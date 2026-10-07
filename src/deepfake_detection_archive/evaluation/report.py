"""Failure-aware reports over complete score records and diagnostic buckets."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol

from .metrics import BinaryMetrics, MetricInputError, evaluate_binary


class EvaluationRecord(Protocol):
    """Read-only structural interface, also implemented by runtime ScoreRecord."""

    @property
    def sample_id(self) -> str: ...
    @property
    def label(self) -> int | None: ...
    @property
    def score(self) -> float | None: ...
    @property
    def status(self) -> str: ...
    @property
    def error_code(self) -> str | None: ...
    @property
    def group_id(self) -> str | None: ...
    @property
    def source(self) -> str: ...
    @property
    def content_type(self) -> str: ...
    @property
    def generator(self) -> str: ...
    @property
    def modality(self) -> str: ...


@dataclass(frozen=True)
class EvaluationReport:
    sample_count: int
    success_count: int
    failure_count: int
    unlabeled_count: int
    coverage: float
    failure_rate: float
    fullset_eer: float | None
    fullset_reason: str | None
    successful_subset_metrics: BinaryMetrics | None
    successful_subset_reason: str | None
    failures_by_code: dict[str, int]
    failed_sample_ids: tuple[str, ...]
    real_count: int
    fake_count: int
    group_count: int
    real_group_count: int
    fake_group_count: int
    missing_group_count: int
    frozen_threshold: float | None
    successful_subset_far: float | None
    successful_subset_frr: float | None


def _validate_record(record: EvaluationRecord) -> None:
    if record.label is not None and (
        isinstance(record.label, bool)
        or not isinstance(record.label, int)
        or record.label not in (0, 1)
    ):
        raise MetricInputError(f"invalid label: sample_id={record.sample_id}")
    if record.status == "ok":
        if (
            record.score is None
            or isinstance(record.score, bool)
            or not isinstance(record.score, (float, int))
            or not isfinite(record.score)
            or record.error_code is not None
        ):
            raise MetricInputError(f"invalid success: sample_id={record.sample_id}")
    elif record.status == "error":
        if not record.error_code or record.score is not None:
            raise MetricInputError(f"invalid failure: sample_id={record.sample_id}")
    else:
        raise MetricInputError(f"invalid status: sample_id={record.sample_id}")


def evaluate_records(
    records: Sequence[EvaluationRecord],
    *,
    frozen_threshold: float | None = None,
    expected_sample_ids: Sequence[str] | None = None,
) -> EvaluationReport:
    """Report all failures; never assign artificial failure scores using truth.

    A finite score is required for every ``ok`` row; every error row must carry
    an error code and score=None. If expected IDs are supplied, their exact set
    must be represented once. Unlabeled rows make fullset EER undefined. Single
    class buckets can report error rates only at an externally frozen threshold.
    """
    if not records:
        raise MetricInputError("cannot report an empty sample collection")
    ids = [record.sample_id for record in records]
    if any(not sample_id for sample_id in ids) or len(set(ids)) != len(ids):
        raise MetricInputError("sample IDs must be nonempty and unique")
    if expected_sample_ids is not None:
        if len(set(expected_sample_ids)) != len(expected_sample_ids):
            raise MetricInputError("expected sample IDs must be unique")
        if set(ids) != set(expected_sample_ids):
            raise MetricInputError("score records do not match expected sample IDs")
    if frozen_threshold is not None and not isfinite(frozen_threshold):
        raise MetricInputError("frozen threshold must be finite")
    labels: list[int] = []
    scores: list[float] = []
    errors: Counter[str] = Counter()
    failed_ids = []
    for record in records:
        _validate_record(record)
        if record.status == "ok":
            if record.label is not None and record.score is not None:
                labels.append(record.label)
                scores.append(record.score)
        elif record.error_code is not None:
            errors[record.error_code] += 1
            failed_ids.append(record.sample_id)
    metrics = evaluate_binary(labels, scores) if set(labels) == {0, 1} else None
    subset_reason = None if metrics else "successful_labeled_subset_missing_class"
    unlabeled_count = sum(record.label is None for record in records)
    reason = (
        "failed_samples_have_no_score"
        if failed_ids
        else "unlabeled_samples"
        if unlabeled_count
        else subset_reason
    )
    real_scores = [
        score for label, score in zip(labels, scores, strict=True) if label == 0
    ]
    fake_scores = [
        score for label, score in zip(labels, scores, strict=True) if label == 1
    ]
    known_groups = [
        record
        for record in records
        if record.group_id is not None
        and record.group_id.strip()
        and record.group_id.strip().casefold() != "unknown"
    ]
    return EvaluationReport(
        sample_count=len(records),
        success_count=len(records) - len(failed_ids),
        failure_count=len(failed_ids),
        unlabeled_count=unlabeled_count,
        coverage=(len(records) - len(failed_ids)) / len(records),
        failure_rate=len(failed_ids) / len(records),
        fullset_eer=metrics.eer if reason is None and metrics else None,
        fullset_reason=reason,
        successful_subset_metrics=metrics,
        successful_subset_reason=subset_reason,
        failures_by_code=dict(sorted(errors.items())),
        failed_sample_ids=tuple(failed_ids),
        real_count=sum(record.label == 0 for record in records),
        fake_count=sum(record.label == 1 for record in records),
        group_count=len({record.group_id for record in known_groups}),
        real_group_count=len({r.group_id for r in known_groups if r.label == 0}),
        fake_group_count=len({r.group_id for r in known_groups if r.label == 1}),
        missing_group_count=len(records) - len(known_groups),
        frozen_threshold=frozen_threshold,
        successful_subset_far=(
            sum(score < frozen_threshold for score in fake_scores) / len(fake_scores)
            if frozen_threshold is not None and fake_scores
            else None
        ),
        successful_subset_frr=(
            sum(score >= frozen_threshold for score in real_scores) / len(real_scores)
            if frozen_threshold is not None and real_scores
            else None
        ),
    )


def grouped_reports(
    records: Sequence[EvaluationRecord],
    field: Literal["source", "content_type", "generator", "modality"],
    *,
    frozen_threshold: float | None = None,
) -> dict[str, EvaluationReport]:
    """Produce source/content/generator/modality diagnostics without tuning."""
    evaluate_records(records, frozen_threshold=frozen_threshold)
    if field not in ("source", "content_type", "generator", "modality"):
        raise MetricInputError(f"unsupported grouping field: {field}")
    buckets: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        key: str = getattr(record, field)
        buckets.setdefault(key, []).append(record)
    return {
        key: evaluate_records(bucket, frozen_threshold=frozen_threshold)
        for key, bucket in sorted(buckets.items())
    }
