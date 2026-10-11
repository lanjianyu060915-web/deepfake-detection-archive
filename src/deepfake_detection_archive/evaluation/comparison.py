"""Compare aligned scores without confusing different successful subsets."""

from collections.abc import Sequence
from dataclasses import dataclass

from .metrics import MetricInputError
from .report import EvaluationRecord, EvaluationReport, evaluate_records


@dataclass(frozen=True)
class PairedReport:
    baseline: EvaluationReport
    candidate: EvaluationReport
    fullset_delta_eer: float | None
    fullset_reason: str | None
    common_sample_count: int
    common_coverage: float
    excluded_sample_ids: tuple[str, ...]
    common_baseline: EvaluationReport | None
    common_candidate: EvaluationReport | None
    common_delta_eer: float | None
    common_reason: str | None


def _delta(baseline: EvaluationReport, candidate: EvaluationReport) -> float | None:
    if baseline.fullset_eer is None or candidate.fullset_eer is None:
        return None
    return candidate.fullset_eer - baseline.fullset_eer


def compare_records(
    baseline: Sequence[EvaluationRecord],
    candidate: Sequence[EvaluationRecord],
    *,
    frozen_threshold: float | None = None,
) -> PairedReport:
    """Return candidate-minus-baseline EER on full and common-success sets.

    IDs and paired metadata must agree. Both full reports keep all failures;
    the common-success subset is descriptive and never replaces their coverage.
    A supplied threshold is used unchanged on both sides, with score >= threshold
    predicting fake. Score rows alone cannot prove model or dataset provenance.
    """
    left = sorted(baseline, key=lambda row: row.sample_id)
    right = sorted(candidate, key=lambda row: row.sample_id)
    left_report = evaluate_records(left, frozen_threshold=frozen_threshold)
    right_report = evaluate_records(
        right,
        frozen_threshold=frozen_threshold,
        expected_sample_ids=[row.sample_id for row in left],
    )
    if len({row.modality for row in left}) != 1:
        raise MetricInputError("paired comparison requires a single modality")
    fields = ("label", "modality", "source", "content_type", "generator", "group_id")
    for before, after in zip(left, right, strict=True):
        for field in fields:
            if getattr(before, field) != getattr(after, field):
                raise MetricInputError(
                    f"paired metadata mismatch: {before.sample_id}.{field}"
                )
    excluded = tuple(
        before.sample_id
        for before, after in zip(left, right, strict=True)
        if before.status != "ok" or after.status != "ok"
    )
    excluded_set = set(excluded)
    common_left = [row for row in left if row.sample_id not in excluded_set]
    common_right = [row for row in right if row.sample_id not in excluded_set]
    common_baseline = common_candidate = None
    common_delta = None
    common_reason: str | None = "no_common_successful_samples"
    if common_left:
        common_baseline = (
            evaluate_records(common_left, frozen_threshold=frozen_threshold)
            if excluded
            else left_report
        )
        common_candidate = (
            evaluate_records(common_right, frozen_threshold=frozen_threshold)
            if excluded
            else right_report
        )
        common_delta = _delta(common_baseline, common_candidate)
        common_reason = common_baseline.fullset_reason
    return PairedReport(
        baseline=left_report,
        candidate=right_report,
        fullset_delta_eer=_delta(left_report, right_report),
        fullset_reason=(
            "failed_samples_have_no_score" if excluded else left_report.fullset_reason
        ),
        common_sample_count=len(common_left),
        common_coverage=len(common_left) / len(left),
        excluded_sample_ids=excluded,
        common_baseline=common_baseline,
        common_candidate=common_candidate,
        common_delta_eer=common_delta,
        common_reason=common_reason,
    )
