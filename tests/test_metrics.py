"""Hand-computable operating points, tie invariance and failure accounting."""

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from itertools import permutations, product
from pathlib import Path

import pytest

from deepfake_detection_archive.evaluation.metrics import (
    MetricInputError,
    UndefinedMetricError,
    evaluate_binary,
)
from deepfake_detection_archive.evaluation.report import (
    evaluate_records,
    grouped_reports,
)


@pytest.mark.parametrize(
    ("labels", "scores", "eer", "auc"),
    [
        ([0, 0, 1, 1], [0, 0, 1, 1], 0, 1),
        ([0, 0, 1, 1], [1, 1, 0, 0], 1, 0),
        ([0, 1], [0, 0], 0.5, 0.5),
        ([0, 0, 1], [0, 2, 1], 0.5, 0.5),
        ([0, 0, 1, 1], [0, 1, 1, 2], 0.25, 0.875),
    ],
)
def test_hand_computed_metrics(
    labels: list[int], scores: list[float], eer: float, auc: float
) -> None:
    result = evaluate_binary(labels, scores)
    assert result.eer == pytest.approx(eer)
    assert result.roc_auc == pytest.approx(auc)
    assert result.real_count + result.fake_count == len(labels)
    json.dumps(asdict(result), allow_nan=False)


def test_det_points_match_actual_threshold_decisions() -> None:
    labels, scores = [0, 1, 0, 1, 0], [0.0, 1.0, 1.0, 2.0, 2.0]
    result = evaluate_binary(labels, scores)
    assert len(result.det_curve) == len(set(scores)) + 1
    assert result.det_curve[0].threshold is None
    assert (result.det_curve[0].far, result.det_curve[0].frr) == (1, 0)
    assert (result.det_curve[-1].far, result.det_curve[-1].frr) == (0, 1)
    for point in result.det_curve[1:]:
        assert point.threshold is not None
        assert point.far == sum(
            y == 1 and score < point.threshold
            for y, score in zip(labels, scores, strict=True)
        ) / sum(labels)
        assert point.frr == sum(
            y == 0 and score >= point.threshold
            for y, score in zip(labels, scores, strict=True)
        ) / labels.count(0)


def test_interpolation_is_not_an_attainable_threshold() -> None:
    result = evaluate_binary([0, 0, 1, 1], [0, 1, 1, 2])
    assert result.lower_point.threshold == 2
    assert result.upper_point.threshold == 1
    assert result.interpolation_weight == 0.5
    assert result.nearest_point == result.lower_point
    assert result.nearest_eer == 0.25
    assert all(point.far != point.frr for point in result.det_curve)


def test_tie_group_row_permutations_do_not_change_any_metric() -> None:
    pairs = [(0, 0), (0, 1), (1, 1), (1, 2)]
    expected = evaluate_binary([y for y, _ in pairs], [s for _, s in pairs])
    for rows in permutations(pairs):
        assert evaluate_binary([y for y, _ in rows], [s for _, s in rows]) == expected


def test_auc_agrees_with_independent_pair_ranking_exhaustively() -> None:
    labels = [0, 0, 1, 1]
    for scores in product([0.0, 1.0, 2.0], repeat=4):
        expected = (
            sum(
                float(fake > real) + 0.5 * (fake == real)
                for fake in scores[2:]
                for real in scores[:2]
            )
            / 4
        )
        result = evaluate_binary(labels, scores)
        assert result.roc_auc == pytest.approx(expected)
        assert 0 <= result.eer <= 1
        assert result.lower_point.far >= result.lower_point.frr
        assert result.upper_point.far <= result.upper_point.frr


@pytest.mark.parametrize("labels", [[], [0], [1, 1]])
def test_missing_classes_raise(labels: list[int]) -> None:
    with pytest.raises(UndefinedMetricError):
        evaluate_binary(labels, [0.1] * len(labels))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_score_names_row(score: float) -> None:
    with pytest.raises(MetricInputError, match=r"rows=\[1\]"):
        evaluate_binary([0, 1], [0, score])


@pytest.mark.parametrize("labels", [[0, 2], [0, -1], [0, True], [0, 1.0]])
def test_invalid_labels_are_not_coerced(labels: list[int]) -> None:
    with pytest.raises(MetricInputError, match="labels must"):
        evaluate_binary(labels, [0, 1])


def test_shape_mismatch_and_boolean_scores() -> None:
    with pytest.raises(MetricInputError, match="equal length"):
        evaluate_binary([0, 1], [0])
    with pytest.raises(MetricInputError, match="finite numbers"):
        evaluate_binary([0, 1], [False, True])


@dataclass(frozen=True)
class Record:
    sample_id: str
    label: int | None
    score: float | None
    status: str = "ok"
    error_code: str | None = None
    group_id: str | None = "group"
    source: str = "source"
    content_type: str = "speech"
    generator: str = "unknown"
    modality: str = "audio"


def test_failure_uses_full_denominator_and_nulls_fullset() -> None:
    rows = [
        Record("real", 0, 0),
        Record("fake", 1, 1),
        Record("bad", 1, None, "error", "decode_error"),
    ]
    report = evaluate_records(rows, expected_sample_ids=["bad", "real", "fake"])
    assert report.failure_rate == pytest.approx(1 / 3)
    assert report.coverage == pytest.approx(2 / 3)
    assert report.fullset_eer is None
    assert report.fullset_reason == "failed_samples_have_no_score"
    assert report.successful_subset_metrics is not None
    assert report.successful_subset_metrics.eer == 0
    assert report.failed_sample_ids == ("bad",)
    assert report.failures_by_code == {"decode_error": 1}
    assert report.fake_count == 2
    json.dumps(asdict(report), allow_nan=False)


def test_complete_success_and_unlabeled_rows() -> None:
    rows = [Record("real", 0, 0), Record("fake", 1, 1)]
    assert evaluate_records(rows).fullset_eer == 0
    result = evaluate_records([*rows, Record("unknown", None, 0.1)])
    assert result.unlabeled_count == 1
    assert result.fullset_eer is None
    assert result.fullset_reason == "unlabeled_samples"
    assert result.coverage == 1


@pytest.mark.parametrize(
    "group",
    [None, "", " ", "unknown", " UNKNOWN ", "aime-prompt:tags", " AIME-PROMPT:x "],
)
def test_unknown_groups_are_missing_evidence(group: str | None) -> None:
    rows = [
        Record("real", 0, 0, group_id=group),
        Record("fake", 1, 1, group_id="known"),
    ]
    report = evaluate_records(rows)
    assert report.fullset_eer == 0
    assert report.group_count == report.fake_group_count == 1
    assert report.real_group_count == 0
    assert report.missing_group_count == 1


def test_single_class_buckets_use_frozen_threshold_only() -> None:
    rows = [
        Record("real-a", 0, 0.5, source="real"),
        Record("real-b", 0, 0.2, source="real"),
        Record("fake", 1, 0.5, source="fake"),
    ]
    reports = grouped_reports(rows, "source", frozen_threshold=0.5)
    assert reports["real"].fullset_eer is None
    assert reports["real"].successful_subset_frr == 0.5
    assert reports["real"].successful_subset_far is None
    assert reports["fake"].successful_subset_far == 0
    assert evaluate_records(rows[:2]).successful_subset_frr is None


@pytest.mark.parametrize(
    "row",
    [
        Record("x", 0, float("nan")),
        Record("x", 0, None),
        Record("x", 0, 0.5, "error", "decode_error"),
        Record("x", 0, None, "error"),
        Record("x", 0, 0.5, "unknown"),
        Record("x", 2, 0.5),
    ],
)
def test_invalid_records_raise(row: Record) -> None:
    with pytest.raises(MetricInputError):
        evaluate_records([row])


def test_report_detects_missing_duplicate_and_all_failed_records() -> None:
    row = Record("one", 0, 0.5)
    with pytest.raises(MetricInputError, match="unique"):
        evaluate_records([row, row])
    with pytest.raises(MetricInputError, match="expected sample IDs"):
        evaluate_records([row], expected_sample_ids=["one", "two"])
    result = evaluate_records(
        [replace(row, score=None, status="error", error_code="oom")]
    )
    assert result.coverage == 0
    assert result.failure_rate == 1
    assert result.successful_subset_metrics is None


def test_readonly_asvspoof_reference_in_isolated_process_when_available() -> None:
    """Reference is optional locally and never a production import/dependency."""
    pytest.importorskip("numpy")
    root = Path(__file__).resolve().parents[1]
    reference = root / "external/asvspoof2021/eval-package/eval_metrics.py"
    if not reference.exists():
        pytest.skip("fixed ASVspoof reference checkout is not present")
    revision = subprocess.run(
        ["git", "-C", str(reference.parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == "9b33f5eca887bd3bf629e3fb79428cd9bb7ac972"
    code = """
import importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location('asvspoof_reference', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pairs = [([0, 1], [2, 3]), ([2, 3], [0, 1]), ([0, 2], [1])]
print(json.dumps([float(module.compute_eer(-np.array(real, dtype=float),
    -np.array(fake, dtype=float))[0]) for real, fake in pairs]))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(reference)],
        check=True,
        capture_output=True,
        text=True,
    )
    reference_eers = json.loads(result.stdout)
    assert reference_eers == [0.0, 1.0, 0.75]
    assert evaluate_binary([0, 0, 1, 1], [0, 1, 2, 3]).eer == reference_eers[0]
    assert evaluate_binary([0, 0, 1, 1], [2, 3, 0, 1]).eer == reference_eers[1]
    assert evaluate_binary([0, 0, 1], [0, 2, 1]).eer == 0.5
