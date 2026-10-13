"""Paired diagnostics must compare the same IDs and successful sample subset."""

import json
import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from deepfake_detection_archive.contracts import ScoreRecord
from deepfake_detection_archive.evaluation.comparison import compare_records


def records(values: list[float]) -> list[ScoreRecord]:
    return [
        ScoreRecord(
            sample_id=f"s{index}",
            modality="audio",
            score=value,
            status="ok",
            elapsed_seconds=0,
            label=int(index >= len(values) // 2),
            source="synthetic",
            content_type="speech",
            generator="real" if index < len(values) // 2 else "A01",
            group_id="unknown",
        )
        for index, value in enumerate(values)
    ]


def failed(row: ScoreRecord) -> ScoreRecord:
    return replace(
        row,
        score=None,
        status="error",
        error_code="decode_error",
        error_message="corrupt media",
    )


def test_identical_scores_and_reordered_files_have_zero_delta() -> None:
    baseline = records([0.1, 0.2, 0.8, 0.9])
    report = compare_records(baseline, baseline)
    assert compare_records(baseline[::-1], baseline[1:] + baseline[:1]) == report
    assert report.fullset_delta_eer == 0
    assert report.common_delta_eer == 0
    assert report.common_sample_count == 4
    assert report.common_coverage == 1
    assert report.excluded_sample_ids == ()
    assert report.fullset_reason is None
    assert report.common_reason is None


def test_legacy_prompt_score_groups_are_not_counted_as_independence() -> None:
    baseline = [
        replace(row, group_id=f"aime-prompt:{index}")
        for index, row in enumerate(records([0.1, 0.2, 0.8, 0.9]))
    ]
    report = compare_records(baseline, baseline)
    assert report.fullset_delta_eer == 0
    assert report.baseline.group_count == report.candidate.group_count == 0
    assert report.baseline.missing_group_count == 4


def test_worse_scores_have_positive_candidate_minus_baseline_delta() -> None:
    report = compare_records(
        records([0.1, 0.2, 0.8, 0.9]), records([0.9, 0.8, 0.2, 0.1])
    )
    assert report.baseline.fullset_eer == 0
    assert report.candidate.fullset_eer == 1
    assert report.fullset_delta_eer == 1


def test_different_failure_locations_use_common_success_intersection() -> None:
    baseline = records([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    candidate = records([0.9, 0.2, 0.8, 0.3, 0.8, 0.1])
    baseline[1] = failed(baseline[1])
    candidate[4] = failed(candidate[4])
    report = compare_records(baseline, candidate)
    assert report.baseline.sample_count == report.candidate.sample_count == 6
    assert report.baseline.failure_count == report.candidate.failure_count == 1
    assert report.fullset_delta_eer is None
    assert report.fullset_reason == "failed_samples_have_no_score"
    assert report.common_sample_count == 4
    assert report.common_coverage == pytest.approx(2 / 3)
    assert report.excluded_sample_ids == ("s1", "s4")
    assert report.common_baseline is not None
    assert report.common_candidate is not None
    assert report.common_baseline.fullset_eer == 0
    assert report.common_candidate.fullset_eer == 1
    assert report.common_delta_eer == 1
    assert compare_records(candidate, baseline).common_delta_eer == -1


def test_no_common_success_keeps_full_failure_report() -> None:
    baseline = records([0.1, 0.2, 0.8, 0.9])
    report = compare_records(baseline, [failed(row) for row in baseline])
    assert report.candidate.failure_count == 4
    assert report.candidate.coverage == 0
    assert report.common_sample_count == report.common_coverage == 0
    assert report.common_baseline is None
    assert report.common_candidate is None
    assert report.common_delta_eer is None
    assert report.common_reason == "no_common_successful_samples"


def test_common_subset_missing_class_reports_reason() -> None:
    baseline = records([0.1, 0.2, 0.8, 0.9])
    candidate = [failed(row) if row.label == 0 else row for row in baseline]
    report = compare_records(baseline, candidate)
    assert report.common_sample_count == 2
    assert report.common_delta_eer is None
    assert report.common_reason == "successful_labeled_subset_missing_class"


def test_unlabeled_samples_do_not_produce_fullset_delta() -> None:
    baseline = records([0.1, 0.2, 0.8, 0.9])
    baseline[0] = replace(baseline[0], label=None)
    report = compare_records(baseline, baseline)
    assert report.fullset_delta_eer is None
    assert report.common_delta_eer is None
    assert report.fullset_reason == report.common_reason == "unlabeled_samples"


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_mismatched_or_duplicate_ids_are_rejected(change: str) -> None:
    baseline = records([0.1, 0.2, 0.8, 0.9])
    candidate = {
        "missing": baseline[1:],
        "extra": [*baseline, replace(baseline[0], sample_id="extra")],
        "duplicate": [*baseline, baseline[0]],
    }[change]
    with pytest.raises(ValueError, match="sample IDs"):
        compare_records(baseline, candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label", 1),
        ("modality", "video"),
        ("source", "different"),
        ("content_type", "music"),
        ("generator", "A02"),
        ("group_id", "other"),
    ],
)
def test_mismatched_pair_metadata_is_rejected(field: str, value: object) -> None:
    baseline = records([0.1, 0.2, 0.8, 0.9])
    candidate = baseline.copy()
    candidate[0] = replace(candidate[0], **{field: value})
    with pytest.raises(ValueError, match=field):
        compare_records(baseline, candidate)


def test_audio_and_video_are_not_combined_into_one_eer() -> None:
    baseline = records([0.1, 0.2, 0.8, 0.9])
    baseline[0] = replace(baseline[0], modality="video")
    with pytest.raises(ValueError, match="single modality"):
        compare_records(baseline, baseline)


def test_fixed_threshold_exposes_calibration_shift_despite_unchanged_eer() -> None:
    baseline = records([0.1, 0.2, 0.8, 0.9])
    candidate = records([0.6, 0.7, 1.3, 1.4])
    report = compare_records(baseline, candidate, frozen_threshold=0.5)
    assert report.fullset_delta_eer == 0
    assert report.baseline.frozen_threshold == report.candidate.frozen_threshold == 0.5
    assert report.baseline.successful_subset_frr == 0
    assert report.candidate.successful_subset_frr == 1
    assert report.candidate.successful_subset_far == 0


def test_score_file_tool_writes_paired_report_and_input_hashes(tmp_path: Path) -> None:
    baseline = records([0.1, 0.2, 0.8, 0.9])
    inputs = [tmp_path / "baseline.jsonl", tmp_path / "candidate.jsonl"]
    for path in inputs:
        path.write_text(
            "".join(json.dumps(row.to_dict()) + "\n" for row in baseline),
            encoding="utf-8",
        )
    output = tmp_path / "comparison.json"
    main = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/compare_scores.py")
    )["main"]
    main(
        [
            "--baseline-scores",
            str(inputs[0]),
            "--candidate-scores",
            str(inputs[1]),
            "--threshold",
            "0.5",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["comparison"]["fullset_delta_eer"] == 0
    assert (
        report["inputs"]["baseline"]["sha256"]
        == report["inputs"]["candidate"]["sha256"]
    )
    assert report["model_identity_verified"] is False
    assert report["manifest_metadata_verified"] is False
