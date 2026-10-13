"""Inference stage attribution stays session-local and leaves score rows unchanged."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from deepfake_detection_archive.data.manifest import SampleRecord
from deepfake_detection_archive.submission.runner import run_inference
from deepfake_detection_archive.timing import (
    InferenceStageTimings,
    collect_inference_stages,
    inference_stage,
)


@dataclass
class Clock:
    now: float = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    value = Clock()
    monkeypatch.setattr(
        "deepfake_detection_archive.timing.time.perf_counter", lambda: value.now
    )
    return value


def records() -> list[SampleRecord]:
    return [
        SampleRecord("real", "real.wav", "audio", 0, "test"),
        SampleRecord("fake", "fake.wav", "audio", 1, "test"),
    ]


def test_stages_are_disjoint_and_preserve_legacy_sample_time(
    tmp_path: Path, clock: Clock
) -> None:
    def score(record: SampleRecord) -> float:
        with inference_stage("input_validation"):
            clock.advance(1)
        with inference_stage("decode_and_preprocess"):
            clock.advance(2)
        with inference_stage("model_and_feature_scoring"):
            clock.advance(3)
        clock.advance(0.5)
        return float(record.label or 0)

    result = run_inference(records(), score, tmp_path, identity={"model": "test"})
    assert [row.elapsed_seconds for row in result] == [6.5, 6.5]
    timing = json.loads((tmp_path / "report.json").read_text())["timing"]
    assert timing["sample_call_seconds"] == 13
    assert timing["current_session_sample_call_seconds"] == 13
    assert timing["current_session_instrumented_rows"] == 2
    assert timing["current_session_stage_seconds"] == {
        "input_validation": 2,
        "decode_and_preprocess": 4,
        "model_and_feature_scoring": 6,
        "unattributed_sample_call": 1,
    }
    assert timing["current_session_setup_seconds"] == 0
    assert timing["current_session_report_build_seconds"] == 0
    assert timing["current_session_elapsed_seconds_before_report_write"] == 13
    assert "no added CUDA synchronization" in timing["stage_clock"]


def test_failed_stage_is_measured_without_changing_failure_accounting(
    tmp_path: Path, clock: Clock
) -> None:
    def score(record: SampleRecord) -> float:
        with inference_stage("decode_and_preprocess"):
            clock.advance(2)
            raise ValueError("decode failed")

    result = run_inference(records(), score, tmp_path, identity={"model": "test"})
    assert all(row.status == "error" for row in result)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["failure_count"] == 2
    timing = report["timing"]
    assert timing["current_session_instrumented_rows"] == 2
    assert timing["current_session_stage_seconds"]["decode_and_preprocess"] == 4
    assert timing["current_session_sample_call_seconds"] == 4


def test_resume_measures_only_new_rows_and_preserves_completed_report(
    tmp_path: Path, clock: Clock
) -> None:
    def interrupted(record: SampleRecord) -> float:
        with inference_stage("decode_and_preprocess"):
            clock.advance(1)
            if record.sample_id == "fake":
                raise KeyboardInterrupt
        with inference_stage("model_and_feature_scoring"):
            clock.advance(3)
        return 0.0

    with pytest.raises(KeyboardInterrupt):
        run_inference(records(), interrupted, tmp_path, identity={"model": "test"})
    prefix = (tmp_path / "scores.jsonl").read_bytes()

    def resumed(record: SampleRecord) -> float:
        assert record.sample_id == "fake"
        with inference_stage("decode_and_preprocess"):
            clock.advance(5)
        with inference_stage("model_and_feature_scoring"):
            clock.advance(7)
        return 1.0

    result = run_inference(
        records(), resumed, tmp_path, identity={"model": "test"}, resume=True
    )
    assert [row.elapsed_seconds for row in result] == [4, 12]
    assert (tmp_path / "scores.jsonl").read_bytes().startswith(prefix)
    report_bytes = (tmp_path / "report.json").read_bytes()
    timing = json.loads(report_bytes)["timing"]
    assert timing["sample_call_seconds"] == 16
    assert timing["current_session_sample_call_seconds"] == 12
    assert timing["current_session_new_rows"] == 1
    assert timing["current_session_instrumented_rows"] == 1
    assert timing["current_session_stage_seconds"] == {
        "input_validation": 0,
        "decode_and_preprocess": 5,
        "model_and_feature_scoring": 7,
        "unattributed_sample_call": 0,
    }
    run_inference(records(), resumed, tmp_path, identity={"model": "test"}, resume=True)
    assert (tmp_path / "report.json").read_bytes() == report_bytes


def test_uninstrumented_scorer_is_reported_as_unattributed(
    tmp_path: Path, clock: Clock
) -> None:
    def score(record: SampleRecord) -> float:
        clock.advance(4)
        return float(record.label or 0)

    run_inference(records(), score, tmp_path, identity={"model": "test"})
    timing = json.loads((tmp_path / "report.json").read_text())["timing"]
    assert timing["current_session_instrumented_rows"] == 0
    assert timing["current_session_stage_seconds"] == {
        "input_validation": 0,
        "decode_and_preprocess": 0,
        "model_and_feature_scoring": 0,
        "unattributed_sample_call": 8,
    }


def test_collection_resets_after_exception_and_is_inactive_outside_runner(
    clock: Clock,
) -> None:
    first, second = InferenceStageTimings(), InferenceStageTimings()

    def interrupted_collection() -> None:
        with collect_inference_stages(second), inference_stage("decode_and_preprocess"):
            clock.advance(2)
            raise KeyboardInterrupt

    with collect_inference_stages(first):
        with pytest.raises(KeyboardInterrupt):
            interrupted_collection()
        with inference_stage("input_validation"):
            clock.advance(3)
    with inference_stage("model_and_feature_scoring"):
        clock.advance(5)
    assert first.seconds["input_validation"] == 3
    assert second.seconds["decode_and_preprocess"] == 2
    assert first.measured_scopes == second.measured_scopes == 1
    assert first.seconds["model_and_feature_scoring"] == 0
    assert second.seconds["model_and_feature_scoring"] == 0
