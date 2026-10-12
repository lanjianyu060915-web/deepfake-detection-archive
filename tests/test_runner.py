"""Persistence, recovery, failure accounting and OS-lock behavior."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from deepfake_detection_archive.data.manifest import SampleRecord
from deepfake_detection_archive.submission.runner import run_inference


IDENTITY: dict[str, object] = {
    "model_hash": "model-v1",
    "config_hash": "config-v1",
    "code_hash": "code-v1",
}


def inputs() -> list[SampleRecord]:
    return [
        SampleRecord("b", "b.wav", "audio", 0, "source", group_id="b"),
        SampleRecord("a", "a.wav", "audio", 1, "source", group_id="a"),
    ]


def infer(record: SampleRecord) -> float:
    return float(record.label or 0)


def test_ordered_inference_and_failure_aware_report(tmp_path: Path) -> None:
    result = run_inference(inputs(), infer, tmp_path, identity=IDENTITY)
    assert [score.sample_id for score in result] == ["b", "a"]
    assert all(score.elapsed_seconds >= 0 for score in result)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["fullset_eer"] == 0
    assert report["sample_count"] == 2
    assert report["timing"]["T10k_safe_seconds"] == pytest.approx(
        1.25 * report["timing"]["T10k_raw_seconds"]
    )
    assert report["timing"]["representativeness_reason"] == "fewer_than_500_samples"
    assert report["official_evaluation"] is False


def test_completed_resume_never_recomputes_or_changes_report(tmp_path: Path) -> None:
    first = run_inference(inputs(), infer, tmp_path, identity=IDENTITY)
    report = (tmp_path / "report.json").read_bytes()

    def forbidden(record: SampleRecord) -> float:
        pytest.fail("completed row must not be recomputed")

    assert (
        run_inference(inputs(), forbidden, tmp_path, identity=IDENTITY, resume=True)
        == first
    )
    assert (tmp_path / "report.json").read_bytes() == report
    with pytest.raises(ValueError, match="already exists"):
        run_inference(inputs(), infer, tmp_path, identity=IDENTITY)


def test_keyboard_interrupt_and_partial_tail_recovery(tmp_path: Path) -> None:
    def interrupted(record: SampleRecord) -> float:
        if record.sample_id == "a":
            raise KeyboardInterrupt
        return infer(record)

    with pytest.raises(KeyboardInterrupt):
        run_inference(inputs(), interrupted, tmp_path, identity=IDENTITY)
    path = tmp_path / "scores.jsonl"
    prefix = path.read_bytes()
    assert prefix.count(b"\n") == 1
    with path.open("ab") as stream:
        stream.write(b'{"sample_id":"a",')
    called: list[str] = []

    def resumed(record: SampleRecord) -> float:
        called.append(record.sample_id)
        return infer(record)

    result = run_inference(inputs(), resumed, tmp_path, identity=IDENTITY, resume=True)
    assert called == ["a"]
    assert len(result) == 2
    assert path.read_bytes().startswith(prefix)
    assert len(path.read_text().splitlines()) == 2


@pytest.mark.parametrize("change", ["identity", "order", "path", "label", "metadata"])
def test_resume_rejects_identity_or_input_mutation(tmp_path: Path, change: str) -> None:
    records = inputs()
    run_inference(records, infer, tmp_path, identity=IDENTITY)
    saved = (tmp_path / "scores.jsonl").read_bytes()
    identity = dict(IDENTITY)
    if change == "identity":
        identity["model_hash"] = "changed"
    elif change == "order":
        records.reverse()
    elif change == "path":
        records[0] = replace(records[0], path="other.wav")
    elif change == "label":
        records[0] = replace(records[0], label=1)
    else:
        records[0] = replace(records[0], source="other")
    with pytest.raises(ValueError, match="identity changed"):
        run_inference(records, infer, tmp_path, identity=identity, resume=True)
    assert (tmp_path / "scores.jsonl").read_bytes() == saved


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_id",
        "wrong_metadata",
        "duplicate",
        "malformed",
        "duplicate_key",
        "non_object",
        "nonfinite",
    ],
)
def test_complete_invalid_rows_are_rejected_before_tail_truncation(
    tmp_path: Path, corruption: str
) -> None:
    run_inference(inputs(), infer, tmp_path, identity=IDENTITY)
    path = tmp_path / "scores.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    if corruption == "wrong_id":
        row["sample_id"] = "other"
        lines[0] = json.dumps(row)
    elif corruption == "wrong_metadata":
        row["generator"] = "changed"
        lines[0] = json.dumps(row)
    elif corruption == "duplicate":
        lines[1] = lines[0]
    elif corruption == "malformed":
        lines[0] = '{"bad":'
    elif corruption == "non_object":
        lines[0] = "[]"
    elif corruption == "nonfinite":
        row["score"] = float("nan")
        lines[0] = json.dumps(row)
    else:
        lines[0] = lines[0][:-1] + ',"status":"ok"}'
    corrupt = ("\n".join(lines) + '\n{"unfinished":').encode()
    path.write_bytes(corrupt)
    with pytest.raises(ValueError, match=r"."):
        run_inference(inputs(), infer, tmp_path, identity=IDENTITY, resume=True)
    assert path.read_bytes() == corrupt


def test_unserializable_identity_is_rejected_before_output(tmp_path: Path) -> None:
    output = tmp_path / "run"
    with pytest.raises(TypeError, match="not JSON serializable"):
        run_inference(inputs(), infer, output, identity={"model": object()})
    assert not output.exists()


def test_failures_count_in_full_denominator_and_keep_custom_code(
    tmp_path: Path,
) -> None:
    class DecodeError(Exception):
        code = "decode_error"

    def failed(record: SampleRecord) -> float:
        if record.sample_id == "a":
            raise DecodeError("corrupted audio")
        return infer(record)

    result = run_inference(inputs(), failed, tmp_path, identity=IDENTITY)
    assert result[1].status == "error"
    assert result[1].error_code == "decode_error"
    assert result[1].score is None
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["sample_count"] == 2
    assert report["failure_rate"] == 0.5
    assert report["fullset_eer"] is None
    assert report["fullset_reason"] == "failed_samples_have_no_score"


def test_invalid_score_becomes_explicit_failure(tmp_path: Path) -> None:
    result = run_inference(
        inputs(), lambda record: float("nan"), tmp_path, identity=IDENTITY
    )
    assert all(
        score.status == "error" and score.error_code == "inference_error"
        for score in result
    )


def test_unavailable_sample_is_not_sent_to_scorer(tmp_path: Path) -> None:
    records = [replace(inputs()[0], status="missing", failure_reason="not downloaded")]

    def forbidden(record: SampleRecord) -> float:
        pytest.fail("known missing file must not be scored")

    result = run_inference(records, forbidden, tmp_path, identity=IDENTITY)
    assert result[0].error_code == "missing"


def test_concurrent_writer_is_rejected_by_os_lock(tmp_path: Path) -> None:
    entered, release = Event(), Event()

    def slow(record: SampleRecord) -> float:
        entered.set()
        assert release.wait(5), "test synchronization timed out"
        return infer(record)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_inference, inputs(), slow, tmp_path, identity=IDENTITY)
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="holds this run lock"):
                run_inference(inputs(), infer, tmp_path, identity=IDENTITY, resume=True)
        finally:
            release.set()
        assert len(future.result(timeout=5)) == 2


def test_stale_pid_file_does_not_block_authorized_resume(tmp_path: Path) -> None:
    (tmp_path / ".runner.lock").write_text("pid=1\n")
    assert len(run_inference(inputs(), infer, tmp_path, identity=IDENTITY)) == 2


def test_missing_scores_from_completed_run_are_not_silently_recomputed(
    tmp_path: Path,
) -> None:
    run_inference(inputs(), infer, tmp_path, identity=IDENTITY)
    (tmp_path / "scores.jsonl").unlink()
    with pytest.raises(ValueError, match="scores are missing"):
        run_inference(inputs(), infer, tmp_path, identity=IDENTITY, resume=True)


def test_duplicate_inputs_empty_identity_and_orphan_artifacts_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unique"):
        run_inference([inputs()[0]] * 2, infer, tmp_path, identity=IDENTITY)
    with pytest.raises(ValueError, match="identity"):
        run_inference(inputs(), infer, tmp_path, identity={})
    with pytest.raises(ValueError, match="without metadata"):
        run_inference(inputs(), infer, tmp_path, identity=IDENTITY, resume=True)
    (tmp_path / "scores.jsonl").write_text("orphan\n")
    with pytest.raises(ValueError, match="without metadata"):
        run_inference(inputs(), infer, tmp_path, identity=IDENTITY)


@pytest.mark.parametrize(
    "exception", [MemoryError("exhausted"), ImportError("backend")]
)
def test_fatal_failure_aborts_and_preserves_resumable_prefix(
    tmp_path: Path,
    exception: Exception,
) -> None:
    called = []

    def failing(record: SampleRecord) -> float:
        called.append(record.sample_id)
        if record.sample_id == "a":
            raise exception
        return infer(record)

    with pytest.raises(type(exception)):
        run_inference(inputs(), failing, tmp_path, identity=IDENTITY)
    assert called == ["b", "a"]
    assert not (tmp_path / "report.json").exists()
    prefix = (tmp_path / "scores.jsonl").read_bytes()
    assert prefix.count(b"\n") == 1
    result = run_inference(inputs(), infer, tmp_path, identity=IDENTITY, resume=True)
    assert len(result) == 2
    assert (tmp_path / "scores.jsonl").read_bytes().startswith(prefix)


def test_backend_fatal_exception_is_injected_without_torch(tmp_path: Path) -> None:
    class BackendOutOfMemoryError(RuntimeError):
        pass

    def failing(record: SampleRecord) -> float:
        raise BackendOutOfMemoryError("device exhausted")

    with pytest.raises(BackendOutOfMemoryError):
        run_inference(
            inputs(),
            failing,
            tmp_path,
            identity=IDENTITY,
            fatal_exceptions=(BackendOutOfMemoryError,),
        )
    assert (tmp_path / "scores.jsonl").read_bytes() == b""
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("code", ["missing_dependency", "process_error"])
def test_environment_failure_is_not_a_media_failure(tmp_path: Path, code: str) -> None:
    class EnvironmentFailureError(ValueError):
        pass

    error = EnvironmentFailureError("decoder cannot start")
    error.code = code  # type: ignore[attr-defined]

    def failing(record: SampleRecord) -> float:
        raise error

    with pytest.raises(EnvironmentFailureError):
        run_inference(inputs(), failing, tmp_path, identity=IDENTITY)
    assert not (tmp_path / "report.json").exists()
