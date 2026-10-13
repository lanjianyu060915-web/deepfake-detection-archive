"""Durable, ordered inference with identity-bound resume and process locking."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO, cast

from deepfake_detection_archive.contracts import ScoreRecord, score_from_dict
from deepfake_detection_archive.data.manifest import SampleRecord
from deepfake_detection_archive.evaluation.report import (
    evaluate_records,
    grouped_reports,
)
from deepfake_detection_archive.timing import (
    InferenceStageTimings,
    collect_inference_stages,
)
from deepfake_detection_archive.tracking import (
    atomic_json,
    canonical_json,
    object_hash,
    parse_json,
)


@contextmanager
def _run_lock(path: Path) -> Iterator[None]:
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another inference process holds this run lock") from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()} (diagnostic only)\n")
            stream.flush()
            os.fsync(stream.fileno())
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_object(data: bytes) -> dict[str, object]:
    value = parse_json(data)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")  # noqa: TRY004 -- persisted schema
    return cast(dict[str, object], value)


def _sample_metadata(record: SampleRecord) -> dict[str, object]:
    return {
        key: getattr(record, key)
        for key in (
            "sample_id",
            "modality",
            "label",
            "group_id",
            "source",
            "content_type",
            "generator",
        )
    }


def _read_prefix(path: Path, records: Sequence[SampleRecord]) -> list[ScoreRecord]:
    """Validate all committed rows before truncating a non-newline-terminated tail."""
    if not path.exists():
        return []
    data = path.read_bytes()
    last_newline = data.rfind(b"\n") + 1
    complete = data[:last_newline]
    result: list[ScoreRecord] = []
    for index, line in enumerate(complete.splitlines()):
        if index >= len(records):
            raise ValueError("saved scores exceed the input manifest")
        score = score_from_dict(_read_object(line))
        metadata = _sample_metadata(records[index])
        saved_metadata = {key: getattr(score, key) for key in metadata}
        if object_hash(saved_metadata) != object_hash(metadata):
            raise ValueError(
                f"saved score metadata/order differs at input index {index}"
            )
        result.append(score)
    if len(complete) != len(data):
        with path.open("r+b") as stream:
            stream.truncate(last_newline)
            stream.flush()
            os.fsync(stream.fileno())
    return result


def _prepare_run(
    records: Sequence[SampleRecord],
    output_dir: Path,
    identity: dict[str, object],
    identity_hash: str,
    resume: bool,
) -> list[ScoreRecord]:
    ordered = [record.to_dict() for record in records]
    metadata = {
        "protocol": "ordered_inference_v1",
        "ordered_samples": ordered,
        "input_hash": object_hash(ordered),
        "identity": identity,
        "identity_hash": identity_hash,
        "score_direction": "fake_high",
    }
    metadata_path = output_dir / "metadata.json"
    scores_path = output_dir / "scores.jsonl"
    if metadata_path.exists():
        if not resume:
            raise ValueError("run already exists; use resume with the same identity")
        saved = _read_object(metadata_path.read_bytes())
        if object_hash(saved) != object_hash(metadata):
            raise ValueError(
                "resume rejected: input/config/model/code/runtime identity changed; "
                "use a new output directory"
            )
        if not scores_path.exists() and (output_dir / "report.json").exists():
            raise ValueError("completed run report exists but its scores are missing")
        return _read_prefix(scores_path, records)
    if scores_path.exists() or (output_dir / "report.json").exists():
        raise ValueError("run artifacts exist without metadata; refusing to overwrite")
    if resume:
        raise ValueError("cannot resume a run without metadata")
    atomic_json(metadata_path, metadata)
    return []


def _infer_one(
    record: SampleRecord,
    scorer: Callable[[SampleRecord], float],
    fatal_exceptions: tuple[type[Exception], ...],
) -> ScoreRecord:
    started = time.perf_counter()
    metadata = _sample_metadata(record)
    try:
        if record.status != "available":
            return score_from_dict(
                {
                    **metadata,
                    "score": None,
                    "status": "error",
                    "elapsed_seconds": 0.0,
                    "error_code": record.status,
                    "error_message": record.failure_reason,
                }
            )
        value = scorer(record)
        return score_from_dict(
            {
                **metadata,
                "score": value,
                "status": "ok",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:  # Each media failure remains a persisted row.
        if isinstance(exc, (MemoryError, ImportError, *fatal_exceptions)):
            raise
        code = getattr(exc, "code", "inference_error")
        if code in ("missing_dependency", "process_error"):
            raise
        return score_from_dict(
            {
                **metadata,
                "score": None,
                "status": "error",
                "elapsed_seconds": time.perf_counter() - started,
                "error_code": code
                if isinstance(code, str) and code
                else "inference_error",
                "error_message": str(exc) or type(exc).__name__,
            }
        )


def _append_score(stream: BinaryIO, score: ScoreRecord) -> float:
    started = time.perf_counter()
    stream.write((canonical_json(score.to_dict()) + "\n").encode("utf-8"))
    stream.flush()
    os.fsync(stream.fileno())
    return time.perf_counter() - started


def _report(
    scores: list[ScoreRecord],
    write_seconds: float,
    new_rows: int,
    expected_sample_ids: Sequence[str],
) -> dict[str, object]:
    evaluation = asdict(
        evaluate_records(scores, expected_sample_ids=expected_sample_ids)
    )
    seconds = sum(score.elapsed_seconds for score in scores)
    write_per_row = write_seconds / new_rows if new_rows else None
    raw = (seconds / len(scores) + (write_per_row or 0.0)) * 10000
    return {
        **evaluation,
        "by_modality": {
            key: asdict(value)
            for key, value in grouped_reports(scores, "modality").items()
        },
        "by_source": {
            key: asdict(value)
            for key, value in grouped_reports(scores, "source").items()
        },
        "by_content_type": {
            key: asdict(value)
            for key, value in grouped_reports(scores, "content_type").items()
        },
        "by_generator": {
            key: asdict(value)
            for key, value in grouped_reports(scores, "generator").items()
        },
        "timing": {
            "sample_call_seconds": seconds,
            "current_session_write_seconds": write_seconds,
            "current_session_new_rows": new_rows,
            "samples_per_second": len(scores) / seconds if seconds > 0 else None,
            "T10k_raw_seconds": raw,
            "T10k_safe_seconds": 1.25 * raw,
            "startup_seconds": None,
            "projection_scope": "sample calls and current measured row writes; excludes model startup",
            "representative": False,
            "representativeness_reason": "fewer_than_500_samples"
            if len(scores) < 500
            else "duration_content_distribution_not_verified",
            "write_estimate_available": write_per_row is not None,
        },
        "official_evaluation": False,
        "det_endpoint_note": "null threshold denotes a threshold strictly above max(score)",
    }


def run_inference(  # noqa: PLR0913 -- explicit backend failure boundary
    records: Sequence[SampleRecord],
    scorer: Callable[[SampleRecord], float],
    output_dir: Path,
    *,
    identity: dict[str, object],
    resume: bool = False,
    fatal_exceptions: tuple[type[Exception], ...] = (),
) -> list[ScoreRecord]:
    """Persist one row per input, preserving order and never silently dropping errors.

    The caller supplies config/model/code/runtime identity and must use immutable
    local model/media resources. Interrupts, memory/dependency failures and injected
    backend fatal exceptions propagate after committed rows, without a final report.
    An advisory OS flock guards every mutation; lock-file PID is only diagnostic.
    Stage timings are scoped to newly processed rows in this invocation and are
    additive report fields; persisted score rows and resume identity are unchanged.
    """
    started = time.perf_counter()
    records = tuple(records)
    ids = [record.sample_id for record in records]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("inference requires nonempty input with unique sample IDs")
    if not identity:
        raise ValueError("a nonempty config/model/code identity is required")
    # Validate serializability before making run artifacts.
    identity_hash = object_hash(identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _run_lock(output_dir / ".runner.lock"):
        scores = _prepare_run(records, output_dir, identity, identity_hash, resume)
        new_rows = len(records) - len(scores)
        write_seconds = 0.0
        sample_seconds = 0.0
        instrumented_rows = 0
        stages = InferenceStageTimings()
        with (output_dir / "scores.jsonl").open("ab") as stream:
            setup_seconds = time.perf_counter() - started
            for record in records[len(scores) :]:
                previous_scopes = stages.measured_scopes
                with collect_inference_stages(stages):
                    score = _infer_one(record, scorer, fatal_exceptions)
                instrumented_rows += stages.measured_scopes > previous_scopes
                sample_seconds += score.elapsed_seconds
                write_seconds += _append_score(stream, score)
                scores.append(score)
        report_path = output_dir / "report.json"
        # Preserve the measured write timings when an already complete run resumes.
        if new_rows or not report_path.exists():
            report_started = time.perf_counter()
            report = _report(scores, write_seconds, new_rows, ids)
            timing = cast(dict[str, object], report["timing"])
            stage_seconds: dict[str, float] = dict(stages.seconds.items())
            stage_seconds["unattributed_sample_call"] = max(
                0.0, sample_seconds - sum(stages.seconds.values())
            )
            timing.update(
                {
                    "current_session_setup_seconds": setup_seconds,
                    "current_session_sample_call_seconds": sample_seconds,
                    "current_session_instrumented_rows": instrumented_rows,
                    "current_session_stage_seconds": stage_seconds,
                    "current_session_report_build_seconds": time.perf_counter()
                    - report_started,
                    "current_session_elapsed_seconds_before_report_write": time.perf_counter()
                    - started,
                    "current_session_scope": (
                        "runner entry through report preparation; excludes caller "
                        "preflight/model startup, final report write and lock release; "
                        "stage timings cover only newly processed rows, including "
                        "sample failures; uninstrumented work is unattributed"
                    ),
                    "stage_clock": (
                        "host wall clock, no added CUDA synchronization; built-in "
                        "successful scoring includes blocking CPU feature/score "
                        "copies, transfers and pooling, not kernel-only time; "
                        "failed CUDA calls may leave asynchronous work pending"
                    ),
                }
            )
            atomic_json(report_path, report)
        return scores
