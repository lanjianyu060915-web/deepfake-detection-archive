"""Prepare paired dev media under fixed compression/resampling conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)


TEMPORAL_CANONICAL = "video_h264_crf23_448x256_16fps_3p5s"


CONDITIONS = {
    TEMPORAL_CANONICAL: (
        "video",
        "mp4",
        "h264",
        (
            "-map",
            "0:v:0",
            "-an",
            "-t",
            "3.5",
            "-vf",
            "setpts=PTS-STARTPTS,scale=448:256:flags=bicubic,setsar=1,fps=16",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-fps_mode",
            "passthrough",
        ),
    ),
    "audio_aac64k": (
        "audio",
        "m4a",
        "aac",
        (
            "-map",
            "0:a:0",
            "-vn",
            "-ar",
            "16000",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
        ),
    ),
    "audio_bandlimit8k": (
        "audio",
        "flac",
        "flac",
        (
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            "aresample=8000,aresample=16000",
            "-ar",
            "16000",
            "-c:a",
            "flac",
        ),
    ),
    "video_h264_crf23": (
        "video",
        "mp4",
        "h264",
        (
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-fps_mode",
            "passthrough",
        ),
    ),
    "video_h264_crf35": (
        "video",
        "mp4",
        "h264",
        (
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "35",
            "-fps_mode",
            "passthrough",
        ),
    ),
    # Size-canonical conditions: both labels get one shared output size, codec and
    # pixel format, so residual width/height differences cannot drive the score.
    "video_h264_crf18_448x256": (
        "video",
        "mp4",
        "h264",
        (
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "scale=448:256:flags=bicubic",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-fps_mode",
            "passthrough",
        ),
    ),
    "video_h264_crf35_448x256": (
        "video",
        "mp4",
        "h264",
        (
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "scale=448:256:flags=bicubic",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "35",
            "-fps_mode",
            "passthrough",
        ),
    ),
}


def video_timing(path: Path, timeout: float) -> tuple[float, int | None]:
    """Probe actual video duration; frame count also verifies canonical output."""
    completed = subprocess.run(  # noqa: S603
        [  # noqa: S607 -- configured local media executable
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    data = json.loads(completed.stdout)
    stream = data["streams"][0]
    duration = float(stream.get("duration", data.get("format", {}).get("duration")))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video duration must be finite and positive")
    frames = stream.get("nb_frames")
    return duration, int(frames) if frames is not None and frames != "N/A" else None


def transform(
    row: SampleRecord,
    data_root: Path,
    output: Path,
    condition: str,
    timeout: float,
) -> SampleRecord:
    modality, container, codec, arguments = CONDITIONS[condition]
    name = hashlib.sha256(row.sample_id.encode("utf-8")).hexdigest()
    relative = f"media/{name}.{container}"
    target = output / relative
    derived = replace(
        row,
        path=relative,
        sha256="unknown",
        container=container,
        codec=codec,
        sample_rate=16000 if modality == "audio" else None,
        duration_seconds=None,
        frame_rate=None,
        decode_status="unknown",
    )
    if row.status != "available" or row.decode_status == "failed":
        return replace(
            derived,
            status="decode_failed",
            decode_status="failed",
            failure_reason=row.failure_reason,
        )
    try:
        if condition == TEMPORAL_CANONICAL:
            duration, _ = video_timing(data_root / row.path, timeout)
            if duration < 3.5:
                raise ValueError(  # noqa: TRY301 -- persist per-sample failure
                    f"video is shorter than 3.5 seconds: {duration}"
                )
        subprocess.run(  # noqa: S603 - media paths are separate arguments
            [  # noqa: S607 - use the configured local FFmpeg
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-xerror",
                "-threads",
                "1",
                "-err_detect",
                "explode",
                "-i",
                str(data_root / row.path),
                *arguments,
                "-threads",
                "1",
                "-filter_threads",
                "1",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        if condition == TEMPORAL_CANONICAL:
            duration, frames = video_timing(target, timeout)
            if duration != 3.5 or frames != 56:
                raise ValueError(  # noqa: TRY301 -- remove invalid output and persist failure
                    "canonical video must contain exactly 56 frames / 3.5s"
                )
            derived = replace(derived, duration_seconds=3.5, frame_rate=16)
        return replace(
            derived,
            sha256=file_sha256(target),
            status="available",
            failure_reason="unknown",
        )
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ) as exc:
        target.unlink(missing_ok=True)
        reason = str(exc)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            reason += f": {exc.stderr.strip()}"
        return replace(
            derived,
            status="decode_failed",
            decode_status="failed",
            failure_reason=reason,
        )


def prepare(
    manifest: Path,
    data_root: Path,
    output_dir: Path,
    conditions: list[str],
    timeout_seconds: float = 60,
) -> dict[str, Any]:
    """Validate the experiment once, then keep every row in each condition."""
    started = time.monotonic()
    records = load_manifest(manifest)
    if any(row.split != "dev" for row in records):
        raise ValueError("robustness preparation requires only dev rows")
    modalities = {row.modality for row in records}
    if len(modalities) != 1:
        raise ValueError("manifest must contain a single modality")
    if {row.label for row in records} != {0, 1}:
        raise ValueError("dev manifest must contain both binary labels 0 and 1")
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("choose nonempty, distinct conditions")
    for condition in conditions:
        if condition not in CONDITIONS or CONDITIONS[condition][0] not in modalities:
            raise ValueError(f"condition does not match manifest modality: {condition}")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    version = subprocess.run(
        ["ffmpeg", "-version"],  # noqa: S607 - configured local FFmpeg
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    ).stdout.splitlines()[0]
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    baseline = output_dir / "baseline.jsonl"
    baseline_hash = write_manifest(records, baseline)
    summary: dict[str, Any] = {
        "input_manifest": str(manifest.resolve()),
        "input_manifest_hash": baseline_hash,
        "data_root": str(data_root),
        "ffmpeg_version": version,
        "timeout_seconds": timeout_seconds,
        "baseline": {
            "path": str(baseline),
            "sha256": baseline_hash,
            "data_root": str(data_root),
        },
        "conditions": {},
    }
    for condition in conditions:
        condition_started = time.monotonic()
        destination = output_dir / condition
        (destination / "media").mkdir(parents=True)
        derived = [
            transform(row, data_root, destination, condition, timeout_seconds)
            for row in records
        ]
        path = destination / "manifest.jsonl"
        digest = write_manifest(derived, path)
        errors = {
            row.sample_id: row.failure_reason
            for row in derived
            if row.status != "available"
        }
        summary["conditions"][condition] = {
            "path": str(path),
            "sha256": digest,
            "data_root": str(destination),
            "parameters": list(CONDITIONS[condition][3]),
            "sample_count": len(derived),
            "success_count": len(derived) - len(errors),
            "failure_count": len(errors),
            "errors": errors,
            "elapsed_seconds": time.monotonic() - condition_started,
        }
    summary["elapsed_seconds"] = time.monotonic() - started
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition", action="append", choices=CONDITIONS, required=True
    )
    parser.add_argument("--timeout-seconds", type=float, default=60)
    args = parser.parse_args()
    summary = prepare(
        args.manifest,
        args.data_root,
        args.output_dir,
        args.condition,
        args.timeout_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
