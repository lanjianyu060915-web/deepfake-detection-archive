"""Deterministic, bounded local media decoding with explicit failures.

Video sampling uses decoded presentation timestamps, including VFR sources. Audio
is resampled and downmixed only; no loudness normalization or denoising is applied.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


class MediaError(ValueError):
    """A sample-level failure whose stable code can be recorded in a manifest."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    max_duration_seconds: float = 600.0
    timeout_seconds: float = 30.0
    max_samples: int = 9_600_000
    backend: Literal["ffmpeg", "soundfile"] = "ffmpeg"

    def __post_init__(self) -> None:
        for name in ("sample_rate", "max_samples"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _positive(self.max_duration_seconds, "max_duration_seconds")
        _positive(self.timeout_seconds, "timeout_seconds")
        if self.backend not in ("ffmpeg", "soundfile"):
            raise ValueError("audio backend must be ffmpeg or soundfile")


@dataclass(frozen=True)
class VideoConfig:
    n_frames: int = 16
    height: int = 224
    width: int = 224
    resize_mode: Literal["pad", "center_crop", "native_center_crop"] = "pad"
    max_duration_seconds: float = 600.0
    timeout_seconds: float = 60.0
    max_source_pixels: int = 40_000_000
    max_source_frames: int = 100_000
    max_output_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "n_frames",
            "height",
            "width",
            "max_source_pixels",
            "max_source_frames",
            "max_output_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _positive(self.max_duration_seconds, "max_duration_seconds")
        _positive(self.timeout_seconds, "timeout_seconds")
        if self.resize_mode not in ("pad", "center_crop", "native_center_crop"):
            raise ValueError(
                "resize_mode must be pad, center_crop or native_center_crop"
            )
        if self.n_frames > 1024:
            raise ValueError("n_frames cannot exceed 1024")
        if self.n_frames * self.height * self.width * 3 > self.max_output_bytes:
            raise ValueError("requested video tensor exceeds max_output_bytes")


@dataclass(frozen=True)
class AudioData:
    waveform: NDArray[np.float32]
    sample_rate: int
    original_sample_rate: int
    original_duration_seconds: float


@dataclass(frozen=True)
class VideoData:
    frames: NDArray[np.uint8]
    timestamps_seconds: tuple[float, ...]
    frame_indices: tuple[int, ...]
    requested_timestamps_seconds: tuple[float, ...]
    original_width: int
    original_height: int
    original_duration_seconds: float


def _run(command: list[str], timeout: float, max_bytes: int) -> bytes:
    """Bound wall time and captured output without retaining unbounded pipes."""
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed executable, shell=False
                command, stdin=subprocess.DEVNULL, stdout=output, stderr=errors
            )
        except FileNotFoundError as exc:
            raise MediaError("missing_dependency", f"{command[0]} unavailable") from exc
        except OSError as exc:
            raise MediaError("process_error", str(exc)) from exc
        deadline = time.monotonic() + timeout
        try:
            while True:
                if os.fstat(output.fileno()).st_size > max_bytes:
                    raise MediaError("output_limit", "decoded output exceeds limit")
                if os.fstat(errors.fileno()).st_size > 1_048_576:
                    raise MediaError("decode_error", "decoder emitted excessive errors")
                if process.poll() is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MediaError("timeout", "media subprocess exceeded timeout")
                time.sleep(min(0.02, remaining))
            errors.seek(0)
            diagnostic = errors.read(4096).decode("utf-8", errors="replace").strip()
            if process.returncode != 0 or diagnostic:
                raise MediaError("decode_error", diagnostic or "decoder failed")
            output.seek(0)
            result = output.read(max_bytes + 1)
            if len(result) > max_bytes:
                raise MediaError("output_limit", "decoded output exceeds limit")
            return result
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def _local_file(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise MediaError("missing_file", f"media file does not exist: {path}")
    return str(resolved)


def _probe(path: str, stream: str, timeout: float) -> dict[str, Any]:
    raw = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            stream,
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            path,
        ],
        timeout,
        1_048_576,
    )
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or not data.get("streams"):
            raise MediaError("missing_stream", f"no {stream} stream")
        return dict(data)
    except (ValueError, TypeError) as exc:
        if isinstance(exc, MediaError):
            raise
        raise MediaError("invalid_metadata", "invalid probe metadata") from exc


def _duration(probe: dict[str, Any], maximum: float) -> float:
    try:
        duration = float(
            probe["streams"][0].get("duration", probe.get("format", {}).get("duration"))
        )
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise MediaError("invalid_metadata", "missing media duration") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise MediaError("invalid_metadata", "invalid media duration")
    if duration > maximum:
        raise MediaError("duration_limit", "media duration exceeds configured limit")
    return duration


def _decode_audio_soundfile(source: str, config: AudioConfig) -> AudioData:
    """Read native-rate mono audio in-process; no subprocess timeout applies."""
    import soundfile as sf  # type: ignore[import-untyped]  # noqa: PLC0415

    try:
        with sf.SoundFile(source) as stream:
            rate = int(stream.samplerate)
            if rate != config.sample_rate or stream.channels != 1:
                raise MediaError(
                    "unsupported_audio_format",
                    "soundfile backend requires mono audio at the target sample rate",
                )
            frames = int(stream.frames)
            if frames <= 0:
                raise MediaError("no_samples", "audio contains no complete samples")
            duration = frames / rate
            if duration > config.max_duration_seconds:
                raise MediaError(
                    "duration_limit", "media duration exceeds configured limit"
                )
            limit = min(
                config.max_samples,
                math.ceil(config.max_duration_seconds * config.sample_rate),
            )
            if frames > limit:
                raise MediaError("output_limit", "decoded output exceeds limit")
            waveform: NDArray[np.float32] = stream.read(limit + 1, dtype="float32")
    except (RuntimeError, OSError) as exc:
        raise MediaError("decode_error", str(exc)) from exc
    if waveform.size > limit:
        raise MediaError("output_limit", "decoded output exceeds limit")
    if waveform.size != frames:
        raise MediaError("decode_error", "audio ended before all samples were decoded")
    if not np.isfinite(waveform).all():
        raise MediaError("nonfinite_audio", "decoded audio contains nonfinite samples")
    return AudioData(waveform, rate, rate, duration)


def decode_audio(path: str | Path, config: AudioConfig | None = None) -> AudioData:
    config = config or AudioConfig()
    source = _local_file(path)
    if config.backend == "soundfile":
        return _decode_audio_soundfile(source, config)
    metadata = _probe(source, "a:0", config.timeout_seconds)
    duration = _duration(metadata, config.max_duration_seconds)
    try:
        rate = int(metadata["streams"][0]["sample_rate"])
        channels = int(metadata["streams"][0]["channels"])
    except (ValueError, TypeError, KeyError) as exc:
        raise MediaError("invalid_metadata", "invalid original sample rate") from exc
    if rate <= 0:
        raise MediaError("invalid_metadata", "invalid original sample rate")
    if not 1 <= channels <= 64:
        raise MediaError("channel_limit", "audio channel count outside 1..64")
    # An explicit arithmetic mean preserves identical-channel amplitude, unlike
    # FFmpeg's default energy-preserving stereo mixing matrix.
    downmix = "pan=mono|c0=" + "+".join(
        f"{1 / channels:.17g}*c{index}" for index in range(channels)
    )
    limit = min(
        config.max_samples, math.ceil(config.max_duration_seconds * config.sample_rate)
    )
    raw = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-xerror",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-err_detect",
            "explode",
            "-i",
            source,
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            downmix,
            "-ac",
            "1",
            "-ar",
            str(config.sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        config.timeout_seconds,
        limit * 4,
    )
    if not raw or len(raw) % 4:
        raise MediaError("no_samples", "audio contains no complete samples")
    waveform = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if not np.isfinite(waveform).all():
        raise MediaError("nonfinite_audio", "decoded audio contains nonfinite samples")
    return AudioData(waveform, config.sample_rate, rate, duration)


def _frame_timestamps(source: str, config: VideoConfig) -> NDArray[np.float64]:
    raw = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            source,
        ],
        config.timeout_seconds,
        config.max_source_frames * 256,
    )
    try:
        frames = json.loads(raw)["frames"]
        if not frames:
            raise MediaError("no_frames", "video has no decoded frames")
        if len(frames) > config.max_source_frames:
            raise MediaError("frame_limit", "source frame count exceeds limit")
        timestamps = np.asarray(
            [float(frame["best_effort_timestamp_time"]) for frame in frames],
            dtype=np.float64,
        )
    except (KeyError, ValueError, TypeError) as exc:
        if isinstance(exc, MediaError):
            raise
        raise MediaError(
            "invalid_timestamps", "frame presentation time missing"
        ) from exc
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise MediaError("invalid_timestamps", "frame timestamps invalid or unsorted")
    if timestamps[-1] - timestamps[0] > config.max_duration_seconds:
        raise MediaError("duration_limit", "decoded video timeline exceeds limit")
    return timestamps


def decode_video(path: str | Path, config: VideoConfig | None = None) -> VideoData:
    config = config or VideoConfig()
    source = _local_file(path)
    metadata = _probe(source, "v:0", config.timeout_seconds)
    duration = _duration(metadata, config.max_duration_seconds)
    try:
        width = int(metadata["streams"][0]["width"])
        height = int(metadata["streams"][0]["height"])
    except (ValueError, TypeError, KeyError) as exc:
        raise MediaError("invalid_metadata", "source dimensions missing") from exc
    if width <= 0 or height <= 0 or width * height > config.max_source_pixels:
        raise MediaError("dimension_limit", "source dimensions exceed limits")
    timestamps = _frame_timestamps(source, config)
    targets = np.linspace(timestamps[0], timestamps[-1], config.n_frames)
    right = np.searchsorted(timestamps, targets).clip(0, len(timestamps) - 1)
    left = (right - 1).clip(0, len(timestamps) - 1)
    indices = np.where(
        targets - timestamps[left] <= timestamps[right] - targets, left, right
    )
    unique, inverse = np.unique(indices, return_inverse=True)
    selection = "+".join(f"eq(n\\,{int(index)})" for index in unique)
    mode = "decrease" if config.resize_mode == "pad" else "increase"
    scale = (
        f"scale={config.width}:{config.height}:force_original_aspect_ratio={mode}"
        ":flags=bicubic"
    )
    finish = (
        f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2:black"
        if config.resize_mode == "pad"
        else f"crop={config.width}:{config.height}"
    )
    spatial_filter = f"{scale},{finish}"
    if config.resize_mode == "native_center_crop":
        # Work in displayed dimensions after FFmpeg autorotation. PIL CenterCrop
        # pads small images, then rounds odd center offsets with ties to even.
        # RGB avoids chroma-subsampling rounding of crop/pad offsets.
        crop_x = (
            f"floor((iw-{config.width})/2)"
            f"+mod(floor((iw-{config.width})/2),2)*mod(iw-{config.width},2)"
        )
        crop_y = (
            f"floor((ih-{config.height})/2)"
            f"+mod(floor((ih-{config.height})/2),2)*mod(ih-{config.height},2)"
        )
        spatial_filter = (
            f"format=rgb24,pad='max(iw,{config.width})':'max(ih,{config.height})':"
            "'floor((ow-iw)/2)':'floor((oh-ih)/2)':black,"
            f"crop={config.width}:{config.height}:'{crop_x}':'{crop_y}'"
        )
    raw = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-xerror",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-err_detect",
            "explode",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-an",
            "-filter_threads",
            "1",
            "-vf",
            f"select={selection},{spatial_filter},setsar=1",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        config.timeout_seconds,
        config.max_output_bytes,
    )
    expected = len(unique) * config.height * config.width * 3
    if len(raw) != expected:
        raise MediaError("frame_count_mismatch", "not all selected frames decoded")
    frames = (
        np.frombuffer(raw, dtype=np.uint8)
        .reshape(len(unique), config.height, config.width, 3)[inverse]
        .copy()
    )
    return VideoData(
        frames,
        tuple(float(timestamps[i]) for i in indices),
        tuple(int(i) for i in indices),
        tuple(float(t) for t in targets),
        width,
        height,
        duration,
    )


def sample_audio_segments(
    waveform: NDArray[np.float32], sample_count: int, n_segments: int
) -> NDArray[np.float32]:
    """Uniform start offsets; loop-pad short audio without changing amplitude."""
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if type(n_segments) is not int or n_segments <= 0:
        raise ValueError("n_segments must be a positive integer")
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise MediaError("invalid_audio", "expected nonempty finite mono waveform")
    if sample_count * n_segments * 4 > 512 * 1024 * 1024:
        raise MediaError("output_limit", "requested segments exceed 512 MiB")
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.size < sample_count:
        samples = np.resize(samples, sample_count)
    starts = np.linspace(0, len(samples) - sample_count, n_segments).astype(np.int64)
    return np.stack([samples[start : start + sample_count] for start in starts])
