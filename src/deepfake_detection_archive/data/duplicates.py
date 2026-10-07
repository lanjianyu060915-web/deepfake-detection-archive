"""Conservative near-duplicate candidates, never inferred speaker/group labels.

Video pHash is not rotation invariant and may match unrelated similar scenes.
Audio summaries compare whole aligned clips, not speaker identity; trimming,
time stretching and local copies may be missed. Review candidates with provenance
before assigning shared groups. Low-information media remain explicitly unaudited.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from deepfake_detection_archive.media import (
    AudioConfig,
    MediaError,
    VideoConfig,
    decode_audio,
    decode_video,
)

from .manifest import UNKNOWN, SampleRecord, file_sha256, manifest_hash


@dataclass(frozen=True)
class DuplicateSample:
    sample_id: str
    modality: str
    source_sha256: str
    fingerprint_sha256: str
    method: str
    status: str
    reason: str
    duration_seconds: float | None
    identity_id: str
    identity_verified: bool = False
    timestamps_seconds: tuple[float, ...] = ()


@dataclass(frozen=True)
class DuplicatePair:
    left_id: str
    right_id: str
    modality: str
    similarity: float
    cross_split: bool
    label_conflict: bool
    evidence: str = "near_duplicate_candidate_only"


@dataclass(frozen=True)
class DuplicateFailure:
    sample_id: str
    code: str
    message: str


@dataclass(frozen=True)
class DuplicateAudit:
    manifest_hash: str
    candidate_pairs: tuple[DuplicatePair, ...]
    samples: tuple[DuplicateSample, ...]
    failures: tuple[DuplicateFailure, ...]
    min_similarity: float
    max_duration_ratio: float
    max_pairs: int
    compared_pairs: int
    audio_config: dict[str, object]
    video_config: dict[str, object]
    grouping_applied: bool = False
    speaker_verification_performed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Fingerprint:
    record: SampleRecord
    duration: float
    values: NDArray[np.float64]


def image_phash(image: NDArray[np.uint8]) -> tuple[int, ...]:
    """63 DCT AC bits from an RGB image, with explicit flat-image rejection.

    This is an image helper for video keyframes; it does not add an image modality
    to the competition manifest. The hash includes no crop, face detection or rotation.
    """
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise MediaError("invalid_image", "expected RGB image of at least 2x2 pixels")
    if image.dtype != np.uint8:
        raise MediaError("invalid_image", "expected uint8 RGB image")
    gray = np.asarray(image, dtype=np.float64) @ np.array([0.299, 0.587, 0.114])
    if float(gray.std()) < 3.0:
        raise MediaError("low_information", "nearly flat video keyframe")
    # Separable linear interpolation keeps this deterministic without a new codec
    # or imaging dependency. DCT-II is computed directly on a fixed 32x32 grid.
    x = np.linspace(0, gray.shape[1] - 1, 32)
    y = np.linspace(0, gray.shape[0] - 1, 32)
    horizontal = np.stack([np.interp(x, np.arange(gray.shape[1]), row) for row in gray])
    resized = np.stack(
        [np.interp(y, np.arange(gray.shape[0]), column) for column in horizontal.T],
        axis=1,
    )
    transform = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32) + 0.5))
    coefficients = (transform @ resized @ transform.T).ravel()[1:]
    return tuple(int(bit) for bit in coefficients > np.median(coefficients))


def _audio_fingerprint(waveform: NDArray[np.float32]) -> NDArray[np.float64]:
    if len(waveform) < 2048 or float(np.std(waveform)) < 1e-5:
        raise MediaError("low_information", "silent, constant or too-short audio")
    # Sixteen uniform chunks retain temporal ordering. Average spectra over
    # overlapping 512-sample windows are resilient to codec and resampling noise.
    chunks = np.array_split(waveform.astype(np.float64), 16)
    spectral: list[NDArray[np.float64]] = []
    envelope: list[float] = []
    window = np.hanning(512)
    band_edges = np.linspace(1, 257, 33, dtype=np.int64)
    for original_chunk in chunks:
        chunk = (
            np.pad(original_chunk, (0, 512 - len(original_chunk)), mode="wrap")
            if len(original_chunk) < 512
            else original_chunk
        )
        frames = np.lib.stride_tricks.sliding_window_view(chunk, 512)[::256]
        power = np.mean(np.abs(np.fft.rfft(frames * window, axis=1)) ** 2, axis=0)
        bands = np.array(
            [np.mean(power[start:stop]) for start, stop in pairwise(band_edges)]
        )
        bands /= max(float(bands.sum()), 1e-20)
        log_bands = np.log(bands + 1e-7)
        spectral.append(log_bands - log_bands.mean())
        envelope.append(float(np.sqrt(np.mean(chunk**2))))
    matrix = np.stack(spectral)
    variation = float(np.linalg.norm(matrix - matrix.mean(axis=0)))
    if variation / max(float(np.linalg.norm(matrix)), 1e-20) < 0.01:
        raise MediaError(
            "low_information", "stationary spectrum lacks temporal evidence"
        )
    energy = np.array(envelope)
    energy = np.log(energy / max(float(energy.mean()), 1e-20) + 1e-7)
    features = np.concatenate([matrix.ravel(), energy * 4])
    norm = float(np.linalg.norm(features))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise MediaError("low_information", "audio fingerprint has no finite variation")
    return features / norm


def _fingerprint(
    record: SampleRecord, root: Path, audio: AudioConfig, video: VideoConfig
) -> tuple[DuplicateSample, _Fingerprint | None, DuplicateFailure | None]:
    method = (
        "audio_chunk_log_spectrum_v1"
        if record.modality == "audio"
        else "video_phash_v1"
    )
    checksum = UNKNOWN
    duration: float | None = None
    timestamps: tuple[float, ...] = ()
    try:
        if record.status != "available":
            raise MediaError("unavailable", record.failure_reason)  # noqa: TRY301
        path = (root / record.path).resolve()
        if not path.is_relative_to(root):
            raise MediaError("path_escape", "resolved media path escapes root")  # noqa: TRY301
        if not path.is_file():
            raise MediaError("missing_file", record.path)  # noqa: TRY301
        checksum = file_sha256(path)
        if record.sha256 not in (UNKNOWN, checksum):
            raise MediaError("hash_mismatch", "file differs from registered sha256")  # noqa: TRY301
        if record.modality == "audio":
            decoded_audio = decode_audio(path, audio)
            duration = len(decoded_audio.waveform) / decoded_audio.sample_rate
            values = _audio_fingerprint(decoded_audio.waveform)
        else:
            decoded_video = decode_video(path, video)
            duration = decoded_video.original_duration_seconds
            timestamps = decoded_video.timestamps_seconds
            # Padding itself must not turn a pure-color clip into visual evidence.
            # Inspect the interior of the original picture before hashing pads.
            if video.resize_mode == "pad":
                ratio = min(
                    video.width / decoded_video.original_width,
                    video.height / decoded_video.original_height,
                )
                content_width = max(1, round(decoded_video.original_width * ratio))
                content_height = max(1, round(decoded_video.original_height * ratio))
                left = (video.width - content_width) // 2
                top = (video.height - content_height) // 2
                margin_x, margin_y = content_width // 10, content_height // 10
                for frame in decoded_video.frames:
                    interior = frame[
                        top + margin_y : top + content_height - margin_y,
                        left + margin_x : left + content_width - margin_x,
                    ]
                    image_phash(interior)
            # Reject rather than drop flat frames: every requested keyframe must
            # carry evidence, preventing unrelated title/black screens matching.
            values = np.asarray(
                [image_phash(frame) for frame in decoded_video.frames], dtype=np.float64
            )
        digest = hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
        sample = DuplicateSample(
            record.sample_id,
            record.modality,
            checksum,
            digest,
            method,
            "fingerprinted",
            "candidate search only; identity remains unverified",
            duration,
            record.identity_id,
            timestamps_seconds=timestamps,
        )
        return sample, _Fingerprint(record, duration, values), None
    except (MediaError, OSError) as exc:
        code = exc.code if isinstance(exc, MediaError) else "file_read_error"
        message = exc.message if isinstance(exc, MediaError) else str(exc)
        sample = DuplicateSample(
            record.sample_id,
            record.modality,
            checksum,
            UNKNOWN,
            method,
            "low_information" if code == "low_information" else "failed",
            message,
            duration,
            record.identity_id,
            timestamps_seconds=timestamps,
        )
        return sample, None, DuplicateFailure(record.sample_id, code, message)


def find_near_duplicates(  # noqa: PLR0912, PLR0913 - explicit audit policy arguments
    records: Sequence[SampleRecord],
    root: Path,
    *,
    audio_config: AudioConfig | None = None,
    video_config: VideoConfig | None = None,
    min_similarity: float = 0.97,
    max_pairs: int = 2_000_000,
    max_duration_ratio: float = 1.03,
) -> DuplicateAudit:
    """Return review candidates without modifying records or assigning groups.

    All same-modality pairs are searched within a duration ratio. The conservative
    quadratic-work guard is checked before decoding; larger collections require an
    explicitly raised budget (or a future indexed search), never silent subsampling.
    """
    if not math.isfinite(min_similarity) or not 0 < min_similarity <= 1:
        raise ValueError("min_similarity must be in (0, 1]")
    if not math.isfinite(max_duration_ratio) or max_duration_ratio < 1:
        raise ValueError("max_duration_ratio must be finite and at least 1")
    if type(max_pairs) is not int or max_pairs <= 0:
        raise ValueError("max_pairs must be a positive integer")
    if len({r.sample_id for r in records}) != len(records):
        raise ValueError("duplicate sample_id in records")
    counts = Counter(r.modality for r in records)
    upper_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    if upper_pairs > max_pairs:
        raise ValueError(f"pair budget exceeded: {upper_pairs} > {max_pairs}")
    audio = audio_config or AudioConfig()
    video = video_config or VideoConfig(n_frames=8, height=64, width=64)
    root = root.resolve()
    samples: list[DuplicateSample] = []
    fingerprints: list[_Fingerprint] = []
    failures: list[DuplicateFailure] = []
    for record in records:
        sample, fingerprint, failure = _fingerprint(record, root, audio, video)
        samples.append(sample)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
        if failure is not None:
            failures.append(failure)
    pairs: list[DuplicatePair] = []
    compared = 0
    for index, left in enumerate(fingerprints):
        for right in fingerprints[index + 1 :]:
            if left.record.modality != right.record.modality:
                continue
            if (
                max(left.duration, right.duration) / min(left.duration, right.duration)
                > max_duration_ratio
            ):
                continue
            compared += 1
            if left.record.modality == "audio":
                similarity = float(np.clip(np.dot(left.values, right.values), 0, 1))
            else:
                # Minimum frame similarity prevents one coincidentally shared
                # frame from dominating whole-video candidates.
                similarity = float((left.values == right.values).mean(axis=1).min())
            if similarity < min_similarity:
                continue
            splits = {left.record.split, right.record.split} - {"unassigned"}
            pairs.append(
                DuplicatePair(
                    left.record.sample_id,
                    right.record.sample_id,
                    left.record.modality,
                    similarity,
                    len(splits) > 1,
                    left.record.label is not None
                    and right.record.label is not None
                    and left.record.label != right.record.label,
                )
            )
    return DuplicateAudit(
        manifest_hash(list(records)),
        tuple(pairs),
        tuple(samples),
        tuple(failures),
        min_similarity,
        max_duration_ratio,
        max_pairs,
        compared,
        asdict(audio),
        asdict(video),
    )
