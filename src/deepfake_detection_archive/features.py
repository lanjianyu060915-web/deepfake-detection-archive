"""Shared local-only media preprocessing for frozen feature extraction."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .data.manifest import SampleRecord, file_sha256
from .media import (
    AudioConfig,
    MediaError,
    VideoConfig,
    decode_audio,
    decode_video,
    sample_audio_segments,
)
from .models.encoders import FrozenAudioEncoder, FrozenVisionEncoder
from .timing import inference_stage


Encoder = FrozenAudioEncoder | FrozenVisionEncoder


def validate_encoder_config(config: dict[str, Any]) -> None:
    """Check the encoder boundary's schema without loading a backbone."""
    if config.get("kind") == "audio":
        if set(config) != {"kind", "model_dir", "expected_hashes"}:
            raise ValueError(
                "audio encoder requires kind, model_dir and expected_hashes"
            )
        if (
            not isinstance(config["model_dir"], str)
            or not config["model_dir"].strip()
            or not isinstance(config["expected_hashes"], dict)
        ):
            raise ValueError("invalid audio encoder configuration")
    elif config.get("kind") == "vision":
        if set(config) != {"kind", "architecture", "checkpoint", "expected_sha256"}:
            raise ValueError(
                "vision encoder requires kind, architecture, checkpoint and expected_sha256"
            )
        if any(
            not isinstance(config[name], str) or not config[name].strip()
            for name in ("architecture", "checkpoint", "expected_sha256")
        ):
            raise ValueError("invalid vision encoder configuration")
    else:
        raise ValueError("encoder kind must be audio or vision")


def load_encoder(config: dict[str, Any], base_dir: Path, device: str) -> Encoder:
    """Construct a frozen encoder from explicit local, hash-verified resources."""
    validate_encoder_config(config)
    if config["kind"] == "audio":
        encoder: Encoder = FrozenAudioEncoder(
            base_dir / config["model_dir"], expected_hashes=config["expected_hashes"]
        )
    else:
        encoder = FrozenVisionEncoder(
            config["architecture"],
            base_dir / config["checkpoint"],
            expected_sha256=config["expected_sha256"],
        )
    encoder.to(device)
    encoder.eval()
    return encoder


@inference_stage("input_validation")
def media_path(record: SampleRecord, root: Path) -> Path:
    """Recheck containment and bytes at the consumption boundary."""
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / record.path).resolve(strict=True)
    if not path.is_relative_to(resolved_root):
        raise MediaError("path_escape", "resolved media path escapes data_root")
    if record.sha256 == "unknown":
        raise MediaError("unregistered_hash", "media needs a registered SHA-256")
    if file_sha256(path) != record.sha256:
        raise MediaError("hash_mismatch", "media differs from registered bytes")
    return path


def extract_audio_feature_batch(  # noqa: PLR0913, PLR0917
    records: list[SampleRecord],
    root: Path,
    encoder: FrozenAudioEncoder,
    audio_config: AudioConfig,
    segment_samples: int,
    segments: int,
    device: str,
    batch_size: int,
) -> list[Tensor]:
    """Encode a caller-bounded media chunk, batching segments across recordings.

    Return one CPU float32 [segments, dimensions] tensor per input record in
    order. Decode errors propagate so the caller can record individual failures.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("feature batch_size must be a positive integer")
    if not records:
        return []
    inputs = []
    for record in records:
        if record.modality != "audio":
            raise ValueError("audio feature batch requires audio media")
        audio = decode_audio(media_path(record, root), audio_config)
        inputs.append(
            torch.from_numpy(
                sample_audio_segments(audio.waveform, segment_samples, segments)
            )
        )
    encoder.eval()
    result = []
    with torch.inference_mode():
        for batch in torch.cat(inputs).split(batch_size):  # type: ignore[no-untyped-call]
            features = encoder(batch.to(device), sample_rate=audio_config.sample_rate)
            if (
                features.ndim != 2
                or features.shape[0] != batch.shape[0]
                or not bool(torch.isfinite(features).all())
            ):
                raise ValueError("encoder returned invalid feature shape or values")
            result.append(features.cpu().float())
    # Each cache file must own just its media, rather than serialize the batch storage.
    return [
        features.clone()
        for features in torch.cat(result).reshape(len(records), segments, -1).unbind()
    ]


def extract_features(  # noqa: PLR0913, PLR0917
    record: SampleRecord,
    root: Path,
    encoder: Encoder,
    audio_config: AudioConfig,
    video_config: VideoConfig,
    segment_samples: int,
    segments: int,
    device: str,
    batch_size: int,
) -> Tensor:
    """Return CPU float32 [observations, dimensions], preserving each media.

    Vision decoding uses the explicitly configured FFmpeg resize/crop or pad,
    then the encoder's registered mean/std. This preprocessing choice is shared
    verbatim with inference and recorded in the experiment configuration.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("feature batch_size must be a positive integer")
    path = media_path(record, root)
    with inference_stage("decode_and_preprocess"):
        if record.modality == "audio":
            if not isinstance(encoder, FrozenAudioEncoder):
                raise ValueError("audio media requires an audio encoder")
            audio = decode_audio(path, audio_config)
            inputs = torch.from_numpy(
                sample_audio_segments(audio.waveform, segment_samples, segments)
            )
        else:
            if not isinstance(encoder, FrozenVisionEncoder):
                raise ValueError("video media requires a vision encoder")
            size = tuple(encoder.preprocessing["input_size"])
            if size != (3, video_config.height, video_config.width):
                raise ValueError("video dimensions must match encoder input_size")
            required_mode = encoder.preprocessing.get("resize_mode")
            if required_mode is not None and video_config.resize_mode != required_mode:
                raise ValueError("video resize_mode must match encoder preprocessing")
            video = decode_video(path, video_config)
            inputs = (
                torch.from_numpy(video.frames).permute(0, 3, 1, 2).float().div_(255)
            )
            mean = inputs.new_tensor(encoder.preprocessing["mean"])[None, :, None, None]
            std = inputs.new_tensor(encoder.preprocessing["std"])[None, :, None, None]
            inputs = (inputs - mean) / std
    with inference_stage("model_and_feature_scoring"):
        encoder.eval()
        result: list[Tensor] = []
        with torch.inference_mode():
            for batch in inputs.split(batch_size):  # type: ignore[no-untyped-call]
                if isinstance(encoder, FrozenAudioEncoder):
                    features = encoder(
                        batch.to(device), sample_rate=audio_config.sample_rate
                    )
                else:
                    features = encoder(batch.to(device))
                if (
                    features.ndim != 2
                    or features.shape[0] != batch.shape[0]
                    or not bool(torch.isfinite(features).all())
                ):
                    raise ValueError("encoder returned invalid feature shape or values")
                result.append(features.cpu().float())
        return torch.cat(result)


def extract_video_feature_batch(  # noqa: PLR0913, PLR0917
    records: list[SampleRecord],
    root: Path,
    encoder: FrozenVisionEncoder,
    video_config: VideoConfig,
    device: str,
    batch_size: int,
    num_workers: int,
) -> list[Tensor]:
    """Decode a bounded chunk concurrently, then batch frames across videos.

    Only the main thread runs the encoder. Cache entries own their CPU FP32
    tensors, so serializing one video's features never saves a whole GPU batch.
    """
    if not records:
        return []
    required_mode = encoder.preprocessing.get("resize_mode")
    if required_mode is not None and video_config.resize_mode != required_mode:
        raise ValueError("video resize_mode must match encoder preprocessing")
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        futures = [
            pool.submit(
                extract_video_frames, record, root, video_config, encoder.preprocessing
            )
            for record in records
        ]
        inputs = [future.result() for future in futures]
    with torch.inference_mode():
        features = [
            encoder(frames.to(device)).cpu().float()
            for frames in torch.cat(inputs).split(batch_size)  # type: ignore[no-untyped-call]
        ]
    values = torch.cat(features).reshape(len(records), video_config.n_frames, -1)
    return [value.clone() for value in values.unbind()]


def extract_video_frames(
    record: SampleRecord,
    root: Path,
    video_config: VideoConfig,
    preprocessing: dict[str, Any],
) -> Tensor:
    """Decode one media into normalized frames [frames, 3, height, width].

    Resize/crop policy, channel order and mean/std normalization are the ones
    the frozen encoder path already uses, so a trainable backbone sees exactly
    the preprocessing that the experiment configuration records.
    """
    if record.modality != "video":
        raise ValueError("extract_video_frames requires a video record")
    size = tuple(preprocessing["input_size"])
    if size != (3, video_config.height, video_config.width):
        raise ValueError("video dimensions must match the backbone input_size")
    path = media_path(record, root)
    with inference_stage("decode_and_preprocess"):
        video = decode_video(path, video_config)
        frames = torch.from_numpy(video.frames).permute(0, 3, 1, 2).float().div_(255)
        mean = frames.new_tensor(preprocessing["mean"])[None, :, None, None]
        std = frames.new_tensor(preprocessing["std"])[None, :, None, None]
        result: Tensor = (frames - mean) / std
        if result.ndim != 4 or not bool(torch.isfinite(result).all()):
            raise ValueError("decoded frames must be finite [frames, 3, height, width]")
        return result.contiguous()
