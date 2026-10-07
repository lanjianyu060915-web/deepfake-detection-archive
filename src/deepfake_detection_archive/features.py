"""Shared local-only media preprocessing for frozen feature extraction."""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .data.manifest import SampleRecord
from .media import (
    AudioConfig,
    VideoConfig,
    decode_audio,
    decode_video,
    sample_audio_segments,
)
from .models.checkpoint import verify_sha256
from .models.encoders import FrozenAudioEncoder, FrozenVisionEncoder


Encoder = FrozenAudioEncoder | FrozenVisionEncoder


def load_encoder(config: dict[str, Any], base_dir: Path, device: str) -> Encoder:
    """Construct a frozen encoder from explicit local, hash-verified resources."""
    if config.get("kind") == "audio":
        if set(config) != {"kind", "model_dir", "expected_hashes"}:
            raise ValueError(
                "audio encoder requires kind, model_dir and expected_hashes"
            )
        if not isinstance(config["model_dir"], str) or not isinstance(
            config["expected_hashes"], dict
        ):
            raise ValueError("invalid audio encoder configuration")
        encoder: Encoder = FrozenAudioEncoder(
            base_dir / config["model_dir"], expected_hashes=config["expected_hashes"]
        )
    elif config.get("kind") == "vision":
        if set(config) != {"kind", "architecture", "checkpoint", "expected_sha256"}:
            raise ValueError(
                "vision encoder requires kind, architecture, checkpoint and expected_sha256"
            )
        if any(
            not isinstance(config[name], str)
            for name in ("architecture", "checkpoint", "expected_sha256")
        ):
            raise ValueError("invalid vision encoder configuration")
        encoder = FrozenVisionEncoder(
            config["architecture"],
            base_dir / config["checkpoint"],
            expected_sha256=config["expected_sha256"],
        )
    else:
        raise ValueError("encoder kind must be audio or vision")
    encoder.to(device)
    encoder.eval()
    return encoder


def media_path(record: SampleRecord, root: Path) -> Path:
    """Recheck containment and bytes at the consumption boundary."""
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / record.path).resolve(strict=True)
    if not path.is_relative_to(resolved_root):
        raise ValueError("media path escapes data_root")
    verify_sha256(path, record.sha256)
    return path


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
        video = decode_video(path, video_config)
        inputs = torch.from_numpy(video.frames).permute(0, 3, 1, 2).float().div_(255)
        mean = inputs.new_tensor(encoder.preprocessing["mean"])[None, :, None, None]
        std = inputs.new_tensor(encoder.preprocessing["std"])[None, :, None, None]
        inputs = (inputs - mean) / std
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
