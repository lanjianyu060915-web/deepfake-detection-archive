"""Versioned tensor-only checkpoints with fixed constructors and file digests."""

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .aasist import AASISTLite
from .ast import ASTClassifier, ASTConfig
from .core import (
    FeatureHead,
    HeadConfig,
    LogMelCNN,
    LogMelConfig,
    MediaFeatureClassifier,
    MediaFeatureConfig,
)
from .finetune import VideoFrameClassifier, VideoFrameConfig
from .ufd import UFDConfig, UFDVideoClassifier
from .waverep import WaveRepConfig, WaveRepVideoClassifier


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256: str) -> None:
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")


def create_model(architecture: str, config: dict[str, Any]) -> nn.Module:  # noqa: PLR0911
    """Instantiate only built-in architectures; never resolve arbitrary classes."""
    if architecture == "logmel_cnn_v1":
        return LogMelCNN(LogMelConfig(**config))
    if architecture == "feature_head_v1":
        return FeatureHead(HeadConfig(**config))
    if architecture == "media_feature_head_v1":
        return MediaFeatureClassifier(MediaFeatureConfig(**config))
    if architecture == "video_frame_finetune_v1":
        return VideoFrameClassifier(VideoFrameConfig(**config))
    if architecture == "ufd_video_v1":
        return UFDVideoClassifier(UFDConfig(**config))
    if architecture == "waverep_video_v1":
        return WaveRepVideoClassifier(WaveRepConfig(**config))
    if architecture == "aasist_l_v1" and config == {}:
        return AASISTLite()
    if architecture == "audio_ast_v1":
        return ASTClassifier(ASTConfig(**config))
    raise ValueError(f"unsupported architecture/config: {architecture}")


def model_spec(model: nn.Module) -> tuple[str, dict[str, Any]]:  # noqa: PLR0911
    if type(model) is LogMelCNN:
        return "logmel_cnn_v1", asdict(model.config)
    if type(model) is FeatureHead:
        return "feature_head_v1", asdict(model.config)
    if type(model) is MediaFeatureClassifier:
        return "media_feature_head_v1", asdict(model.config)
    if type(model) is VideoFrameClassifier:
        return "video_frame_finetune_v1", asdict(model.config)
    if type(model) is UFDVideoClassifier:
        return "ufd_video_v1", asdict(model.config)
    if type(model) is WaveRepVideoClassifier:
        return "waverep_video_v1", asdict(model.config)
    if type(model) is AASISTLite:
        return "aasist_l_v1", {}
    if type(model) is ASTClassifier:
        return "audio_ast_v1", asdict(model.config)
    raise ValueError("checkpoint supports only registered model types")


def save_checkpoint(path: Path, model: nn.Module) -> dict[str, Any]:
    """Atomically save CPU tensors and a JSON descriptor; return the descriptor.

    A caller must separately retain its trusted descriptor/hash. A same-directory
    sidecar documents integrity but is not a signature against malicious edits.
    Experiment metadata and training-resume state belong to the runner.
    """
    architecture, config = model_spec(model)
    state_dict = {
        key: tensor.detach().cpu().contiguous()
        for key, tensor in model.state_dict().items()
    }
    if any(not bool(torch.isfinite(tensor).all()) for tensor in state_dict.values()):
        raise ValueError("cannot save a checkpoint with non-finite tensors")
    descriptor: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "architecture": architecture,
        "config": config,
        "score_direction": "higher_is_fake",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save({**descriptor, "state_dict": state_dict}, stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    descriptor["sha256"] = sha256_file(path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return descriptor


def load_checkpoint(
    path: Path, *, expected_sha256: str
) -> tuple[nn.Module, dict[str, Any]]:
    """Verify a trusted hash before safe deserialization and strict state loading."""
    verify_sha256(path, expected_sha256)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "architecture",
        "config",
        "score_direction",
        "state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("invalid checkpoint schema fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema version")
    if payload["score_direction"] != "higher_is_fake":
        raise ValueError("unsupported checkpoint score direction")
    if not isinstance(payload["architecture"], str) or not isinstance(
        payload["config"], dict
    ):
        raise TypeError("invalid checkpoint architecture/config")
    state = payload["state_dict"]
    if (
        not isinstance(state, dict)
        or not state
        or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state.items()
        )
    ):
        raise ValueError("state_dict must be a nonempty string-to-tensor mapping")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
        raise ValueError("checkpoint contains non-finite tensors")
    model = create_model(payload["architecture"], payload["config"])
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    metadata["sha256"] = expected_sha256
    return model, cast(dict[str, Any], metadata)
