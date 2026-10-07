"""Strict JSON/TOML configuration with explicit local paths and version."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from .tracking import parse_json


def load_object(path: Path) -> dict[str, object]:
    value: object
    if path.suffix == ".toml":
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        value = parse_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")  # noqa: TRY004 -- schema
    return cast(dict[str, object], value)


def reject_unknown(value: dict[str, object], allowed: set[str]) -> None:
    if extra := set(value) - allowed:
        raise ValueError(f"unknown configuration fields: {sorted(extra)}")


@dataclass(frozen=True)
class InferenceConfig:
    manifest: str
    data_root: str
    output_dir: str
    modality: str
    checkpoint: str
    checkpoint_sha256: str
    device: str = "cuda"
    model_kind: str = "audio_waveform"
    audio: dict[str, object] = field(default_factory=dict)
    video: dict[str, object] = field(default_factory=dict)
    encoder: dict[str, object] = field(default_factory=dict)
    segments: int = 1
    segment_samples: int = 64600
    aggregation: str = "mean"
    batch_size: int = 8
    seed: int = 42
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported inference schema_version")
        if self.modality not in ("audio", "video"):
            raise ValueError("modality must be audio or video")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.model_kind not in (
            "audio_waveform",
            "audio_features",
            "video_features",
        ):
            raise ValueError("unsupported model_kind")
        if (self.modality == "video") != (self.model_kind == "video_features"):
            raise ValueError("model_kind and modality disagree")
        if self.aggregation not in ("mean", "median", "logsumexp", "trimmed_mean"):
            raise ValueError("unsupported aggregation")
        self._validate_resources()

    def _validate_resources(self) -> None:
        for key in ("manifest", "data_root", "output_dir", "checkpoint"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty path")
        if (
            not isinstance(self.checkpoint_sha256, str)
            or len(self.checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256)
        ):
            raise ValueError("checkpoint_sha256 must be lowercase SHA-256")
        for key in ("segments", "segment_samples", "batch_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be nonnegative integer")
        for key in ("audio", "video", "encoder"):
            if not isinstance(getattr(self, key), dict):
                raise ValueError(f"{key} must be an object")  # noqa: TRY004 -- schema

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_inference_config(path: Path) -> InferenceConfig:
    value = load_object(path)
    try:
        return InferenceConfig(**value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"invalid inference configuration: {exc}") from exc
