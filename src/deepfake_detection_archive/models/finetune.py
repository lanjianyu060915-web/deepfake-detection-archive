"""Partially trainable video frame classifier for local offline fine-tuning.

Only the registered trainable prefix and the media head receive gradients.
Every other backbone submodule stays in evaluation mode, so fine-tuning never
updates their batch-normalization statistics or dropout state. Nothing is
downloaded: the backbone state dict comes from a hash-verified local file and
the same preprocessing contract is reused at inference time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import torch
from torch import Tensor, nn

from .core import FeatureHead, HeadConfig, PoolingMethod, pool_media_scores


@dataclass(frozen=True)
class VideoFrameConfig:
    architecture: str
    encoder_checkpoint: str
    expected_sha256: str
    hidden_dim: int = 64
    dropout: float = 0.1
    aggregation: PoolingMethod = "mean"
    trim_fraction: float = 0.1
    trainable_prefix: str = "layer4"

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, str) or not self.architecture.strip():
            raise ValueError("architecture must be a nonempty name")
        if (
            not isinstance(self.encoder_checkpoint, str)
            or not self.encoder_checkpoint.strip()
        ):
            raise ValueError("encoder_checkpoint must be a nonempty path")
        if (
            not isinstance(self.expected_sha256, str)
            or len(self.expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.trainable_prefix, str) or not self.trainable_prefix:
            raise ValueError("trainable_prefix must be a nonempty module name")
        if type(self.hidden_dim) is not int or self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.aggregation not in ("mean", "median", "logsumexp", "trimmed_mean"):
            raise ValueError(f"unsupported aggregation: {self.aggregation}")
        if not 0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must lie in [0, 0.5)")


class VideoFrameClassifier(nn.Module):
    """Media-level fake logit from a partially fine-tuned frame backbone.

    The forward pass accepts an already preprocessed frame batch with shape
    [batch, frames, 3, height, width] and returns one logit per media. Frames
    are encoded independently and aggregated by the registered pooling method,
    so the model never depends on frame order beyond that aggregation. Callers
    that need bounded scores may apply a sigmoid to the media logit.
    """

    def __init__(self, config: VideoFrameConfig) -> None:
        super().__init__()
        import timm  # noqa: PLC0415
        from timm.data.config import resolve_model_data_config  # noqa: PLC0415

        # Imported lazily: checkpoint registers this architecture, so a
        # module-level import would close a cycle.
        from .checkpoint import verify_sha256  # noqa: PLC0415

        checkpoint = Path(config.encoder_checkpoint)
        verify_sha256(checkpoint, config.expected_sha256)
        self.backbone = timm.create_model(
            config.architecture, pretrained=False, num_classes=0
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(
            not isinstance(key, str) or not isinstance(value, Tensor)
            for key, value in state.items()
        ):
            raise ValueError("vision checkpoint must contain a bare tensor state_dict")
        if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
            raise ValueError("vision checkpoint contains non-finite tensors")
        self.backbone.load_state_dict(state, strict=True)
        if not hasattr(self.backbone, config.trainable_prefix):
            raise ValueError("trainable_prefix is not a backbone submodule")
        self.head = FeatureHead(
            HeadConfig(int(cast(int, self.backbone.num_features)), config.hidden_dim,
                       config.dropout)
        )
        self.preprocessing: dict[str, Any] = resolve_model_data_config(  # type: ignore[no-untyped-call]
            self.backbone
        )
        self.config = config
        self.apply_trainable_policy()

    def _is_trainable(self, qualified_name: str) -> bool:
        root = f"backbone.{self.config.trainable_prefix}"
        return qualified_name in (root, "head") or qualified_name.startswith(
            (root + ".", "head.")
        )

    def apply_trainable_policy(self) -> None:
        """Freeze everything except the trainable prefix and the media head."""
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(self._is_trainable(name))
        self._set_module_modes(False)

    def _set_module_modes(self, mode: bool) -> None:
        for name, module in self.named_children():
            if name != "backbone":
                module.train(mode)
                continue
            for child_name, child in module.named_children():
                child.train(mode and child_name == self.config.trainable_prefix)

    def train(self, mode: bool = True) -> Self:
        if not isinstance(mode, bool):
            raise TypeError("training mode must be a bool")
        super().train(mode)
        self._set_module_modes(mode)
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[2] != 3 or min(images.shape) == 0:
            raise ValueError(
                "images must have nonempty shape [batch, frames, 3, height, width]"
            )
        if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
            raise ValueError("images must contain finite floating-point values")
        batch, frames = images.shape[0], images.shape[1]
        flattened = images.reshape(batch * frames, *images.shape[2:])
        logits = self.head.forward(self.backbone(flattened)).reshape(batch, frames)
        return pool_media_scores(
            logits, None, self.config.aggregation, self.config.trim_fraction
        )
