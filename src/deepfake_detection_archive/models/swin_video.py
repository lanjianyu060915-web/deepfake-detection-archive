"""Offline Video Swin with a frozen spatiotemporal prefix."""

from dataclasses import dataclass
from typing import Any, Self, cast

import torch
from torch import Tensor, nn

from .core import MediaFeatureClassifier, MediaFeatureConfig


@dataclass(frozen=True)
class SwinVideoConfig:
    hidden_dim: int = 64
    dropout: float = 0.1
    aggregation: str = "mean"
    n_frames: int = 16
    trainable_stages: int = 1

    def __post_init__(self) -> None:
        MediaFeatureConfig(768, self.hidden_dim, self.dropout)
        if self.aggregation != "mean":
            raise ValueError("Video Swin requires mean aggregation")
        if type(self.n_frames) is not int or self.n_frames != 16:
            raise ValueError("Video Swin requires n_frames=16")
        if type(self.trainable_stages) is not int or self.trainable_stages not in (
            0,
            1,
        ):
            raise ValueError("trainable_stages must be 0 or 1")


class SwinVideoClassifier(nn.Module):
    """Pool temporal/spatial features before the binary head.

    The research frontend samples 16 native-center-cropped 224-pixel frames.
    Frame batch hints split complete clips only: even a hint below 16 retains a
    full clip, so adaptive execution cannot shrink a single clip's frame count.
    All backbone modules stay in evaluation mode, including an adapted last
    stage; gradients still flow through the selected stage and final norm.
    """

    def __init__(self, config: SwinVideoConfig) -> None:
        super().__init__()
        from torchvision.models.video import swin3d_t  # type: ignore[import-untyped] # noqa: PLC0415

        self.config = config
        self.backbone: Any = swin3d_t(weights=None)
        self.feature_dim = int(self.backbone.head.in_features)
        self.backbone.head = nn.Identity()
        self.head = MediaFeatureClassifier(
            MediaFeatureConfig(self.feature_dim, config.hidden_dim, config.dropout)
        )
        self.preprocessing: dict[str, Any] = {
            "input_size": (3, 224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "resize_mode": "native_center_crop",
        }
        self.backbone.requires_grad_(False)
        if config.trainable_stages:
            self.backbone.features[-1].requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
        self.train()

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def prefix(self, images: Tensor) -> Tensor:
        """Cache FP32 final-stage inputs as [batch, time * 7 * 7, channels]."""
        hidden = self.backbone.patch_embed(images.permute(0, 2, 1, 3, 4))
        hidden = self.backbone.pos_drop(hidden)
        hidden = self.backbone.features[:-1](hidden)
        return cast(Tensor, hidden.flatten(1, 3))

    def _suffix(self, tokens: Tensor) -> Tensor:
        hidden = tokens.reshape(-1, self.config.n_frames // 2, 7, 7, self.feature_dim)
        hidden = self.backbone.features[-1](hidden)
        hidden = self.backbone.norm(hidden)
        return cast(Tensor, hidden.mean(dim=(1, 2, 3)))

    def _clip_batch_size(self, batch: int, frame_batch_size: int | None) -> int:
        return batch if frame_batch_size is None else max(1, frame_batch_size // 16)

    def encode_tokens(
        self, tokens: Tensor, *, frame_batch_size: int | None = None
    ) -> Tensor:
        parameter = next(self.backbone.parameters())
        chunks = torch.split(
            tokens, self._clip_batch_size(len(tokens), frame_batch_size)
        )
        return torch.cat(
            [
                self._suffix(chunk.to(device=parameter.device, dtype=parameter.dtype))
                for chunk in chunks
            ]
        ).unsqueeze(1)

    def encode(self, images: Tensor, *, frame_batch_size: int | None = None) -> Tensor:
        parameter = next(self.backbone.parameters())
        chunks = torch.split(
            images, self._clip_batch_size(len(images), frame_batch_size)
        )
        return torch.cat(
            [
                self._suffix(
                    self.prefix(
                        chunk.to(device=parameter.device, dtype=parameter.dtype)
                    )
                )
                for chunk in chunks
            ]
        ).unsqueeze(1)

    def forward_tokens(
        self, tokens: Tensor, *, frame_batch_size: int | None = None
    ) -> Tensor:
        return self.head.forward(
            self.encode_tokens(tokens, frame_batch_size=frame_batch_size)
        )

    def forward(self, images: Tensor, *, frame_batch_size: int | None = None) -> Tensor:
        if images.ndim == 3:
            return self.forward_tokens(images, frame_batch_size=frame_batch_size)
        return self.head.forward(self.encode(images, frame_batch_size=frame_batch_size))
