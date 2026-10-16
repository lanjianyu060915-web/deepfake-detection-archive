"""Self-contained DINOv2 last-block adaptation with a frozen token prefix."""

from dataclasses import dataclass
from typing import Any, Self, cast

import torch
from torch import Tensor, nn

from .core import MediaFeatureClassifier, MediaFeatureConfig


@dataclass(frozen=True)
class DinoAdaptConfig:
    hidden_dim: int = 64
    dropout: float = 0.1
    aggregation: str = "mean"
    trainable_blocks: int = 1

    def __post_init__(self) -> None:
        MediaFeatureConfig(768, self.hidden_dim, self.dropout)
        if self.aggregation != "mean":
            raise ValueError("DINOv2 adaptation requires mean aggregation")
        if type(self.trainable_blocks) is not int or self.trainable_blocks not in (
            0,
            1,
        ):
            raise ValueError("trainable_blocks must be 0 or 1")


class DinoAdaptVideoClassifier(nn.Module):
    """Full backbone plus media head; raw frames and cached prefix tokens share suffix.

    The checkpoint retains the original 518-pixel positional grid. Dynamic input
    interpolation supports the fixed 224-pixel inference frontend. Prefix caches
    may use BF16, which is an approximate training input, not FP32 equivalence.
    """

    def __init__(self, config: DinoAdaptConfig) -> None:
        super().__init__()
        import timm  # noqa: PLC0415

        self.config = config
        self.backbone: Any = cast(
            Any,
            timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
            ),
        )
        self.head = MediaFeatureClassifier(
            MediaFeatureConfig(768, config.hidden_dim, config.dropout)
        )
        self.preprocessing: dict[str, Any] = {
            "input_size": (3, 224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "resize_mode": "native_center_crop",
        }
        self.backbone.requires_grad_(False)
        if config.trainable_blocks:
            self.backbone.blocks[-1].requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
        self.train()

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.backbone.eval()
        self.backbone.blocks[-1].train(mode and bool(self.config.trainable_blocks))
        self.backbone.norm.train(mode and bool(self.config.trainable_blocks))
        return self

    @torch.no_grad()
    def prefix(self, images: Tensor) -> Tensor:
        """Return every CLS/patch token immediately before the final block."""
        hidden = self.backbone.patch_embed(images)
        hidden = self.backbone._pos_embed(hidden)
        hidden = self.backbone.patch_drop(hidden)
        hidden = self.backbone.norm_pre(hidden)
        hidden = self.backbone.blocks[:-1](hidden)
        return cast(Tensor, hidden)

    def _suffix(self, tokens: Tensor) -> Tensor:
        hidden = self.backbone.blocks[-1](tokens)
        hidden = self.backbone.norm(hidden)
        return cast(Tensor, self.backbone.forward_head(hidden, pre_logits=True))

    def forward_tokens(
        self, tokens: Tensor, *, frame_batch_size: int | None = None
    ) -> Tensor:
        batch, frames = tokens.shape[:2]
        parameter = next(self.backbone.parameters())
        flattened = tokens.flatten(0, 1)
        chunks = torch.split(flattened, frame_batch_size or len(flattened))
        features = torch.cat(
            [
                self._suffix(chunk.to(device=parameter.device, dtype=parameter.dtype))
                for chunk in chunks
            ]
        )
        return self.head.forward(features.reshape(batch, frames, 768))

    def forward(self, images: Tensor, *, frame_batch_size: int | None = None) -> Tensor:
        if images.ndim == 4:
            return self.forward_tokens(images, frame_batch_size=frame_batch_size)
        batch, frames = images.shape[:2]
        flattened = images.flatten(0, 1)
        parameter = next(self.backbone.parameters())
        features = torch.cat(
            [
                self._suffix(
                    self.prefix(
                        chunk.to(device=parameter.device, dtype=parameter.dtype)
                    )
                )
                for chunk in torch.split(flattened, frame_batch_size or len(flattened))
            ]
        )
        return self.head.forward(features.reshape(batch, frames, 768))
