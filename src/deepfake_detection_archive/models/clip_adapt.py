"""Local CLIP last-block adaptation with reusable frozen prefix tokens."""

from dataclasses import dataclass
from typing import Self, cast

import torch
from torch import Tensor, nn

from .core import FeatureHead, HeadConfig
from .ufd import UFDConfig, UFDVideoClassifier


@dataclass(frozen=True)
class ClipAdaptConfig(UFDConfig):
    hidden_dim: int = 64
    dropout: float = 0.1
    train_last_block: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.train_last_block, bool):
            raise TypeError("train_last_block must be a bool")


class ClipAdaptVideoClassifier(UFDVideoClassifier):
    """Train the final visual block and MLP; keep the prefix/projection frozen.

    Raw input is [batch, frames, 3, height, width]. Cached prefix input is
    [batch, frames, tokens, hidden_size], preserving every CLS/patch token.
    Projected features remain unnormalized, as in the UFD inference path.
    """

    config: ClipAdaptConfig

    def __init__(self, config: ClipAdaptConfig) -> None:
        super().__init__(config)
        self.head = FeatureHead(
            HeadConfig(
                self.backbone.config.projection_dim, config.hidden_dim, config.dropout
            )
        )
        self.backbone.requires_grad_(False)
        if config.train_last_block:
            vision = self.backbone.vision_model
            vision.encoder.layers[-1].requires_grad_(True)
            vision.post_layernorm.requires_grad_(True)
        self.train()

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        cast(nn.Module, self.backbone).eval()
        vision = self.backbone.vision_model
        vision.encoder.layers[-1].train(mode and self.config.train_last_block)
        vision.post_layernorm.train(mode and self.config.train_last_block)
        return self

    @torch.no_grad()
    def encode_prefix(self, images: Tensor) -> Tensor:
        """Encode frames through the frozen prefix; FP32 inputs yield FP32 tokens."""
        vision = self.backbone.vision_model
        hidden = vision.pre_layrnorm(vision.embeddings(images))
        for layer in vision.encoder.layers[:-1]:
            hidden = layer(hidden, attention_mask=None)
        return cast(Tensor, hidden)

    def forward_tokens(self, tokens: Tensor) -> Tensor:
        batch, frames = tokens.shape[:2]
        vision = self.backbone.vision_model
        hidden = tokens.flatten(0, 1).to(dtype=next(self.backbone.parameters()).dtype)
        hidden = vision.encoder.layers[-1](hidden, attention_mask=None)
        pooled = vision.post_layernorm(hidden[:, 0, :])
        features = self.backbone.visual_projection(pooled)
        logits = self.head(features).reshape(batch, frames)
        return cast(Tensor, logits.mean(dim=1))

    def forward(self, images: Tensor, *, frame_batch_size: int | None = None) -> Tensor:
        if images.ndim == 4:
            return self.forward_tokens(images)
        return super().forward(images, frame_batch_size=frame_batch_size)
