"""Frozen CLIP-MLP and WaveRep video fusion, with all resources in one checkpoint."""

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from .core import MediaFeatureClassifier, MediaFeatureConfig
from .ufd import UFDConfig, UFDVideoClassifier
from .waverep import WaveRepConfig, WaveRepVideoClassifier


@dataclass(frozen=True)
class VideoFusionConfig:
    clip: dict[str, Any]
    clip_head: dict[str, Any]
    waverep: dict[str, Any]
    mean: list[float]
    std: list[float]
    coef: list[float]
    intercept: float
    aggregation: str = "mean"
    n_frames: int = 4

    def __post_init__(self) -> None:
        if self.aggregation != "mean" or self.n_frames != 4:
            raise ValueError("frozen video fusion requires four frames and mean logits")
        for name in ("mean", "std", "coef"):
            values = getattr(self, name)
            if len(values) != 2 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"fusion {name} requires two finite values")
        if min(self.std) <= 0 or not math.isfinite(self.intercept):
            raise ValueError("fusion requires positive std and finite intercept")


class VideoFusionClassifier(nn.Module):
    """Fuse raw video logits in CLIP, WaveRep order; failures propagate to the runner.

    Each branch keeps its own native crop and normalization. Only independent
    backbone frames are batched; CLIP's media head sees its complete sequence.
    No encoder path, hub name or external weight file is resolved at inference.
    """

    def __init__(self, config: VideoFusionConfig) -> None:
        super().__init__()
        self.config = config
        clip = UFDVideoClassifier(UFDConfig(**config.clip))
        self.clip_backbone = clip.backbone
        self.clip_preprocessing = clip.preprocessing
        self.clip_head = MediaFeatureClassifier(MediaFeatureConfig(**config.clip_head))
        if (
            self.clip_head.config.aggregation != "mean"
            or self.clip_head.config.input_dim != clip.head.in_features
        ):
            raise ValueError("fusion requires a matching CLIP mean-logit head")
        self.waverep = WaveRepVideoClassifier(WaveRepConfig(**config.waverep))

    def forward(
        self, clip_frames: Tensor, waverep_frames: Tensor, *, frame_batch_size: int
    ) -> Tensor:
        batch, frames = clip_frames.shape[:2]
        device = next(self.parameters()).device
        features = torch.cat(
            [
                self.clip_backbone(pixel_values=chunk.to(device)).image_embeds
                for chunk in torch.split(clip_frames.flatten(0, 1), frame_batch_size)
            ]
        ).float()
        clip_score = self.clip_head(features.reshape(batch, frames, -1))
        waverep_score = self.waverep(waverep_frames, frame_batch_size=frame_batch_size)
        mean, std, coef = self.config.mean, self.config.std, self.config.coef
        return cast(
            Tensor,
            coef[0] * ((clip_score - mean[0]) / std[0])
            + coef[1] * ((waverep_score - mean[1]) / std[1])
            + self.config.intercept,
        )
