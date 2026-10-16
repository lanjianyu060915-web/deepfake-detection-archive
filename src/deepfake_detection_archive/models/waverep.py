"""Self-contained WaveRep DINOv2 frame detector (Corvi et al., NeurIPS 2025)."""

from dataclasses import dataclass, field
from typing import Any, cast

from torch import Tensor, nn


def _dinov2_g1_config() -> dict[str, Any]:
    # timm vit_base_patch14_reg4_dinov2.lvd142m geometry used by the author demo.
    return {"img_size": 504}


@dataclass(frozen=True)
class WaveRepConfig:
    timm_name: str = "vit_base_patch14_reg4_dinov2.lvd142m"
    backbone_overrides: dict[str, Any] = field(default_factory=_dinov2_g1_config)
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if self.timm_name != "vit_base_patch14_reg4_dinov2.lvd142m":
            raise ValueError("WaveRep requires the pinned G1 timm architecture")
        structural = {"img_size", "embed_dim", "depth", "num_heads"}
        if set(self.backbone_overrides) - structural:
            raise ValueError("WaveRep backbone accepts only structural overrides")
        if self.aggregation != "mean":
            raise ValueError("WaveRep requires mean frame-logit aggregation")
        if "img_size" not in self.backbone_overrides:
            raise ValueError("WaveRep backbone requires an explicit img_size")


class WaveRepVideoClassifier(nn.Module):
    """Apply the author's end-to-end DINOv2 detector to each frame.

    Input is [batch, frames, 3, height, width]; output is the mean raw frame logit
    (author LLR, higher is fake). The network is built with ``pretrained=False``
    and receives every parameter from the local checkpoint; it never downloads.
    """

    def __init__(self, config: WaveRepConfig) -> None:
        super().__init__()
        import timm  # noqa: PLC0415

        self.config = config
        self.backbone = timm.create_model(
            config.timm_name,
            pretrained=False,
            num_classes=1,
            **config.backbone_overrides,
        )
        size = int(config.backbone_overrides["img_size"])
        self.preprocessing: dict[str, Any] = {
            "input_size": (3, size, size),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "resize_mode": "native_center_crop",
        }

    def forward(self, images: Tensor) -> Tensor:
        batch, frames = images.shape[:2]
        flattened = images.reshape(batch * frames, *images.shape[2:])
        logits = self.backbone(flattened)[:, -1].reshape(batch, frames)
        return cast(Tensor, logits.mean(dim=1))
