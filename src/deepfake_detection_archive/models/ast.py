"""Local AST binary classifier with a cached AudioSet filterbank frontend."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

from torch import Tensor, nn


@dataclass(frozen=True)
class ASTConfig:
    backbone_config: dict[str, Any]
    trainable_layers: int = 4
    sample_rate: int = 16000
    feature_mean: float = -4.2677393
    feature_std: float = 4.5689974
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if type(self.trainable_layers) is not int or self.trainable_layers < 0:
            raise ValueError("trainable_layers must be a nonnegative integer")
        if self.sample_rate != 16000:
            raise ValueError("AST requires sample_rate=16000")
        if not math.isfinite(self.feature_mean) or not (
            math.isfinite(self.feature_std) and self.feature_std > 0
        ):
            raise ValueError("feature mean/std must be finite, with std > 0")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


class ASTClassifier(nn.Module):
    """Fine-tune the final AST blocks; return one fake-oriented raw logit."""

    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        from transformers import (  # noqa: PLC0415
            ASTConfig as BackboneConfig,
            ASTFeatureExtractor,
            ASTModel,
        )

        self.config = config
        backbone_config = BackboneConfig(**config.backbone_config)
        backbone_config._attn_implementation = "sdpa"
        if config.trainable_layers > backbone_config.num_hidden_layers:
            raise ValueError("trainable_layers exceeds backbone layer count")
        self.backbone = ASTModel(backbone_config)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(backbone_config.hidden_size, 1)
        )
        self._extractor = ASTFeatureExtractor(  # type: ignore[no-untyped-call]
            sampling_rate=config.sample_rate,
            num_mel_bins=backbone_config.num_mel_bins,
            max_length=backbone_config.max_length,
            mean=config.feature_mean,
            std=config.feature_std,
            do_normalize=True,
            return_attention_mask=False,
        )
        self.backbone.requires_grad_(False)
        for layer in self.backbone.layers[
            len(self.backbone.layers) - config.trainable_layers :
        ]:
            layer.requires_grad_(True)
        self.backbone.layernorm.requires_grad_(True)
        self.train()

    @property
    def preprocessing(self) -> dict[str, Any]:
        """Effective frontend configuration, suitable for a JSON cache identity."""
        return {
            "feature_extractor_type": "ASTFeatureExtractor",
            "sampling_rate": self._extractor.sampling_rate,
            "num_mel_bins": self._extractor.num_mel_bins,
            "max_length": self._extractor.max_length,
            "mean": self._extractor.mean,
            "std": self._extractor.std,
            "do_normalize": self._extractor.do_normalize,
            "padding_value": self._extractor.padding_value,
            "return_attention_mask": self._extractor.return_attention_mask,
        }

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.backbone.eval()  # type: ignore[no-untyped-call]
        for layer in self.backbone.layers[
            len(self.backbone.layers) - self.config.trainable_layers :
        ]:
            layer.train(mode)
        self.backbone.layernorm.train(mode)
        return self

    def prepare_waveforms(self, waveforms: Tensor) -> Tensor:
        features = self._extractor(
            waveforms.detach().float().cpu().numpy(),
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
        )
        return cast(Tensor, features["input_values"]).float()

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 2:
            inputs = self.prepare_waveforms(inputs).to(next(self.parameters()).device)
        pooled = self.backbone(input_values=inputs).pooler_output
        return cast(Tensor, self.head(pooled).squeeze(-1))

    def initialize_pretrained(self, path: Path, expected_sha256: str) -> None:
        """Load a verified, single-file HF AST checkpoint entirely offline."""
        from transformers import ASTModel  # noqa: PLC0415

        from .checkpoint import verify_sha256  # noqa: PLC0415

        if path.name != "model.safetensors":
            raise ValueError("AST pretrained weights must be named model.safetensors")
        verify_sha256(path, expected_sha256)
        backbone = ASTModel.from_pretrained(
            str(path.parent),
            config=self.backbone.config,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        self.backbone.load_state_dict(backbone.state_dict(), strict=True)
