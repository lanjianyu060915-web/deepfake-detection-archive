"""Local AST binary classifier with a cached AudioSet filterbank frontend."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
from numpy.typing import NDArray
from torch import Tensor, cat, from_numpy, nn

from deepfake_detection_archive.media import MediaError, sample_audio_segments


@dataclass(frozen=True)
class ASTConfig:
    backbone_config: dict[str, Any]
    trainable_layers: int = 4
    sample_rate: int = 16000
    feature_mean: float = -4.2677393
    feature_std: float = 4.5689974
    dropout: float = 0.1
    waveform_preparation: str = "legacy"

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
        if self.waveform_preparation not in {"legacy", "center_pad_64000_peak095_v1"}:
            raise ValueError("unsupported AST waveform_preparation")


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
        result = {
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
        if self.config.waveform_preparation != "legacy":
            result["waveform_preparation"] = self.config.waveform_preparation
        return result

    def validate_audio_sampling(self, sample_count: int, n_segments: int) -> None:
        if self.config.waveform_preparation != "legacy" and (
            sample_count != 64000 or n_segments != 1
        ):
            raise ValueError(
                "AST center_pad_64000_peak095_v1 requires "
                "segment_samples=64000 and segments=1"
            )

    def prepare_audio(
        self, waveform: NDArray[np.float32], sample_count: int, n_segments: int
    ) -> Tensor:
        """Shared decoded-mono preparation for training caches and inference.

        The versioned policy mirrors the offline training-media crop/pad/peak
        operations, without an extra lossy PCM16 roundtrip. Silence stays zero.
        Sampling compatibility is checked by the training/scoring entry point.
        """
        if self.config.waveform_preparation == "legacy":
            segments = sample_audio_segments(waveform, sample_count, n_segments)
        else:
            if (
                waveform.ndim != 1
                or not waveform.size
                or not np.isfinite(waveform).all()
            ):
                raise MediaError(
                    "invalid_audio", "expected nonempty finite mono waveform"
                )
            samples = np.asarray(waveform, dtype=np.float32)
            if len(samples) < 64000:
                missing = 64000 - len(samples)
                samples = np.pad(samples, (missing // 2, missing - missing // 2))
            start = (len(samples) - 64000) // 2
            samples = samples[start : start + 64000]
            peak = float(np.max(np.abs(samples)))
            if peak > 0:
                samples = samples * (0.95 / peak)
            segments = samples[None, :]
        return self.prepare_waveforms(from_numpy(segments))

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
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        state = backbone.state_dict()
        source, target = backbone.config, self.backbone.config
        if target.max_length != source.max_length:
            if target.max_length > source.max_length:
                raise ValueError("AST pretrained input length can only be shortened")
            frequency, old_time = backbone.embeddings.get_shape(source)  # type: ignore[no-untyped-call]
            _, new_time = self.backbone.embeddings.get_shape(target)  # type: ignore[no-untyped-call]
            key = "embeddings.position_embeddings"
            position = state[key]
            grid = position[:, 2:].reshape(1, frequency, old_time, source.hidden_size)
            start = old_time // 2 - new_time // 2
            patches = grid[:, :, start : start + new_time].reshape(
                1, frequency * new_time, source.hidden_size
            )
            state[key] = cat((position[:, :2], patches), dim=1)
        self.backbone.load_state_dict(state, strict=True)
