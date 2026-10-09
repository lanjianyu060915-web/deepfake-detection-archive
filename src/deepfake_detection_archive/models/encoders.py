"""Frozen local-only encoder boundaries; no model IDs or remote code accepted."""

import json
from pathlib import Path
from typing import Any, Self, cast

import torch
from torch import Tensor, nn

from .checkpoint import verify_sha256
from .core import validate_waveforms


VISION_ARCHITECTURES = frozenset(
    {
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "vit_small_patch16_224",
        "convnext_tiny",
    }
)
AUDIO_ARCHITECTURES = frozenset({"wav2vec2", "hubert", "wavlm"})


class _FrozenEncoder(nn.Module):
    def train(self, mode: bool = True) -> Self:
        # Parent .train() calls must never activate encoder dropout or update BN.
        if not isinstance(mode, bool):
            raise TypeError("training mode must be a bool")
        super().train(False)
        return self

    def _freeze(self) -> None:
        self.requires_grad_(False)
        self.eval()


class FrozenVisionEncoder(_FrozenEncoder):
    """Load a timm feature-only state_dict from an explicit hash-verified file.

    The file must contain the exact `num_classes=0` state_dict, without wrappers.
    Pretrained classifiers must be explicitly converted before this boundary.
    Input RGB tensors must already use the registered model preprocessing;
    `preprocessing` exposes its size, interpolation, mean and standard deviation.
    """

    def __init__(
        self, architecture: str, checkpoint: Path, *, expected_sha256: str
    ) -> None:
        super().__init__()
        if architecture not in VISION_ARCHITECTURES:
            raise ValueError(f"unsupported local vision architecture: {architecture}")
        verify_sha256(checkpoint, expected_sha256)
        import timm  # noqa: PLC0415
        from timm.data.config import resolve_model_data_config  # noqa: PLC0415

        self.backbone = timm.create_model(architecture, pretrained=False, num_classes=0)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(
            not isinstance(key, str) or not isinstance(value, Tensor)
            for key, value in state.items()
        ):
            raise ValueError("vision checkpoint must contain a bare tensor state_dict")
        if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
            raise ValueError("vision checkpoint contains non-finite tensors")
        self.backbone.load_state_dict(state, strict=True)
        self.feature_dim = int(cast(int, self.backbone.num_features))
        self.preprocessing: dict[str, Any] = resolve_model_data_config(  # type: ignore[no-untyped-call]
            self.backbone
        )
        self.architecture = architecture
        self.checkpoint_sha256 = expected_sha256
        self._freeze()

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape) == 0:
            raise ValueError(
                "images must have nonempty shape [batch, 3, height, width]"
            )
        if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
            raise ValueError("images must contain finite floating-point values")
        return cast(Tensor, self.backbone(images))


class FrozenAudioEncoder(_FrozenEncoder):
    """Mean-pooled SSL speech embeddings from a fully verified local bundle.

    This does not imply coverage of music/environmental sounds. Unpadded inputs
    are batched; variable-length waveforms are encoded separately so padding
    cannot change attention or group normalization. Cache embeddings to reuse.
    """

    def __init__(self, model_dir: Path, *, expected_hashes: dict[str, str]) -> None:
        super().__init__()
        model_dir = model_dir.resolve(strict=True)
        if not model_dir.is_dir():
            raise ValueError("model_dir must be a local directory")
        weights = {path.name for path in model_dir.glob("*.safetensors")}
        if not weights:
            raise ValueError("local SSL bundle requires safetensors weights")
        required = weights | {"config.json", "preprocessor_config.json"}
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            required.add(index.name)
        if not required <= expected_hashes.keys():
            raise ValueError(
                "trusted hashes must cover config, preprocessor and weights"
            )
        for name, digest in expected_hashes.items():
            if Path(name).name != name:
                raise ValueError("bundle hash keys must be filenames without paths")
            verify_sha256(model_dir / name, digest)
        if index.exists():
            contents = json.loads(index.read_text(encoding="utf-8"))
            if not set(contents["weight_map"].values()) <= weights:
                raise ValueError("weight index references unverified or absent shards")
        config_json = json.loads(
            (model_dir / "config.json").read_text(encoding="utf-8")
        )
        if config_json.get("model_type") not in AUDIO_ARCHITECTURES:
            raise ValueError("unsupported local SSL architecture")
        from transformers import AutoFeatureExtractor, AutoModel  # noqa: PLC0415

        extractor = AutoFeatureExtractor.from_pretrained(  # type: ignore[no-untyped-call]
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        self.sample_rate = int(extractor.sampling_rate)
        self.do_normalize = bool(extractor.do_normalize)
        self.backbone, loading_info = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            output_loading_info=True,
        )
        if any(
            loading_info.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError("SSL weights do not exactly match the local architecture")
        self.feature_dim = int(self.backbone.config.hidden_size)
        self.bundle_hashes = dict(expected_hashes)
        self._freeze()

    @torch.no_grad()
    def forward(
        self,
        waveforms: Tensor,
        lengths: Tensor | None = None,
        *,
        sample_rate: int = 16000,
    ) -> Tensor:
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"expected sample rate {self.sample_rate}, got {sample_rate}"
            )
        lengths = validate_waveforms(waveforms, lengths)
        if bool((lengths == waveforms.shape[1]).all()):
            values = waveforms.float()
            if self.do_normalize:
                values = (values - values.mean(dim=1, keepdim=True)) / (
                    values.var(dim=1, unbiased=False, keepdim=True) + 1e-7
                ).sqrt()
            hidden = self.backbone(input_values=values).last_hidden_state
            return cast(Tensor, hidden.mean(dim=1))
        embeddings: list[Tensor] = []
        for waveform, length in zip(waveforms, lengths, strict=True):
            values = waveform[: int(length)].float()
            if self.do_normalize:
                values = (values - values.mean()) / (
                    values.var(unbiased=False) + 1e-7
                ).sqrt()
            hidden = self.backbone(input_values=values[None, :]).last_hidden_state
            embeddings.append(hidden.mean(dim=1).squeeze(0))
        return torch.stack(embeddings)
