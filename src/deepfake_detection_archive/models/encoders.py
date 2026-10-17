"""Frozen local-only encoder boundaries; no model IDs or remote code accepted."""

from pathlib import Path
from typing import Any, Self, cast

import torch
from torch import Tensor, nn

from deepfake_detection_archive.config import load_object

from .checkpoint import load_checkpoint, verify_sha256
from .core import validate_waveforms
from .ufd import UFDVideoClassifier


VISION_ARCHITECTURES = frozenset(
    {
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "vit_small_patch16_224",
        "convnext_tiny",
        "ufd_clip_vit_l14",
        "vit_base_patch14_dinov2.lvd142m",
    }
)
AUDIO_ARCHITECTURES = frozenset({"wav2vec2", "hubert", "wavlm"})


def verify_audio_bundle(model_dir: Path, expected_hashes: dict[str, str]) -> Path:
    """Verify the local SSL bundle without constructing or loading its model."""
    if not isinstance(expected_hashes, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str)
        for name, digest in expected_hashes.items()
    ):
        raise ValueError("expected_hashes must map filenames to SHA-256 strings")
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
        raise ValueError("trusted hashes must cover config, preprocessor and weights")
    for name, digest in expected_hashes.items():
        if Path(name).name != name:
            raise ValueError("bundle hash keys must be filenames without paths")
        verify_sha256(model_dir / name, digest)
    if index.exists():
        mapping = load_object(index).get("weight_map")
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(
                not isinstance(name, str) or not isinstance(shard, str)
                for name, shard in mapping.items()
            )
        ):
            raise ValueError("weight index requires a nonempty string weight_map")
        if not set(mapping.values()) <= weights:
            raise ValueError("weight index references unverified or absent shards")
    config_json = load_object(model_dir / "config.json")
    if config_json.get("model_type") not in AUDIO_ARCHITECTURES:
        raise ValueError("unsupported local SSL architecture")
    return model_dir


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
    """Load local timm features or the projected visual encoder of a UFD model.

    Timm files contain the exact `num_classes=0` state_dict without wrappers;
    UFD uses a standard full checkpoint containing its projected visual encoder.
    Input RGB tensors must use the encoder's size and normalization; research
    feature heads configure their spatial view separately through VideoConfig.
    DINOv2 keeps its pretrained 518 position grid and interpolates it for the
    experiment's 224 RGB center crops, using ImageNet normalization.
    """

    def __init__(
        self, architecture: str, checkpoint: Path, *, expected_sha256: str
    ) -> None:
        super().__init__()
        if architecture not in VISION_ARCHITECTURES:
            raise ValueError(f"unsupported local vision architecture: {architecture}")
        self.architecture = architecture
        self.checkpoint_sha256 = expected_sha256
        if architecture == "ufd_clip_vit_l14":
            model, _ = load_checkpoint(checkpoint, expected_sha256=expected_sha256)
            if not isinstance(model, UFDVideoClassifier):
                raise ValueError("ufd_clip_vit_l14 requires a ufd_video_v1 checkpoint")
            self.backbone: nn.Module = model.backbone
            self.feature_dim = int(model.backbone.config.projection_dim)
            self.preprocessing: dict[str, Any] = dict(model.preprocessing)
            self._freeze()
            return
        verify_sha256(checkpoint, expected_sha256)
        import timm  # noqa: PLC0415
        from timm.data.config import resolve_model_data_config  # noqa: PLC0415

        dinov2 = architecture == "vit_base_patch14_dinov2.lvd142m"
        model_kwargs: dict[str, Any] = {"dynamic_img_size": True} if dinov2 else {}
        self.backbone = timm.create_model(
            architecture, pretrained=False, num_classes=0, **model_kwargs
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
        self.feature_dim = int(cast(int, self.backbone.num_features))
        self.preprocessing = resolve_model_data_config(  # type: ignore[no-untyped-call]
            self.backbone
        )
        if dinov2:
            self.preprocessing.update(
                input_size=(3, 224, 224), resize_mode="native_center_crop"
            )
        self._freeze()

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape) == 0:
            raise ValueError(
                "images must have nonempty shape [batch, 3, height, width]"
            )
        if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
            raise ValueError("images must contain finite floating-point values")
        if self.architecture == "ufd_clip_vit_l14":
            return cast(Tensor, self.backbone(pixel_values=images).image_embeds)
        return cast(Tensor, self.backbone(images))


class FrozenAudioEncoder(_FrozenEncoder):
    """Mean-pooled SSL speech embeddings from a fully verified local bundle.

    This does not imply coverage of music/environmental sounds. Unpadded inputs
    are batched; variable-length waveforms are encoded separately so padding
    cannot change attention or group normalization. Cache embeddings to reuse.
    """

    def __init__(self, model_dir: Path, *, expected_hashes: dict[str, str]) -> None:
        super().__init__()
        model_dir = verify_audio_bundle(model_dir, expected_hashes)
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
