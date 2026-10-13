"""Scoped runtime identities for prepared tensors and resumed offline scores.

Only libraries/backends that execute the selected pipeline belong here. This is
not an environment or lockfile snapshot. Resolve once at preparation/run startup;
never probe executables or hash dependencies in the per-sample/epoch hot path.
"""

from __future__ import annotations

import os
import subprocess
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from .media import AudioConfig
from .models.ast import ASTClassifier
from .models.encoders import FrozenAudioEncoder, FrozenVisionEncoder
from .models.finetune import VideoFrameClassifier
from .tracking import object_hash


if TYPE_CHECKING:
    from .scoring import OfflineScorer


def _executable_version(name: str) -> dict[str, str]:
    try:
        result = subprocess.run(  # noqa: S603 -- internal ffmpeg/ffprobe names only
            [name, "-version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot identify required media backend {name}") from exc
    lines = result.stdout.strip().splitlines()
    if not lines or not lines[0].startswith(f"{name} version "):
        raise RuntimeError(f"invalid version output from media backend {name}")
    # Includes build configuration and linked codec/filter library versions.
    # Exclude its install path: moving the same tool must not invalidate inputs.
    return {"version": lines[0], "build_sha256": object_hash(lines)}


def _torch_runtime(device: str) -> dict[str, Any]:
    target = torch.device(device)
    # New per-backend precision getters also reflect legacy precision settings.
    # Legacy getters can raise when a caller has configured the new API instead.
    result: dict[str, Any] = {
        "version": str(torch.__version__),
        "build_sha256": object_hash(torch.__config__.show()),
        "device_type": target.type,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "fp32_precision": torch.backends.fp32_precision,  # type: ignore[attr-defined]
    }
    if target.type == "cpu":
        result.update(
            cpu_capability=torch.backends.cpu.get_cpu_capability(),
            num_threads=torch.get_num_threads(),
            mkldnn_enabled=torch.backends.mkldnn.enabled,
            mkldnn_deterministic=torch.backends.mkldnn.deterministic,
            mkldnn_fp32_precision=torch.backends.mkldnn.fp32_precision,
            matmul_fp32_precision=torch.backends.mkldnn.matmul.fp32_precision,  # type: ignore[attr-defined]
            conv_fp32_precision=torch.backends.mkldnn.conv.fp32_precision,  # type: ignore[attr-defined]
            rnn_fp32_precision=torch.backends.mkldnn.rnn.fp32_precision,  # type: ignore[attr-defined]
        )
    elif target.type == "cuda":
        result.update(
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
            device_name=torch.cuda.get_device_name(target),
            device_capability=list(torch.cuda.get_device_capability(target)),
            cudnn_enabled=torch.backends.cudnn.enabled,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_fp32_precision=torch.backends.cudnn.fp32_precision,
            matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
            conv_fp32_precision=torch.backends.cudnn.conv.fp32_precision,  # type: ignore[attr-defined]
            rnn_fp32_precision=torch.backends.cudnn.rnn.fp32_precision,  # type: ignore[attr-defined]
            flash_sdp=torch.backends.cuda.flash_sdp_enabled(),  # type: ignore[no-untyped-call]
            mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),  # type: ignore[no-untyped-call]
            math_sdp=torch.backends.cuda.math_sdp_enabled(),  # type: ignore[no-untyped-call]
            cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),  # type: ignore[no-untyped-call]
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        )
    else:
        raise ValueError(f"unsupported runtime identity device: {target.type}")
    return result


def _ast_frontend() -> dict[str, Any]:
    from transformers.models.audio_spectrogram_transformer import (  # noqa: PLC0415
        feature_extraction_audio_spectrogram_transformer as frontend,
    )

    # Match the predicate in the loaded extractor, rather than guessing from a
    # package inventory: its optional torchaudio path changes the actual fbank.
    speech_available = frontend.is_speech_available()  # type: ignore[attr-defined]
    result: dict[str, Any] = {
        "transformers": version("transformers"),
        "backend": "torchaudio.compliance.kaldi" if speech_available else "numpy",
        "torch_version": str(torch.__version__),
    }
    if speech_available:
        result["torchaudio"] = version("torchaudio")
        result["torch"] = _torch_runtime("cpu")
    return result


def preprocessing_runtime(
    modality: str,
    audio: AudioConfig,
    *,
    ast: bool = False,
    normalized_frames: bool = False,
) -> dict[str, Any]:
    """Identify only the decoder and arithmetic used to prepare cached inputs."""
    result: dict[str, Any] = {"schema_version": 1, "numpy": np.__version__}
    if modality == "video" or audio.backend == "ffmpeg":
        result["decoder"] = {
            "backend": "ffmpeg",
            "ffmpeg": _executable_version("ffmpeg"),
            "ffprobe": _executable_version("ffprobe"),
        }
    else:
        import soundfile as sf  # type: ignore[import-untyped]  # noqa: PLC0415

        result["decoder"] = {
            "backend": "soundfile",
            "soundfile": sf.__version__,
            "libsndfile": sf.__libsndfile_version__,
        }
    if ast:
        result["ast_frontend"] = _ast_frontend()
    if normalized_frames:
        # Elementwise normalization has no model, matmul or device backend.
        result["frame_normalization"] = {"torch_version": str(torch.__version__)}
    return result


def model_runtime(model: nn.Module, device: str) -> dict[str, Any]:
    """Record executed model dependencies and resolved attention implementations."""
    result: dict[str, Any] = {
        "torch": _torch_runtime(device),
        "tensor_dtypes": sorted(
            {str(tensor.dtype) for tensor in (*model.parameters(), *model.buffers())}
        ),
    }
    if isinstance(model, (ASTClassifier, FrozenAudioEncoder)):
        result["transformers"] = version("transformers")
        result["attention_implementation"] = model.backbone.config._attn_implementation
    elif isinstance(model, (FrozenVisionEncoder, VideoFrameClassifier)):
        result["timm"] = version("timm")
        result["torchvision"] = version("torchvision")
        result["fused_attention"] = {
            name: module.fused_attn
            for name, module in model.named_modules()
            if isinstance(getattr(module, "fused_attn", None), bool)
        }
    return result


def inference_runtime(scorer: OfflineScorer) -> dict[str, Any]:
    """Bind resume to the loaded scorer's actual preprocessing/model backends."""
    result = preprocessing_runtime(
        scorer.config.modality,
        scorer.audio,
        ast=isinstance(scorer.model, ASTClassifier),
        normalized_frames=scorer.config.modality == "video",
    )
    result["model"] = model_runtime(scorer.model, str(scorer.device))
    if scorer.encoder is not None:
        result["encoder"] = model_runtime(scorer.encoder, str(scorer.device))
    return result
