"""One offline scoring path shared by local evaluation and submission runner."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from .config import InferenceConfig
from .data.manifest import SampleRecord
from .execution import ExecutionPolicy
from .features import (
    extract_features,
    extract_video_frames,
    load_encoder,
    media_path,
    normalized_video_frames,
)
from .media import (
    AudioConfig,
    MediaError,
    VideoConfig,
    decode_audio,
    sample_audio_segments,
)
from .media_tools import media_tool
from .models import (
    AASISTLite,
    DinoAdaptVideoClassifier,
    LogMelCNN,
    MediaFeatureClassifier,
    SwinVideoClassifier,
    UFDVideoClassifier,
    VideoFrameClassifier,
    VideoFusionClassifier,
    WaveRepVideoClassifier,
    load_checkpoint,
    pool_media_scores,
)
from .models.ast import ASTClassifier
from .models.checkpoint import verify_sha256
from .models.core import PoolingMethod
from .models.ssl_antispoofing import build_model, pad_or_tile
from .timing import inference_stage


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _validate_media_executables(config: InferenceConfig, audio: AudioConfig) -> None:
    if config.modality == "video" or audio.backend == "ffmpeg":
        missing = [
            name for name in ("ffmpeg", "ffprobe") if not shutil.which(media_tool(name))
        ]
        if missing:
            raise RuntimeError(
                f"required media executables missing: {', '.join(missing)}"
            )


class OfflineScorer:
    """Load resources once; media errors are recorded by the outer runner."""

    def __init__(  # noqa: C901, PLR0912, PLR0915
        self, config: InferenceConfig, base_dir: Path, *, resume: bool = False
    ) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.config = config
        self.root = resolve_path(base_dir, config.data_root).resolve(strict=True)
        self.audio = AudioConfig(**config.audio)  # type: ignore[arg-type]
        self.video = VideoConfig(**config.video)  # type: ignore[arg-type]
        _validate_media_executables(config, self.audio)
        self.device = torch.device(config.device)
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        loading_started = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        torch.manual_seed(config.seed)
        if config.model_kind == "audio_ssl_antispoofing":
            weights = resolve_path(base_dir, config.checkpoint)
            xlsr_config = resolve_path(base_dir, config.ssl["xlsr_config"])
            verify_sha256(xlsr_config, config.ssl["xlsr_config_sha256"])
            self.model, self.metadata = build_model(
                weights,
                resolve_path(base_dir, config.ssl["upstream_model"]),
                xlsr_config,
                str(self.device),
                expected_weights_sha256=config.checkpoint_sha256,
            )
        else:
            self.model, self.metadata = load_checkpoint(
                resolve_path(base_dir, config.checkpoint),
                expected_sha256=config.checkpoint_sha256,
            )
        self.model.to(self.device).eval()
        self.frame_preprocessing: dict[str, Any] | None = None
        if config.model_kind == "video_frames":
            if not isinstance(
                self.model,
                (
                    VideoFrameClassifier,
                    DinoAdaptVideoClassifier,
                    SwinVideoClassifier,
                    UFDVideoClassifier,
                    WaveRepVideoClassifier,
                    VideoFusionClassifier,
                ),
            ):
                raise ValueError(
                    "video_frames requires a video_frame_finetune_v1, ufd_video_v1 "
                    "waverep_video_v1, video_dinov2_adapt_v1, video_swin3d_v1 "
                    "or video_fusion_v1 checkpoint"
                )
            if self.model.config.aggregation != config.aggregation:
                raise ValueError(
                    "head aggregation and inference configuration disagree"
                )
            if isinstance(
                self.model,
                (UFDVideoClassifier, WaveRepVideoClassifier, DinoAdaptVideoClassifier),
            ) and (
                self.video.resize_mode != "native_center_crop"
                or self.model.preprocessing["input_size"]
                != (3, self.video.height, self.video.width)
            ):
                raise ValueError(
                    "UFD/WaveRep/DINOv2 require native_center_crop at their input size"
                )
            if isinstance(self.model, SwinVideoClassifier) and (
                self.video.n_frames != self.model.config.n_frames
                or self.video.resize_mode != "native_center_crop"
                or (self.video.height, self.video.width) != (224, 224)
            ):
                raise ValueError(
                    "Video Swin requires 16 frames and native_center_crop at 224 pixels"
                )
            self.encoder = None
            if isinstance(self.model, VideoFusionClassifier):
                if (
                    self.video.n_frames != self.model.config.n_frames
                    or self.video.resize_mode != "native_center_crop"
                    or (3, self.video.height, self.video.width)
                    != self.model.clip_preprocessing["input_size"]
                ):
                    raise ValueError(
                        "fusion requires four frames and native_center_crop at CLIP size"
                    )
            else:
                self.frame_preprocessing = self.model.preprocessing
        elif config.model_kind == "audio_ssl_antispoofing":
            self.encoder = None
        elif config.model_kind == "audio_waveform":
            if not isinstance(self.model, (LogMelCNN, AASISTLite, ASTClassifier)):
                raise ValueError("audio_waveform requires LogMel, AASIST-L or AST")
            if isinstance(self.model, (LogMelCNN, ASTClassifier)) and (
                self.model.config.sample_rate != self.audio.sample_rate
            ):
                raise ValueError("decoder/model sample rates disagree")
            if isinstance(self.model, AASISTLite) and (
                self.audio.sample_rate != 16000 or config.segment_samples != 64600
            ):
                raise ValueError("AASIST-L requires 16kHz and 64600-sample segments")
            if isinstance(self.model, ASTClassifier):
                self.model.validate_audio_sampling(
                    config.segment_samples, config.segments
                )
            self.encoder = None
        else:
            if not isinstance(self.model, MediaFeatureClassifier):
                raise ValueError("feature inference requires a media_feature_head_v1")
            if self.model.config.aggregation != config.aggregation:
                raise ValueError(
                    "head aggregation and inference configuration disagree"
                )
            self.encoder = load_encoder(config.encoder, base_dir, str(self.device))
        self.execution = ExecutionPolicy(
            self.device,
            config.execution_mode,
            config.batch_size,
            self.video.n_frames if config.modality == "video" else config.segments,
            resolve_path(base_dir, config.output_dir),
            resume=resume,
        )
        self.execution.resources["model_load_seconds"] = (
            time.monotonic() - loading_started
        )

    @torch.inference_mode()
    def __call__(
        self,
        record: SampleRecord,
        *,
        audio_config: AudioConfig | None = None,
        video_config: VideoConfig | None = None,
    ) -> float:
        if record.status != "available":
            raise MediaError(record.status, record.failure_reason)
        if record.modality != self.config.modality:
            raise MediaError("wrong_modality", "sample differs from run modality")
        try:
            return self.execution.score(
                lambda batch_size: self._score(
                    record,
                    batch_size=batch_size,
                    audio_config=audio_config,
                    video_config=video_config,
                )
            )
        except RuntimeError as exc:
            # PyTorch's CPU allocator raises RuntimeError rather than OutOfMemoryError.
            if "DefaultCPUAllocator: can't allocate memory" in str(exc):
                raise MemoryError(str(exc)) from exc
            raise

    def _score(
        self,
        record: SampleRecord,
        *,
        batch_size: int,
        audio_config: AudioConfig | None = None,
        video_config: VideoConfig | None = None,
    ) -> float:
        if self.config.model_kind == "video_frames" and isinstance(
            self.model, VideoFusionClassifier
        ):
            path = media_path(record, self.root)
            clip_video = video_config or self.video
            _, height, width = self.model.waverep.preprocessing["input_size"]
            waverep_video = replace(clip_video, height=height, width=width)
            clip_frames = normalized_video_frames(
                path, clip_video, self.model.clip_preprocessing
            )
            waverep_frames = normalized_video_frames(
                path, waverep_video, self.model.waverep.preprocessing
            )
            with inference_stage("model_and_feature_scoring"):
                fused = self.model(
                    clip_frames[None, ...],
                    waverep_frames[None, ...],
                    frame_batch_size=batch_size,
                )
                return float(fused[0].float().cpu())
        if self.frame_preprocessing is not None:
            frames = extract_video_frames(
                record,
                self.root,
                video_config or self.video,
                self.frame_preprocessing,
            )
            with inference_stage("model_and_feature_scoring"):
                frame_result: Tensor = self.model(
                    frames[None, ...], frame_batch_size=batch_size
                )
                return float(frame_result[0].float().cpu())
        if self.encoder is not None:
            features = extract_features(
                record,
                self.root,
                self.encoder,
                audio_config or self.audio,
                video_config or self.video,
                self.config.segment_samples,
                self.config.segments,
                str(self.device),
                batch_size,
            )
            with inference_stage("model_and_feature_scoring"):
                result: Tensor = self.model(features[None, ...].to(self.device))
                return float(result[0].float().cpu())
        path = media_path(record, self.root)
        with inference_stage("decode_and_preprocess"):
            decoded = decode_audio(
                path,
                audio_config or self.audio,
            )
            if self.config.model_kind == "audio_ssl_antispoofing":
                observations = torch.from_numpy(pad_or_tile(decoded.waveform))[None, :]
            elif isinstance(self.model, ASTClassifier):
                observations = self.model.prepare_audio(
                    decoded.waveform,
                    self.config.segment_samples,
                    self.config.segments,
                )
            else:
                observations = torch.from_numpy(
                    sample_audio_segments(
                        decoded.waveform,
                        self.config.segment_samples,
                        self.config.segments,
                    )
                )
        with inference_stage("model_and_feature_scoring"):
            if self.config.model_kind == "audio_ssl_antispoofing":
                logits = self.model(observations.to(self.device)).float()
                return -float(logits[0, 1].cpu())
            scores = [
                self.model(
                    observations[start : start + batch_size].to(self.device)
                ).float()
                for start in range(0, len(observations), batch_size)
            ]
            pooled = pool_media_scores(
                torch.cat(scores)[None, :],
                method=cast(PoolingMethod, self.config.aggregation),
            )
            return float(pooled[0].cpu())
