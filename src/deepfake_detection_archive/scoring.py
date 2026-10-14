"""One offline scoring path shared by local evaluation and submission runner."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from .config import InferenceConfig
from .data.manifest import SampleRecord
from .features import extract_features, extract_video_frames, load_encoder, media_path
from .media import (
    AudioConfig,
    MediaError,
    VideoConfig,
    decode_audio,
    sample_audio_segments,
)
from .models import (
    AASISTLite,
    LogMelCNN,
    MediaFeatureClassifier,
    VideoFrameClassifier,
    load_checkpoint,
    pool_media_scores,
)
from .models.ast import ASTClassifier
from .models.core import PoolingMethod
from .timing import inference_stage


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _validate_media_executables(config: InferenceConfig, audio: AudioConfig) -> None:
    if config.modality == "video" or audio.backend == "ffmpeg":
        missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
        if missing:
            raise RuntimeError(
                f"required media executables missing: {', '.join(missing)}"
            )


class OfflineScorer:
    """Load resources once; media errors are recorded by the outer runner."""

    def __init__(self, config: InferenceConfig, base_dir: Path) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.config = config
        self.root = resolve_path(base_dir, config.data_root).resolve(strict=True)
        self.audio = AudioConfig(**config.audio)  # type: ignore[arg-type]
        self.video = VideoConfig(**config.video)  # type: ignore[arg-type]
        _validate_media_executables(config, self.audio)
        self.device = torch.device(config.device)
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        torch.manual_seed(config.seed)
        self.model, self.metadata = load_checkpoint(
            resolve_path(base_dir, config.checkpoint),
            expected_sha256=config.checkpoint_sha256,
        )
        self.model.to(self.device).eval()
        self.frame_preprocessing: dict[str, Any] | None = None
        if config.model_kind == "video_frames":
            if not isinstance(self.model, VideoFrameClassifier):
                raise ValueError(
                    "video_frames requires a video_frame_finetune_v1 checkpoint"
                )
            if self.model.config.aggregation != config.aggregation:
                raise ValueError(
                    "head aggregation and inference configuration disagree"
                )
            self.encoder = None
            self.frame_preprocessing = self.model.preprocessing
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

    @torch.inference_mode()
    def __call__(
        self,
        record: SampleRecord,
        *,
        audio_config: AudioConfig | None = None,
        video_config: VideoConfig | None = None,
    ) -> float:
        try:
            return self._score(
                record, audio_config=audio_config, video_config=video_config
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
        audio_config: AudioConfig | None = None,
        video_config: VideoConfig | None = None,
    ) -> float:
        if record.status != "available":
            raise MediaError(record.status, record.failure_reason)
        if record.modality != self.config.modality:
            raise MediaError("wrong_modality", "sample differs from run modality")
        if self.frame_preprocessing is not None:
            frames = extract_video_frames(
                record,
                self.root,
                video_config or self.video,
                self.frame_preprocessing,
            )
            with inference_stage("model_and_feature_scoring"):
                frame_result: Tensor = self.model(frames[None, ...].to(self.device))
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
                self.config.batch_size,
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
            if isinstance(self.model, ASTClassifier):
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
            scores = [
                self.model(
                    observations[start : start + self.config.batch_size].to(self.device)
                ).float()
                for start in range(0, len(observations), self.config.batch_size)
            ]
            pooled = pool_media_scores(
                torch.cat(scores)[None, :],
                method=cast(PoolingMethod, self.config.aggregation),
            )
            return float(pooled[0].cpu())
