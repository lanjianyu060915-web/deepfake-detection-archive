"""One offline scoring path shared by local evaluation and submission runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from .config import InferenceConfig
from .data.manifest import SampleRecord
from .features import extract_features, load_encoder, media_path
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
    load_checkpoint,
    pool_media_scores,
)
from .models.core import PoolingMethod


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


class OfflineScorer:
    """Load resources once; media errors are recorded by the outer runner."""

    def __init__(self, config: InferenceConfig, base_dir: Path) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.config = config
        self.root = resolve_path(base_dir, config.data_root).resolve(strict=True)
        self.device = torch.device(config.device)
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        torch.manual_seed(config.seed)
        self.model, self.metadata = load_checkpoint(
            resolve_path(base_dir, config.checkpoint),
            expected_sha256=config.checkpoint_sha256,
        )
        self.model.to(self.device).eval()
        self.audio = AudioConfig(**config.audio)  # type: ignore[arg-type]
        self.video = VideoConfig(**config.video)  # type: ignore[arg-type]
        if config.model_kind == "audio_waveform":
            if not isinstance(self.model, (LogMelCNN, AASISTLite)):
                raise ValueError("audio_waveform requires logmel_cnn or AASIST-L")
            if isinstance(self.model, LogMelCNN) and (
                self.model.config.sample_rate != self.audio.sample_rate
            ):
                raise ValueError("decoder/model sample rates disagree")
            if isinstance(self.model, AASISTLite) and (
                self.audio.sample_rate != 16000 or config.segment_samples != 64600
            ):
                raise ValueError("AASIST-L requires 16kHz and 64600-sample segments")
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
    def __call__(self, record: SampleRecord) -> float:
        if record.status != "available":
            raise MediaError(record.status, record.failure_reason)
        if record.modality != self.config.modality:
            raise MediaError("wrong_modality", "sample differs from run modality")
        if self.encoder is not None:
            features = extract_features(
                record,
                self.root,
                self.encoder,
                self.audio,
                self.video,
                self.config.segment_samples,
                self.config.segments,
                str(self.device),
                self.config.batch_size,
            )
            result: Tensor = self.model(features[None, ...].to(self.device))
            return float(result[0].float().cpu())
        decoded = decode_audio(media_path(record, self.root), self.audio)
        observations = torch.from_numpy(
            sample_audio_segments(
                decoded.waveform,
                self.config.segment_samples,
                self.config.segments,
            )
        )
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
