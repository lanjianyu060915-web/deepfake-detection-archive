"""Offline audio/video model components; importing them requires a torch extra."""

from .aasist import AASISTLite
from .checkpoint import create_model, load_checkpoint, save_checkpoint, sha256_file
from .clip_adapt import ClipAdaptConfig, ClipAdaptVideoClassifier
from .core import (
    FeatureHead,
    HeadConfig,
    LogMelCNN,
    LogMelConfig,
    MediaFeatureClassifier,
    MediaFeatureConfig,
    pool_media_scores,
)
from .dino_adapt import DinoAdaptConfig, DinoAdaptVideoClassifier
from .encoders import FrozenAudioEncoder, FrozenVisionEncoder
from .finetune import VideoFrameClassifier, VideoFrameConfig
from .ufd import UFDConfig, UFDVideoClassifier
from .video_fusion import VideoFusionClassifier, VideoFusionConfig
from .waverep import WaveRepConfig, WaveRepVideoClassifier


__all__ = [
    "AASISTLite",
    "ClipAdaptConfig",
    "ClipAdaptVideoClassifier",
    "DinoAdaptConfig",
    "DinoAdaptVideoClassifier",
    "FeatureHead",
    "FrozenAudioEncoder",
    "FrozenVisionEncoder",
    "HeadConfig",
    "LogMelCNN",
    "LogMelConfig",
    "MediaFeatureClassifier",
    "MediaFeatureConfig",
    "UFDConfig",
    "UFDVideoClassifier",
    "VideoFrameClassifier",
    "VideoFrameConfig",
    "VideoFusionClassifier",
    "VideoFusionConfig",
    "WaveRepConfig",
    "WaveRepVideoClassifier",
    "create_model",
    "load_checkpoint",
    "pool_media_scores",
    "save_checkpoint",
    "sha256_file",
]
