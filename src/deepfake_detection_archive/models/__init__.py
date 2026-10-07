"""Offline audio/video model components; importing them requires a torch extra."""

from .aasist import AASISTLite
from .checkpoint import create_model, load_checkpoint, save_checkpoint, sha256_file
from .core import (
    FeatureHead,
    HeadConfig,
    LogMelCNN,
    LogMelConfig,
    MediaFeatureClassifier,
    MediaFeatureConfig,
    pool_media_scores,
)
from .encoders import FrozenAudioEncoder, FrozenVisionEncoder


__all__ = [
    "AASISTLite",
    "FeatureHead",
    "FrozenAudioEncoder",
    "FrozenVisionEncoder",
    "HeadConfig",
    "LogMelCNN",
    "LogMelConfig",
    "MediaFeatureClassifier",
    "MediaFeatureConfig",
    "create_model",
    "load_checkpoint",
    "pool_media_scores",
    "save_checkpoint",
    "sha256_file",
]
