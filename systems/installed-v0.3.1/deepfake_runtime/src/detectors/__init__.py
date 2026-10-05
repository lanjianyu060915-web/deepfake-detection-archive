"""深度伪造检测器统一接口与实现。"""

from .base import BaseDetector, DetectorError
from .frequency_stat import (
    FEATURE_DIMENSION,
    FEATURE_NAMES,
    MODEL_NAME,
    MODEL_VERSION,
    FrequencyStatDetector,
)
from .fusion_detector import FusionDetector
from .spatial_mobilenet import SpatialMobileNetDetector

__all__ = [
    "BaseDetector",
    "DetectorError",
    "FEATURE_DIMENSION",
    "FEATURE_NAMES",
    "MODEL_NAME",
    "MODEL_VERSION",
    "FrequencyStatDetector",
    "FusionDetector",
    "SpatialMobileNetDetector",
]
