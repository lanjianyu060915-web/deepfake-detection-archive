from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


class DetectorError(RuntimeError):
    """可直接展示给中文界面的检测器错误。"""


class BaseDetector(ABC):
    """图片检测器统一接口。"""

    @abstractmethod
    def load_model(self) -> None:
        """加载并验证模型文件。"""

    @abstractmethod
    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """对单张图片执行真实模型推理。"""

    @abstractmethod
    def predict_batch(self, image_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
        """批量执行图片推理。"""

    @abstractmethod
    def extract_features(self, image_path: str | Path) -> np.ndarray:
        """提取固定维度检测特征。"""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """返回模型版本、阈值和校验信息。"""
