from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.fusion.fusion_optimizer import apply_fusion
from src.preprocessing import OpenCVFaceDetector, PreprocessConfig

from .base import BaseDetector, DetectorError
from .frequency_stat import FrequencyStatDetector
from .spatial_finetuning import load_verified_finetuned_model
from .spatial_mobilenet import SpatialMobileNetDetector


class FusionDetector(BaseDetector):
    """可供后续界面直接调用的空间域/频域正式融合检测器。"""

    def __init__(
        self,
        config_path: str | Path,
        device: str = "auto",
        project_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else self._infer_project_root(self.config_path)
        )
        self.requested_device = device
        self.device = self._select_device(device)
        self.config: dict[str, Any] = {}
        self.spatial_model: torch.nn.Module | None = None
        self.spatial_checkpoint: dict[str, Any] = {}
        self.frequency_detector: FrequencyStatDetector | None = None
        self.spatial_preprocessor: SpatialMobileNetDetector | None = None

    @staticmethod
    def _infer_project_root(config_path: Path) -> Path:
        if config_path.parent.parent.name.lower() == "models":
            return config_path.parent.parent.parent
        return Path.cwd().resolve()

    @staticmethod
    def _select_device(device: str) -> torch.device:
        normalized = str(device).lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized.startswith("cuda"):
            if not torch.cuda.is_available():
                raise DetectorError("请求使用CUDA，但当前PyTorch CUDA不可用。请改用device='cpu'或检查CUDA环境。")
            selected = torch.device(normalized)
            if selected.index is not None and selected.index >= torch.cuda.device_count():
                raise DetectorError(f"请求的CUDA设备不存在：{device}")
            return selected
        if normalized == "cpu":
            return torch.device("cpu")
        raise DetectorError(f"不支持的推理设备：{device}")

    def _resolve_config_path(self, value: Any, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise DetectorError(f"融合配置缺少有效字段：{field}")
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def load_model(self) -> None:
        if not self.config_path.is_file():
            raise DetectorError(f"正式融合配置文件不存在：{self.config_path}")
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DetectorError(f"正式融合配置损坏或无法读取：{self.config_path}") from exc
        required = {
            "spatial_model_path",
            "frequency_model_path",
            "fusion_method",
            "spatial_weight",
            "frequency_weight",
            "frame_threshold",
            "video_aggregation",
            "video_threshold",
            "label_mapping",
        }
        missing = sorted(required - set(config))
        if missing:
            raise DetectorError(f"正式融合配置缺少字段：{missing}")
        try:
            spatial_weight = float(config["spatial_weight"])
            frequency_weight = float(config["frequency_weight"])
            frame_threshold = float(config["frame_threshold"])
            video_threshold = float(config["video_threshold"])
        except (TypeError, ValueError) as exc:
            raise DetectorError("融合权重或阈值不是有效数值。") from exc
        if not math.isclose(spatial_weight + frequency_weight, 1.0, abs_tol=1e-9):
            raise DetectorError("空间域和频域融合权重之和必须为1。")
        if min(spatial_weight, frequency_weight, frame_threshold, video_threshold) < 0.0:
            raise DetectorError("融合权重和阈值不得为负数。")
        if max(spatial_weight, frequency_weight, frame_threshold, video_threshold) > 1.0:
            raise DetectorError("融合权重和阈值不得超过1。")
        mapping = {str(key).upper(): int(value) for key, value in dict(config["label_mapping"]).items()}
        if mapping != {"REAL": 0, "FAKE": 1}:
            raise DetectorError("正式融合配置标签映射必须为REAL=0、FAKE=1。")

        spatial_path = self._resolve_config_path(config["spatial_model_path"], "spatial_model_path")
        frequency_path = self._resolve_config_path(config["frequency_model_path"], "frequency_model_path")
        if not spatial_path.is_file():
            raise DetectorError(f"GPU微调空间模型文件不存在：{spatial_path}")
        if not frequency_path.is_file():
            raise DetectorError(f"频域正式模型文件不存在：{frequency_path}")

        try:
            spatial_model, checkpoint = load_verified_finetuned_model(spatial_path.parent, self.device)
        except Exception as exc:
            raise DetectorError(f"GPU微调空间模型加载失败：{exc}") from exc
        shared_face_detector = OpenCVFaceDetector()
        preprocess_config = PreprocessConfig(face_size=(224, 224))
        spatial_preprocessor = SpatialMobileNetDetector(
            device="cpu",
            face_detector=shared_face_detector,
            preprocess_config=preprocess_config,
        )
        frequency_detector = FrequencyStatDetector(
            model_path=frequency_path,
            metadata_path=frequency_path.with_name("metadata.json"),
            face_detector=shared_face_detector,
            preprocess_config=preprocess_config,
        )
        frequency_detector.load_model()
        self.config = config
        self.spatial_model = spatial_model
        self.spatial_checkpoint = checkpoint
        self.frequency_detector = frequency_detector
        self.spatial_preprocessor = spatial_preprocessor

    def _spatial_predict(self, image_path: str | Path) -> dict[str, Any]:
        assert self.spatial_model is not None and self.spatial_preprocessor is not None
        preprocessing_started = time.perf_counter()
        tensor, bbox, original_size, crop_size = self.spatial_preprocessor._prepare(image_path)
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
        tensor = tensor.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_started = time.perf_counter()
        with torch.inference_mode():
            logits = self.spatial_model(tensor)
            probability = torch.softmax(logits, dim=1)[0, 1]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_ms = (time.perf_counter() - model_started) * 1000.0
        fake_probability = float(probability.detach().cpu().item())
        if not math.isfinite(fake_probability) or not 0.0 <= fake_probability <= 1.0:
            raise DetectorError("GPU微调空间模型返回了无效FAKE概率。")
        return {
            "fake_probability": fake_probability,
            "preprocessing_ms": preprocessing_ms,
            "model_inference_ms": model_ms,
            "face_bbox": bbox,
            "original_size": original_size,
            "crop_size": crop_size,
        }

    @staticmethod
    def _risk_level(score: float, threshold: float) -> str:
        if score >= max(0.75, threshold):
            return "高风险"
        if score >= threshold:
            return "中风险"
        return "低风险"

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        if self.spatial_model is None or self.frequency_detector is None:
            self.load_model()
        assert self.frequency_detector is not None
        started = time.perf_counter()
        spatial = self._spatial_predict(image_path)
        frequency = self.frequency_detector.predict(image_path)
        method = str(self.config["fusion_method"])
        alpha = float(self.config["spatial_weight"])
        fusion_score = float(
            apply_fusion(
                [spatial["fake_probability"]],
                [frequency["fake_probability"]],
                method,
                alpha,
            )[0]
        )
        threshold = float(self.config["frame_threshold"])
        predicted_label = "FAKE" if fusion_score >= threshold else "REAL"
        return {
            "spatial_score": spatial["fake_probability"],
            "frequency_score": frequency["fake_probability"],
            "fusion_score": fusion_score,
            "threshold": threshold,
            "predicted_label": predicted_label,
            "label": predicted_label,
            "risk_level": self._risk_level(fusion_score, threshold),
            "device": str(self.device),
            "fusion_method": method,
            "spatial_weight": alpha,
            "frequency_weight": 1.0 - alpha,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
            "spatial_preprocessing_ms": spatial["preprocessing_ms"],
            "spatial_model_inference_ms": spatial["model_inference_ms"],
            "frequency_inference_ms": frequency["inference_ms"],
            "face_bbox": spatial["face_bbox"],
            "original_size": spatial["original_size"],
            "crop_size": spatial["crop_size"],
        }

    def predict_batch(self, image_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
        return [self.predict(path) for path in image_paths]

    def extract_features(self, image_path: str | Path) -> np.ndarray:
        result = self.predict(image_path)
        return np.asarray([result["spatial_score"], result["frequency_score"]], dtype=np.float64)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": "FusionDetector",
            "loaded": self.spatial_model is not None and self.frequency_detector is not None,
            "device": str(self.device),
            "config_path": str(self.config_path),
            "fusion_method": self.config.get("fusion_method"),
            "spatial_weight": self.config.get("spatial_weight"),
            "frequency_weight": self.config.get("frequency_weight"),
            "frame_threshold": self.config.get("frame_threshold"),
            "video_aggregation": self.config.get("video_aggregation"),
            "video_threshold": self.config.get("video_threshold"),
        }
