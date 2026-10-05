from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.preprocessing import OpenCVFaceDetector, PreprocessConfig, PreprocessingError, load_face_crop

from .base import BaseDetector, DetectorError


MODEL_NAME = "SpatialMobileNetDetector"
BACKBONE = "mobilenet_v3_small"
WEIGHTS_NAME = "project_checkpoint_full_state_dict"
WEIGHTS_FILENAME = "mobilenet_v3_small-047dcff4.pth"
WEIGHTS_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
DEFAULT_THRESHOLD = 0.5


def official_weights_path() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_FILENAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class SpatialMobileNetDetector(BaseDetector):
    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu",
        face_detector: OpenCVFaceDetector | None = None,
        preprocess_config: PreprocessConfig | None = None,
    ) -> None:
        if device != "cpu":
            raise DetectorError("SpatialMobileNetDetector当前仅支持CPU设备。")
        if not 0.0 <= threshold <= 1.0:
            raise DetectorError("空间域检测器阈值必须位于0到1之间。")
        self.model_path = Path(model_path).resolve() if model_path else None
        self.threshold = float(threshold)
        self.device = torch.device("cpu")
        self.face_detector = face_detector or OpenCVFaceDetector()
        self.preprocess_config = preprocess_config or PreprocessConfig(face_size=(224, 224))
        self.model: nn.Module | None = None
        self.processor = None
        self.label_mapping: dict[str, int] = {"REAL": 0, "FAKE": 1}
        self.model_version = "1.0.0"

    @staticmethod
    def build_model(
        num_classes: int = 2,
        *,
        use_imagenet_pretrained: bool = False,
    ) -> nn.Module:
        """Construct the architecture without a runtime download dependency.

        Formal inference always uses ``weights=None`` because the project
        checkpoint contains all backbone and classifier tensors.  The explicit
        opt-in remains available only for the historical training pipeline.
        """

        weights = None
        if use_imagenet_pretrained:
            path = official_weights_path()
            if not path.is_file():
                raise DetectorError(f"训练所需的官方MobileNetV3-Small权重不存在：{path}")
            if sha256_file(path) != WEIGHTS_SHA256:
                raise DetectorError("训练所需的官方MobileNetV3-Small权重SHA-256校验失败。")
            weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    def validate_label_mapping(self, mapping: dict[str, int] | None = None) -> dict[str, int]:
        mapping = mapping or self.label_mapping
        normalized = {str(key).upper(): int(value) for key, value in mapping.items()}
        if set(normalized) != {"REAL", "FAKE"} or set(normalized.values()) != {0, 1}:
            raise DetectorError("空间域检测器标签映射异常：必须明确包含REAL=0和FAKE=1。")
        self.label_mapping = {"REAL": normalized["REAL"], "FAKE": normalized["FAKE"]}
        return dict(self.label_mapping)

    def load_model(self) -> None:
        if self.model_path is None:
            raise DetectorError("未配置SpatialMobileNetDetector模型文件。")
        if not self.model_path.is_file():
            raise DetectorError(f"SpatialMobileNetDetector模型文件不存在：{self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            mapping = checkpoint.get("label_mapping") if isinstance(checkpoint, dict) else None
            if mapping:
                self.validate_label_mapping(mapping)
            else:
                self.validate_label_mapping()
            model = self.build_model()
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"SpatialMobileNetDetector模型损坏或不兼容：{exc}") from exc
        self.model = model

    def _prepare_with_evidence(
        self,
        image_path: str | Path,
    ) -> tuple[
        torch.Tensor,
        tuple[int, int, int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int, int, int],
    ]:
        """Return the normal tensor plus the measured pre-expansion face box.

        The first four values are byte-for-byte compatible with the historical
        ``_prepare`` contract.  Retaining the actual Haar box avoids running a
        second high-resolution face detection pass merely for visualization.
        """

        path = Path(image_path).resolve()
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
            if image is None:
                raise PreprocessingError(f"无法读取图片，文件可能已损坏：{path.name}")
            original_size = (int(image.shape[1]), int(image.shape[0]))
            box = self.face_detector.detect_largest(image)
            if box is None:
                raise PreprocessingError(f"图片中未检测到人脸：{path.name}")
            from src.preprocessing import expand_face_box

            x1, y1, x2, y2 = expand_face_box(box, image.shape[1], image.shape[0], self.preprocess_config.face_expand_ratio)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                raise PreprocessingError(f"人脸裁剪为空：{path.name}")
            crop = cv2.resize(crop, self.preprocess_config.face_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            weights = MobileNet_V3_Small_Weights.DEFAULT
            tensor = weights.transforms()(pil).unsqueeze(0)
            return (
                tensor,
                (x1, y1, x2 - x1, y2 - y1),
                original_size,
                (crop.shape[1], crop.shape[0]),
                tuple(int(value) for value in box),
            )
        except PreprocessingError as exc:
            raise DetectorError(str(exc)) from exc
        except OSError as exc:
            raise DetectorError(f"无法读取图片：{path}") from exc

    def _prepare(
        self,
        image_path: str | Path,
    ) -> tuple[
        torch.Tensor,
        tuple[int, int, int, int],
        tuple[int, int],
        tuple[int, int],
    ]:
        tensor, bbox, original_size, crop_size, _raw_bbox = self._prepare_with_evidence(
            image_path
        )
        return tensor, bbox, original_size, crop_size

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        if self.model is None:
            self.load_model()
        assert self.model is not None
        started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        tensor, bbox, original_size, crop_size = self._prepare(image_path)
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
        model_started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor.to(self.device))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        real_index = self.label_mapping["REAL"]
        fake_index = self.label_mapping["FAKE"]
        real_probability = float(probabilities[real_index])
        fake_probability = float(probabilities[fake_index])
        model_inference_ms = (time.perf_counter() - model_started) * 1000.0
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (real_probability, fake_probability)):
            raise DetectorError("空间域模型返回了无效概率。")
        if abs(real_probability + fake_probability - 1.0) > 1e-5:
            raise DetectorError("空间域模型概率和不等于1。")
        return {
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "label": "FAKE" if fake_probability >= self.threshold else "REAL",
            "threshold": self.threshold,
            "model_name": MODEL_NAME,
            "backbone": BACKBONE,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
            "preprocessing_ms": preprocessing_ms,
            "model_inference_ms": model_inference_ms,
            "device": "cpu",
            "face_bbox": bbox,
            "original_size": original_size,
            "crop_size": crop_size,
            "evidence": {"real_probability": real_probability, "fake_probability": fake_probability},
        }

    def predict_batch(self, image_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
        return [self.predict(path) for path in image_paths]

    def extract_features(self, image_path: str | Path) -> np.ndarray:
        tensor, *_ = self._prepare(image_path)
        return tensor.squeeze(0).numpy().astype(np.float32, copy=False)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "backbone": BACKBONE,
            "weights": WEIGHTS_NAME,
            "checkpoint_scope": "full_backbone_and_classifier",
            "official_base_weight_required_for_inference": False,
            "label_mapping": dict(self.label_mapping),
            "threshold": self.threshold,
            "device": "cpu",
            "loaded": self.model is not None,
            "model_path": str(self.model_path) if self.model_path else "",
        }
