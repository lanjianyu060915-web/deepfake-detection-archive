from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from scipy.fft import dctn

from src.preprocessing import OpenCVFaceDetector, PreprocessConfig, PreprocessingError, load_face_crop

from .base import BaseDetector, DetectorError


MODEL_NAME = "FrequencyStatDetector"
MODEL_VERSION = "1.0.0"
RADIAL_BIN_COUNT = 16
EPSILON = 1e-12


def _feature_names() -> list[str]:
    names: list[str] = []
    for index in range(RADIAL_BIN_COUNT):
        names.extend(
            [
                f"fft_radial_{index:02d}_log_mean",
                f"fft_radial_{index:02d}_log_std",
                f"fft_radial_{index:02d}_energy_ratio",
            ]
        )
    names.extend(
        [
            "fft_low_energy_ratio",
            "fft_mid_energy_ratio",
            "fft_high_energy_ratio",
            "fft_high_low_energy_ratio",
            "fft_center_energy_ratio",
            "fft_peripheral_energy_ratio",
            "fft_center_peripheral_energy_difference",
            "dct_abs_mean",
            "dct_abs_std",
            "dct_abs_max",
            "dct_abs_q25",
            "dct_abs_median",
            "dct_abs_q75",
            "dct_low_energy_ratio",
            "dct_mid_energy_ratio",
            "dct_high_energy_ratio",
            "dct_high_low_energy_ratio",
            "block_dct_abs_mean",
            "block_dct_abs_std",
            "block_dct_rms",
        ]
    )
    for band in ("low", "mid", "high"):
        names.extend(
            [
                f"block_dct_{band}_abs_mean",
                f"block_dct_{band}_abs_std",
                f"block_dct_{band}_energy_ratio",
            ]
        )
    names.extend(
        [
            "pixel_mean",
            "pixel_std",
            "pixel_min",
            "pixel_max",
            "pixel_skewness",
            "pixel_excess_kurtosis",
            "gradient_magnitude_mean",
            "gradient_magnitude_std",
            "laplacian_variance",
            "gray_entropy_32bin",
        ]
    )
    return names


FEATURE_NAMES = tuple(_feature_names())
FEATURE_DIMENSION = len(FEATURE_NAMES)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(float(denominator), EPSILON))


def _normalized_radius(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.ogrid[:height, :width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    return radius / max(float(radius.max()), EPSILON)


def _origin_radius(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.ogrid[:height, :width]
    radius = np.sqrt(y**2 + x**2)
    return radius / max(float(radius.max()), EPSILON)


def _band_masks(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return radius < 0.20, (radius >= 0.20) & (radius < 0.50), radius >= 0.50


def _moments(values: np.ndarray) -> tuple[float, float]:
    centered = values - float(values.mean())
    std = float(values.std())
    if std <= EPSILON:
        return 0.0, 0.0
    normalized = centered / std
    return float(np.mean(normalized**3)), float(np.mean(normalized**4) - 3.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class FrequencyStatDetector(BaseDetector):
    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        face_detector: OpenCVFaceDetector | None = None,
        preprocess_config: PreprocessConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve() if model_path else None
        self.metadata_path = Path(metadata_path).resolve() if metadata_path else None
        self.face_detector = face_detector or OpenCVFaceDetector()
        self.preprocess_config = preprocess_config or PreprocessConfig(face_size=(224, 224))
        self.pipeline: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.threshold: float | None = None
        self.progress_callback: Any | None = None

    def _emit_progress(
        self,
        stage: str,
        message: str,
        *,
        started_at: str,
        elapsed_seconds: float,
        progress_current: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        callback = self.progress_callback
        if not callable(callback):
            return
        event = {
            "module": "deepfake",
            "stage": stage,
            "status": "success",
            "message": message,
            "progress_current": int(progress_current),
            "progress_total": 11,
            "progress_percent": round(100.0 * progress_current / 11.0, 3),
            "started_at": started_at,
            "ended_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "elapsed_seconds": max(0.0, float(elapsed_seconds)),
            "payload": dict(payload or {}),
        }
        try:
            callback(event)
        except Exception:
            # Progress reporting is observational and must not break inference.
            return

    @staticmethod
    def feature_schema() -> list[dict[str, Any]]:
        return [{"index": index, "name": name} for index, name in enumerate(FEATURE_NAMES)]

    def load_model(self) -> None:
        if self.model_path is None:
            raise DetectorError("未配置频域统计检测器模型文件。")
        if not self.model_path.exists():
            raise DetectorError(f"频域统计检测器模型文件不存在：{self.model_path}")
        metadata_path = self.metadata_path or self.model_path.with_name("metadata.json")
        if not metadata_path.exists():
            raise DetectorError(f"频域统计检测器元数据文件不存在：{metadata_path}")

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DetectorError(f"频域统计检测器元数据损坏或无法读取：{metadata_path}") from exc

        expected_hash = str(metadata.get("model_file_sha256", "")).strip().lower()
        if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
            raise DetectorError("频域统计检测器元数据缺少有效的模型文件SHA-256。")
        if _sha256(self.model_path) != expected_hash:
            raise DetectorError("频域统计检测器模型文件SHA-256校验失败。")

        try:
            pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise DetectorError(f"频域统计检测器模型损坏或不兼容：{exc}") from exc
        if not callable(getattr(pipeline, "predict_proba", None)):
            raise DetectorError("频域统计检测器模型不兼容：缺少predict_proba接口。")
        feature_dim = int(metadata.get("feature_dimension", -1))
        if feature_dim != FEATURE_DIMENSION:
            raise DetectorError(
                f"频域统计检测器特征维度不兼容：模型要求{feature_dim}，当前实现为{FEATURE_DIMENSION}。"
            )
        try:
            threshold = float(metadata["classification_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DetectorError("频域统计检测器元数据缺少有效分类阈值。") from exc
        if not 0.0 <= threshold <= 1.0:
            raise DetectorError("频域统计检测器分类阈值必须位于0到1之间。")

        classes = np.asarray(getattr(pipeline, "classes_", []))
        if not np.any(classes == 1):
            raise DetectorError("频域统计检测器模型不兼容：未找到FAKE类别1。")

        self.pipeline = pipeline
        self.metadata = metadata
        self.metadata_path = metadata_path
        self.threshold = threshold

    def _extract_with_evidence(self, image_path: str | Path) -> tuple[np.ndarray, dict[str, float]]:
        try:
            face = load_face_crop(
                Path(image_path),
                config=self.preprocess_config,
                face_detector=self.face_detector,
            )
        except PreprocessingError as exc:
            raise DetectorError(str(exc)) from exc

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        fft_values = np.fft.fftshift(np.fft.fft2(gray))
        fft_magnitude = np.abs(fft_values)
        fft_power = fft_magnitude**2
        fft_log = np.log1p(fft_magnitude)
        fft_radius = _normalized_radius(gray.shape)
        fft_total_energy = float(fft_power.sum())

        features: list[float] = []
        edges = np.linspace(0.0, 1.0, RADIAL_BIN_COUNT + 1)
        for index in range(RADIAL_BIN_COUNT):
            if index == RADIAL_BIN_COUNT - 1:
                mask = (fft_radius >= edges[index]) & (fft_radius <= edges[index + 1])
            else:
                mask = (fft_radius >= edges[index]) & (fft_radius < edges[index + 1])
            features.extend(
                [
                    float(fft_log[mask].mean()),
                    float(fft_log[mask].std()),
                    _safe_ratio(float(fft_power[mask].sum()), fft_total_energy),
                ]
            )

        fft_low_mask, fft_mid_mask, fft_high_mask = _band_masks(fft_radius)
        fft_low = _safe_ratio(float(fft_power[fft_low_mask].sum()), fft_total_energy)
        fft_mid = _safe_ratio(float(fft_power[fft_mid_mask].sum()), fft_total_energy)
        fft_high = _safe_ratio(float(fft_power[fft_high_mask].sum()), fft_total_energy)
        fft_high_low = _safe_ratio(fft_high, fft_low)
        fft_center = fft_low
        fft_peripheral = fft_high
        fft_difference = fft_center - fft_peripheral
        features.extend(
            [fft_low, fft_mid, fft_high, fft_high_low, fft_center, fft_peripheral, fft_difference]
        )

        dct_values = dctn(gray, type=2, norm="ortho")
        dct_abs = np.abs(dct_values)
        dct_power = dct_values**2
        dct_radius = _origin_radius(gray.shape)
        dct_low_mask, dct_mid_mask, dct_high_mask = _band_masks(dct_radius)
        dct_total_energy = float(dct_power.sum())
        dct_low = _safe_ratio(float(dct_power[dct_low_mask].sum()), dct_total_energy)
        dct_mid = _safe_ratio(float(dct_power[dct_mid_mask].sum()), dct_total_energy)
        dct_high = _safe_ratio(float(dct_power[dct_high_mask].sum()), dct_total_energy)
        features.extend(
            [
                float(dct_abs.mean()),
                float(dct_abs.std()),
                float(dct_abs.max()),
                float(np.quantile(dct_abs, 0.25)),
                float(np.median(dct_abs)),
                float(np.quantile(dct_abs, 0.75)),
                dct_low,
                dct_mid,
                dct_high,
                _safe_ratio(dct_high, dct_low),
            ]
        )

        blocks = gray.reshape(28, 8, 28, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
        block_dct = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
        block_abs = np.abs(block_dct)
        block_power = block_dct**2
        block_total_energy = float(block_power.sum())
        features.extend(
            [
                float(block_abs.mean()),
                float(block_abs.std()),
                math.sqrt(float(np.mean(block_power))),
            ]
        )
        block_radius = _origin_radius((8, 8))
        for mask in _band_masks(block_radius):
            band_abs = block_abs[:, mask]
            band_power = block_power[:, mask]
            features.extend(
                [
                    float(band_abs.mean()),
                    float(band_abs.std()),
                    _safe_ratio(float(band_power.sum()), block_total_energy),
                ]
            )

        skewness, excess_kurtosis = _moments(gray)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.hypot(gradient_x, gradient_y)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        histogram, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
        probabilities = histogram.astype(np.float64) / max(float(histogram.sum()), EPSILON)
        nonzero = probabilities[probabilities > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        features.extend(
            [
                float(gray.mean()),
                float(gray.std()),
                float(gray.min()),
                float(gray.max()),
                skewness,
                excess_kurtosis,
                float(gradient.mean()),
                float(gradient.std()),
                float(laplacian.var()),
                entropy,
            ]
        )

        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (FEATURE_DIMENSION,):
            raise DetectorError(
                f"频域特征维度异常：期望{FEATURE_DIMENSION}，实际{vector.size}。"
            )
        if not np.all(np.isfinite(vector)):
            raise DetectorError("频域特征包含NaN或Inf，无法执行检测。")

        evidence = {
            "fft_low_energy_ratio": fft_low,
            "fft_mid_energy_ratio": fft_mid,
            "fft_high_energy_ratio": fft_high,
            "fft_high_low_energy_ratio": fft_high_low,
            "fft_center_peripheral_energy_difference": fft_difference,
            "dct_low_energy_ratio": dct_low,
            "dct_mid_energy_ratio": dct_mid,
            "dct_high_energy_ratio": dct_high,
        }
        return vector, evidence

    def extract_features(self, image_path: str | Path) -> np.ndarray:
        vector, _ = self._extract_with_evidence(image_path)
        return vector

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        if self.pipeline is None:
            self.load_model()
        assert self.pipeline is not None and self.threshold is not None

        started = time.perf_counter()
        feature_started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        feature_started = time.perf_counter()
        vector, evidence = self._extract_with_evidence(image_path)
        feature_extraction_ms = (time.perf_counter() - feature_started) * 1000.0
        self._emit_progress(
            "frequency_features_extracted",
            "频域87维统计特征已从本次实际人脸裁剪中提取。",
            started_at=feature_started_at,
            elapsed_seconds=feature_extraction_ms / 1000.0,
            progress_current=6,
            payload={"feature_dimension": int(vector.size)},
        )
        prediction_started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        prediction_started = time.perf_counter()
        probabilities = np.asarray(self.pipeline.predict_proba(vector.reshape(1, -1))[0])
        classes = np.asarray(self.pipeline.classes_)
        fake_probability = float(probabilities[np.flatnonzero(classes == 1)[0]])
        prediction_ms = (time.perf_counter() - prediction_started) * 1000.0
        self._emit_progress(
            "frequency_model_inference_completed",
            "频域LogisticRegression已对真实87维输入完成推理。",
            started_at=prediction_started_at,
            elapsed_seconds=prediction_ms / 1000.0,
            progress_current=7,
            payload={"frequency_score": fake_probability},
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not 0.0 <= fake_probability <= 1.0 or not math.isfinite(fake_probability):
            raise DetectorError("模型返回了无效的伪造概率。")
        return {
            "fake_probability": fake_probability,
            "label": "FAKE" if fake_probability >= self.threshold else "REAL",
            "threshold": self.threshold,
            "model_name": MODEL_NAME,
            "model_version": str(self.metadata.get("model_version", MODEL_VERSION)),
            "inference_ms": inference_ms,
            "feature_extraction_ms": feature_extraction_ms,
            "prediction_ms": prediction_ms,
            "device": "cpu",
            "evidence": evidence,
        }

    def predict_batch(self, image_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
        return [self.predict(path) for path in image_paths]

    def get_model_info(self) -> dict[str, Any]:
        info = dict(self.metadata)
        info.update(
            {
                "model_name": MODEL_NAME,
                "model_version": str(info.get("model_version", MODEL_VERSION)),
                "feature_dimension": FEATURE_DIMENSION,
                "device": "cpu",
                "loaded": self.pipeline is not None,
            }
        )
        return info
