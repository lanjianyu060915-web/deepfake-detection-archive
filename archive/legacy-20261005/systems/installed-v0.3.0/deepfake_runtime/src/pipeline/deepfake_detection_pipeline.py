from __future__ import annotations

import json
import logging
import math
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.detectors import DetectorError, FusionDetector
from src.preprocessing import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    OpenCVFaceDetector,
    PreprocessConfig,
    calculate_sha256,
    extract_largest_face,
    read_image,
    write_image,
)
from src.reporting import RuntimeReportService


class RuntimePipelineError(RuntimeError):
    """可直接展示给中文界面的端到端管线错误。"""


@dataclass(frozen=True)
class RuntimePipelineConfig:
    target_valid_frames: int
    min_valid_frames: int
    candidate_multiplier: int
    max_candidate_frames: int
    keyframe_count: int
    cleanup_temporary: bool

    @classmethod
    def load(cls, path: Path) -> "RuntimePipelineConfig":
        if not path.is_file():
            raise RuntimePipelineError(f"运行时管线配置不存在：{path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = cls(
                target_valid_frames=int(payload["target_valid_frames"]),
                min_valid_frames=int(payload["min_valid_frames"]),
                candidate_multiplier=int(payload["candidate_multiplier"]),
                max_candidate_frames=int(payload["max_candidate_frames"]),
                keyframe_count=int(payload["keyframe_count"]),
                cleanup_temporary=bool(payload["cleanup_temporary"]),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimePipelineError(f"运行时管线配置损坏或字段无效：{path}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.target_valid_frames <= 0:
            raise RuntimePipelineError("目标有效帧数必须大于0。")
        if self.min_valid_frames <= 0 or self.min_valid_frames > self.target_valid_frames:
            raise RuntimePipelineError("最低有效帧数必须大于0且不超过目标有效帧数。")
        if self.candidate_multiplier < 1:
            raise RuntimePipelineError("候选帧倍数不得小于1。")
        if self.max_candidate_frames < self.target_valid_frames:
            raise RuntimePipelineError("最大候选帧数不得小于目标有效帧数。")
        if self.keyframe_count <= 0:
            raise RuntimePipelineError("关键帧数量必须大于0。")

    def with_overrides(
        self,
        target_valid_frames: int | None,
        min_valid_frames: int | None,
    ) -> "RuntimePipelineConfig":
        updated = replace(
            self,
            target_valid_frames=self.target_valid_frames if target_valid_frames is None else int(target_valid_frames),
            min_valid_frames=self.min_valid_frames if min_valid_frames is None else int(min_valid_frames),
        )
        updated.validate()
        return updated


class DeepfakeDetectionPipeline:
    """正式图片/视频端到端检测管线；一个实例只加载一次模型。"""

    def __init__(
        self,
        project_root: str | Path,
        device: str = "auto",
        output_root: str | Path | None = None,
        writable_root: str | Path | None = None,
        fusion_config_path: str | Path | None = None,
        runtime_config_path: str | Path | None = None,
        detector: Any | None = None,
        face_detector: Any | None = None,
        report_service: RuntimeReportService | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.writable_root = (
            Path(writable_root).resolve() if writable_root is not None else self.project_root
        )
        self.output_root = (
            Path(output_root).resolve()
            if output_root
            else self.writable_root / "outputs" / "runtime"
        )
        self._require_inside_writable_root(self.output_root, "运行结果目录")
        self.fusion_config_path = (
            Path(fusion_config_path).resolve()
            if fusion_config_path
            else self.project_root / "models" / "fusion_formal" / "fusion_config.json"
        )
        self.runtime_config_path = (
            Path(runtime_config_path).resolve()
            if runtime_config_path
            else self.project_root / "config" / "runtime_pipeline.json"
        )
        self.runtime_config = RuntimePipelineConfig.load(self.runtime_config_path)
        self.fusion_config = self._load_fusion_config()
        self.fusion_config_hash = calculate_sha256(self.fusion_config_path)
        self.detector = detector or FusionDetector(
            self.fusion_config_path, device=device, project_root=self.project_root
        )
        self.fallback_face_detector = face_detector or OpenCVFaceDetector()
        self.preprocess_config = PreprocessConfig(face_size=(224, 224), face_expand_ratio=0.25)
        self.report_service = report_service or RuntimeReportService()
        self._model_loaded = False
        self._model_load_count = 0
        self._load_lock = threading.Lock()

    def _require_inside_writable_root(self, path: Path, label: str) -> None:
        try:
            path.resolve().relative_to(self.writable_root)
        except ValueError as exc:
            if self.writable_root == self.project_root:
                raise RuntimePipelineError(f"{label}必须位于当前项目目录内：{path}") from exc
            raise RuntimePipelineError(f"{label}必须位于应用运行数据目录内：{path}") from exc

    def _load_fusion_config(self) -> dict[str, Any]:
        if not self.fusion_config_path.is_file():
            raise RuntimePipelineError(f"正式融合配置不存在：{self.fusion_config_path}")
        try:
            config = json.loads(self.fusion_config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimePipelineError(f"正式融合配置损坏：{self.fusion_config_path}") from exc
        required = {
            "spatial_model_path",
            "frequency_model_path",
            "fusion_method",
            "spatial_weight",
            "frequency_weight",
            "frame_threshold",
            "video_aggregation",
            "video_threshold",
        }
        missing = sorted(required - set(config))
        if missing:
            raise RuntimePipelineError(f"正式融合配置缺少字段：{missing}")
        if not math.isclose(
            float(config["spatial_weight"]) + float(config["frequency_weight"]), 1.0, abs_tol=1e-9
        ):
            raise RuntimePipelineError("正式融合配置的两分支权重之和不为1。")
        if str(config["video_aggregation"]) != "median":
            raise RuntimePipelineError(
                f"本阶段正式视频流程要求median聚合，配置实际为：{config['video_aggregation']}"
            )
        return config

    def _ensure_model_loaded(self) -> float:
        if self._model_loaded:
            return 0.0
        with self._load_lock:
            if self._model_loaded:
                return 0.0
            started = time.perf_counter()
            self.detector.load_model()
            self._model_loaded = True
            self._model_load_count += 1
            return time.perf_counter() - started

    def _active_face_detector(self) -> Any:
        preprocessor = getattr(self.detector, "spatial_preprocessor", None)
        if preprocessor is not None and getattr(preprocessor, "face_detector", None) is not None:
            return preprocessor.face_detector
        return self.fallback_face_detector

    @staticmethod
    def _run_id() -> str:
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        return f"{timestamp}_{uuid.uuid4().hex[:12]}"

    def _create_run(self) -> tuple[str, Path, Path, logging.Logger]:
        run_id = self._run_id()
        run_dir = (self.output_root / run_id).resolve()
        self._require_inside_writable_root(run_dir, "单次运行目录")
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "keyframes").mkdir()
        temp_dir = run_dir / f".tmp_{uuid.uuid4().hex}"
        temp_dir.mkdir()
        logger = logging.getLogger(f"runtime_pipeline.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(run_dir / "runtime.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)
        return run_id, run_dir, temp_dir, logger

    @staticmethod
    def _close_logger(logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    def _base_result(self, run_id: str, run_dir: Path, input_path: Path, media_type: str) -> dict[str, Any]:
        spatial_path = self._resolve_model_path(self.fusion_config["spatial_model_path"])
        frequency_path = self._resolve_model_path(self.fusion_config["frequency_model_path"])
        actual_device = str(getattr(self.detector, "device", "unknown"))
        return {
            "run_id": run_id,
            "status": "PROCESSING",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "media_type": media_type,
            "input_file_name": input_path.name,
            "input_sha256": "",
            "file_size": int(input_path.stat().st_size) if input_path.is_file() else 0,
            "actual_device": actual_device,
            "model_paths": {"spatial": str(spatial_path), "frequency": str(frequency_path)},
            "model_versions": {
                "fusion_method": self.fusion_config["fusion_method"],
                "preprocessing_version": self.fusion_config.get("preprocessing_version", "未记录"),
            },
            "fusion_config_path": str(self.fusion_config_path),
            "fusion_config_hash": self.fusion_config_hash,
            "spatial_weight": float(self.fusion_config["spatial_weight"]),
            "frequency_weight": float(self.fusion_config["frequency_weight"]),
            "frame_threshold": float(self.fusion_config["frame_threshold"]),
            "video_threshold": float(self.fusion_config["video_threshold"]),
            "video_aggregation": str(self.fusion_config["video_aggregation"]),
            "decision_threshold": None,
            "final_fake_score": None,
            "final_label": "证据不足",
            "provisional_label": None,
            "valid_frame_count": 0,
            "invalid_frame_count": 0,
            "candidate_frame_count": 0,
            "warnings": [],
            "error_message": "",
            "total_elapsed_seconds": 0.0,
            "timing": {},
            "media_metadata": {},
            "spatial_score_summary": {},
            "frequency_score_summary": {},
            "fusion_score_summary": {},
            "frame_results": [],
            "keyframe_paths": [],
            "run_directory": str(run_dir),
            "model_load_count_for_pipeline_instance": self._model_load_count,
            "temporary_files_cleaned": False,
        }

    def _resolve_model_path(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    @staticmethod
    def _score_summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    def _safe_cleanup(self, temp_dir: Path, run_dir: Path) -> bool:
        resolved_temp = temp_dir.resolve()
        try:
            resolved_temp.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise RuntimePipelineError(f"拒绝清理运行目录之外的临时路径：{resolved_temp}") from exc
        if resolved_temp.exists():
            shutil.rmtree(resolved_temp)
        return True

    def _finish(
        self,
        result: dict[str, Any],
        run_dir: Path,
        temp_dir: Path,
        logger: logging.Logger,
        started: float,
        cleanup: bool,
    ) -> dict[str, Any]:
        if cleanup:
            result["temporary_files_cleaned"] = self._safe_cleanup(temp_dir, run_dir)
        else:
            result["warnings"].append("已按调用参数保留临时候选帧，仅供调试使用。")
        result["model_load_count_for_pipeline_instance"] = self._model_load_count
        result["total_elapsed_seconds"] = time.perf_counter() - started
        report_started = time.perf_counter()
        self.report_service.write_reports(result, run_dir)
        result["timing"]["report_seconds"] = time.perf_counter() - report_started
        result["total_elapsed_seconds"] = time.perf_counter() - started
        self.report_service.write_reports(result, run_dir)
        logger.info("运行完成：status=%s label=%s score=%s", result["status"], result["final_label"], result["final_fake_score"])
        self._close_logger(logger)
        return result

    def detect_media(
        self,
        input_path: str | Path,
        target_valid_frames: int | None = None,
        min_valid_frames: int | None = None,
        cleanup_temporary: bool | None = None,
    ) -> dict[str, Any]:
        path = Path(input_path).resolve()
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return self.detect_image(path, cleanup_temporary=cleanup_temporary)
        if suffix in VIDEO_EXTENSIONS:
            return self.detect_video(
                path,
                target_valid_frames=target_valid_frames,
                min_valid_frames=min_valid_frames,
                cleanup_temporary=cleanup_temporary,
            )
        return self._unsupported_media_result(path, cleanup_temporary)

    def _validate_input_file(self, path: Path) -> None:
        if not path.exists():
            raise RuntimePipelineError(f"输入文件不存在：{path}")
        if not path.is_file():
            raise RuntimePipelineError(f"输入路径不是文件：{path}")

    def _unsupported_media_result(self, path: Path, cleanup_temporary: bool | None) -> dict[str, Any]:
        started = time.perf_counter()
        run_id, run_dir, temp_dir, logger = self._create_run()
        result = self._base_result(run_id, run_dir, path, "unknown")
        try:
            self._validate_input_file(path)
            hash_started = time.perf_counter()
            result["input_sha256"] = calculate_sha256(path)
            result["timing"]["sha256_seconds"] = time.perf_counter() - hash_started
            supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
            raise RuntimePipelineError(f"不支持的文件格式：{path.suffix.lower() or '无扩展名'}；支持：{supported}")
        except RuntimePipelineError as exc:
            result["status"] = "ERROR"
            result["error_message"] = str(exc)
            result["warnings"].append(str(exc))
            logger.error(str(exc))
        cleanup = self.runtime_config.cleanup_temporary if cleanup_temporary is None else cleanup_temporary
        return self._finish(result, run_dir, temp_dir, logger, started, cleanup)

    def detect_image(
        self,
        input_path: str | Path,
        cleanup_temporary: bool | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        path = Path(input_path).resolve()
        run_id, run_dir, temp_dir, logger = self._create_run()
        result = self._base_result(run_id, run_dir, path, "image")
        cleanup = self.runtime_config.cleanup_temporary if cleanup_temporary is None else cleanup_temporary
        try:
            self._validate_input_file(path)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise RuntimePipelineError(f"不支持的图片格式：{path.suffix.lower() or '无扩展名'}")
            hash_started = time.perf_counter()
            result["input_sha256"] = calculate_sha256(path)
            result["timing"]["sha256_seconds"] = time.perf_counter() - hash_started
            decode_started = time.perf_counter()
            image = read_image(path)
            if image is None:
                raise RuntimePipelineError(f"无法读取图片，文件可能已损坏：{path.name}")
            height, width = image.shape[:2]
            result["media_metadata"] = {
                "width": int(width),
                "height": int(height),
                "format": path.suffix.lower().lstrip(".").upper(),
                "channels": int(image.shape[2]) if image.ndim == 3 else 1,
            }
            result["timing"]["metadata_and_decode_seconds"] = time.perf_counter() - decode_started
            model_load_seconds = self._ensure_model_loaded()
            result["timing"]["model_load_seconds"] = model_load_seconds
            result["actual_device"] = str(getattr(self.detector, "device", result["actual_device"]))
            face_started = time.perf_counter()
            face = extract_largest_face(image, self._active_face_detector(), self.preprocess_config)
            result["timing"]["face_detection_seconds"] = time.perf_counter() - face_started
            result["candidate_frame_count"] = 1
            if face is None:
                message = "未检测到满足质量要求的人脸，无法形成可靠判断。"
                result["status"] = "INSUFFICIENT_EVIDENCE"
                result["invalid_frame_count"] = 1
                result["warnings"].append(message)
                result["frame_results"] = [
                    {
                        "candidate_index": 1,
                        "source_frame_index": 0,
                        "timestamp_seconds": 0.0,
                        "status": "NO_FACE",
                        "invalid_reason": message,
                    }
                ]
                logger.warning(message)
                return self._finish(result, run_dir, temp_dir, logger, started, cleanup)
            preview_path = run_dir / "keyframes" / "keyframe_01.jpg"
            write_image(preview_path, face)
            inference_started = time.perf_counter()
            prediction = self.detector.predict(path)
            result["timing"]["inference_seconds"] = time.perf_counter() - inference_started
            frame = self._valid_frame_result(1, 0, 0.0, prediction)
            result["frame_results"] = [frame]
            result["valid_frame_count"] = 1
            result["keyframe_paths"] = ["keyframes/keyframe_01.jpg"]
            result["final_fake_score"] = float(prediction["fusion_score"])
            result["decision_threshold"] = float(self.fusion_config["frame_threshold"])
            result["final_label"] = str(prediction["predicted_label"])
            result["provisional_label"] = result["final_label"]
            result["status"] = "SUCCESS"
            self._set_score_summaries(result)
        except (RuntimePipelineError, DetectorError, OSError, ValueError) as exc:
            result["status"] = "ERROR"
            result["error_message"] = str(exc)
            result["warnings"].append(str(exc))
            logger.error("图片检测失败：%s", exc)
        return self._finish(result, run_dir, temp_dir, logger, started, cleanup)

    def detect_video(
        self,
        input_path: str | Path,
        target_valid_frames: int | None = None,
        min_valid_frames: int | None = None,
        cleanup_temporary: bool | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        path = Path(input_path).resolve()
        run_id, run_dir, temp_dir, logger = self._create_run()
        result = self._base_result(run_id, run_dir, path, "video")
        config = self.runtime_config.with_overrides(target_valid_frames, min_valid_frames)
        cleanup = config.cleanup_temporary if cleanup_temporary is None else cleanup_temporary
        capture: cv2.VideoCapture | None = None
        valid_frames: list[tuple[dict[str, Any], np.ndarray]] = []
        try:
            self._validate_input_file(path)
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise RuntimePipelineError(f"不支持的视频格式：{path.suffix.lower() or '无扩展名'}")
            hash_started = time.perf_counter()
            result["input_sha256"] = calculate_sha256(path)
            result["timing"]["sha256_seconds"] = time.perf_counter() - hash_started
            metadata_started = time.perf_counter()
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimePipelineError(f"无法打开视频，文件可能已损坏或编码不受支持：{path.name}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            width = max(0, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = max(0, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")
            duration = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
            if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
                raise RuntimePipelineError(f"无法读取有效的视频元数据：{path.name}")
            result["media_metadata"] = {
                "duration_seconds": float(duration),
                "fps": float(fps),
                "total_frames": int(total_frames),
                "width": int(width),
                "height": int(height),
                "codec": codec or "unknown",
                "target_valid_frames": config.target_valid_frames,
                "min_valid_frames": config.min_valid_frames,
            }
            result["timing"]["metadata_seconds"] = time.perf_counter() - metadata_started
            result["timing"]["model_load_seconds"] = self._ensure_model_loaded()
            result["actual_device"] = str(getattr(self.detector, "device", result["actual_device"]))
            candidate_indices = self._candidate_indices(total_frames, config)
            inference_total = 0.0
            face_total = 0.0
            for candidate_number, frame_index in enumerate(candidate_indices, start=1):
                if len(valid_frames) >= config.target_valid_frames:
                    break
                timestamp = frame_index / fps
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_image = capture.read()
                if not ok or frame_image is None:
                    result["frame_results"].append(
                        self._invalid_frame_result(candidate_number, frame_index, timestamp, "DECODE_FAILED", "候选帧解码失败。")
                    )
                    continue
                face_started = time.perf_counter()
                face = extract_largest_face(frame_image, self._active_face_detector(), self.preprocess_config)
                face_total += time.perf_counter() - face_started
                if face is None:
                    result["frame_results"].append(
                        self._invalid_frame_result(
                            candidate_number,
                            frame_index,
                            timestamp,
                            "NO_FACE",
                            "未检测到满足质量要求的人脸。",
                        )
                    )
                    continue
                candidate_path = temp_dir / f"candidate_{candidate_number:03d}_idx{frame_index:08d}.jpg"
                write_image(candidate_path, frame_image)
                inference_started = time.perf_counter()
                try:
                    prediction = self.detector.predict(candidate_path)
                except DetectorError as exc:
                    inference_total += time.perf_counter() - inference_started
                    result["frame_results"].append(
                        self._invalid_frame_result(candidate_number, frame_index, timestamp, "INFERENCE_FAILED", str(exc))
                    )
                    continue
                inference_total += time.perf_counter() - inference_started
                frame_result = self._valid_frame_result(candidate_number, frame_index, timestamp, prediction)
                result["frame_results"].append(frame_result)
                valid_frames.append((frame_result, frame_image.copy()))
            result["timing"]["face_detection_seconds"] = face_total
            result["timing"]["inference_seconds"] = inference_total
            result["candidate_frame_count"] = len(result["frame_results"])
            result["valid_frame_count"] = len(valid_frames)
            result["invalid_frame_count"] = len(result["frame_results"]) - len(valid_frames)
            if result["candidate_frame_count"] == 0:
                raise RuntimePipelineError(f"视频中没有可读取的候选画面：{path.name}")
            self._save_keyframes(valid_frames, run_dir, config.keyframe_count, result)
            self._set_score_summaries(result)
            fusion_values = [float(item[0]["fusion_score"]) for item in valid_frames]
            if fusion_values:
                result["final_fake_score"] = self._aggregate(fusion_values, result["video_aggregation"])
                result["decision_threshold"] = float(self.fusion_config["video_threshold"])
                result["provisional_label"] = (
                    "FAKE" if result["final_fake_score"] >= result["decision_threshold"] else "REAL"
                )
            if len(valid_frames) == 0:
                message = "视频候选帧中未检测到满足质量要求的人脸，无法形成可靠判断。"
                result["status"] = "INSUFFICIENT_EVIDENCE"
                result["warnings"].append(message)
                logger.warning(message)
            elif len(valid_frames) < config.min_valid_frames:
                message = (
                    f"有效人脸帧仅{len(valid_frames)}张，低于配置要求的{config.min_valid_frames}张；"
                    "仅保留暂定评分供人工查看，不形成高可信结论。"
                )
                result["status"] = "LOW_CONFIDENCE"
                result["warnings"].append(message)
                logger.warning(message)
            else:
                result["status"] = "SUCCESS"
                result["final_label"] = str(result["provisional_label"])
        except (RuntimePipelineError, DetectorError, OSError, ValueError) as exc:
            result["status"] = "ERROR"
            result["error_message"] = str(exc)
            result["warnings"].append(str(exc))
            logger.error("视频检测失败：%s", exc)
        finally:
            if capture is not None:
                capture.release()
        return self._finish(result, run_dir, temp_dir, logger, started, cleanup)

    @staticmethod
    def _candidate_indices(total_frames: int, config: RuntimePipelineConfig) -> list[int]:
        primary_count = min(config.target_valid_frames, total_frames)
        candidate_count = min(
            total_frames,
            config.max_candidate_frames,
            max(config.target_valid_frames, config.target_valid_frames * config.candidate_multiplier),
        )
        primary = np.linspace(0, total_frames - 1, num=primary_count, dtype=np.int64).tolist()
        fallback = np.linspace(0, total_frames - 1, num=candidate_count, dtype=np.int64).tolist()
        ordered: list[int] = []
        seen: set[int] = set()
        for index in [*primary, *fallback]:
            value = int(index)
            if value not in seen:
                ordered.append(value)
                seen.add(value)
        return ordered

    @staticmethod
    def _valid_frame_result(
        candidate_index: int,
        source_frame_index: int,
        timestamp_seconds: float,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "candidate_index": int(candidate_index),
            "source_frame_index": int(source_frame_index),
            "timestamp_seconds": float(timestamp_seconds),
            "status": "VALID",
            "invalid_reason": "",
            "spatial_score": float(prediction["spatial_score"]),
            "frequency_score": float(prediction["frequency_score"]),
            "fusion_score": float(prediction["fusion_score"]),
            "frame_threshold": float(prediction["threshold"]),
            "predicted_label": str(prediction["predicted_label"]),
            "risk_level": str(prediction["risk_level"]),
            "inference_ms": float(prediction.get("inference_ms", 0.0)),
        }

    @staticmethod
    def _invalid_frame_result(
        candidate_index: int,
        source_frame_index: int,
        timestamp_seconds: float,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "candidate_index": int(candidate_index),
            "source_frame_index": int(source_frame_index),
            "timestamp_seconds": float(timestamp_seconds),
            "status": status,
            "invalid_reason": reason,
        }

    def _save_keyframes(
        self,
        valid_frames: list[tuple[dict[str, Any], np.ndarray]],
        run_dir: Path,
        keyframe_count: int,
        result: dict[str, Any],
    ) -> None:
        ranked = sorted(valid_frames, key=lambda item: float(item[0]["fusion_score"]), reverse=True)
        for rank, (_, image) in enumerate(ranked[:keyframe_count], start=1):
            relative = f"keyframes/keyframe_{rank:02d}.jpg"
            write_image(run_dir / relative, image)
            result["keyframe_paths"].append(relative)

    def _set_score_summaries(self, result: dict[str, Any]) -> None:
        valid = [item for item in result["frame_results"] if item.get("status") == "VALID"]
        result["spatial_score_summary"] = self._score_summary([float(item["spatial_score"]) for item in valid])
        result["frequency_score_summary"] = self._score_summary([float(item["frequency_score"]) for item in valid])
        result["fusion_score_summary"] = self._score_summary([float(item["fusion_score"]) for item in valid])

    @staticmethod
    def _aggregate(values: list[float], method: str) -> float:
        array = np.asarray(values, dtype=np.float64)
        if method == "median":
            return float(np.median(array))
        if method == "mean":
            return float(np.mean(array))
        if method in {"top2_mean", "top3_mean"}:
            k = 2 if method == "top2_mean" else 3
            return float(np.mean(np.sort(array)[-min(k, len(array)) :]))
        raise RuntimePipelineError(f"不支持的视频聚合方法：{method}")

    @property
    def model_load_count(self) -> int:
        return self._model_load_count
