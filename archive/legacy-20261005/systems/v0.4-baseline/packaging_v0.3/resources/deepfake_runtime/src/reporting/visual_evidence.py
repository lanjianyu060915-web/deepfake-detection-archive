from __future__ import annotations

import hashlib
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.preprocessing import read_image, write_image


class VisualEvidenceError(RuntimeError):
    """Raised only inside the optional AIGC explanation boundary."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(task_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(task_root.resolve()).as_posix()
    except ValueError as exc:
        raise VisualEvidenceError(f"证据文件不在任务目录内：{path}") from exc


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_u8(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low = float(finite.min())
    high = float(finite.max())
    if math.isclose(low, high, abs_tol=1e-15):
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


class _ArtifactManifest:
    def __init__(self, task_root: Path) -> None:
        self.task_root = task_root.resolve()
        self.items: list[dict[str, Any]] = []

    def add(
        self,
        artifact_id: str,
        kind: str,
        path: Path,
        description: str,
    ) -> str:
        resolved = path.resolve()
        relative = _relative(self.task_root, resolved)
        if not resolved.is_file():
            raise VisualEvidenceError(f"准备登记的证据文件不存在：{resolved}")
        self.items.append(
            {
                "id": str(artifact_id),
                "module": "deepfake",
                "kind": str(kind),
                "path": relative,
                "sha256": _sha256(resolved),
                "size_bytes": int(resolved.stat().st_size),
                "status": "generated",
                "description": str(description),
            }
        )
        return relative


def _event(
    stage: str,
    status: str,
    message: str,
    *,
    started_at: str,
    elapsed_seconds: float,
    progress_current: int,
    progress_total: int,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "module": "deepfake",
        "stage": stage,
        "status": status,
        "message": message,
        "progress_current": int(progress_current),
        "progress_total": int(progress_total),
        "progress_percent": round(100.0 * progress_current / progress_total, 3),
        "started_at": started_at,
        "ended_at": _now_iso(),
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "payload": dict(payload or {}),
    }


def _detect_eye_candidates(
    image: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    """Return only points actually emitted by OpenCV's eye cascade.

    The AIGC detector does not ship a five-point landmark model.  It is safer
    to expose an empty list than to synthesize anatomical positions.  When the
    packaged eye cascade detects ocular regions, their measured centres are
    reported as eye candidates and are explicitly not described as alignment
    landmarks.
    """

    x, y, width, height = face_box
    roi_height = max(1, int(round(height * 0.68)))
    roi = image[y : y + roi_height, x : x + width]
    if roi.size == 0:
        return []
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
    classifier = cv2.CascadeClassifier(str(cascade_path))
    if classifier.empty():
        return []
    gray = cv2.equalizeHist(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    detections = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(12, width // 12), max(8, height // 16)),
    )
    ranked = sorted(
        detections,
        key=lambda box: int(box[2]) * int(box[3]),
        reverse=True,
    )[:2]
    points = []
    for index, (eye_x, eye_y, eye_w, eye_h) in enumerate(sorted(ranked, key=lambda box: box[0]), start=1):
        points.append(
            {
                "id": f"eye_candidate_{index}",
                "x": int(x + eye_x + eye_w / 2),
                "y": int(y + eye_y + eye_h / 2),
                "detector": "OpenCV Haar eye cascade",
                "role": "detected_eye_region_center",
            }
        )
    return points


def _save_face_evidence(
    *,
    query_path: Path,
    evidence_dir: Path,
    prefix: str,
    detector: Any,
    manifest: _ArtifactManifest,
    native_frame: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    spatial_preprocessor = getattr(detector, "spatial_preprocessor", None)
    if spatial_preprocessor is None:
        raise VisualEvidenceError("空间域预处理器不可用，无法复现模型实际人脸裁剪。")
    image = read_image(query_path)
    if image is None:
        raise VisualEvidenceError(f"无法读取解释输入图片：{query_path.name}")

    frame = native_frame if isinstance(native_frame, dict) else {}
    stored_expanded = frame.get("face_bbox_xywh")
    stored_raw = frame.get("raw_face_bbox_xywh")
    has_stored_boxes = (
        isinstance(stored_expanded, list)
        and len(stored_expanded) == 4
        and all(_finite(value) is not None for value in stored_expanded)
    )
    if has_stored_boxes:
        # These coordinates were retained at the real inference point.  Reuse
        # them so evidence generation does not repeat a costly high-resolution
        # face detector pass.  The tensor transform below is the same official
        # MobileNetV3 transform used by SpatialMobileNetDetector._prepare.
        expanded_bbox = tuple(int(value) for value in stored_expanded)
        raw_box = (
            tuple(int(value) for value in stored_raw)
            if isinstance(stored_raw, list) and len(stored_raw) == 4
            else None
        )
        original_size = tuple(
            int(value) for value in frame.get("original_size", [image.shape[1], image.shape[0]])
        )
        model_crop_size = tuple(
            int(value)
            for value in frame.get(
                "model_input_size", spatial_preprocessor.preprocess_config.face_size
            )
        )
        detected_face_count = 1
    else:
        tensor, expanded_bbox, original_size, model_crop_size, raw_box = (
            spatial_preprocessor._prepare_with_evidence(query_path)
        )
        raw_box = tuple(int(value) for value in raw_box)
        detected_face_count = 1

    expanded_x, expanded_y, expanded_w, expanded_h = [int(value) for value in expanded_bbox]
    crop = image[
        expanded_y : expanded_y + expanded_h,
        expanded_x : expanded_x + expanded_w,
    ]
    if crop.size == 0:
        raise VisualEvidenceError("模型实际人脸裁剪为空。")
    input_width, input_height = spatial_preprocessor.preprocess_config.face_size
    crop = cv2.resize(crop, (input_width, input_height), interpolation=cv2.INTER_AREA)

    if has_stored_boxes:
        from PIL import Image
        from torchvision.models import MobileNet_V3_Small_Weights

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = MobileNet_V3_Small_Weights.DEFAULT.transforms()(
            Image.fromarray(rgb)
        ).unsqueeze(0)

    point_search_box = raw_box or tuple(int(value) for value in expanded_bbox)
    keypoints = _detect_eye_candidates(image, point_search_box)
    annotated = image.copy()
    if raw_box is not None:
        raw_x, raw_y, raw_w, raw_h = raw_box
        cv2.rectangle(
            annotated,
            (raw_x, raw_y),
            (raw_x + raw_w, raw_y + raw_h),
            (54, 226, 123),
            8,
        )
        label_origin = (max(0, raw_x), max(35, raw_y - 18))
    else:
        label_origin = (max(0, expanded_x), max(35, expanded_y - 18))
    cv2.rectangle(
        annotated,
        (expanded_x, expanded_y),
        (expanded_x + expanded_w, expanded_y + expanded_h),
        (255, 190, 60),
        8,
    )
    cv2.putText(
        annotated,
        "FACE 1 (selected model crop)",
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (54, 226, 123),
        3,
        cv2.LINE_AA,
    )
    for point in keypoints:
        centre = (int(point["x"]), int(point["y"]))
        cv2.circle(annotated, centre, 14, (20, 230, 255), -1, cv2.LINE_AA)

    annotated_path = evidence_dir / f"{prefix}_face_detected.jpg"
    crop_path = evidence_dir / f"{prefix}_face_crop_224.png"
    write_image(annotated_path, annotated)
    write_image(crop_path, crop)
    annotated_rel = manifest.add(
        "deepfake.face_detection",
        "face_detection_image",
        annotated_path,
        "本次任务实际人脸检测框、扩展裁剪框及真实检测到的眼部候选点。",
    )
    crop_rel = manifest.add(
        "deepfake.face_crop",
        "model_input_crop",
        crop_path,
        "按空间域正式预处理规则生成的224×224实际模型人脸裁剪。",
    )
    face_evidence = {
        "status": "generated",
        "face_detection_image": annotated_rel,
        "face_crop_image": crop_rel,
        "detected_face_count": detected_face_count,
        "selected_face_index": 1,
        "raw_face_bbox_xywh": list(raw_box) if raw_box is not None else None,
        "expanded_face_bbox_xywh": list(expanded_bbox),
        "crop_expand_ratio": float(spatial_preprocessor.preprocess_config.face_expand_ratio),
        "original_size": [int(original_size[0]), int(original_size[1])],
        "model_input_size": [int(model_crop_size[0]), int(model_crop_size[1])],
        "keypoints": keypoints,
        "keypoint_status": "detected_eye_candidates" if keypoints else "not_detected",
        "keypoint_method": "OpenCV Haar eye cascade" if keypoints else None,
        "keypoint_limitation": (
            "空间域模型未配置五点关键点模型；仅展示眼部级联分类器真实检出的候选中心，"
            "不将几何估算点冒充真实关键点。"
        ),
        "selection_rule": "OpenCV Haar frontal-face detector返回的最大人脸。",
        "box_source": (
            "retained_from_actual_inference"
            if has_stored_boxes
            else "recomputed_with_formal_preprocessor"
        ),
    }
    return face_evidence, tensor, crop


def _spatial_occlusion_evidence(
    *,
    tensor: Any,
    crop: np.ndarray,
    detector: Any,
    evidence_dir: Path,
    prefix: str,
    manifest: _ArtifactManifest,
) -> dict[str, Any]:
    import torch

    model = getattr(detector, "spatial_model", None)
    if model is None:
        raise VisualEvidenceError("空间域模型尚未加载。")
    device = getattr(detector, "device", torch.device("cpu"))
    grid_rows = 6
    grid_columns = 6
    source = tensor.detach().clone()
    _, _, height, width = source.shape
    variants = [source]
    cells: list[tuple[int, int, int, int]] = []
    for row in range(grid_rows):
        y1 = int(round(row * height / grid_rows))
        y2 = int(round((row + 1) * height / grid_rows))
        for column in range(grid_columns):
            x1 = int(round(column * width / grid_columns))
            x2 = int(round((column + 1) * width / grid_columns))
            occluded = source.clone()
            # Zero is the ImageNet-normalized channel mean.  This is a stable,
            # deterministic occluder and never touches model parameters.
            occluded[:, :, y1:y2, x1:x2] = 0.0
            variants.append(occluded)
            cells.append((x1, y1, x2, y2))

    started = time.perf_counter()
    batch = torch.cat(variants, dim=0).to(device)
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        logits = model(batch)
        scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    baseline = float(scores[0])
    signed_changes = baseline - scores[1:]
    sensitivity = np.abs(signed_changes).reshape(grid_rows, grid_columns)
    signed_matrix = signed_changes.reshape(grid_rows, grid_columns)
    upsampled = cv2.resize(
        _normalize_u8(sensitivity),
        (crop.shape[1], crop.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    heatmap = cv2.applyColorMap(upsampled, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(crop, 0.58, heatmap, 0.42, 0.0)
    maximum_index = int(np.argmax(sensitivity))
    max_row, max_column = divmod(maximum_index, grid_columns)
    max_x1, max_y1, max_x2, max_y2 = cells[maximum_index]
    cv2.rectangle(
        overlay,
        (max_x1, max_y1),
        (max_x2 - 1, max_y2 - 1),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    heatmap_path = evidence_dir / f"{prefix}_spatial_occlusion_heatmap.png"
    overlay_path = evidence_dir / f"{prefix}_spatial_occlusion_overlay.jpg"
    write_image(heatmap_path, heatmap)
    write_image(overlay_path, overlay)
    heatmap_rel = manifest.add(
        "deepfake.spatial_heatmap",
        "spatial_sensitivity_heatmap",
        heatmap_path,
        "基于当前MobileNetV3-Small真实输出变化的遮挡敏感性纯热力图。",
    )
    overlay_rel = manifest.add(
        "deepfake.spatial_overlay",
        "spatial_sensitivity_overlay",
        overlay_path,
        "遮挡敏感性热力图叠加到本次实际模型人脸裁剪。",
    )
    return {
        "status": "generated",
        "method": "occlusion_sensitivity",
        "target_class": "FAKE",
        "baseline_spatial_score_recomputed": baseline,
        "grid_shape": [grid_rows, grid_columns],
        "occluder": "ImageNet-normalized channel mean (tensor value 0)",
        "signed_score_change_matrix": signed_matrix.tolist(),
        "absolute_sensitivity_matrix": sensitivity.tolist(),
        "heatmap_image": heatmap_rel,
        "overlay_image": overlay_rel,
        "strongest_region": {
            "grid_row": int(max_row),
            "grid_column": int(max_column),
            "model_input_bbox_xyxy": [max_x1, max_y1, max_x2, max_y2],
            "signed_score_change": float(signed_matrix[max_row, max_column]),
            "absolute_score_change": float(sensitivity[max_row, max_column]),
        },
        "generation_seconds": float(elapsed),
        "disclaimer": "该图反映模型对遮挡区域的输出敏感性，不等于像素级伪造定位或已确认换脸区域。",
    }


def _draw_radial_profile(values: list[float]) -> np.ndarray:
    width, height = 900, 460
    canvas = np.full((height, width, 3), (28, 37, 48), dtype=np.uint8)
    left, right, top, bottom = 90, 45, 48, 70
    cv2.line(canvas, (left, height - bottom), (width - right, height - bottom), (170, 180, 190), 2)
    cv2.line(canvas, (left, top), (left, height - bottom), (170, 180, 190), 2)
    if values:
        array = np.asarray(values, dtype=np.float64)
        minimum, maximum = float(array.min()), float(array.max())
        span = max(maximum - minimum, 1e-12)
        points = []
        for index, value in enumerate(array):
            x = left + int(round(index * (width - left - right) / max(1, len(array) - 1)))
            y = height - bottom - int(round((float(value) - minimum) / span * (height - top - bottom)))
            points.append((x, y))
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (61, 213, 255), 4, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 4, (82, 242, 188), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"max {maximum:.4f}", (left + 10, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 235, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"min {minimum:.4f}", (left + 10, height - bottom - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 235, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Radial FFT log-magnitude profile", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 248, 250), 2, cv2.LINE_AA)
    cv2.putText(canvas, "normalized radius (center -> edge)", (280, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 218, 226), 2, cv2.LINE_AA)
    return canvas


def _frequency_contributions(frequency_detector: Any, query_path: Path) -> dict[str, Any]:
    vector, summary = frequency_detector._extract_with_evidence(query_path)
    pipeline = frequency_detector.pipeline
    if pipeline is None:
        raise VisualEvidenceError("频域LogisticRegression管线尚未加载。")
    steps = list(getattr(pipeline, "steps", []))
    if not steps:
        raise VisualEvidenceError("频域模型不是可解释的sklearn Pipeline。")
    classifier = steps[-1][1]
    if not hasattr(classifier, "coef_") or not hasattr(classifier, "intercept_"):
        raise VisualEvidenceError("频域末级分类器没有LogisticRegression系数。")
    if len(steps) > 1:
        model_input = np.asarray(pipeline[:-1].transform(vector.reshape(1, -1)), dtype=np.float64)[0]
        preprocessing_steps = [name for name, _ in steps[:-1]]
    else:
        model_input = vector.astype(np.float64, copy=False)
        preprocessing_steps = []
    classes = np.asarray(getattr(classifier, "classes_", getattr(pipeline, "classes_", [])))
    fake_locations = np.flatnonzero(classes == 1)
    if fake_locations.size != 1:
        raise VisualEvidenceError("频域LogisticRegression没有唯一FAKE类别1。")
    fake_location = int(fake_locations[0])
    coefficients = np.asarray(classifier.coef_, dtype=np.float64)
    intercepts = np.asarray(classifier.intercept_, dtype=np.float64).reshape(-1)
    if coefficients.shape[0] == 1 and classes.size == 2:
        coefficient = coefficients[0]
        intercept = float(intercepts[0])
        if fake_location == 0:
            coefficient = -coefficient
            intercept = -intercept
    else:
        coefficient = coefficients[fake_location]
        intercept = float(intercepts[fake_location])
    if coefficient.shape != model_input.shape:
        raise VisualEvidenceError(
            f"频域系数维度与模型输入不一致：{coefficient.shape} != {model_input.shape}"
        )
    contributions = model_input * coefficient
    decision_logit = float(intercept + contributions.sum())
    reconstructed_probability = float(1.0 / (1.0 + math.exp(-decision_logit)))
    probabilities = np.asarray(pipeline.predict_proba(vector.reshape(1, -1))[0], dtype=np.float64)
    actual_probability = float(probabilities[fake_location])
    if abs(reconstructed_probability - actual_probability) > 1e-8:
        raise VisualEvidenceError(
            "无法严格用标准化特征值×回归系数重建频域FAKE概率。"
        )

    from src.detectors.frequency_stat import FEATURE_NAMES

    rows: list[dict[str, Any]] = []
    for index, contribution in enumerate(contributions):
        rows.append(
            {
                "index": int(index),
                "name": str(FEATURE_NAMES[index]),
                "raw_value": float(vector[index]),
                "model_input_value": float(model_input[index]),
                "coefficient": float(coefficient[index]),
                "contribution": float(contribution),
                "direction": "increase_fake_logit" if contribution >= 0.0 else "decrease_fake_logit",
            }
        )
    by_absolute = sorted(rows, key=lambda item: abs(float(item["contribution"])), reverse=True)
    positive = sorted(
        (item for item in rows if float(item["contribution"]) > 0.0),
        key=lambda item: float(item["contribution"]),
        reverse=True,
    )[:10]
    negative = sorted(
        (item for item in rows if float(item["contribution"]) < 0.0),
        key=lambda item: float(item["contribution"]),
    )[:10]
    return {
        "status": "supported",
        "feature_dimension": int(vector.size),
        "feature_names_source": "FrequencyStatDetector.FEATURE_NAMES",
        "preprocessing_steps": preprocessing_steps,
        "contribution_formula": "actual_model_input_value × LogisticRegression coefficient",
        "intercept": intercept,
        "decision_logit": decision_logit,
        "reconstructed_fake_probability": reconstructed_probability,
        "actual_fake_probability": actual_probability,
        "reconstruction_absolute_error": abs(reconstructed_probability - actual_probability),
        "frequency_summary": summary,
        "top_by_absolute_contribution": by_absolute[:10],
        "top_positive_contributions": positive,
        "top_negative_contributions": negative,
        "all_feature_contributions": rows,
    }


def _frequency_evidence(
    *,
    crop: np.ndarray,
    query_path: Path,
    detector: Any,
    evidence_dir: Path,
    prefix: str,
    manifest: _ArtifactManifest,
) -> dict[str, Any]:
    frequency_detector = getattr(detector, "frequency_detector", None)
    if frequency_detector is None:
        raise VisualEvidenceError("频域检测器尚未加载。")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    fft_unshifted = np.fft.fft2(gray)
    unshifted_log = np.log1p(np.abs(fft_unshifted))
    shifted_log = np.log1p(np.abs(np.fft.fftshift(fft_unshifted)))
    unshifted_image = cv2.applyColorMap(_normalize_u8(unshifted_log), cv2.COLORMAP_VIRIDIS)
    shifted_image = cv2.applyColorMap(_normalize_u8(shifted_log), cv2.COLORMAP_VIRIDIS)

    height, width = shifted_log.shape
    y_grid, x_grid = np.ogrid[:height, :width]
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    radius = np.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
    normalized_radius = radius / max(float(radius.max()), 1e-12)
    radial_bin_count = 32
    radial_profile: list[dict[str, float]] = []
    profile_values: list[float] = []
    edges = np.linspace(0.0, 1.0, radial_bin_count + 1)
    for index in range(radial_bin_count):
        if index == radial_bin_count - 1:
            mask = (normalized_radius >= edges[index]) & (normalized_radius <= edges[index + 1])
        else:
            mask = (normalized_radius >= edges[index]) & (normalized_radius < edges[index + 1])
        value = float(shifted_log[mask].mean()) if np.any(mask) else 0.0
        profile_values.append(value)
        radial_profile.append(
            {
                "bin": int(index),
                "radius_start": float(edges[index]),
                "radius_end": float(edges[index + 1]),
                "mean_log_magnitude": value,
            }
        )
    profile_image = _draw_radial_profile(profile_values)

    unshifted_path = evidence_dir / f"{prefix}_fft_log_magnitude.png"
    shifted_path = evidence_dir / f"{prefix}_fft_centered_log_spectrum.png"
    profile_path = evidence_dir / f"{prefix}_fft_radial_profile.png"
    write_image(unshifted_path, unshifted_image)
    write_image(shifted_path, shifted_image)
    write_image(profile_path, profile_image)
    unshifted_rel = manifest.add(
        "deepfake.frequency_fft",
        "frequency_fft_log_magnitude",
        unshifted_path,
        "本次模型人脸裁剪的二维FFT对数幅度谱（未中心化显示）。",
    )
    shifted_rel = manifest.add(
        "deepfake.frequency_fft_centered",
        "frequency_fft_centered_spectrum",
        shifted_path,
        "本次模型人脸裁剪的中心化二维FFT对数幅度谱。",
    )
    profile_rel = manifest.add(
        "deepfake.frequency_radial_profile",
        "frequency_radial_profile",
        profile_path,
        "从中心到频谱边缘的32段真实径向对数幅度均值曲线。",
    )
    contribution_started = time.perf_counter()
    contribution_error = None
    try:
        contributions = _frequency_contributions(frequency_detector, query_path)
    except Exception as exc:  # explanation failure is deliberately non-fatal
        contributions = {
            "status": "unsupported",
            "message": "当前模型仅支持频域整体分数解释。",
        }
        contribution_error = f"{type(exc).__name__}: {exc}"
        contributions["technical_reason"] = contribution_error
    return {
        "status": "generated" if contribution_error is None else "partial",
        "face_crop_image": next(
            (item["path"] for item in manifest.items if item["id"] == "deepfake.face_crop"),
            None,
        ),
        "fft_log_magnitude_image": unshifted_rel,
        "fft_centered_spectrum_image": shifted_rel,
        "radial_profile_image": profile_rel,
        "radial_profile": radial_profile,
        "feature_contributions": contributions,
        "contribution_generation_seconds": time.perf_counter() - contribution_started,
    }


def _fusion_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    spatial_score = _finite(metrics.get("spatial_score"))
    frequency_score = _finite(metrics.get("frequency_score"))
    spatial_weight = _finite(metrics.get("spatial_weight"))
    frequency_weight = _finite(metrics.get("frequency_weight"))
    fusion_score = _finite(metrics.get("fusion_score"))
    threshold = _finite(metrics.get("decision_threshold"))
    margin = _finite(metrics.get("decision_margin"))
    complete = all(
        value is not None
        for value in (
            spatial_score,
            frequency_score,
            spatial_weight,
            frequency_weight,
            fusion_score,
            threshold,
            margin,
        )
    )
    if not complete:
        return {
            "status": "unavailable",
            "message": "本次任务未形成完整的空间域、频域和融合分数。",
        }
    assert spatial_score is not None
    assert frequency_score is not None
    assert spatial_weight is not None
    assert frequency_weight is not None
    assert fusion_score is not None
    assert threshold is not None
    assert margin is not None
    return {
        "status": "generated",
        "spatial_score": spatial_score,
        "spatial_weight": spatial_weight,
        "spatial_weighted_contribution": spatial_score * spatial_weight,
        "frequency_score": frequency_score,
        "frequency_weight": frequency_weight,
        "frequency_weighted_contribution": frequency_score * frequency_weight,
        "fusion_score": fusion_score,
        "decision_threshold": threshold,
        "decision_margin": margin,
        "decision_rule": "fusion_score >= decision_threshold",
        "formula": "spatial_weight × spatial_score + frequency_weight × frequency_score",
        "formula_substitution": (
            f"{spatial_weight:.12g} × {spatial_score:.12g} + "
            f"{frequency_weight:.12g} × {frequency_score:.12g} = {fusion_score:.12g}"
        ),
        "decision": (
            "检出较高换脸风险" if fusion_score >= threshold else "未检出明显换脸风险"
        ),
        "score_semantics": "当前任务的FAKE风险筛查分数，不是模型准确率或司法鉴定结论。",
    }


def _register_native_artifacts(
    native_result: dict[str, Any] | None,
    manifest: _ArtifactManifest,
) -> list[str]:
    if not isinstance(native_result, dict):
        return []
    run_value = native_result.get("run_directory")
    if not run_value:
        return []
    run_dir = Path(str(run_value)).resolve(strict=False)
    registered: list[str] = []
    candidates: list[tuple[str, str, Path, str]] = []
    report_paths = native_result.get("report_paths")
    if isinstance(report_paths, dict):
        for key, value in report_paths.items():
            if value:
                candidates.append(
                    (
                        f"deepfake.native.{key}",
                        f"native_{key}",
                        run_dir / str(value),
                        f"冻结AIGC管线本次运行生成的{key}。",
                    )
                )
    keyframes = native_result.get("keyframe_paths")
    if isinstance(keyframes, list):
        for index, value in enumerate(keyframes, start=1):
            candidates.append(
                (
                    f"deepfake.native.keyframe_{index:02d}",
                    "native_keyframe",
                    run_dir / str(value),
                    f"冻结AIGC管线本次运行的第{index}张有效关键帧/人脸预览。",
                )
            )
    known_paths = {item["path"] for item in manifest.items}
    for artifact_id, kind, path, description in candidates:
        if not path.is_file():
            continue
        relative = _relative(manifest.task_root, path)
        if relative in known_paths:
            continue
        registered.append(manifest.add(artifact_id, kind, path, description))
        known_paths.add(relative)
    return registered


def generate_deepfake_visual_evidence(
    *,
    query_path: str | Path,
    task_root: str | Path,
    detector: Any,
    result_payload: dict[str, Any],
    native_result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate deterministic, task-local evidence without changing scores.

    The caller must keep this function behind a broad exception boundary: all
    computations are explanatory follow-up work and must never change a valid
    model verdict into a failed detection.
    """

    started = time.perf_counter()
    started_at = _now_iso()
    query = Path(query_path).expanduser().resolve(strict=False)
    task = Path(task_root).expanduser().resolve(strict=False)
    evidence_dir = task / "visualizations" / "deepfake" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = _ArtifactManifest(task)
    events: list[dict[str, Any]] = []
    input_hash = _sha256(query)
    prefix = f"deepfake_{input_hash[:12]}"

    if str(result_payload.get("label", "")) == "no_face":
        image = read_image(query)
        if image is None:
            raise VisualEvidenceError(f"无法读取无人脸解释输入图片：{query.name}")
        annotated = image.copy()
        cv2.rectangle(
            annotated,
            (1, 1),
            (max(1, image.shape[1] - 2), max(1, image.shape[0] - 2)),
            (40, 90, 235),
            max(2, min(image.shape[:2]) // 120),
        )
        cv2.putText(
            annotated,
            "NO VALID FACE DETECTED",
            (max(8, image.shape[1] // 30), max(35, image.shape[0] // 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.55, min(image.shape[:2]) / 1000.0),
            (40, 90, 235),
            max(2, min(image.shape[:2]) // 240),
            cv2.LINE_AA,
        )
        no_face_path = evidence_dir / f"{prefix}_no_valid_face.jpg"
        write_image(no_face_path, annotated)
        no_face_rel = manifest.add(
            "deepfake.face_detection_no_face",
            "face_detection_failure_image",
            no_face_path,
            "本次真实人脸检测未检出有效人脸的可视化状态。",
        )
        native_artifacts = _register_native_artifacts(native_result, manifest)
        events.extend(
            [
                _event(
                    "visual_face_evidence_generated",
                    "warning",
                    "已生成未检出有效人脸的明确可视化状态。",
                    started_at=started_at,
                    elapsed_seconds=time.perf_counter() - started,
                    progress_current=10,
                    progress_total=11,
                    payload={"detected_face_count": 0},
                ),
                _event(
                    "visual_evidence_completed",
                    "success",
                    "无人脸任务的结构化状态与artifact manifest已生成。",
                    started_at=started_at,
                    elapsed_seconds=time.perf_counter() - started,
                    progress_current=11,
                    progress_total=11,
                    payload={"artifact_count": len(manifest.items)},
                ),
            ]
        )
        visual_evidence = {
            "deepfake": {
                "schema_version": "1.0",
                "status": "partial",
                "input_sha256": input_hash,
                "face_detection": {
                    "status": "no_face",
                    "face_detection_image": no_face_rel,
                    "detected_face_count": 0,
                    "selected_face_index": None,
                    "message": "未检测到有效人脸，当前图片无法完成空间域人脸伪造分析。",
                },
                "spatial": {"status": "unavailable", "message": "本次任务未生成该证据。"},
                "frequency": {"status": "unavailable", "message": "本次任务未生成该证据。"},
                "fusion": {"status": "unavailable", "message": "本次任务未形成可融合的分支分数。"},
                "native_artifacts": native_artifacts,
                "generation": {
                    "status": "partial",
                    "started_at": started_at,
                    "ended_at": _now_iso(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "artifact_count": len(manifest.items),
                    "failure_isolation": "无人脸状态不等同于模型加载失败或证据不足的真假结论。",
                },
            }
        }
        return visual_evidence, manifest.items, events

    face_started_at = _now_iso()
    face_started = time.perf_counter()
    native_frames = (
        native_result.get("frame_results", [])
        if isinstance(native_result, dict)
        else []
    )
    native_frame = next(
        (
            item
            for item in native_frames
            if isinstance(item, dict) and item.get("status") == "VALID"
        ),
        None,
    )
    face, tensor, crop = _save_face_evidence(
        query_path=query,
        evidence_dir=evidence_dir,
        prefix=prefix,
        detector=detector,
        manifest=manifest,
        native_frame=native_frame,
    )
    events.append(
        _event(
            "visual_face_evidence_generated",
            "success",
            "已按正式预处理路径生成真实人脸框与模型裁剪证据。",
            started_at=face_started_at,
            elapsed_seconds=time.perf_counter() - face_started,
            progress_current=10,
            progress_total=11,
            payload={
                "face_count": face["detected_face_count"],
                "selected_face_index": face["selected_face_index"],
                "keypoint_count": len(face["keypoints"]),
            },
        )
    )

    spatial_started_at = _now_iso()
    spatial_started = time.perf_counter()
    spatial = _spatial_occlusion_evidence(
        tensor=tensor,
        crop=crop,
        detector=detector,
        evidence_dir=evidence_dir,
        prefix=prefix,
        manifest=manifest,
    )
    recorded_spatial_score = _finite(
        (result_payload.get("metrics") or {}).get("spatial_score")
        if isinstance(result_payload.get("metrics"), dict)
        else None
    )
    spatial["recorded_detection_spatial_score"] = recorded_spatial_score
    spatial["baseline_recompute_absolute_error"] = (
        abs(spatial["baseline_spatial_score_recomputed"] - recorded_spatial_score)
        if recorded_spatial_score is not None
        else None
    )
    events.append(
        _event(
            "spatial_explanation_generated",
            "success",
            "已基于空间域模型真实输出生成遮挡敏感性证据。",
            started_at=spatial_started_at,
            elapsed_seconds=time.perf_counter() - spatial_started,
            progress_current=10,
            progress_total=11,
            payload={
                "method": spatial["method"],
                "grid_shape": spatial["grid_shape"],
                "baseline_spatial_score_recomputed": spatial[
                    "baseline_spatial_score_recomputed"
                ],
            },
        )
    )

    frequency_started_at = _now_iso()
    frequency_started = time.perf_counter()
    frequency = _frequency_evidence(
        crop=crop,
        query_path=query,
        detector=detector,
        evidence_dir=evidence_dir,
        prefix=prefix,
        manifest=manifest,
    )
    frequency_status = "success" if frequency["status"] == "generated" else "warning"
    events.append(
        _event(
            "frequency_explanation_generated",
            frequency_status,
            (
                "已生成真实FFT、径向频谱和LogisticRegression输入特征贡献。"
                if frequency_status == "success"
                else "已生成真实FFT与径向频谱；逐特征贡献不可严格恢复。"
            ),
            started_at=frequency_started_at,
            elapsed_seconds=time.perf_counter() - frequency_started,
            progress_current=10,
            progress_total=11,
            payload={
                "feature_contribution_status": frequency["feature_contributions"].get(
                    "status"
                )
            },
        )
    )

    fusion = _fusion_evidence(result_payload)
    native_artifacts = _register_native_artifacts(native_result, manifest)
    events.append(
        _event(
            "visual_evidence_completed",
            "success",
            "AIGC结构化证据与任务级artifact manifest已生成。",
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
            progress_current=11,
            progress_total=11,
            payload={
                "artifact_count": len(manifest.items),
                "native_artifact_count": len(native_artifacts),
            },
        )
    )
    evidence_status = (
        "generated"
        if frequency["status"] == "generated" and spatial["status"] == "generated"
        else "partial"
    )
    visual_evidence = {
        "deepfake": {
            "schema_version": "1.0",
            "status": evidence_status,
            "input_sha256": input_hash,
            "face_detection": face,
            "spatial": spatial,
            "frequency": frequency,
            "fusion": fusion,
            "native_artifacts": native_artifacts,
            "generation": {
                "status": evidence_status,
                "started_at": started_at,
                "ended_at": _now_iso(),
                "elapsed_seconds": time.perf_counter() - started,
                "artifact_count": len(manifest.items),
                "failure_isolation": "解释生成失败不会改变检测状态、分数、阈值或结论。",
            },
        }
    }
    return visual_evidence, manifest.items, events
