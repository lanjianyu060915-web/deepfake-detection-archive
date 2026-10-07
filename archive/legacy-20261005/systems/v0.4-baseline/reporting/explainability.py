"""Pure, deterministic explanations derived from persisted module results.

This module intentionally has no Streamlit, model, network, or filesystem
dependency.  It never manufactures a model observation.  Missing diagnostics
remain missing and are called out as limitations instead of being guessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any


_UNCERTAIN_LABELS = {
    "low_confidence",
    "insufficient",
    "insufficient_evidence",
    "no_face",
    "undetermined",
    "unknown",
}
_WINDOWS_PATH_RE = re.compile(r"(?i)(?<![\w])(?:[a-z]:\\|\\\\)[^\r\n；;，,。]+")
_FLOAT_TOLERANCE = 2e-6


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _at(data: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = data
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            if current is not None:
                return current
    return None


def _lookup(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if data.get(key) is not None:
            return data[key]
    for container_name in (
        "metrics",
        "details",
        "diagnostics",
        "method",
        "runtime",
        "provenance",
        "explanation",
    ):
        container = data.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            if container.get(key) is not None:
                return container[key]
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _strings(*values: Any) -> list[str]:
    output: list[str] = []
    for value in values:
        items: Iterable[Any]
        if isinstance(value, str):
            items = (value,)
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
            items = value
        else:
            continue
        for item in items:
            text = str(item).strip()
            if text and text not in output:
                output.append(text)
    return output


def _fmt(value: float | None, digits: int = 6) -> str:
    return "未记录" if value is None else f"{value:.{digits}f}"


def display_name(value: Any, fallback: str = "未记录") -> str:
    """Return a path's final component without exposing its parent directory."""

    text = _text(value)
    if not text:
        return fallback
    if "\\" in text or re.match(r"(?i)^[a-z]:", text):
        return PureWindowsPath(text).name or fallback
    return PurePosixPath(text).name or fallback


def redact_text(value: Any) -> str:
    """Redact Windows absolute paths from ordinary-user prose."""

    text = str(value or "")

    def replace(match: re.Match[str]) -> str:
        return f"[…\\{display_name(match.group(0).rstrip())}]"

    return _WINDOWS_PATH_RE.sub(replace, text)


def _status(data: Mapping[str, Any]) -> str:
    value = str(data.get("status") or "error").strip().lower()
    return value if value in {"success", "error", "skipped"} else "error"


def _common_lists(data: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    warnings = _strings(data.get("warnings"), _mapping(data.get("details")).get("warnings"))
    limitations = _strings(data.get("limitations"))
    return warnings, limitations


def _weight_mapping(data: Mapping[str, Any], names: tuple[str, str]) -> tuple[float | None, float | None, str | None]:
    first_name, second_name = names
    containers: list[tuple[str, Mapping[str, Any]]] = []
    for container_name in ("method", "metrics", "details", "diagnostics"):
        container = data.get(container_name)
        if isinstance(container, Mapping):
            containers.append((container_name, container))
            for nested_name in ("weights", "fusion_weights", "score_weights"):
                nested = container.get(nested_name)
                if isinstance(nested, Mapping):
                    containers.append((f"{container_name}.{nested_name}", nested))
    containers.insert(0, ("result", data))
    aliases = {
        "spatial": ("spatial", "spatial_weight", "spatial_score_weight"),
        "frequency": ("frequency", "frequency_weight", "frequency_score_weight"),
        "global": ("global", "global_weight", "global_score_weight"),
        "local": ("local", "local_weight", "local_score_weight"),
    }
    for source, container in containers:
        first = next((_number(container.get(key)) for key in aliases[first_name] if _number(container.get(key)) is not None), None)
        second = next((_number(container.get(key)) for key in aliases[second_name] if _number(container.get(key)) is not None), None)
        if first is not None and second is not None:
            return first, second, source
    return None, None, None


def _verified_legacy_weights(
    first_score: float | None,
    second_score: float | None,
    reported_score: float | None,
    weights: tuple[float, float],
) -> bool:
    if first_score is None or second_score is None or reported_score is None:
        return False
    calculated = weights[0] * first_score + weights[1] * second_score
    return abs(calculated - reported_score) <= _FLOAT_TOLERANCE


def _formula(
    *,
    first_name: str,
    second_name: str,
    first_score: float | None,
    second_score: float | None,
    first_weight: float | None,
    second_weight: float | None,
    reported_score: float | None,
    weight_source: str | None,
) -> dict[str, Any]:
    available = all(
        value is not None
        for value in (first_score, second_score, first_weight, second_weight, reported_score)
    )
    calculated = (
        float(first_weight * first_score + second_weight * second_score)
        if available
        else None
    )
    delta = calculated - reported_score if calculated is not None and reported_score is not None else None
    return {
        "available": available,
        "expression": f"{first_name}权重 × {first_name}分数 + {second_name}权重 × {second_name}分数",
        "first_name": first_name,
        "second_name": second_name,
        "first_score": first_score,
        "second_score": second_score,
        "first_weight": first_weight,
        "second_weight": second_weight,
        "reported_score": reported_score,
        "calculated_score": calculated,
        "calculation_difference": delta,
        "consistent": abs(delta) <= _FLOAT_TOLERANCE if delta is not None else None,
        "weight_source": weight_source,
        "substitution": (
            f"{_fmt(first_weight)} × {_fmt(first_score)} + "
            f"{_fmt(second_weight)} × {_fmt(second_score)} = {_fmt(calculated)}"
            if available
            else None
        ),
    }


def _base_result(module: str, data: Mapping[str, Any]) -> dict[str, Any]:
    warnings, limitations = _common_lists(data)
    return {
        "module": module,
        "status": _status(data),
        "headline": "",
        "summary": "",
        "basis": [],
        "technical": {},
        "warnings": warnings,
        "limitations": limitations,
    }


def build_deepfake_explanation(result: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _mapping(result)
    output = _base_result("deepfake", data)
    status = output["status"]
    if status == "skipped":
        reason = _text(_mapping(data.get("details")).get("reason")) or "本次任务未选择该模块。"
        output.update(headline="未运行AI换脸/深伪风险甄别", summary=redact_text(reason))
        return output
    if status == "error":
        message = _text(data.get("error_message")) or "模块未返回可复核结果。"
        output.update(headline="AI换脸/深伪风险甄别执行失败", summary=redact_text(message))
        output["technical"] = {"error_message": message}
        return output

    details = _mapping(data.get("details"))
    metrics = _mapping(data.get("metrics"))
    method = _mapping(data.get("method"))
    provenance = _mapping(data.get("provenance"))
    spatial = _number(_lookup(data, "spatial_score"))
    frequency = _number(_lookup(data, "frequency_score"))
    fusion = _number(_lookup(data, "fusion_score"))
    score = _number(data.get("score"))
    if fusion is None:
        fusion = score
    threshold = _number(_lookup(data, "decision_threshold", "threshold"))
    margin = fusion - threshold if fusion is not None and threshold is not None else None

    spatial_weight, frequency_weight, weight_source = _weight_mapping(data, ("spatial", "frequency"))
    if spatial_weight is None and _verified_legacy_weights(spatial, frequency, fusion, (0.62, 0.38)):
        spatial_weight, frequency_weight = 0.62, 0.38
        weight_source = "legacy_v2_verified_against_reported_fusion_score"
    formula = _formula(
        first_name="空间域",
        second_name="频域",
        first_score=spatial,
        second_score=frequency,
        first_weight=spatial_weight,
        second_weight=frequency_weight,
        reported_score=fusion,
        weight_source=weight_source,
    )

    label = str(data.get("label") or "").strip().lower().replace("-", "_")
    is_deepfake = data.get("is_deepfake")
    uncertain = is_deepfake is None or label in _UNCERTAIN_LABELS
    if label == "no_face":
        output["headline"] = "未检测到有效人脸，无法完成空间域人脸伪造分析"
    elif uncertain:
        output["headline"] = "检测结论：不确定（证据不足，未形成真假判断）"
    elif is_deepfake is True or label == "deepfake":
        output["headline"] = "检测结论：疑似AI换脸"
    elif is_deepfake is False or label == "real":
        output["headline"] = "检测结论：未检出明显换脸风险"
    else:
        output["headline"] = "检测结论：返回字段不一致，需人工复核"
        output["warnings"].append("模型标签与布尔结论无法一致解释。")

    label_verdict = True if label == "deepfake" else False if label == "real" else None
    if isinstance(is_deepfake, bool) and label_verdict is not None and is_deepfake != label_verdict:
        output["warnings"].append("结果中的模型标签与布尔结论相互矛盾，已保留原字段并提示人工复核。")

    summary_parts = [output["headline"]]
    if fusion is not None:
        summary_parts.append(f"风险分数为{_fmt(fusion)}")
    if threshold is not None:
        summary_parts.append(f"判定阈值为{_fmt(threshold)}")
    if margin is not None:
        summary_parts.append(f"与阈值差值为{margin:+.6f}")
    output["summary"] = "；".join(summary_parts) + "。"

    if spatial is not None:
        output["basis"].append(f"空间域模型输出分数为{_fmt(spatial)}。")
    if frequency is not None:
        output["basis"].append(f"频域模型输出分数为{_fmt(frequency)}。")
    if formula["available"]:
        output["basis"].append(f"融合计算：{formula['substitution']}；报告风险分数为{_fmt(fusion)}。")
        if formula["consistent"] is False:
            output["warnings"].append("按已记录权重复算的融合分数与报告分数不一致，未据此改写原结果。")
    else:
        output["warnings"].append("结果未同时记录两个分支分数和可验证融合权重，无法完整复核融合公式。")
    if margin is not None:
        relation = "高于或等于" if margin >= 0 else "低于"
        output["basis"].append(
            f"最终风险分数{relation}现有判定阈值，实际差值为{margin:+.6f}；本说明不把风险分数称为系统准确率。"
        )
        if isinstance(is_deepfake, bool) and is_deepfake != (margin >= 0):
            output["warnings"].append("结果中的真假结论与已记录阈值比较不一致，未根据阈值反向改写原始结论。")

    band = _number(_lookup(data, "decision_boundary_band", "boundary_margin"))
    band_validated = _boolean(_lookup(data, "decision_boundary_band_validated", "boundary_band_validated"))
    if band is not None and band_validated is True and margin is not None and abs(margin) <= band:
        output["warnings"].append("本次结果落入已记录的决策边界带，稳定性有限，建议结合原始文件和其他证据人工复核。")
    elif band_validated is not True:
        output["limitations"].append("结果未提供经独立验证的决策边界带；这里只列出实际阈值差值，不声称高置信或低置信。")
    output["limitations"].append("风险分数是本模块的模型输出，不是系统准确率、司法结论或跨模块综合概率。")

    model_versions = _mapping(details.get("model_versions"))
    model_hashes = _mapping(_lookup(data, "model_hashes", "model_sha256"))
    method_name = _text(method.get("name"))
    spatial_model_name = _text(method.get("spatial_model"))
    frequency_model_name = _text(method.get("frequency_model"))
    model_name = _text(_lookup(data, "model_name", "architecture"))
    if not model_name:
        model_name = " + ".join(
            value for value in (spatial_model_name, frequency_model_name) if value
        ) or method_name
    output["technical"] = {
        "model_name": model_name,
        "method_name": method_name,
        "spatial_model_name": spatial_model_name,
        "frequency_model_name": frequency_model_name,
        "model_version": _lookup(data, "model_version") or model_versions,
        "spatial_score": spatial,
        "frequency_score": frequency,
        "spatial_weight": spatial_weight,
        "frequency_weight": frequency_weight,
        "fusion_score": fusion,
        "decision_threshold": threshold,
        "threshold_margin": margin,
        "formula": formula,
        "actual_device": _lookup(data, "actual_device", "device"),
        "valid_face_or_frame_count": _lookup(data, "valid_face_count", "valid_frame_count", "query_face_count"),
        "input_sha256": _lookup(data, "input_sha256", "input_file_sha256"),
        "model_hashes": model_hashes,
        "spatial_model_sha256": _lookup(data, "spatial_model_sha256") or model_hashes.get("spatial"),
        "frequency_model_sha256": _lookup(data, "frequency_model_sha256") or model_hashes.get("frequency"),
        "fusion_config_sha256": _lookup(data, "fusion_config_hash", "fusion_config_sha256"),
        "runtime_seconds": _number(data.get("runtime_seconds")),
        "native_status": details.get("native_status"),
        "native_run_directory": details.get("native_run_directory"),
        "native_report_paths": details.get("native_report_paths"),
        "keyframe_paths": details.get("keyframe_paths"),
        "warnings": list(output["warnings"]),
        "method": method,
        "metrics": metrics,
        "provenance": provenance,
    }
    return output


def _identity_candidates(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        return []
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates, start=1):
        candidate = _mapping(raw)
        mean = _number(candidate.get("mean_similarity"))
        if mean is None:
            mean = _number(candidate.get("score"))
        best_image = candidate.get("best_reference_image") or candidate.get("reference_image")
        reference_scores: list[dict[str, Any]] = []
        for item in candidate.get("reference_scores", []) if isinstance(candidate.get("reference_scores"), list) else []:
            record = _mapping(item)
            reference_scores.append(
                {
                    "reference_image": record.get("reference_image"),
                    "reference_name": display_name(record.get("reference_image")),
                    "similarity": _number(record.get("similarity")),
                }
            )
        output.append(
            {
                "rank": _integer(candidate.get("rank")) or index,
                "identity_id": candidate.get("identity_id"),
                "identity_name": candidate.get("identity_name") or candidate.get("identity_id"),
                "mean_similarity": mean,
                "score": _number(candidate.get("score")),
                "reference_count": _integer(candidate.get("reference_count")),
                "best_reference_score": _number(candidate.get("best_reference_score")),
                "best_reference_image": best_image,
                "best_reference_name": display_name(best_image),
                "reference_scores": reference_scores,
            }
        )
    return output


def build_identity_explanation(result: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _mapping(result)
    output = _base_result("identity", data)
    status = output["status"]
    metrics = _mapping(data.get("metrics"))
    method = _mapping(data.get("method"))
    provenance = _mapping(data.get("provenance"))
    query_face_count = _integer(_lookup(data, "query_face_count", "detected_face_count"))
    candidates = _identity_candidates(data)

    if status == "skipped":
        reason = _text(_mapping(data.get("details")).get("reason")) or "本次任务未选择该模块。"
        output.update(headline="未运行人脸身份搜索", summary=redact_text(reason))
        return output
    if status == "error":
        message = _text(data.get("error_message")) or "模块未返回可复核结果。"
        if query_face_count == 0 or "no face" in message.lower() or "无人脸" in message:
            headline = "未检测到可用于身份检索的人脸"
        else:
            headline = "人脸身份搜索执行失败"
        output.update(headline=headline, summary=redact_text(message))
        output["technical"] = {"query_face_count": query_face_count, "error_message": message}
        output["limitations"].append("本系统仅提供当前身份库内的闭集候选排序，不能单独确认人物身份。")
        return output

    if not candidates:
        output.update(headline="当前身份库未返回候选", summary="检索已完成，但结果中没有可展示的库内候选。")
        output["warnings"].append("没有候选分数可用于排序解释。")
    else:
        top1 = candidates[0]
        output["headline"] = f"当前身份库内第一候选：{top1['identity_name'] or '未命名人员'}"
        output["summary"] = (
            f"第一候选的人员级平均余弦相似度为{_fmt(top1['mean_similarity'])}。"
            "这是当前身份库中的候选排序，不是最终身份确认。"
        )

    top1 = candidates[0] if candidates else None
    top2 = candidates[1] if len(candidates) > 1 else None
    margin = (
        top1["mean_similarity"] - top2["mean_similarity"]
        if top1 and top2 and top1["mean_similarity"] is not None and top2["mean_similarity"] is not None
        else None
    )
    reported_margin = _number(_lookup(data, "top1_top2_margin"))
    if margin is not None and reported_margin is not None and abs(margin - reported_margin) > _FLOAT_TOLERANCE:
        output["warnings"].append("候选分数复算的Top1与Top2差值和结果诊断字段不一致；解释采用候选原始分数复算值。")

    embedding_dimension = _integer(_lookup(data, "embedding_dimension"))
    aggregation = _lookup(data, "aggregation_method", "aggregation")
    if isinstance(aggregation, Mapping):
        aggregation = aggregation.get("name") or aggregation.get("method")
    if candidates:
        dimension_text = f"{embedding_dimension}维归一化身份特征" if embedding_dimension else "归一化身份特征"
        output["basis"].append(f"系统检测查询图片中的人脸并提取{dimension_text}。")
        output["basis"].append("查询特征与身份库中每张有效参考图进行余弦相似度比较；余弦相似度不是身份概率。")
        if str(aggregation or "").strip().lower() in {"mean", "average", "arithmetic_mean"}:
            output["basis"].append("同一人员有多张有效参考图时，系统对逐图余弦相似度取算术均值，作为人员级分数。")
        elif aggregation:
            output["basis"].append(f"人员级聚合方法记录为“{aggregation}”。")
        else:
            output["basis"].append("当前兼容结果未单独记录聚合方法；人员级分数按结果JSON中的实际候选分数展示。")
        if top1:
            detail = f"第一候选{top1['identity_name']}的人员级平均相似度为{_fmt(top1['mean_similarity'])}"
            if top2:
                detail += f"，第二候选{top2['identity_name']}为{_fmt(top2['mean_similarity'])}，差值为{_fmt(margin)}"
            output["basis"].append(detail + "，因此其在当前库内排序第一。")
            if top1["reference_count"] is not None:
                output["basis"].append(f"第一候选使用了{top1['reference_count']}张有效参考图。")
            if top1["best_reference_score"] is not None:
                output["basis"].append(
                    f"最佳单张参考图为{top1['best_reference_name']}，单张余弦相似度为{_fmt(top1['best_reference_score'])}。"
                )

    if query_face_count is None:
        output["limitations"].append("兼容结果未保存查询图片的人脸检测数量。")
    if top1 and top1["reference_count"] is None:
        output["limitations"].append("兼容结果未保存第一候选的有效参考图数量及逐图相似度。")
    output["limitations"].append("当前系统未启用经独立验证的开放集拒识阈值；第一候选不能表述为已确认身份。")

    output["technical"] = {
        "query_face_count": query_face_count,
        "selected_face_bbox": _lookup(data, "selected_face_bbox"),
        "face_detection_score": _number(_lookup(data, "face_detection_score")),
        "embedding_dimension": embedding_dimension,
        "model_name": _lookup(data, "model_name"),
        "aggregation_method": aggregation,
        "library_identity_count": _integer(_lookup(data, "library_identity_count", "identity_count")),
        "library_image_count": _integer(_lookup(data, "library_image_count", "image_count")),
        "valid_embedding_count": _integer(_lookup(data, "valid_embedding_count", "embedding_count")),
        "skipped_library_image_count": _integer(_lookup(data, "skipped_library_image_count")),
        "top1_top2_margin": margin,
        "reported_top1_top2_margin": reported_margin,
        "runtime_seconds": _number(data.get("runtime_seconds")),
        "candidates": candidates,
        "method": method,
        "metrics": metrics,
        "provenance": provenance,
        "warnings": list(output["warnings"]),
    }
    return output


def _source_local_formula(candidate: Mapping[str, Any]) -> dict[str, Any]:
    good = _integer(candidate.get("good_match_count"))
    inliers = _integer(candidate.get("ransac_inlier_count"))
    ratio = _number(candidate.get("ransac_inlier_ratio"))
    if ratio is None and good is not None and inliers is not None:
        ratio = inliers / max(good, 1)
    reported = _number(candidate.get("local_score"))
    components = _mapping(candidate.get("local_score_components"))
    mode = str(components.get("mode") or "").strip().lower()
    if mode == "insufficient_matches" or (good is not None and good < 4):
        calculated = min(good / 40.0, 0.15) if good is not None else None
        delta = calculated - reported if calculated is not None and reported is not None else None
        return {
            "available": calculated is not None and reported is not None,
            "mode": "insufficient_matches",
            "expression": "min(有效匹配点数 / 40, 0.15)",
            "substitution": f"min({good} / 40, 0.15) = {_fmt(calculated)}" if good is not None else None,
            "good_match_count": good,
            "ransac_inlier_count": inliers,
            "ransac_inlier_ratio": ratio,
            "calculated_score": calculated,
            "reported_score": reported,
            "calculation_difference": delta,
            "consistent": abs(delta) <= _FLOAT_TOLERANCE if delta is not None else None,
        }

    ratio_weight = _number(
        components.get("ratio_weight")
        if components.get("ratio_weight") is not None
        else components.get("inlier_ratio_weight")
    )
    count_weight = _number(
        components.get("count_weight")
        if components.get("count_weight") is not None
        else components.get("inlier_count_weight")
    )
    count_score = _number(
        components.get("count_score")
        if components.get("count_score") is not None
        else components.get("normalized_inlier_count")
    )
    if count_score is None and inliers is not None:
        count_score = min(inliers / 40.0, 1.0)
    if ratio_weight is None or count_weight is None:
        if ratio is not None and count_score is not None and reported is not None:
            legacy = 0.6 * ratio + 0.4 * count_score
            if abs(legacy - reported) <= _FLOAT_TOLERANCE:
                ratio_weight, count_weight = 0.6, 0.4
    available = all(value is not None for value in (ratio, count_score, ratio_weight, count_weight, reported))
    calculated = ratio_weight * ratio + count_weight * count_score if available else None
    delta = calculated - reported if calculated is not None and reported is not None else None
    return {
        "available": available,
        "mode": mode or "ransac",
        "expression": "内点率权重 × RANSAC内点率 + 内点数量权重 × min(RANSAC内点数 / 40, 1)",
        "substitution": (
            f"{_fmt(ratio_weight)} × {_fmt(ratio)} + {_fmt(count_weight)} × {_fmt(count_score)} = {_fmt(calculated)}"
            if available
            else None
        ),
        "good_match_count": good,
        "ransac_inlier_count": inliers,
        "ransac_inlier_ratio": ratio,
        "count_score": count_score,
        "ratio_weight": ratio_weight,
        "count_weight": count_weight,
        "calculated_score": calculated,
        "reported_score": reported,
        "calculation_difference": delta,
        "consistent": abs(delta) <= _FLOAT_TOLERANCE if delta is not None else None,
    }


def _source_candidates(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list):
        return []
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates, start=1):
        candidate = _mapping(raw)
        record = {
            **candidate,
            "rank": _integer(candidate.get("rank")) or index,
            "file_name": display_name(candidate.get("file_path")),
            "visualization_name": display_name(candidate.get("visualization_path")),
            "global_score": _number(candidate.get("global_score")),
            "local_score": _number(candidate.get("local_score")),
            "final_score": _number(candidate.get("final_score")),
            "query_keypoint_count": _integer(candidate.get("query_keypoint_count")),
            "candidate_keypoint_count": _integer(candidate.get("candidate_keypoint_count")),
            "good_match_count": _integer(candidate.get("good_match_count")),
            "ransac_inlier_count": _integer(candidate.get("ransac_inlier_count")),
            "ransac_inlier_ratio": _number(candidate.get("ransac_inlier_ratio")),
            "lowe_ratio": _number(candidate.get("lowe_ratio")),
            "ransac_threshold": _number(candidate.get("ransac_threshold")),
        }
        record["local_formula"] = _source_local_formula(record)
        output.append(record)
    return output


def build_source_explanation(result: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _mapping(result)
    output = _base_result("source", data)
    status = output["status"]
    metrics = _mapping(data.get("metrics"))
    diagnostics = _mapping(data.get("diagnostics"))
    method = _mapping(data.get("method"))
    provenance = _mapping(data.get("provenance"))
    candidates = _source_candidates(data)
    query_face_detected = _boolean(_lookup(data, "query_face_detected"))
    face_mask_applied = _boolean(_lookup(data, "face_mask_applied"))
    library_decoded = _integer(_lookup(data, "library_decoded"))

    if status == "skipped":
        reason = _text(_mapping(data.get("details")).get("reason")) or "本次任务未选择该模块。"
        output.update(headline="未运行原始底图搜索", summary=redact_text(reason))
        return output
    if status == "error":
        message = _text(data.get("error_message")) or "模块未返回可复核结果。"
        headline = "底图库为空或没有可解码图片" if library_decoded == 0 or "empty" in message.lower() or "图库为空" in message else "原始底图搜索执行失败"
        output.update(headline=headline, summary=redact_text(message))
        output["technical"] = {"library_decoded": library_decoded, "error_message": message}
        output["limitations"].append("底图检索结果仅是当前库内候选排序，不能单独确认原始底图。")
        return output

    if not candidates:
        output.update(headline="当前底图库未返回候选", summary="检索已完成，但结果中没有可展示的库内候选。")
        output["warnings"].append("没有候选分数可用于排序解释。")
    else:
        top1 = candidates[0]
        output["headline"] = f"当前底图库内最相似候选：{top1['file_name']}"
        output["summary"] = (
            f"第一候选的全局相似度为{_fmt(top1['global_score'])}，"
            f"局部相似度为{_fmt(top1['local_score'])}，最终相似度为{_fmt(top1['final_score'])}。"
        )

    top1 = candidates[0] if candidates else None
    top2 = candidates[1] if len(candidates) > 1 else None
    margin = (
        top1["final_score"] - top2["final_score"]
        if top1 and top2 and top1["final_score"] is not None and top2["final_score"] is not None
        else None
    )
    reported_margin = _number(_lookup(data, "top1_top2_margin"))
    if margin is not None and reported_margin is not None and abs(margin - reported_margin) > _FLOAT_TOLERANCE:
        output["warnings"].append("候选分数复算的Top1与Top2差值和结果诊断字段不一致；解释采用候选原始分数复算值。")

    if face_mask_applied is True:
        output["basis"].append("系统已屏蔽查询图片中的主要人脸区域，以降低身份区域对底图检索的干扰。")
    elif query_face_detected is False or face_mask_applied is False:
        output["basis"].append("查询图未检测到可屏蔽的人脸区域，本次按完整图像提取底图特征。")
    else:
        output["basis"].append("兼容结果未记录查询图人脸屏蔽状态，不能据此声称已成功屏蔽。")
    output["basis"].append("系统利用颜色、灰度、边缘和外围区域等全局特征进行库内粗检索，再对粗排候选执行SIFT局部复核。")

    global_weight, local_weight, weight_source = _weight_mapping(data, ("global", "local"))
    if top1:
        candidate_global_weight = _number(top1.get("global_weight"))
        candidate_local_weight = _number(top1.get("local_weight"))
        if candidate_global_weight is not None and candidate_local_weight is not None:
            global_weight, local_weight, weight_source = candidate_global_weight, candidate_local_weight, "candidate"
        if global_weight is None and _verified_legacy_weights(
            top1["global_score"], top1["local_score"], top1["final_score"], (0.45, 0.55)
        ):
            global_weight, local_weight = 0.45, 0.55
            weight_source = "legacy_v2_verified_against_reported_final_score"
        final_formula = _formula(
            first_name="全局",
            second_name="局部",
            first_score=top1["global_score"],
            second_score=top1["local_score"],
            first_weight=global_weight,
            second_weight=local_weight,
            reported_score=top1["final_score"],
            weight_source=weight_source,
        )
        local_formula = top1["local_formula"]
        if local_formula["available"]:
            if local_formula["mode"] == "insufficient_matches":
                output["basis"].append(
                    f"有效匹配点少于4个，未执行RANSAC；局部分按特殊分支计算：{local_formula['substitution']}。"
                )
            else:
                output["basis"].append(
                    f"Lowe筛选后有效匹配点{local_formula['good_match_count']}个，RANSAC内点{local_formula['ransac_inlier_count']}个，"
                    f"内点率{_fmt(local_formula['ransac_inlier_ratio'])}；局部分计算：{local_formula['substitution']}。"
                )
            if local_formula["consistent"] is False:
                output["warnings"].append("按已记录局部诊断复算的局部分与报告局部分不一致，未据此改写原结果。")
        else:
            output["limitations"].append("兼容结果未保存局部匹配点和RANSAC诊断，无法完整复核局部分公式。")
        if final_formula["available"]:
            output["basis"].append(f"最终分计算：{final_formula['substitution']}；报告最终分为{_fmt(top1['final_score'])}。")
            if final_formula["consistent"] is False:
                output["warnings"].append("按已记录权重复算的最终分与报告最终分不一致，未据此改写原结果。")
        else:
            output["warnings"].append("结果未同时记录可验证的全局/局部权重，无法完整复核最终分公式。")
        if top2 and margin is not None:
            output["basis"].append(
                f"第二候选{top2['file_name']}的最终分为{_fmt(top2['final_score'])}，第一名与第二名差值为{_fmt(margin)}。"
            )
    else:
        final_formula = _formula(
            first_name="全局", second_name="局部", first_score=None, second_score=None,
            first_weight=None, second_weight=None, reported_score=None, weight_source=None,
        )

    output["limitations"].append("全局分、局部分和最终分都是本检索算法的库内排序分数，不是图像来源概率。")
    output["limitations"].append("当前底图库内最相似候选不能表述为已确认原始底图。")
    output["technical"] = {
        "query_face_detected": query_face_detected,
        "query_face_bbox": _lookup(data, "query_face_bbox"),
        "face_mask_applied": face_mask_applied,
        "masked_query_path": _lookup(data, "masked_query_path"),
        "library_file_count": _integer(_lookup(data, "library_file_count")),
        "library_decoded": library_decoded,
        "library_skipped": _integer(_lookup(data, "library_skipped_count", "library_skipped")),
        "shortlist_size": _integer(_lookup(data, "shortlist_size")),
        "lowe_ratio": _number(_lookup(data, "lowe_ratio")),
        "ransac_threshold": _number(_lookup(data, "ransac_threshold")),
        "global_weight": global_weight,
        "local_weight": local_weight,
        "top1_top2_margin": margin,
        "reported_top1_top2_margin": reported_margin,
        "final_formula": final_formula,
        "runtime_seconds": _number(data.get("runtime_seconds")),
        "candidates": candidates,
        "method": method,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "provenance": provenance,
        "warnings": list(output["warnings"]),
    }
    return output


def build_explanations(
    deepfake_result: Mapping[str, Any] | None,
    identity_result: Mapping[str, Any] | None,
    source_result: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Build all three independent explanations without a fused score."""

    return {
        "deepfake": build_deepfake_explanation(deepfake_result),
        "identity": build_identity_explanation(identity_result),
        "source": build_source_explanation(source_result),
    }


# Concise aliases for callers that prefer verb-style names.
explain_deepfake = build_deepfake_explanation
explain_identity = build_identity_explanation
explain_source = build_source_explanation


__all__ = [
    "build_deepfake_explanation",
    "build_explanations",
    "build_identity_explanation",
    "build_source_explanation",
    "display_name",
    "explain_deepfake",
    "explain_identity",
    "explain_source",
    "redact_text",
]
