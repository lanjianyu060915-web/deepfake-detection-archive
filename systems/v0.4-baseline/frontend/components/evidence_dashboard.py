from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import streamlit as st

from components.live_progress import load_progress_events
from components.path_security import PathContainmentError, ResourcePathPolicy
from components.result_views import format_score, render_image


PAGES = (
    "研判总览",
    "AIGC鉴伪证据",
    "人脸身份检索",
    "原始底图溯源",
    "证据链与完整报告",
)

PAGE_KEYS = {
    "deepfake": "AIGC鉴伪证据",
    "identity": "人脸身份检索",
    "source": "原始底图溯源",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _six(value: Any) -> str:
    number = _number(value)
    return f"{number:.6f}" if number is not None else "—"


def _signed_six(value: Any) -> str:
    number = _number(value)
    return f"{number:+.6f}" if number is not None else "—"


def _safe_text(value: Any, fallback: str = "—") -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _table_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _basename(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "—"
    return Path(value).name or value


def _module_visual(data: Mapping[str, Any], module: str) -> dict[str, Any]:
    raw = data.get("visual_evidence")
    if not isinstance(raw, Mapping):
        details = data.get("details")
        raw = details.get("visual_evidence") if isinstance(details, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    nested = raw.get(module)
    return dict(nested) if isinstance(nested, Mapping) else dict(raw)


def _walk_values(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)


def _find_value(data: Mapping[str, Any], *keys: str) -> Any:
    wanted = set(keys)
    for key in keys:
        if key in data and data[key] not in (None, "", [], {}):
            return data[key]
    for key, value in _walk_values(data):
        if key in wanted and value not in (None, "", [], {}):
            return value
    return None


def _find_list(data: Mapping[str, Any], *keys: str) -> list[Any]:
    value = _find_value(data, *keys)
    return list(value) if isinstance(value, list) else []


def _path_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("path", "relative_path", "artifact_path", "file"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def resolve_evidence_path(
    value: Any,
    policy: ResourcePathPolicy,
    *,
    kind: str = "visualization",
    label: str,
) -> Path | None:
    raw = _path_value(value)
    if raw is None:
        return None
    try:
        return policy.resolve(raw, kind, label=label)  # type: ignore[arg-type]
    except PathContainmentError:
        return None


def _render_evidence_image(
    value: Any,
    policy: ResourcePathPolicy,
    *,
    caption: str,
    key: str,
    kind: str = "visualization",
) -> bool:
    path = resolve_evidence_path(value, policy, kind=kind, label=caption)
    if path is None or not path.is_file():
        st.warning(f"本次任务未生成该证据：{caption}")
        return False
    render_image(path, caption, key)
    return True


def _query_path(data: Mapping[str, Any], policy: ResourcePathPolicy) -> Path | None:
    value = data.get("query_image")
    if not isinstance(value, str):
        return None
    try:
        path = policy.resolve(value, "query", label="query_image")
    except PathContainmentError:
        return None
    return path if path.is_file() else None


def collect_artifact_manifest(*results: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        module = str(result.get("module") or "unknown")
        raw = result.get("artifact_manifest")
        if not isinstance(raw, list):
            visual = result.get("visual_evidence")
            raw = visual.get("artifact_manifest") if isinstance(visual, Mapping) else None
        if not isinstance(raw, list):
            continue
        for index, item in enumerate(raw, 1):
            if not isinstance(item, Mapping):
                continue
            record = dict(item)
            record.setdefault("module", module)
            record.setdefault("id", f"{module}-{index}")
            path = _path_value(record.get("path") or record.get("relative_path")) or ""
            key = (str(record["module"]), path)
            if key in seen:
                continue
            seen.add(key)
            output.append(record)
    return output


def build_task_overview(
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    task_status: Mapping[str, Any] | None = None,
    final_report: Mapping[str, Any] | None = None,
    task_dir: Path | None = None,
    product_version: str = "0.4.0",
) -> dict[str, Any]:
    status = _mapping(task_status)
    report = _mapping(final_report)
    report_input = _mapping(report.get("input"))
    report_product = _mapping(report.get("product"))
    details = _mapping(deepfake.get("details"))
    metadata = _mapping(details.get("media_metadata"))
    deepfake_runtime = _mapping(deepfake.get("runtime"))
    identity_runtime = _mapping(identity.get("runtime"))
    source_runtime = _mapping(source.get("runtime"))
    query = str(
        deepfake.get("query_image")
        or identity.get("query_image")
        or source.get("query_image")
        or status.get("query_image")
        or ""
    )
    width = metadata.get("width") or report_input.get("width")
    height = metadata.get("height") or report_input.get("height")
    dimensions = f"{width} × {height}" if width and height else "—"
    input_hash = (
        details.get("input_sha256")
        or _mapping(deepfake.get("provenance")).get("input_sha256")
        or _mapping(source.get("provenance")).get("input_sha256")
        or report_input.get("sha256")
    )
    total_runtime = _number(status.get("runtime_seconds"))
    if total_runtime is None:
        module_values = [
            _number(deepfake.get("runtime_seconds")),
            _number(identity.get("runtime_seconds")),
            _number(source.get("runtime_seconds")),
        ]
        total_runtime = sum(value or 0.0 for value in module_values)
    device = (
        deepfake_runtime.get("actual_device")
        or details.get("actual_device")
        or identity_runtime.get("device")
        or "CPU"
    )
    return {
        "system_name": report_product.get("name") or "AI换脸涉案图像智能鉴伪与原始素材溯源系统",
        "product_version": report_product.get("version") or product_version,
        "task_id": status.get("task_id") or (task_dir.name if task_dir else "—"),
        "query_file": Path(query).name if query else "—",
        "dimensions": dimensions,
        "input_sha256": input_hash or "—",
        "task_status": status.get("status") or "已载入",
        "total_runtime": total_runtime,
        "deepfake_runtime": _number(deepfake_runtime.get("module_elapsed_seconds")) or _number(deepfake.get("runtime_seconds")),
        "identity_runtime": _number(identity_runtime.get("module_elapsed_seconds")) or _number(identity.get("runtime_seconds")),
        "source_runtime": _number(source_runtime.get("module_elapsed_seconds")) or _number(source.get("runtime_seconds")),
        "device": str(device),
        "started_at": status.get("started_at") or "—",
        "finished_at": status.get("finished_at") or "—",
    }


def _queue_page(page: str, request_key: str) -> None:
    st.session_state[request_key] = page


def _result_status(data: Mapping[str, Any]) -> str:
    return str(data.get("status") or "unknown")


def _render_task_header(info: Mapping[str, Any]) -> None:
    st.markdown(
        f"### {info['system_name']}  ·  v{info['product_version']}"
    )
    first = st.columns([1.2, 1.2, 1, 1, 1.6])
    first[0].metric("任务编号", _safe_text(info.get("task_id")))
    first[1].metric("查询文件", _safe_text(info.get("query_file")))
    first[2].metric("图片尺寸", _safe_text(info.get("dimensions")))
    first[3].metric("任务状态", _safe_text(info.get("task_status")))
    total = _number(info.get("total_runtime"))
    first[4].metric("任务总耗时", f"{total:.3f} 秒" if total is not None else "—")
    second = st.columns(5)
    second[0].metric("AIGC耗时", f"{_number(info.get('deepfake_runtime')) or 0.0:.3f} 秒")
    second[1].metric("身份检索耗时", f"{_number(info.get('identity_runtime')) or 0.0:.3f} 秒")
    second[2].metric("底图溯源耗时", f"{_number(info.get('source_runtime')) or 0.0:.3f} 秒")
    second[3].metric("运行设备", _safe_text(info.get("device")))
    second[4].metric("检测完成时间", _safe_text(info.get("finished_at")))
    digest = _safe_text(info.get("input_sha256"))
    st.caption(f"输入文件 SHA-256：{digest}")


def _render_overview_cards(
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    request_key: str,
    key_prefix: str,
) -> None:
    columns = st.columns(3)
    details = _mapping(deepfake.get("details"))
    metrics = _mapping(deepfake.get("metrics"))
    with columns[0]:
        with st.container(border=True):
            st.markdown("#### AIGC鉴伪")
            st.metric("检测结论", _safe_text(_mapping(deepfake.get("explanation")).get("decision") or deepfake.get("label")))
            st.metric("融合风险分数", _six(metrics.get("fusion_score") or deepfake.get("score")))
            pair = st.columns(2)
            pair[0].metric("空间域", _six(metrics.get("spatial_score") or details.get("spatial_score")))
            pair[1].metric("频域", _six(metrics.get("frequency_score") or details.get("frequency_score")))
            st.caption(
                f"阈值 {_six(metrics.get('decision_threshold') or details.get('decision_threshold'))} · "
                f"差值 {_signed_six(metrics.get('decision_margin'))} · 状态 {_result_status(deepfake)}"
            )
            st.button(
                "查看AIGC证据 →",
                key=f"{key_prefix}_goto_deepfake",
                width="stretch",
                on_click=_queue_page,
                args=(PAGE_KEYS["deepfake"], request_key),
            )
    candidates = _list(identity.get("candidates"))
    identity_metrics = _mapping(identity.get("metrics"))
    top_identity = _mapping(candidates[0]) if candidates else {}
    top2 = _mapping(candidates[1]) if len(candidates) > 1 else {}
    with columns[1]:
        with st.container(border=True):
            st.markdown("#### 人脸身份检索")
            st.metric("当前库内第一候选", _safe_text(top_identity.get("identity_name") or top_identity.get("identity_id")))
            st.metric("人员级平均余弦相似度", _six(top_identity.get("mean_similarity") or top_identity.get("score")))
            pair = st.columns(2)
            pair[0].metric("Top1-Top2", _six(identity_metrics.get("top1_top2_margin")))
            pair[1].metric("有效参考图", identity_metrics.get("valid_embedding_count") or "—")
            st.caption(
                f"第二候选 {_safe_text(top2.get('identity_name') or top2.get('identity_id'))} · "
                f"身份库 {identity_metrics.get('library_identity_count') or '—'} 人"
            )
            st.info("库内候选，尚未确认。余弦相似度不是身份概率。")
            st.button(
                "查看身份证据 →",
                key=f"{key_prefix}_goto_identity",
                width="stretch",
                on_click=_queue_page,
                args=(PAGE_KEYS["identity"], request_key),
            )
    source_candidates = _list(source.get("candidates"))
    source_metrics = _mapping(source.get("metrics"))
    top_source = _mapping(source_candidates[0]) if source_candidates else {}
    with columns[2]:
        with st.container(border=True):
            st.markdown("#### 原始底图溯源")
            st.metric("当前库内第一候选", _basename(top_source.get("file_path")))
            score_row = st.columns(3)
            score_row[0].metric("全局", _six(top_source.get("global_score")))
            score_row[1].metric("局部", _six(top_source.get("local_score")))
            score_row[2].metric("最终", _six(top_source.get("final_score")))
            st.caption(
                f"Top1-Top2 {_six(source_metrics.get('top1_top2_margin'))} · "
                f"RANSAC {top_source.get('ransac_inlier_count') or '—'} 点 / {_six(top_source.get('ransac_inlier_ratio'))}"
            )
            st.info("库内候选，尚未确认。检索分数不是来源概率。")
            st.button(
                "查看底图证据 →",
                key=f"{key_prefix}_goto_source",
                width="stretch",
                on_click=_queue_page,
                args=(PAGE_KEYS["source"], request_key),
            )


def _render_threshold_chart(score: Any, threshold: Any) -> None:
    score_value = _number(score)
    threshold_value = _number(threshold)
    if score_value is None or threshold_value is None:
        st.warning("本次结果缺少风险分数或阈值，无法绘制阈值尺。")
        return
    maximum = max(1.0, score_value, threshold_value)
    spec = {
        "data": {"values": [{"start": 0.0, "score": score_value, "threshold": threshold_value}]},
        "layer": [
            {
                "mark": {"type": "bar", "height": 34, "cornerRadiusEnd": 5, "color": "#315f86"},
                "encoding": {
                    "x": {"field": "start", "type": "quantitative", "scale": {"domain": [0, maximum]}, "title": "融合风险分数"},
                    "x2": {"field": "score"},
                    "tooltip": [{"field": "score", "type": "quantitative", "format": ".6f", "title": "风险分数"}],
                },
            },
            {
                "mark": {"type": "rule", "color": "#c55a3d", "strokeWidth": 4},
                "encoding": {
                    "x": {"field": "threshold", "type": "quantitative"},
                    "tooltip": [{"field": "threshold", "type": "quantitative", "format": ".6f", "title": "判定阈值"}],
                },
            },
        ],
        "height": 70,
    }
    st.vega_lite_chart(spec, width="stretch")


def _render_signed_contributions(contributions: Sequence[Any]) -> None:
    values: list[dict[str, Any]] = []
    for index, raw in enumerate(contributions):
        if not isinstance(raw, Mapping):
            continue
        contribution_value = raw.get("contribution")
        if contribution_value is None:
            contribution_value = raw.get("signed_contribution")
        contribution = _number(contribution_value)
        if contribution is None:
            continue
        name = raw.get("feature_name") or raw.get("feature") or raw.get("name")
        if not name:
            feature_index = raw.get("feature_index", raw.get("index", index))
            name = f"feature_{int(feature_index):02d}" if isinstance(feature_index, int) else f"feature_{index:02d}"
        values.append(
            {
                "feature": str(name),
                "contribution": contribution,
                "direction": "推动风险上升" if contribution >= 0 else "推动风险下降",
                "feature_value": raw.get("model_input_value")
                if raw.get("model_input_value") is not None
                else raw.get("feature_value")
                if raw.get("feature_value") is not None
                else raw.get("standardized_value"),
                "coefficient": raw.get("coefficient"),
            }
        )
    values = sorted(values, key=lambda item: abs(item["contribution"]), reverse=True)[:10]
    if not values:
        st.info("当前模型仅支持频域整体分数解释，本次没有可严格恢复的逐特征贡献。")
        return
    spec = {
        "data": {"values": values},
        "mark": {"type": "bar", "cornerRadiusEnd": 4},
        "encoding": {
            "y": {"field": "feature", "type": "nominal", "sort": "-x", "title": "真实特征索引/名称"},
            "x": {"field": "contribution", "type": "quantitative", "title": "标准化特征值 × 回归系数"},
            "color": {
                "field": "direction",
                "type": "nominal",
                "scale": {"domain": ["推动风险上升", "推动风险下降"], "range": ["#b75d3e", "#3b6f9c"]},
                "title": "作用方向",
            },
            "tooltip": [
                {"field": "feature", "title": "特征"},
                {"field": "contribution", "type": "quantitative", "format": ".8f", "title": "有符号贡献"},
                {"field": "feature_value", "title": "模型输入值"},
                {"field": "coefficient", "title": "回归系数"},
            ],
        },
        "height": max(260, 28 * len(values)),
    }
    st.vega_lite_chart(spec, width="stretch")


def _render_deepfake_page(data: Mapping[str, Any], policy: ResourcePathPolicy, key_prefix: str) -> None:
    st.subheader("AIGC鉴伪证据")
    st.info("适用边界：模型敏感性图反映当前模型输出对输入区域的响应，不等于像素级伪造定位或司法鉴定结论。")
    if _result_status(data) != "success":
        st.error(_safe_text(data.get("error_message"), "AIGC模块未成功完成。"))
        return
    visual = _module_visual(data, "deepfake")
    details = _mapping(data.get("details"))
    metrics = _mapping(data.get("metrics"))
    st.markdown("#### 1. 查询图、人脸检测与模型输入")
    columns = st.columns(3)
    query = _query_path(data, policy)
    with columns[0]:
        if query is not None:
            render_image(query, "原始查询图", f"{key_prefix}_df_query")
        else:
            st.warning("原始查询图不可用。")
    with columns[1]:
        _render_evidence_image(
            _find_value(visual, "face_detection_image", "face_annotated_image", "annotated_face_image"),
            policy,
            caption="真实人脸框、关键点与序号",
            key=f"{key_prefix}_df_face_detection",
        )
    with columns[2]:
        crop_value = _find_value(visual, "face_crop_image", "selected_face_crop", "model_input_image")
        if crop_value is None:
            keyframes = details.get("keyframe_paths")
            crop_value = keyframes[0] if isinstance(keyframes, list) and keyframes else None
        _render_evidence_image(
            crop_value,
            policy,
            caption="实际送入模型的人脸裁剪",
            key=f"{key_prefix}_df_face_crop",
        )
    face_meta = _mapping(_find_value(visual, "face_detection", "face_metadata"))
    meta_columns = st.columns(4)
    meta_columns[0].metric("检测人脸数", face_meta.get("detected_face_count") if face_meta.get("detected_face_count") is not None else face_meta.get("face_count") or metrics.get("valid_frame_count") or details.get("valid_frame_count") or "—")
    meta_columns[1].metric("采用人脸", face_meta.get("selected_face_index") or "1（若检测成功）")
    crop_expand = face_meta.get("crop_expand_ratio")
    if crop_expand is None:
        crop_expand = face_meta.get("crop_expansion") or face_meta.get("expansion_ratio")
    meta_columns[2].metric("裁剪扩展", _safe_text(crop_expand))
    meta_columns[3].metric("模型输入尺寸", _safe_text(face_meta.get("model_input_size") or "224 × 224"))

    st.markdown("#### 2. 空间域模型敏感性")
    heat_columns = st.columns(2)
    with heat_columns[0]:
        _render_evidence_image(
            _find_value(visual, "spatial_heatmap", "heatmap_image", "sensitivity_heatmap"),
            policy,
            caption="空间域模型敏感性热力图",
            key=f"{key_prefix}_df_heatmap",
        )
    with heat_columns[1]:
        _render_evidence_image(
            _find_value(visual, "spatial_overlay", "heatmap_overlay", "sensitivity_overlay", "overlay_image"),
            policy,
            caption="空间域模型关注区域（叠加图）",
            key=f"{key_prefix}_df_overlay",
        )
    spatial_meta = _mapping(_find_value(visual, "spatial_explanation", "spatial_sensitivity", "spatial"))
    spatial_columns = st.columns(3)
    spatial_columns[0].metric("解释方法", _safe_text(spatial_meta.get("method") or _find_value(visual, "spatial_explanation_method")))
    spatial_columns[1].metric("空间域分数", _six(metrics.get("spatial_score") or details.get("spatial_score")))
    elapsed = _number(
        spatial_meta.get("generation_seconds")
        if spatial_meta.get("generation_seconds") is not None
        else spatial_meta.get("elapsed_seconds")
        or _find_value(visual, "spatial_explanation_elapsed_seconds")
    )
    spatial_columns[2].metric("解释图耗时", f"{elapsed:.3f} 秒" if elapsed is not None else "—")
    st.caption("该图仅展示模型敏感性区域，不表示已经确认的换脸区域。")

    st.markdown("#### 3. 频域证据")
    frequency_columns = st.columns(3)
    with frequency_columns[0]:
        _render_evidence_image(
            _find_value(visual, "frequency_input_image", "face_crop_image", "frequency_face_crop"),
            policy,
            caption="频域模块实际输入人脸",
            key=f"{key_prefix}_df_frequency_input",
        )
    with frequency_columns[1]:
        _render_evidence_image(
            _find_value(
                visual,
                "frequency_spectrum",
                "fft_log_spectrum",
                "centered_spectrum",
                "fft_log_magnitude_image",
                "fft_centered_spectrum_image",
            ),
            policy,
            caption="二维FFT中心化对数幅度谱",
            key=f"{key_prefix}_df_fft",
        )
    with frequency_columns[2]:
        _render_evidence_image(
            _find_value(visual, "frequency_profile", "radial_frequency_profile", "band_energy_chart", "radial_profile_image"),
            policy,
            caption="径向频谱/分频段能量",
            key=f"{key_prefix}_df_frequency_profile",
        )
    st.metric("频域模型分数", _six(metrics.get("frequency_score") or details.get("frequency_score")))
    st.markdown("**对当前 LogisticRegression 输出贡献绝对值最大的真实特征**")
    contribution_value = _find_value(
        visual, "frequency_contributions", "feature_contributions", "contributions"
    )
    if isinstance(contribution_value, Mapping):
        contribution_rows = _list(contribution_value.get("top_by_absolute_contribution"))
    else:
        contribution_rows = list(contribution_value) if isinstance(contribution_value, list) else []
    _render_signed_contributions(contribution_rows)

    st.markdown("#### 4. 融合与阈值判定")
    spatial = _number(metrics.get("spatial_score") or details.get("spatial_score"))
    frequency = _number(metrics.get("frequency_score") or details.get("frequency_score"))
    sw = _number(metrics.get("spatial_weight") or details.get("spatial_weight"))
    fw = _number(metrics.get("frequency_weight") or details.get("frequency_weight"))
    fused = _number(metrics.get("fusion_score") or details.get("fusion_score") or data.get("score"))
    threshold = _number(metrics.get("decision_threshold") or details.get("decision_threshold"))
    margin = _number(metrics.get("decision_margin"))
    score_columns = st.columns(5)
    score_columns[0].metric("空间域", _six(spatial))
    score_columns[1].metric("空间权重", _six(sw))
    score_columns[2].metric("频域", _six(frequency))
    score_columns[3].metric("频域权重", _six(fw))
    score_columns[4].metric("融合风险", _six(fused), _signed_six(margin), delta_color="off")
    if None not in (spatial, frequency, sw, fw, fused):
        st.code(
            f"{sw:.6f} × {spatial:.6f} + {fw:.6f} × {frequency:.6f} = {fused:.6f}",
            language=None,
        )
    _render_threshold_chart(fused, threshold)
    conclusion = _mapping(data.get("explanation")).get("decision") or data.get("label")
    st.warning(f"当前规范结论：{_safe_text(conclusion)}；与阈值差值 {_signed_six(margin)}。")


def _candidate_reference_values(candidates: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points: list[dict[str, Any]] = []
    means: list[dict[str, Any]] = []
    for candidate in candidates[:5]:
        if not isinstance(candidate, Mapping):
            continue
        identity = str(candidate.get("identity_name") or candidate.get("identity_id") or "unknown")
        mean = _number(candidate.get("mean_similarity") or candidate.get("score"))
        if mean is not None:
            means.append({"identity": identity, "mean": mean})
        best_path = str(candidate.get("best_reference_image") or "")
        best_file_name = str(candidate.get("best_reference_file_name") or Path(best_path).name)
        for index, raw in enumerate(_list(candidate.get("reference_scores")), 1):
            if not isinstance(raw, Mapping):
                continue
            score = _number(raw.get("similarity"))
            if score is None:
                continue
            path = str(raw.get("reference_image") or "")
            reference_file_name = str(raw.get("reference_file_name") or Path(path).name or f"参考图{index}")
            points.append(
                {
                    "identity": identity,
                    "reference": reference_file_name,
                    "similarity": score,
                    "is_best": path == best_path or reference_file_name == best_file_name,
                }
            )
    return points, means


def _render_identity_score_chart(candidates: Sequence[Any]) -> None:
    points, means = _candidate_reference_values(candidates)
    if not points:
        st.warning("本次任务未保存逐参考图真实相似度。")
        return
    spec = {
        "layer": [
            {
                "data": {"values": points},
                "mark": {"type": "circle", "size": 100, "opacity": 0.82},
                "encoding": {
                    "x": {"field": "similarity", "type": "quantitative", "title": "逐图余弦相似度", "scale": {"domain": [-1, 1]}},
                    "y": {"field": "identity", "type": "nominal", "sort": None, "title": "人员候选"},
                    "color": {"field": "is_best", "type": "nominal", "scale": {"domain": [True, False], "range": ["#b75d3e", "#315f86"]}, "title": "最佳单图"},
                    "tooltip": [
                        {"field": "identity", "title": "人员"},
                        {"field": "reference", "title": "参考图"},
                        {"field": "similarity", "type": "quantitative", "format": ".6f", "title": "余弦相似度"},
                    ],
                },
            },
            {
                "data": {"values": means},
                "mark": {"type": "tick", "thickness": 4, "size": 28, "color": "#d39b37"},
                "encoding": {
                    "x": {"field": "mean", "type": "quantitative"},
                    "y": {"field": "identity", "type": "nominal", "sort": None},
                    "tooltip": [{"field": "mean", "type": "quantitative", "format": ".6f", "title": "人员级平均"}],
                },
            },
        ],
        "height": max(260, 54 * min(5, len(candidates))),
    }
    st.vega_lite_chart(spec, width="stretch")
    st.caption("蓝/橙圆点为逐张参考图余弦相似度，金色刻线为人员级平均值；排序依据是人员级平均，不是单张图或身份概率。")


def _render_identity_page(data: Mapping[str, Any], policy: ResourcePathPolicy, key_prefix: str) -> None:
    st.subheader("人脸身份检索证据")
    st.info("适用边界：这是当前身份库内的闭集候选排序，余弦相似度不是身份概率，第一候选尚未确认。")
    if _result_status(data) != "success":
        st.error(_safe_text(data.get("error_message"), "身份模块未成功完成。"))
        return
    visual = _module_visual(data, "identity")
    metrics = _mapping(data.get("metrics"))
    candidates = _list(data.get("candidates"))
    st.markdown("#### 1. 人脸检测、关键点与对齐")
    image_columns = st.columns(4)
    query = _query_path(data, policy)
    with image_columns[0]:
        if query is not None:
            render_image(query, "查询原图", f"{key_prefix}_id_query")
        else:
            st.warning("查询原图不可用。")
    with image_columns[1]:
        _render_evidence_image(
            _find_value(visual, "face_detection_image", "query_face_annotated"),
            policy,
            caption="真实人脸框与五点关键点",
            key=f"{key_prefix}_id_detection",
        )
    with image_columns[2]:
        _render_evidence_image(
            _find_value(visual, "unaligned_face_image", "face_crop_before_alignment", "selected_face_crop"),
            policy,
            caption="对齐前查询人脸",
            key=f"{key_prefix}_id_before",
        )
    with image_columns[3]:
        _render_evidence_image(
            _find_value(visual, "aligned_face_image", "query_aligned_face"),
            policy,
            caption="对齐后标准人脸",
            key=f"{key_prefix}_id_aligned",
        )
    stat_columns = st.columns(4)
    stat_columns[0].metric("检测人脸数", metrics.get("query_face_count") if metrics.get("query_face_count") is not None else "—")
    stat_columns[1].metric("检测置信度", _six(metrics.get("face_detection_score")))
    stat_columns[2].metric("特征维度", metrics.get("embedding_dimension") or "—")
    stat_columns[3].metric("特征状态", "已提取" if candidates else "未生成候选")

    st.markdown("#### 2. Top-5 当前库内候选")
    top_score = _number(_mapping(candidates[0]).get("mean_similarity")) if candidates else None
    cards = st.columns(min(5, max(1, len(candidates))))
    for index, raw in enumerate(candidates[:5]):
        candidate = _mapping(raw)
        with cards[index]:
            with st.container(border=True):
                st.markdown(f"**第 {candidate.get('rank') or index + 1} 名 · {_safe_text(candidate.get('identity_name') or candidate.get('identity_id'))}**")
                image_value = (
                    candidate.get("candidate_card_image")
                    or candidate.get("best_reference_image")
                    or candidate.get("reference_image")
                )
                image_path = resolve_evidence_path(image_value, policy, kind="identity", label="身份候选参考图")
                if image_path is not None and image_path.is_file():
                    st.image(str(image_path), width="stretch")
                else:
                    st.caption("参考图不可用")
                mean = _number(candidate.get("mean_similarity") or candidate.get("score"))
                st.metric("人员级平均", _six(mean))
                st.metric("最佳单图", _six(candidate.get("best_reference_score")))
                valid_count = candidate.get("valid_reference_count")
                if valid_count is None:
                    valid_count = candidate.get("reference_count")
                st.caption(f"有效参考图 {valid_count if valid_count is not None else '—'} 张")
                st.caption(f"距第一候选 {_signed_six((mean - top_score) if mean is not None and top_score is not None else None)}")
                st.caption(f"最佳文件：{_safe_text(candidate.get('best_reference_file_name'), _basename(candidate.get('best_reference_image') or candidate.get('reference_image')))}")

    st.markdown("#### 3. 人员级分数形成过程")
    _render_identity_score_chart(candidates)
    skipped: list[Any] = []
    for candidate in candidates[:5]:
        if isinstance(candidate, Mapping):
            skipped.extend(_list(candidate.get("skipped_reference_images")))
    if not skipped:
        skipped = _find_list(visual, "skipped_reference_images", "skipped_references")
    if skipped:
        with st.expander("被跳过参考图及原因"):
            st.dataframe(skipped, width="stretch", hide_index=True)
    else:
        skipped_count = metrics.get("skipped_library_image_count")
        st.caption(f"跳过参考图数量：{skipped_count if skipped_count is not None else '本次未记录详细清单'}")

    st.markdown("#### 4. 查询人脸与第一候选最佳参考图")
    compare_columns = st.columns(2)
    with compare_columns[0]:
        aligned = _find_value(visual, "aligned_face_image", "query_aligned_face")
        _render_evidence_image(aligned, policy, caption="查询人脸（对齐后）", key=f"{key_prefix}_id_compare_query")
    top = _mapping(candidates[0]) if candidates else {}
    with compare_columns[1]:
        best_reference_visual = _mapping(visual.get("best_reference"))
        top_path = resolve_evidence_path(
            best_reference_visual.get("face_detection_image")
            or top.get("candidate_card_image")
            or top.get("best_reference_image")
            or top.get("reference_image"),
            policy,
            kind="identity",
            label="最佳参考图",
        )
        if top_path is not None and top_path.is_file():
            render_image(top_path, "第一候选最佳参考图", f"{key_prefix}_id_compare_reference")
        else:
            st.warning("第一候选最佳参考图不可用。")
    compare_stats = st.columns(3)
    compare_stats[0].metric("最佳单图余弦相似度", _six(top.get("best_reference_score")))
    compare_stats[1].metric("人员级平均余弦相似度", _six(top.get("mean_similarity") or top.get("score")))
    compare_stats[2].metric("Top1与Top2差值", _six(metrics.get("top1_top2_margin")))
    comparison_image = _find_value(visual, "best_reference_comparison_image")
    if comparison_image is not None:
        _render_evidence_image(
            comparison_image,
            policy,
            caption="查询人脸与第一候选最佳参考图证据对照",
            key=f"{key_prefix}_id_best_comparison",
            kind="identity",
        )


def _render_source_score_chart(candidates: Sequence[Any]) -> None:
    values: list[dict[str, Any]] = []
    for raw in candidates[:5]:
        candidate = _mapping(raw)
        name = f"#{candidate.get('rank')} {_basename(candidate.get('file_path'))}"
        for label, key in (("全局分", "global_score"), ("局部分", "local_score"), ("最终分", "final_score")):
            value = _number(candidate.get(key))
            if value is not None:
                values.append({"candidate": name, "score_type": label, "score": value})
    if not values:
        st.warning("本次任务没有可绘制的底图候选分数。")
        return
    spec = {
        "data": {"values": values},
        "mark": {"type": "bar", "cornerRadiusEnd": 3},
        "encoding": {
            "x": {"field": "candidate", "type": "nominal", "title": "候选"},
            "y": {"field": "score", "type": "quantitative", "title": "检索分数"},
            "xOffset": {"field": "score_type"},
            "color": {"field": "score_type", "type": "nominal", "scale": {"range": ["#617a8a", "#b0834c", "#315f86"]}},
            "tooltip": [
                {"field": "candidate", "title": "候选"},
                {"field": "score_type", "title": "分数类型"},
                {"field": "score", "type": "quantitative", "format": ".6f", "title": "数值"},
            ],
        },
        "height": 320,
    }
    st.vega_lite_chart(spec, width="stretch")


def _render_global_components(candidates: Sequence[Any], visual: Mapping[str, Any]) -> None:
    values: list[dict[str, Any]] = []
    visual_candidates = _find_list(visual, "top_candidates", "candidates")
    global_scoring = _mapping(visual.get("global_scoring"))
    scoring_candidates = _list(global_scoring.get("candidates"))
    by_file: dict[str, Mapping[str, Any]] = {}
    for raw in visual_candidates:
        if isinstance(raw, Mapping):
            by_file[_basename(raw.get("file_path") or raw.get("candidate_path") or raw.get("file_name"))] = raw
    for raw in scoring_candidates:
        if isinstance(raw, Mapping):
            by_file[_basename(raw.get("file_path") or raw.get("candidate_path") or raw.get("file_name"))] = raw
    for raw in candidates[:5]:
        candidate = _mapping(raw)
        name = _basename(candidate.get("file_path"))
        components = candidate.get("global_score_components") or candidate.get("global_components")
        if not isinstance(components, Mapping):
            visual_candidate = by_file.get(name, {})
            components = (
                visual_candidate.get("components") or visual_candidate.get("global_components")
                if isinstance(visual_candidate, Mapping)
                else None
            )
        if not isinstance(components, Mapping):
            continue
        for component, value in components.items():
            if component in {"weights", "weighted_terms", "formula", "score", "border_common_pixel_count"}:
                continue
            score = _number(value)
            if score is not None:
                values.append({"candidate": name, "component": str(component), "score": score})
    if not values:
        st.info("现有任务只保存了最终全局分，本次未生成可严格追溯的全局子分量；不会为了绘图凭空拆分。")
        return
    spec = {
        "data": {"values": values},
        "mark": {"type": "bar"},
        "encoding": {
            "x": {"field": "candidate", "type": "nominal", "title": "候选文件"},
            "y": {"field": "score", "type": "quantitative", "title": "实际子分量"},
            "xOffset": {"field": "component"},
            "color": {"field": "component", "type": "nominal", "title": "全局特征分量"},
            "tooltip": [
                {"field": "candidate", "title": "候选"},
                {"field": "component", "title": "分量"},
                {"field": "score", "type": "quantitative", "format": ".6f", "title": "分数"},
            ],
        },
        "height": 300,
    }
    st.vega_lite_chart(spec, width="stretch")


def _render_source_page(data: Mapping[str, Any], policy: ResourcePathPolicy, key_prefix: str) -> None:
    st.subheader("原始底图溯源证据")
    st.info("适用边界：结果是当前底图库内的候选排序，检索相似度不是来源概率，不能单独确认原始底图。")
    if _result_status(data) != "success":
        st.error(_safe_text(data.get("error_message"), "底图模块未成功完成。"))
        return
    metrics = _mapping(data.get("metrics"))
    visual = _module_visual(data, "source")
    candidates = _list(data.get("candidates"))
    st.markdown("#### 1. 查询图与真实检索输入")
    query_columns = st.columns(3)
    query = _query_path(data, policy)
    with query_columns[0]:
        if query is not None:
            render_image(query, "原始查询图", f"{key_prefix}_src_query")
        else:
            st.warning("查询原图不可用。")
    with query_columns[1]:
        _render_evidence_image(
            _find_value(visual, "face_detection_image", "query_face_annotated"),
            policy,
            caption="查询图真实人脸检测框",
            key=f"{key_prefix}_src_face_detection",
        )
    with query_columns[2]:
        masked = _find_value(visual, "face_masked_image", "masked_query_image") or metrics.get("masked_query_path")
        _render_evidence_image(masked, policy, caption="后端实际使用的人脸屏蔽图", key=f"{key_prefix}_src_masked")
    st.caption(
        f"是否应用屏蔽：{'是' if metrics.get('face_mask_applied') is True else '否' if metrics.get('face_mask_applied') is False else '未记录'}。"
        "屏蔽用于降低查询人脸身份对背景/底图检索的干扰，不改变底图库候选内容。"
    )

    st.markdown("#### 2. 全库粗排漏斗")
    total = _integer(metrics.get("library_file_count"))
    decoded = _integer(metrics.get("library_decoded"))
    skipped = _integer(metrics.get("library_skipped_count"))
    shortlist = _integer(metrics.get("shortlist_size"))
    displayed = min(5, len(candidates))
    funnel = st.columns(5)
    labels = [
        ("底图库文件", total),
        ("成功读取", decoded),
        ("跳过文件", skipped),
        ("进入局部复核", shortlist),
        ("最终展示", displayed),
    ]
    for index, (label, value) in enumerate(labels):
        funnel[index].metric(label, value if value is not None else "—")
        if index < 4:
            funnel[index].caption("→ 下一真实处理阶段")

    st.markdown("#### 3. 全局特征组成与候选对比")
    _render_global_components(candidates, visual)
    _render_source_score_chart(candidates)

    st.markdown("#### 4. Top-5 当前库内候选")
    cards = st.columns(min(5, max(1, len(candidates))))
    top_final = _number(_mapping(candidates[0]).get("final_score")) if candidates else None
    visual_top_candidates = {
        int(item.get("rank")): item
        for item in _find_list(visual, "top_candidates")
        if isinstance(item, Mapping) and isinstance(item.get("rank"), int)
    }
    for index, raw in enumerate(candidates[:5]):
        candidate = _mapping(raw)
        with cards[index]:
            with st.container(border=True):
                st.markdown(f"**第 {candidate.get('rank') or index + 1} 名 · {_basename(candidate.get('file_path'))}**")
                visual_candidate = _mapping(visual_top_candidates.get(int(candidate.get("rank") or index + 1)))
                candidate_image = (
                    candidate.get("candidate_image_artifact")
                    or visual_candidate.get("candidate_image")
                    or candidate.get("file_path")
                )
                path = resolve_evidence_path(candidate_image, policy, kind="source", label="底图候选")
                if path is not None and path.is_file():
                    st.image(str(path), width="stretch")
                else:
                    st.caption("候选图不可用")
                st.metric("最终分", _six(candidate.get("final_score")))
                st.caption(f"全局 {_six(candidate.get('global_score'))} · 局部 {_six(candidate.get('local_score'))}")
                st.caption(f"Lowe有效匹配 {candidate.get('good_match_count') or 0} 点")
                st.caption(f"RANSAC {candidate.get('ransac_inlier_count') or 0} 点 / {_six(candidate.get('ransac_inlier_ratio'))}")
                final = _number(candidate.get("final_score"))
                st.caption(f"距第一候选 {_signed_six((final - top_final) if final is not None and top_final is not None else None)}")

    st.markdown("#### 5. SIFT、Lowe与RANSAC几何证据（第一候选）")
    match_columns = st.columns(2)
    top = _mapping(candidates[0]) if candidates else {}
    with match_columns[0]:
        _render_evidence_image(
            _find_value(visual, "query_keypoints", "query_keypoints_image"),
            policy,
            caption="查询图SIFT关键点",
            key=f"{key_prefix}_src_query_keypoints",
        )
    with match_columns[1]:
        _render_evidence_image(
            _find_value(visual, "candidate_keypoints", "candidate_keypoints_image"),
            policy,
            caption="第一候选SIFT关键点",
            key=f"{key_prefix}_src_candidate_keypoints",
        )
    match_columns = st.columns(2)
    with match_columns[0]:
        all_matches = _find_value(visual, "matches_all", "matches_lowe", "lowe_matches_image") or top.get("visualization_path")
        _render_evidence_image(all_matches, policy, caption="Lowe筛选后的全部有效匹配线", key=f"{key_prefix}_src_matches_all")
    with match_columns[1]:
        inlier_matches = _find_value(
            visual,
            "matches_inliers_outliers",
            "ransac_inliers_outliers_image",
            "matches_inliers",
            "ransac_inliers_image",
            "match_inliers",
        ) or top.get("visualization_path")
        _render_evidence_image(inlier_matches, policy, caption="RANSAC内点（绿）与外点（红/灰）", key=f"{key_prefix}_src_matches_inliers")
    homography = _find_value(visual, "homography_overlay", "homography_projection")
    if homography is not None:
        _render_evidence_image(homography, policy, caption="单应性投影/几何对齐叠加", key=f"{key_prefix}_src_homography")
    local_components = _mapping(top.get("local_score_components"))
    local_stats = st.columns(4)
    local_stats[0].metric("Lowe有效匹配", top.get("good_match_count") or "—")
    local_stats[1].metric("RANSAC内点", top.get("ransac_inlier_count") or "—")
    local_stats[2].metric("RANSAC内点率", _six(top.get("ransac_inlier_ratio")))
    local_stats[3].metric("内点数归一化", _six(local_components.get("normalized_inlier_count")))
    st.code(_safe_text(local_components.get("formula"), "本次未记录局部分公式"), language=None)

    st.markdown("#### 6. 最终分数形成")
    global_score = _number(top.get("global_score"))
    local_score = _number(top.get("local_score"))
    gw = _number(top.get("global_weight") or metrics.get("global_weight"))
    lw = _number(top.get("local_weight") or metrics.get("local_weight"))
    final_score = _number(top.get("final_score"))
    final_columns = st.columns(5)
    final_columns[0].metric("全局分", _six(global_score))
    final_columns[1].metric("全局权重", _six(gw))
    final_columns[2].metric("局部分", _six(local_score))
    final_columns[3].metric("局部权重", _six(lw))
    final_columns[4].metric("最终分", _six(final_score))
    if None not in (global_score, local_score, gw, lw, final_score):
        st.code(f"{gw:.6f} × {global_score:.6f} + {lw:.6f} × {local_score:.6f} = {final_score:.6f}", language=None)
    second = _mapping(candidates[1]) if len(candidates) > 1 else {}
    st.caption(f"第二候选最终分 {_six(second.get('final_score'))}；第一名与第二名差值 {_six(metrics.get('top1_top2_margin'))}。")


def _render_evidence_chain_page(
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    task_dir: Path,
    task_status: Mapping[str, Any] | None,
    policy: ResourcePathPolicy,
    task_artifact_manifest: Mapping[str, Any] | None = None,
) -> None:
    st.subheader("证据链与完整报告")
    st.info("本页只汇总当前任务实际结果、配置、模型/输入哈希、进度事件和已落盘证据；不生成跨模块综合概率。")
    authoritative = _mapping(task_artifact_manifest)
    authoritative_records = authoritative.get("artifacts")
    manifest = (
        [dict(item) for item in authoritative_records if isinstance(item, Mapping)]
        if isinstance(authoritative_records, list)
        else collect_artifact_manifest(deepfake, identity, source)
    )
    st.markdown("#### 任务级 artifact manifest")
    if manifest:
        rows: list[dict[str, Any]] = []
        for item in manifest:
            path_value = _path_value(item.get("path") or item.get("relative_path")) or ""
            path = resolve_evidence_path(path_value, policy, label=f"artifact {item.get('id')}") if path_value else None
            rows.append(
                {
                    "模块": item.get("module"),
                    "证据类型": item.get("kind") or item.get("type"),
                    "相对路径": path_value,
                    "状态": item.get("status") or ("present" if path and path.is_file() else "missing"),
                    "大小(字节)": item.get("size_bytes") or (path.stat().st_size if path and path.is_file() else None),
                    "SHA-256": item.get("sha256"),
                    "说明": item.get("description"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.warning("该任务是旧结构或本次未生成 artifact manifest；原结果仍可读取，不会因此崩溃。")

    st.markdown("#### 输入、模型与配置溯源")
    provenance_rows: list[dict[str, Any]] = []
    for module, result in (("AIGC", deepfake), ("身份", identity), ("底图", source)):
        provenance = _mapping(result.get("provenance"))
        for key, value in provenance.items():
            if "path" in key.lower() and isinstance(value, str):
                value = _basename(value)
            provenance_rows.append({"模块": module, "字段": key, "值": _table_value(value)})
    if provenance_rows:
        st.dataframe(provenance_rows, width="stretch", hide_index=True)

    st.markdown("#### 实际运行事件时间线")
    events = load_progress_events(task_dir)
    if events:
        rows = []
        for event in events[-300:]:
            rows.append(
                {
                    "时间": event.get("timestamp"),
                    "模块": event.get("module"),
                    "阶段": event.get("stage"),
                    "状态": event.get("status"),
                    "真实进度": event.get("progress_percent"),
                    "耗时(秒)": event.get("elapsed_seconds"),
                    "说明": event.get("message"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True, height=420)
    else:
        st.caption("该任务没有保存可读进度事件。")
    status = _mapping(task_status)
    if status:
        with st.expander("任务状态与三个模块原始状态"):
            st.json(status)


def render_evidence_workspace(
    *,
    deepfake: Mapping[str, Any],
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    policy: ResourcePathPolicy,
    task_dir: Path,
    task_status: Mapping[str, Any] | None = None,
    final_report: Mapping[str, Any] | None = None,
    task_artifact_manifest: Mapping[str, Any] | None = None,
    product_version: str = "0.4.0",
    key_prefix: str = "evidence",
) -> str:
    request_key = f"{key_prefix}_page_request"
    navigation_key = f"{key_prefix}_page"
    requested = st.session_state.pop(request_key, None)
    if requested in PAGES:
        st.session_state[navigation_key] = requested
    if navigation_key not in st.session_state:
        st.session_state[navigation_key] = PAGES[0]
    selected = st.radio(
        "研判页面",
        PAGES,
        key=navigation_key,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()
    if selected == "研判总览":
        overview = build_task_overview(
            deepfake,
            identity,
            source,
            task_status=task_status,
            final_report=final_report,
            task_dir=task_dir,
            product_version=product_version,
        )
        _render_task_header(overview)
        st.warning("三个模块解决不同问题；本页并列呈现证据，不相加、不生成所谓综合可信概率。")
        _render_overview_cards(
            deepfake,
            identity,
            source,
            request_key=request_key,
            key_prefix=key_prefix,
        )
    elif selected == "AIGC鉴伪证据":
        _render_deepfake_page(deepfake, policy, key_prefix)
    elif selected == "人脸身份检索":
        _render_identity_page(identity, policy, key_prefix)
    elif selected == "原始底图溯源":
        _render_source_page(source, policy, key_prefix)
    else:
        _render_evidence_chain_page(
            deepfake,
            identity,
            source,
            task_dir=task_dir,
            task_status=task_status,
            policy=policy,
            task_artifact_manifest=task_artifact_manifest,
        )
    return str(selected)


__all__ = [
    "PAGES",
    "build_task_overview",
    "collect_artifact_manifest",
    "render_evidence_workspace",
    "resolve_evidence_path",
]
