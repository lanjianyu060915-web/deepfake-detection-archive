from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch

from src.core.paths import resolve_resource_root, resolve_user_data_root
from src.pipeline import DeepfakeDetectionPipeline
from src.reporting import DISCLAIMER
from src.ui import (
    build_frame_dataframe,
    cleanup_upload_directory,
    download_artifacts,
    keyframe_details,
    load_ui_config,
    model_disagreement,
    safe_download_stem,
    save_uploaded_file,
    score_display,
    status_presentation,
)
from src.ui.streamlit_helpers import StreamlitUIError


ROOT = resolve_resource_root(Path(__file__).resolve().parent)
USER_DATA_ROOT = resolve_user_data_root(ROOT)
UI_CONFIG = load_ui_config(ROOT / "config" / "streamlit_ui.json")
FUSION_CONFIG = json.loads(
    (ROOT / "models" / "fusion_formal" / "fusion_config.json").read_text(encoding="utf-8")
)
RUNTIME_CONFIG = json.loads(
    (ROOT / "config" / "runtime_pipeline.json").read_text(encoding="utf-8")
)
OUTPUT_DIR = USER_DATA_ROOT / "outputs" / "streamlit_formal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_OUTPUT_DIR = USER_DATA_ROOT / "outputs" / "runtime"
RUNTIME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = USER_DATA_ROOT / "logs" if USER_DATA_ROOT != ROOT else OUTPUT_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_TEMP_DIRECTORY = (
    "runtime/streamlit_uploads"
    if USER_DATA_ROOT != ROOT
    else str(UI_CONFIG["upload_temp_directory"])
)
LOGGER = logging.getLogger("streamlit_formal")
if not LOGGER.handlers:
    handler = logging.FileHandler(LOG_DIR / "streamlit_ui.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


@st.cache_resource(show_spinner=False)
def load_pipeline(device: str) -> DeepfakeDetectionPipeline:
    """每种设备模式只构造一个管线实例，模型由管线延迟加载并跨帧复用。"""
    return DeepfakeDetectionPipeline(
        project_root=ROOT,
        device=device,
        output_root=RUNTIME_OUTPUT_DIR,
        writable_root=USER_DATA_ROOT,
    )


def clear_detection_state() -> None:
    st.session_state.pop("detection_result", None)
    st.session_state.pop("detection_source_name", None)


def _configure_page() -> None:
    st.set_page_config(
        page_title=UI_CONFIG["application_name"],
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
<style>
.block-container{padding-top:1.6rem;padding-bottom:2.5rem;max-width:1280px}
.system-subtitle{color:#526777;font-size:1.05rem;margin-top:-0.6rem;margin-bottom:1.4rem}
.evidence-note{border-left:4px solid #3b6f8f;background:#f2f7fa;padding:0.8rem 1rem;margin:0.7rem 0}
.disclaimer{border:1px solid #d5a94f;background:#fff8e8;padding:1rem;border-radius:0.4rem}
</style>
""",
        unsafe_allow_html=True,
    )


def _sidebar() -> tuple[str, int, int, bool]:
    st.sidebar.header("运行设置")
    device_label = st.sidebar.selectbox(
        "运行设备",
        ["自动选择", "CUDA GPU", "CPU"],
        index=0,
        help="自动选择会在CUDA可用时使用GPU，否则使用CPU。",
    )
    device_map = {"自动选择": "auto", "CUDA GPU": "cuda", "CPU": "cpu"}
    device = device_map[device_label]
    cuda_unavailable = device == "cuda" and not torch.cuda.is_available()
    if cuda_unavailable:
        st.sidebar.error("当前PyTorch无法使用CUDA。请选择“自动选择”或“CPU”，并检查CUDA环境。")
    if device in {"auto", "cuda"} and torch.cuda.is_available():
        actual_hint = f"CUDA · {torch.cuda.get_device_name(0)}"
    else:
        actual_hint = "CPU"
    st.sidebar.caption(f"预计实际设备：{actual_hint}")

    st.sidebar.subheader("视频抽帧设置")
    formal_target = int(RUNTIME_CONFIG["target_valid_frames"])
    formal_minimum = int(RUNTIME_CONFIG["min_valid_frames"])
    target = st.sidebar.slider(
        "目标有效帧",
        min_value=3,
        max_value=10,
        value=formal_target,
    )
    default_minimum = min(formal_minimum, target)
    minimum = st.sidebar.slider(
        "最低有效帧",
        min_value=1,
        max_value=target,
        value=default_minimum,
    )
    if target != formal_target or minimum != formal_minimum:
        st.sidebar.warning(
            f"正式实验默认参数为目标{formal_target}帧、最低{formal_minimum}帧。"
            "修改参数可能使运行结果与正式实验口径不同。"
        )
    else:
        st.sidebar.caption(
            f"当前使用正式实验默认口径：目标{formal_target}帧、最低{formal_minimum}帧。"
        )

    st.sidebar.divider()
    st.sidebar.subheader("系统信息")
    st.sidebar.markdown(
        f"""
- 空间模型：MobileNetV3-Small GPU微调版
- 频域模型：FrequencyStatDetector
- 融合方式：加权概率融合
- 视频聚合：{FUSION_CONFIG['video_aggregation']}
- 正式版本：v{UI_CONFIG['application_version']}
- 空间/频域权重：{FUSION_CONFIG['spatial_weight']:.2f} / {FUSION_CONFIG['frequency_weight']:.2f}
"""
    )
    st.sidebar.info("系统仅用于深度伪造风险筛查和辅助研判。")
    return device, target, minimum, cuda_unavailable


def _render_uploaded_preview(uploaded_file: Any) -> None:
    extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
    left, right = st.columns([1, 2])
    with left:
        st.markdown("#### 文件信息")
        st.write(f"**原始文件名：** {uploaded_file.name}")
        st.write(f"**文件大小：** {uploaded_file.size / 1024:.2f} KiB")
        media_label = "图片" if extension in UI_CONFIG["image_extensions"] else "视频"
        st.write(f"**文件类型：** {media_label}（.{extension}）")
    with right:
        st.markdown("#### 上传预览")
        if extension in UI_CONFIG["image_extensions"]:
            try:
                st.image(uploaded_file, caption="原始上传图片", width="stretch")
            except Exception:
                LOGGER.exception("上传图片预览失败")
                st.warning("图片预览失败，文件可能已损坏；仍可点击检测获取正式错误报告。")
        else:
            st.video(uploaded_file.getvalue())


def _emit_status(presentation: dict[str, Any]) -> None:
    message = f"**{presentation['title']}**\n\n{presentation['detail']}"
    if presentation["kind"] == "success":
        st.success(message)
    elif presentation["kind"] == "warning":
        st.warning(message)
    else:
        st.error(message)


def _render_core_metrics(result: dict[str, Any], show_score: bool) -> None:
    columns = st.columns(5)
    columns[0].metric("最终判断", result.get("final_label", "证据不足"))
    if show_score:
        columns[1].metric("融合伪造风险评分", score_display(result.get("final_fake_score")))
    else:
        columns[1].metric("融合伪造风险评分", "不可用")
    threshold = result.get("decision_threshold")
    columns[2].metric("判定阈值", "不可用" if threshold is None else f"{float(threshold):.4f}")
    columns[3].metric("有效帧数量", str(result.get("valid_frame_count", 0)))
    columns[4].metric("总耗时", f"{float(result.get('total_elapsed_seconds', 0.0)):.3f} 秒")

    spatial = result.get("spatial_score_summary", {}).get("median")
    frequency = result.get("frequency_score_summary", {}).get("median")
    fusion = result.get("fusion_score_summary", {}).get("median")
    if any(value is not None for value in (spatial, frequency, fusion)):
        st.markdown("#### 模型评分")
        score_columns = st.columns(3)
        score_columns[0].metric("空间域评分（中位数）", score_display(spatial))
        score_columns[1].metric("频域评分（中位数）", score_display(frequency))
        score_columns[2].metric("逐帧融合评分（中位数）", score_display(fusion))
        st.caption("上述百分比为模型伪造风险评分，不等于系统总体准确率。")


def _render_media_metadata(result: dict[str, Any]) -> None:
    metadata = result.get("media_metadata", {})
    with st.expander("文件与运行信息", expanded=True):
        st.code(f"SHA-256: {result.get('input_sha256', '')}", language=None)
        if result.get("media_type") == "video":
            cols = st.columns(5)
            cols[0].metric("时长", f"{float(metadata.get('duration_seconds', 0.0)):.3f}s")
            cols[1].metric("FPS", f"{float(metadata.get('fps', 0.0)):.3f}")
            cols[2].metric("分辨率", f"{metadata.get('width', 0)}×{metadata.get('height', 0)}")
            cols[3].metric("总帧数", str(metadata.get("total_frames", 0)))
            cols[4].metric("无效候选帧", str(result.get("invalid_frame_count", 0)))
        else:
            st.write(
                f"图片尺寸：{metadata.get('width', 0)}×{metadata.get('height', 0)}；"
                f"格式：{metadata.get('format', '未知')}"
            )
        st.write(f"实际运行设备：`{result.get('actual_device', 'unknown')}`")
        st.write(f"运行编号：`{result.get('run_id', '')}`")


def _render_keyframes(result: dict[str, Any]) -> None:
    details = keyframe_details(result)
    if not result.get("keyframe_paths"):
        if result.get("status") not in {"ERROR", "INSUFFICIENT_EVIDENCE"}:
            st.warning("后端未生成可展示的关键帧。")
        return
    title = "人脸裁剪结果" if result.get("media_type") == "image" else "视频关键帧（按融合评分排序）"
    st.markdown(f"#### {title}")
    columns = st.columns(min(3, len(result["keyframe_paths"])))
    run_dir = Path(str(result.get("run_directory", "")))
    for index, relative_path in enumerate(result["keyframe_paths"]):
        detail = details[index] if index < len(details) else {}
        path = run_dir / relative_path
        with columns[index % len(columns)]:
            if path.is_file():
                st.image(str(path), caption=f"关键帧 {index + 1}", width="stretch")
                if detail:
                    st.caption(
                        f"时间 {float(detail.get('timestamp_seconds', 0.0)):.3f}s · "
                        f"空间 {float(detail.get('spatial_score', 0.0)):.4f} · "
                        f"频域 {float(detail.get('frequency_score', 0.0)):.4f} · "
                        f"融合 {float(detail.get('fusion_score', 0.0)):.4f} · "
                        f"{detail.get('predicted_label', '')}"
                    )
            else:
                st.warning(f"关键帧文件不存在：{relative_path}")


def _render_frame_analysis(result: dict[str, Any]) -> None:
    frame = build_frame_dataframe(result)
    if frame.empty:
        return
    st.markdown("#### 逐帧分析")
    st.dataframe(frame, width="stretch", hide_index=True)
    valid = frame[frame["状态"] == "VALID"].copy()
    if result.get("media_type") == "video" and not valid.empty:
        chart = valid[
            ["时间戳（秒）", "空间域评分", "频域评分", "融合评分", "帧级阈值"]
        ].set_index("时间戳（秒）")
        st.markdown("##### 逐帧评分趋势")
        st.line_chart(chart)


def _render_downloads(result: dict[str, Any]) -> None:
    st.markdown("#### 报告下载")
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
    stem = safe_download_stem(result.get("input_file_name", "media"))
    run_id = str(result.get("run_id", "run"))
    columns = st.columns(3)
    for index, artifact in enumerate(download_artifacts(result)):
        with columns[index]:
            if artifact["exists"]:
                try:
                    data = artifact["path"].read_bytes()
                    st.download_button(
                        artifact["label"],
                        data=data,
                        file_name=f"{stem}_{run_id}_{artifact['output_suffix']}",
                        mime=artifact["mime"],
                        key=f"download_{run_id}_{index}",
                        width="stretch",
                    )
                except OSError as exc:
                    st.warning(f"无法读取{artifact['label']}：{exc}")
            else:
                st.warning(f"{artifact['label']}不存在，其他结果仍可查看。")


def render_result(result: dict[str, Any]) -> None:
    st.divider()
    st.header("检测结果")
    presentation = status_presentation(result)
    _emit_status(presentation)
    _render_core_metrics(result, bool(presentation["show_score"]))
    _render_media_metadata(result)
    disagreement = model_disagreement(result, float(UI_CONFIG["disagreement_threshold"]))
    if disagreement["is_disagreement"]:
        st.warning(
            f"两个检测分支存在明显分歧（评分绝对差 {disagreement['absolute_difference']:.4f} > "
            f"{disagreement['threshold']:.2f}），建议重点进行人工复核。"
        )
        st.caption("空间域主要分析纹理、边缘和局部视觉痕迹；频域主要分析频率分布和生成伪影，两者可能对不同伪造产生不同响应。")
    for warning in result.get("warnings", []):
        st.warning(str(warning))
    _render_keyframes(result)
    _render_frame_analysis(result)
    if result.get("media_type") == "image":
        st.info("单张图片仅包含有限视觉信息，模型可能无法覆盖视频时序中的更多伪造痕迹，建议结合多帧或原始视频进行研判。")
    _render_downloads(result)


def main() -> None:
    _configure_page()
    device, target_frames, min_frames, cuda_unavailable = _sidebar()
    st.title(UI_CONFIG["application_name"])
    st.markdown(
        '<div class="system-subtitle">基于空间域与频域特征融合的图片及视频伪造风险检测原型</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="evidence-note"><strong>系统定位：</strong>面向公安实战场景的多域融合深度伪造辅助研判原型系统。'
        "检测结果不是司法鉴定结论。</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "上传待检测图片或视频",
        type=UI_CONFIG["upload_extensions"],
        accept_multiple_files=False,
        on_change=clear_detection_state,
        key="media_upload",
        help="支持JPG、JPEG、PNG、WEBP、MP4、AVI、MOV和MKV。上传后不会自动启动模型。",
    )
    if uploaded_file is not None:
        _render_uploaded_preview(uploaded_file)
    else:
        st.info("请先上传一个图片或视频文件。尚未点击检测前，系统不会执行模型计算。")

    start = st.button(
        "开始检测",
        type="primary",
        disabled=uploaded_file is None or cuda_unavailable,
        width="stretch",
    )
    if start and uploaded_file is not None:
        upload_dir: Path | None = None
        try:
            upload_path, upload_dir = save_uploaded_file(
                uploaded_file.name,
                uploaded_file.getvalue(),
                USER_DATA_ROOT,
                UPLOAD_TEMP_DIRECTORY,
                UI_CONFIG["upload_extensions"],
            )
            with st.status("正在初始化或读取缓存检测管线……", expanded=True) as status:
                st.write("正在检查媒体文件与运行设备。")
                pipeline = load_pipeline(device)
                st.write("正式管线已就绪，正在完成抽帧、人脸检测、多域分析与报告生成。")
                result = pipeline.detect_media(
                    upload_path,
                    target_valid_frames=target_frames,
                    min_valid_frames=min_frames,
                )
                status.update(label="检测与报告生成完成", state="complete", expanded=False)
            st.session_state["detection_result"] = result
            st.session_state["detection_source_name"] = uploaded_file.name
            LOGGER.info("检测完成：run_id=%s status=%s device=%s", result.get("run_id"), result.get("status"), result.get("actual_device"))
        except Exception as exc:
            LOGGER.exception("Streamlit检测流程异常")
            st.error(f"检测未能完成：{exc}。请重新上传文件或切换设备后重试。")
        finally:
            if upload_dir is not None:
                try:
                    cleanup_upload_directory(upload_dir, USER_DATA_ROOT)
                except StreamlitUIError:
                    LOGGER.exception("上传临时目录清理失败")

    result = st.session_state.get("detection_result")
    if isinstance(result, dict):
        render_result(result)

    st.divider()
    st.subheader("适用场景与边界")
    st.markdown(
        "网络谣言视频初筛、诈骗视频线索核验、涉案音视频材料辅助检查、网络传播内容风险排查。"
        "当前系统是课程原型，未接入公安业务系统，也未用于实战部署。"
    )
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
