from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


class StreamlitUIError(RuntimeError):
    """可直接展示给界面用户的安全输入或工件错误。"""


def load_ui_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StreamlitUIError(f"界面配置文件不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StreamlitUIError(f"界面配置损坏：{path}") from exc
    required = {
        "application_name",
        "application_version",
        "disagreement_threshold",
        "upload_extensions",
        "image_extensions",
        "video_extensions",
        "upload_temp_directory",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise StreamlitUIError(f"界面配置缺少字段：{missing}")
    threshold = float(payload["disagreement_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise StreamlitUIError("模型分歧阈值必须位于0到1之间。")
    return payload


def safe_download_stem(filename: str) -> str:
    stem = Path(str(filename)).stem
    safe = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", stem).strip("._")
    return safe[:80] or "media"


def save_uploaded_file(
    filename: str,
    data: bytes,
    project_root: Path,
    temp_relative_root: str,
    allowed_extensions: list[str],
) -> tuple[Path, Path]:
    suffix = Path(str(filename)).suffix.lower().lstrip(".")
    normalized_allowed = {str(item).lower().lstrip(".") for item in allowed_extensions}
    if suffix not in normalized_allowed:
        raise StreamlitUIError(f"不支持的上传格式：.{suffix or '无扩展名'}")
    if not data:
        raise StreamlitUIError("上传文件为空，无法检测。")
    root = project_root.resolve()
    temp_root = (root / temp_relative_root).resolve()
    try:
        temp_root.relative_to(root)
    except ValueError as exc:
        raise StreamlitUIError("上传临时目录必须位于项目目录内。") from exc
    upload_dir = temp_root / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=False)
    target = upload_dir / f"uploaded.{suffix}"
    try:
        target.write_bytes(data)
    except OSError as exc:
        cleanup_upload_directory(upload_dir, root)
        raise StreamlitUIError(f"上传文件写入失败：{exc}") from exc
    return target, upload_dir


def cleanup_upload_directory(upload_dir: Path, project_root: Path) -> None:
    resolved = upload_dir.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise StreamlitUIError(f"拒绝清理项目目录之外的路径：{resolved}") from exc
    if resolved.exists():
        shutil.rmtree(resolved)


def score_display(value: Any) -> str:
    if value is None or value == "":
        return "不可用"
    score = float(value)
    return f"{score:.4f}（{score * 100:.2f}%）"


def status_presentation(result: dict[str, Any]) -> dict[str, Any]:
    status = str(result.get("status", "ERROR"))
    label = str(result.get("final_label", "证据不足"))
    if status == "SUCCESS" and label == "REAL":
        return {
            "kind": "success",
            "title": "未发现明显深度伪造风险",
            "detail": "模型结果仍需结合文件来源、传播链路和其他证据核验。",
            "show_score": True,
        }
    if status == "SUCCESS" and label == "FAKE":
        return {
            "kind": "error",
            "title": "检测到较高深度伪造风险",
            "detail": "建议重点开展人工复核，并结合原始文件与传播来源综合研判。",
            "show_score": True,
        }
    if status == "LOW_CONFIDENCE":
        return {
            "kind": "warning",
            "title": "有效人脸帧不足，当前结果可信度较低",
            "detail": "仅展示有限分析结果，建议上传更清晰或更长的视频。",
            "show_score": result.get("final_fake_score") is not None,
        }
    if status == "INSUFFICIENT_EVIDENCE":
        return {
            "kind": "warning",
            "title": "未检测到满足质量要求的人脸，无法形成可靠判断",
            "detail": "请更换包含清晰正面人脸的素材，或转由人工核查。",
            "show_score": False,
        }
    return {
        "kind": "error",
        "title": "检测未能完成",
        "detail": str(result.get("error_message") or "请重新上传文件后重试。"),
        "show_score": False,
    }


def _valid_frames(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in result.get("frame_results", []) if row.get("status") == "VALID"]


def keyframe_details(result: dict[str, Any]) -> list[dict[str, Any]]:
    paths = list(result.get("keyframe_paths", []))
    ranked = sorted(_valid_frames(result), key=lambda row: float(row["fusion_score"]), reverse=True)
    return [
        {"relative_path": relative_path, **frame}
        for relative_path, frame in zip(paths, ranked, strict=False)
    ]


def build_frame_dataframe(result: dict[str, Any]) -> pd.DataFrame:
    keyframe_candidates = {int(item["candidate_index"]) for item in keyframe_details(result)}
    rows = []
    for frame in result.get("frame_results", []):
        candidate = int(frame.get("candidate_index", 0))
        rows.append(
            {
                "候选序号": candidate,
                "源帧序号": frame.get("source_frame_index"),
                "时间戳（秒）": frame.get("timestamp_seconds"),
                "状态": frame.get("status"),
                "空间域评分": frame.get("spatial_score"),
                "频域评分": frame.get("frequency_score"),
                "融合评分": frame.get("fusion_score"),
                "帧级阈值": frame.get("frame_threshold"),
                "帧级判断": frame.get("predicted_label", ""),
                "是否关键帧": "是" if candidate in keyframe_candidates else "否",
                "无效原因": frame.get("invalid_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def model_disagreement(result: dict[str, Any], threshold: float) -> dict[str, Any]:
    spatial = result.get("spatial_score_summary", {}).get("median")
    frequency = result.get("frequency_score_summary", {}).get("median")
    if spatial is None or frequency is None:
        return {"is_disagreement": False, "absolute_difference": None, "threshold": float(threshold)}
    difference = abs(float(spatial) - float(frequency))
    return {
        "is_disagreement": difference > float(threshold),
        "absolute_difference": difference,
        "threshold": float(threshold),
    }


def download_artifacts(result: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = Path(str(result.get("run_directory", ""))).resolve()
    mapping = [
        ("HTML辅助研判报告", "report.html", "text/html", "report.html"),
        ("JSON结构化结果", "result.json", "application/json", "result.json"),
        ("CSV逐帧结果", "frame_predictions.csv", "text/csv", "frames.csv"),
    ]
    artifacts = []
    for label, filename, mime, output_suffix in mapping:
        path = run_dir / filename
        artifacts.append(
            {
                "label": label,
                "path": path,
                "exists": path.is_file(),
                "mime": mime,
                "output_suffix": output_suffix,
            }
        )
    return artifacts
