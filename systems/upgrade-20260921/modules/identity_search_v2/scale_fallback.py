# -*- coding: utf-8 -*-
"""身份检索的检测尺度回退策略（CLI worker 与常驻 worker 共用）。

证据见 docs/M2_人脸预处理_尺度回退实验.md：
冻结 worker 固定 det_size=(1280,1280)，InsightFace 会把整幅图直接缩放到该尺寸，
竖屏 720x1280 视频帧因此被横向拉伸 1.78 倍，人脸在检测网络里过大而漏检。

本模块只做两件事：

1. 把检测输入尺度切换到指定档位（复用同一个 FaceAnalysis 实例，不重复加载模型）；
2. 按尺度链依次尝试查询，只对「一个尺度都没检出人脸」触发下一档。

检索算法、库索引、特征口径、单脸约束都不在这里改动。
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from identity_search.errors import InputValidationError, NoFaceError


DEFAULT_SCALE_CHAIN = (1280, 640, 320)
FALLBACK_POLICY = "scale_fallback_on_no_face"


def parse_scale_chain(text: str) -> tuple[int, ...]:
    """解析形如 "1280,640,320" 的检测尺度链，去重并保持顺序。"""

    scales: list[int] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise InputValidationError(f"检测尺度必须是整数: {token}") from exc
        if value < 64:
            raise InputValidationError(f"检测尺度不得小于 64: {value}")
        if value not in scales:
            scales.append(value)
    if not scales:
        raise InputValidationError("检测尺度链不得为空")
    return tuple(scales)


def apply_detection_scale(extractor: Any, scale: int, ctx_id: int) -> Any:
    """切换检测输入尺度；复用已有 FaceAnalysis 实例，不重新加载模型。"""

    app = getattr(extractor, "app", None)
    if app is None:
        raise InputValidationError("特征提取器没有可切换的检测模型实例。")
    app.prepare(ctx_id=ctx_id, det_size=(scale, scale))
    if hasattr(extractor, "det_size"):
        extractor.det_size = (scale, scale)
    return extractor


def attempt_scale_chain(
    searcher: Any,
    query: str,
    scales: Sequence[int],
    extractor_factory: Callable[[int], Any],
    *,
    top_k: int,
    progress_reporter: Any = None,
    emit_progress: Callable[..., None] | None = None,
    **search_kwargs: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """按尺度链依次尝试；返回 (成功结果或 None, 逐档记录)。

    与冻结实现的差异只有一处：某一档「一个人脸都检不到」时继续下一档。
    其他错误（多人脸、解码失败等）保持冻结行为，立即上抛，不做重试。
    """

    def _emit(stage: str, status: str, message: str, **fields: Any) -> None:
        if emit_progress is not None:
            emit_progress(progress_reporter, stage, status, message, **fields)

    attempts: list[dict[str, Any]] = []
    for index, scale in enumerate(scales):
        entry: dict[str, Any] = {"scale": scale, "tier": index}
        # 只替换查询侧检测尺度；库索引已按第一档建好，缓存键不受影响。
        searcher.extractor = extractor_factory(scale)
        _emit(
            "identity_detection_scale",
            "running",
            f"正在按检测尺度 {scale} 提取查询图人脸。",
            progress_percent=None,
            details={"scale": scale, "tier": index, "chain": list(scales)},
            force=True,
        )
        try:
            result = searcher.search(
                query, top_k=top_k, progress_reporter=progress_reporter, **search_kwargs
            )
        except NoFaceError as exc:
            diagnostics = getattr(exc, "diagnostics", {}) or {}
            entry.update(
                {
                    "status": "error",
                    "error_type": "NoFaceError",
                    "face_count": diagnostics.get("face_count"),
                }
            )
            attempts.append(entry)
            _emit(
                "identity_scale_fallback",
                "warning",
                f"检测尺度 {scale} 未检出人脸。",
                progress_percent=None,
                details={"scale": scale, "tier": index, "next_tier": index + 1},
                force=True,
            )
            continue
        entry.update({"status": "success", "face_count": 1})
        attempts.append(entry)
        return result, attempts
    return None, attempts


def annotate_result(
    result: dict[str, Any], scales: Sequence[int], attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    """把尺度链与实际生效档位写入结果 JSON，供人工复核。"""

    used = attempts[-1]["scale"] if attempts else scales[0]
    method = result.get("method")
    if not isinstance(method, dict):
        method = {}
        result["method"] = method
    method["preprocessing"] = {
        "policy": FALLBACK_POLICY,
        "det_size_chain": list(scales),
        "library_det_size": scales[0],
        "detection_scale_used": used,
        "fallback_applied": used != scales[0],
        "attempts": attempts,
    }
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        metrics["detection_scale_used"] = used
        metrics["detection_attempt_count"] = len(attempts)
    return result


__all__ = [
    "DEFAULT_SCALE_CHAIN",
    "FALLBACK_POLICY",
    "annotate_result",
    "apply_detection_scale",
    "attempt_scale_chain",
    "parse_scale_chain",
]
