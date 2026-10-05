from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .fusion_optimizer import search_threshold


AGGREGATION_SIMPLICITY = {"mean": 0, "median": 1, "top2_mean": 2, "top3_mean": 3}


def _aggregate(values: pd.Series, method: str) -> float:
    array = values.astype(float).to_numpy()
    if not len(array):
        raise ValueError("视频没有可聚合的帧。")
    if method == "mean":
        return float(np.mean(array))
    if method == "median":
        return float(np.median(array))
    if method in {"top2_mean", "top3_mean"}:
        k = 2 if method == "top2_mean" else 3
        return float(np.mean(np.sort(array)[-min(k, len(array)) :]))
    raise ValueError(f"未知视频聚合方法：{method}")


def aggregate_video_scores(
    frame_predictions: pd.DataFrame,
    method: str,
    score_columns: tuple[str, ...] = (
        "spatial_fake_probability",
        "frequency_fake_probability",
        "fusion_fake_probability",
    ),
) -> pd.DataFrame:
    required = {"video_id", "pair_id", "true_label", *score_columns}
    missing = sorted(required - set(frame_predictions.columns))
    if missing:
        raise ValueError(f"帧级预测缺少视频聚合字段：{missing}")
    rows: list[dict[str, Any]] = []
    for video_id, group in frame_predictions.groupby("video_id", sort=True):
        labels = set(group["true_label"].astype(int))
        pair_ids = set(group["pair_id"].astype(str))
        if len(labels) != 1 or len(pair_ids) != 1:
            raise ValueError(f"视频{video_id}存在标签或视频对ID冲突。")
        row: dict[str, Any] = {
            "video_id": str(video_id),
            "pair_id": next(iter(pair_ids)),
            "true_label": next(iter(labels)),
            "frame_count": int(len(group)),
        }
        for column in score_columns:
            row[column] = _aggregate(group[column], method)
        rows.append(row)
    return pd.DataFrame(rows)


def select_video_aggregation(
    validation_frames: pd.DataFrame,
    auc_tolerance: float = 0.002,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    aggregated_by_method: dict[str, pd.DataFrame] = {}
    for method in AGGREGATION_SIMPLICITY:
        aggregated = aggregate_video_scores(validation_frames, method)
        threshold, metrics, _ = search_threshold(
            aggregated["true_label"], aggregated["fusion_fake_probability"]
        )
        aggregated_by_method[method] = aggregated
        rows.append(
            {
                "aggregation": method,
                "threshold": threshold,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "Specificity": metrics["Specificity"],
                "F1": metrics["F1"],
                "ROC-AUC": metrics["ROC-AUC"],
                "PR-AUC": metrics["PR-AUC"],
                "video_count": metrics["sample_count"],
            }
        )
    table = pd.DataFrame(rows)
    maximum_auc = float(table["ROC-AUC"].max())
    eligible = table[table["ROC-AUC"] >= maximum_auc - float(auc_tolerance)].copy()
    eligible["simplicity"] = eligible["aggregation"].map(AGGREGATION_SIMPLICITY)
    selected = eligible.sort_values(
        ["F1", "Accuracy", "simplicity", "ROC-AUC"],
        ascending=[False, False, True, False],
        kind="stable",
    ).iloc[0]
    result = {
        "aggregation": str(selected["aggregation"]),
        "threshold": float(selected["threshold"]),
        "selection_metrics": {
            key: float(selected[key])
            for key in ("Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "PR-AUC")
        },
        "maximum_candidate_roc_auc": maximum_auc,
        "auc_tolerance": float(auc_tolerance),
        "selection_rule": "ROC-AUC最高；差异<0.002时依次比较F1、Accuracy和聚合简单性",
    }
    return result, table, aggregated_by_method
