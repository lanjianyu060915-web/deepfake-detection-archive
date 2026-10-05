from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


EPSILON = 1e-6
METHOD_SIMPLICITY = {
    "equal_probability": 0,
    "weighted_probability": 1,
    "weighted_logit": 2,
}


def _as_probability(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("概率必须是一维有限数值数组。")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("概率必须位于0到1之间。")
    return array


def apply_fusion(
    spatial_probability: np.ndarray | pd.Series | list[float],
    frequency_probability: np.ndarray | pd.Series | list[float],
    method: str,
    alpha: float,
) -> np.ndarray:
    spatial = _as_probability(spatial_probability)
    frequency = _as_probability(frequency_probability)
    if spatial.shape != frequency.shape:
        raise ValueError("空间域和频域概率数量不一致。")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("融合权重alpha必须位于0到1之间。")
    alpha = float(alpha)
    if method == "equal_probability":
        if not np.isclose(alpha, 0.5):
            raise ValueError("等权概率融合的alpha必须为0.5。")
        return 0.5 * spatial + 0.5 * frequency
    if method == "weighted_probability":
        return alpha * spatial + (1.0 - alpha) * frequency
    if method == "weighted_logit":
        spatial_safe = np.clip(spatial, EPSILON, 1.0 - EPSILON)
        frequency_safe = np.clip(frequency, EPSILON, 1.0 - EPSILON)
        spatial_logit = np.log(spatial_safe / (1.0 - spatial_safe))
        frequency_logit = np.log(frequency_safe / (1.0 - frequency_safe))
        fused_logit = alpha * spatial_logit + (1.0 - alpha) * frequency_logit
        return 1.0 / (1.0 + np.exp(-fused_logit))
    raise ValueError(f"未知融合方法：{method}")


def classification_metrics(
    targets: np.ndarray | pd.Series | list[int],
    scores: np.ndarray | pd.Series | list[float],
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(targets, dtype=np.int64)
    probability = _as_probability(scores)
    if y_true.shape != probability.shape or y_true.ndim != 1:
        raise ValueError("标签和概率数量不一致。")
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("标签必须使用REAL=0、FAKE=1编码。")
    predicted = (probability >= float(threshold)).astype(np.int64)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    roc_auc = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None
    pr_auc = float(average_precision_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None
    return {
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sample_count": int(len(y_true)),
        "threshold": float(threshold),
        "false_positive_count": fp,
        "false_negative_count": fn,
    }


def search_threshold(
    targets: np.ndarray | pd.Series | list[int],
    scores: np.ndarray | pd.Series | list[float],
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    y_true = np.asarray(targets, dtype=np.int64)
    probability = _as_probability(scores)
    candidates = np.unique(np.concatenate((probability, np.asarray([0.0, 0.5, 1.0]))))
    predictions = probability[np.newaxis, :] >= candidates[:, np.newaxis]
    positive = y_true[np.newaxis, :] == 1
    negative = ~positive
    tp = np.sum(predictions & positive, axis=1).astype(np.float64)
    fp = np.sum(predictions & negative, axis=1).astype(np.float64)
    fn = np.sum((~predictions) & positive, axis=1).astype(np.float64)
    tn = np.sum((~predictions) & negative, axis=1).astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) != 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    accuracy = (tp + tn) / max(float(len(y_true)), 1.0)
    roc_auc = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else np.nan
    pr_auc = float(average_precision_score(y_true, probability)) if len(np.unique(y_true)) == 2 else np.nan
    table = pd.DataFrame(
        {
            "threshold": candidates,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
        }
    )
    table["distance_to_0_5"] = (table["threshold"] - 0.5).abs()
    best = table.sort_values(
        ["F1", "Accuracy", "distance_to_0_5", "threshold"],
        ascending=[False, False, True, True],
        kind="stable",
    ).iloc[0]
    threshold = float(best["threshold"])
    return threshold, classification_metrics(y_true, probability, threshold), table.drop(columns="distance_to_0_5")


def optimize_frame_fusion(
    validation: pd.DataFrame,
    auc_tolerance: float = 0.002,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required = {"true_label", "spatial_fake_probability", "frequency_fake_probability"}
    missing = sorted(required - set(validation.columns))
    if missing:
        raise ValueError(f"验证集对齐预测缺少字段：{missing}")
    targets = validation["true_label"].astype(int).to_numpy()
    spatial = validation["spatial_fake_probability"].astype(float).to_numpy()
    frequency = validation["frequency_fake_probability"].astype(float).to_numpy()
    candidates: list[tuple[str, float]] = [("equal_probability", 0.5)]
    alphas = np.round(np.linspace(0.0, 1.0, 101), 2)
    candidates.extend(("weighted_probability", float(alpha)) for alpha in alphas)
    candidates.extend(("weighted_logit", float(alpha)) for alpha in alphas)
    rows: list[dict[str, Any]] = []
    threshold_tables: list[pd.DataFrame] = []
    for method, alpha in candidates:
        scores = apply_fusion(spatial, frequency, method, alpha)
        threshold, metrics, threshold_table = search_threshold(targets, scores)
        rows.append(
            {
                "method": method,
                "alpha": alpha,
                "threshold": threshold,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "Specificity": metrics["Specificity"],
                "F1": metrics["F1"],
                "ROC-AUC": metrics["ROC-AUC"],
                "PR-AUC": metrics["PR-AUC"],
            }
        )
        threshold_table.insert(0, "alpha", alpha)
        threshold_table.insert(0, "method", method)
        threshold_tables.append(threshold_table)
    table = pd.DataFrame(rows)
    maximum_auc = float(table["ROC-AUC"].max())
    eligible = table[table["ROC-AUC"] >= maximum_auc - float(auc_tolerance)].copy()
    eligible["simplicity"] = eligible["method"].map(METHOD_SIMPLICITY)
    eligible["alpha_stability"] = (eligible["alpha"] - 0.5).abs()
    selected = eligible.sort_values(
        ["F1", "Accuracy", "simplicity", "alpha_stability", "ROC-AUC", "alpha"],
        ascending=[False, False, True, True, False, True],
        kind="stable",
    ).iloc[0]
    result = {
        "method": str(selected["method"]),
        "alpha": float(selected["alpha"]),
        "frequency_weight": 1.0 - float(selected["alpha"]),
        "threshold": float(selected["threshold"]),
        "selection_metrics": {
            key: float(selected[key])
            for key in ("Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "PR-AUC")
        },
        "maximum_candidate_roc_auc": maximum_auc,
        "auc_tolerance": float(auc_tolerance),
        "candidate_count": int(len(table)),
        "tie_break_rule": "ROC-AUC最高；差异<0.002时依次比较F1、Accuracy、简单性和alpha距0.5稳定性",
    }
    return result, table, pd.concat(threshold_tables, ignore_index=True)
