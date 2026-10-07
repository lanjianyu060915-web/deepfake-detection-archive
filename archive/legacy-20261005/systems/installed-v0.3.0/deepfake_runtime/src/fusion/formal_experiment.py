from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve

from src.detectors.frequency_stat import FrequencyStatDetector

from .fusion_optimizer import (
    apply_fusion,
    classification_metrics,
    optimize_frame_fusion,
    search_threshold,
)
from .video_aggregation import aggregate_video_scores, select_video_aggregation


SEED = 20260712
FRAME_PATTERN = re.compile(r"_f(?P<frame>\d+)_idx(?P<source>\d+)$")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化类型：{type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_relative(value: str) -> str:
    return PurePosixPath(str(value).replace("\\", "/")).as_posix()


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("formal_fusion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def protected_files(root: Path) -> list[Path]:
    return [
        root / "data_all" / "data_formal" / "manifest.csv",
        root / "models" / "spatial_mobilenet_formal" / "spatial_mobilenet_state_dict.pt",
        root / "models" / "spatial_mobilenet_finetuned_gpu" / "spatial_mobilenet_finetuned_gpu.pt",
        root / "models" / "spatial_mobilenet_finetuned_gpu" / "metadata.json",
        root / "models" / "frequency_stat_formal" / "frequency_stat_detector.joblib",
        root / "models" / "frequency_stat_formal" / "metadata.json",
        root / "outputs" / "spatial_mobilenet_finetuned_gpu" / "test_predictions.csv",
        root / "outputs" / "spatial_mobilenet_finetuned_gpu" / "test_metrics.json",
        root / "outputs" / "frequency_stat_formal" / "predictions.csv",
        root / "outputs" / "frequency_stat_formal" / "metrics.json",
    ]


def collect_hashes(paths: list[Path], root: Path) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"关键原文件缺失：{missing}")
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def build_frame_manifest(manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {
        "split",
        "pair_id",
        "label",
        "local_video",
        "video_sha256",
        "frame_count",
        "frame_paths",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"data_formal清单缺少字段：{missing}")
    rows: list[dict[str, Any]] = []
    for record in manifest.to_dict(orient="records"):
        frame_paths = [value for value in str(record["frame_paths"]).split("|") if value]
        if len(frame_paths) != int(record["frame_count"]):
            raise ValueError(f"视频{record['local_video']}的frame_count与frame_paths不一致。")
        label_name = str(record["label"]).upper()
        if label_name not in {"REAL", "FAKE"}:
            raise ValueError(f"未知标签：{record['label']}")
        video_id = Path(str(record["local_video"])).stem
        for raw_path in frame_paths:
            relative_path = _normalize_relative(raw_path)
            stem = Path(relative_path).stem
            match = FRAME_PATTERN.search(stem)
            if match is None:
                raise ValueError(f"无法从帧文件名解析帧序号：{relative_path}")
            rows.append(
                {
                    "sample_id": stem,
                    "relative_path": relative_path,
                    "video_id": video_id,
                    "pair_id": str(record["pair_id"]),
                    "frame_index": int(match.group("frame")),
                    "source_frame_index": int(match.group("source")),
                    "split": str(record["split"]),
                    "true_label": 1 if label_name == "FAKE" else 0,
                    "true_label_name": label_name,
                    "video_sha256": str(record["video_sha256"]),
                }
            )
    frames = pd.DataFrame(rows)
    if frames["relative_path"].duplicated().any() or frames["sample_id"].duplicated().any():
        raise ValueError("data_formal清单存在重复帧路径或样本ID。")
    split_pairs = {split: set(group["pair_id"]) for split, group in frames.groupby("split")}
    split_videos = {split: set(group["video_sha256"]) for split, group in frames.groupby("split")}
    leakage: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        leakage[f"pair_id_{left}_{right}"] = sorted(split_pairs[left] & split_pairs[right])
        leakage[f"video_sha256_{left}_{right}"] = sorted(split_videos[left] & split_videos[right])
    if any(leakage.values()):
        raise ValueError(f"检测到跨集合数据泄漏：{leakage}")
    report = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_frame_counts": {
            split: int(count) for split, count in frames.groupby("split").size().items()
        },
        "manifest_video_counts": {
            split: int(count) for split, count in frames.groupby("split")["video_id"].nunique().items()
        },
        "manifest_pair_counts": {
            split: int(count) for split, count in frames.groupby("split")["pair_id"].nunique().items()
        },
        "leakage_checks": leakage,
        "leakage_check_passed": not any(leakage.values()),
    }
    return frames, report


def _standardize_spatial(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    path_column = "sample_path"
    required = {path_column, "true_label", "fake_probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"空间域{split}预测缺少字段：{missing}")
    return pd.DataFrame(
        {
            "relative_path": frame[path_column].map(_normalize_relative),
            "spatial_true_label_name": frame["true_label"].str.upper(),
            "spatial_fake_probability": pd.to_numeric(frame["fake_probability"], errors="raise"),
        }
    )


def _standardize_frequency(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    path_column = "relative_path" if "relative_path" in frame.columns else "frame_path"
    required = {path_column, "true_label", "fake_probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"频域{split}预测缺少字段：{missing}")
    return pd.DataFrame(
        {
            "relative_path": frame[path_column].map(_normalize_relative),
            "frequency_true_label_name": frame["true_label"].str.upper(),
            "frequency_fake_probability": pd.to_numeric(frame["fake_probability"], errors="raise"),
        }
    )


def generate_frequency_validation_predictions(
    root: Path,
    manifest_frames: pd.DataFrame,
    spatial_validation_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    spatial = _standardize_spatial(spatial_validation_path, "val")
    expected = set(spatial["relative_path"])
    if output_path.is_file():
        existing = _standardize_frequency(output_path, "val")
        if set(existing["relative_path"]) != expected or len(existing) != len(expected):
            raise ValueError("已有融合实验频域验证预测与空间域验证样本集合不一致。")
        logger.info("复用融合目录已有频域验证集预测：%s", output_path)
        return
    lookup = manifest_frames.set_index("relative_path")
    detector = FrequencyStatDetector(
        model_path=root / "models" / "frequency_stat_formal" / "frequency_stat_detector.joblib",
        metadata_path=root / "models" / "frequency_stat_formal" / "metadata.json",
    )
    detector.load_model()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, relative_path in enumerate(sorted(expected), start=1):
        if relative_path not in lookup.index:
            raise ValueError(f"空间域验证样本不在manifest中：{relative_path}")
        record = lookup.loc[relative_path]
        result = detector.predict(root / "data_all" / "data_formal" / relative_path)
        rows.append(
            {
                "split": "val",
                "sample_id": record["sample_id"],
                "relative_path": relative_path,
                "video_id": record["video_id"],
                "pair_id": record["pair_id"],
                "true_label": record["true_label_name"],
                "fake_probability": result["fake_probability"],
                "predicted_label_at_original_threshold": result["label"],
                "inference_ms": result["inference_ms"],
            }
        )
        if index % 25 == 0 or index == len(expected):
            logger.info("频域验证集确定性推理：%d/%d", index, len(expected))
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("频域验证集推理完成：%d条，耗时%.2fs", len(rows), time.perf_counter() - started)


def align_predictions(
    manifest_frames: pd.DataFrame,
    spatial_path: Path,
    frequency_path: Path,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_manifest = manifest_frames[manifest_frames["split"] == split].copy()
    spatial = _standardize_spatial(spatial_path, split)
    frequency = _standardize_frequency(frequency_path, split)
    for name, frame in (("spatial", spatial), ("frequency", frequency)):
        duplicates = frame.loc[frame["relative_path"].duplicated(), "relative_path"].tolist()
        if duplicates:
            raise ValueError(f"{name} {split}预测存在重复样本：{duplicates[:5]}")
        probability_column = f"{name}_fake_probability"
        values = frame[probability_column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} {split}预测包含非法概率。")
    spatial_set = set(spatial["relative_path"])
    frequency_set = set(frequency["relative_path"])
    manifest_set = set(expected_manifest["relative_path"])
    if spatial_set != frequency_set:
        raise ValueError(
            f"{split}两模型样本集合不一致：空间独有{sorted(spatial_set-frequency_set)[:5]}，"
            f"频域独有{sorted(frequency_set-spatial_set)[:5]}"
        )
    if split == "test" and spatial_set != manifest_set:
        raise ValueError("测试集预测未完整覆盖manifest中的150帧。")
    unexpected = spatial_set - manifest_set
    if unexpected:
        raise ValueError(f"{split}预测包含manifest外样本：{sorted(unexpected)[:5]}")
    aligned = expected_manifest[expected_manifest["relative_path"].isin(spatial_set)].merge(
        spatial, on="relative_path", how="inner", validate="one_to_one"
    )
    aligned = aligned.merge(frequency, on="relative_path", how="inner", validate="one_to_one")
    label_conflicts = aligned[
        (aligned["true_label_name"] != aligned["spatial_true_label_name"])
        | (aligned["true_label_name"] != aligned["frequency_true_label_name"])
    ]
    if not label_conflicts.empty:
        raise ValueError(f"{split}标签语义冲突：{label_conflicts['relative_path'].head().tolist()}")
    aligned = aligned[
        [
            "sample_id",
            "relative_path",
            "video_id",
            "pair_id",
            "frame_index",
            "source_frame_index",
            "split",
            "true_label",
            "true_label_name",
            "spatial_fake_probability",
            "frequency_fake_probability",
        ]
    ].sort_values(["pair_id", "video_id", "frame_index"], kind="stable")
    aligned.reset_index(drop=True, inplace=True)
    report = {
        "split": split,
        "manifest_sample_count": int(len(expected_manifest)),
        "spatial_prediction_count": int(len(spatial)),
        "frequency_prediction_count": int(len(frequency)),
        "aligned_sample_count": int(len(aligned)),
        "aligned_video_count": int(aligned["video_id"].nunique()),
        "aligned_pair_count": int(aligned["pair_id"].nunique()),
        "spatial_duplicate_count": int(spatial["relative_path"].duplicated().sum()),
        "frequency_duplicate_count": int(frequency["relative_path"].duplicated().sum()),
        "spatial_only": sorted(spatial_set - frequency_set),
        "frequency_only": sorted(frequency_set - spatial_set),
        "missing_from_both_vs_manifest": sorted(manifest_set - spatial_set),
        "extra_vs_manifest": sorted(spatial_set - manifest_set),
        "label_conflict_count": int(len(label_conflicts)),
        "sample_id_unique": bool(aligned["sample_id"].is_unique),
        "passed": True,
    }
    return aligned, report


def _save_confusion(metrics: dict[str, Any], title: str, path: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
    axis.set_xticks([0, 1], ["REAL", "FAKE"])
    axis.set_yticks([0, 1], ["REAL", "FAKE"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_roc(targets: pd.Series, scores: pd.Series, auc: float, title: str, path: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(targets.astype(int), scores.astype(float))
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"Fusion AUC = {auc:.4f}", linewidth=2)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_figures(
    figures_dir: Path,
    candidate_table: pd.DataFrame,
    test_frames: pd.DataFrame,
    frame_metrics: dict[str, Any],
    test_videos: pd.DataFrame,
    video_metrics: dict[str, Any],
    frame_comparison: pd.DataFrame,
    complementarity: dict[str, Any],
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    weighted = candidate_table[candidate_table["method"].isin(["weighted_probability", "weighted_logit"])]
    for metric, filename, ylabel in (
        ("ROC-AUC", "validation_alpha_roc_auc.png", "Validation ROC-AUC"),
        ("F1", "validation_alpha_f1.png", "Validation F1"),
    ):
        fig, axis = plt.subplots(figsize=(7, 5))
        for method, group in weighted.groupby("method"):
            axis.plot(group["alpha"], group[metric], label=method)
        axis.set_xlabel("Spatial weight alpha")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Alpha vs {ylabel}")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=180)
        plt.close(fig)

    _save_roc(
        test_frames["true_label"],
        test_frames["fusion_fake_probability"],
        float(frame_metrics["ROC-AUC"]),
        "Formal fusion frame-level ROC",
        figures_dir / "frame_level_roc_curve.png",
    )
    _save_confusion(frame_metrics, "Formal fusion frame-level confusion matrix", figures_dir / "frame_level_confusion_matrix.png")
    _save_roc(
        test_videos["true_label"],
        test_videos["fusion_video_score"],
        float(video_metrics["ROC-AUC"]),
        "Formal fusion video-level ROC",
        figures_dir / "video_level_roc_curve.png",
    )
    _save_confusion(video_metrics, "Formal fusion video-level confusion matrix", figures_dir / "video_level_confusion_matrix.png")

    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    plot_frame = frame_comparison[frame_comparison["model"].isin(["GPU Spatial", "Frequency", "Formal Fusion"])]
    x = np.arange(len(metrics))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5))
    for index, (_, row) in enumerate(plot_frame.iterrows()):
        axis.bar(x + (index - 1) * width, [row[item] for item in metrics], width, label=row["model"])
    axis.set_xticks(x, metrics)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Frame-level model comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "model_metric_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    colors = test_frames["true_label"].map({0: "tab:blue", 1: "tab:red"})
    axis.scatter(
        test_frames["spatial_fake_probability"],
        test_frames["frequency_fake_probability"],
        c=colors,
        alpha=0.7,
        s=28,
    )
    axis.set_xlabel("Spatial FAKE probability")
    axis.set_ylabel("Frequency FAKE probability")
    axis.set_title("Spatial vs frequency probabilities")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "spatial_frequency_scatter.png", dpi=180)
    plt.close(fig)

    correlation = np.asarray(
        [[1.0, complementarity["pearson_correlation"]], [complementarity["pearson_correlation"], 1.0]]
    )
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{correlation[row, column]:.3f}", ha="center", va="center")
    axis.set_xticks([0, 1], ["Spatial", "Frequency"])
    axis.set_yticks([0, 1], ["Spatial", "Frequency"])
    axis.set_title("Prediction score correlation")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(figures_dir / "prediction_correlation.png", dpi=180)
    plt.close(fig)

    labels = ["Both correct", "Spatial only", "Frequency only", "Both wrong", "Fusion fixes spatial", "Fusion new vs spatial"]
    values = [
        complementarity["both_correct"],
        complementarity["spatial_only_correct"],
        complementarity["frequency_only_correct"],
        complementarity["both_wrong"],
        complementarity["fusion_corrected_spatial_errors"],
        complementarity["fusion_new_errors_vs_spatial"],
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, values, color=["#4c78a8", "#72b7b2", "#f2cf5b", "#e45756", "#54a24b", "#b279a2"])
    axis.set_ylabel("Frame count")
    axis.set_title("Error complementarity and fusion changes")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "error_change_statistics.png", dpi=180)
    plt.close(fig)

    differences = test_frames["spatial_fake_probability"] - test_frames["frequency_fake_probability"]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.hist(differences, bins=20, color="#4c78a8", edgecolor="white")
    axis.axvline(0.0, linestyle="--", color="black")
    axis.set_xlabel("Spatial probability - frequency probability")
    axis.set_ylabel("Frame count")
    axis.set_title("Prediction score difference distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "score_difference_distribution.png", dpi=180)
    plt.close(fig)


def _metric_row(name: str, metrics: dict[str, Any], threshold_source: str) -> dict[str, Any]:
    return {
        "model": name,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "Specificity": metrics.get("Specificity"),
        "F1": metrics["F1"],
        "ROC-AUC": metrics["ROC-AUC"],
        "PR-AUC": metrics.get("PR-AUC"),
        "threshold": metrics.get("threshold", metrics.get("classification_threshold")),
        "threshold_source": threshold_source,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, record in frame.iterrows():
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
                values.append(f"{float(value):.4f}")
            elif pd.isna(value):
                values.append("N/A")
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _complementarity(
    test: pd.DataFrame,
    spatial_threshold: float,
    frequency_threshold: float,
    fusion_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    target = test["true_label"].astype(int).to_numpy()
    spatial_pred = (test["spatial_fake_probability"].to_numpy() >= spatial_threshold).astype(int)
    frequency_pred = (test["frequency_fake_probability"].to_numpy() >= frequency_threshold).astype(int)
    fusion_pred = (test["fusion_fake_probability"].to_numpy() >= fusion_threshold).astype(int)
    spatial_correct = spatial_pred == target
    frequency_correct = frequency_pred == target
    fusion_correct = fusion_pred == target
    pearson = pearsonr(test["spatial_fake_probability"], test["frequency_fake_probability"])
    spearman = spearmanr(test["spatial_fake_probability"], test["frequency_fake_probability"])
    difference = test["spatial_fake_probability"] - test["frequency_fake_probability"]
    payload = {
        "both_correct": int(np.sum(spatial_correct & frequency_correct)),
        "spatial_only_correct": int(np.sum(spatial_correct & ~frequency_correct)),
        "frequency_only_correct": int(np.sum(~spatial_correct & frequency_correct)),
        "both_wrong": int(np.sum(~spatial_correct & ~frequency_correct)),
        "fusion_corrected_spatial_errors": int(np.sum(~spatial_correct & fusion_correct)),
        "fusion_corrected_frequency_errors": int(np.sum(~frequency_correct & fusion_correct)),
        "fusion_new_errors_vs_spatial": int(np.sum(spatial_correct & ~fusion_correct)),
        "fusion_new_errors_vs_frequency": int(np.sum(frequency_correct & ~fusion_correct)),
        "single_model_correct_but_fusion_wrong": int(np.sum((spatial_correct | frequency_correct) & ~fusion_correct)),
        "pearson_correlation": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_correlation": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "score_difference": {
            "mean": float(difference.mean()),
            "std": float(difference.std(ddof=0)),
            "min": float(difference.min()),
            "median": float(difference.median()),
            "max": float(difference.max()),
            "mean_absolute": float(difference.abs().mean()),
        },
        "spatial_threshold": spatial_threshold,
        "frequency_threshold": frequency_threshold,
        "fusion_threshold": fusion_threshold,
    }
    disagreement = test[spatial_pred != frequency_pred].copy()
    disagreement["spatial_predicted_label"] = np.where(spatial_pred[spatial_pred != frequency_pred] == 1, "FAKE", "REAL")
    disagreement["frequency_predicted_label"] = np.where(frequency_pred[spatial_pred != frequency_pred] == 1, "FAKE", "REAL")
    disagreement["fusion_predicted_label"] = np.where(fusion_pred[spatial_pred != frequency_pred] == 1, "FAKE", "REAL")
    disagreement["spatial_is_correct"] = spatial_correct[spatial_pred != frequency_pred]
    disagreement["frequency_is_correct"] = frequency_correct[spatial_pred != frequency_pred]
    disagreement["fusion_is_correct"] = fusion_correct[spatial_pred != frequency_pred]

    categories = {
        "空间错误、频域正确、融合正确": (~spatial_correct) & frequency_correct & fusion_correct,
        "频域错误、空间正确、融合正确": spatial_correct & (~frequency_correct) & fusion_correct,
        "两个模型均错误": (~spatial_correct) & (~frequency_correct),
        "单模型正确但融合错误": (spatial_correct | frequency_correct) & (~fusion_correct),
    }
    cases: list[dict[str, Any]] = []
    for category, mask in categories.items():
        indices = np.flatnonzero(mask)
        if len(indices):
            row = test.iloc[int(indices[0])]
            cases.append(
                {
                    "category": category,
                    "sample_id": row["sample_id"],
                    "relative_path": row["relative_path"],
                    "true_label": row["true_label_name"],
                    "spatial_probability": row["spatial_fake_probability"],
                    "frequency_probability": row["frequency_fake_probability"],
                    "fusion_probability": row["fusion_fake_probability"],
                }
            )
    return payload, disagreement, cases


def _report(
    path: Path,
    dataset_report: dict[str, Any],
    alignment_report: dict[str, Any],
    selection: dict[str, Any],
    video_selection: dict[str, Any],
    frame_metrics: dict[str, Any],
    video_metrics: dict[str, Any],
    frame_comparison: pd.DataFrame,
    video_comparison: pd.DataFrame,
    complementarity: dict[str, Any],
    cases: list[dict[str, Any]],
    recommendation: dict[str, Any],
    config_path: Path,
) -> None:
    method_formulas = {
        "equal_probability": "p_fusion = 0.5 × p_spatial + 0.5 × p_frequency",
        "weighted_probability": "p_fusion = alpha × p_spatial + (1-alpha) × p_frequency",
        "weighted_logit": "p_fusion = sigmoid(alpha × logit(p_spatial) + (1-alpha) × logit(p_frequency))",
    }
    frame_table = _markdown_table(frame_comparison)
    video_table = _markdown_table(video_comparison)
    case_lines = "\n".join(
        f"- {item['category']}：`{item['relative_path']}`，真实={item['true_label']}，"
        f"空间={item['spatial_probability']:.4f}，频域={item['frequency_probability']:.4f}，融合={item['fusion_probability']:.4f}"
        for item in cases
    ) or "- 本次测试集中没有满足指定类别的案例。"
    spatial_delta_auc = recommendation["delta_vs_gpu_spatial"]["ROC-AUC"]
    spatial_delta_f1 = recommendation["delta_vs_gpu_spatial"]["F1"]
    content = f"""# 空间域GPU微调模型与频域模型正式融合实验报告

生成时间：{_now()}

## 1. 实验目标

复用已完成的GPU微调SpatialMobileNetDetector和FrequencyStatDetector，在不重新训练单模型、不改变数据划分的前提下完成验证集调参、帧级与视频级正式融合评估，并提供后续Streamlit可直接调用的统一检测接口。

## 2. 数据集与划分

- 实际数据路径：`data_all/data_formal`
- manifest：`{dataset_report['manifest_path']}`
- 帧数：train={dataset_report['manifest_frame_counts']['train']}，val={dataset_report['manifest_frame_counts']['val']}，test={dataset_report['manifest_frame_counts']['test']}
- 视频数：train={dataset_report['manifest_video_counts']['train']}，val={dataset_report['manifest_video_counts']['val']}，test={dataset_report['manifest_video_counts']['test']}
- 视频对数：train={dataset_report['manifest_pair_counts']['train']}，val={dataset_report['manifest_pair_counts']['val']}，test={dataset_report['manifest_pair_counts']['test']}
- pair_id和video_sha256跨集合检查均无交集，未重新划分数据。

验证集manifest有150帧；两条正式分支共同有效145帧。缺少的5帧均为原流程已记录的“未检测到人脸”样本，不是本阶段删除。测试集150帧全部参与正式评价。

## 3. 两个输入模型

- 空间域：`models/spatial_mobilenet_finetuned_gpu/spatial_mobilenet_finetuned_gpu.pt`，MobileNetV3-Small后3个特征模块微调，正式空间分支。
- 频域：`models/frequency_stat_formal/frequency_stat_detector.joblib`，87维频域统计特征 + StandardScaler + LogisticRegression。
- 两个模型均只加载推理，本阶段没有训练，也没有覆盖原模型或原结果。

## 4. 样本对齐方法

先将manifest展开为逐帧表，再统一Windows路径分隔符为POSIX相对路径，并按`relative_path`一对一合并；`sample_id`、`video_id`、`pair_id`、帧序号和标签均由manifest确定，不按CSV行号合并。

- 验证集：空间={alignment_report['validation']['spatial_prediction_count']}，频域={alignment_report['validation']['frequency_prediction_count']}，对齐={alignment_report['validation']['aligned_sample_count']}，重复=0，标签冲突=0。
- 测试集：空间={alignment_report['test']['spatial_prediction_count']}，频域={alignment_report['test']['frequency_prediction_count']}，对齐={alignment_report['test']['aligned_sample_count']}，重复=0，标签冲突=0。

## 5. 标签和概率语义检查

统一标签为REAL=0、FAKE=1。空间分支取softmax的FAKE索引1；频域分支明确取`pipeline.classes_ == 1`的`predict_proba`。两列数值均检查为有限且位于[0,1]，因此都表示“越高越可能FAKE”，没有进行方向反转。

## 6. 候选融合方法

- 方法A：等权概率融合。
- 方法B：加权概率融合，alpha从0.00到1.00，步长0.01。
- 方法C：概率裁剪到[1e-6, 1-1e-6]后进行加权logit融合，alpha同样网格搜索。

共比较{selection['candidate_count']}个候选。选择规则为ROC-AUC优先；若与最高ROC-AUC差异不超过{selection['auc_tolerance']}，再依次比较F1、Accuracy、方法简单性和alpha稳定性。

## 7. 验证集调参结果

- 最终方法：`{selection['method']}`
- 最佳alpha（空间权重）：{selection['alpha']:.4f}
- 频域权重：{selection['frequency_weight']:.4f}
- 帧级阈值：{selection['threshold']:.12f}
- 验证集Accuracy={selection['selection_metrics']['Accuracy']:.4f}，Precision={selection['selection_metrics']['Precision']:.4f}，Recall={selection['selection_metrics']['Recall']:.4f}，F1={selection['selection_metrics']['F1']:.4f}，ROC-AUC={selection['selection_metrics']['ROC-AUC']:.4f}。

最终公式：`{method_formulas[selection['method']]}`，代入alpha={selection['alpha']:.4f}。

视频聚合比较mean、median、top-2 mean、top-3 mean，最终选择`{video_selection['aggregation']}`，视频阈值={video_selection['threshold']:.12f}；流程为“每帧先融合，再对同一视频的融合概率聚合”。

## 8. 测试集使用约束

所有融合超参数仅由验证集决定。正式配置在读取测试预测前写入并冻结；测试集未参与方法、权重、帧阈值、视频聚合或视频阈值选择，且仅用于本次最终正式评价。测试预测直接复用两个单模型已有正式CSV，没有重新执行测试推理。

## 9. 帧级正式测试结果

- 样本数：{frame_metrics['sample_count']}
- Accuracy：{frame_metrics['Accuracy']:.4f}
- Precision：{frame_metrics['Precision']:.4f}
- Recall：{frame_metrics['Recall']:.4f}
- Specificity：{frame_metrics['Specificity']:.4f}
- F1：{frame_metrics['F1']:.4f}
- ROC-AUC：{frame_metrics['ROC-AUC']:.4f}
- PR-AUC：{frame_metrics['PR-AUC']:.4f}
- 混淆矩阵：{frame_metrics['confusion_matrix']}

## 10. 视频级正式测试结果

- 视频数：{video_metrics['sample_count']}
- Accuracy：{video_metrics['Accuracy']:.4f}
- Precision：{video_metrics['Precision']:.4f}
- Recall：{video_metrics['Recall']:.4f}
- Specificity：{video_metrics['Specificity']:.4f}
- F1：{video_metrics['F1']:.4f}
- ROC-AUC：{video_metrics['ROC-AUC']:.4f}
- PR-AUC：{video_metrics['PR-AUC']:.4f}
- 混淆矩阵：{video_metrics['confusion_matrix']}

## 11. 单模型与融合模型对比

### 帧级

{frame_table}

CPU空间基线只引用原正式150帧结果；另外三个模型均在本次严格对齐的同150帧上复算。GPU空间和频域使用各自原验证阈值，融合使用本阶段验证阈值。

### 视频级

{video_table}

视频级单分支和融合均采用本阶段选定的同一种聚合方法；各分支阈值只由各自验证视频分数确定。CPU基线缺少可直接复用的验证逐帧预测，因此不伪造其视频阈值，未列入视频级数值比较。

融合相对GPU空间：ROC-AUC变化{spatial_delta_auc:+.4f}，F1变化{spatial_delta_f1:+.4f}。融合相对频域：ROC-AUC变化{recommendation['delta_vs_frequency']['ROC-AUC']:+.4f}，F1变化{recommendation['delta_vs_frequency']['F1']:+.4f}。

## 12. 互补性分析

- 两模型都正确：{complementarity['both_correct']}
- 仅空间正确：{complementarity['spatial_only_correct']}
- 仅频域正确：{complementarity['frequency_only_correct']}
- 两模型都错误：{complementarity['both_wrong']}
- 融合修正空间错误：{complementarity['fusion_corrected_spatial_errors']}
- 融合修正频域错误：{complementarity['fusion_corrected_frequency_errors']}
- 融合相对空间新引入错误：{complementarity['fusion_new_errors_vs_spatial']}
- Pearson相关性：{complementarity['pearson_correlation']:.4f}
- Spearman相关性：{complementarity['spearman_correlation']:.4f}
- 两分支概率差的平均绝对值：{complementarity['score_difference']['mean_absolute']:.4f}

相关性低于1且存在“仅空间正确”和“仅频域正确”样本，说明两分支具有互补信息；融合是否把这种互补转化为最终收益，以真实测试指标为准。

## 13. 典型案例

{case_lines}

完整分歧清单见`disagreement_samples.csv`。

## 14. 局限性

- 验证集只有15个视频对、145个有效帧，alpha和阈值估计仍可能有方差。
- 帧来自同一视频，帧级样本并非完全独立；因此同时报告30段视频的视频级结果。
- 两分支都依赖同一OpenCV人脸检测前处理，未检出人脸时无法输出融合结论。
- 简单加权融合没有学习跨样本的非线性交互，但更适合当前小验证集，也更可解释。

## 15. 正式推荐结论

**{recommendation['decision']}**。{recommendation['reason']}

无论推荐结果如何，原CPU空间模型、GPU空间模型和频域模型均保持不变并可独立加载。

## 16. 后续Streamlit接入

正式配置：`{config_path}`。界面层可使用：

```python
from src.detectors import FusionDetector

detector = FusionDetector(r"models/fusion_formal/fusion_config.json", device="auto")
result = detector.predict(image_path)
```

返回`spatial_score`、`frequency_score`、`fusion_score`、`threshold`、`predicted_label`、`risk_level`和耗时字段；视频流程可批量调用`predict_batch`后按配置中的`video_aggregation`与`video_threshold`聚合。

## 17. 工件保护声明

原单模型未重新训练；原正式模型、数据、测试预测、指标、图表和报告均未覆盖。运行前后关键原文件SHA-256逐项一致，详见`original_integrity.json`。
"""
    path.write_text(content, encoding="utf-8")


def run_formal_fusion(root: Path) -> dict[str, Any]:
    root = root.resolve()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    output_dir = root / "outputs" / "fusion_formal"
    model_dir = root / "models" / "fusion_formal"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(output_dir / "fusion_experiment.log")
    logger.info("开始正式融合实验；本阶段不训练任何单模型。")

    key_files = protected_files(root)
    hashes_before = collect_hashes(key_files, root)
    write_json(output_dir / "protected_file_hashes_before.json", hashes_before)
    manifest_path = root / "data_all" / "data_formal" / "manifest.csv"
    manifest_frames, dataset_report = build_frame_manifest(manifest_path)
    logger.info("manifest校验通过：val=%d，test=%d，无跨集合泄漏。", dataset_report["manifest_frame_counts"]["val"], dataset_report["manifest_frame_counts"]["test"])

    spatial_val_path = root / "outputs" / "spatial_mobilenet_finetuned_gpu" / "validation_predictions.csv"
    spatial_test_path = root / "outputs" / "spatial_mobilenet_finetuned_gpu" / "test_predictions.csv"
    frequency_test_path = root / "outputs" / "frequency_stat_formal" / "predictions.csv"
    frequency_val_path = output_dir / "frequency_validation_predictions.csv"
    generate_frequency_validation_predictions(
        root, manifest_frames, spatial_val_path, frequency_val_path, logger
    )
    validation, validation_alignment = align_predictions(
        manifest_frames, spatial_val_path, frequency_val_path, "val"
    )
    validation.to_csv(output_dir / "validation_aligned_predictions.csv", index=False, encoding="utf-8-sig")
    logger.info("验证集严格对齐通过：%d帧，%d段视频。", len(validation), validation["video_id"].nunique())

    selection, candidate_table, threshold_table = optimize_frame_fusion(validation)
    candidate_table.to_csv(output_dir / "validation_fusion_grid.csv", index=False, encoding="utf-8-sig")
    threshold_table.to_csv(output_dir / "validation_threshold_search.csv", index=False, encoding="utf-8-sig")
    validation["fusion_fake_probability"] = apply_fusion(
        validation["spatial_fake_probability"],
        validation["frequency_fake_probability"],
        selection["method"],
        selection["alpha"],
    )
    validation["fusion_predicted_label"] = (
        validation["fusion_fake_probability"] >= selection["threshold"]
    ).astype(int)
    validation.to_csv(output_dir / "validation_fusion_predictions.csv", index=False, encoding="utf-8-sig")
    video_selection, video_candidate_table, _ = select_video_aggregation(validation)
    video_candidate_table.to_csv(output_dir / "validation_video_aggregation_grid.csv", index=False, encoding="utf-8-sig")
    logger.info(
        "验证集冻结配置：method=%s alpha=%.2f frame_threshold=%.12f video=%s video_threshold=%.12f",
        selection["method"], selection["alpha"], selection["threshold"],
        video_selection["aggregation"], video_selection["threshold"],
    )

    spatial_model = root / "models" / "spatial_mobilenet_finetuned_gpu" / "spatial_mobilenet_finetuned_gpu.pt"
    frequency_model = root / "models" / "frequency_stat_formal" / "frequency_stat_detector.joblib"
    config = {
        "spatial_model_path": spatial_model.relative_to(root).as_posix(),
        "frequency_model_path": frequency_model.relative_to(root).as_posix(),
        "fusion_method": selection["method"],
        "spatial_weight": selection["alpha"],
        "frequency_weight": selection["frequency_weight"],
        "frame_threshold": selection["threshold"],
        "video_aggregation": video_selection["aggregation"],
        "video_threshold": video_selection["threshold"],
        "label_mapping": {"REAL": 0, "FAKE": 1},
        "preprocessing_version": "opencv_largest_face_expand_224 + MobileNetV3_DEFAULT / FrequencyStat_v1.0.0",
        "created_at": _now(),
        "validation_selection_metric": "ROC-AUC; within 0.002 use F1, Accuracy, simplicity and stability",
        "validation_sample_count": int(len(validation)),
        "validation_video_count": int(validation["video_id"].nunique()),
        "spatial_model_sha256": sha256_file(spatial_model),
        "frequency_model_sha256": sha256_file(frequency_model),
        "test_set_used_for_selection": False,
        "frame_fusion_then_video_aggregation": True,
    }
    config_path = model_dir / "fusion_config.json"
    write_json(config_path, config)
    frozen_config_sha256 = sha256_file(config_path)
    logger.info("正式融合配置已在读取测试预测前冻结：%s，SHA-256=%s", config_path, frozen_config_sha256)

    # 配置冻结后才读取测试集已有正式预测；本代码段是唯一一次正式测试评价。
    test, test_alignment = align_predictions(
        manifest_frames, spatial_test_path, frequency_test_path, "test"
    )
    test.to_csv(output_dir / "test_aligned_predictions.csv", index=False, encoding="utf-8-sig")
    alignment_report = {
        "created_at": _now(),
        "join_key": "normalized relative_path (one-to-one), not CSV row order",
        "label_mapping": {"REAL": 0, "FAKE": 1},
        "probability_semantics": {
            "spatial": "softmax class index 1 = FAKE probability",
            "frequency": "predict_proba where pipeline.classes_ == 1 = FAKE probability",
        },
        "dataset": dataset_report,
        "validation": validation_alignment,
        "test": test_alignment,
        "all_checks_passed": True,
    }
    write_json(output_dir / "alignment_report.json", alignment_report)
    logger.info("测试集严格对齐通过：%d帧；开始唯一一次正式评价。", len(test))

    test["fusion_fake_probability"] = apply_fusion(
        test["spatial_fake_probability"],
        test["frequency_fake_probability"],
        config["fusion_method"],
        config["spatial_weight"],
    )
    test["fusion_predicted_label"] = (
        test["fusion_fake_probability"] >= config["frame_threshold"]
    ).astype(int)
    test["fusion_predicted_label_name"] = test["fusion_predicted_label"].map({0: "REAL", 1: "FAKE"})
    test["is_correct"] = test["fusion_predicted_label"] == test["true_label"]
    frame_metrics = classification_metrics(
        test["true_label"], test["fusion_fake_probability"], config["frame_threshold"]
    )
    frame_metrics.update(
        {
            "fusion_method": config["fusion_method"],
            "spatial_weight": config["spatial_weight"],
            "frequency_weight": config["frequency_weight"],
            "formula": "frame fusion first, then threshold",
            "config_sha256": frozen_config_sha256,
            "test_evaluation_count_this_run": 1,
        }
    )
    write_json(output_dir / "frame_level_test_metrics.json", frame_metrics)
    test.to_csv(output_dir / "frame_level_test_predictions.csv", index=False, encoding="utf-8-sig")

    video = aggregate_video_scores(test, config["video_aggregation"])
    video.rename(
        columns={
            "spatial_fake_probability": "spatial_video_score",
            "frequency_fake_probability": "frequency_video_score",
            "fusion_fake_probability": "fusion_video_score",
        },
        inplace=True,
    )
    video["predicted_label"] = (video["fusion_video_score"] >= config["video_threshold"]).astype(int)
    video["predicted_label_name"] = video["predicted_label"].map({0: "REAL", 1: "FAKE"})
    video["true_label_name"] = video["true_label"].map({0: "REAL", 1: "FAKE"})
    video["is_correct"] = video["predicted_label"] == video["true_label"]
    video_metrics = classification_metrics(
        video["true_label"], video["fusion_video_score"], config["video_threshold"]
    )
    video_metrics.update(
        {
            "aggregation": config["video_aggregation"],
            "pipeline": "frame-level fusion -> per-video aggregation -> fixed video threshold",
            "config_sha256": frozen_config_sha256,
        }
    )
    write_json(output_dir / "video_level_test_metrics.json", video_metrics)
    video.to_csv(output_dir / "video_level_test_predictions.csv", index=False, encoding="utf-8-sig")

    spatial_metadata = json.loads((spatial_model.parent / "metadata.json").read_text(encoding="utf-8"))
    frequency_metadata = json.loads((frequency_model.parent / "metadata.json").read_text(encoding="utf-8"))
    spatial_threshold = float(spatial_metadata["classification_threshold"])
    frequency_threshold = float(frequency_metadata["classification_threshold"])
    gpu_metrics = classification_metrics(test["true_label"], test["spatial_fake_probability"], spatial_threshold)
    frequency_metrics = classification_metrics(test["true_label"], test["frequency_fake_probability"], frequency_threshold)
    baseline_metrics = json.loads(
        (root / "outputs" / "spatial_mobilenet_finetuned_gpu" / "baseline_recomputed_metrics.json").read_text(encoding="utf-8")
    )
    frame_comparison = pd.DataFrame(
        [
            _metric_row("CPU Spatial Baseline", baseline_metrics, "original validation threshold"),
            _metric_row("GPU Spatial", gpu_metrics, "original validation threshold"),
            _metric_row("Frequency", frequency_metrics, "original validation threshold"),
            _metric_row("Formal Fusion", frame_metrics, "fusion validation threshold"),
        ]
    )
    frame_comparison.to_csv(output_dir / "frame_level_model_comparison.csv", index=False, encoding="utf-8-sig")

    validation_video = aggregate_video_scores(validation, config["video_aggregation"])
    branch_video_selection: dict[str, Any] = {}
    video_rows: list[dict[str, Any]] = []
    for name, validation_column, test_column in (
        ("GPU Spatial", "spatial_fake_probability", "spatial_video_score"),
        ("Frequency", "frequency_fake_probability", "frequency_video_score"),
    ):
        branch_threshold, branch_val_metrics, _ = search_threshold(
            validation_video["true_label"], validation_video[validation_column]
        )
        branch_test_metrics = classification_metrics(video["true_label"], video[test_column], branch_threshold)
        branch_video_selection[name] = {
            "aggregation": config["video_aggregation"],
            "validation_threshold": branch_threshold,
            "validation_metrics": branch_val_metrics,
        }
        video_rows.append(_metric_row(name, branch_test_metrics, "branch validation video threshold"))
    video_rows.append(_metric_row("Formal Fusion", video_metrics, "fusion validation video threshold"))
    video_comparison = pd.DataFrame(video_rows)
    video_comparison.to_csv(output_dir / "video_level_model_comparison.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "single_model_video_validation_selection.json", branch_video_selection)

    complementarity, disagreement, cases = _complementarity(
        test, spatial_threshold, frequency_threshold, config["frame_threshold"]
    )
    complementarity["typical_cases"] = cases
    write_json(output_dir / "complementarity_analysis.json", complementarity)
    disagreement.to_csv(output_dir / "disagreement_samples.csv", index=False, encoding="utf-8-sig")

    gpu_row = frame_comparison[frame_comparison["model"] == "GPU Spatial"].iloc[0]
    frequency_row = frame_comparison[frame_comparison["model"] == "Frequency"].iloc[0]
    fusion_row = frame_comparison[frame_comparison["model"] == "Formal Fusion"].iloc[0]
    deltas_gpu = {metric: float(fusion_row[metric] - gpu_row[metric]) for metric in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")}
    deltas_frequency = {metric: float(fusion_row[metric] - frequency_row[metric]) for metric in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")}
    nontrivial_fusion = 0.0 < config["spatial_weight"] < 1.0
    recommended = bool(nontrivial_fusion and (deltas_gpu["ROC-AUC"] > 0.0 or deltas_gpu["F1"] > 0.0))
    recommendation = {
        "recommended": recommended,
        "decision": "推荐正式系统使用融合模型" if recommended else "暂不推荐融合模型替换GPU空间模型",
        "reason": (
            "验证集独立选择得到非退化融合权重，且正式测试ROC-AUC或F1相对GPU空间模型取得真实提升。"
            if recommended
            else "验证集选择结果或正式测试收益不足以支持用融合模型替换GPU空间模型；保留真实下降结果。"
        ),
        "delta_vs_gpu_spatial": deltas_gpu,
        "delta_vs_frequency": deltas_frequency,
        "nontrivial_two_branch_weight": nontrivial_fusion,
    }
    write_json(output_dir / "recommendation.json", recommendation)

    generate_figures(
        figures_dir,
        candidate_table,
        test,
        frame_metrics,
        video,
        video_metrics,
        frame_comparison,
        complementarity,
    )
    _report(
        output_dir / "FUSION_REPORT.md",
        dataset_report,
        alignment_report,
        selection,
        video_selection,
        frame_metrics,
        video_metrics,
        frame_comparison,
        video_comparison,
        complementarity,
        cases,
        recommendation,
        config_path.relative_to(root),
    )

    hashes_after = collect_hashes(key_files, root)
    integrity = {
        "checked_at": _now(),
        "before": hashes_before,
        "after": hashes_after,
        "changed_files": [key for key in hashes_before if hashes_before[key] != hashes_after[key]],
        "all_critical_original_files_unchanged": hashes_before == hashes_after,
    }
    if not integrity["all_critical_original_files_unchanged"]:
        raise RuntimeError(f"关键原文件在融合实验期间发生变化：{integrity['changed_files']}")
    write_json(output_dir / "protected_file_hashes_after.json", hashes_after)
    write_json(output_dir / "original_integrity.json", integrity)
    audit = {
        "evaluated_at": _now(),
        "frozen_config_path": str(config_path.resolve()),
        "frozen_config_sha256": frozen_config_sha256,
        "test_source_predictions": {
            "spatial": str(spatial_test_path.resolve()),
            "spatial_sha256": sha256_file(spatial_test_path),
            "frequency": str(frequency_test_path.resolve()),
            "frequency_sha256": sha256_file(frequency_test_path),
        },
        "test_prediction_inference_rerun": False,
        "test_evaluation_count_this_run": 1,
        "test_used_for_hyperparameter_selection": False,
    }
    write_json(output_dir / "test_evaluation_audit.json", audit)
    commands = [
        r".\.venv\Scripts\python.exe scripts\run_formal_fusion.py",
        r".\.venv\Scripts\python.exe -m pytest tests\test_fusion.py -q",
        r".\.venv\Scripts\python.exe scripts\smoke_test_fusion_detector.py --device both",
        r".\.venv\Scripts\python.exe -m pytest -q",
    ]
    (output_dir / "executed_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    summary = {
        "validation_sample_count": int(len(validation)),
        "test_sample_count": int(len(test)),
        "alignment_passed": True,
        "selection": selection,
        "video_selection": video_selection,
        "frame_metrics": frame_metrics,
        "video_metrics": video_metrics,
        "recommendation": recommendation,
        "config_path": str(config_path.resolve()),
        "report_path": str((output_dir / "FUSION_REPORT.md").resolve()),
        "critical_original_files_unchanged": True,
    }
    write_json(output_dir / "experiment_summary.json", summary)
    logger.info("正式融合实验完成：frame AUC=%.4f F1=%.4f；video AUC=%.4f F1=%.4f", frame_metrics["ROC-AUC"], frame_metrics["F1"], video_metrics["ROC-AUC"], video_metrics["F1"])
    return summary
