from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import DetectorError
from .frequency_stat import (
    FEATURE_DIMENSION,
    FEATURE_NAMES,
    MODEL_NAME,
    MODEL_VERSION,
    FrequencyStatDetector,
)


RANDOM_STATE = 20260712
REQUIRED_COLUMNS = {
    "split",
    "pair_id",
    "label",
    "source_repo",
    "source_filename",
    "local_video",
    "video_sha256",
    "frame_count",
    "frame_paths",
}
SPLITS = ("train", "val", "test")
LABEL_TO_TARGET = {"real": 0, "fake": 1}
TARGET_TO_LABEL = {0: "REAL", 1: "FAKE"}


class DatasetValidationError(RuntimeError):
    """数据清单或数据划分不满足训练约束。"""


@dataclass(frozen=True)
class FrameSample:
    split: str
    pair_id: str
    label: str
    target: int
    frame_path: Path
    relative_frame_path: str
    local_video: str
    local_video_path: Path
    source_filename: str
    video_sha256: str


@dataclass(frozen=True)
class ManifestData:
    manifest_path: Path
    dataset_root: Path
    manifest_sha256: str
    samples: tuple[FrameSample, ...]
    video_row_counts: dict[str, int]
    pair_counts: dict[str, int]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def label_to_target(label: str) -> int:
    normalized = str(label).strip().lower()
    if normalized not in LABEL_TO_TARGET:
        raise DatasetValidationError(f"未知标签：{label}；仅支持real和fake。")
    return LABEL_TO_TARGET[normalized]


def _resolve_dataset_path(dataset_root: Path, raw_path: str, field_name: str) -> Path:
    normalized = Path(raw_path.replace("\\", "/"))
    resolved = (dataset_root / normalized).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise DatasetValidationError(f"{field_name}超出数据集目录：{raw_path}") from exc
    return resolved


def load_manifest(manifest_path: str | Path) -> ManifestData:
    path = Path(manifest_path).resolve()
    if not path.exists():
        raise DatasetValidationError(f"数据清单不存在：{path}")
    dataset_root = path.parent.resolve()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing_columns = sorted(REQUIRED_COLUMNS - columns)
            if missing_columns:
                raise DatasetValidationError(f"数据清单缺少列：{', '.join(missing_columns)}")
            rows = list(reader)
    except UnicodeError as exc:
        raise DatasetValidationError(f"数据清单编码无法读取：{path}") from exc

    samples: list[FrameSample] = []
    video_row_counts = Counter()
    pairs_by_split: dict[str, set[str]] = defaultdict(set)
    for row_number, row in enumerate(rows, start=2):
        split = str(row["split"]).strip().lower()
        if split not in SPLITS:
            raise DatasetValidationError(f"第{row_number}行包含未知split：{row['split']}")
        label = str(row["label"]).strip().lower()
        target = label_to_target(label)
        pair_id = str(row["pair_id"]).strip()
        if not pair_id:
            raise DatasetValidationError(f"第{row_number}行pair_id为空。")
        raw_frame_paths = str(row["frame_paths"])
        frame_paths = raw_frame_paths.split("|") if raw_frame_paths else []
        if not frame_paths or any(not value.strip() for value in frame_paths):
            raise DatasetValidationError(f"第{row_number}行frame_paths格式无效。")
        try:
            declared_frame_count = int(str(row["frame_count"]).strip())
        except ValueError as exc:
            raise DatasetValidationError(f"第{row_number}行frame_count不是整数。") from exc
        if declared_frame_count != len(frame_paths):
            raise DatasetValidationError(
                f"第{row_number}行frame_count={declared_frame_count}，"
                f"但frame_paths实际包含{len(frame_paths)}项。"
            )

        local_video = str(row["local_video"]).strip()
        local_video_path = _resolve_dataset_path(dataset_root, local_video, "local_video")
        video_row_counts[split] += 1
        pairs_by_split[split].add(pair_id)
        for raw_frame_path in frame_paths:
            relative_frame_path = raw_frame_path.strip()
            samples.append(
                FrameSample(
                    split=split,
                    pair_id=pair_id,
                    label=label,
                    target=target,
                    frame_path=_resolve_dataset_path(
                        dataset_root,
                        relative_frame_path,
                        "frame_paths",
                    ),
                    relative_frame_path=relative_frame_path.replace("\\", "/"),
                    local_video=local_video.replace("\\", "/"),
                    local_video_path=local_video_path,
                    source_filename=str(row["source_filename"]).strip(),
                    video_sha256=str(row["video_sha256"]).strip().lower(),
                )
            )

    return ManifestData(
        manifest_path=path,
        dataset_root=dataset_root,
        manifest_sha256=file_sha256(path),
        samples=tuple(samples),
        video_row_counts={split: int(video_row_counts[split]) for split in SPLITS},
        pair_counts={split: len(pairs_by_split[split]) for split in SPLITS},
    )


def _cross_split_values(samples: tuple[FrameSample, ...], attribute: str) -> list[str]:
    split_sets: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        split_sets[str(getattr(sample, attribute))].add(sample.split)
    return sorted(value for value, splits in split_sets.items() if len(splits) > 1)


def _label_conflicts(samples: tuple[FrameSample, ...], attribute: str) -> list[str]:
    targets: dict[str, set[int]] = defaultdict(set)
    for sample in samples:
        targets[str(getattr(sample, attribute))].add(sample.target)
    return sorted(value for value, labels in targets.items() if len(labels) > 1)


def validate_manifest_data(data: ManifestData, check_files: bool = True) -> dict[str, Any]:
    samples = data.samples
    path_counts = Counter(str(sample.frame_path) for sample in samples)
    duplicate_entries = sorted(path for path, count in path_counts.items() if count > 1)
    report: dict[str, Any] = {
        "frame_path_cross_split": _cross_split_values(samples, "frame_path"),
        "pair_id_cross_split": _cross_split_values(samples, "pair_id"),
        "video_sha256_cross_split": _cross_split_values(samples, "video_sha256"),
        "local_video_cross_split": _cross_split_values(samples, "local_video"),
        "source_filename_cross_split": _cross_split_values(samples, "source_filename"),
        "duplicate_frame_entries": duplicate_entries,
        "frame_path_label_conflicts": _label_conflicts(samples, "frame_path"),
        "video_sha256_label_conflicts": _label_conflicts(samples, "video_sha256"),
        "local_video_label_conflicts": _label_conflicts(samples, "local_video"),
        "missing_frame_paths": [],
        "missing_video_paths": [],
    }
    if check_files:
        report["missing_frame_paths"] = sorted(
            str(sample.frame_path) for sample in samples if not sample.frame_path.is_file()
        )
        report["missing_video_paths"] = sorted(
            {str(sample.local_video_path) for sample in samples if not sample.local_video_path.is_file()}
        )

    split_label_counts = {
        split: {
            label: sum(1 for sample in samples if sample.split == split and sample.label == label)
            for label in LABEL_TO_TARGET
        }
        for split in SPLITS
    }
    report["split_label_frame_counts"] = split_label_counts
    report["passed"] = all(
        not value
        for key, value in report.items()
        if key not in {"passed", "split_label_frame_counts"}
    ) and all(count > 0 for counts in split_label_counts.values() for count in counts.values())
    if not report["passed"]:
        raise DatasetValidationError(
            "数据划分检查失败，存在跨集合泄漏、标签冲突、重复项、缺失文件或空类别。"
        )
    return report


def extract_feature_matrix(
    detector: FrequencyStatDetector,
    samples: list[FrameSample],
) -> tuple[np.ndarray, np.ndarray, list[FrameSample], list[dict[str, str]]]:
    vectors: list[np.ndarray] = []
    targets: list[int] = []
    successful: list[FrameSample] = []
    failures: list[dict[str, str]] = []
    for sample in samples:
        try:
            vector = detector.extract_features(sample.frame_path)
        except DetectorError as exc:
            failures.append(
                {
                    "stage": "feature_extraction",
                    "split": sample.split,
                    "pair_id": sample.pair_id,
                    "label": TARGET_TO_LABEL[sample.target],
                    "frame_path": sample.relative_frame_path,
                    "reason": str(exc),
                }
            )
            continue
        vectors.append(vector)
        targets.append(sample.target)
        successful.append(sample)
    if not vectors:
        raise DatasetValidationError("没有可用于训练或评估的有效人脸样本。")
    return np.vstack(vectors), np.asarray(targets, dtype=np.int64), successful, failures


def search_best_threshold(
    y_true: np.ndarray,
    fake_probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], fake_probabilities)))
    rows: list[dict[str, float]] = []
    for threshold in candidates:
        predicted = (fake_probabilities >= threshold).astype(np.int64)
        rows.append(
            {
                "threshold": float(threshold),
                "F1": float(f1_score(y_true, predicted, zero_division=0)),
                "Accuracy": float(accuracy_score(y_true, predicted)),
                "Precision": float(precision_score(y_true, predicted, zero_division=0)),
                "Recall": float(recall_score(y_true, predicted, zero_division=0)),
                "distance_to_0_5": abs(float(threshold) - 0.5),
            }
        )
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best = table.sort_values(
        ["F1", "Accuracy", "distance_to_0_5", "threshold"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return float(best["threshold"]), table


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _feature_category(name: str) -> str:
    if name.startswith("fft_radial"):
        return "FFT径向分桶"
    if name.startswith("fft_"):
        return "FFT能量关系"
    if name.startswith("block_dct"):
        return "分块DCT"
    if name.startswith("dct_"):
        return "全局DCT"
    return "灰度与纹理统计"


def _write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    columns = ["stage", "split", "pair_id", "label", "frame_path", "reason"]
    pd.DataFrame(failures, columns=columns).drop_duplicates().to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def _write_integrity_report(
    report: dict[str, Any], model_dir: Path, output_dir: Path
) -> None:
    _write_json(model_dir / "data_integrity_report.json", report)
    _write_json(output_dir / "data_integrity_report.json", report)


def _write_sample_groups(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["split", "pair_id", "frame_path", "true_label", "predicted_label", "fake_probability"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def train_frequency_model(
    manifest_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_manifest(manifest_path)
    leakage_report = validate_manifest_data(data)
    detector = FrequencyStatDetector()
    train_samples = [sample for sample in data.samples if sample.split == "train"]
    val_samples = [sample for sample in data.samples if sample.split == "val"]
    test_samples = [sample for sample in data.samples if sample.split == "test"]

    x_train, y_train, successful_train, train_failures = extract_feature_matrix(detector, train_samples)
    x_val, y_val, successful_val, val_failures = extract_feature_matrix(detector, val_samples)
    _, _, successful_test, test_integrity_failures = extract_feature_matrix(detector, test_samples)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise DatasetValidationError("train和val都必须同时包含REAL与FAKE有效样本。")

    classifier = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000,
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)
    val_probabilities = pipeline.predict_proba(x_val)[:, int(np.flatnonzero(pipeline.classes_ == 1)[0])]
    threshold, threshold_table = search_best_threshold(y_val, val_probabilities)

    model_path = model_dir / "frequency_stat_detector.joblib"
    joblib.dump(pipeline, model_path)
    model_hash = file_sha256(model_path)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest_frame_counts = {
        split: sum(1 for sample in data.samples if sample.split == split) for split in SPLITS
    }
    demo_only = data.dataset_root.name.lower() == "data_demo"
    integrity_report = {
        "manifest_path": str(data.manifest_path),
        "manifest_sha256": data.manifest_sha256,
        "manifest_row_count": sum(data.video_row_counts.values()),
        "pair_counts": data.pair_counts,
        "video_row_counts": data.video_row_counts,
        "frame_counts": manifest_frame_counts,
        "label_mapping": {"real": 0, "fake": 1},
        "leakage_check": leakage_report,
        "missing_or_unreadable_frame_count": 0,
        "face_or_feature_failure_count": len(train_failures) + len(val_failures) + len(test_integrity_failures),
        "face_or_feature_success_count": len(successful_train) + len(successful_val) + len(successful_test),
        "failures": train_failures + val_failures + test_integrity_failures,
        "passed": bool(leakage_report["passed"]),
        "demo_only": demo_only,
        "not_for_final_metrics": demo_only,
    }
    _write_integrity_report(integrity_report, model_dir, output_dir)
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "created_at": created_at,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "feature_dimension": FEATURE_DIMENSION,
        "train_sample_count": len(successful_train),
        "val_sample_count": len(successful_val),
        "test_sample_count": len(successful_test),
        "manifest_frame_counts": manifest_frame_counts,
        "video_row_counts": data.video_row_counts,
        "pair_counts": data.pair_counts,
        "random_state": RANDOM_STATE,
        "classification_threshold": threshold,
        "manifest_sha256": data.manifest_sha256,
        "model_file_sha256": model_hash,
        "model_file": model_path.name,
        "demo_only": demo_only,
        "not_for_final_metrics": demo_only,
    }
    _write_json(model_dir / "metadata.json", metadata)
    _write_json(
        model_dir / "feature_schema.json",
        {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "feature_dimension": FEATURE_DIMENSION,
            "features": [
                {"index": index, "name": name, "category": _feature_category(name)}
                for index, name in enumerate(FEATURE_NAMES)
            ],
        },
    )

    val_predicted = (val_probabilities >= threshold).astype(np.int64)
    training_summary = {
        "created_at": created_at,
        "manifest_path": str(data.manifest_path),
        "manifest_sha256": data.manifest_sha256,
        "label_mapping": {"real": 0, "fake": 1},
        "frame_paths_storage": "以竖线|分隔的Windows相对路径",
        "manifest_frame_counts": manifest_frame_counts,
        "usable_sample_counts": {
            "train": len(successful_train),
            "val": len(successful_val),
            "test": len(test_samples),
        },
        "failed_feature_extractions": len(train_failures) + len(val_failures) + len(test_integrity_failures),
        "leakage_check": leakage_report,
        "pipeline": {
            "steps": ["StandardScaler", "LogisticRegression"],
            "standard_scaler_params": pipeline.named_steps["scaler"].get_params(deep=False),
            "logistic_regression_params": classifier.get_params(deep=False),
        },
        "threshold_selection": {
            "primary_metric": "F1",
            "tie_breakers": ["Accuracy降序", "与0.5距离升序", "阈值升序"],
            "selected_threshold": threshold,
            "validation_F1": float(f1_score(y_val, val_predicted, zero_division=0)),
        },
        "demo_only": demo_only,
        "not_for_final_metrics": demo_only,
    }
    _write_json(model_dir / "training_summary.json", training_summary)
    threshold_table.to_csv(output_dir / "threshold_search.csv", index=False, encoding="utf-8-sig")
    _write_failures(
        output_dir / "failed_samples.csv",
        train_failures + val_failures + test_integrity_failures,
    )

    statistics = []
    for index, name in enumerate(FEATURE_NAMES):
        values = x_train[:, index]
        statistics.append(
            {
                "index": index,
                "name": name,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    _write_json(
        output_dir / "feature_statistics.json",
        {
            "source_split": "train",
            "sample_count": len(successful_train),
            "feature_dimension": FEATURE_DIMENSION,
            "features": statistics,
            "demo_only": demo_only,
            "not_for_final_metrics": demo_only,
        },
    )
    return {
        "metadata": metadata,
        "training_summary": training_summary,
        "model_path": str(model_path),
        "failures": train_failures + val_failures,
    }


def _metric_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = float(true_negative / (true_negative + false_positive)) if true_negative + false_positive else None
    try:
        roc_auc: float | None = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        roc_auc = None
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": specificity,
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": roc_auc,
        "confusion_matrix": matrix.tolist(),
        "REAL_sample_count": int(np.sum(y_true == 0)),
        "FAKE_sample_count": int(np.sum(y_true == 1)),
        "false_positive_count": int(matrix[0, 1]),
        "false_negative_count": int(matrix[1, 0]),
    }


def evaluate_frequency_model(
    manifest_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    data = load_manifest(manifest_path)
    leakage_report = validate_manifest_data(data)
    detector = FrequencyStatDetector(model_path=model_dir / "frequency_stat_detector.joblib")
    detector.load_model()

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    test_samples = [sample for sample in data.samples if sample.split == "test"]
    for sample in test_samples:
        try:
            result = detector.predict(sample.frame_path)
        except DetectorError as exc:
            failures.append(
                {
                    "stage": "test_inference",
                    "split": sample.split,
                    "pair_id": sample.pair_id,
                    "label": TARGET_TO_LABEL[sample.target],
                    "frame_path": sample.relative_frame_path,
                    "reason": str(exc),
                }
            )
            continue
        predictions.append(
            {
                "split": sample.split,
                "pair_id": sample.pair_id,
                "frame_path": sample.relative_frame_path,
                "video_sha256": sample.video_sha256,
                "true_label": TARGET_TO_LABEL[sample.target],
                "predicted_label": result["label"],
                "fake_probability": result["fake_probability"],
                "threshold": result["threshold"],
                "inference_ms": result["inference_ms"],
                "feature_extraction_ms": result["feature_extraction_ms"],
                "prediction_ms": result["prediction_ms"],
            }
        )
    if not predictions:
        raise DatasetValidationError("test没有成功完成推理的样本。")

    frame = pd.DataFrame(predictions)
    frame.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    y_true = frame["true_label"].map({"REAL": 0, "FAKE": 1}).to_numpy(dtype=np.int64)
    y_pred = frame["predicted_label"].map({"REAL": 0, "FAKE": 1}).to_numpy(dtype=np.int64)
    probabilities = frame["fake_probability"].to_numpy(dtype=np.float64)
    metrics = _metric_payload(y_true, y_pred, probabilities)
    metrics.update(
        {
            "classification_threshold": detector.threshold,
            "face_detection_failure_count": len(failures),
            "face_detection_success_count": len(frame),
            "average_inference_ms": float(frame["inference_ms"].mean()),
            "average_feature_extraction_ms": float(frame["feature_extraction_ms"].mean()),
            "average_prediction_ms": float(frame["prediction_ms"].mean()),
            "evaluated_test_sample_count": len(frame),
            "manifest_test_sample_count": len(test_samples),
            "leakage_check_passed": bool(leakage_report["passed"]),
            "demo_only": data.dataset_root.name.lower() == "data_demo",
            "not_for_final_metrics": data.dataset_root.name.lower() == "data_demo",
        }
    )
    _write_json(output_dir / "metrics.json", metrics)

    failure_path = output_dir / "failed_samples.csv"
    existing_failures: list[dict[str, str]] = []
    if failure_path.exists():
        existing = pd.read_csv(failure_path, dtype=str, keep_default_na=False)
        existing_failures = existing.to_dict(orient="records")
    _write_failures(failure_path, existing_failures + failures)

    false_positive_rows = [
        row for row in predictions if row["true_label"] == "REAL" and row["predicted_label"] == "FAKE"
    ]
    false_negative_rows = [
        row for row in predictions if row["true_label"] == "FAKE" and row["predicted_label"] == "REAL"
    ]
    _write_sample_groups(output_dir / "false_positives.csv", false_positive_rows)
    _write_sample_groups(output_dir / "false_negatives.csv", false_negative_rows)
    summary_lines = [
        "# FrequencyStatDetector正式评估摘要",
        "",
        f"- 测试样本数：{len(frame)}（REAL={metrics['REAL_sample_count']}，FAKE={metrics['FAKE_sample_count']}）",
        f"- 阈值：{metrics['classification_threshold']:.12f}",
        f"- Accuracy：{metrics['Accuracy']:.6f}",
        f"- Precision：{metrics['Precision']:.6f}",
        f"- Recall：{metrics['Recall']:.6f}",
        f"- Specificity：{metrics['Specificity']}",
        f"- F1：{metrics['F1']:.6f}",
        f"- ROC-AUC：{metrics['ROC-AUC']}",
        f"- 混淆矩阵：{metrics['confusion_matrix']}",
        f"- 人脸/特征失败数：{len(failures)}",
        f"- 平均特征提取耗时：{metrics['average_feature_extraction_ms']:.3f} ms",
        f"- 平均预测耗时：{metrics['average_prediction_ms']:.3f} ms",
        f"- 误报数：{metrics['false_positive_count']}，漏报数：{metrics['false_negative_count']}",
    ]
    (output_dir / "evaluation_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], labels=["REAL", "FAKE"])
    axis.set_yticks([0, 1], labels=["REAL", "FAKE"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("FrequencyStatDetector confusion matrix (demo only)")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    return {"metrics": metrics, "predictions": predictions, "failures": failures}
