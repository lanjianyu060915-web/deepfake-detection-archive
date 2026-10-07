from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

from src.detectors.frequency_training import LABEL_TO_TARGET, TARGET_TO_LABEL, load_manifest, validate_manifest_data
from src.preprocessing import OpenCVFaceDetector, PreprocessConfig, PreprocessingError, expand_face_box

from .spatial_mobilenet import (
    BACKBONE,
    MODEL_NAME,
    WEIGHTS_FILENAME,
    WEIGHTS_SHA256,
    SpatialMobileNetDetector,
    official_weights_path,
    sha256_file,
)

RANDOM_STATE = 20260712
LABEL_MAPPING = {"REAL": 0, "FAKE": 1}


def seed_everything() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


def _safe_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def build_face_cache(manifest_path: str | Path, cache_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_manifest(manifest_path)
    validate_manifest_data(data)
    cache_dir = Path(cache_dir).resolve()
    image_dir = cache_dir / "faces"
    image_dir.mkdir(parents=True, exist_ok=True)
    detector = OpenCVFaceDetector()
    config = PreprocessConfig(face_size=(224, 224))
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, sample in enumerate(data.samples, start=1):
        record: dict[str, Any] = {
            "source_path": str(sample.frame_path),
            "relative_frame_path": sample.relative_frame_path,
            "split": sample.split,
            "label": sample.label,
            "target": sample.target,
            "pair_id": sample.pair_id,
            "face_bbox": "",
            "crop_path": "",
            "failure_reason": "",
        }
        try:
            raw = np.fromfile(sample.frame_path, dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
            if image is None:
                raise PreprocessingError(f"无法读取图片，文件可能已损坏：{sample.frame_path.name}")
            box = detector.detect_largest(image)
            if box is None:
                raise PreprocessingError(f"图片中未检测到人脸：{sample.frame_path.name}")
            x1, y1, x2, y2 = expand_face_box(box, image.shape[1], image.shape[0], config.face_expand_ratio)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                raise PreprocessingError(f"人脸裁剪为空：{sample.frame_path.name}")
            crop = cv2.resize(crop, config.face_size, interpolation=cv2.INTER_AREA)
            crop_path = image_dir / f"{_safe_key(sample.relative_frame_path)}.png"
            ok, encoded = cv2.imencode(".png", crop)
            if not ok:
                raise PreprocessingError(f"无法编码人脸裁剪：{sample.frame_path.name}")
            encoded.tofile(crop_path)
            record["face_bbox"] = json.dumps([x1, y1, x2 - x1, y2 - y1])
            record["crop_path"] = str(crop_path)
            records.append(record)
        except (OSError, PreprocessingError) as exc:
            record["failure_reason"] = str(exc)
            failures.append(record)
        if index % 100 == 0:
            print(f"人脸缓存进度：{index}/{len(data.samples)}")
    fields = list(records[0].keys()) if records else ["source_path", "relative_frame_path", "split", "label", "target", "pair_id", "face_bbox", "crop_path", "failure_reason"]
    pd.DataFrame(records + failures, columns=fields).to_csv(cache_dir / "cache_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "manifest_path": str(data.manifest_path),
        "cache_dir": str(cache_dir),
        "total_count": len(data.samples),
        "success_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "label_mapping": LABEL_MAPPING,
        "face_size": [224, 224],
    }
    (cache_dir / "cache_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return records, summary


class FaceCacheDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], train: bool) -> None:
        self.records = records
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                weights.transforms(),
            ]
            if train
            else [weights.transforms()]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record["crop_path"]) as image:
            image = image.convert("RGB")
            return self.transform(image), int(record["target"])


def search_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(np.concatenate((np.linspace(0.0, 1.0, 1001), probabilities)))
    rows = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(np.int64)
        rows.append({"threshold": float(threshold), "F1": f1_score(y_true, pred, zero_division=0), "Accuracy": accuracy_score(y_true, pred), "distance_to_0_5": abs(float(threshold) - 0.5)})
    table = pd.DataFrame(rows)
    best = table.sort_values(["F1", "Accuracy", "distance_to_0_5", "threshold"], ascending=[False, False, True, True]).iloc[0]
    return float(best["threshold"]), table


def _predict_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module | None = None) -> tuple[np.ndarray, np.ndarray, float, float]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    total_loss = 0.0
    total_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            if criterion is not None:
                total_loss += float(criterion(logits, targets).item()) * len(targets)
                total_count += len(targets)
            output = torch.softmax(logits, dim=1)[:, LABEL_MAPPING["FAKE"]]
            probabilities.extend(output.cpu().numpy().tolist())
            labels.extend(targets.numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64), (time.perf_counter() - started) * 1000.0 / max(len(labels), 1), total_loss / max(total_count, 1)


def train_spatial_model(manifest_path: str | Path, cache_dir: str | Path, model_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    seed_everything()
    model_dir, output_dir = Path(model_dir).resolve(), Path(output_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, cache_summary = build_face_cache(manifest_path, cache_dir)
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    test_records = [r for r in records if r["split"] == "test"]
    model = SpatialMobileNetDetector.build_model(use_imagenet_pretrained=True)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    weights = torch.bincount(torch.tensor([int(r["target"]) for r in train_records]), minlength=2).float()
    class_weights = weights.sum() / (2.0 * weights.clamp_min(1.0))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(FaceCacheDataset(train_records, True), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(FaceCacheDataset(val_records, False), batch_size=32, shuffle=False, num_workers=0)
    history: list[dict[str, Any]] = []
    best_f1, best_epoch, stale = -1.0, 0, 0
    best_state = None
    best_val_labels = None
    best_val_probabilities = None
    for epoch in range(1, 13):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
        y_val, p_val, _, val_loss = _predict_loader(model, val_loader, criterion)
        pred_val = (p_val >= 0.5).astype(np.int64)
        row = {"epoch": epoch, "train_loss": total_loss / max(len(train_records), 1), "val_loss": val_loss, "val_accuracy": accuracy_score(y_val, pred_val), "val_precision": precision_score(y_val, pred_val, zero_division=0), "val_recall": recall_score(y_val, pred_val, zero_division=0), "val_f1": f1_score(y_val, pred_val, zero_division=0), "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"epoch={epoch} train_loss={row['train_loss']:.5f} val_f1={row['val_f1']:.5f}")
        if row["val_f1"] > best_f1:
            best_f1, best_epoch, stale = row["val_f1"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_labels = y_val.copy()
            best_val_probabilities = p_val.copy()
        else:
            stale += 1
            if stale >= 3:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model_path = model_dir / "spatial_mobilenet_state_dict.pt"
    torch.save({"state_dict": model.state_dict(), "label_mapping": LABEL_MAPPING, "backbone": BACKBONE}, model_path)
    assert best_val_labels is not None and best_val_probabilities is not None
    threshold, threshold_table = search_threshold(best_val_labels, best_val_probabilities)
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    threshold_table.to_csv(output_dir / "threshold_search.csv", index=False, encoding="utf-8-sig")
    metadata = {"model_name": MODEL_NAME, "backbone": BACKBONE, "pretraining": "ImageNet预训练MobileNetV3-Small，使用data_formal进行二分类迁移学习", "official_weights_path": str(official_weights_path()), "official_weights_sha256": WEIGHTS_SHA256, "model_file_sha256": sha256_file(model_path), "label_mapping": LABEL_MAPPING, "best_epoch": best_epoch, "classification_threshold": threshold, "train_valid_count": len(train_records), "val_valid_count": len(val_records), "test_valid_count": len(test_records), "face_detection_failure_count": cache_summary["failure_count"], "device": "cpu", "random_state": RANDOM_STATE, "retrained_on_data_formal": True}
    (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "label_mapping.json").write_text(json.dumps(LABEL_MAPPING, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "model_file_sha256.json").write_text(json.dumps({"file": model_path.name, "sha256": metadata["model_file_sha256"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "training_summary.json").write_text(json.dumps({"metadata": metadata, "cache_summary": cache_summary, "best_epoch": best_epoch, "threshold": threshold}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "metadata": metadata, "history": history, "records": records}


def _metric_payload(y_true: np.ndarray, pred: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred, zero_division=0), "Recall": recall_score(y_true, pred, zero_division=0), "Specificity": tn / max(tn + fp, 1), "F1": f1_score(y_true, pred, zero_division=0), "ROC-AUC": roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) == 2 else None, "confusion_matrix": matrix.tolist(), "REAL_sample_count": int((y_true == 0).sum()), "FAKE_sample_count": int((y_true == 1).sum()), "face_detection_success_count": len(y_true), "face_detection_failure_count": 0, "false_positive_count": int(fp), "false_negative_count": int(fn)}


def evaluate_spatial_model(manifest_path: str | Path, cache_dir: str | Path, model_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_manifest(manifest_path)
    validate_manifest_data(data)
    model = SpatialMobileNetDetector(model_path=Path(model_dir) / "spatial_mobilenet_state_dict.pt", threshold=float(json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))["classification_threshold"]))
    model.load_model()
    records = pd.read_csv(Path(cache_dir) / "cache_manifest.csv", dtype=str, keep_default_na=False).to_dict(orient="records")
    test_records = [r for r in records if r["split"] == "test" and r["crop_path"]]
    predictions, failures = [], []
    for index, record in enumerate(test_records, start=1):
        try:
            result = model.predict(record["source_path"])
            predictions.append({"split": "test", "pair_id": record["pair_id"], "frame_path": record["relative_frame_path"], "true_label": TARGET_TO_LABEL[int(record["target"])], "predicted_label": result["label"], "fake_probability": result["fake_probability"], "real_probability": result["real_probability"], "threshold": result["threshold"], "inference_ms": result["inference_ms"], "preprocessing_ms": result["preprocessing_ms"], "model_inference_ms": result["model_inference_ms"], "face_bbox": result["face_bbox"]})
        except Exception as exc:
            failures.append({"stage": "test_inference", "split": "test", "pair_id": record["pair_id"], "label": TARGET_TO_LABEL[int(record["target"])], "frame_path": record["relative_frame_path"], "reason": str(exc)})
        if index % 25 == 0:
            print(f"空间模型评估进度：{index}/{len(test_records)}")
    frame = pd.DataFrame(predictions)
    y_true = frame["true_label"].map({"REAL": 0, "FAKE": 1}).to_numpy(dtype=np.int64)
    probabilities = frame["fake_probability"].to_numpy(dtype=np.float64)
    pred = frame["predicted_label"].map({"REAL": 0, "FAKE": 1}).to_numpy(dtype=np.int64)
    metrics = _metric_payload(y_true, pred, probabilities)
    metrics.update({"classification_threshold": model.threshold, "average_inference_ms": float(frame["inference_ms"].mean()), "average_preprocessing_ms": float(frame["preprocessing_ms"].mean()), "average_model_inference_ms": float(frame["model_inference_ms"].mean()), "evaluated_test_sample_count": len(frame), "face_detection_success_count": len(frame), "face_detection_failure_count": len(failures), "demo_only": False, "not_for_final_metrics": False})
    frame.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(output_dir / "failed_samples.csv", index=False, encoding="utf-8-sig")
    frame[(frame.true_label == "REAL") & (frame.predicted_label == "FAKE")].to_csv(output_dir / "false_positives.csv", index=False, encoding="utf-8-sig")
    frame[(frame.true_label == "FAKE") & (frame.predicted_label == "REAL")].to_csv(output_dir / "false_negatives.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    frequency_path = Path("outputs/frequency_stat_formal/predictions.csv").resolve()
    if frequency_path.is_file():
        frequency = pd.read_csv(frequency_path, dtype={"frame_path": str})
        joined = frame.merge(frequency[["frame_path", "fake_probability", "predicted_label"]], on="frame_path", how="inner", suffixes=("_spatial", "_frequency"))
        if len(joined) >= 2:
            spatial_correct = joined["predicted_label_spatial"] == joined["true_label"]
            frequency_correct = joined["predicted_label_frequency"] == joined["true_label"]
            comparison = {"common_test_sample_count": len(joined), "probability_pearson": float(joined["fake_probability_spatial"].corr(joined["fake_probability_frequency"], method="pearson")), "probability_spearman": float(joined["fake_probability_spatial"].corr(joined["fake_probability_frequency"], method="spearman")), "both_correct_count": int((spatial_correct & frequency_correct).sum()), "spatial_only_correct_count": int((spatial_correct & ~frequency_correct).sum()), "frequency_only_correct_count": int((~spatial_correct & frequency_correct).sum()), "both_wrong_count": int((~spatial_correct & ~frequency_correct).sum())}
            metrics["frequency_comparison"] = comparison
            (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics["confusion_matrix"])
    plt.figure(figsize=(5, 4)); plt.imshow(matrix, cmap="Blues"); plt.colorbar(); plt.xticks([0, 1], ["REAL", "FAKE"]); plt.yticks([0, 1], ["REAL", "FAKE"]); plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("SpatialMobileNetDetector");
    for row in range(2):
        for column in range(2): plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(output_dir / "confusion_matrix.png", dpi=150); plt.close()
    sample = frame.groupby("true_label", group_keys=False).head(3)
    sample.to_json(output_dir / "six_sample_predictions.json", orient="records", force_ascii=False, indent=2)
    (output_dir / "evaluation_summary.md").write_text("\n".join(["# SpatialMobileNetDetector正式评估", "", f"- Accuracy：{metrics['Accuracy']:.6f}", f"- F1：{metrics['F1']:.6f}", f"- ROC-AUC：{metrics['ROC-AUC']}", f"- 阈值：{metrics['classification_threshold']:.12f}", f"- 混淆矩阵：{metrics['confusion_matrix']}", f"- 误报：{metrics['false_positive_count']}，漏报：{metrics['false_negative_count']}", "- 本模型使用ImageNet预训练MobileNetV3-Small骨干，并在data_formal上进行了二分类迁移学习。", "- 本轮未进行双模型融合。", ""]) , encoding="utf-8")
    return {"metrics": metrics, "predictions": predictions, "failures": failures}
