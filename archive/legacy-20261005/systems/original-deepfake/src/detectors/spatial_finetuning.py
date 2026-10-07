from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

from src.detectors.frequency_training import load_manifest, validate_manifest_data

from .spatial_mobilenet import BACKBONE, SpatialMobileNetDetector, sha256_file
from .spatial_training import LABEL_MAPPING, search_threshold


DEFAULT_SEED = 20260712
MODEL_FILENAME = "spatial_mobilenet_finetuned_gpu.pt"


@dataclass(frozen=True)
class FineTuneConfig:
    device: str = "cuda"
    batch_size: int = 16
    epochs: int = 20
    patience: int = 5
    unfreeze_last: int = 3
    backbone_lr: float = 1e-5
    classifier_lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = DEFAULT_SEED
    amp: bool = True
    num_workers: int = 0
    image_size: int = 224
    smoke_train_samples: int | None = None
    smoke_val_samples: int | None = None


TRAIN_AUGMENTATION = {
    "RandomHorizontalFlip": {"p": 0.5},
    "ColorJitter": {"brightness": 0.08, "contrast": 0.08, "saturation": 0.05, "hue": 0.01},
    "RandomRotation": {"degrees": 3, "interpolation": "bilinear"},
    "RandomGaussianBlur": {"p": 0.15, "kernel_size": 3, "sigma": [0.1, 0.5]},
    "normalize": "MobileNet_V3_Small_Weights.DEFAULT.transforms()",
}


def configure_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collect_environment() -> dict[str, Any]:
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(properties.total_memory),
                    "total_memory_mib": round(properties.total_memory / 1024**2, 2),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
        ).stdout
    except OSError as exc:
        nvidia_smi = f"nvidia-smi执行失败：{exc}"
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": str(torch.__version__),
        "torchvision_version": str(__import__("torchvision").__version__),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpus": gpus,
        "nvidia_smi": nvidia_smi,
    }


def require_cuda(device: str, amp: bool) -> torch.device:
    if not device.startswith("cuda"):
        raise RuntimeError("GPU微调实验必须显式使用CUDA设备，禁止回退到CPU冒充GPU训练。")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前PyTorch无法使用CUDA。请在项目.venv中安装CUDA版PyTorch后重试：\n"
            ".\\.venv\\Scripts\\python.exe -m pip install --force-reinstall --no-deps "
            "torch==2.11.0+cu128 torchvision==0.26.0+cu128 "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    selected = torch.device(device)
    if selected.index is not None and selected.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA设备不存在：{device}")
    if not amp:
        raise RuntimeError("本实验要求启用CUDA AMP混合精度。")
    return selected


def make_logger(log_path: Path, name: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_environment(logger: logging.Logger, environment: dict[str, Any], device: torch.device, amp: bool) -> None:
    logger.info("Python版本：%s", environment["python_version"].replace("\n", " "))
    logger.info("PyTorch版本：%s", environment["pytorch_version"])
    logger.info("torchvision版本：%s", environment["torchvision_version"])
    logger.info("PyTorch CUDA版本：%s", environment["torch_cuda_version"])
    logger.info("CUDA是否可用：%s", environment["cuda_available"])
    logger.info("CUDA设备数量：%s", environment["cuda_device_count"])
    for gpu in environment["gpus"]:
        logger.info("GPU %d：%s，总显存 %.2f MiB，计算能力 %s", gpu["index"], gpu["name"], gpu["total_memory_mib"], gpu["compute_capability"])
    logger.info("当前训练设备：%s", device)
    logger.info("AMP混合精度：%s", amp)


def validate_and_load_cache(manifest_path: Path, cache_manifest_path: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    validate_manifest_data(manifest)
    if not cache_manifest_path.is_file():
        raise FileNotFoundError(f"空间域人脸缓存清单不存在：{cache_manifest_path}")
    frame = pd.read_csv(cache_manifest_path, dtype=str, keep_default_na=False)
    required = {"source_path", "relative_frame_path", "split", "label", "target", "pair_id", "crop_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"空间域缓存清单缺少字段：{missing}")
    valid = frame[frame["crop_path"] != ""].copy()
    if valid["relative_frame_path"].duplicated().any():
        raise ValueError("空间域缓存清单存在重复帧路径。")
    if set(valid["split"]) - {"train", "val", "test"}:
        raise ValueError("空间域缓存清单存在未知数据划分。")
    split_pairs = {split: set(group["pair_id"]) for split, group in valid.groupby("split")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_pairs.get(left, set()) & split_pairs.get(right, set())
        if overlap:
            raise ValueError(f"检测到视频对跨集合泄漏：{left}/{right}，{sorted(overlap)[:5]}")
    for path in valid["crop_path"]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"缓存人脸图片不存在：{path}")
    return valid.to_dict(orient="records")


class RandomGaussianBlur:
    def __init__(self, probability: float = 0.15) -> None:
        self.probability = probability
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.blur(image) if random.random() < self.probability else image


def build_transform(train: bool) -> transforms.Compose:
    weights_transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    if not train:
        return transforms.Compose([weights_transform])
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01),
            transforms.RandomRotation(3, interpolation=transforms.InterpolationMode.BILINEAR),
            RandomGaussianBlur(probability=0.15),
            weights_transform,
        ]
    )


class CachedFaceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], train: bool) -> None:
        self.records = records
        self.transform = build_transform(train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        with Image.open(record["crop_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(record["target"]), index


def balanced_subset(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    labels = sorted({int(record["target"]) for record in records})
    per_label = max(1, limit // max(len(labels), 1))
    selected: list[dict[str, Any]] = []
    for label in labels:
        selected.extend([record for record in records if int(record["target"]) == label][:per_label])
    return selected[:limit]


def load_verified_baseline(baseline_model_dir: Path) -> tuple[nn.Module, str]:
    model_path = baseline_model_dir / "spatial_mobilenet_state_dict.pt"
    hash_path = baseline_model_dir / "model_file_sha256.json"
    if not model_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"CPU基线模型或SHA-256记录缺失：{baseline_model_dir}")
    expected = json.loads(hash_path.read_text(encoding="utf-8"))["sha256"]
    actual = sha256_file(model_path)
    if actual != expected:
        raise RuntimeError(f"CPU基线模型SHA-256校验失败：期望{expected}，实际{actual}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model = SpatialMobileNetDetector.build_model()
    model.load_state_dict(state_dict, strict=True)
    return model, actual


def configure_finetuning(model: nn.Module, unfreeze_last: int) -> list[int]:
    feature_modules = list(model.features.children())
    if unfreeze_last <= 0 or unfreeze_last > len(feature_modules):
        raise ValueError(f"unfreeze_last必须位于1到{len(feature_modules)}之间。")
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    unfrozen_indices = list(range(len(feature_modules) - unfreeze_last, len(feature_modules)))
    for index in unfrozen_indices:
        for parameter in feature_modules[index].parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return unfrozen_indices


def set_finetune_train_mode(model: nn.Module, unfrozen_indices: list[int]) -> None:
    model.train()
    for index, module in enumerate(model.features.children()):
        module.train(index in unfrozen_indices)
    model.classifier.train()


def metric_payload(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
        "Specificity": float(tn / max(tn + fp, 1)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": matrix.tolist(),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "sample_count": int(len(y_true)),
        "classification_threshold": float(threshold),
    }


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    indices: list[int] = []
    total_loss = 0.0
    total_count = 0
    model_seconds = 0.0
    with torch.inference_mode():
        for images, targets, record_indices in loader:
            images = images.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets_device) if criterion is not None else None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_seconds += time.perf_counter() - started
            if loss is not None:
                total_loss += float(loss.item()) * len(targets)
                total_count += len(targets)
            output = torch.softmax(logits.float(), dim=1)[:, LABEL_MAPPING["FAKE"]]
            probabilities.extend(output.cpu().numpy().tolist())
            labels.extend(targets.numpy().tolist())
            indices.extend(record_indices.numpy().tolist())
    count = max(len(labels), 1)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
        total_loss / max(total_count, 1),
        model_seconds * 1000.0 / count,
    )


def _checkpoint_payload(
    model: nn.Module,
    config: FineTuneConfig,
    epoch: int,
    best_val_metrics: dict[str, Any],
    baseline_sha256: str,
    unfrozen_indices: list[int],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "label_mapping": LABEL_MAPPING,
        "backbone": BACKBONE,
        "source_baseline_sha256": baseline_sha256,
        "config": asdict(config),
        "best_epoch": epoch,
        "best_val_metrics_at_0_5": best_val_metrics,
        "unfrozen_feature_indices": unfrozen_indices,
        "environment": {key: value for key, value in environment.items() if key != "nvidia_smi"},
    }


def _is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    candidate_auc = candidate["ROC-AUC"] if candidate["ROC-AUC"] is not None else -1.0
    best_auc = best["ROC-AUC"] if best["ROC-AUC"] is not None else -1.0
    return candidate_auc > best_auc + 1e-12 or (
        abs(candidate_auc - best_auc) <= 1e-12 and candidate["F1"] > best["F1"] + 1e-12
    )


def train_finetuned_model(
    manifest_path: Path,
    cache_manifest_path: Path,
    baseline_model_dir: Path,
    model_dir: Path,
    output_dir: Path,
    config: FineTuneConfig,
) -> dict[str, Any]:
    device = require_cuda(config.device, config.amp)
    configure_reproducibility(config.seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    logger = make_logger(log_path, f"spatial_finetune_{model_dir.name}")
    environment = collect_environment()
    log_environment(logger, environment, device, config.amp)
    logger.info("配置：%s", json.dumps(asdict(config), ensure_ascii=False))
    logger.info("训练增强：%s", json.dumps(TRAIN_AUGMENTATION, ensure_ascii=False))
    records = validate_and_load_cache(manifest_path, cache_manifest_path)
    train_records = balanced_subset([record for record in records if record["split"] == "train"], config.smoke_train_samples)
    val_records = balanced_subset([record for record in records if record["split"] == "val"], config.smoke_val_samples)
    if not train_records or not val_records:
        raise RuntimeError("训练集或验证集没有有效缓存样本。")
    logger.info("有效样本：train=%d，val=%d；训练流程未读取test图像。", len(train_records), len(val_records))

    model, baseline_sha256 = load_verified_baseline(baseline_model_dir)
    unfrozen_indices = configure_finetuning(model, config.unfreeze_last)
    model.to(device)
    model_parameter_device = next(model.parameters()).device
    if model_parameter_device.type != "cuda":
        raise RuntimeError(f"模型未位于CUDA设备：{model_parameter_device}")
    logger.info("CPU基线权重SHA-256校验通过：%s", baseline_sha256)
    logger.info("模型参数实际设备：%s；解冻features模块索引：%s", model_parameter_device, unfrozen_indices)

    backbone_parameters = [parameter for parameter in model.features.parameters() if parameter.requires_grad]
    classifier_parameters = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.backbone_lr, "name": "backbone"},
            {"params": classifier_parameters, "lr": config.classifier_lr, "name": "classifier"},
        ],
        weight_decay=config.weight_decay,
    )
    counts = torch.bincount(torch.tensor([int(record["target"]) for record in train_records]), minlength=2).float()
    class_weights = (counts.sum() / (2.0 * counts.clamp_min(1.0))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        CachedFaceDataset(train_records, train=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        CachedFaceDataset(val_records, train=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = model_dir / MODEL_FILENAME
    total_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    first_batch_verified = False

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        set_finetune_train_mode(model, unfrozen_indices)
        total_loss = 0.0
        total_count = 0
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if not first_batch_verified:
                if images.device.type != "cuda" or targets.device.type != "cuda":
                    raise RuntimeError(f"首批输入未位于CUDA：images={images.device}, targets={targets.device}")
                logger.info("首批输入实际设备：images=%s，targets=%s；AMP=%s", images.device, targets.device, config.amp)
                first_batch_verified = True
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(targets)
            total_count += len(targets)

        y_val, p_val, _, val_loss, _ = predict_loader(model, val_loader, device, config.amp, criterion)
        val_metrics = metric_payload(y_val, p_val, threshold=0.5)
        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["Accuracy"],
            "val_precision": val_metrics["Precision"],
            "val_recall": val_metrics["Recall"],
            "val_f1": val_metrics["F1"],
            "val_roc_auc": val_metrics["ROC-AUC"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "classifier_lr": optimizer.param_groups[1]["lr"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f val_accuracy=%.6f val_precision=%.6f "
            "val_recall=%.6f val_f1=%.6f val_roc_auc=%.6f lr_backbone=%.2e lr_classifier=%.2e time=%.2fs",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_accuracy"],
            row["val_precision"],
            row["val_recall"],
            row["val_f1"],
            row["val_roc_auc"],
            row["backbone_lr"],
            row["classifier_lr"],
            epoch_seconds,
        )
        if _is_better(val_metrics, best_metrics):
            best_metrics = val_metrics
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                _checkpoint_payload(model, config, epoch, val_metrics, baseline_sha256, unfrozen_indices, environment),
                checkpoint_path,
            )
            logger.info("保存新的验证集最佳模型：epoch=%d，ROC-AUC=%.6f，F1=%.6f", epoch, val_metrics["ROC-AUC"], val_metrics["F1"])
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                logger.info("早停：连续%d轮验证集ROC-AUC/F1未改善。", config.patience)
                break

    total_seconds = time.perf_counter() - total_started
    if not checkpoint_path.is_file() or best_metrics is None:
        raise RuntimeError("训练结束但未生成验证集最佳模型。")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    y_val, p_val, indices, _, val_model_ms = predict_loader(model, val_loader, device, config.amp)
    threshold, threshold_table = search_threshold(y_val, p_val)
    threshold_metrics = metric_payload(y_val, p_val, threshold)
    checkpoint["classification_threshold"] = threshold
    checkpoint["best_val_metrics_at_selected_threshold"] = threshold_metrics
    torch.save(checkpoint, checkpoint_path)

    validation_rows = []
    for target, probability, record_index in zip(y_val, p_val, indices, strict=True):
        record = val_records[int(record_index)]
        validation_rows.append(
            {
                "sample_path": record["relative_frame_path"],
                "video_id": Path(record["relative_frame_path"]).stem.split("_f")[0],
                "true_label": "FAKE" if int(target) == 1 else "REAL",
                "fake_probability": float(probability),
                "predicted_label": "FAKE" if probability >= threshold else "REAL",
            }
        )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    threshold_table.to_csv(output_dir / "threshold_search_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(validation_rows).to_csv(output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "training_config.json").write_text(
        json.dumps({"config": asdict(config), "augmentation": TRAIN_AUGMENTATION}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "model_name": "SpatialMobileNetDetector GPU Finetuned",
        "backbone": BACKBONE,
        "source_baseline_path": str((baseline_model_dir / "spatial_mobilenet_state_dict.pt").resolve()),
        "source_baseline_sha256": baseline_sha256,
        "model_path": str(checkpoint_path.resolve()),
        "model_sha256": sha256_file(checkpoint_path),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "classification_threshold": threshold,
        "best_val_metrics_at_0_5": best_metrics,
        "best_val_metrics_at_selected_threshold": threshold_metrics,
        "unfrozen_feature_indices": unfrozen_indices,
        "train_valid_count": len(train_records),
        "val_valid_count": len(val_records),
        "test_images_used_during_training": 0,
        "training_seconds": total_seconds,
        "average_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "validation_model_inference_ms_per_sample": val_model_ms,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "actual_batch_size": config.batch_size,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "amp_enabled": config.amp,
        "random_seed": config.seed,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "model_file_sha256.json").write_text(
        json.dumps({"file": checkpoint_path.name, "sha256": metadata["model_sha256"]}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("训练完成：best_epoch=%d，val_threshold=%.12f，总耗时=%.2fs，峰值显存=%.2f MiB", best_epoch, threshold, total_seconds, metadata["peak_gpu_memory_mib"])
    return {"metadata": metadata, "history": history, "model_path": str(checkpoint_path), "log_path": str(log_path)}


def load_verified_finetuned_model(model_dir: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    metadata_path = model_dir / "metadata.json"
    hash_path = model_dir / "model_file_sha256.json"
    model_path = model_dir / MODEL_FILENAME
    if not metadata_path.is_file() or not hash_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"GPU微调模型工件不完整：{model_dir}")
    expected = json.loads(hash_path.read_text(encoding="utf-8"))["sha256"]
    actual = sha256_file(model_path)
    if actual != expected:
        raise RuntimeError(f"GPU微调模型SHA-256校验失败：期望{expected}，实际{actual}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = SpatialMobileNetDetector.build_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint
