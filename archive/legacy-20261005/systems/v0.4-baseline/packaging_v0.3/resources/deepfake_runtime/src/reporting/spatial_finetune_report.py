from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from src.detectors.spatial_finetuning import (
    CachedFaceDataset,
    collect_environment,
    load_verified_baseline,
    load_verified_finetuned_model,
    make_logger,
    metric_payload,
    predict_loader,
    require_cuda,
    validate_and_load_cache,
)


def _evaluate_records(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    amp: bool,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    loader = DataLoader(
        CachedFaceDataset(records, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    started = time.perf_counter()
    y_true, probabilities, indices, _, model_ms = predict_loader(model, loader, device, amp)
    wall_ms = (time.perf_counter() - started) * 1000.0 / max(len(records), 1)
    metrics = metric_payload(y_true, probabilities, threshold)
    metrics["average_model_inference_ms_per_sample"] = model_ms
    metrics["average_cached_pipeline_ms_per_sample"] = wall_ms
    rows = []
    for target, probability, record_index in zip(y_true, probabilities, indices, strict=True):
        record = records[int(record_index)]
        predicted = 1 if probability >= threshold else 0
        rows.append(
            {
                "sample_path": record["relative_frame_path"],
                "sample_id": Path(record["relative_frame_path"]).stem,
                "video_id": Path(record["relative_frame_path"]).stem.split("_f")[0],
                "pair_id": record["pair_id"],
                "true_label": "FAKE" if int(target) == 1 else "REAL",
                "fake_probability": float(probability),
                "predicted_label": "FAKE" if predicted == 1 else "REAL",
                "is_correct": bool(predicted == int(target)),
            }
        )
    return metrics, pd.DataFrame(rows)


def _save_training_plots(history: pd.DataFrame, output_dir: Path) -> list[Path]:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    plots = [
        ("training_loss_curve.png", ["train_loss", "val_loss"], "Loss", "Train and validation loss"),
        ("val_accuracy_curve.png", ["val_accuracy"], "Accuracy", "Validation accuracy"),
        ("val_f1_curve.png", ["val_f1"], "F1", "Validation F1"),
        ("val_roc_auc_curve.png", ["val_roc_auc"], "ROC-AUC", "Validation ROC-AUC"),
    ]
    for filename, columns, ylabel, title in plots:
        path = output_dir / filename
        plt.figure(figsize=(7, 4.5))
        for column in columns:
            plt.plot(history["epoch"], history[column], marker="o", label=column)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)
    return paths


def _save_evaluation_plots(
    gpu_metrics: dict[str, Any],
    gpu_predictions: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    matrix_path = output_dir / "test_confusion_matrix_gpu.png"
    matrix = np.asarray(gpu_metrics["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["REAL", "FAKE"])
    plt.yticks([0, 1], ["REAL", "FAKE"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("GPU finetuned test confusion matrix")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=160)
    plt.close()
    paths.append(matrix_path)

    roc_path = output_dir / "test_roc_curve_gpu.png"
    y_true = gpu_predictions["true_label"].map({"REAL": 0, "FAKE": 1}).to_numpy(dtype=np.int64)
    probabilities = gpu_predictions["fake_probability"].to_numpy(dtype=np.float64)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(5.5, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"GPU finetuned (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("GPU finetuned test ROC")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()
    paths.append(roc_path)

    comparison_path = output_dir / "baseline_gpu_metric_comparison.png"
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(names))
    width = 0.36
    baseline_values = [baseline_metrics[name] for name in names]
    gpu_values = [gpu_metrics[name] for name in names]
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width / 2, baseline_values, width, label="CPU baseline")
    plt.bar(x + width / 2, gpu_values, width, label="GPU finetuned")
    plt.xticks(x, names, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("CPU baseline vs GPU finetuned")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=160)
    plt.close()
    paths.append(comparison_path)
    return paths


def _obvious_overfit(history: pd.DataFrame, best_epoch: int) -> tuple[bool, str]:
    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]
    last_row = history.iloc[-1]
    auc_drop = float(best_row["val_roc_auc"] - last_row["val_roc_auc"])
    loss_rising = float(last_row["val_loss"]) > float(best_row["val_loss"])
    train_loss_falling = float(last_row["train_loss"]) < float(best_row["train_loss"])
    obvious = best_epoch < int(last_row["epoch"]) and auc_drop > 0.10 and loss_rising and train_loss_falling
    reason = f"best到最后一轮val ROC-AUC变化={-auc_drop:+.6f}，val loss变化={float(last_row['val_loss'] - best_row['val_loss']):+.6f}"
    return obvious, reason


def _write_report(
    output_dir: Path,
    model_dir: Path,
    metadata: dict[str, Any],
    environment: dict[str, Any],
    gpu_val_metrics: dict[str, Any],
    baseline_val_metrics: dict[str, Any],
    gpu_test_metrics: dict[str, Any],
    baseline_test_metrics: dict[str, Any],
    original_baseline_metrics: dict[str, Any],
    recommendation: dict[str, Any],
    plot_paths: list[Path],
    train_command: str,
    eval_command: str,
) -> Path:
    gpu_name = metadata["gpu_name"]
    config = json.loads((output_dir / "training_config.json").read_text(encoding="utf-8"))
    history = pd.read_csv(output_dir / "training_history.csv")
    lines = [
        "# 空间域模型 GPU 微调实验报告",
        "",
        "## 1. 环境信息",
        "",
        f"- 操作系统：{environment['platform']}",
        f"- Python：{environment['python_version'].splitlines()[0]}",
        f"- PyTorch：{environment['pytorch_version']}",
        f"- torchvision：{environment['torchvision_version']}",
        f"- CUDA runtime：{environment['torch_cuda_version']}",
        f"- CUDA可用：{environment['cuda_available']}",
        f"- GPU：{gpu_name}",
        f"- 总显存：{environment['gpus'][0]['total_memory_mib']:.2f} MiB",
        f"- 训练设备：{metadata['device']}；AMP：{metadata['amp_enabled']}",
        "",
        "## 2. 数据集与防泄漏",
        "",
        "- 正式清单：`data_all/data_formal/manifest.csv`（200段视频、100对、1000帧）。",
        "- 原视频对级划分保持为 train/val/test = 70/15/15 对，没有重新随机划分图片。",
        f"- 有效缓存样本：train={metadata['train_valid_count']}，val={metadata['val_valid_count']}，test=150。",
        "- pair_id、视频SHA-256、原视频路径和源文件名跨集合交集均为0。",
        f"- 训练过程中读取test图像数：{metadata['test_images_used_during_training']}；test仅在最终评估脚本中推理一次。",
        "",
        "## 3. 模型结构与解冻策略",
        "",
        "- 基线：torchvision MobileNetV3-Small ImageNet预训练模型，REAL=0、FAKE=1双logit分类头。",
        f"- 初始化：加载并校验原CPU最佳权重 SHA-256 `{metadata['source_baseline_sha256']}`。",
        f"- 冻结前部features，解冻末端模块索引 {metadata['unfrozen_feature_indices']}（最后3个主要模块），分类头保持可训练。",
        "- 模型输出为两个logit，因此FAKE概率使用softmax第二类；不把双logit错误地当作单logit sigmoid。",
        "",
        "## 4. 训练参数与数据增强",
        "",
        f"- 参数：`{json.dumps(config['config'], ensure_ascii=False)}`",
        f"- 增强：`{json.dumps(config['augmentation'], ensure_ascii=False)}`",
        "- 验证集和测试集仅使用MobileNetV3官方确定性预处理与ImageNet归一化。",
        "",
        "## 5. 训练过程",
        "",
        f"- 最佳epoch：{metadata['best_epoch']}；实际完成{metadata['epochs_completed']}轮。",
        f"- 总训练耗时：{metadata['training_seconds']:.3f}秒；单epoch平均：{metadata['average_epoch_seconds']:.3f}秒。",
        f"- 峰值GPU显存：{metadata['peak_gpu_memory_mib']:.2f} MiB；最终batch size：{metadata['actual_batch_size']}。",
        f"- 验证集选择阈值：{metadata['classification_threshold']:.12f}（只由最佳epoch验证集F1搜索确定）。",
        f"- 最佳验证ROC-AUC：{gpu_val_metrics['ROC-AUC']:.6f}；验证F1：{gpu_val_metrics['F1']:.6f}。",
        "- 最佳模型首先按验证ROC-AUC选择，完全相同时再比较F1；测试集不参与选模、解冻、学习率或阈值选择。",
        "",
        "## 6. 最终测试结果",
        "",
        "| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | 混淆矩阵 |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| GPU Finetuned | {gpu_test_metrics['Accuracy']:.6f} | {gpu_test_metrics['Precision']:.6f} | {gpu_test_metrics['Recall']:.6f} | {gpu_test_metrics['F1']:.6f} | {gpu_test_metrics['ROC-AUC']:.6f} | `{gpu_test_metrics['confusion_matrix']}` |",
        "",
        f"固定阈值为验证集确定的 `{gpu_test_metrics['classification_threshold']:.12f}`。测试集未进行阈值搜索。",
        "",
        "## 7. CPU/GPU公平对比",
        "",
        "| 模型 | 训练设备 | 骨干状态 | Accuracy | Precision | Recall | F1 | ROC-AUC | 模型推理ms/样本 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
        f"| CPU Baseline（同脚本复算） | CPU | 冻结骨干，仅训练分类头 | {baseline_test_metrics['Accuracy']:.6f} | {baseline_test_metrics['Precision']:.6f} | {baseline_test_metrics['Recall']:.6f} | {baseline_test_metrics['F1']:.6f} | {baseline_test_metrics['ROC-AUC']:.6f} | {baseline_test_metrics['average_model_inference_ms_per_sample']:.3f} |",
        f"| GPU Finetuned | {gpu_name} | 解冻末端3模块 | {gpu_test_metrics['Accuracy']:.6f} | {gpu_test_metrics['Precision']:.6f} | {gpu_test_metrics['Recall']:.6f} | {gpu_test_metrics['F1']:.6f} | {gpu_test_metrics['ROC-AUC']:.6f} | {gpu_test_metrics['average_model_inference_ms_per_sample']:.3f} |",
        "",
        f"- Accuracy变化：{gpu_test_metrics['Accuracy'] - baseline_test_metrics['Accuracy']:+.6f}",
        f"- F1变化：{gpu_test_metrics['F1'] - baseline_test_metrics['F1']:+.6f}",
        f"- ROC-AUC变化：{gpu_test_metrics['ROC-AUC'] - baseline_test_metrics['ROC-AUC']:+.6f}",
        f"- 模型推理耗时变化：{gpu_test_metrics['average_model_inference_ms_per_sample'] - baseline_test_metrics['average_model_inference_ms_per_sample']:+.3f} ms/样本。",
        f"- 原CPU正式结果（未覆盖）：Accuracy={original_baseline_metrics['Accuracy']:.6f}、F1={original_baseline_metrics['F1']:.6f}、ROC-AUC={original_baseline_metrics['ROC-AUC']:.6f}。",
        f"- 验证ROC-AUC：CPU={baseline_val_metrics['ROC-AUC']:.6f}，GPU={gpu_val_metrics['ROC-AUC']:.6f}。",
        f"- 过拟合判定：{recommendation['overfit_reason']}；明显过拟合={recommendation['obvious_overfit']}。",
        "",
        "## 8. 推荐结论",
        "",
        f"- 是否推荐替换CPU基线：**{'是' if recommendation['recommended'] else '否'}**。",
        f"- 判定依据：{recommendation['reason']}",
        "- 原CPU基线模型和结果始终保留，可独立加载复现。",
        "",
        "## 9. 图表与逐样本结果",
        "",
    ]
    lines.extend(f"- `{path.relative_to(output_dir.parent.parent).as_posix()}`" for path in plot_paths)
    lines.extend(
        [
            "- `outputs/spatial_mobilenet_finetuned_gpu/test_predictions.csv`",
            "- `outputs/spatial_mobilenet_finetuned_gpu/test_errors.csv`",
            "- `outputs/spatial_mobilenet_finetuned_gpu/baseline_recomputed_predictions.csv`",
            "",
            "## 10. 局限性",
            "",
            "- 测试集仅15对视频、150帧，帧间相关性较强，指标不等同于更大规模跨数据集泛化能力。",
            "- 人脸缓存由OpenCV Haar检测生成，全部1000帧中27帧未检出；test恰好150/150有效。",
            "- 本实验只微调MobileNetV3后部模块，不包含音频检测，也未训练大型网络。",
            "- GPU与CPU推理速度使用同一缓存、相同batch size和确定性预处理复算，但训练设备与数值精度不同。",
            "",
            "## 11. 实际执行过的关键命令",
            "",
            "- `nvidia-smi`",
            "- `.\\.venv\\Scripts\\python.exe scripts\\check_gpu_env.py --device cuda`",
            "- `.\\.venv\\Scripts\\python.exe -m pip install --force-reinstall --no-deps torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128`",
            "- `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_spatial_finetuning.py tests\\test_spatial_mobilenet.py -q`",
            "- `.\\.venv\\Scripts\\python.exe scripts\\finetune_spatial_mobilenet_gpu.py --device cuda --batch-size 16 --smoke`",
            "- `.\\.venv\\Scripts\\python.exe scripts\\finetune_spatial_mobilenet_gpu.py --device cuda --batch-size 16 --epochs 20 --patience 5 --unfreeze-last 3`",
            "- `.\\.venv\\Scripts\\python.exe scripts\\evaluate_spatial_mobilenet_gpu.py --device cuda`（新模型test仅本次推理）",
            "- `.\\.venv\\Scripts\\python.exe -m compileall -q src scripts tests`",
            "- `.\\.venv\\Scripts\\python.exe -m pytest -q`",
            "- `.\\.venv\\Scripts\\python.exe -m pip check`",
            "",
            "## 12. 复现命令",
            "",
            f"完整GPU训练并最终评估：`{train_command}`",
            "",
            f"仅执行测试评估：`{eval_command}`",
            "",
            f"训练历史共{len(history)}行，完整日志见 `outputs/spatial_mobilenet_finetuned_gpu/training.log`。",
            "",
        ]
    )
    report_path = output_dir / "GPU_FINETUNE_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def evaluate_experiment(
    manifest_path: Path,
    cache_manifest_path: Path,
    baseline_model_dir: Path,
    finetuned_model_dir: Path,
    output_dir: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    device = require_cuda(device_name, amp=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(output_dir / "evaluation.log", "spatial_finetune_evaluation")
    records = validate_and_load_cache(manifest_path, cache_manifest_path)
    val_records = [record for record in records if record["split"] == "val"]
    test_records = [record for record in records if record["split"] == "test"]
    if len(test_records) != 150:
        raise RuntimeError(f"test有效样本应为150，实际为{len(test_records)}；为保护测试口径已停止。")
    metadata = json.loads((finetuned_model_dir / "metadata.json").read_text(encoding="utf-8"))
    batch_size = int(metadata["actual_batch_size"])
    gpu_model, gpu_checkpoint = load_verified_finetuned_model(finetuned_model_dir, device)
    gpu_threshold = float(gpu_checkpoint["classification_threshold"])
    logger.info("GPU微调模型已通过SHA-256校验并加载到%s；固定验证阈值=%.12f", next(gpu_model.parameters()).device, gpu_threshold)
    gpu_val_metrics, _ = _evaluate_records(gpu_model, val_records, device, batch_size, True, gpu_threshold)

    baseline_model, baseline_sha = load_verified_baseline(baseline_model_dir)
    baseline_metadata = json.loads((baseline_model_dir / "metadata.json").read_text(encoding="utf-8"))
    baseline_threshold = float(baseline_metadata["classification_threshold"])
    baseline_model.to(torch.device("cpu")).eval()
    baseline_val_metrics, _ = _evaluate_records(
        baseline_model, val_records, torch.device("cpu"), batch_size, False, baseline_threshold
    )
    logger.info("CPU基线SHA-256校验通过：%s；开始最终test评估（GPU新模型只执行本次推理）。", baseline_sha)

    gpu_test_metrics, gpu_predictions = _evaluate_records(gpu_model, test_records, device, batch_size, True, gpu_threshold)
    baseline_test_metrics, baseline_predictions = _evaluate_records(
        baseline_model, test_records, torch.device("cpu"), batch_size, False, baseline_threshold
    )
    gpu_predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    gpu_predictions[~gpu_predictions["is_correct"]].to_csv(output_dir / "test_errors.csv", index=False, encoding="utf-8-sig")
    baseline_predictions.to_csv(output_dir / "baseline_recomputed_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "test_metrics.json").write_text(json.dumps(gpu_test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "baseline_recomputed_metrics.json").write_text(
        json.dumps(baseline_test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    history = pd.read_csv(output_dir / "training_history.csv")
    obvious_overfit, overfit_reason = _obvious_overfit(history, int(metadata["best_epoch"]))
    validation_better = gpu_val_metrics["ROC-AUC"] > baseline_val_metrics["ROC-AUC"] + 1e-12
    test_better = (
        gpu_test_metrics["ROC-AUC"] > baseline_test_metrics["ROC-AUC"] + 1e-12
        or gpu_test_metrics["F1"] > baseline_test_metrics["F1"] + 1e-12
    )
    recommended = validation_better and test_better and not obvious_overfit
    reason = (
        f"验证ROC-AUC变化{gpu_val_metrics['ROC-AUC'] - baseline_val_metrics['ROC-AUC']:+.6f}，"
        f"测试ROC-AUC变化{gpu_test_metrics['ROC-AUC'] - baseline_test_metrics['ROC-AUC']:+.6f}，"
        f"测试F1变化{gpu_test_metrics['F1'] - baseline_test_metrics['F1']:+.6f}；"
        f"按‘验证更好、测试ROC-AUC或F1提升、无明显过拟合’规则判定。"
    )
    recommendation = {
        "recommended": recommended,
        "validation_better": validation_better,
        "test_auc_or_f1_better": test_better,
        "obvious_overfit": obvious_overfit,
        "overfit_reason": overfit_reason,
        "reason": reason,
    }
    (output_dir / "recommendation.json").write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_paths = _save_training_plots(history, output_dir)
    plot_paths.extend(_save_evaluation_plots(gpu_test_metrics, gpu_predictions, baseline_test_metrics, output_dir))
    environment = collect_environment()
    original_baseline_metrics = json.loads((output_dir.parent / "spatial_mobilenet_formal" / "metrics.json").read_text(encoding="utf-8"))
    train_command = (
        ".\\.venv\\Scripts\\python.exe scripts\\finetune_spatial_mobilenet_gpu.py --device cuda "
        "--batch-size 16 --epochs 20 --patience 5 --unfreeze-last 3 --evaluate-test"
    )
    eval_command = ".\\.venv\\Scripts\\python.exe scripts\\evaluate_spatial_mobilenet_gpu.py --device cuda"
    report_path = _write_report(
        output_dir,
        finetuned_model_dir,
        metadata,
        environment,
        gpu_val_metrics,
        baseline_val_metrics,
        gpu_test_metrics,
        baseline_test_metrics,
        original_baseline_metrics,
        recommendation,
        plot_paths,
        train_command,
        eval_command,
    )
    comparison = {
        "CPU Baseline": baseline_test_metrics,
        "GPU Finetuned": gpu_test_metrics,
        "delta": {
            "Accuracy": gpu_test_metrics["Accuracy"] - baseline_test_metrics["Accuracy"],
            "F1": gpu_test_metrics["F1"] - baseline_test_metrics["F1"],
            "ROC-AUC": gpu_test_metrics["ROC-AUC"] - baseline_test_metrics["ROC-AUC"],
            "model_inference_ms": gpu_test_metrics["average_model_inference_ms_per_sample"]
            - baseline_test_metrics["average_model_inference_ms_per_sample"],
        },
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("最终评估完成：Accuracy=%.6f F1=%.6f ROC-AUC=%.6f matrix=%s", gpu_test_metrics["Accuracy"], gpu_test_metrics["F1"], gpu_test_metrics["ROC-AUC"], gpu_test_metrics["confusion_matrix"])
    logger.info("推荐替换CPU基线：%s；报告：%s", recommended, report_path)
    return {
        "gpu_metrics": gpu_test_metrics,
        "baseline_metrics": baseline_test_metrics,
        "recommendation": recommendation,
        "report_path": str(report_path),
        "plot_paths": [str(path) for path in plot_paths],
    }
