"""核对本轮四个完整实验并汇总 dev；不访问 holdout/eval，不混入旧实验."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

from deepfake_detection_archive.contracts import load_scores
from deepfake_detection_archive.data.manifest import (
    file_sha256,
    load_manifest,
    manifest_hash,
)
from deepfake_detection_archive.evaluation.report import (
    EvaluationReport,
    evaluate_records,
)


RUNS = ("logmel-seed42", "logmel-seed43", "hubert-seed42", "hubert-seed43")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def compact_report(report: EvaluationReport) -> dict[str, Any]:
    """逐阈值曲线留在原始产物，Git摘要仅保留指标与失败分母."""
    result = asdict(report)
    if result["successful_subset_metrics"] is not None:
        result["successful_subset_metrics"].pop("det_curve")
    return result


def summarize(task_root: Path, output_dir: Path) -> None:  # noqa: PLR0912 -- one experiment
    task_root = task_root.resolve()
    runs = {name: task_root / "runs" / name for name in RUNS}
    experiments = {
        name: read_json(path / "experiment.json") for name, path in runs.items()
    }
    trainers = {
        name: read_json(path / "training/run.json") for name, path in runs.items()
    }
    if any(
        r["status"] != "complete" for r in [*experiments.values(), *trainers.values()]
    ):
        raise ValueError("四个实验及训练过程必须全部 complete 才能汇总")
    manifest = task_root / "data/main/manifest-training.jsonl"
    samples = load_manifest(manifest)
    dev = {r.sample_id: r for r in samples if r.split == "dev"}
    expected_counts = {
        (0, "human"): 2548,
        **{(1, f"A{i:02d}"): 3716 for i in range(1, 7)},
    }
    if Counter((r.label, r.generator) for r in dev.values()) != expected_counts:
        raise ValueError("dev 必须覆盖 24,844 条：2,548 real，各 A01–A06 3,716 fake")
    digest = manifest_hash(samples)
    sources = artifact(task_root / "data/main/sources.jsonl")
    histories, results = {}, {}
    for name, path in runs.items():
        experiment, trainer = experiments[name], trainers[name]
        if (
            experiment["holdout_accessed"]
            or experiment["provenance"] != trainer["provenance"]
        ):
            raise ValueError(f"{name}: holdout/provenance 与本轮协议不符")
        provenance = experiment["provenance"]
        if (
            not provenance["manifest_sha256"] == provenance["data_sha256"] == digest
            or provenance["sources_sha256"] != sources["sha256"]
        ):
            raise ValueError(f"{name}: 实验数据与当前训练清单不符")
        if trainer["config"]["seed"] != int(name[-2:]):
            raise ValueError(f"{name}: seed 不符")
        scores = load_scores(path / "dev-scores.jsonl")
        overall = evaluate_records(scores, expected_sample_ids=list(dev))
        fields = (
            "label",
            "source",
            "generator",
            "group_id",
            "modality",
            "content_type",
        )
        for score in scores:
            record = dev[score.sample_id]
            if any(getattr(score, field) != getattr(record, field) for field in fields):
                raise ValueError(f"{name}: score 元数据不符 {score.sample_id}")
        if overall.fullset_eer is not None and not math.isclose(
            overall.fullset_eer, trainer["best_dev_eer"], rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"{name}: 重算 EER 与训练最佳值不符")
        history = read_json(path / "training/history.json")
        if len(history) != trainer["epochs_completed"] or [
            r["epoch"] for r in history
        ] != list(range(1, len(history) + 1)):
            raise ValueError(f"{name}: 训练曲线不完整")
        checkpoint = artifact(path / "training/best.pt")
        if (
            checkpoint["sha256"] != trainer["checkpoint"]["sha256"]
            or checkpoint["sha256"] != experiment["checkpoint_sha256"]
        ):
            raise ValueError(f"{name}: 最佳 checkpoint SHA256 不符")
        histories[name] = history
        attacks = {}
        for attack in (f"A{i:02d}" for i in range(1, 7)):
            selected = [s for s in scores if s.label == 0 or s.generator == attack]
            attacks[attack] = compact_report(evaluate_records(selected))
        keys = [
            "best_epoch",
            "epochs_completed",
            "stop_reason",
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
        ]
        results[name] = {
            "status": "complete",
            "overall": compact_report(overall),
            "per_attack_with_all_real": attacks,
            "train_count": experiment["train_count"],
            "dev_count": experiment["dev_count"],
            **{key: trainer[key] for key in keys},
            "skipped_updates": sum(r["skipped_updates"] for r in history),
            "experiment_seconds": experiment["seconds"],
            "training_seconds": trainer["seconds"],
            "median_train_samples_per_second": median(
                r["train_samples_per_second"] for r in history
            ),
            "dev_scoring_seconds": sum(s.elapsed_seconds for s in scores),
            "resource_scope": "训练器显存峰值与缓存输入吞吐；不代表端到端推理资源开销",
            "score_scope": "冻结 HuBERT BF16 提取特征后的 FP32 检测头重评分"
            if name.startswith("hubert")
            else "缓存解码波形上的 FP32 log-mel 模型重评分",
            "training_config": trainer["config"],
            "code_at_run": experiment["code"],
            "provenance": provenance,
            "artifacts": {
                "checkpoint": checkpoint,
                "dev-scores.jsonl": artifact(path / "dev-scores.jsonl"),
            },
        }
    summary = {
        "experiment": "asv-cloud-20261011",
        "status": "complete",
        "task_root": str(task_root),
        "scope": "同一 train/dev、两种模型各两个 seed；dev 选最佳轮次；仅已知 A01–A06 语音生成器，不是独立验收或端到端离线推理成绩",
        "score_direction": "higher_is_fake",
        "score_type": "continuous_logits",
        "holdout_accessed": False,
        "eval_accessed": False,
        "end_to_end": False,
        "manifest": artifact(manifest),
        "manifest_record_sha256": digest,
        "sources": sources,
        "summarizer": artifact(Path(__file__)),
        "runs": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in (
        ("summary.json", summary),
        ("training-history.json", histories),
    ):
        (output_dir / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    summarize(arguments.task_root, arguments.output_dir)
