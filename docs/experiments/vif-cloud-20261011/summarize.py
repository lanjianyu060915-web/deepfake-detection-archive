"""Recompute the Issue #10 comparison from complete local development scores."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepfake_detection_archive.contracts import ScoreRecord, load_scores
from deepfake_detection_archive.data.manifest import file_sha256, load_manifest
from deepfake_detection_archive.evaluation.comparison import compare_records
from deepfake_detection_archive.evaluation.report import evaluate_records


DOC = Path(__file__).resolve().parent
ASSETS = DOC.parents[2] / "outputs/experiments/vif-cloud-20261011"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def evidence(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def main() -> None:
    manifest = ASSETS / "control/hunyuan_geometry_duration.jsonl"
    dev = {r.sample_id: r for r in load_manifest(manifest) if r.split == "dev"}
    metadata_path = ASSETS / "control/hunyuan_geometry_duration-scores.jsonl"
    metadata_rows = [
        json.loads(line) for line in metadata_path.read_text().splitlines()
    ]
    baselines: dict[str, list[ScoreRecord]] = {}
    for name in ("width_only", "numeric_media"):
        scores = []
        for row in metadata_rows:
            if row["model"] != name:
                continue
            record = dev[row["sample_id"]]
            if row["label"] != record.label:
                raise ValueError("metadata score label differs from manifest")
            scores.append(
                ScoreRecord(
                    sample_id=record.sample_id,
                    modality=record.modality,
                    score=row["score"],
                    status="ok",
                    elapsed_seconds=0.0,  # Original metadata scorer did not time rows.
                    label=record.label,
                    group_id=record.group_id,
                    source=record.source,
                    content_type=record.content_type,
                    generator=record.generator,
                )
            )
        evaluate_records(scores, expected_sample_ids=list(dev))
        baselines[name] = scores
    summary: dict[str, Any] = {
        "scope": "HunyuanVideo single-source controlled development comparison",
        "score_direction": "higher_is_fake; visual mean frame logits",
        "selection_split": "dev",
        "holdout_probe_decode_scoring": False,
        "competition_approved": False,
        "assets_root": str(ASSETS.resolve()),
        "manifest": evidence(manifest),
        "sources": evidence(ASSETS / "vif-panda-research/sources.jsonl"),
        "metadata_scores": evidence(metadata_path),
        "data_preparation": read(ASSETS / "control/summary.json"),
        "data_isolation_audit": evidence(ASSETS / "control/data-isolation-audit.json"),
        "encoder_restoration": read(ASSETS / "resnet18-a1/evidence/restoration.json"),
        "profile": read(ASSETS / "profile.json"),
        "runs": {},
    }
    for seed in (42, 43):
        output = ASSETS / f"seed{seed}"
        experiment = read(output / "experiment.json")
        run = read(output / "training/run.json")
        history = read(output / "training/history.json")
        failures = read(output / "decode_failures.json")
        if experiment["status"] != "complete" or run["status"] != "complete":
            raise ValueError("training is incomplete")
        if experiment["holdout_accessed"] or failures:
            raise ValueError("holdout accessed or sample preparation failed")
        checkpoint = evidence(output / "training/best.pt")
        if not (
            checkpoint["sha256"]
            == experiment["checkpoint_sha256"]
            == run["checkpoint"]["sha256"]
        ):
            raise ValueError("selected checkpoint hash mismatch")
        scores = load_scores(output / "dev-scores.jsonl")
        report = evaluate_records(scores, expected_sample_ids=list(dev))
        if (
            report.fullset_eer is None
            or abs(report.fullset_eer - run["best_dev_eer"]) >= 1e-12
        ):
            raise ValueError("recomputed complete-dev EER differs from selected model")
        comparisons = {}
        for name, baseline in baselines.items():
            paired = compare_records(baseline, scores)
            comparisons[name] = {
                "baseline_eer": paired.baseline.fullset_eer,
                "candidate_eer": paired.candidate.fullset_eer,
                "candidate_minus_baseline_eer": paired.fullset_delta_eer,
                "common_sample_count": paired.common_sample_count,
                "common_coverage": paired.common_coverage,
                "excluded_sample_ids": paired.excluded_sample_ids,
            }
        evaluation = asdict(report)
        if evaluation["successful_subset_metrics"]:
            evaluation["successful_subset_metrics"].pop("det_curve", None)
        summary["runs"][f"seed{seed}"] = {
            "config": evidence(DOC / f"seed{seed}.json"),
            "code": experiment["code"],
            "provenance": experiment["provenance"],
            "train_count": experiment["train_count"],
            "dev_count": experiment["dev_count"],
            "best_epoch": run["best_epoch"],
            "epochs_completed": run["epochs_completed"],
            "stop_reason": run["stop_reason"],
            "checkpoint": checkpoint,
            "scores": evidence(output / "dev-scores.jsonl"),
            "decode_failure_count": len(failures),
            "skipped_optimizer_updates": sum(r["skipped_updates"] for r in history),
            "experiment_seconds": experiment["seconds"],
            "training_seconds": run["seconds"],
            "median_training_samples_per_second": statistics.median(
                r["train_samples_per_second"] for r in history
            ),
            "peak_cuda_allocated_bytes": run["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": run["peak_cuda_reserved_bytes"],
            "dev_scoring_seconds_excluding_decode": read(
                output / "dev-evaluation.json"
            )["seconds"],
            "evaluation": evaluation,
            "comparisons": comparisons,
        }
    (DOC / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: v["evaluation"]["fullset_eer"] for k, v in summary["runs"].items()}
        )
    )


if __name__ == "__main__":
    main()
