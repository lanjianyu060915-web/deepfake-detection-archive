"""Verify archived crop/pad dev results without reading media or retraining.

Print a fresh summary to stdout. Original execution records are read-only.
EER is recomputed by a standalone threshold sweep, without project metrics.
"""

# ruff: noqa: S101, PT018, PLR0915
# This evidence checker deliberately uses linear assertions; -O is rejected below.

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from deepfake_detection_archive.data.manifest import (
    file_sha256,
    load_manifest,
    manifest_hash,
)
from deepfake_detection_archive.models.checkpoint import load_checkpoint
from deepfake_detection_archive.tracking import code_identity, object_hash


ROOT = Path(__file__).resolve().parents[4]
CONTROL = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/experiments/vif-pilot-20261007"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def independent_eer(rows: list[dict[str, Any]]) -> float:
    """Group ties and interpolate FAR/FRR across a score>=threshold sweep."""
    counts = Counter(row["label"] for row in rows)
    assert set(counts) == {0, 1}
    assert all(math.isfinite(row["score"]) for row in rows)
    points = [(1.0, 0.0)]
    for threshold in sorted({row["score"] for row in rows}, reverse=True):
        far = sum(r["label"] == 1 and r["score"] < threshold for r in rows)
        frr = sum(r["label"] == 0 and r["score"] >= threshold for r in rows)
        points.append((far / counts[1], frr / counts[0]))
    for (far0, frr0), (far1, frr1) in pairwise(points):
        if far0 == frr0:
            return far0
        if far1 <= frr1:
            weight = (far0 - frr0) / ((far0 - frr0) - (far1 - frr1))
            return far0 + weight * (far1 - far0)
    raise AssertionError("no EER crossing")


def differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    assert left.keys() == right.keys()
    changed: list[str] = []
    for key, value in left.items():
        if isinstance(value, dict):
            changed.extend(f"{key}.{k}" for k in differences(value, right[key]))
        elif value != right[key]:
            changed.append(key)
    return sorted(changed)


def resources(directory: Path, stage: str) -> dict[str, Any]:
    execution = read(directory / f"{stage}.execution.json")
    assert execution["returncode"] == 0
    assert not (directory / f"{stage}.gpu.stderr").read_text().strip()
    with (directory / f"{stage}.gpu.csv").open() as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    assert rows
    peaks = {
        key: max(float(row[key].split()[0]) for row in rows)
        for key in (
            "utilization.gpu [%]",
            "memory.used [MiB]",
            "temperature.gpu",
            "power.draw [W]",
        )
    }
    stderr = (directory / f"{stage}.stderr").read_text()
    rss_line = next(
        line for line in stderr.splitlines() if "Maximum resident set size" in line
    )
    return {
        "seconds": execution["seconds"],
        "returncode": execution["returncode"],
        "telemetry_samples": len(rows),
        "device_sampled_peaks": peaks,
        "process_peak_rss_kib": int(rss_line.rsplit(":", 1)[1]),
    }


def verify_run(directory: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    config = read(directory / "train-config.json")
    experiment = read(directory / "experiment.json")
    run = read(directory / "training-run.json")
    history = read(directory / "training-history.json")
    preflight = read(directory / "preflight.json")
    diagnostics = read(directory / "dev-diagnostics.json")
    report = read(directory / "infer-report.json")
    infer = read(directory / "infer-config.json")
    output = Path(config["output_dir"])
    records = load_manifest(Path(config["manifest"]))
    dev = {r.sample_id: r for r in records if r.split == "dev"}
    selected = [r for r in records if r.split in {"train", "dev"}]
    assert len(selected) == 870 and len(dev) == 154
    assert Counter(r.label for r in dev.values()) == {0: 77, 1: 77}
    assert experiment["status"] == run["status"] == "complete"
    assert experiment["holdout_accessed"] is diagnostics["holdout_accessed"] is False
    assert preflight["config_sha256"] == file_sha256(directory / "train-config.json")
    assert preflight["uv_lock_sha256"] == file_sha256(ROOT / "uv.lock")
    assert preflight["encoder_sha256"] == file_sha256(
        Path(config["encoder"]["checkpoint"])
    )
    assert preflight["encoder_sha256"] == config["encoder"]["expected_sha256"]
    assert preflight["manifest_sha256"] == file_sha256(Path(config["manifest"]))
    assert run["provenance"] == experiment["provenance"]
    provenance = run["provenance"]
    assert provenance["manifest_sha256"] == manifest_hash(records)
    assert provenance["data_sha256"] == manifest_hash(selected)
    assert provenance["sources_sha256"] == preflight["source_sha256"]
    assert provenance["sources_sha256"] == file_sha256(Path(config["sources"]))
    assert provenance["encoder_sha256"] == object_hash(config["encoder"])
    assert provenance["code_sha256"] == code_identity()["code_sha256"]
    assert preflight["code"] == experiment["code"]
    assert experiment["code"]["code_sha256"] == provenance["code_sha256"]
    assert experiment["config_sha256"] == object_hash(experiment["config"])
    assert run["config_sha256"] == object_hash(run["config"])
    assert run["provenance_sha256"] == object_hash(provenance)
    checkpoint_sha = file_sha256(output / "training/best.pt")
    assert checkpoint_sha == experiment["checkpoint_sha256"]
    assert checkpoint_sha == run["checkpoint"]["sha256"]
    assert (
        checkpoint_sha == infer["checkpoint_sha256"] == diagnostics["checkpoint_sha256"]
    )
    initial_sha = file_sha256(output / "initial.pt")
    assert initial_sha == experiment["initial_checkpoint"]["sha256"]
    assert initial_sha == provenance["weights_sha256"]
    initial, _ = load_checkpoint(output / "initial.pt", expected_sha256=initial_sha)
    best, _ = load_checkpoint(
        output / "training/best.pt", expected_sha256=checkpoint_sha
    )
    initial_parameters = dict(initial.named_parameters())
    best_parameters = dict(best.named_parameters())
    assert initial_parameters.keys() == best_parameters.keys()
    updates = {
        name: {
            "elements": tensor.numel(),
            "changed_elements": int(
                torch.count_nonzero(tensor != best_parameters[name])
            ),
            "max_abs_delta": float(
                (tensor - best_parameters[name]).detach().abs().max()
            ),
        }
        for name, tensor in initial_parameters.items()
    }
    assert all(item["changed_elements"] > 0 for item in updates.values())
    assert len(history) == run["epochs_completed"]
    assert all(h["train_samples"] == 716 and h["dev_samples"] == 154 for h in history)
    assert all(
        h["optimizer_updates"] == 12 and h["skipped_updates"] == 0 for h in history
    )
    assert min(h["dev_eer"] for h in history) == run["best_dev_eer"]
    assert history[run["best_epoch"] - 1]["dev_eer"] == run["best_dev_eer"]
    scores = [
        json.loads(line)
        for line in (directory / "dev-scores.jsonl").read_text().splitlines()
    ]
    assert len(scores) == len({r["sample_id"] for r in scores}) == len(dev)
    assert {r["sample_id"] for r in scores} == set(dev)
    assert all(r["status"] == "ok" and r["error_code"] is None for r in scores)
    for score in scores:
        for field in ("label", "generator", "source", "modality", "content_type"):
            assert score[field] == getattr(dev[score["sample_id"]], field)
    dev_manifest = load_manifest(Path(infer["manifest"]))
    assert {r.sample_id for r in dev_manifest} == set(dev)
    assert all(r.split == "dev" for r in dev_manifest)
    assert diagnostics["dev_manifest_sha256"] == file_sha256(Path(infer["manifest"]))
    scores_sha = file_sha256(directory / "dev-scores.jsonl")
    assert scores_sha == diagnostics["scores_sha256"]
    assert scores_sha == file_sha256(output / "dev-inference/scores.jsonl")
    eer = independent_eer(scores)
    assert math.isclose(eer, report["fullset_eer"], abs_tol=1e-12)
    assert math.isclose(eer, diagnostics["overall"]["fullset_eer"], abs_tol=1e-12)
    # Inference uses fp32; checkpoint selection used bf16. Compare, do not assume.
    assert math.isclose(eer, run["best_dev_eer"], abs_tol=1e-12)
    assert report["failure_count"] == 0 and report["success_count"] == 154
    assert read(directory / "decode-failures.json") == []
    audit = read(directory / "data-audit.json")
    assert audit["audit"]["passed"] and audit["split_errors"] == []
    assert set(audit["audit"]["trainable_ids"]) == {r.sample_id for r in selected}
    return {
        "dev_eer": eer,
        "dev_count": len(scores),
        "failures": 0,
        "known_independent_groups": report["group_count"],
        "best_epoch": run["best_epoch"],
        "epochs_completed": run["epochs_completed"],
        "optimizer_updates": sum(h["optimizer_updates"] for h in history),
        "skipped_updates": 0,
        "checkpoint_sha256": checkpoint_sha,
        "initial_checkpoint_sha256": initial_sha,
        "parameter_updates": updates,
        "scores_sha256": scores_sha,
        "archived_evidence_sha256": {
            path.name: file_sha256(path)
            for path in sorted(directory.iterdir())
            if path.is_file()
        },
        "provenance": provenance,
        "execution_code": experiment["code"],
        "resources": {
            "train": resources(directory, "train"),
            "infer": resources(directory, "infer"),
            "head_training_seconds": run["seconds"],
            "head_peak_cuda_allocated_bytes": run["peak_cuda_allocated_bytes"],
            "head_peak_cuda_reserved_bytes": run["peak_cuda_reserved_bytes"],
        },
    }, initial.state_dict()


def main() -> None:
    if not __debug__:
        raise RuntimeError("verification requires assertions; do not use python -O")
    prereg = read(CONTROL / "preregistration.json")
    assert prereg["baseline_seed42_seen"] and not prereg["formal_candidate"]
    assert file_sha256(CONTROL / "source/run_control.py") == prereg["script_sha256"]
    assert (
        file_sha256(BASELINE / "training/run_pilot.py") == prereg["base_runner_sha256"]
    )
    assert (
        file_sha256(CONTROL / "configs/comparison.json") == prereg["comparison_sha256"]
    )
    assert (
        file_sha256(ROOT / "docs/plans/vif-shortcut-controls.md")
        == prereg["plan_sha256"]
    )
    runs = []
    for seed in (42, 43):
        crop_dir = CONTROL / f"training/seed{seed}"
        pad_dir = BASELINE / f"training/seed{seed}"
        crop_config = read(crop_dir / "train-config.json")
        pad_config = read(pad_dir / "train-config.json")
        changed = differences(crop_config, pad_config)
        assert changed == sorted(prereg["allowed_configuration_changes"])
        assert crop_config["video"]["resize_mode"] == "center_crop"
        assert pad_config["video"]["resize_mode"] == "pad"
        config_name = f"train-seed{seed}.json"
        config_sha = file_sha256(CONTROL / "configs" / config_name)
        assert config_sha == prereg["training_config_sha256"][config_name]
        assert config_sha == file_sha256(crop_dir / "train-config.json")
        crop, initial_crop = verify_run(crop_dir)
        pad, initial_pad = verify_run(pad_dir)
        assert initial_crop.keys() == initial_pad.keys()
        assert all(torch.equal(initial_crop[k], initial_pad[k]) for k in initial_crop)
        assert crop["provenance"] == pad["provenance"]
        assert crop["provenance"]["manifest_sha256"] == prereg["manifest_sha256"]
        runs.append(
            {
                "seed": seed,
                "changed_config_fields": changed,
                "same_initial_parameters": True,
                "crop": crop,
                "pad": pad,
                "crop_minus_pad_eer": crop["dev_eer"] - pad["dev_eer"],
            }
        )
    assert runs[0]["crop_minus_pad_eer"] > 0 > runs[1]["crop_minus_pad_eer"]
    print(
        json.dumps(
            {
                "verification": "PASS",
                "verifier_script_sha256": file_sha256(Path(__file__)),
                "eer_method": "independent tie-group threshold sweep with linear interpolation",
                "scope": "Archived train/dev artifacts and local checkpoints; no media read or training",
                "formal_preregistered_candidate": False,
                "baseline_seed42_seen_before_materialization": True,
                "holdout_media_or_scores_read_by_verifier": False,
                "historical_holdout_evidence": "runner/source inspection and archived flags; no OS access trace",
                "formal_promotion": False,
                "decision": "No promotion: crop effect changes direction across seeds; exploratory only",
                "independent_group_confidence_interval": None,
                "runs": runs,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
