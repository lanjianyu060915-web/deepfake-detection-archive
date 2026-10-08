"""Run one frozen ViF exploratory seed and score only its development split."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.contracts import load_scores
from deepfake_detection_archive.data.manifest import file_sha256, load_manifest
from deepfake_detection_archive.evaluation.report import evaluate_records
from deepfake_detection_archive.tracking import atomic_json, code_identity, utc_now


ROOT = Path(__file__).resolve().parents[4]
PLAN = ROOT / "outputs/experiment-planning/vif-pilot"
DOC = Path(__file__).parent
MANIFEST_SHA = "f3b004bb94174d02899b9a8763c3619da121f90f53f919e5884e171c716d6638"


def execute(command: list[str], directory: Path, prefix: str) -> None:
    """Capture command completion and owned telemetry, leaving no monitor process."""
    started = utc_now()
    clock = time.monotonic()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    with (
        (directory / f"{prefix}.stdout").open("x") as stdout,
        (directory / f"{prefix}.stderr").open("x") as stderr,
        (directory / f"{prefix}.gpu.csv").open("x") as telemetry,
        (directory / f"{prefix}.gpu.stderr").open("x") as monitor_errors,
    ):
        monitor = subprocess.Popen(  # noqa: S603 -- installed local GPU telemetry tool
            [
                shutil.which("nvidia-smi") or "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw",
                "--format=csv",
                "--loop=1",
            ],
            stdout=telemetry,
            stderr=monitor_errors,
        )
        try:
            result = subprocess.run(  # noqa: S603
                ["/usr/bin/time", "-v", *command],
                cwd=ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
    atomic_json(
        directory / f"{prefix}.execution.json",
        {
            "command": ["/usr/bin/time", "-v", *command],
            "cwd": str(ROOT),
            "started_at": started,
            "finished_at": utc_now(),
            "seconds": time.monotonic() - clock,
            "returncode": result.returncode,
            "telemetry_returncode": monitor.returncode,
        },
    )
    if result.returncode:
        raise RuntimeError(f"{prefix} failed: inspect captured stderr")


def main() -> None:  # noqa: PLR0912, PLR0915 -- linear auditable execution stages
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--audit-commit", required=True)
    args = parser.parse_args()
    config_path = PLAN / f"train-seed{args.seed}.json"
    config = json.loads(config_path.read_text())
    manifest = Path(config["manifest"])
    if file_sha256(manifest) != MANIFEST_SHA:
        raise ValueError("locked manifest hash differs")
    prereg = json.loads(
        (ROOT / "docs/research/source/vif-pilot/preregistration.json").read_text()
    )
    if file_sha256(config_path) != prereg["training_config_sha256"][config_path.name]:
        raise ValueError("frozen training configuration differs")
    records = load_manifest(manifest)
    if Counter(r.split for r in records) != {"train": 716, "dev": 154, "holdout": 154}:
        raise ValueError("locked split counts differ")
    dev = [record for record in records if record.split == "dev"]
    if Counter(record.label for record in dev) != {0: 77, 1: 77}:
        raise ValueError("development class counts differ")
    encoder = Path(config["encoder"]["checkpoint"])
    if file_sha256(encoder) != prereg["weights_sha256"]:
        raise ValueError("encoder hash differs")
    evidence = DOC / f"seed{args.seed}"
    evidence.mkdir(exist_ok=False)
    out = Path(config["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("training output must be empty")
    atomic_json(
        evidence / "preflight.json",
        {
            "created_at": utc_now(),
            "audit_commit": args.audit_commit,
            "code": code_identity(),
            "script_sha256": file_sha256(Path(__file__)),
            "uv_lock_sha256": file_sha256(ROOT / "uv.lock"),
            "config_sha256": file_sha256(config_path),
            "manifest_sha256": file_sha256(manifest),
            "source_sha256": file_sha256(Path(config["sources"])),
            "encoder_sha256": file_sha256(encoder),
            "python": sys.version,
            "installed_distributions": sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
            ),
            "scope": "Exploratory; known aspect-ratio shortcut; no holdout or CI.",
        },
    )
    (evidence / "train-config.json").write_bytes(config_path.read_bytes())
    execute(
        [
            sys.executable,
            "-m",
            "deepfake_detection_archive",
            "train",
            "--config",
            str(config_path),
        ],
        evidence,
        "train",
    )
    run = json.loads((out / "training/run.json").read_text())
    checkpoint = out / "training/best.pt"
    checkpoint_sha = file_sha256(checkpoint)
    if run["status"] != "complete" or checkpoint_sha != run["checkpoint"]["sha256"]:
        raise ValueError("training/checkpoint verification failed")
    dev_manifest = out / "dev.jsonl"
    with dev_manifest.open("x") as stream:
        for record in dev:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    infer = InferenceConfig(
        manifest=str(dev_manifest),
        data_root=config["data_root"],
        output_dir=str(out / "dev-inference"),
        modality="video",
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha,
        device="cuda",
        model_kind="video_features",
        video=config["video"],
        encoder=config["encoder"],
        batch_size=config["batch_size"],
        seed=args.seed,
    )
    infer_path = out / "dev-inference-config.json"
    atomic_json(infer_path, infer.to_dict())
    execute(
        [
            sys.executable,
            "-m",
            "deepfake_detection_archive",
            "infer",
            "--config",
            str(infer_path),
        ],
        evidence,
        "infer",
    )
    scores_path = out / "dev-inference/scores.jsonl"
    scores = load_scores(scores_path)
    expected = {record.sample_id: record for record in dev}
    for score in scores:
        record = expected[score.sample_id]
        for field in ("label", "generator", "source", "modality", "content_type"):
            if getattr(score, field) != getattr(record, field):
                raise ValueError("score metadata differs")
    overall = evaluate_records(scores, expected_sample_ids=list(expected))
    atomic_json(
        evidence / "dev-diagnostics.json",
        {
            "overall": asdict(overall),
            "by_generator_versus_all_77_real": {
                generator: asdict(
                    evaluate_records(
                        [
                            score
                            for score in scores
                            if score.label == 0 or score.generator == generator
                        ]
                    )
                )
                for generator in sorted({r.generator for r in dev if r.label == 1})
            },
            "checkpoint_sha256": checkpoint_sha,
            "scores_sha256": file_sha256(scores_path),
            "dev_manifest_sha256": file_sha256(dev_manifest),
            "holdout_accessed": False,
            "confidence_interval": None,
            "note": "Dev selected early stopping; shared real reference; known aspect-ratio shortcut; exploratory only.",
        },
    )
    for source, name in (
        (out / "experiment.json", "experiment.json"),
        (out / "training/run.json", "training-run.json"),
        (out / "training/history.json", "training-history.json"),
        (out / "decode_failures.json", "decode-failures.json"),
        (out / "audit.json", "data-audit.json"),
        (scores_path, "dev-scores.jsonl"),
        (out / "dev-inference/report.json", "infer-report.json"),
        (out / "dev-inference/state.json", "infer-state.json"),
        (infer_path, "infer-config.json"),
    ):
        if source.exists():
            (evidence / name).write_bytes(source.read_bytes())
    print(
        json.dumps(
            {
                "seed": args.seed,
                "dev_eer": overall.fullset_eer,
                "failures": overall.failure_count,
            }
        )
    )


if __name__ == "__main__":
    main()
