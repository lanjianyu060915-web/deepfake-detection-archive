"""Read-only audit of completed audio trials using existing independent verifiers."""

# ruff: noqa: S101
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

from deepfake_detection_archive.data.manifest import file_sha256, load_manifest
from deepfake_detection_archive.tracking import code_identity


ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parents[1]
RUNS = ("logmel-seed42", "logmel-seed43", "aasist-l-seed42", "aasist-l-seed43-retry1")


def module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main() -> None:
    training_probe = module(
        ROOT / "docs/research/offline-rehearsal/resume/model_rehearsal.py"
    )
    metric_probe = module(
        ROOT / "docs/experiments/vif-crop-control-20261007/source/verify_results.py"
    )
    dev = load_manifest(ROOT / "data/asvspoof2019-la-pilot-quarantine/dev.jsonl")
    ids = [r.sample_id for r in dev]
    assert len(ids) == 512
    rows_out = []
    for name in RUNS:
        directory = ROOT / "outputs/experiments/asv-pilot-quarantine" / name
        recorded = DOC / "training" / name
        config_path = (
            ROOT / "outputs/experiment-planning/asv-pilot-quarantine" / f"{name}.json"
        )
        experiment = json.loads((directory / "experiment.json").read_text())
        job = {
            "experiment": str(directory / "experiment.json"),
            "experiment_config": str(config_path),
            "checkpoint": str(directory / "training/best.pt"),
            "checkpoint_sha256": experiment["checkpoint_sha256"],
            "training_manifest": experiment["config"]["manifest"],
            "dev_manifest": str(
                ROOT / "data/asvspoof2019-la-pilot-quarantine/dev.jsonl"
            ),
            "sources": experiment["config"]["sources"],
            "data_root": experiment["config"]["data_root"],
        }
        training = training_probe.verify_training(job)
        scores = [
            json.loads(line)
            for line in (recorded / "dev-scores.jsonl").read_text().splitlines()
        ]
        assert [r["sample_id"] for r in scores] == ids
        assert len({r["sample_id"] for r in scores}) == 512
        assert all(r["status"] == "ok" and r["error_code"] is None for r in scores)
        assert all(
            r["label"] == sample.label
            and r["generator"] == sample.generator
            and r["source"] == sample.source
            for r, sample in zip(scores, dev, strict=True)
        )
        eer = metric_probe.independent_eer(scores)
        diagnostics = json.loads((recorded / "dev-diagnostics.json").read_text())
        trainer = json.loads((recorded / "training-run.json").read_text())
        assert abs(eer - diagnostics["overall"]["fullset_eer"]) < 1e-12
        assert abs(eer - trainer["best_dev_eer"]) < 1e-12
        assert training["provenance"]["code_sha256"] == code_identity()["code_sha256"]
        assert file_sha256(directory / "dev-inference/scores.jsonl") == file_sha256(
            recorded / "dev-scores.jsonl"
        )
        telemetry = recorded / "telemetry"
        assert (telemetry / "exit-code.txt").read_text().strip() == "0"
        assert not (telemetry / "gpu-monitor.stderr").read_text().strip()
        with (telemetry / "gpu.csv").open() as stream:
            samples = list(csv.DictReader(stream, skipinitialspace=True))
        assert samples
        peaks = {
            key: max(float(r[key].split()[0]) for r in samples)
            for key in ("memory.used [MiB]", "utilization.gpu [%]", "temperature.gpu")
        }
        if name != "logmel-seed42":
            scoring = json.loads((recorded / "dev-score-execution.json").read_text())
            assert scoring["exit_code"] == 0
            scorer = (
                DOC
                / "source"
                / (
                    "score_retry1.executed.py.txt"
                    if name.endswith("retry1")
                    else "score_pilot.py"
                )
            )
            assert scoring["scorer_sha256"] == file_sha256(scorer)
            assert scoring["wrapper_sha256"] == file_sha256(
                DOC / "source/score_with_telemetry.executed.py.txt"
            )
        rows_out.append(
            {
                "run": name,
                "dev_count": 512,
                "failures": 0,
                "dev_eer": eer,
                "training": training,
                "trainer_resources": {
                    k: trainer[k]
                    for k in (
                        "seconds",
                        "peak_cuda_allocated_bytes",
                        "peak_cuda_reserved_bytes",
                    )
                },
                "device_sampled_train_peaks": peaks,
                "checkpoint_sha256": experiment["checkpoint_sha256"],
                "scores_sha256": file_sha256(recorded / "dev-scores.jsonl"),
            }
        )
    print(
        json.dumps(
            {
                "verification": "PASS",
                "verifier_sha256": file_sha256(Path(__file__)),
                "runs": rows_out,
                "formal_promotion": False,
                "formal_ci": None,
                "holdout_or_eval_media_read": False,
                "independent_eer_implementation": str(
                    ROOT
                    / "docs/experiments/vif-crop-control-20261007/source/verify_results.py"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Verification requires non-optimized Python")
    main()
