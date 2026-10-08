"""Revalidate quarantine inputs before any training; this command never trains."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from prepare_quarantine import (
    DOC,
    FROZEN,
    INPUT_HASHES,
    MEDIA,
    NEW,
    OLD,
    ROOT,
    RUNS,
    WORK,
    require,
    validate_active,
)

from deepfake_detection_archive.data.audit import audit_manifest
from deepfake_detection_archive.data.manifest import (
    file_sha256,
    load_manifest,
    load_sources,
    manifest_hash,
)
from deepfake_detection_archive.data.split import connected_groups, validate_splits
from deepfake_detection_archive.experiments import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run = args.config.stem
    require(run in RUNS, "unknown run")
    for name, digest in INPUT_HASHES.items():
        require(file_sha256(OLD / name) == digest, f"original input changed: {name}")
    summary = json.loads((NEW / "summary.json").read_text())
    require(
        summary["sources_sha256"] == INPUT_HASHES["sources.jsonl"],
        "source binding changed",
    )
    require(
        file_sha256(DOC / "source/prepare_quarantine.py") == summary["script_sha256"],
        "preparation source changed",
    )
    require(
        file_sha256(args.config) == summary["config_bindings"][run]["new_sha256"],
        "config hash changed",
    )
    require(
        file_sha256(FROZEN / args.config.name)
        == summary["config_bindings"][run]["original_sha256"],
        "preregistration hash changed",
    )
    config = load_experiment_config(args.config)
    require(
        Path(config.manifest) == NEW / "manifest-decoded.jsonl", "manifest path differs"
    )
    require(
        Path(config.sources) == OLD / "sources.jsonl"
        and Path(config.data_root) == MEDIA,
        "source/media paths differ",
    )
    old_config = json.loads((FROZEN / args.config.name).read_text())
    new_config = json.loads(args.config.read_text())
    require(
        {
            k
            for k in old_config.keys() | new_config.keys()
            if old_config.get(k) != new_config.get(k)
        }
        == {"manifest", "feature_cache", "output_dir"},
        "configuration delta differs",
    )
    require(
        config.output_dir
        == str(ROOT / "outputs/experiments/asv-pilot-quarantine" / run),
        "output isolation differs",
    )
    require(
        not Path(config.output_dir).exists(), "experiment already exists; no overwrite"
    )
    require(
        not (WORK / "telemetry" / run).exists(),
        "telemetry already exists; no overwrite",
    )
    original = load_manifest(OLD / "manifest-decoded.jsonl")
    active = load_manifest(NEW / "manifest-decoded.jsonl")
    ledger = load_manifest(NEW / "all-records-ledger.jsonl")
    counts = validate_active(active)
    require(
        manifest_hash(active) == summary["active_manifest_hash"],
        "active manifest changed",
    )
    require(manifest_hash(ledger) == summary["ledger_manifest_hash"], "ledger changed")
    candidate_report = json.loads((OLD / "near-duplicates.json").read_text())
    candidates = candidate_report["candidate_pairs"]
    require(
        candidates == summary["candidate_pairs_original"]
        and summary["candidates_confirmed"] is False,
        "candidate evidence changed",
    )
    edges = tuple((p["left_id"], p["right_id"]) for p in candidates)
    old_groups = connected_groups(original)
    conservative = connected_groups(original, duplicate_pairs=edges)
    excluded_ids = {
        r.sample_id for r in original if old_groups[r.sample_id] == "LA_T_1006969"
    }
    expected = [
        replace(
            r,
            derivation_group_id=f"asv-quarantine-20261008:{conservative[r.sample_id]}",
        )
        for r in original
    ]
    require(
        active == [r for r in expected if r.sample_id not in excluded_ids],
        "active membership or preserved fields changed",
    )
    require(
        len(ledger) == 1024 and len(excluded_ids) == 26,
        "ledger or exclusion size changed",
    )
    for before, after in zip(expected, ledger, strict=True):
        if before.sample_id in excluded_ids:
            require(
                after.status == "excluded"
                and after.split == "unassigned"
                and "candidate" in after.failure_reason,
                "exclusion accounting differs",
            )
            require(
                replace(
                    after,
                    status=before.status,
                    split=before.split,
                    failure_reason=before.failure_reason,
                )
                == before,
                "excluded identity or media fields changed",
            )
        else:
            require(before == after, "retained ledger row changed")
    require(
        not validate_splits(
            ledger, duplicate_pairs=edges, require_class_coverage=False
        ),
        "candidate leakage in ledger",
    )
    dev = load_manifest(NEW / "dev.jsonl")
    require(dev == [r for r in active if r.split == "dev"], "dev manifest differs")
    sources = load_sources(Path(config.sources))
    require(
        all(
            s.license_status == "approved" and "training" in s.allowed_uses
            for s in sources.values()
        ),
        "research permission missing",
    )
    audit = audit_manifest(
        [replace(r, split="unassigned") for r in original], MEDIA, sources
    )
    require(
        audit.passed and len(audit.trainable_ids) == 1024,
        "fresh byte/source audit failed",
    )
    print(
        json.dumps(
            {
                "run": run,
                "counts": counts,
                "all_original_media_hashes_rechecked": 1024,
                "split_errors": [],
                "decode_failures_in_bound_prior_evidence": 0,
                "training_started": False,
            }
        )
    )


if __name__ == "__main__":
    main()
