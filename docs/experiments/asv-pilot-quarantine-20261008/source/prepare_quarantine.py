"""Create the predeclared ASV candidate quarantine; never train or overwrite."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from deepfake_detection_archive.data.audit import audit_manifest
from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    load_sources,
    manifest_hash,
    write_manifest,
)
from deepfake_detection_archive.data.split import (
    connected_groups,
    statistical_groups,
    validate_splits,
)
from deepfake_detection_archive.experiments import load_experiment_config
from deepfake_detection_archive.tracking import code_identity, utc_now


ROOT = Path(__file__).resolve().parents[4]
DOC = ROOT / "docs/experiments/asv-pilot-quarantine-20261008"
OLD = ROOT / "data/asvspoof2019-la-pilot"
NEW = ROOT / "data/asvspoof2019-la-pilot-quarantine"
FROZEN = ROOT / "docs/experiments/asv-pilot-20261007/configs"
WORK = ROOT / "outputs/experiment-planning/asv-pilot-quarantine"
MEDIA = ROOT / "data/asvspoof2019-la/LA"
INPUT_HASHES = {
    "preflight-summary.json": "526389068dd6bbd8e4b7995e98d4f37b60f83e93dfe4d7064cd45fdac3164549",
    "near-duplicates.json": "a5785da209a75b875348d842703f805c97e6703c651304fdcd2fd5eeb93fee96",
    "manifest-decoded.jsonl": "a93063bb743325c7258fc044c1034813285f6e60200f8edc006f3f5bdf4d04ed",
    "sources.jsonl": "6ecc3a94e6b11d60cf4da4b1b5790aa251d95ee75c03df0ce63a4f5f62187039",
    "audit-train-dev.json": "2b2557c3b86ace907075c081fab48d502ada22c94d6d92f6410eefef22aa63f5",
    "import-summary.json": "af515f1c55df445e2e403b6d40630cfdb61f6c0f93af84cf770d8e0ae4f6aeac",
}
RUNS = ("logmel-seed42", "aasist-l-seed42", "logmel-seed43", "aasist-l-seed43")


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        stream.write("\n")


def copy_exact(source: Path, target: Path) -> None:
    with target.open("xb") as stream:
        stream.write(source.read_bytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def split_counts(records: list[SampleRecord]) -> dict[str, Any]:
    groups = statistical_groups(records)
    counts = {}
    for split in ("train", "dev"):
        selected = [r for r in records if r.split == split]
        counts[split] = {
            "samples": len(selected),
            "labels": dict(Counter(str(r.label) for r in selected)),
            "speakers": len({r.identity_id for r in selected}),
            "statistical_groups": len({groups[r.sample_id] for r in selected}),
            "speakers_per_label": {
                str(label): len({r.identity_id for r in selected if r.label == label})
                for label in (0, 1)
            },
            "statistical_groups_per_label": {
                str(label): len(
                    {groups[r.sample_id] for r in selected if r.label == label}
                )
                for label in (0, 1)
            },
            "attacks": dict(sorted(Counter(r.generator for r in selected).items())),
        }
    return counts


def validate_active(records: list[SampleRecord]) -> dict[str, Any]:
    require(len(records) == 998, "requires 998 active records")
    require(
        all(r.status == "available" and r.decode_status == "ok" for r in records),
        "decode/status differs",
    )
    require(
        all(r.split in {"train", "dev"} and r.modality == "audio" for r in records),
        "unexpected scope",
    )
    require(
        not validate_splits(records, require_class_coverage=False),
        "known split leakage",
    )
    counts = split_counts(records)
    for split, label_count, speakers in (("train", 243, 19), ("dev", 256, 20)):
        item = counts[split]
        require(
            item["labels"] == {"0": label_count, "1": label_count},
            "class counts differ",
        )
        require(
            item["speakers"] == speakers and item["statistical_groups"] == 19,
            "group counts differ",
        )
        require(
            set(item["attacks"]) == {"human", *(f"A{i:02}" for i in range(1, 7))},
            "attack coverage differs",
        )
    require(
        counts["dev"]["speakers_per_label"] == {"0": 20, "1": 10},
        "dev speaker coverage differs",
    )
    require(
        counts["dev"]["statistical_groups_per_label"] == {"0": 19, "1": 9},
        "dev conservative groups differ",
    )
    return counts


def main() -> None:  # noqa: PLR0915 -- one ordered, non-overwriting research preparation
    require(not NEW.exists() and not WORK.exists(), "output exists; never overwrite")
    for name, digest in INPUT_HASHES.items():
        require(file_sha256(OLD / name) == digest, f"frozen input changed: {name}")
    records = load_manifest(OLD / "manifest-decoded.jsonl")
    require(len(records) == 1024, "original size differs")
    preflight = json.loads((OLD / "preflight-summary.json").read_text())
    require(
        preflight["decode_and_source_audit_passed"] is True, "decode evidence missing"
    )
    require(
        preflight["manifest_hash"] == manifest_hash(records),
        "decode manifest binding differs",
    )
    duplicate = json.loads((OLD / "near-duplicates.json").read_text())
    require(
        duplicate["manifest_hash"] == manifest_hash(records)
        and duplicate["failures"] == [],
        "fingerprint binding differs",
    )
    candidates = duplicate["candidate_pairs"]
    edges = tuple((p["left_id"], p["right_id"]) for p in candidates)
    require(
        edges == (("LA_D_1032020", "LA_D_1073474"), ("LA_D_1062379", "LA_T_1023667")),
        "candidate set differs",
    )
    original_groups = connected_groups(records)
    conservative_groups = connected_groups(records, duplicate_pairs=edges)
    excluded = [r for r in records if original_groups[r.sample_id] == "LA_T_1006969"]
    require(
        len(excluded) == 26 and Counter(r.label for r in excluded) == {0: 13, 1: 13},
        "excluded component differs",
    )
    require(
        all(
            r.split == "train" and r.identity_id.endswith(":LA_0087") for r in excluded
        ),
        "excluded scope differs",
    )
    excluded_ids = {r.sample_id for r in excluded}
    updated = [
        replace(
            r,
            derivation_group_id=f"asv-quarantine-20261008:{conservative_groups[r.sample_id]}",
        )
        for r in records
    ]
    active = [r for r in updated if r.sample_id not in excluded_ids]
    ledger = [
        replace(
            r,
            split="unassigned",
            status="excluded",
            failure_reason="unconfirmed cross-split fingerprint candidate; original train speaker component quarantined",
        )
        if r.sample_id in excluded_ids
        else r
        for r in updated
    ]
    counts = validate_active(active)
    require(not validate_splits(ledger, require_class_coverage=False), "ledger leakage")
    require(
        [r.sample_id for r in active if r.split == "dev"]
        == [r.sample_id for r in records if r.split == "dev"],
        "dev changed",
    )
    retained_ids = {r.sample_id for r in active}
    retained_edges = tuple(
        (a, b) for a, b in edges if a in retained_ids and b in retained_ids
    )
    require(
        not validate_splits(active, retained_edges, require_class_coverage=False),
        "candidate crosses active splits",
    )
    sources = load_sources(OLD / "sources.jsonl")
    require(
        all(
            s.license_status == "approved"
            and "training" in s.allowed_uses
            and s.downloaded_at != "unknown"
            for s in sources.values()
        ),
        "research source permission/date missing",
    )
    # Rehash every original media file, including quarantined files; no decode or eval.
    audit = audit_manifest(
        [replace(r, split="unassigned") for r in records], MEDIA, sources
    )
    require(
        audit.passed and len(audit.trainable_ids) == 1024,
        "actual byte/source audit failed",
    )
    NEW.mkdir()
    WORK.mkdir(parents=True)
    write_manifest(active, NEW / "manifest-decoded.jsonl")
    write_manifest(ledger, NEW / "all-records-ledger.jsonl")
    write_manifest([r for r in active if r.split == "dev"], NEW / "dev.jsonl")
    excluded_rows: list[dict[str, Any]] = [
        {
            "original_record": r.to_dict(),
            "original_component": original_groups[r.sample_id],
            "bytes": (MEDIA / r.path).stat().st_size,
            "reason": "unconfirmed cross-split candidate; whole original training speaker component excluded",
        }
        for r in excluded
    ]
    write_json(NEW / "excluded.json", excluded_rows)
    write_json(NEW / "byte-source-audit.json", audit.to_dict())
    config_bindings = {}
    for run in RUNS:
        old_path = FROZEN / f"{run}.json"
        old = json.loads(old_path.read_text())
        config = dict(old)
        config.update(
            manifest=str(NEW / "manifest-decoded.jsonl"),
            feature_cache=str(
                ROOT / "cache/asvspoof2019-la-pilot-quarantine" / old["architecture"]
            ),
            output_dir=str(ROOT / "outputs/experiments/asv-pilot-quarantine" / run),
        )
        changed = {k for k in old.keys() | config.keys() if old.get(k) != config.get(k)}
        require(
            changed == {"manifest", "feature_cache", "output_dir"},
            "unapproved config change",
        )
        require(
            config["sources"] == str(OLD / "sources.jsonl"), "source registry changed"
        )
        write_json(WORK / f"{run}.json", config)
        load_experiment_config(WORK / f"{run}.json")
        copy_exact(WORK / f"{run}.json", DOC / "configs" / f"{run}.json")
        config_bindings[run] = {
            "original_path": str(old_path.relative_to(ROOT)),
            "original_sha256": file_sha256(old_path),
            "new_sha256": file_sha256(WORK / f"{run}.json"),
            "changed_keys": sorted(changed),
            "all_other_fields_equal": True,
        }
    summary = {
        "schema_version": 1,
        "finished_at": utc_now(),
        "script_sha256": file_sha256(Path(__file__)),
        "code": code_identity(),
        "input_sha256": INPUT_HASHES,
        "original_manifest_hash": manifest_hash(records),
        "active_manifest_hash": manifest_hash(active),
        "ledger_manifest_hash": manifest_hash(ledger),
        "counts": counts,
        "excluded_count": len(excluded),
        "excluded_total_bytes": sum(r["bytes"] for r in excluded_rows),
        "candidate_pairs_original": candidates,
        "candidates_confirmed": False,
        "candidate_policy": "conservative grouping and whole original train component quarantine before model scores",
        "active_cross_split_candidates": 0,
        "actual_all_original_media_rehashed": 1024,
        "byte_source_audit_passed": audit.passed,
        "audit_warnings": sum(i.severity == "warning" for i in audit.issues),
        "decode_failures": 0,
        "decode_evidence": "original decoded manifest and preflight hashes verified; no repeated decode",
        "fingerprint_recomputed": False,
        "sources": {k: asdict(v) for k, v in sources.items()},
        "source_path": str(OLD / "sources.jsonl"),
        "sources_sha256": file_sha256(OLD / "sources.jsonl"),
        "config_bindings": config_bindings,
        "eval_or_holdout_accessed": False,
        "training_started": False,
        "root_review_required_before_training": True,
        "formal_ci_or_promotion": False,
        "official_competition_usage": "unconfirmed",
    }
    write_json(NEW / "summary.json", summary)
    write_json(WORK / "plan.json", summary)
    for name in (
        "manifest-decoded.jsonl",
        "all-records-ledger.jsonl",
        "excluded.json",
        "summary.json",
    ):
        copy_exact(NEW / name, DOC / name)
    for name, digest in INPUT_HASHES.items():
        require(
            file_sha256(OLD / name) == digest,
            f"old input changed during preparation: {name}",
        )
    print(
        json.dumps(
            {
                "active": len(active),
                "excluded": len(excluded),
                "counts": counts,
                "training_started": False,
            }
        )
    )


if __name__ == "__main__":
    main()
