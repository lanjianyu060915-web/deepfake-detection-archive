"""Audit the predeclared ASV train/dev pilot; never inspect eval or train models."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from deepfake_detection_archive.data.audit import audit_manifest
from deepfake_detection_archive.data.duplicates import find_near_duplicates
from deepfake_detection_archive.data.manifest import (
    file_sha256,
    load_manifest,
    load_sources,
    manifest_hash,
    write_manifest,
)
from deepfake_detection_archive.data.split import validate_splits
from deepfake_detection_archive.preparation import decode_manifest
from deepfake_detection_archive.tracking import atomic_json, code_identity, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--downloaded-at", required=True)
    args = parser.parse_args()
    project = args.project_root.resolve(strict=True)
    acquisition_path = project / "data/asvspoof2019-la/acquisition.json"
    acquisition = json.loads(acquisition_path.read_text())
    if not acquisition["official_checksum_verified"]:
        raise ValueError("archive checksum verification is required")
    if acquisition["eval_member_contents_read"] or acquisition["eval_media_extracted"]:
        raise ValueError("unexpected eval access in acquisition")
    media_root = acquisition_path.parent / acquisition["dataset_root"]
    pilot = project / "data/asvspoof2019-la-pilot"
    decoded_path = pilot / "manifest-decoded.jsonl"
    if decoded_path.exists() or (pilot / "audit.json").exists():
        raise ValueError("existing audit artifacts must not be overwritten")
    started = utc_now()
    monotonic = time.perf_counter()
    records = load_manifest(pilot / "manifest.jsonl")
    if len(records) != 1024 or any(r.split not in ("train", "dev") for r in records):
        raise ValueError("requires exactly 1024 train/dev-only pilot samples")
    sources_path = pilot / "sources.jsonl"
    original_sources = sources_path.read_text()
    source_rows = [json.loads(line) for line in original_sources.splitlines()]
    if len(source_rows) != 1 or source_rows[0]["downloaded_at"] != "unknown":
        raise ValueError("requires the original unfilled single-source import")
    (pilot / "sources-import-original.jsonl").write_text(original_sources)
    source_rows[0]["downloaded_at"] = args.downloaded_at
    source_rows[0]["archive_sha256"] = acquisition["archive_sha256"]
    sources_path.write_text(json.dumps(source_rows[0], sort_keys=True) + "\n")
    decoded = decode_manifest(records, media_root)
    write_manifest(decoded, decoded_path)
    audit = audit_manifest(decoded, media_root, load_sources(sources_path))
    atomic_json(pilot / "audit.json", audit.to_dict())
    counts = {
        split: {
            "labels": dict(Counter(r.label for r in decoded if r.split == split)),
            "speakers": len({r.identity_id for r in decoded if r.split == split}),
            "attacks": dict(Counter(r.generator for r in decoded if r.split == split)),
        }
        for split in ("train", "dev")
    }
    if not audit.passed or validate_splits(decoded, require_class_coverage=False):
        raise ValueError("decode/provenance/split audit failed; inspect audit.json")
    for split, speakers in (("train", 20), ("dev", 10)):
        selected = [r for r in decoded if r.split == split]
        if Counter(r.label for r in selected) != {0: 256, 1: 256}:
            raise ValueError(f"unexpected labels in {split}")
        if len({r.identity_id for r in selected}) != speakers:
            raise ValueError(f"unexpected speaker count in {split}")
        if {r.generator for r in selected if r.label} != {
            f"A{i:02d}" for i in range(1, 7)
        }:
            raise ValueError(f"incomplete attack coverage in {split}")
    duplicate_audit = find_near_duplicates(decoded, media_root)
    atomic_json(pilot / "near-duplicates.json", duplicate_audit.to_dict())
    write_manifest([r for r in decoded if r.split == "dev"], pilot / "dev.jsonl")
    summary = {
        "started_at": started,
        "finished_at": utc_now(),
        "seconds": time.perf_counter() - monotonic,
        "script_sha256": file_sha256(Path(__file__)),
        "code": code_identity(),
        "acquisition_sha256": file_sha256(acquisition_path),
        "downloaded_at": args.downloaded_at,
        "manifest_hash": manifest_hash(decoded),
        "sources_sha256": file_sha256(sources_path),
        "counts": counts,
        "decode_and_source_audit_passed": audit.passed,
        "audit_warning_count": len(audit.issues),
        "duplicate_candidate_count": len(duplicate_audit.candidate_pairs),
        "cross_split_candidate_count": sum(
            pair.cross_split for pair in duplicate_audit.candidate_pairs
        ),
        "fingerprint_failures": len(duplicate_audit.failures),
        "training_authorized_by_this_script": False,
        "candidate_review_required": True,
        "eval_or_holdout_accessed": False,
        "official_competition_usage": "unconfirmed",
    }
    atomic_json(pilot / "preflight-summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
