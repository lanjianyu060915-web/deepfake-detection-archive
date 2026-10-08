"""Audit existing decoded train/dev bytes without requiring or inspecting holdout."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import replace
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
from deepfake_detection_archive.tracking import atomic_json, code_identity, utc_now


ROOT = Path(__file__).resolve().parents[4]
PILOT = ROOT / "data/asvspoof2019-la-pilot"
DOC = Path(__file__).parent


def main() -> None:  # noqa: PLR0912 -- sequential provenance and coverage checks
    started, tick = utc_now(), time.perf_counter()
    outputs = (
        "audit-train-dev.json",
        "near-duplicates.json",
        "dev.jsonl",
        "preflight-summary.json",
    )
    if any((PILOT / name).exists() for name in outputs):
        raise FileExistsError("retain existing audited artifacts")
    acquisition = json.loads(
        (ROOT / "data/asvspoof2019-la/acquisition.json").read_text()
    )
    if not acquisition["official_checksum_verified"] or (
        acquisition["eval_member_contents_read"] or acquisition["eval_media_extracted"]
    ):
        raise ValueError("verified train/dev-only extraction required")
    media_root = ROOT / "data/asvspoof2019-la" / acquisition["dataset_root"]
    original = load_manifest(PILOT / "manifest.jsonl")
    decoded = load_manifest(PILOT / "manifest-decoded.jsonl")
    if len(decoded) != 1024 or len(original) != 1024:
        raise ValueError("expected exactly 1024 samples")
    for before, after in zip(original, decoded, strict=True):
        if (
            any(
                getattr(before, key) != getattr(after, key)
                for key in (
                    "sample_id",
                    "path",
                    "sha256",
                    "label",
                    "identity_id",
                    "split",
                    "generator",
                )
            )
            or after.decode_status != "ok"
        ):
            raise ValueError("decoded bytes/membership or decoding differs")
    counts = {}
    for split in ("train", "dev"):
        rows = [r for r in decoded if r.split == split]
        labels = Counter(r.label for r in rows)
        per_label = {
            str(label): len({r.identity_id for r in rows if r.label == label})
            for label in (0, 1)
        }
        if labels != {0: 256, 1: 256} or per_label != {
            "0": 20,
            "1": 20 if split == "train" else 10,
        }:
            raise ValueError("label/speaker coverage differs")
        if {r.generator for r in rows if r.label == 1} != {
            f"A{i:02d}" for i in range(1, 7)
        }:
            raise ValueError("missing attack coverage")
        counts[split] = {
            "labels": dict(labels),
            "speakers": len({r.identity_id for r in rows}),
            "speakers_per_label": per_label,
            "attacks": dict(Counter(r.generator for r in rows)),
        }
    if any(r.split not in {"train", "dev"} for r in decoded):
        raise ValueError("eval/holdout input forbidden")
    errors = validate_splits(decoded, require_class_coverage=False)
    sources_path = PILOT / "sources.jsonl"
    sources = load_sources(sources_path)
    # Same bytes/registry audit as training: validate original split relations
    # separately; a train/dev pilot does not require a third holdout split.
    audit = audit_manifest(
        [replace(r, split="unassigned") for r in decoded], media_root, sources
    )
    atomic_json(
        PILOT / "audit-train-dev.json",
        {
            "scope": "train/dev bytes only; original splits validated separately",
            "split_errors": errors,
            "audit": audit.to_dict(),
        },
    )
    if errors or not audit.passed:
        raise ValueError("actual media/source/leakage audit failed")
    for source in sources.values():
        if (
            source.archive_sha256 != acquisition["archive_sha256"]
            or source.downloaded_at == "unknown"
        ):
            raise ValueError("acquisition provenance incomplete")
    duplicates = find_near_duplicates(decoded, media_root)
    atomic_json(PILOT / "near-duplicates.json", duplicates.to_dict())
    write_manifest([r for r in decoded if r.split == "dev"], PILOT / "dev.jsonl")
    summary = {
        "started_at": started,
        "finished_at": utc_now(),
        "seconds": time.perf_counter() - tick,
        "script_sha256": file_sha256(Path(__file__)),
        "code": code_identity(),
        "manifest_hash": manifest_hash(decoded),
        "manifest_file_sha256": file_sha256(PILOT / "manifest-decoded.jsonl"),
        "sources_sha256": file_sha256(sources_path),
        "counts": counts,
        "decode_and_source_audit_passed": audit.passed,
        "audit_warning_count": len(audit.issues),
        "duplicate_candidate_count": len(duplicates.candidate_pairs),
        "cross_split_candidate_count": sum(
            pair.cross_split for pair in duplicates.candidate_pairs
        ),
        "fingerprint_failures": len(duplicates.failures),
        "candidate_review_required": True,
        "training_authorized_by_this_script": False,
        "eval_or_holdout_accessed": False,
        "official_competition_usage": "unconfirmed",
        "prior_failed_audit_retained": "audit.json",
    }
    atomic_json(PILOT / "preflight-summary.json", summary)
    for name in (
        *outputs,
        "manifest-decoded.jsonl",
        "sources.jsonl",
        "import-summary.json",
    ):
        (DOC / name).write_bytes((PILOT / name).read_bytes())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
