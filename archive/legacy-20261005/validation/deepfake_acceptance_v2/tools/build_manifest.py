#!/usr/bin/env python3
"""Freeze a human-curated D-v2 ground-truth CSV after recomputing file facts.

Labels are never inferred from names or directories.  The curator must supply an
exact REAL/FAKE label and all provenance fields in the fixed schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import (
    GROUND_TRUTH_FIELDS,
    LABELS,
    image_dimensions,
    parse_bool,
    read_csv,
    require_exact_header,
    resolve_dataset_path,
    sha256_file,
    write_csv,
    write_json,
)


def build(curation_csv: Path, dataset_root: Path, output: Path) -> dict[str, object]:
    fields, rows = read_csv(curation_csv)
    require_exact_header(fields, GROUND_TRUTH_FIELDS, "curation CSV")
    if not rows:
        raise ValueError("curation CSV is empty; a formal manifest cannot be frozen")

    sample_ids: set[str] = set()
    frozen: list[dict[str, object]] = []
    for line_number, row in enumerate(rows, 2):
        sample_id = row["sample_id"]
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"line {line_number}: empty or duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        label = row["label"]
        if label not in LABELS:
            raise ValueError(f"line {line_number}: label must be exactly REAL or FAKE; got {label!r}")
        is_original = parse_bool(row["is_original"], f"line {line_number} is_original")
        path = resolve_dataset_path(dataset_root, row["file_path"])
        if not path.is_file():
            raise ValueError(f"line {line_number}: file does not exist: {row['file_path']}")
        width, height = image_dimensions(path)
        frozen_row: dict[str, object] = dict(row)
        frozen_row["is_original"] = "true" if is_original else "false"
        frozen_row["width"] = width
        frozen_row["height"] = height
        frozen_row["sha256"] = sha256_file(path)
        frozen.append(frozen_row)

    for line_number, row in enumerate(frozen, 2):
        is_original = row["is_original"] == "true"
        parent = str(row["parent_sample_id"])
        if is_original:
            if parent or row["perturbation_type"] or row["perturbation_level"]:
                raise ValueError(f"line {line_number}: original sample must not declare perturbation metadata")
        else:
            if not parent or parent not in sample_ids:
                raise ValueError(f"line {line_number}: missing/dangling parent_sample_id {parent!r}")
            parent_row = next(item for item in frozen if item["sample_id"] == parent)
            if parent_row["label"] != row["label"]:
                raise ValueError(f"line {line_number}: perturbation label differs from parent")
            if not row["perturbation_type"] or not row["perturbation_level"]:
                raise ValueError(f"line {line_number}: perturbation type and level are required")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output}")
    write_csv(output, GROUND_TRUTH_FIELDS, frozen)
    manifest_sha = sha256_file(output)
    summary = {
        "schema_version": "D-v2-ground-truth-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "source_curation_csv": str(curation_csv.resolve()),
        "manifest": str(output.resolve()),
        "manifest_sha256": manifest_sha,
        "sample_count": len(frozen),
        "label_counts": {label: sum(row["label"] == label for row in frozen) for label in sorted(LABELS)},
        "unknown_admitted": False,
    }
    write_json(output.with_suffix(output.suffix + ".freeze.json"), summary)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{manifest_sha}  {output.name}\n", encoding="ascii"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curation-csv", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = build(args.curation_csv, args.dataset_root, args.output)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "success", **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

