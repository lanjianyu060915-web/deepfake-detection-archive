#!/usr/bin/env python3
"""Validate a frozen D-v2 dataset and emit machine/human-readable reports."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
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
    write_json,
)


def _counter(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(collections.Counter((row.get(key) or "unknown") for row in rows).items()))


def validate(manifest: Path, dataset_root: Path) -> dict[str, object]:
    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    try:
        fields, rows = read_csv(manifest)
        require_exact_header(fields, GROUND_TRUTH_FIELDS, "ground truth")
    except Exception as exc:
        return {
            "status": "FAIL",
            "manifest": str(manifest.resolve()),
            "dataset_root": str(dataset_root.resolve()),
            "sample_count": 0,
            "errors": [{"code": "MANIFEST_INVALID", "message": str(exc)}],
            "warnings": [],
        }

    if not rows:
        errors.append({"code": "EMPTY_DATASET", "message": "formal ground truth contains no samples"})

    seen_ids: dict[str, int] = {}
    sha_to_ids: dict[str, list[str]] = collections.defaultdict(list)
    facts: dict[str, dict[str, object]] = {}
    parsed_original: dict[str, bool] = {}
    for line_number, row in enumerate(rows, 2):
        sample_id = row.get("sample_id", "")
        prefix = {"line": line_number, "sample_id": sample_id}
        if not sample_id:
            errors.append({**prefix, "code": "SAMPLE_ID_EMPTY", "message": "sample_id is required"})
        elif sample_id in seen_ids:
            errors.append({**prefix, "code": "SAMPLE_ID_DUPLICATE", "message": f"first seen on line {seen_ids[sample_id]}"})
        else:
            seen_ids[sample_id] = line_number

        if row.get("label") not in LABELS:
            errors.append({**prefix, "code": "LABEL_INVALID", "message": "label must be exactly REAL or FAKE"})
        try:
            parsed_original[sample_id] = parse_bool(row.get("is_original"), "is_original")
        except ValueError as exc:
            errors.append({**prefix, "code": "IS_ORIGINAL_INVALID", "message": str(exc)})

        try:
            path = resolve_dataset_path(dataset_root, row.get("file_path", ""))
        except ValueError as exc:
            errors.append({**prefix, "code": "PATH_INVALID", "message": str(exc)})
            continue
        if not path.is_file():
            errors.append({**prefix, "code": "FILE_MISSING", "message": row.get("file_path", "")})
            continue
        try:
            width, height = image_dimensions(path)
        except Exception as exc:
            errors.append({**prefix, "code": "IMAGE_DECODE_FAILED", "message": str(exc)})
            continue
        actual_sha = sha256_file(path)
        facts[sample_id] = {"width": width, "height": height, "sha256": actual_sha}
        if row.get("sha256") != actual_sha:
            errors.append({**prefix, "code": "SHA256_MISMATCH", "message": f"expected {row.get('sha256')!r}, got {actual_sha}"})
        for dimension_name, actual in (("width", width), ("height", height)):
            try:
                recorded = int(row.get(dimension_name, ""))
            except ValueError:
                recorded = -1
            if recorded != actual:
                errors.append({**prefix, "code": "DIMENSION_MISMATCH", "message": f"{dimension_name}: expected {row.get(dimension_name)!r}, got {actual}"})
        sha_to_ids[actual_sha].append(sample_id)

    row_by_id = {row["sample_id"]: row for row in rows if row.get("sample_id") and row["sample_id"] in seen_ids}
    for line_number, row in enumerate(rows, 2):
        sample_id = row.get("sample_id", "")
        if sample_id not in parsed_original:
            continue
        prefix = {"line": line_number, "sample_id": sample_id}
        parent_id = row.get("parent_sample_id", "")
        if parsed_original[sample_id]:
            if parent_id or row.get("perturbation_type") or row.get("perturbation_level"):
                errors.append({**prefix, "code": "ORIGINAL_HAS_PERTURBATION_METADATA", "message": "original rows must leave parent/type/level blank"})
        else:
            if not parent_id or parent_id not in row_by_id:
                errors.append({**prefix, "code": "PARENT_DANGLING", "message": parent_id})
            else:
                parent = row_by_id[parent_id]
                if parent.get("label") != row.get("label"):
                    errors.append({**prefix, "code": "PERTURBATION_LABEL_MISMATCH", "message": f"parent label is {parent.get('label')}"})
                if parsed_original.get(parent_id) is not True:
                    errors.append({**prefix, "code": "PARENT_NOT_ORIGINAL", "message": parent_id})
            if not row.get("perturbation_type") or not row.get("perturbation_level"):
                errors.append({**prefix, "code": "PERTURBATION_METADATA_MISSING", "message": "type and level are required"})

    duplicates = [
        {"sha256": sha, "sample_ids": ids}
        for sha, ids in sorted(sha_to_ids.items())
        if len(ids) > 1
    ]
    for duplicate in duplicates:
        errors.append({"code": "DUPLICATE_IMAGE", "message": duplicate})

    widths = [int(item["width"]) for item in facts.values()]
    heights = [int(item["height"]) for item in facts.values()]
    resolution_stats = {
        "decoded_count": len(facts),
        "width_min": min(widths) if widths else None,
        "width_max": max(widths) if widths else None,
        "width_median": statistics.median(widths) if widths else None,
        "height_min": min(heights) if heights else None,
        "height_max": max(heights) if heights else None,
        "height_median": statistics.median(heights) if heights else None,
    }
    return {
        "schema_version": "D-v2-validation-1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "manifest": str(manifest.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "sample_count": len(rows),
        "label_counts": _counter(rows, "label"),
        "generation_method_counts": _counter(rows, "generation_method"),
        "perturbation_type_counts": _counter(rows, "perturbation_type"),
        "resolution_statistics": resolution_stats,
        "duplicate_groups": duplicates,
        "errors": errors,
        "warnings": warnings,
    }


def render_markdown(report: dict[str, object]) -> str:
    labels = report.get("label_counts", {})
    lines = [
        "# D-v2 Dataset Validation Report",
        "",
        f"- Status: **{report['status']}**",
        f"- Samples: {report.get('sample_count', 0)}",
        f"- REAL: {labels.get('REAL', 0) if isinstance(labels, dict) else 0}",
        f"- FAKE: {labels.get('FAKE', 0) if isinstance(labels, dict) else 0}",
        f"- Errors: {len(report.get('errors', []))}",
        "",
        "## Checks",
        "",
        "Files, decoding, exact labels, unique IDs, SHA-256, dimensions, path containment, parent links, label inheritance, statistics and exact duplicate content were checked.",
        "",
        "## Errors",
        "",
    ]
    if report.get("errors"):
        for error in report["errors"]:
            lines.append(f"- `{error.get('code')}`: {error.get('message')}")
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--report-md", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args.manifest, args.dataset_root)
    write_json(args.report_json, report)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "sample_count": report.get("sample_count", 0), "error_count": len(report.get("errors", []))}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

