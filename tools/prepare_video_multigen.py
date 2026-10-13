"""Prepare registered multi-generator pairs without changing existing splits."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from prepare_robustness import MULTIGEN_CANONICAL, transform
from prepare_vif_bench import component, stem_of
from prepare_vif_panda import _copy

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.data.split import connected_groups


def select_pairs(
    full: list[SampleRecord],
    expanded: list[SampleRecord],
    exclusions: list[SampleRecord],
) -> tuple[list[SampleRecord], int]:
    """Close exclusions over metadata relations and conservative filename groups."""
    registered = {row.sample_id: row for row in full}
    chosen = {row.sample_id: row for row in expanded}
    if not chosen.keys() <= registered.keys():
        raise ValueError("expanded samples must belong to the full registration")
    if any(row.split not in ("train", "dev") for row in expanded):
        raise ValueError("expanded manifest requires train/dev assignments")
    for row in expanded:
        fixed_split = registered[row.sample_id].split
        if fixed_split in ("train", "dev", "holdout") and row.split != fixed_split:
            raise ValueError("expanded manifest changes a registered fixed split")
    combined = {row.sample_id: row for row in exclusions}
    combined.update(registered)
    # Alternate registrations retain their own relation values and share an
    # explicit edge to the canonical ID. This unions rather than overwrites
    # full, expanded and exclusion metadata, including external parent nodes.
    relation_records = list(combined.values())
    edges: list[tuple[str, str]] = []
    for index, row in enumerate(full + expanded + exclusions):
        if row != combined[row.sample_id]:
            alias = f"metadata-version:{index}:{row.sample_id}"
            relation_records.append(replace(row, sample_id=alias))
            edges.append((alias, row.sample_id))
    by_clip: dict[str, list[str]] = defaultdict(list)
    for row in relation_records:
        by_clip[component(stem_of(row.path))].append(row.sample_id)
    edges.extend((ids[0], sid) for ids in by_clip.values() for sid in ids[1:])
    groups = connected_groups(relation_records, tuple(edges))
    blocked_ids = (registered.keys() - chosen.keys()) | {
        row.sample_id for row in exclusions
    }
    blocked = {groups[sid] for sid in blocked_ids}
    selected = [row for row in expanded if groups[row.sample_id] not in blocked]
    assignments: dict[str, str] = {}
    selected_ids = {row.sample_id for row in selected}
    paired_reals = {row.parent_id for row in selected if row.label == 1}
    for row in selected:
        group = groups[row.sample_id]
        if assignments.setdefault(group, row.split) != row.split:
            raise ValueError("metadata connects incompatible train/dev assignments")
        if row.label == 1 and row.parent_id not in selected_ids:
            raise ValueError("fake sample is missing its registered real parent")
        if row.label == 0 and row.sample_id not in paired_reals:
            raise ValueError("real sample is missing its registered fake pair")
    return selected, len(expanded) - len(selected)


def prepare(  # noqa: PLR0913 -- explicit preparation inputs and resource limits
    *,
    full_manifest: Path,
    expanded_manifest: Path,
    archive_path: Path,
    sources: Path,
    exclusion_manifests: list[Path],
    output: Path,
    workers: int = 8,
    timeout: float = 60,
) -> dict[str, Any]:
    started = time.monotonic()
    if workers < 1 or timeout <= 0:
        raise ValueError("workers and timeout must be positive")
    excluded = [row for path in exclusion_manifests for row in load_manifest(path)]
    rows, excluded_count = select_pairs(
        load_manifest(full_manifest), load_manifest(expanded_manifest), excluded
    )
    if not rows:
        raise ValueError("no paired samples remain after exclusions")
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    (output / "media").mkdir()
    shutil.copyfile(sources, output / "sources.jsonl")
    with zipfile.ZipFile(archive_path) as archive:
        for row in rows:
            target = raw / row.path
            digest = _copy(archive, archive.getinfo(row.path), target)
            if digest != row.sha256:
                raise ValueError(f"registered SHA256 mismatch: {row.sample_id}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        derived = list(
            pool.map(
                lambda row: transform(row, raw, output, MULTIGEN_CANONICAL, timeout),
                rows,
            )
        )
    failures = {
        row.sample_id: row.failure_reason
        for row in derived
        if row.status != "available"
    }
    summary: dict[str, Any] = {
        "condition": MULTIGEN_CANONICAL,
        "full_manifest": str(full_manifest.resolve()),
        "expanded_manifest": str(expanded_manifest.resolve()),
        "exclusion_manifests": [str(path.resolve()) for path in exclusion_manifests],
        "excluded_rows": excluded_count,
        "manifest_sha256": write_manifest(derived, output / "manifest.jsonl"),
        "samples": len(derived),
        "split_counts": dict(Counter(row.split for row in derived)),
        "fake_generators": dict(
            Counter(row.generator for row in derived if row.label == 1)
        ),
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if not failures:
        shutil.rmtree(raw)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--expanded-manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    result = prepare(
        full_manifest=args.full_manifest,
        expanded_manifest=args.expanded_manifest,
        archive_path=args.archive,
        sources=args.sources,
        exclusion_manifests=args.exclude_manifest,
        output=args.output,
        workers=args.workers,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 2 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
