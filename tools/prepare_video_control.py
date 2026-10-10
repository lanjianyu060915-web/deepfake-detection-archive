"""Prepare group-separated video metadata controls without decoding frames or GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.data.split import connected_groups
from deepfake_detection_archive.evaluation.metrics import evaluate_binary


FEATURES = {
    "width_only": ("width",),
    "numeric_media": ("width", "height", "fps", "duration_seconds", "bit_rate"),
}


def rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def partition(
    full: list[SampleRecord], pilot: list[SampleRecord], seed: int
) -> tuple[list[SampleRecord], dict[str, str], int]:
    """Keep pilot assignments and exclude every component touching old holdout."""
    pilot_groups: dict[str, list[str]] = defaultdict(list)
    for sid, group in connected_groups(pilot).items():
        pilot_groups[group].append(sid)
    edges = tuple((ids[0], sid) for ids in pilot_groups.values() for sid in ids[1:])
    groups = connected_groups(full, edges)
    fixed: dict[str, str] = {}
    for row in pilot:
        group = groups[row.sample_id]
        if group in fixed and fixed[group] != row.split:
            raise ValueError("full metadata connects incompatible pilot splits")
        fixed[group] = row.split
    result = []
    for row in full:
        group = groups[row.sample_id]
        split = fixed.get(group)
        if split == "holdout":
            continue
        if split is None:
            split = "dev" if int(rank(group, seed)[:8], 16) % 5 == 0 else "train"
        result.append(replace(row, split=split, derivation_group_id=group))
    return result, groups, len(full) - len(result)


def probe(row: SampleRecord, root: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(  # noqa: S603 - local path is a separate argument
            [  # noqa: S607 - project-managed media executable
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate:format=duration,bit_rate",
                "-of",
                "json",
                str(root / row.path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        data = json.loads(completed.stdout)
        stream, format_ = data["streams"][0], data["format"]
        return {
            "sample_id": row.sample_id,
            "sha256": row.sha256,
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": float(Fraction(stream["avg_frame_rate"])),
            "duration_seconds": float(format_["duration"]),
            "bit_rate": float(format_["bit_rate"]),
        }
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
        return {"sample_id": row.sample_id, "error": str(exc)}


def matched(
    rows: list[SampleRecord],
    metadata: dict[str, dict[str, Any]],
    seed: int,
    duration: bool = False,
) -> list[SampleRecord]:
    """Balance labels per split and exact width/height/FPS (optional 1s duration)."""
    buckets: dict[tuple[Any, ...], dict[int, list[SampleRecord]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for row in rows:
        meta = metadata[row.sample_id]
        key: tuple[Any, ...] = (row.split, meta["width"], meta["height"], meta["fps"])
        if duration:
            key += (int(meta["duration_seconds"]),)
        if row.label not in (0, 1):
            raise ValueError("metadata control requires binary labels")
        buckets[key][row.label].append(row)
    selected = []
    for labels in buckets.values():
        count = min(len(labels[0]), len(labels[1]))
        for candidates in labels.values():
            selected.extend(
                sorted(candidates, key=lambda r: rank(r.sample_id, seed))[:count]
            )
    return sorted(selected, key=lambda r: r.sample_id)


def counts(rows: list[SampleRecord]) -> dict[str, Any]:
    return {
        split: {
            "total": sum(r.split == split for r in rows),
            "labels": dict(Counter(str(r.label) for r in rows if r.split == split)),
            "fake_generators": dict(
                Counter(r.generator for r in rows if r.split == split and r.label == 1)
            ),
            "components": len(
                {r.derivation_group_id for r in rows if r.split == split}
            ),
        }
        for split in ("train", "dev")
    }


def baseline(
    rows: list[SampleRecord],
    metadata: dict[str, dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    subsets = {
        split: [r for r in rows if r.split == split] for split in ("train", "dev")
    }
    if any({r.label for r in items} != {0, 1} for items in subsets.values()):
        return {"unavailable": "both labels required in train and dev"}
    results: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    for name, fields in FEATURES.items():
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
        )
        matrices = {
            split: [[metadata[r.sample_id][field] for field in fields] for r in items]
            for split, items in subsets.items()
        }
        model.fit(matrices["train"], [r.label for r in subsets["train"]])
        scores = model.decision_function(matrices["dev"])
        results[name] = {
            "fields": fields,
            "eer": evaluate_binary(
                [int(r.label) for r in subsets["dev"] if r.label is not None],
                scores.tolist(),
            ).eer,
        }
        predictions.extend(
            {
                "model": name,
                "sample_id": row.sample_id,
                "label": row.label,
                "score": float(score),
            }
            for row, score in zip(subsets["dev"], scores, strict=True)
        )
    output.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    full, pilot = load_manifest(args.manifest), load_manifest(args.pilot)
    rows, _, excluded = partition(full, pilot, args.seed)
    cache = args.output / "metadata.jsonl"
    metadata = {
        row["sample_id"]: row
        for row in (
            [json.loads(line) for line in cache.read_text().splitlines()]
            if cache.exists()
            else []
        )
    }
    missing = [
        r for r in rows if metadata.get(r.sample_id, {}).get("sha256") != r.sha256
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, cache.open("a") as file:
        for index, meta in enumerate(
            pool.map(lambda r: probe(r, args.data_root), missing), 1
        ):
            metadata[meta["sample_id"]] = meta
            file.write(json.dumps(meta) + "\n")
            if index % 500 == 0:
                file.flush()
                print(f"Probed {index}/{len(missing)}", flush=True)
    errors = [metadata[r.sample_id] for r in rows if "error" in metadata[r.sample_id]]
    valid = [r for r in rows if "error" not in metadata[r.sample_id]]
    summary: dict[str, Any] = {
        "manifest_sha256": file_sha256(args.manifest),
        "pilot_sha256": file_sha256(args.pilot),
        "seed": args.seed,
        "excluded_holdout_related": excluded,
        "probe_failures": errors,
        "source_count": len({r.source for r in valid}),
        "source_holdout": "unavailable: all records have the same registered source",
        "variants": {},
        "generator_dev_diagnostics": {},
    }
    variants = {
        "expanded": valid,
        "matched_geometry": matched(valid, metadata, args.seed),
        "matched_geometry_duration": matched(valid, metadata, args.seed, duration=True),
        "hunyuan_geometry_duration": matched(
            [r for r in valid if r.label == 0 or r.generator == "HunyuanVideo"],
            metadata,
            args.seed,
            duration=True,
        ),
    }
    for name, selected in variants.items():
        write_manifest(selected, args.output / f"{name}.jsonl")
        summary["variants"][name] = {
            "counts": counts(selected),
            "metadata_models": baseline(
                selected, metadata, args.output / f"{name}-scores.jsonl"
            ),
        }
    # Every diagnostic trains without any component containing its target generator.
    # Dev uses only target fakes and their paired real parents, then exact matching.
    for generator in sorted({r.generator for r in valid if r.label == 1}):
        target_groups = {
            r.derivation_group_id
            for r in valid
            if r.label == 1 and r.generator == generator
        }
        target_dev = [
            r
            for r in valid
            if r.split == "dev" and r.label == 1 and r.generator == generator
        ]
        dev_ids = {r.sample_id for r in target_dev} | {r.parent_id for r in target_dev}
        selected = [
            r
            for r in valid
            if (r.split == "train" and r.derivation_group_id not in target_groups)
            or r.sample_id in dev_ids
        ]
        selected = matched(selected, metadata, args.seed, duration=True)
        name = "leave-" + generator.replace("/", "_")
        write_manifest(selected, args.output / f"{name}.jsonl")
        summary["generator_dev_diagnostics"][generator] = {
            "counts": counts(selected),
            "metadata_models": baseline(
                selected, metadata, args.output / f"{name}-scores.jsonl"
            ),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
