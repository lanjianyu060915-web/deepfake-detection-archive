"""Build a fixed size-controlled ViF-Bench view and its canonical re-encodes.

Matching rule: keep a fake only when its registered paired real has exactly the
same probed width and height; inside every (generator, width, height) bucket take
min(fake, real) rows in a seeded order. The view keeps original media paths so the
original-size condition can be scored without duplicating bytes, and both labels
are then re-encoded with identical output size and codec settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from prepare_robustness import prepare

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)


CANONICAL_CONDITIONS = ("video_h264_crf18_448x256", "video_h264_crf35_448x256")


def rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def probe(row: SampleRecord, root: Path, timeout: float) -> dict[str, Any]:
    """Read width, height, frame rate and codec without decoding frames."""
    try:
        completed = subprocess.run(  # noqa: S603 - local path is a separate argument
            [  # noqa: S607 - project-managed media executable
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,codec_name:format=format_name",
                "-of",
                "json",
                str(root / row.path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        data = json.loads(completed.stdout)
        stream = data["streams"][0]
        return {
            "sample_id": row.sample_id,
            "sha256": row.sha256,
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": float(Fraction(stream["avg_frame_rate"])),
            "codec": str(stream["codec_name"]),
            "container": str(data["format"]["format_name"]),
        }
    except (
        subprocess.SubprocessError,
        ValueError,
        ZeroDivisionError,
        KeyError,
        IndexError,
    ) as exc:
        return {"sample_id": row.sample_id, "sha256": row.sha256, "error": str(exc)}


def select_pairs(
    rows: list[SampleRecord],
    metadata: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[int, int, int]]], Counter[str]]:
    """Return matched fake/real pairs, per-generator buckets and exclusion reasons."""
    usable = {
        row.sample_id: row
        for row in rows
        if "error" not in metadata.get(row.sample_id, {"error": "missing"})
    }
    reasons: Counter[str] = Counter()
    buckets: dict[tuple[str, int, int], dict[str, dict[str, SampleRecord]]] = (
        defaultdict(lambda: {"fake": {}, "real": {}})
    )
    for row in rows:
        if row.label != 1:
            continue
        meta = metadata.get(row.sample_id)
        if meta is None or "error" in meta:
            reasons["fake_probe_failed"] += 1
            continue
        partner = usable.get(row.parent_id)
        if partner is None:
            reasons["no_paired_real"] += 1
            continue
        key = (row.generator, int(meta["width"]), int(meta["height"]))
        partner_meta = metadata[partner.sample_id]
        if (partner_meta["width"], partner_meta["height"]) != key[1:]:
            reasons["paired_real_size_differs"] += 1
            continue
        buckets[key]["fake"][row.sample_id] = row
        buckets[key]["real"][partner.sample_id] = partner
    pairs: list[dict[str, Any]] = []
    geometry: dict[str, list[tuple[int, int, int]]] = {}
    for (generator, width, height), group in sorted(buckets.items()):
        fakes = sorted(
            group["fake"].values(), key=lambda row: rank(row.sample_id, seed)
        )
        reals = sorted(
            group["real"].values(), key=lambda row: rank(row.sample_id, seed)
        )
        count = min(len(fakes), len(reals))
        if len(fakes) > len(reals):
            reasons["paired_real_supply_exhausted"] += len(fakes) - len(reals)
        geometry.setdefault(generator, []).append((width, height, count))
        pairs.extend(
            {
                "fake_sample_id": fake.sample_id,
                "real_sample_id": real.sample_id,
                "generator": generator,
                "width": width,
                "height": height,
            }
            for fake, real in zip(fakes[:count], reals[:count], strict=True)
        )
    return pairs, geometry, reasons


def summarise(
    rows: list[SampleRecord],
    metadata: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    geometry: dict[str, list[tuple[int, int, int]]],
    reasons: Counter[str],
) -> dict[str, Any]:
    """Record selection counts, matched composition and leftover size confounders."""
    fake_ids = {pair["fake_sample_id"] for pair in pairs}
    real_ids = {pair["real_sample_id"] for pair in pairs}
    generators = Counter(pair["generator"] for pair in pairs)
    reused = Counter(
        sum(pair["real_sample_id"] == real_id for pair in pairs) for real_id in real_ids
    )
    per_generator: dict[str, Any] = {}
    for generator in sorted(geometry):
        buckets = sorted(geometry[generator])
        per_generator[generator] = {
            "pairs": generators[generator],
            "matched_real": len(
                {
                    pair["real_sample_id"]
                    for pair in pairs
                    if pair["generator"] == generator
                }
            ),
            "buckets": [
                {"width": width, "height": height, "pairs": count}
                for width, height, count in buckets
            ],
            "input_fake": sum(
                row.generator == generator and row.label == 1 for row in rows
            ),
        }

    def sizes(ids: set[str]) -> dict[str, Any]:
        by_label: dict[str, dict[str, Any]] = {}
        for label in (0, 1):
            selected = [
                row for row in rows if row.sample_id in ids and row.label == label
            ]
            pairs_seen = Counter(
                (metadata[row.sample_id]["width"], metadata[row.sample_id]["height"])
                for row in selected
            )
            by_label[str(label)] = {
                "count": len(selected),
                "width_height": {
                    f"{width}x{height}": count
                    for (width, height), count in sorted(pairs_seen.items())
                },
                "fps": dict(
                    Counter(metadata[row.sample_id]["fps"] for row in selected)
                ),
                "codec": dict(
                    Counter(metadata[row.sample_id]["codec"] for row in selected)
                ),
            }
        return by_label

    return {
        "input_rows": len(rows),
        "probed": sum("error" not in value for value in metadata.values()),
        "probe_failures": {
            key: value["error"] for key, value in metadata.items() if "error" in value
        },
        "input_by_label": dict(Counter(str(row.label) for row in rows)),
        "input_by_generator": dict(
            Counter(row.generator for row in rows if row.label == 1)
        ),
        "unmatched_fakes": dict(sorted(reasons.items())),
        "pairs": len(pairs),
        "unique_samples": len(fake_ids | real_ids),
        "unique_fake": len(fake_ids),
        "unique_real": len(real_ids),
        "matched_sizes": sizes(fake_ids | real_ids),
        "per_generator": per_generator,
        "generators_without_pairs": sorted(
            {
                row.generator
                for row in rows
                if row.label == 1 and row.generator not in geometry
            }
        ),
        "real_reuse": {
            "distinct_real": len(real_ids),
            "pairs_per_real": {
                str(count): occurrences for count, occurrences in sorted(reused.items())
            },
            "unused_real": len(
                {row.sample_id for row in rows if row.label == 0} - real_ids
            ),
        },
    }


def canonical_geometry(manifest: Path, root: Path, timeout: float) -> dict[str, Any]:
    """Confirm the derived condition really has one identical geometry per label."""
    rows = load_manifest(manifest)
    result: dict[str, Any] = {}
    for label in (0, 1):
        found: Counter[str] = Counter()
        failures: dict[str, str] = {}
        for row in rows:
            if row.label != label:
                continue
            value = probe(row, root, timeout)
            if "error" in value:
                failures[row.sample_id] = str(value["error"])
                continue
            found[
                f"{value['width']}x{value['height']}:{value['codec']}"
                f":{value['container']}"
            ] += 1
        result[str(label)] = {
            "count": sum(found.values()),
            "geometry": dict(sorted(found.items())),
            "failures": failures,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)
    cache = args.output_dir / "metadata.jsonl"
    metadata: dict[str, dict[str, Any]] = {
        row["sample_id"]: row
        for row in (
            [json.loads(line) for line in cache.read_text().splitlines()]
            if cache.exists()
            else []
        )
    }
    missing = [
        row
        for row in rows
        if metadata.get(row.sample_id, {}).get("sha256") != row.sha256
    ]

    def probe_one(row: SampleRecord) -> dict[str, Any]:
        return probe(row, args.data_root, args.timeout_seconds)

    with ThreadPoolExecutor(max_workers=args.workers) as pool, cache.open("a") as file:
        for index, value in enumerate(pool.map(probe_one, missing), 1):
            metadata[value["sample_id"]] = value
            file.write(json.dumps(value, sort_keys=True) + "\n")
            if index % 500 == 0:
                file.flush()
                print(f"Probed {index}/{len(missing)}", flush=True)
    pairs, geometry, reasons = select_pairs(rows, metadata, args.seed)
    selected = {
        row.sample_id: replace(row, split="dev")
        for row in rows
        if row.sample_id in {pair["fake_sample_id"] for pair in pairs}
        or row.sample_id in {pair["real_sample_id"] for pair in pairs}
    }
    matched = sorted(selected.values(), key=lambda row: row.sample_id)
    matched_sha = write_manifest(matched, args.output_dir / "matched.jsonl")
    (args.output_dir / "pairs.jsonl").write_text(
        "".join(
            json.dumps(pair, sort_keys=True, ensure_ascii=False) + "\n"
            for pair in sorted(pairs, key=lambda pair: pair["fake_sample_id"])
        ),
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "data_root": str(args.data_root.resolve()),
        "seed": args.seed,
        "matching_rule": (
            "fake kept only when its registered paired real has identical probed "
            "width and height; per (generator, width, height) bucket take "
            "min(fake, real) by seeded rank(sample_id)"
        ),
        "matched_manifest": str((args.output_dir / "matched.jsonl").resolve()),
        "matched_manifest_sha256": matched_sha,
        "selection": summarise(rows, metadata, pairs, geometry, reasons),
        "original_condition": {
            "manifest": str((args.output_dir / "matched.jsonl").resolve()),
            "data_root": str(args.data_root.resolve()),
        },
        "canonical": prepare(
            args.output_dir / "matched.jsonl",
            args.data_root,
            args.output_dir / "canonical",
            list(CANONICAL_CONDITIONS),
            args.timeout_seconds,
        ),
        "canonical_geometry": {},
        "elapsed_seconds": time.monotonic() - started,
    }
    for name, condition in summary["canonical"]["conditions"].items():
        summary["canonical_geometry"][name] = canonical_geometry(
            Path(condition["path"]), Path(condition["data_root"]), args.timeout_seconds
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pairs": summary["selection"]["pairs"],
                "unique_samples": summary["selection"]["unique_samples"],
                "unmatched_fakes": summary["selection"]["unmatched_fakes"],
                "generators_without_pairs": summary["selection"][
                    "generators_without_pairs"
                ],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
