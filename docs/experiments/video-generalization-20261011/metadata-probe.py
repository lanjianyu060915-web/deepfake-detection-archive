"""Probe decoded-media metadata of an imported diagnostic set without touching frames.

Reports per-class and per-generator width, height, frame rate and duration so that a
metadata-only shortcut can be judged before trusting any content-model score. The
width-only figure is an oracle threshold chosen on this same set, so it is an
optimistic bound, not a generalising baseline.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import subprocess
from pathlib import Path


def probe(path: Path) -> dict[str, object]:
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,codec_name,duration",
            "-show_entries", "format=format_name,duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if done.returncode != 0:
        raise ValueError(done.stderr.strip() or "ffprobe failed")
    payload = json.loads(done.stdout)
    stream = (payload.get("streams") or [{}])[0]
    container = payload.get("format") or {}
    rate = stream.get("avg_frame_rate") or "0/0"
    numerator, _, denominator = rate.partition("/")
    fps = float(numerator) / float(denominator) if float(denominator or 0) else None
    duration = stream.get("duration") or container.get("duration")
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "duration": float(duration) if duration is not None else None,
        "codec": stream.get("codec_name"),
        "container": container.get("format_name"),
    }


def oracle_width_eer(rows: list[dict[str, object]]) -> float | None:
    """Optimistic width-only bound with the threshold chosen on the same rows."""
    reals = sorted(
        (row["width"], row["label"]) for row in rows if row["width"] is not None
    )
    if {label for _, label in reals} != {0, 1}:
        return None
    best = 1.0
    candidates = sorted({width for width, _ in reals})
    for threshold in candidates:
        false_positive = sum(1 for w, label in reals if label == 0 and w >= threshold)
        false_negative = sum(1 for w, label in reals if label == 1 and w < threshold)
        real_total = sum(1 for _, label in reals if label == 0)
        fake_total = sum(1 for _, label in reals if label == 1)
        rate = max(false_positive / real_total, false_negative / fake_total)
        best = min(best, rate)
    return best


def summarise(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    real = [row for row in rows if row["label"] == 0]
    fake = [row for row in rows if row["label"] == 1]

    def stats(group: list[dict[str, object]], key: str) -> dict[str, object]:
        values = [row[key] for row in group if row[key] is not None]
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "distinct": len(set(values)),
        }

    return {
        "name": name,
        "rows": len(rows),
        "real": len(real),
        "fake": len(fake),
        "width": {"real": stats(real, "width"), "fake": stats(fake, "width")},
        "height": {"real": stats(real, "height"), "fake": stats(fake, "height")},
        "fps": {"real": stats(real, "fps"), "fake": stats(fake, "fps")},
        "duration": {
            "real": stats(real, "duration"),
            "fake": stats(fake, "duration"),
        },
        "codec": {
            "real": dict(collections.Counter(row["codec"] for row in real)),
            "fake": dict(collections.Counter(row["codec"] for row in fake)),
        },
        "oracle_width_eer": oracle_width_eer(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    probed: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    for index, row in enumerate(rows, 1):
        try:
            metadata = probe(args.data_root / row["path"])
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            failures[row["sample_id"]] = str(exc)
            continue
        probed.append(
            {
                "sample_id": row["sample_id"],
                "label": row["label"],
                "generator": row["generator"],
                **metadata,
            }
        )
        if index % 200 == 0:
            print(f"probed {index}/{len(rows)}", flush=True)
    parents: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        if row["label"] == 1 and row.get("parent_id", "unknown") != "unknown":
            parents[str(row["parent_id"])].add(str(row["generator"]))
    by_generator: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for entry in probed:
        if entry["label"] == 1:
            by_generator[str(entry["generator"])].append(entry)
        else:
            for generator in parents.get(str(entry["sample_id"]), ()):
                by_generator[generator].append(entry)
    summary = {
        "manifest": str(args.manifest),
        "rows": len(rows),
        "probed": len(probed),
        "failures": failures,
        "overall": summarise("all", probed),
        "per_generator": {
            key: summarise(key, value) for key, value in sorted(by_generator.items())
        },
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    probe_table = args.output.with_name("metadata.jsonl")
    probe_table.write_text(
        "".join(
            json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"
            for entry in sorted(probed, key=lambda item: str(item["sample_id"]))
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_generator"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
