"""Build a fixed H.264 augmentation view of one controlled video manifest.

Training-time compression augmentation is expressed as derived media plus one
manifest, so every epoch reads ready files instead of running FFmpeg per item.
Arms are assigned per source clip, which keeps a fake and its paired real on the
same arm; dev rows stay on unmodified bytes because the same-domain dev split
chooses the epoch. Originals are hardlinked into the output root so a single
data root covers both the clean and the transcoded arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from prepare_robustness import transform

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)


CLEAN = "clean"
ARMS = ("clean", "h264-crf23", "h264-crf35")
ARM_CONDITIONS = {"h264-crf23": "video_h264_crf23", "h264-crf35": "video_h264_crf35"}


def clip_key(row: SampleRecord) -> str:
    """Return the source clip identity shared by a fake and its paired real."""
    name = Path(row.path).name
    if not name:
        raise ValueError(f"cannot derive a clip key from {row.path!r}")
    return name


def arm_for(key: str, seed: int, arms: Sequence[str]) -> str:
    """Assign one arm per clip from a seeded hash of the clip ID.

    Indexing into the sorted arms keeps the condition tied to the clip, not to
    the order in which the caller lists the arms.
    """
    digest = hashlib.sha256(f"video-augmentation:{seed}:{key}".encode()).digest()
    ordered = sorted(arms)
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def plan(
    records: list[SampleRecord], seed: int, arms: Sequence[str]
) -> list[tuple[str, SampleRecord]]:
    """Pair every row with its arm; dev rows keep the unmodified clean arm."""
    return [
        (
            CLEAN if row.split == "dev" else arm_for(clip_key(row), seed, arms),
            row,
        )
        for row in records
    ]


def _link_originals(
    records: list[SampleRecord], data_root: Path, output_dir: Path
) -> Counter[str]:
    """Publish original bytes under the derived root without duplicating them."""
    modes: Counter[str] = Counter()
    for row in records:
        target = output_dir / row.path
        if target.exists():
            modes["reused"] += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = data_root / row.path
        try:
            os.link(source, target)
            modes[
                "linked" if target.stat().st_ino == source.stat().st_ino else "copied"
            ] += 1
        except OSError:
            shutil.copy2(source, target)
            if file_sha256(target) != row.sha256:
                raise ValueError(f"copied original differs from {row.sha256}") from None
            modes["copied"] += 1
    return modes


def prepare_augmentation(  # noqa: PLR0912, PLR0913, PLR0917 -- explicit inputs
    manifest: Path,
    data_root: Path,
    output_dir: Path,
    arms: Sequence[str] = ARMS,
    seed: int = 42,
    timeout_seconds: float = 60,
) -> dict[str, Any]:
    """Validate once, then write hardlinked originals and one derived arm per row."""
    started = time.monotonic()
    records = load_manifest(manifest)
    splits = {row.split for row in records}
    if not splits or not splits <= {"train", "dev"}:
        raise ValueError("augmentation preparation requires train and/or dev rows")
    if "train" not in splits:
        raise ValueError("augmentation preparation requires at least one train row")
    if {row.modality for row in records} != {"video"}:
        raise ValueError("augmentation preparation requires video media")
    for split in sorted(splits):
        if {row.label for row in records if row.split == split} != {0, 1}:
            raise ValueError(f"{split} requires both binary labels 0 and 1")
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ValueError("choose distinct arms from " + ", ".join(ARMS))
    if CLEAN not in arms:
        raise ValueError("arms must include clean: dev rows need unmodified bytes")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not timeout_seconds > 0:
        raise ValueError("timeout_seconds must be positive")
    version = subprocess.run(
        ["ffmpeg", "-version"],  # noqa: S607 - configured local FFmpeg
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    ).stdout.splitlines()[0]
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    baseline_hash = write_manifest(records, output_dir / "baseline.jsonl")
    assigned = plan(records, seed, arms)
    media = output_dir / "media"
    modes = _link_originals(records, data_root, output_dir)
    per_arm: dict[str, dict[str, Any]] = {}
    combined: list[SampleRecord] = []
    for arm in arms:
        selected = [row for assigned_arm, row in assigned if assigned_arm == arm]
        failures: dict[str, str] = {}
        if arm == CLEAN:
            combined.extend(selected)
        else:
            media.mkdir(parents=True, exist_ok=True)
            condition = ARM_CONDITIONS[arm]
            for row in selected:
                derived = transform(
                    replace(row, sample_id=f"{row.sample_id}#{arm}"),
                    data_root,
                    output_dir,
                    condition,
                    timeout_seconds,
                )
                combined.append(derived)
                if derived.status != "available":
                    failures[derived.sample_id] = derived.failure_reason
        per_arm[arm] = {
            "condition": ARM_CONDITIONS.get(arm),
            "rows": len(selected),
            "success_count": len(selected) - len(failures),
            "failure_count": len(failures),
            "failures": failures,
            "split_label_counts": {
                f"{split}/{label}": sum(
                    row.split == split and row.label == label for row in selected
                )
                for split in sorted(splits)
                for label in (0, 1)
            },
            "generators": dict(
                sorted(Counter(row.generator for row in selected).items())
            ),
        }
    manifest_path = output_dir / "manifest.jsonl"
    digest = write_manifest(combined, manifest_path)
    summary: dict[str, Any] = {
        "input_manifest": str(manifest.resolve()),
        "input_manifest_sha256": baseline_hash,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "ffmpeg_version": version,
        "seed": seed,
        "arms": list(arms),
        "arm_rule": (
            "arm = sha256('video-augmentation:<seed>:<clip name>') mod len(arms); "
            "dev rows stay clean; a fake and its paired real share the clip name"
        ),
        "conditions": {arm: ARM_CONDITIONS[arm] for arm in arms if arm != CLEAN},
        "original_modes": dict(modes),
        "baseline": {
            "path": str(output_dir / "baseline.jsonl"),
            "sha256": baseline_hash,
            "data_root": str(data_root),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": digest,
            "data_root": str(output_dir),
            "sample_count": len(combined),
        },
        "per_arm": per_arm,
        "failure_count": sum(entry["failure_count"] for entry in per_arm.values()),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", action="append", choices=ARMS, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    args = parser.parse_args()
    summary = prepare_augmentation(
        args.manifest,
        args.data_root,
        args.output_dir,
        tuple(args.arm) if args.arm else ARMS,
        args.seed,
        args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "manifest": summary["manifest"],
                "per_arm": {
                    arm: {
                        "rows": entry["rows"],
                        "failure_count": entry["failure_count"],
                        "split_label_counts": entry["split_label_counts"],
                    }
                    for arm, entry in summary["per_arm"].items()
                },
                "original_modes": summary["original_modes"],
                "failure_count": summary["failure_count"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
