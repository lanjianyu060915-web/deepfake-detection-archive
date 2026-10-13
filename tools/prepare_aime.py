"""Import prompt-isolated AIME music for internal noncommercial research."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.data.split import (
    assign_splits,
    connected_groups,
    statistical_groups,
)
from deepfake_detection_archive.tracking import atomic_json


REPO = "disco-eth/AIME"
REVISION = "b84d4be5eda830b6eb714998569dba73530f2601"
SOURCE = "aime-music-research"
LICENSE = "CC-BY-4.0 AND CC-BY-NC-SA-4.0 AND LicenseRef-MTG-CC-per-track"
REAL = "MTG-Jamendo"
TRAIN_MODELS = {
    "MusicGen Small",
    "MusicGen Medium",
    "AudioLDM 2 Large",
    "Riffusion",
    "Mustango",
}
MODELS = TRAIN_MODELS | {
    REAL,
    "MusicGen Large",
    "AudioLDM 2 Music",
    "Stable Audio v1",
    "Stable Audio v2",
    "Suno v3",
    "Suno v3.5",
    "Udio",
}
RATE = 16000
SAMPLES = 64000
PROCESSING = {
    "sample_rate": RATE,
    "samples": SAMPLES,
    "channels": "mean",
    "resampling": "scipy.signal.resample_poly",
    "crop": "center",
    "padding": "symmetric_zero",
    "peak": 0.95,
    "subtype": "PCM_16",
}


def _description(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("description must contain three comma-separated tags")  # noqa: TRY004 -- input schema error
    tags = [tag.strip().casefold() for tag in value.split(",")]
    if len(tags) != 3 or not all(tags) or len(set(tags)) != 3:
        raise ValueError(f"description must contain three distinct tags: {value!r}")
    return ",".join(sorted(tags))


def _inputs(parquet_dir: Path, resources: Path) -> tuple[dict[str, Any], list[Path]]:
    specification: dict[str, Any] = json.loads(resources.read_text())
    if specification["repo"] != REPO or specification["revision"] != REVISION:
        raise ValueError("resources must refer to the fixed official AIME revision")
    paths = []
    for item in specification["files"]:
        path = parquet_dir / item["path"]
        if path.stat().st_size != item["size"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"official size/SHA256 mismatch: {path}")
        paths.append(path)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("resources must contain unique parquet paths")
    return specification, paths


def _real_descriptions(paths: list[Path]) -> set[str]:
    descriptions: set[str] = set()
    ids: set[str] = set()
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["id", "model", "description"]
        ):
            for row in batch.to_pylist():
                if not isinstance(row["id"], str) or not row["id"].strip():
                    raise ValueError("AIME id must be a nonempty string")
                if row["id"] in ids:
                    raise ValueError(f"duplicate AIME row id: {row['id']}")
                ids.add(row["id"])
                if row["model"] not in MODELS:
                    raise ValueError(f"unknown AIME model: {row['model']}")
                description = _description(row["description"])
                if row["model"] == REAL:
                    descriptions.add(description)
    if not descriptions:
        raise ValueError("no real MTG-Jamendo descriptions in selected resources")
    return descriptions


def _normalize(raw: bytes, destination: Path) -> dict[str, Any]:
    waveform, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("empty or non-finite source audio")
    original = {
        "sample_rate": rate,
        "frames": len(waveform),
        "duration_seconds": len(waveform) / rate,
        "channels": waveform.shape[1],
    }
    mono = waveform.mean(axis=1)
    divisor = math.gcd(rate, RATE)
    mono = resample_poly(mono, RATE // divisor, rate // divisor)
    if len(mono) < SAMPLES:
        missing = SAMPLES - len(mono)
        mono = np.pad(mono, (missing // 2, missing - missing // 2))
    start = (len(mono) - SAMPLES) // 2
    segment = mono[start : start + SAMPLES]
    peak = float(np.max(np.abs(segment)))
    if peak == 0:
        raise ValueError("silent center segment")
    sf.write(destination, segment * (0.95 / peak), RATE, subtype="PCM_16")
    return {"original": original, "pre_normalization_peak": peak}


def _row(
    row: dict[str, Any], real_descriptions: set[str], output: Path
) -> tuple[SampleRecord, dict[str, Any]]:
    description = _description(row["description"])
    digest = hashlib.sha256(row["id"].encode()).hexdigest()
    record = SampleRecord(
        sample_id=f"aime:{row['id']}",
        path=f"media/{digest}.wav",
        modality="audio",
        label=0 if row["model"] == REAL else 1,
        source=SOURCE,
        content_type="music",
        generator="unknown" if row["model"] == REAL else row["model"],
        original_id=f"aime-row:{row['id']}",
        group_id=f"aime-prompt:{hashlib.sha256(description.encode()).hexdigest()}",
        license_id=LICENSE,
    )
    raw: bytes = row["audio"]["bytes"]
    provenance: dict[str, Any] = {
        "sample_id": record.sample_id,
        "row_id": row["id"],
        "model": row["model"],
        "description": row["description"],
        "canonical_description": description,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "artist": "unknown",
        "album": "unknown",
        "processing": PROCESSING,
    }
    if description not in real_descriptions:
        return replace(
            record, status="excluded", failure_reason="no_matching_real_description"
        ), provenance
    try:
        provenance.update(_normalize(raw, output / record.path))
    except (OSError, ValueError, sf.LibsndfileError) as exc:
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        return replace(
            record,
            status="decode_failed",
            decode_status="failed",
            failure_reason=provenance["error"],
        ), provenance
    return replace(
        record,
        sha256=file_sha256(output / record.path),
        container="wav",
        codec="pcm_s16le",
        duration_seconds=4.0,
        sample_rate=RATE,
        decode_status="ok",
    ), provenance


def _counts(records: list[SampleRecord]) -> dict[str, Any]:
    return {
        "total": len(records),
        "split": dict(sorted(Counter(r.split for r in records).items())),
        "status": dict(sorted(Counter(r.status for r in records).items())),
        "decode_status": dict(
            sorted(Counter(r.decode_status for r in records).items())
        ),
        "model": dict(
            sorted(
                Counter(REAL if r.label == 0 else r.generator for r in records).items()
            )
        ),
        "split_label": dict(
            sorted(
                Counter(
                    f"{r.split}/{'real' if r.label == 0 else 'fake'}" for r in records
                ).items()
            )
        ),
    }


def _preserve_splits(
    records: list[SampleRecord], previous: list[SampleRecord]
) -> tuple[list[SampleRecord], int]:
    # Separate IDs keep historical hashes/groups even when a current row overlaps.
    # Parent edges connect each historical anchor to its original sample ID.
    anchors = [
        replace(r, sample_id=f"previous:{r.sample_id}", parent_id=r.sample_id)
        for r in previous
    ]
    groups = connected_groups(records + anchors)
    fixed: dict[str, set[str]] = defaultdict(set)
    for anchor in anchors:
        if anchor.split != "unassigned":
            fixed[groups[anchor.sample_id]].add(anchor.split)
    conflicts = {group for group, splits in fixed.items() if len(splits) > 1}
    conflicts.update(
        groups[r.sample_id]
        for r in anchors
        if r.failure_reason == "previous_split_conflict"
    )
    result = []
    for record in records:
        group = groups[record.sample_id]
        if group in conflicts:
            result.append(
                replace(
                    record,
                    split="unassigned",
                    status="excluded",
                    failure_reason="previous_split_conflict",
                )
            )
        elif splits := fixed.get(group):
            result.append(replace(record, split=next(iter(splits))))
        else:
            result.append(record)
    return result, len(conflicts & {groups[r.sample_id] for r in records})


def prepare(
    parquet_dir: Path,
    resources: Path,
    output: Path,
    *,
    seed: int = 42,
    previous_manifest: Path | None = None,
) -> dict[str, Any]:
    previous = load_manifest(previous_manifest) if previous_manifest else []
    if any(r.source != SOURCE or r.split == "test" for r in previous):
        raise ValueError("previous manifest must contain AIME research splits")
    specification, paths = _inputs(parquet_dir, resources)
    descriptions = _real_descriptions(paths)
    output.mkdir(parents=True, exist_ok=False)
    (output / "media").mkdir()
    records = []
    with (output / "provenance.jsonl").open("w", encoding="utf-8") as stream:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
                record, provenance = _row(batch.to_pylist()[0], descriptions, output)
                records.append(record)
                provenance["parquet"] = path.relative_to(parquet_dir).as_posix()
                stream.write(json.dumps(provenance, ensure_ascii=False) + "\n")
    conflicts = 0
    if previous:
        records, conflicts = _preserve_splits(records, previous)
    # Failed media keeps its prompt relation; the splitter counts only available
    # records. Include reserved generators here so duplicate hashes cannot leak.
    selected = assign_splits([r for r in records if r.status != "excluded"], seed=seed)
    main = [
        r
        for r in selected
        if (r.label == 0 or r.generator in TRAIN_MODELS)
        and (r.status == "available" or r.split != "train")
    ]
    unseen = [
        r
        for r in selected
        if r.split in {"dev", "holdout"}
        and (r.label == 0 or r.generator not in TRAIN_MODELS)
    ]
    by_id = {r.sample_id: r for r in selected}
    inventory = [by_id.get(r.sample_id, r) for r in records]
    inventory = [
        replace(
            r, status="excluded", failure_reason="generator_reserved_for_evaluation"
        )
        if r.split == "train" and r.label == 1 and r.generator not in TRAIN_MODELS
        else r
        for r in inventory
    ]
    hashes = {
        name: write_manifest(items, output / f"{name}.jsonl")
        for name, items in {
            "manifest": main,
            "unseen": unseen,
            "inventory": inventory,
        }.items()
    }
    source = SourceRecord(
        source=SOURCE,
        url=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
        version=REVISION,
        license_id=LICENSE,
        license_url="https://mtg.github.io/mtg-jamendo-dataset/#license",
        downloaded_at=datetime.now(UTC).isoformat(),
        allowed_uses=("training", "research"),
        license_status="approved",
    )
    (output / "sources.jsonl").write_text(
        json.dumps(asdict(source), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "source": SOURCE,
        "repo": REPO,
        "revision": REVISION,
        "seed": seed,
        "previous_manifest": str(previous_manifest) if previous_manifest else None,
        "previous_split_conflicts": conflicts,
        "resources": specification["files"],
        "processing": PROCESSING,
        "train_models": sorted(TRAIN_MODELS),
        "real_descriptions": len(descriptions),
        "connected_groups": len(set(connected_groups(selected).values())),
        "known_statistical_groups": len(
            set(statistical_groups(selected).values()) - {"unknown"}
        ),
        "grouping_evidence": {
            "group_id": "prompt_label_combination",
            "independent_artists": "unknown",
            "independent_source_recordings": "unknown",
        },
        "manifest": _counts(main),
        "unseen": _counts(unseen),
        "inventory": _counts(inventory),
        "sha256": hashes,
        "exclusions": dict(
            Counter(r.failure_reason for r in inventory if r.status != "available")
        ),
        "competition_approved": False,
        "license_id": LICENSE,
        "limitations": [
            "仅批准内部非商业研究；原曲逐曲许可未知，正式比赛未批准。",
            "按标签组合保守隔离，不代表同曲派生；artist/album未知。",
            "MusicGen Large与训练中的Small/Medium同家族，不是全新家族。",
        ],
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--resources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--previous-manifest",
        type=Path,
        help="Preserve prior AIME splits; prefer the complete inventory.jsonl",
    )
    arguments = parser.parse_args()
    print(json.dumps(prepare(**vars(arguments)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
