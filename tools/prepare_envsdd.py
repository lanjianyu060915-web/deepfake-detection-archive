"""Import UrbanSound8K EnvSDD families for noncommercial detection research."""

from __future__ import annotations

import argparse
import csv
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
from deepfake_detection_archive.data.split import assign_splits, connected_groups
from deepfake_detection_archive.tracking import atomic_json


REPO = "EnvSDD/EnvSDD"
REVISION = "b0832726deb55484dbce869c6f09c41f7e9e5fbb"
SOURCE = "envsdd-urbansound8k-research"
LICENSE = "CC-BY-4.0 AND CC-BY-NC-3.0"
MEMBERS = {
    ("real", "real"),
    ("audiogen", "tta"),
    ("audioldm1", "tta"),
    ("audioldm2", "tta"),
    ("audioldm1", "ata"),
}
RATE = 16000
SAMPLES = 64000


def _normalize(raw: bytes, destination: Path) -> dict[str, Any]:
    waveform, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("empty or non-finite source audio")
    original = {
        "sample_rate": rate,
        "frames": len(waveform),
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


def _prepare_row(
    row: dict[str, Any],
    metadata: dict[str, str],
    uploader: str,
    reason: str,
    output: Path,
) -> tuple[SampleRecord, dict[str, Any]]:
    filename = row["filename"]
    model, attack = row["generative_model"], row["attack_type"]
    if (model, attack) not in MEMBERS or row["label"] != (
        "real" if model == "real" else "fake"
    ):
        raise ValueError(f"unexpected EnvSDD member: {filename}/{model}/{attack}")
    stem = f"{Path(filename).stem}-{model}-{attack}"
    sample_id = f"envsdd:{stem}"
    raw_path = output / "raw" / f"{stem}.wav"
    raw_path.write_bytes(row["audio"]["bytes"])
    record = SampleRecord(
        sample_id=sample_id,
        path=f"media/{stem}.wav",
        modality="audio",
        label=0 if model == "real" else 1,
        source=SOURCE,
        content_type="environment",
        generator="unknown" if model == "real" else model,
        original_id=f"freesound:{metadata['fsID']}",
        group_id=f"freesound-uploader:{uploader.casefold()}",
        parent_id="unknown"
        if model == "real"
        else f"envsdd:{Path(filename).stem}-real-real",
        derivation_group_id=f"freesound:{metadata['fsID']}",
        license_id=LICENSE,
    )
    provenance: dict[str, Any] = {
        "sample_id": sample_id,
        "filename": filename,
        "generative_model": model,
        "attack_type": attack,
        "metadata": metadata,
        "uploader": uploader,
        "event": row.get("event"),
        "scene": row.get("scene"),
        "caption": row.get("caption"),
        "raw_path": raw_path.relative_to(output).as_posix(),
        "raw_sha256": file_sha256(raw_path),
        "raw_bytes": raw_path.stat().st_size,
    }
    if reason:
        return replace(record, status="excluded", failure_reason=reason), provenance
    try:
        provenance.update(_normalize(row["audio"]["bytes"], output / record.path))
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


def _exclude_incomplete(
    records: list[SampleRecord],
    families: dict[str, list[int]],
    members: dict[str, set[tuple[str, str]]],
) -> None:
    for name, indices in families.items():
        reason = ""
        if members[name] != MEMBERS:
            reason = "incomplete_five_member_family"
        elif any(records[i].status != "available" for i in indices):
            reason = "family_member_unavailable"
        if reason:
            for i in indices:
                if records[i].status == "available":
                    records[i] = replace(
                        records[i], status="excluded", failure_reason=reason
                    )


def prepare(  # noqa: PLR0913, PLR0915 -- one streaming dataset import
    parquet_dir: Path,
    credits_path: Path,
    urban_metadata: Path,
    exclude_manifest: Path,
    output: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    parquet_paths = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_paths:
        raise ValueError("no parquet files found")
    uploaders = {}
    for line in credits_path.read_text(encoding="utf-8").splitlines():
        identity, separator, uploader = line.partition(" by ")
        if separator and identity.isdigit():
            uploaders[identity] = uploader.strip()
    with urban_metadata.open(newline="", encoding="utf-8") as stream:
        metadata = {row["slice_file_name"]: row for row in csv.DictReader(stream)}
    excluded = load_manifest(exclude_manifest)
    old_originals = {r.original_id for r in excluded}
    old_uploaders = {r.group_id.casefold() for r in excluded}
    inputs = {
        str(path): file_sha256(path)
        for path in [*parquet_paths, credits_path, urban_metadata, exclude_manifest]
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "media").mkdir()
    records: list[SampleRecord] = []
    families: dict[str, list[int]] = defaultdict(list)
    members: dict[str, set[tuple[str, str]]] = defaultdict(set)
    methods: dict[str, str] = {}
    input_counts: Counter[str] = Counter()
    with (output / "provenance.jsonl").open("w", encoding="utf-8") as stream:
        for path in parquet_paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    input_counts[row["dataset"]] += 1
                    if row["dataset"] != "UrbanSound8K":
                        continue
                    name = row["filename"]
                    member = (row["generative_model"], row["attack_type"])
                    if member in members[name]:
                        raise ValueError(f"duplicate EnvSDD member: {name}/{member}")
                    members[name].add(member)
                    meta = metadata[name]
                    uploader = uploaders[meta["fsID"]]
                    reason = ""
                    if float(meta["end"]) - float(meta["start"]) < 3.99:
                        reason = "original_slice_shorter_than_3.99s"
                    elif (
                        f"freesound:{meta['fsID']}" in old_originals
                        or f"freesound-uploader:{uploader.casefold()}" in old_uploaders
                    ):
                        reason = "overlap_with_previous_dataset"
                    record, provenance = _prepare_row(
                        row, meta, uploader, reason, output
                    )
                    families[name].append(len(records))
                    records.append(record)
                    methods[record.sample_id] = f"{member[0]}/{member[1]}"
                    provenance["parquet"] = str(path)
                    stream.write(json.dumps(provenance, ensure_ascii=False) + "\n")
    _exclude_incomplete(records, families, members)
    eligible = [record for record in records if record.status == "available"]
    # Publish the complete failure inventory even if too few groups remain to split.
    write_manifest(records, output / "inventory.jsonl")
    split_records = assign_splits(eligible, seed=seed)
    by_id = {record.sample_id: record for record in split_records}
    inventory_hash = write_manifest(
        [by_id.get(record.sample_id, record) for record in records],
        output / "inventory.jsonl",
    )
    manifest_hash = write_manifest(split_records, output / "manifest.jsonl")
    source = SourceRecord(
        source=SOURCE,
        url=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
        version=REVISION,
        license_id=LICENSE,
        license_url="https://creativecommons.org/licenses/by-nc/3.0/",
        downloaded_at=datetime.now(UTC).isoformat(),
        allowed_uses=("training", "research"),
        license_status="approved",
    )
    (output / "sources.jsonl").write_text(
        json.dumps(asdict(source)) + "\n", encoding="utf-8"
    )
    summary: dict[str, Any] = {
        "source": SOURCE,
        "revision": REVISION,
        "input_sha256": inputs,
        "seed": seed,
        "input_dataset_counts": dict(input_counts),
        "urban_families": len(families),
        "urban_samples": len(records),
        "eligible_samples": len(split_records),
        "status_counts": dict(Counter(r.status for r in records)),
        "failure_reasons": dict(
            Counter(r.failure_reason for r in records if r.status != "available")
        ),
        "incomplete_families": sum(value != MEMBERS for value in members.values()),
        "split_label_counts": dict(
            Counter(f"{r.split}/{r.label}" for r in split_records)
        ),
        "generator_method_counts": dict(
            Counter(methods[r.sample_id] for r in split_records)
        ),
        "groups": len(set(connected_groups(split_records).values())),
        "decode_attempted": sum(r.decode_status != "unknown" for r in records),
        "decode_failed": sum(r.decode_status == "failed" for r in records),
        "normalized_coverage": len(split_records) / len(records),
        "manifest_sha256": manifest_hash,
        "inventory_sha256": inventory_hash,
        "preprocessing": "mono, polyphase 16kHz, center crop/pad 64000 samples, peak 0.95, PCM16",
        "training_license_scope": "internal noncommercial detection research only",
        "competition_approved": False,
        "upstream_split": "selected train/validation parquet; repartitioned by connected source groups",
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parquet-dir",
        "credits",
        "urban-metadata",
        "exclude-manifest",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.parquet_dir,
                args.credits,
                args.urban_metadata,
                args.exclude_manifest,
                args.output,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
