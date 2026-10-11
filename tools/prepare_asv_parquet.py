"""Materialize pinned ASVspoof LA mirror shards without decoding or re-encoding.

Official train/dev protocols must be obtained separately. Every mirror row is
checked against those protocols before writing media. The resulting LA layout
is consumed by prepare_asvspoof2019.py and then prepare_asv_main.py; this tool
does not replace either manifest creation or decode auditing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import file_sha256


MIRROR_REPO = "Bisher/ASVspoof_2019_LA"
MIRROR_REVISION = "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c"
SHARDS = {
    "train": (
        "train-00000-of-00001.parquet",
        "b4eea1063bbcfa0c1cef1b69a96ad8b787c32f662005562b899cd4b461739619",
        25380,
    ),
    "dev": (
        "validation-00000-of-00001.parquet",
        "9d2c340bb7f04c2d63ac018b224ecdb8c1855c789247232608b650f86c189b16",
        24844,
    ),
}
PROTOCOLS = {
    "train": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
}
SCHEMA = pa.schema(
    [
        ("speaker_id", pa.string()),
        ("audio_file_name", pa.string()),
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("system_id", pa.string()),
        ("key", pa.int64()),
    ]
)
METADATA_COLUMNS = ["speaker_id", "audio_file_name", "system_id", "key"]


def _protocol_rows(path: Path, split: str) -> dict[str, tuple[str, str, int]]:
    records: dict[str, tuple[str, str, int]] = {}
    prefix = "T" if split == "train" else "D"
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{number}: expected five protocol columns")
        speaker, sample_id, placeholder, system, key = fields
        if (
            re.fullmatch(rf"LA_{prefix}_\d+", sample_id) is None
            or re.fullmatch(r"LA_\d+", speaker) is None
            or placeholder != "-"
            or key not in {"bonafide", "spoof"}
            or (key == "bonafide" and system != "-")
            or (key == "spoof" and system not in {f"A{i:02d}" for i in range(1, 7)})
        ):
            raise ValueError(f"{path}:{number}: invalid train/dev protocol row")
        if sample_id in records:
            raise ValueError(f"{path}:{number}: duplicate protocol sample {sample_id}")
        records[sample_id] = (speaker, system, int(key == "spoof"))
    return records


def _check_metadata(
    parquet: Any, protocol: dict[str, tuple[str, str, int]], batch_size: int
) -> dict[str, int]:
    if not parquet.schema_arrow.equals(SCHEMA, check_metadata=False):
        raise ValueError("mirror Parquet schema differs from the pinned schema")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=METADATA_COLUMNS):
        for row in batch.to_pylist():
            sample_id = row["audio_file_name"]
            expected = protocol.get(sample_id)
            actual = (row["speaker_id"], row["system_id"], row["key"])
            if expected is None or actual != expected:
                raise ValueError(f"mirror metadata mismatch with protocol: {sample_id}")
            if sample_id in seen:
                raise ValueError(f"duplicate mirror sample: {sample_id}")
            seen.add(sample_id)
            counts["fake" if row["key"] else "real"] += 1
    if seen != set(protocol):
        raise ValueError("mirror samples do not cover the complete official protocol")
    return dict(sorted(counts.items()))


def _write_media(parquet: Any, destination: Path, batch_size: int) -> int:
    destination.mkdir(parents=True)
    total_bytes = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=["audio_file_name", "audio"]
    ):
        for row in batch.to_pylist():
            sample_id = row["audio_file_name"]
            audio = row["audio"]
            if (
                not isinstance(audio, dict)
                or audio.get("path") != f"{sample_id}.flac"
                or not isinstance(audio.get("bytes"), bytes)
                or not audio["bytes"].startswith(b"fLaC")
            ):
                raise ValueError(f"invalid embedded FLAC/path: {sample_id}")
            with (destination / f"{sample_id}.flac").open("xb") as stream:
                stream.write(audio["bytes"])
            total_bytes += len(audio["bytes"])
    return total_bytes


def materialize(
    parquet_root: Path,
    protocol_root: Path,
    output_root: Path,
    *,
    batch_size: int = 64,
) -> dict[str, object]:
    """Validate pinned inputs, then write new media and a completion summary."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output_root = output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("output root already exists; choose a new directory")
    started = time.monotonic()
    parquet_root = parquet_root.resolve()
    protocol_root = protocol_root.resolve()
    shard_hashes = {}
    # Hash each immutable shard exactly once, before reading its Parquet content.
    for name, expected, _ in SHARDS.values():
        checksum = file_sha256(parquet_root / name)
        if checksum != expected:
            raise ValueError(f"pinned Parquet SHA256 mismatch: {name}")
        shard_hashes[name] = checksum
    attribution = [
        protocol_root / name
        for name in ("README.LA.txt", "README.txt", "README", "LICENSE", "LICENSE.txt")
        if (protocol_root / name).is_file()
    ]
    if not attribution:
        raise ValueError("official protocol root must include README or license text")
    parquets = {}
    counts = {}
    protocol_hashes = {}
    for split, (name, _, expected_count) in SHARDS.items():
        protocol_path = protocol_root / PROTOCOLS[split]
        protocol = _protocol_rows(protocol_path, split)
        parquet = pq.ParquetFile(parquet_root / name)
        if (
            len(protocol) != expected_count
            or parquet.metadata.num_rows != expected_count
        ):
            raise ValueError(f"{split}: official protocol or mirror row count mismatch")
        counts[split] = _check_metadata(parquet, protocol, batch_size)
        protocol_hashes[PROTOCOLS[split]] = file_sha256(protocol_path)
        parquets[split] = parquet
    # Exclusive creation prevents mixing old or interrupted output with this run.
    # A failed write leaves a partial directory without a completion summary;
    # rerun into a new directory instead of treating it as an import/resume.
    output_root.mkdir(parents=True)
    bytes_written = {
        split: _write_media(
            parquet, output_root / f"ASVspoof2019_LA_{split}" / "flac", batch_size
        )
        for split, parquet in parquets.items()
    }
    for relative in PROTOCOLS.values():
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(protocol_root / relative, destination)
    for path in attribution:
        shutil.copyfile(path, output_root / path.name)
    elapsed = time.monotonic() - started
    summary: dict[str, object] = {
        "schema_version": 1,
        "source_kind": "community-mirror-repack",
        "mirror_repo": MIRROR_REPO,
        "mirror_revision": MIRROR_REVISION,
        "parquet_root": str(parquet_root),
        "parquet_sha256": shard_hashes,
        "protocol_root": str(protocol_root),
        "protocol_sha256": protocol_hashes,
        "attribution_sha256": {p.name: file_sha256(p) for p in attribution},
        "output_root": str(output_root),
        "split_label_counts": counts,
        "media_bytes_written": bytes_written,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "label_mapping": {"bonafide": 0, "spoof": 1},
        "protocol_fields_verified": ["sample_id", "speaker_id", "system_id", "key"],
        "audio_bytes_preserved": True,
        "decode_verified": False,
        "official_audio_byte_equality_verified": False,
        "eval_accessed": False,
        "competition_approved": False,
        "next_steps": "prepare_asvspoof2019.py, then prepare_asv_main.py",
    }
    (output_root / "materialization-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    try:
        summary = materialize(
            args.parquet_root,
            args.protocol_root,
            args.output_root,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"materialization failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
