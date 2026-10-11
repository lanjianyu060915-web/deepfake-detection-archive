"""Import the pinned ASVspoof 2019 LA eval partition for unseen-attack scoring.

The official CM eval protocol supplies labels and attack IDs; media bytes come
from the pinned public mirror shard. Every mirror row is verified against the
official protocol before any media is written, the protocol file is copied
verbatim, and each attack keeps its own generator label and speaker group. No
train/dev media, decode audit, near-duplicate search or model scoring happens
here, so this import cannot be used to claim an official ASVspoof score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import (
    UNKNOWN,
    SampleRecord,
    SourceRecord,
    file_sha256,
    manifest_hash,
    write_manifest,
)


SOURCE = "asvspoof2019-la-eval-official"
SOURCE_URL = "https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45"
LICENSE_URL = "https://opendatacommons.org/licenses/by/1-0/"
MIRROR_REPO = "Bisher/ASVspoof_2019_LA"
MIRROR_REVISION = "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c"
EVAL_SHARD = "test-00000-of-00001.parquet"
EVAL_SHARD_SHA256 = (
    "8159935a94426bc36308278b23cecd8c4ca2a15b778a1087a4a0441d079c86af"
)
EVAL_PROTOCOL = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
EVAL_PROTOCOL_SHA256 = (
    "84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036"
)
SPLIT = "test"
ATTACKS = tuple(f"A{index:02d}" for index in range(7, 20))
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
ROOT_FILES = ("README.LA.txt", "README.txt", "README", "LICENSE", "LICENSE.txt")


def _protocol_rows(path: Path) -> dict[str, tuple[str, str, int]]:
    records: dict[str, tuple[str, str, int]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{number}: expected five protocol columns")
        speaker, sample_id, placeholder, attack, key = fields
        if (
            re.fullmatch(r"LA_E_\d+", sample_id) is None
            or re.fullmatch(r"LA_\d+", speaker) is None
            or placeholder != "-"
            or key not in {"bonafide", "spoof"}
            or (key == "bonafide" and attack != "-")
            or (key == "spoof" and attack not in ATTACKS)
        ):
            raise ValueError(f"{path}:{number}: invalid eval protocol row")
        if sample_id in records:
            raise ValueError(f"{path}:{number}: duplicate protocol sample {sample_id}")
        records[sample_id] = (speaker, attack, int(key == "spoof"))
    return records


def _check_metadata(
    parquet: Any,
    protocol: dict[str, tuple[str, str, int]],
    batch_size: int,
) -> dict[str, Any]:
    if not parquet.schema_arrow.equals(SCHEMA, check_metadata=False):
        raise ValueError("mirror Parquet schema differs from the pinned schema")
    seen: set[str] = set()
    labels: Counter[str] = Counter()
    attacks: Counter[str] = Counter()
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
            labels["fake" if row["key"] else "real"] += 1
            attacks[expected[1] if expected[2] else "human"] += 1
    if seen != set(protocol):
        raise ValueError("mirror samples do not cover the complete eval protocol")
    return {
        "label_counts": dict(sorted(labels.items())),
        "generator_counts": dict(sorted(attacks.items())),
        "speakers": len({row[0] for row in protocol.values()}),
    }


def _attribution(root: Path) -> list[Path]:
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    if not paths:
        raise ValueError("attribution root must include README or license text")
    return paths


def _materialize_records(
    parquet: Any, media_root: Path, batch_size: int
) -> tuple[list[SampleRecord], int]:
    media_root.mkdir(parents=True)
    records: list[SampleRecord] = []
    total_bytes = 0
    shared: dict[str, tuple[int, str]] = {}
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["speaker_id", "audio_file_name", "audio", "system_id", "key"],
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
            label = int(row["key"])
            attack = row["system_id"] if label else "human"
            checksum = _bytes_sha256(audio["bytes"])
            previous = shared.get(checksum)
            if previous is not None and previous != (label, sample_id):
                raise ValueError(f"identical bytes have conflicting labels: {sample_id}")
            shared[checksum] = (label, sample_id)
            with (media_root / f"{sample_id}.flac").open("xb") as stream:
                stream.write(audio["bytes"])
            total_bytes += len(audio["bytes"])
            speaker = row["speaker_id"]
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    path=f"ASVspoof2019_LA_eval/flac/{sample_id}.flac",
                    modality="audio",
                    label=label,
                    source=SOURCE,
                    content_type="speech",
                    generator=attack,
                    group_id=f"{SOURCE}:speaker:{speaker}",
                    sha256=checksum,
                    license_id="ODC-By-1.0",
                    identity_id=f"{SOURCE}:speaker:{speaker}",
                    container="flac",
                    codec="flac",
                    sample_rate=16000,
                    split=SPLIT,
                    # CM protocols identify speakers, not source-utterance
                    # derivation chains. original_id/parent_id stay unknown.
                    original_id=UNKNOWN,
                    parent_id=UNKNOWN,
                    decode_status=UNKNOWN,
                )
            )
    return records, total_bytes


def materialize(
    parquet_path: Path,
    protocol_path: Path,
    output_root: Path,
    attribution_root: Path,
    *,
    batch_size: int = 64,
) -> dict[str, object]:
    """Verify pinned inputs and the complete protocol, then publish the import."""
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output_root = output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("output root already exists; choose a new directory")
    started = time.monotonic()
    parquet_path = parquet_path.resolve()
    protocol_path = protocol_path.resolve()
    shard_sha256 = file_sha256(parquet_path)
    if parquet_path.name != EVAL_SHARD or shard_sha256 != EVAL_SHARD_SHA256:
        raise ValueError(f"pinned eval shard SHA256 mismatch: {parquet_path.name}")
    protocol_sha256 = file_sha256(protocol_path)
    if protocol_sha256 != EVAL_PROTOCOL_SHA256:
        raise ValueError("pinned official eval protocol SHA256 mismatch")
    attribution = _attribution(attribution_root.resolve())
    protocol = _protocol_rows(protocol_path)
    parquet = pq.ParquetFile(parquet_path)
    counts = _check_metadata(parquet, protocol, batch_size)
    output_root.mkdir(parents=True)
    media_root = output_root / "ASVspoof2019_LA_eval" / "flac"
    records, media_bytes = _materialize_records(parquet, media_root, batch_size)
    if {r.sample_id for r in records} != set(protocol):
        raise ValueError("mirror records do not cover the complete eval protocol")
    # Exclusive creation prevents mixing old or interrupted output with this run.
    # A failed write leaves a partial directory without a summary; rerun into a
    # new directory instead of treating it as an import/resume.
    destination = output_root / EVAL_PROTOCOL
    destination.parent.mkdir(parents=True)
    shutil.copyfile(protocol_path, destination)
    for path in attribution:
        shutil.copyfile(path, output_root / path.name)
    manifest_sha256 = write_manifest(records, output_root / "manifest.jsonl")
    source = SourceRecord(
        source=SOURCE,
        url=SOURCE_URL,
        version=f"ASVspoof2019-LA-eval; mirror {MIRROR_REVISION}; official CM eval protocol",
        license_id="ODC-By-1.0",
        license_url=LICENSE_URL,
        downloaded_at=UNKNOWN,
        allowed_uses=("research", "evaluation"),
        license_status="approved",
        archive_sha256=UNKNOWN,
    )
    (output_root / "sources.jsonl").write_text(
        json.dumps(asdict(source), sort_keys=True) + "\n", encoding="utf-8"
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "source_kind": "official-protocol-labels-with-pinned-mirror-media",
        "source": SOURCE,
        "mirror_repo": MIRROR_REPO,
        "mirror_revision": MIRROR_REVISION,
        "parquet": str(parquet_path),
        "parquet_sha256": shard_sha256,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "attribution_sha256": {p.name: file_sha256(p) for p in attribution},
        "output_root": str(output_root),
        "media_root": str(media_root),
        "rows": len(records),
        "label_counts": counts["label_counts"],
        "generator_counts": counts["generator_counts"],
        "speakers": counts["speakers"],
        "label_mapping": {"bonafide": 0, "spoof": 1},
        "split": SPLIT,
        "manifest_sha256": manifest_sha256,
        "manifest_hash": manifest_hash(records),
        "media_bytes_written": media_bytes,
        "batch_size": batch_size,
        "elapsed_seconds": time.monotonic() - started,
        "decode_verified": False,
        "official_evaluation": False,
        "official_evaluation_note": (
            "labels and attack IDs come from the official CM eval protocol, but "
            "scores use the project EER metric, not the official ASVspoof "
            "min-tDCF pipeline; this is a research diagnostic only"
        ),
    }
    (output_root / "import-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--attribution-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    try:
        summary = materialize(
            args.parquet,
            args.protocol,
            args.output_root,
            args.attribution_root,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"eval import failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
