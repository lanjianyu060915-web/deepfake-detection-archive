"""Register the In-the-Wild speech deepfake release as a frozen external test set.

Reads the extracted ``release_in_the_wild`` directory (``meta.csv`` plus one WAV
per row) in place and writes a ``test`` manifest grouped by the celebrity named
in ``meta.csv``. No media is copied, decoded or scored here; the set is meant
for one diagnostic pass over already frozen speech candidates, never for
epoch selection or tuning. Dataset terms are recorded as unresolved.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    write_manifest,
)


SOURCE = "in-the-wild-release"
SOURCE_URL = "https://deepfake-demo.aisec.fraunhofer.de/in_the_wild"
LABELS = {"bona-fide": 0, "spoof": 1}
SPLIT = "test"


def _speaker_key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", name.strip().lower()).strip("-") or "unknown"


def materialize(
    release_root: Path, output_root: Path, archive_sha256: str, downloaded_at: str
) -> dict[str, Any]:
    """Write ``manifest.jsonl`` and ``sources.jsonl`` for the extracted release."""
    meta = release_root / "meta.csv"
    with meta.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"file", "speaker", "label"}:
        raise ValueError("meta.csv must have exactly file, speaker and label columns")
    records: list[SampleRecord] = []
    seen: set[str] = set()
    for row in rows:
        name = row["file"].strip()
        if name in seen:
            raise ValueError(f"duplicate file in meta.csv: {name}")
        seen.add(name)
        if row["label"] not in LABELS:
            raise ValueError(f"unknown label {row['label']!r} for {name}")
        media = release_root / name
        if not media.is_file():
            raise ValueError(f"missing media file: {name}")
        speaker = f"{SOURCE}:speaker:{_speaker_key(row['speaker'])}"
        records.append(
            SampleRecord(
                sample_id=f"itw-{Path(name).stem}",
                path=name,
                modality="audio",
                label=LABELS[row["label"]],
                source=SOURCE,
                content_type="speech",
                group_id=speaker,
                identity_id=speaker,
                sha256=file_sha256(media),
                license_id="LicenseRef-in-the-wild-unverified",
                container="wav",
                split=SPLIT,
            )
        )
    source = SourceRecord(
        source=SOURCE,
        url=SOURCE_URL,
        version="release_in_the_wild.zip",
        license_id="LicenseRef-in-the-wild-unverified",
        license_url=SOURCE_URL,
        downloaded_at=downloaded_at,
        allowed_uses=("research", "evaluation"),
        license_status="unresolved",
        archive_sha256=archive_sha256,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    digest = write_manifest(records, output_root / "manifest.jsonl")
    source_row = asdict(source) | {"allowed_uses": list(source.allowed_uses)}
    (output_root / "sources.jsonl").write_text(
        json.dumps(source_row, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    labels = Counter(record.label for record in records)
    return {
        "manifest_sha256": digest,
        "samples": len(records),
        "real": labels[0],
        "fake": labels[1],
        "speakers": len({record.group_id for record in records}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--downloaded-at", required=True)
    args = parser.parse_args()
    try:
        summary = materialize(
            args.release_root, args.output_root, args.archive_sha256, args.downloaded_at
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"in-the-wild import failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
