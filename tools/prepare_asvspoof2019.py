"""Import already-extracted official ASVspoof 2019 LA train/dev only.

No download, evaluation-partition access or audio decode is performed. Protocol
metadata are preserved; 16 kHz FLAC is a documented corpus property, not a local
decoder measurement. Decode auditing must precede training.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path

from deepfake_detection_archive.data.manifest import (
    UNKNOWN,
    SampleRecord,
    SourceRecord,
    file_sha256,
    manifest_hash,
    write_manifest,
)


SOURCE = "asvspoof2019-la-official"
SOURCE_URL = "https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45"
LICENSE_URL = "https://opendatacommons.org/licenses/by/1-0/"
OFFICIAL_URL = "https://www.asvspoof.org/index2019.html"
PROTOCOLS = {
    "train": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
}
EXPECTED_COUNTS = {
    "train": {"speakers": 20, "real": 2580, "fake": 22800},
    "dev": {"speakers": 10, "real": 2548, "fake": 22296},
}


def _within_root(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes dataset root: {relative}")
    if not path.is_file():
        raise ValueError(f"required file missing: {relative}")
    return path


def _parse_line(line: str, split: str, location: str) -> tuple[str, str, str, int]:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"{location}: expected five CM protocol columns")
    speaker, sample_id, placeholder, attack, label = fields
    prefix = "T" if split == "train" else "D"
    if re.fullmatch(r"LA_\d+", speaker) is None:
        raise ValueError(f"{location}: invalid speaker ID")
    if re.fullmatch(rf"LA_{prefix}_\d+", sample_id) is None:
        raise ValueError(f"{location}: invalid {split} file ID")
    if placeholder != "-":
        raise ValueError(f"{location}: expected '-' third column")
    if label not in ("bonafide", "spoof"):
        raise ValueError(f"{location}: unknown CM label {label}")
    if (label == "bonafide" and attack != "-") or (
        label == "spoof" and attack not in {f"A{i:02d}" for i in range(1, 7)}
    ):
        raise ValueError(
            f"{location}: label/attack mismatch or unknown train/dev attack"
        )
    return speaker, sample_id, attack, int(label == "spoof")


def _load_records(root: Path) -> tuple[list[SampleRecord], dict[str, str]]:
    records: list[SampleRecord] = []
    protocol_hashes: dict[str, str] = {}
    ids: set[str] = set()
    speaker_splits: dict[str, str] = {}
    hash_records: dict[str, SampleRecord] = {}
    for split, relative in PROTOCOLS.items():
        protocol = _within_root(root, relative)
        protocol_hashes[relative] = file_sha256(protocol)
        lines = protocol.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"empty protocol: {relative}")
        for number, line in enumerate(lines, 1):
            location = f"{relative}:{number}"
            speaker, sample_id, attack, label = _parse_line(line, split, location)
            if sample_id in ids:
                raise ValueError(f"{location}: duplicate protocol sample {sample_id}")
            ids.add(sample_id)
            if speaker in speaker_splits and speaker_splits[speaker] != split:
                raise ValueError(f"speaker crosses official splits: {speaker}")
            speaker_splits[speaker] = split
            media_relative = f"ASVspoof2019_LA_{split}/flac/{sample_id}.flac"
            media = _within_root(root, media_relative)
            checksum = file_sha256(media)
            previous = hash_records.get(checksum)
            if previous is not None and previous.label != label:
                raise ValueError(
                    f"identical bytes have conflicting labels: {sample_id}"
                )
            if previous is not None and previous.split != split:
                raise ValueError(f"identical bytes cross official splits: {sample_id}")
            record = SampleRecord(
                sample_id=sample_id,
                path=media_relative,
                modality="audio",
                label=label,
                source=SOURCE,
                content_type="speech",
                generator=attack if label else "human",
                group_id=f"{SOURCE}:speaker:{speaker}",
                sha256=checksum,
                license_id="ODC-By-1.0",
                identity_id=f"{SOURCE}:speaker:{speaker}",
                container="flac",
                codec="flac",
                sample_rate=16000,
                split=split,
                # CM protocols identify speakers, not source-utterance derivation
                # chains. original_id/parent_id remain explicitly unknown.
                original_id=UNKNOWN,
                parent_id=UNKNOWN,
                decode_status=UNKNOWN,
            )
            hash_records[checksum] = record
            records.append(record)
    return records, protocol_hashes


def _pilot(records: list[SampleRecord], limit: int | None) -> list[SampleRecord]:
    if limit is None:
        return records
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit_per_class must be a positive integer")
    selected: list[SampleRecord] = []
    for split in PROTOCOLS:
        for label in (0, 1):
            by_speaker: dict[str, list[SampleRecord]] = defaultdict(list)
            for record in records:
                if record.split == split and record.label == label:
                    by_speaker[record.identity_id].append(record)
            if not by_speaker:
                raise ValueError(f"{split}: missing label {label}")
            if limit < len(by_speaker):
                raise ValueError(
                    f"limit_per_class {limit} cannot cover {len(by_speaker)} "
                    f"speakers in {split}/label={label}"
                )
            queues = deque(
                deque(sorted(by_speaker[speaker], key=lambda item: item.sample_id))
                for speaker in sorted(by_speaker)
            )
            count = 0
            while queues and count < limit:
                queue = queues.popleft()
                selected.append(queue.popleft())
                count += 1
                if queue:
                    queues.append(queue)
    return selected


def _counts(records: list[SampleRecord]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "speakers": len({r.identity_id for r in records if r.split == split}),
            "real": sum(r.split == split and r.label == 0 for r in records),
            "fake": sum(r.split == split and r.label == 1 for r in records),
        }
        for split in PROTOCOLS
    }


def _local_attribution(root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for name in (
        "README.LA.txt",
        "README.txt",
        "README",
        "LICENSE",
        "LICENSE.txt",
        "LICENCE",
    ):
        if (root / name).exists():
            path = _within_root(root, name)
            entries.append({"path": name, "sha256": file_sha256(path)})
    return entries


def prepare_asvspoof2019(
    root: Path, output_dir: Path, *, limit_per_class: int | None = None
) -> dict[str, object]:
    """Validate complete train/dev inputs, then publish one deterministic import."""
    root = root.resolve()
    output_dir = output_dir.resolve()
    if not root.is_dir():
        raise ValueError("root must be an extracted LA dataset directory")
    filenames = ("manifest.jsonl", "sources.jsonl", "import-summary.json")
    if any((output_dir / name).exists() for name in filenames):
        raise ValueError("import outputs already exist; use a new output directory")
    records, protocol_hashes = _load_records(root)
    counts = _counts(records)
    if any(
        counts[split][label] == 0 for split in PROTOCOLS for label in ("real", "fake")
    ):
        raise ValueError("both classes are required in each official split")
    if counts != EXPECTED_COUNTS:
        raise ValueError(
            "official train/dev counts mismatch; complete official protocols are "
            f"required before pilot selection: actual={counts}, expected={EXPECTED_COUNTS}"
        )
    selected = _pilot(records, limit_per_class)
    source = SourceRecord(
        source=SOURCE,
        url=SOURCE_URL,
        version="ASVspoof2019-LA-train-dev",
        license_id="ODC-By-1.0",
        license_url=LICENSE_URL,
        downloaded_at=UNKNOWN,
        allowed_uses=("research", "training", "evaluation"),
        license_status="approved",
        archive_sha256=UNKNOWN,
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset": SOURCE,
        "root": str(root),
        "protocol_sha256": protocol_hashes,
        "full_protocol_manifest_hash": manifest_hash(records),
        "full_protocol_counts": counts,
        "official_expected_counts": EXPECTED_COUNTS,
        "official_counts_match": counts == EXPECTED_COUNTS,
        "selected_counts": _counts(selected),
        "selected_attack_counts": dict(
            sorted(Counter(r.generator for r in selected).items())
        ),
        "subset": len(selected) < len(records),
        "limit_per_class_per_split": limit_per_class,
        "selection": "sorted-speaker-round-robin-sorted-file-id"
        if limit_per_class
        else "all-official-train-dev",
        "all_protocol_records_validated_before_selection": True,
        "eval_accessed": False,
        "decode_verified": False,
        "media_metadata_basis": "official corpus documentation; not locally decoded",
        "competition_rules_status": "unconfirmed",
        "competition_approved": False,
        "local_research_license_status": "approved",
        "license_scope": "local research/training under source terms; competition use unconfirmed",
        "attribution": {
            "dataset_url": SOURCE_URL,
            "official_protocol_readme_reference": OFFICIAL_URL,
            "license_url": LICENSE_URL,
            "local_files": _local_attribution(root),
            "underlying_corpus": "VCTK; retain official README attribution and source terms",
        },
        "manifest_sha256": manifest_hash(selected),
    }
    # All source checks finish before any deliverable is written. Existing import
    # artifacts are never overwritten; byte hashes bind records to local media.
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(selected, output_dir / "manifest.jsonl")
    (output_dir / "sources.jsonl").write_text(
        json.dumps(asdict(source), sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "import-summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-class", type=int)
    args = parser.parse_args()
    try:
        summary = prepare_asvspoof2019(
            args.root, args.output_dir, limit_per_class=args.limit_per_class
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"import failed: {exc}\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
