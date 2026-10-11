"""Extract and fully decode the pinned FakeSound archive for diagnostics.

Metadata rows absent from the ZIP remain in the manifest. Difficulty buckets are
provenance, never train/dev/holdout assignments. License approval is unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TypedDict

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import (
    UNKNOWN,
    SampleRecord,
    SourceRecord,
    file_sha256,
    write_manifest,
)
from deepfake_detection_archive.data.split import connected_groups


SOURCE = "fakesound-hf-diagnostic"
SOURCE_URL = "https://huggingface.co/datasets/ZeyuXie/FakeSound"
UPSTREAM_README_URL = "https://github.com/FakeSoundData/FakeSound/blob/main/README.md"
UPSTREAM_README_SHA256 = (
    "0e52856c3451a38036f04b0e11d5d3e04fa044ec36cfd96174c7c29f8294d093"
)
ARCHIVE_BYTES = 986_570_159
ARCHIVE_SHA256 = "96cfc28b1e2274ff3e4404be94c4039645e862376ec161549386a9e0577c0623"
BUCKETS = ("easy", "hard", "train", "zeroshot")
METADATA_PREFIX = "FakeSound/deepfake_data/"
ARCHIVE_PREFIX = "FakeSound/audio_data/"


class AudioInfo(TypedDict):
    container: str
    codec: str
    channels: int
    sample_rate: int
    frames: int
    duration_seconds: float


@dataclass(frozen=True)
class MetadataRow:
    bucket: str
    index: int
    audio_id: str
    original_id: str
    label: int
    onset_offset: str
    filepath: str
    member: str

    @property
    def sample_id(self) -> str:
        return f"fakesound:{self.bucket}:{self.audio_id}"


def _safe_member(name: str) -> None:
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if (
        posix.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or "\\" in name
        or "\x00" in name
        or str(posix) in ("", ".")
    ):
        raise ValueError(f"unsafe archive path: {name}")


def _parse_row(raw: object, bucket: str, index: int) -> MetadataRow:
    if not isinstance(raw, dict):
        raise ValueError("metadata row must be an object")  # noqa: TRY004 -- schema
    audio_id, onset, filepath = (
        raw.get("audio_id"),
        raw.get("onset_offset"),
        raw.get("filepath"),
    )
    label = raw.get("label")
    if (
        not isinstance(audio_id, str)
        or not audio_id
        or not isinstance(onset, str)
        or not isinstance(filepath, str)
        or type(label) is not int
        or label not in (0, 1)
    ):
        raise ValueError(f"invalid metadata fields in {bucket}:{index}")
    expected = f"{METADATA_PREFIX}{bucket}/{audio_id}.wav"
    if filepath != expected:
        raise ValueError(f"metadata path disagrees with audio_id: {filepath}")
    member = f"{ARCHIVE_PREFIX}{bucket}/{audio_id}.wav"
    _safe_member(member)
    if "/" in audio_id:
        raise ValueError("audio_id must be a filename stem")
    original = audio_id
    if label == 0:
        if onset != "0_0":
            raise ValueError("original metadata row must use onset_offset=0_0")
    else:
        bounds = onset.split("_")
        if len(bounds) != 2:
            raise ValueError("invalid onset_offset")
        start, end = map(float, bounds)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("invalid onset_offset bounds")
        suffix = "_" + onset
        if not audio_id.endswith(suffix):
            raise ValueError("fake audio_id must end with its explicit onset_offset")
        original = audio_id.removesuffix(suffix)
    return MetadataRow(
        bucket, index, audio_id, original, label, onset, filepath, member
    )


def _read_metadata(
    archive: zipfile.ZipFile,
) -> tuple[list[MetadataRow], dict[str, bytes], set[str]]:
    members: set[str] = set()
    wavs: set[str] = set()
    for info in archive.infolist():
        _safe_member(info.filename)
        if info.filename in members:
            raise ValueError(f"duplicate archive member: {info.filename}")
        members.add(info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"symlink archive member: {info.filename}")
        if not info.is_dir() and info.filename.endswith(".wav"):
            wavs.add(info.filename)
    rows: list[MetadataRow] = []
    metadata: dict[str, bytes] = {}
    for bucket in BUCKETS:
        name = f"FakeSound/meta_data/{bucket}.json"
        data = archive.read(name)
        value = json.loads(data)
        if not isinstance(value, dict) or not isinstance(value.get("audios"), list):
            raise ValueError(f"invalid metadata: {name}")  # noqa: TRY004 -- schema
        parsed = [
            _parse_row(raw, bucket, index)
            for index, raw in enumerate(value["audios"], 1)
        ]
        real_ids = {row.audio_id for row in parsed if row.label == 0}
        if any(row.original_id not in real_ids for row in parsed):
            raise ValueError(f"derived audio lacks an explicit original row: {name}")
        rows.extend(parsed)
        metadata[name] = data
    expected = {row.member for row in rows}
    if not rows:
        raise ValueError("archive metadata has no audio rows")
    if len(expected) != len(rows):
        raise ValueError("duplicate metadata path")
    if wavs - expected:
        raise ValueError("archive contains WAVs without metadata labels")
    if any(row.label == 0 and row.member in wavs for row in rows):
        raise ValueError(
            "manipulated-only distribution unexpectedly contains originals"
        )
    return rows, metadata, wavs


def _decode(path: Path) -> AudioInfo:
    """Read every frame at its native rate/channels without altering the media."""
    with sf.SoundFile(path) as stream:
        frames, rate, channels = (
            int(stream.frames),
            int(stream.samplerate),
            int(stream.channels),
        )
        if frames <= 0 or rate <= 0 or channels <= 0:
            raise ValueError("empty or invalid audio")
        decoded = 0
        for block in stream.blocks(blocksize=65_536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError("nonfinite decoded audio")
            decoded += len(block)
        if decoded != frames:
            raise ValueError("decoded frame count differs from header")
        return {
            "container": str(stream.format),
            "codec": str(stream.subtype),
            "channels": channels,
            "sample_rate": rate,
            "frames": frames,
            "duration_seconds": frames / rate,
        }


def _materialize(
    row: MetadataRow, archive: zipfile.ZipFile, wavs: set[str], root: Path
) -> tuple[SampleRecord, AudioInfo | None]:
    # This identity mapping is specific to the pinned ZIP: the official README
    # declares that only manipulated audio is distributed; every WAV has raw=1
    # and a reconstruction interval. Its absent original counterpart has raw=0.
    # The README's contradictory raw label convention is retained in the summary.
    record = SampleRecord(
        sample_id=row.sample_id,
        path=row.member,
        modality="audio",
        label=row.label,
        source=SOURCE,
        group_id=f"fakesound:original:{row.original_id}",
        original_id=row.original_id,
        parent_id=(
            f"fakesound:{row.bucket}:{row.original_id}" if row.label == 1 else UNKNOWN
        ),
        derivation_group_id=f"fakesound:original:{row.original_id}",
    )
    if row.member not in wavs:
        return replace(record, status="missing", failure_reason="absent_from_zip"), None
    destination = root / row.member
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256()
    try:
        with archive.open(row.member) as source, destination.open("xb") as target:
            while block := source.read(1_048_576):
                target.write(block)
                checksum.update(block)
        record = replace(record, sha256=checksum.hexdigest())
        info = _decode(destination)
        return replace(
            record,
            decode_status="ok",
            container=info["container"],
            codec=info["codec"],
            sample_rate=info["sample_rate"],
            duration_seconds=info["duration_seconds"],
        ), info
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        return replace(
            record,
            status="decode_failed",
            decode_status="failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
        ), None


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def prepare_fakesound(archive_path: Path, output: Path) -> dict[str, object]:
    start = time.monotonic()
    if output.exists() or output.is_symlink():
        raise FileExistsError("output directory must not exist")
    if archive_path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("archive size differs from pinned source")
    if file_sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("archive checksum differs from pinned source")
    with zipfile.ZipFile(archive_path) as archive:
        rows, metadata, wavs = _read_metadata(archive)
        # Exclusive directory creation also prevents concurrent output replacement.
        output.mkdir(parents=True, exist_ok=False)
        (output / "metadata").mkdir()
        for name, data in metadata.items():
            (output / "metadata" / Path(name).name).write_bytes(data)
        records: list[SampleRecord] = []
        formats: Counter[str] = Counter()
        buckets: dict[str, set[str]] = defaultdict(set)
        by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
        with (output / "provenance.jsonl").open("x", encoding="utf-8") as provenance:
            for row in rows:
                record, info = _materialize(row, archive, wavs, output / "media")
                records.append(record)
                buckets[row.original_id].add(row.bucket)
                by_bucket[row.bucket][
                    f"label={record.label}/status={record.status}"
                ] += 1
                if info:
                    formats[
                        f"{info['container']}/{info['codec']}/"
                        f"{info['sample_rate']}Hz/{info['channels']}ch"
                    ] += 1
                provenance.write(
                    json.dumps(
                        {
                            "sample_id": record.sample_id,
                            "metadata": asdict(row),
                            "upstream_bucket": row.bucket,
                            "raw_label": row.label,
                            "internal_label_basis": (
                                "official_manipulated_distribution_and_reconstruction_interval"
                                if row.label == 1
                                else "metadata_original_id_and_zero_interval_missing_media"
                            ),
                            "metadata_only": row.member not in wavs,
                            "format": info,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
    manifest_sha = write_manifest(records, output / "manifest.jsonl")
    source = SourceRecord(
        source=SOURCE,
        url=SOURCE_URL,
        version=f"archive-sha256:{ARCHIVE_SHA256}",
        license_id=UNKNOWN,
        license_url=SOURCE_URL + "/blob/main/README.md",
        downloaded_at="2026-10-10",
        allowed_uses=("diagnostics",),
        archive_sha256=ARCHIVE_SHA256,
    )
    # One JSON object remains valid JSONL even when indented elsewhere for summaries.
    with (output / "sources.jsonl").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(source), sort_keys=True) + "\n")
    elapsed = time.monotonic() - start
    available = [record for record in records if record.decode_status == "ok"]
    summary: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": SOURCE,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_BYTES,
        "manifest_sha256": manifest_sha,
        "metadata_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in metadata.items()
        },
        "path_mapping": {"from": METADATA_PREFIX, "to": ARCHIVE_PREFIX},
        "label_mapping": {"0": "real", "1": "fake"},
        "label_semantics_conflict": {
            "upstream_readme_url": UPSTREAM_README_URL,
            "upstream_readme_sha256": UPSTREAM_README_SHA256,
            "readme_fake_raw_label": 0,
            "archive_manipulated_raw_label": 1,
            "internal_fake_label": 1,
            "basis": (
                "official README distributes manipulated audio only and omits "
                "original AudioCaps audio; present raw=1 rows have reconstructed "
                "intervals, absent raw=0 rows are explicit original counterparts"
            ),
            "scope": "pinned archive only; never apply README raw labels implicitly",
        },
        "grouping_rule": "fake audio_id minus explicit onset_offset suffix matches real audio_id",
        "samples": len(records),
        "archive_wavs": len(wavs),
        "counts": {
            field: dict(
                sorted(Counter(str(getattr(r, field)) for r in records).items())
            )
            for field in ("label", "status", "decode_status")
        },
        "decoded_label_counts": dict(Counter(str(r.label) for r in available)),
        "decode_coverage": len(available) / len(records),
        "formats": dict(sorted(formats.items())),
        "by_upstream_bucket": dict(by_bucket),
        "original_groups": len(buckets),
        "connected_groups": len(set(connected_groups(records).values())),
        "originals_crossing_upstream_buckets": {
            original: sorted(assigned)
            for original, assigned in sorted(buckets.items())
            if len(assigned) > 1
        },
        "elapsed_seconds": elapsed,
        "decoded_samples_per_second": len(available) / elapsed,
        "data_root": "media",
        "split": "unassigned",
        "license_status": "unresolved",
        "training_allowed": False,
        "binary_eer_eligible": False,
        "limitations": [
            "difficulty buckets are not independent data splits",
            "missing and failed records remain in the coverage denominator",
            "content type, generator, and identity are not supplied per sample",
            "license and upstream label-documentation conflict need review",
        ],
    }
    _write_json(output / "import-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = prepare_fakesound(args.archive, args.output)
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
