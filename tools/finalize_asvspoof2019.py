"""Verify official LA.zip and atomically publish only train/dev research media.

The complete archive is hashed as raw bytes, including its compressed eval data.
Eval members are never opened or extracted. This tool does not decode media,
approve competition usage, download files, or change the input archive's name.
Linux renameat2(RENAME_NOREPLACE) provides atomic publication without clobbering.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO


OFFICIAL_SIZE = 7_640_952_520
OFFICIAL_MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
EXPECTED_MEDIA_COUNTS = {"train": 25380, "dev": 24844}
MAX_MEMBER_BYTES = 100 * 1024**2
MAX_EXTRACTED_BYTES = 8 * 1024**3
MAX_ZIP_ENTRIES = 200_000
CHUNK_BYTES = 1024**2
PROTOCOLS = {
    "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt": "train",
    "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt": "dev",
}


@dataclass(frozen=True)
class SelectedMember:
    info: zipfile.ZipInfo
    kind: str
    split: str | None
    prefix: str


def _archive_hashes(stream: BinaryIO) -> tuple[str, str]:
    if os.fstat(stream.fileno()).st_size != OFFICIAL_SIZE:
        raise ValueError(f"archive size differs from official {OFFICIAL_SIZE} bytes")
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    count = 0
    while chunk := stream.read(CHUNK_BYTES):
        md5.update(chunk)
        sha256.update(chunk)
        count += len(chunk)
    if count != OFFICIAL_SIZE or md5.hexdigest() != OFFICIAL_MD5:
        raise ValueError("archive failed official size/MD5 verification")
    stream.seek(0)
    return md5.hexdigest(), sha256.hexdigest()


def _safe_member(info: zipfile.ZipInfo) -> tuple[str, ...]:
    name = info.filename
    parts = tuple(name.removesuffix("/").split("/"))
    if (
        name != info.orig_filename
        or not name
        or PurePosixPath(name).is_absolute()
        or "\\" in name
        or any(not part or part in (".", "..") or ":" in part for part in parts)
        or any(ord(character) < 32 for character in name)
    ):
        raise ValueError(f"unsafe ZIP path: {name!r}")
    mode = stat.S_IFMT(info.external_attr >> 16)
    if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"symlink or special ZIP entry: {name}")
    if mode == stat.S_IFDIR and not info.is_dir():
        raise ValueError(f"inconsistent ZIP directory entry: {name}")
    if info.flag_bits & 1:
        raise ValueError(f"encrypted ZIP entry: {name}")
    return parts


def _selection(info: zipfile.ZipInfo, parts: tuple[str, ...]) -> SelectedMember | None:
    if info.is_dir():
        return None
    if len(parts) >= 3:
        corpus, folder, filename = parts[-3:]
        for split, letter in (("train", "T"), ("dev", "D")):
            if corpus == f"ASVspoof2019_LA_{split}" and folder == "flac":
                if re.fullmatch(rf"LA_{letter}_\d+\.flac", filename) is None:
                    raise ValueError(
                        f"unexpected train/dev media name: {info.filename}"
                    )
                return SelectedMember(info, "media", split, "/".join(parts[:-3]))
    if len(parts) >= 2 and "/".join(parts[-2:]) in PROTOCOLS:
        return SelectedMember(
            info, "protocol", PROTOCOLS["/".join(parts[-2:])], "/".join(parts[:-2])
        )
    return None


def _selected_members(archive: zipfile.ZipFile) -> tuple[list[SelectedMember], str]:
    entries = archive.infolist()
    if len(entries) > MAX_ZIP_ENTRIES:
        raise ValueError("ZIP directory exceeds entry budget")
    selected: list[SelectedMember] = []
    seen: dict[tuple[str, ...], bool] = {}
    documents: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    for info in entries:
        parts = _safe_member(info)
        key = tuple(part.casefold() for part in parts)
        if key in seen:
            raise ValueError(f"duplicate or case-colliding ZIP path: {info.filename}")
        seen[key] = info.is_dir()
        item = _selection(info, parts)
        if item is not None:
            selected.append(item)
        elif not info.is_dir() and parts[-1].upper() in (
            "README",
            "README.TXT",
            "LICENSE",
            "LICENSE.TXT",
        ):
            documents.append((info, parts))
    for key in seen:
        if any(seen.get(key[:index]) is False for index in range(1, len(key))):
            raise ValueError("ZIP file/directory prefix collision")
    prefixes = {item.prefix for item in selected}
    if len(prefixes) != 1:
        raise ValueError("train/dev archive root is missing or inconsistent")
    prefix = prefixes.pop()
    media_counts = Counter(item.split for item in selected if item.kind == "media")
    protocol_counts = Counter(
        item.split for item in selected if item.kind == "protocol"
    )
    if dict(media_counts) != EXPECTED_MEDIA_COUNTS:
        raise ValueError(f"unexpected train/dev media counts: {dict(media_counts)}")
    if protocol_counts != {"train": 1, "dev": 1}:
        raise ValueError("exactly two train/dev CM protocol files are required")
    selected.extend(
        SelectedMember(info, "document", None, prefix)
        for info, parts in documents
        if "/".join(parts[:-1]) in ("", prefix)
    )
    if any(
        item.info.file_size < 0 or item.info.file_size > MAX_MEMBER_BYTES
        for item in selected
    ):
        raise ValueError("selected member exceeds single-file extraction budget")
    if sum(item.info.file_size for item in selected) > MAX_EXTRACTED_BYTES:
        raise ValueError("selected members exceed total extraction budget")
    return sorted(selected, key=lambda item: item.info.filename), prefix


def _extract_member(
    archive: zipfile.ZipFile, item: SelectedMember, stage: Path, remaining: int
) -> int:
    destination = stage.joinpath(*PurePosixPath(item.info.filename).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    with archive.open(item.info, "r") as source, destination.open("xb") as output:
        while chunk := source.read(CHUNK_BYTES):
            copied += len(chunk)
            if copied > min(MAX_MEMBER_BYTES, remaining, item.info.file_size):
                raise ValueError("member expanded beyond verified extraction budget")
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    if copied != item.info.file_size:
        raise ValueError("extracted member size differs from ZIP directory")
    return copied


def _publish(stage: Path, target: Path) -> None:
    """Atomically move a directory only if target does not exist, including races."""
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as exc:
        raise RuntimeError(
            "atomic no-overwrite publication requires Linux renameat2"
        ) from exc
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def finalize_archive(archive_path: Path, output_dir: Path) -> dict[str, object]:
    """Verify official bytes, validate selection and publish a complete research subset."""
    output_dir = output_dir.absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output must not exist: {output_dir}")
    with archive_path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        md5, sha256 = _archive_hashes(stream)
        with zipfile.ZipFile(stream) as archive:
            selected, prefix = _selected_members(archive)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
            ) as temporary:
                stage = Path(temporary)
                extracted_bytes = 0
                for item in selected:
                    extracted_bytes += _extract_member(
                        archive, item, stage, MAX_EXTRACTED_BYTES - extracted_bytes
                    )
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError("archive changed during verification/extraction")
                summary: dict[str, object] = {
                    "schema_version": 1,
                    "archive_name": archive_path.name,
                    "archive_bytes": before.st_size,
                    "archive_md5": md5,
                    "archive_sha256": sha256,
                    "official_checksum_verified": True,
                    "verified_at": datetime.now(UTC).isoformat(),
                    "dataset_root": prefix or ".",
                    "media_counts": dict(EXPECTED_MEDIA_COUNTS),
                    "protocol_count": 2,
                    "selected_file_count": len(selected),
                    "extracted_bytes": extracted_bytes,
                    "eval_media_extracted": False,
                    "eval_member_contents_read": False,
                    "decode_verified": False,
                    "protocol_contents_validated": False,
                    "official_competition_usage": "unconfirmed",
                }
                report = stage / "acquisition.json"
                with report.open("x", encoding="utf-8") as handle:
                    json.dump(
                        summary,
                        handle,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                    )
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                _publish(stage, output_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = finalize_archive(args.archive, args.output)
    except (
        ValueError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        NotImplementedError,
    ) as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
