"""Deterministic, bounded offline bundles with explicit verification evidence.

The internal manifest proves consistency, not provenance. Pin the returned
bundle SHA256 externally and pass expected_sha256 when authenticity matters.
Packaging supplied assets does not prove an official adapter or offline run.
"""

import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, cast


type JSONValue = (
    bool | int | float | str | list[JSONValue] | dict[str, JSONValue] | None
)

MANIFEST_NAME = "MANIFEST.json"
_EXCLUDED = {"data", "archive", "archives", "cache", ".git", "__pycache__"}
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)
}
_CHUNK_BYTES = 1024 * 1024


class BundleError(ValueError):
    """Unsafe bundle, malformed metadata, limits violation or failed integrity."""


@dataclass(frozen=True)
class BundleLimits:
    max_files: int = 10000
    max_file_bytes: int = 8 * 1024**3
    max_total_bytes: int = 32 * 1024**3
    max_manifest_bytes: int = 8 * 1024**2

    def validate(self) -> None:
        for value in (
            self.max_files,
            self.max_file_bytes,
            self.max_total_bytes,
            self.max_manifest_bytes,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise BundleError("bundle limits must be positive integers")


DEFAULT_LIMITS = BundleLimits()


def _safe_name(name: str, *, allow_manifest: bool = False) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise BundleError(f"invalid bundle path: {name!r}")
    parts = name.split("/")
    if any(
        not part
        or part in (".", "..")
        or part.endswith((" ", "."))
        or any(ord(character) < 32 for character in part)
        or ":" in part
        or part.casefold() in _EXCLUDED
        or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED
        for part in parts
    ):
        raise BundleError(f"unsafe or excluded bundle path: {name!r}")
    if PurePosixPath(name).is_absolute():
        raise BundleError(f"absolute bundle path: {name!r}")
    if parts[0].casefold() == MANIFEST_NAME.casefold() and (
        not allow_manifest or name != MANIFEST_NAME
    ):
        raise BundleError("MANIFEST.json is reserved")
    return name


def _no_symlinks(path: Path) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise BundleError(f"symlinks are not allowed: {path}")


def _hash_stream(stream: IO[bytes], maximum: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_CHUNK_BYTES):
        size += len(chunk)
        if size > maximum:
            raise BundleError("file exceeds size limit")
        digest.update(chunk)
    return digest.hexdigest(), size


def _hash_file(path: Path, maximum: int) -> tuple[str, int]:
    with path.open("rb") as stream:
        return _hash_stream(stream, maximum)


def _metadata(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result = dict(value)
    for key in ("official_adapter_status", "offline_rehearsal_status"):
        if not isinstance(result.get(key), str) or not result[key]:
            raise BundleError(f"metadata requires explicit nonempty {key}")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BundleError("metadata must be finite JSON values") from exc
    return result


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _source_files(files: Mapping[str, Path], limits: BundleLimits) -> dict[str, Path]:
    if not files or len(files) > limits.max_files:
        raise BundleError("bundle requires between one and max_files supplied assets")
    sources: dict[str, Path] = {}
    seen: set[str] = set()
    for name, source_path in sorted(files.items()):
        _safe_name(name)
        key = name.casefold()
        if key in seen:
            raise BundleError(f"duplicate/case-colliding bundle path: {name}")
        seen.add(key)
        source = Path(source_path).absolute()
        _no_symlinks(source)
        if any(part.casefold() in _EXCLUDED for part in source.parts):
            raise BundleError(f"excluded source directory: {source}")
        if not source.is_file() or not stat.S_ISREG(source.stat().st_mode):
            raise BundleError(f"source must be a regular file: {source}")
        sources[name] = source
    _check_prefixes(seen)
    return sources


def _check_prefixes(names: set[str]) -> None:
    for name in names:
        if any(str(parent) in names for parent in PurePosixPath(name).parents):
            raise BundleError(f"file/directory path collision: {name}")


def package_bundle(
    output_path: Path,
    files: Mapping[str, Path],
    *,
    metadata: Mapping[str, JSONValue],
    limits: BundleLimits = DEFAULT_LIMITS,
) -> dict[str, JSONValue]:
    """Package explicitly supplied code/config/weights/licenses/wheelhouse files.

    No directory discovery or implicit inclusion occurs. Existing output files
    are never overwritten. Output is reproducible for identical names, bytes
    and metadata, regardless of input mtimes, paths or mapping order.
    """
    limits.validate()
    sources = _source_files(files, limits)
    manifest: dict[str, JSONValue] = {
        "schema_version": 1,
        "metadata": _metadata(metadata),
        "files": [],
    }
    output_path = Path(output_path).absolute()
    _no_symlinks(output_path)
    if output_path.exists():
        raise BundleError("bundle output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        entries: list[JSONValue] = []
        total = 0
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, source in sources.items():
                digest, size = _write_asset(archive, name, source, limits)
                total += size
                if total > limits.max_total_bytes:
                    raise BundleError("bundle exceeds total uncompressed size limit")
                entries.append(
                    {"path": name, "sha256": digest, "size": size, "type": "file"}
                )
            manifest["files"] = entries
            payload = (
                json.dumps(
                    manifest,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
            if len(payload) > limits.max_manifest_bytes:
                raise BundleError("manifest exceeds size limit")
            archive.writestr(_zip_info(MANIFEST_NAME), payload)
        # link is an atomic create-if-absent, so an output race cannot overwrite.
        report = verify_bundle(temporary, limits=limits)
        os.link(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"sha256": report["sha256"], "manifest": manifest}


def _write_asset(
    archive: zipfile.ZipFile, name: str, source: Path, limits: BundleLimits
) -> tuple[str, int]:
    _no_symlinks(source)
    digest = hashlib.sha256()
    size = 0
    with (
        source.open("rb") as stream,
        archive.open(_zip_info(name), "w", force_zip64=True) as target,
    ):
        while chunk := stream.read(_CHUNK_BYTES):
            size += len(chunk)
            if size > limits.max_file_bytes:
                raise BundleError("source exceeds per-file size limit")
            digest.update(chunk)
            target.write(chunk)
    return digest.hexdigest(), size


def _unique_object(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_manifest(
    archive: zipfile.ZipFile, limits: BundleLimits
) -> dict[str, JSONValue]:
    try:
        info = archive.getinfo(MANIFEST_NAME)
    except KeyError as exc:
        raise BundleError("bundle is missing MANIFEST.json") from exc
    if info.file_size > limits.max_manifest_bytes:
        raise BundleError("manifest exceeds size limit")
    try:
        raw: object = json.loads(archive.read(info), object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as exc:
        raise BundleError(f"invalid manifest JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "metadata", "files"}:
        raise BundleError("invalid manifest structure")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise BundleError("unsupported manifest schema")
    if not isinstance(raw["metadata"], dict):
        raise BundleError("manifest metadata must be an object")
    _metadata(raw["metadata"])
    return cast(dict[str, JSONValue], raw)


def _archive_members(
    archive: zipfile.ZipFile, limits: BundleLimits
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > limits.max_files + 1:
        raise BundleError("bundle exceeds file count limit")
    members = {}
    folded = set()
    total = 0
    for info in infos:
        name = _safe_name(info.orig_filename, allow_manifest=True)
        if name.casefold() in folded:
            raise BundleError("duplicate/case-colliding ZIP member")
        folded.add(name.casefold())
        if info.is_dir() or stat.S_IFMT(info.external_attr >> 16) not in (
            0,
            stat.S_IFREG,
        ):
            raise BundleError("ZIP members must be regular files")
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise BundleError("only unencrypted ZIP_STORED bundles are supported")
        maximum = (
            limits.max_manifest_bytes
            if name == MANIFEST_NAME
            else limits.max_file_bytes
        )
        if info.file_size > maximum or info.compress_size != info.file_size:
            raise BundleError("ZIP member violates size limits")
        total += info.file_size
        if total > limits.max_total_bytes + limits.max_manifest_bytes:
            raise BundleError("bundle exceeds total size limit")
        members[name] = info
    _check_prefixes(folded)
    return members


def _verify_assets(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    manifest: dict[str, JSONValue],
    limits: BundleLimits,
) -> int:
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise BundleError("manifest files must be a nonempty list")
    seen: set[str] = set()
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size",
            "type",
        }:
            raise BundleError("invalid manifest file entry")
        name, digest, size = entry["path"], entry["sha256"], entry["size"]
        if not isinstance(name, str):
            raise BundleError("manifest path must be a string")
        _safe_name(name)
        if name.casefold() in seen:
            raise BundleError("duplicate manifest file entry")
        seen.add(name.casefold())
        if (
            entry["type"] != "file"
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise BundleError("invalid manifest size/hash/type")
        if name not in members or members[name].file_size != size:
            raise BundleError(f"missing file or size mismatch: {name}")
        with archive.open(members[name]) as stream:
            actual_hash, actual_size = _hash_stream(stream, limits.max_file_bytes)
        if actual_hash != digest or actual_size != size:
            raise BundleError(f"file hash/size mismatch: {name}")
        total += actual_size
        if total > limits.max_total_bytes:
            raise BundleError("bundle exceeds total asset size limit")
    if seen != {name.casefold() for name in members if name != MANIFEST_NAME}:
        raise BundleError("bundle contains unlisted files")
    return total


def _check_directory_budget(path: Path, limits: BundleLimits) -> None:
    """Bound the central directory before ZipFile allocates its entry objects."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(0, size - 65557))
        tail = stream.read(65557)
        position = tail.rfind(b"PK\x05\x06")
        if position < 0 or len(tail) - position < 22:
            raise BundleError("ZIP end directory record missing")
        end = struct.unpack("<4s4H2LH", tail[position : position + 22])
        if len(tail) - position != 22 + end[-1]:
            raise BundleError("invalid ZIP trailing data")
        count, directory_bytes = end[4], end[5]
        absolute_end = size - len(tail) + position
        if absolute_end >= 20:
            stream.seek(absolute_end - 20)
            locator = stream.read(20)
            if locator.startswith(b"PK\x06\x07"):
                offset = struct.unpack("<4sLQL", locator)[2]
                if offset > size - 56:
                    raise BundleError("invalid ZIP64 directory offset")
                stream.seek(offset)
                record = stream.read(56)
                if not record.startswith(b"PK\x06\x06"):
                    raise BundleError("invalid ZIP64 directory record")
                end64 = struct.unpack("<4sQ2H2L4Q", record)
                count, directory_bytes = end64[7], end64[8]
        if (
            count > limits.max_files + 1
            or directory_bytes > (limits.max_files + 1) * 4096
        ):
            raise BundleError("ZIP central directory exceeds count/memory budget")


def verify_bundle(
    path: Path,
    *,
    expected_sha256: str | None = None,
    extract_to: Path | None = None,
    limits: BundleLimits = DEFAULT_LIMITS,
) -> dict[str, JSONValue]:
    """Verify all assets before optional extraction into a new absent directory.

    ZIP_STORED plus bounded sizes/counts prevents decompression bombs. The
    manifest is bounded separately. Extraction never invokes extractall and
    never writes into an existing directory; partial output is removed on error.
    """
    limits.validate()
    path = Path(path).absolute()
    _no_symlinks(path)
    maximum = (
        limits.max_total_bytes + limits.max_manifest_bytes + limits.max_files * 1024
    )
    if not path.is_file() or path.stat().st_size > maximum:
        raise BundleError("bundle is missing or exceeds archive size limit")
    digest, archive_bytes = _hash_file(path, maximum)
    if expected_sha256 is not None and digest != expected_sha256:
        raise BundleError("bundle SHA256 does not match trusted expected hash")
    _check_directory_budget(path, limits)
    try:
        with zipfile.ZipFile(path) as archive:
            members = _archive_members(archive, limits)
            manifest = _read_manifest(archive, limits)
            total = _verify_assets(archive, members, manifest, limits)
            if extract_to is not None:
                _extract_verified(archive, members, Path(extract_to), limits)
    except (zipfile.BadZipFile, EOFError, RuntimeError) as exc:
        raise BundleError(f"invalid bundle ZIP: {exc}") from exc
    return {
        "sha256": digest,
        "archive_bytes": archive_bytes,
        "file_count": len(members) - 1,
        "total_asset_bytes": total,
        "manifest": manifest,
        "integrity_verified": True,
    }


def _extract_verified(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    destination: Path,
    limits: BundleLimits,
) -> None:
    destination = destination.absolute()
    _no_symlinks(destination)
    if destination.exists():
        raise BundleError("extraction destination must not exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o700)  # exclusive/private; never merge existing content.
    try:
        for name, info in sorted(members.items()):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as output:
                _bounded_copy(
                    source,
                    output,
                    max(limits.max_file_bytes, limits.max_manifest_bytes),
                )
    except BaseException:
        shutil.rmtree(destination)
        raise


def _bounded_copy(source: IO[bytes], output: IO[bytes], maximum: int) -> None:
    size = 0
    while chunk := source.read(_CHUNK_BYTES):
        size += len(chunk)
        if size > maximum:
            raise BundleError("extraction size limit exceeded")
        output.write(chunk)
