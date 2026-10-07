"""Strict, portable JSONL records. Labels follow real=0 and fake=1."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast


UNKNOWN = "unknown"
SPLITS = ("train", "dev", "holdout")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string (use 'unknown')")
    return value


def _sha(value: str) -> None:
    if value != UNKNOWN and re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("sha256 must be lowercase hexadecimal or 'unknown'")


def _relative(value: str) -> None:
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or "\\" in value
        or "\x00" in value
        or value in (".", "")
    ):
        raise ValueError("path must be a relative POSIX path without traversal")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    path: str
    modality: str
    label: int | None
    source: str
    content_type: str = UNKNOWN
    generator: str = UNKNOWN
    group_id: str = UNKNOWN
    sha256: str = UNKNOWN
    license_id: str = UNKNOWN
    original_id: str = UNKNOWN
    parent_id: str = UNKNOWN
    identity_id: str = UNKNOWN
    derivation_group_id: str = UNKNOWN
    generator_version: str = UNKNOWN
    container: str = UNKNOWN
    codec: str = UNKNOWN
    duration_seconds: float | None = None
    frame_rate: float | None = None
    sample_rate: int | None = None
    split: str = "unassigned"
    status: str = "available"
    failure_reason: str = UNKNOWN
    decode_status: str = UNKNOWN

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name not in {
                "label",
                "duration_seconds",
                "frame_rate",
                "sample_rate",
            }:
                _text(getattr(self, field.name), field.name)
        if self.sample_id == UNKNOWN:
            raise ValueError("sample_id cannot be unknown")
        _relative(self.path)
        _sha(self.sha256)
        if self.modality not in ("audio", "video"):
            raise ValueError("modality must be audio or video")
        if self.label is not None and (
            type(self.label) is not int or self.label not in (0, 1)
        ):
            raise ValueError("label must be integer 0, 1 or null")
        if self.split not in (*SPLITS, "unassigned", "test"):
            raise ValueError("invalid split")
        if self.status not in ("available", "missing", "decode_failed", "excluded"):
            raise ValueError("invalid availability status")
        if self.decode_status not in (UNKNOWN, "ok", "failed"):
            raise ValueError("decode_status must be unknown, ok or failed")
        if self.status != "available" and self.failure_reason == UNKNOWN:
            raise ValueError("unavailable samples need an explicit failure_reason")
        if self.decode_status == "failed" and self.failure_reason == UNKNOWN:
            raise ValueError("decode failure needs an explicit failure_reason")
        self._validate_numbers()

    def _validate_numbers(self) -> None:
        for name in ("duration_seconds", "frame_rate", "sample_rate"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive finite or null")
        if self.sample_rate is not None and type(self.sample_rate) is not int:
            raise ValueError("sample_rate must be integer or null")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceRecord:
    source: str
    url: str
    version: str
    license_id: str
    license_url: str
    downloaded_at: str
    allowed_uses: tuple[str, ...]
    license_status: str = "unresolved"
    archive_sha256: str = UNKNOWN

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name != "allowed_uses":
                _text(getattr(self, field.name), field.name)
        if self.source == UNKNOWN:
            raise ValueError("source identifier cannot be unknown")
        if not isinstance(self.allowed_uses, tuple) or any(
            not isinstance(use, str) or not use.strip() for use in self.allowed_uses
        ):
            raise ValueError("allowed_uses must be a tuple of nonempty strings")
        if self.license_status not in ("approved", "unresolved", "denied"):
            raise ValueError("invalid license_status")
        _sha(self.archive_sha256)


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(text: str) -> dict[str, object]:
    value: object = json.loads(text, object_pairs_hook=_object)
    if not isinstance(value, dict):
        raise ValueError("record must be a JSON object")  # noqa: TRY004 -- schema error
    return cast(dict[str, object], value)


def sample_from_dict(value: dict[str, object]) -> SampleRecord:
    allowed = {field.name for field in fields(SampleRecord)}
    extras = set(value) - allowed
    if extras:
        raise ValueError(f"unknown sample fields: {sorted(extras)}")
    try:
        return SampleRecord(**value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"invalid sample schema: {exc}") from exc


def load_manifest(path: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = sample_from_dict(_read_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if record.sample_id in seen:
            raise ValueError(
                f"{path}:{number}: duplicate sample_id: {record.sample_id}"
            )
        seen.add(record.sample_id)
        records.append(record)
    if not records:
        raise ValueError("manifest is empty")
    return records


def manifest_bytes(records: list[SampleRecord]) -> bytes:
    ids = [record.sample_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample_id")
    return "".join(
        json.dumps(
            record.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
        for record in sorted(records, key=lambda item: item.sample_id)
    ).encode("utf-8")


def manifest_hash(records: list[SampleRecord]) -> str:
    """Canonical hash includes split and provenance; independent of input row order."""
    return hashlib.sha256(manifest_bytes(records)).hexdigest()


def write_manifest(records: list[SampleRecord], path: Path) -> str:
    content = manifest_bytes(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def source_from_dict(value: dict[str, object]) -> SourceRecord:
    allowed = {field.name for field in fields(SourceRecord)}
    if set(value) - allowed:
        raise ValueError("unknown source fields")
    uses = value.get("allowed_uses")
    if not isinstance(uses, list):
        raise ValueError("allowed_uses must be a JSON list")  # noqa: TRY004
    value = {**value, "allowed_uses": tuple(uses)}
    try:
        return SourceRecord(**value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"invalid source schema: {exc}") from exc


def load_sources(path: Path) -> dict[str, SourceRecord]:
    """Read a JSONL registry. Permission is asserted by a human-reviewed record."""
    result: dict[str, SourceRecord] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = source_from_dict(_read_json(line))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if record.source in result:
            raise ValueError(f"{path}:{number}: duplicate source: {record.source}")
        result[record.source] = record
    return result


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
