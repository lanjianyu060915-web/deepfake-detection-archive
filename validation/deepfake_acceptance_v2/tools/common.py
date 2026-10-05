"""Shared, dependency-light helpers for D-v2 dataset governance tools."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image


GROUND_TRUTH_FIELDS = [
    "sample_id",
    "file_path",
    "label",
    "source",
    "generation_method",
    "source_face_id",
    "target_face_id",
    "is_original",
    "parent_sample_id",
    "perturbation_type",
    "perturbation_level",
    "width",
    "height",
    "sha256",
    "notes",
]

LABELS = {"REAL", "FAKE"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()
        return image.size


def image_dimensions_bytes(data: bytes) -> tuple[int, int]:
    from io import BytesIO

    with Image.open(BytesIO(data)) as image:
        image.load()
        return image.size


def parse_bool(value: object, field: str = "value") -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"true", "false"}:
        return value == "true"
    raise ValueError(f"{field} must be exactly 'true' or 'false'")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def resolve_dataset_path(dataset_root: Path, file_path: str) -> Path:
    """Resolve a manifest path while rejecting absolute paths and traversal."""
    if not file_path:
        raise ValueError("file_path is empty")
    candidate_text = file_path.replace("\\", "/")
    candidate = Path(candidate_text)
    if candidate.is_absolute() or candidate.drive:
        raise ValueError(f"absolute file_path is forbidden: {file_path}")
    root = dataset_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"file_path escapes dataset root: {file_path}") from exc
    return resolved


def portable_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fields: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_exact_header(actual: list[str], expected: list[str], context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context} header mismatch; expected {expected!r}, got {actual!r}")

