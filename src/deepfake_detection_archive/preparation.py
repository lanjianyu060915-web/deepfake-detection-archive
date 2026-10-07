"""Manifest indexing and real decoding audits without model or label guessing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .data.manifest import SampleRecord, file_sha256


def index_media(root: Path, modality: str, source: str) -> list[SampleRecord]:
    extensions = {
        "audio": {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"},
        "video": {".mp4", ".mkv", ".avi", ".webm", ".mov", ".mpeg", ".m4v"},
    }
    if modality not in extensions:
        raise ValueError("modality must be audio or video")
    root = root.resolve(strict=True)
    records: list[SampleRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions[modality]:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"symlink media must be explicitly registered: {path}")
        relative = path.relative_to(root).as_posix()
        records.append(
            SampleRecord(
                sample_id=relative,
                path=relative,
                modality=modality,
                label=None,
                source=source,
                sha256=file_sha256(path),
                original_id=relative,
                derivation_group_id=relative,
                split="test",
            )
        )
    if not records:
        raise ValueError("no supported media files found")
    return records


def _checked_media_path(record: SampleRecord, root: Path) -> tuple[Path, str]:
    from .media import MediaError  # noqa: PLC0415 -- optional media dependency

    path = (root / record.path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise MediaError("path_escape", "resolved file is outside data_root")
    digest = file_sha256(path)
    if record.sha256 not in ("unknown", digest):
        raise MediaError("hash_mismatch", "registered bytes have changed")
    return path, digest


def decode_manifest(records: list[SampleRecord], root: Path) -> list[SampleRecord]:
    from .media import MediaError, decode_audio, decode_video  # noqa: PLC0415

    root = root.resolve(strict=True)
    result: list[SampleRecord] = []
    for record in records:
        if record.status == "excluded":
            result.append(record)
            continue
        try:
            path, digest = _checked_media_path(record, root)
            if record.modality == "audio":
                audio = decode_audio(path)
                updated = replace(
                    record,
                    status="available",
                    failure_reason="unknown",
                    sha256=digest,
                    decode_status="ok",
                    duration_seconds=audio.original_duration_seconds,
                    sample_rate=audio.original_sample_rate,
                )
            else:
                video = decode_video(path)
                updated = replace(
                    record,
                    status="available",
                    failure_reason="unknown",
                    sha256=digest,
                    decode_status="ok",
                    duration_seconds=video.original_duration_seconds,
                )
            result.append(updated)
        except (OSError, MediaError) as exc:
            result.append(
                replace(
                    record,
                    status="decode_failed",
                    decode_status="failed",
                    failure_reason=f"{getattr(exc, 'code', 'media_io_error')}: {exc}",
                )
            )
    return result


def media_snapshot(
    records: Sequence[SampleRecord], root: Path
) -> list[dict[str, object]]:
    """Hash every present input; persist missing/read-error states without omission.

    Used in run identity before new inference and before any resume. A newly
    available file or changed bytes invalidates the previous run's identity.
    """
    root = root.resolve()
    result: list[dict[str, object]] = []
    for record in records:
        item: dict[str, object] = {
            "sample_id": record.sample_id,
            "path": record.path,
            "sha256": None,
            "state": "missing",
        }
        try:
            path = (root / record.path).resolve(strict=True)
            if not path.is_relative_to(root):
                item["state"] = "path_escape"
            elif not path.is_file():
                item["state"] = "not_file"
            else:
                item.update(sha256=file_sha256(path), state="present")
        except FileNotFoundError:
            pass
        except (OSError, RuntimeError) as exc:
            item.update(
                state="read_error",
                error_type=type(exc).__name__,
                errno=getattr(exc, "errno", None),
            )
        result.append(item)
    return result
