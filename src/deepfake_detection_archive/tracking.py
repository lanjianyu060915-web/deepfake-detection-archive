"""Atomic reports and content identities for reproducible local/offline runs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def parse_json(data: str | bytes) -> object:
    return json.loads(
        data, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def object_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def code_identity() -> dict[str, object]:
    package = Path(__file__).parent
    hashes: dict[str, str] = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".txt"):
            hashes[path.relative_to(package).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    result: dict[str, object] = {"code_sha256": object_hash(hashes)}
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],  # noqa: S607
            cwd=package,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        for key, args in (
            ("commit", ["git", "rev-parse", "HEAD"]),
            ("dirty", ["git", "status", "--porcelain"]),
        ):
            output = subprocess.run(  # noqa: S603
                args,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            result[key] = bool(output) if key == "dirty" else output
    except (OSError, subprocess.SubprocessError):
        result.update(commit="unavailable", dirty=None)
    return result


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
