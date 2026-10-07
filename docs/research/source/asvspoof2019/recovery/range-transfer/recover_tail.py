"""Recover this interrupted archive with three independently resumable ranges."""

# ruff: noqa: TRY301 - retain inline checks in this exact one-off transfer log.

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
ARCHIVE = ROOT / "cache/downloads/asvspoof2019-LA.zip.part"
PARTS = ROOT / "cache/downloads/asvspoof2019-tail"
REPORT = ROOT / "outputs/data-acquisition/asvspoof2019-tail-transfer.json"
URL = "https://zenodo.org/api/records/6906306/files/LA.zip/content"
PREFIX = 5757300704
TOTAL = 7640952520
EXPECTED_MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
LOCK = threading.Lock()
REQUESTS: list[dict[str, object]] = []
RECORD: dict[str, object] = {"requests": REQUESTS}


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def save() -> None:
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text(json.dumps(RECORD, indent=2, sort_keys=True) + "\n")
    temporary.replace(REPORT)


def fetch(item: tuple[int, int, int]) -> dict[str, object]:
    index, begin, end = item
    path = PARTS / f"tail-{index}.part"
    deadline = time.monotonic() + 7200
    attempts = 0
    while (current := path.stat().st_size if path.exists() else 0) < end - begin + 1:
        if time.monotonic() > deadline or attempts >= 12:
            raise RuntimeError(f"range {index} retry/time limit; partial preserved")
        attempts += 1
        start = begin + current
        entry: dict[str, object] = {
            "range_index": index,
            "requested_range": [start, end],
            "started_at_utc": datetime.now(UTC).isoformat(),
            "received_bytes": 0,
        }
        try:
            request = urllib.request.Request(
                URL,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS URL
                content_range = response.headers.get("Content-Range", "")
                entry.update(status=response.status, content_range=content_range)
                parsed = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if (
                    response.status != 206
                    or parsed is None
                    or tuple(map(int, parsed.groups())) != (start, end, TOTAL)
                ):
                    raise ValueError("server did not return the exact requested range")
                if int(response.headers["Content-Length"]) != end - start + 1:
                    raise ValueError("unexpected range Content-Length")
                received = 0
                with path.open("ab") as output:
                    while block := response.read(
                        min(1024**2, end - start + 1 - received)
                    ):
                        output.write(block)
                        received += len(block)
                        entry["received_bytes"] = received
                        if received == end - start + 1:
                            break
                    output.flush()
                    os.fsync(output.fileno())
                if received != end - start + 1:
                    raise OSError("short response; next attempt resumes actual bytes")
        except (OSError, ValueError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            entry["finished_at_utc"] = datetime.now(UTC).isoformat()
            with LOCK:
                REQUESTS.append(entry)
                save()
        if "error" in entry:
            time.sleep(min(5 * attempts, 30))
    if path.stat().st_size != end - begin + 1:
        raise ValueError("range exceeds expected length")
    return {"path": str(path), "begin": begin, "end": end, "sha256": digest(path)}


def main() -> None:
    if ARCHIVE.stat().st_size != PREFIX or REPORT.exists():
        raise ValueError(
            "prefix changed or prior transfer report exists; inspect first"
        )
    PARTS.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    prefix_hash = digest(ARCHIVE)
    RECORD.update(
        scope="fixed ASVspoof LA compressed archive tail; no media extraction",
        started_at_utc=datetime.now(UTC).isoformat(),
        source_url=URL,
        prefix_bytes=PREFIX,
        prefix_sha256=prefix_hash,
        expected_bytes=TOTAL,
        expected_md5=EXPECTED_MD5,
        script_sha256=digest(Path(__file__)),
        maximum_connections=3,
    )
    save()
    step = (TOTAL - PREFIX + 2) // 3
    ranges = [
        (i, PREFIX + i * step, min(TOTAL - 1, PREFIX + (i + 1) * step - 1))
        for i in range(3)
    ]
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            completed = list(pool.map(fetch, ranges))
        if ARCHIVE.stat().st_size != PREFIX or digest(ARCHIVE) != prefix_hash:
            raise ValueError("archive prefix changed during transfer")
        for part in completed:
            path = Path(str(part["path"]))
            if digest(path) != part["sha256"]:
                raise ValueError("completed range bytes changed")
        with ARCHIVE.open("ab") as output:
            for part in completed:
                with Path(str(part["path"])).open("rb") as source:
                    shutil.copyfileobj(source, output, length=8 * 1024**2)
            output.flush()
            os.fsync(output.fileno())
        actual_md5 = digest(ARCHIVE, "md5")
        if ARCHIVE.stat().st_size != TOTAL or actual_md5 != EXPECTED_MD5:
            raise ValueError("complete archive differs from official size/MD5")
        RECORD.update(
            ranges=completed,
            actual_md5=actual_md5,
            archive_sha256=digest(ARCHIVE),
            completed=True,
            finished_at_utc=datetime.now(UTC).isoformat(),
        )
    except Exception as exc:
        RECORD.update(completed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
