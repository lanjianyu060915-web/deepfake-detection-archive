"""Continue the fixed official archive with bounded, validated HTTP byte ranges."""

# ruff: noqa: TRY301 -- retain exact request failures before bounded retry
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).parent
TARGET = ROOT / "cache/downloads/UrbanSound8K.tar.gz.part"
URL = "https://zenodo.org/api/records/1203745/files/UrbanSound8K.tar.gz/content"
TOTAL = 6023741708
CHUNK = 256 * 1024 * 1024


def download_range(start: int, end: int, temporary: Path) -> tuple[int, str]:
    expected = end - start + 1
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310 -- fixed HTTPS official source
        if (
            response.status != 206
            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{TOTAL}"
        ):
            raise ValueError("server did not return the exact range")
        count = 0
        with temporary.open("xb") as output:
            while block := response.read(min(1 << 20, expected - count + 1)):
                count += len(block)
                if count > expected:
                    raise ValueError("range exceeds declared budget")
                output.write(block)
                digest.update(block)
        if count != expected:
            raise ValueError("incomplete range; original prefix intact")
    return count, digest.hexdigest()


def main() -> None:
    log_path = DOC / (
        "range-recovery-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".jsonl"
    )
    with log_path.open("x") as log:
        while TARGET.stat().st_size < TOTAL:
            start = TARGET.stat().st_size
            end = min(TOTAL, start + CHUNK) - 1
            temporary = TARGET.with_suffix(".range-tmp")
            if temporary.exists():
                raise ValueError("existing range temporary requires explicit review")
            for attempt in range(1, 4):
                record = {
                    "start": start,
                    "end": end,
                    "attempt": attempt,
                    "at": datetime.now(UTC).isoformat(),
                }
                try:
                    count, digest = download_range(start, end, temporary)
                    if TARGET.stat().st_size != start:
                        raise ValueError("another writer changed the original prefix")
                    with TARGET.open("ab") as output, temporary.open("rb") as source:
                        for block in iter(lambda: source.read(1 << 20), b""):
                            output.write(block)
                    if TARGET.stat().st_size != end + 1:
                        raise ValueError("append length mismatch")
                    temporary.unlink()
                    record.update(status="appended", bytes=count, sha256=digest)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                    break
                except (
                    Exception
                ) as exc:  # Retain errors and any partial range separately.
                    record.update(status="failed", error=str(exc))
                    if temporary.exists():
                        failed = temporary.with_name(
                            temporary.name
                            + f".failed-{start}-{attempt}-{time.monotonic_ns()}"
                        )
                        temporary.rename(failed)
                        record["partial_range"] = str(failed.relative_to(ROOT))
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    if attempt == 3 or TARGET.stat().st_size != start:
                        raise
                    time.sleep(attempt * 2)
        if TARGET.stat().st_size != TOTAL:
            raise ValueError("archive exceeds fixed source size")
    print(
        json.dumps({"complete_bytes_only": True, "hash_verification_required": True}),
        flush=True,
    )


if __name__ == "__main__":
    main()
