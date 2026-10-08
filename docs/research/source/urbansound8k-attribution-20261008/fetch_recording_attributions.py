"""Record primary Freesound canonical uploader URLs for the fixed 40 source IDs."""

# ruff: noqa: TRY301 -- retain each failed request before bounded retry
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, override


ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).parent
SELECTION = ROOT / "docs/research/source/envsdd/full-family-20261008/selection.json"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    @override
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def canonical_uploader(fsid: int, location: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(
        urllib.parse.urljoin(f"https://freesound.org/s/{fsid}/", location)
    )
    if parsed.scheme != "https" or parsed.netloc != "freesound.org":
        raise ValueError("canonical redirect leaves primary host")
    match = re.fullmatch(r"/people/([^/]+)/sounds/(\d+)/", parsed.path)
    if match is None or int(match[2]) != fsid or parsed.query or parsed.fragment:
        raise ValueError("redirect does not identify the exact source recording")
    return urllib.parse.unquote(match[1]), parsed.geturl()


def fetch(fsid: int) -> dict[str, Any]:
    url = f"https://freesound.org/s/{fsid}/"
    attempts: list[dict[str, Any]] = []
    opener = urllib.request.build_opener(NoRedirect())
    for attempt in range(1, 3):
        try:
            try:
                response = opener.open(
                    urllib.request.Request(url, method="HEAD"), timeout=20
                )
            except urllib.error.HTTPError as error:
                response = error
            with response as result:
                headers = {
                    k.lower(): v
                    for k, v in result.headers.items()
                    if k.lower() in {"date", "location", "last-modified", "etag"}
                }
                if result.status not in {301, 302, 307, 308}:
                    raise ValueError(f"unexpected primary HTTP status {result.status}")
                uploader, canonical = canonical_uploader(fsid, headers["location"])
                return {
                    "fsid": fsid,
                    "url": url,
                    "uploader_username": uploader,
                    "canonical_url": canonical,
                    "status": result.status,
                    "headers": headers,
                    "at": datetime.now(UTC).isoformat(),
                    "verified": True,
                    "failed_attempts": attempts,
                    "evidence_scope": "Canonical uploader URL from primary HTTP redirect; not a per-recording license text or official archive verification",
                }
        except Exception as exc:  # Every request failure is retained.
            attempts.append({"attempt": attempt, "error": str(exc)})
            if attempt < 2:
                time.sleep(1)
    return {"fsid": fsid, "url": url, "verified": False, "attempts": attempts}


def main() -> None:
    if hashlib.sha256(SELECTION.read_bytes()).hexdigest() != (
        "554f77b083bbc3c649516c37045064c27e952147d675c6e3125cc396d79749a6"
    ):
        raise ValueError("frozen source selection changed")
    selection = json.loads(SELECTION.read_text())
    fsids = sorted({r["fsid"] for r in selection["samples"]})
    if len(fsids) != 40:
        raise ValueError("fixed source scope changed")
    path = DOC / (
        "recording-attributions-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + ".jsonl"
    )
    with path.open("x") as stream, ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, fsid) for fsid in fsids]
        for future in as_completed(futures):
            record = future.result()
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    print(
        json.dumps(
            {
                "records": str(path),
                "verified": sum(r["verified"] for r in records),
                "expected": 40,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
