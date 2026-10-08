"""Bounded prefix probe of the official UrbanSound8K archive for attribution text."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
DOC = Path(__file__).parent
LIMIT = 1_048_576
URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"


def main() -> None:
    result: dict[str, Any] = {
        "at": datetime.now(UTC).isoformat(),
        "url": URL,
        "range": f"bytes=0-{LIMIT - 1}",
        "limit": LIMIT,
        "complete_archive_hash_verified": False,
        "members": [],
    }
    request = urllib.request.Request(URL, headers={"Range": f"bytes=0-{LIMIT - 1}"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310
            result["headers"] = {
                k.lower(): v
                for k, v in response.headers.items()
                if k.lower()
                in {"date", "content-range", "content-length", "etag", "content-type"}
            }
            result["status"] = response.status
            raw = response.read(LIMIT)
        result["bytes"] = len(raw)
        result["sha256"] = hashlib.sha256(raw).hexdigest()
        target = (
            ROOT
            / "data/envsdd-urban-full-family/source-excerpts/official-archive-prefix.gz"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        result["local_path"] = str(target.relative_to(ROOT))
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r|gz") as archive:
            for member in archive:
                result["members"].append({"name": member.name, "bytes": member.size})
                if member.name.endswith("FREESOUNDCREDITS.txt"):
                    member_stream = archive.extractfile(member)
                    if member_stream is None or member.size > 1_000_000:
                        raise ValueError("unexpected attribution member")  # noqa: TRY301
                    content = member_stream.read()
                    if len(content) != member.size:
                        raise ValueError("truncated attribution member")  # noqa: TRY301
                    destination = DOC / "license-sources/FREESOUNDCREDITS.txt"
                    with destination.open("xb") as output:
                        output.write(content)
                    result["credits_sha256"] = hashlib.sha256(content).hexdigest()
                    result["credits_bytes"] = len(content)
                    result["credits_recovered"] = True
                    break
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    with (DOC / "license-sources/official-credits-probe.json").open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
