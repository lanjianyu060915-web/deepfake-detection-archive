"""Preserve public primary-source licensing pages, separate from media selection."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


DOC = Path(__file__).parent
SOURCES = {
    "urbansound-home.html": "https://urbansounddataset.weebly.com/",
    "urbansound8k.html": "https://urbansounddataset.weebly.com/urbansound8k.html",
    "envsdd-card.md": "https://huggingface.co/datasets/EnvSDD/EnvSDD/raw/b0832726deb55484dbce869c6f09c41f7e9e5fbb/README.md",
}


def main() -> None:
    destination = DOC / "license-sources"
    destination.mkdir(exist_ok=True)
    evidence = []
    for name, url in SOURCES.items():
        record = {"url": url, "at": datetime.now(UTC).isoformat()}
        try:
            with urllib.request.urlopen(url, timeout=45) as response:  # noqa: S310
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("source exceeds budget")  # noqa: TRY301
                record["final_url"] = response.url
            with (destination / name).open("xb") as stream:
                stream.write(raw)
            record.update(
                status="saved", file=name, sha256=hashlib.sha256(raw).hexdigest()
            )
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        evidence.append(record)
    with (destination / "evidence.json").open("x") as stream:
        stream.write(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
