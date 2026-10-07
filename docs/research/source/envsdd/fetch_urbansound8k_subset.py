"""Fetch only media listed in a reviewed EnvSDD subset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


def fetch(manifest_path: Path, destination: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    records: list[dict[str, Any]] = []
    for sample in manifest["samples"]:
        relative = Path("audio") / sample["label"] / str(sample["fsid"]) / (
            f"{sample['row_idx']}.wav"
        )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            content = target.read_bytes()
        else:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(sample["audio_url"], timeout=90) as source:
                            with temporary.open("wb") as output:
                                shutil.copyfileobj(source, output)
                        break
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        if attempt == 2:
                            raise
                        time.sleep(1 + attempt)
                content = temporary.read_bytes()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if not content.startswith(b"RIFF"):
            raise ValueError(f"download is not a WAV RIFF file: row {sample['row_idx']}")
        records.append(
            {
                **{key: value for key, value in sample.items() if key != "audio_url"},
                "relative_path": str(relative),
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    records = fetch(args.manifest, args.destination)
    (args.destination / "manifest-with-hashes.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    )
