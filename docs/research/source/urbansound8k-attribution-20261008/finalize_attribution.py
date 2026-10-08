"""Verify the complete official archive, then read only attribution text assets."""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).parent
ARCHIVE = ROOT / "cache/downloads/UrbanSound8K.tar.gz.part"
EXPECTED_SIZE = 6023741708
EXPECTED_MD5 = "9aa69802bbf37fb986f71ec1483a196e"
TEXT_MEMBERS = {
    "UrbanSound8K/FREESOUNDCREDITS.txt",
    "UrbanSound8K/UrbanSound8K_README.txt",
    "UrbanSound8K/README.txt",
    "UrbanSound8K/metadata/UrbanSound8K.csv",
}


def main() -> None:
    record = DOC / "official-record.json"
    expected_record = json.loads((DOC / "official-record-evidence.json").read_text())
    if hashlib.sha256(record.read_bytes()).hexdigest() != expected_record["sha256"]:
        raise ValueError("recorded official source metadata changed")
    archive_metadata = expected_record["archive"]
    if (
        archive_metadata["size"] != EXPECTED_SIZE
        or archive_metadata["checksum"] != f"md5:{EXPECTED_MD5}"
        or ARCHIVE.stat().st_size != EXPECTED_SIZE
    ):
        raise ValueError("official archive is not complete; leave download intact")
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with ARCHIVE.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    if md5.hexdigest() != EXPECTED_MD5:
        raise ValueError("complete official archive checksum differs")
    out = DOC / ("assets-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(exist_ok=False)
    assets = {}
    with tarfile.open(ARCHIVE, mode="r|gz") as archive:
        for member in archive:
            if member.name not in TEXT_MEMBERS:
                continue
            if not member.isfile() or not 0 < member.size <= 1_000_000:
                raise ValueError("attribution member is not bounded regular text")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("attribution member cannot be read")
            with source:
                body = source.read(1_000_001)
            if len(body) != member.size:
                raise ValueError("attribution text is truncated")
            target = out / Path(member.name).name
            with target.open("xb") as stream:
                stream.write(body)
            assets[member.name] = {
                "path": str(target.relative_to(ROOT)),
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
    if "UrbanSound8K/FREESOUNDCREDITS.txt" not in assets:
        raise ValueError("official attribution file absent from verified archive")
    result = {
        "at": datetime.now(UTC).isoformat(),
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "size": EXPECTED_SIZE,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets": assets,
        "credits_acquired": True,
        "audio_media_extracted": False,
        "competition_approved": False,
        "per_recording_attribution_review_required": True,
    }
    with (out / "verification.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
