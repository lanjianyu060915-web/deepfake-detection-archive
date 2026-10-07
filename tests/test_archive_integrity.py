"""Protect the historical evidence while the competition workspace evolves."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_files_match_the_migration_snapshot() -> None:
    manifest = json.loads((ROOT / "archive/integrity-manifest.json").read_text())
    archive_root = ROOT / manifest["archive_root"]
    records = manifest["files"]
    assert len(records) == manifest["file_count"]
    assert len({record["path"] for record in records}) == len(records)
    total = 0
    for record in records:
        content = (archive_root / record["path"]).read_bytes()
        assert len(content) == record["bytes"], record["path"]
        assert hashlib.sha256(content).hexdigest() == record["sha256"], record["path"]
        total += len(content)
    assert total == manifest["total_bytes"]


def test_backup_restoration_tool_was_preserved() -> None:
    manifest = json.loads((ROOT / "archive/integrity-manifest.json").read_text())
    for relative, expected in manifest["preserved_tools"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
