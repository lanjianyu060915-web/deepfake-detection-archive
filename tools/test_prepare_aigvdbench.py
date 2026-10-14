"""Behavior checks for the AIGVDBench import; run with unittest discovery."""

from __future__ import annotations

# ruff: noqa: PT009, PT027 - standalone unittest checks alongside the CLI tool
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import prepare_aigvdbench

from deepfake_detection_archive.data.manifest import load_manifest


PROBE = {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "duration": 4.0,
    "nb_frames": 120,
    "vcodec": "h264",
    "has_audio": False,
    "bit_rate": 1,
}


def _row(
    member: str, label: int, generator: str, role: str, digest: str | None, **extra: Any
) -> dict[str, Any]:
    row = {
        "zip": "Real/Real.zip" if label == 0 else f"{generator}.zip",
        "member": member,
        "local_path": member,
        "label": label,
        "generator": "real" if label == 0 else generator,
        "role": role,
        "paired_real": None,
        "youtube_id": member[:11] if label == 0 else None,
        "bytes": 1,
        "crc32_ok": True,
        "sha256": digest,
        "status": "ok",
        "error": None,
        "ffprobe": PROBE,
    }
    row.update(extra)
    return row


def _write(root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    download = root / "download.jsonl"
    download.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    selection = root / "selection.json"
    selection.write_text(json.dumps({"revision": "abc123"}), encoding="utf-8")
    return download, selection


class PrepareAigvdbenchTests(unittest.TestCase):
    def test_pairs_share_group_and_splits_stay_isolated(self) -> None:
        rows = [
            _row("AAAAAAAAAAA_1_0to9.mp4", 0, "real", "dev", "a" * 64),
            _row(
                "AAAAAAAAAAA_1_0to9.mp4",
                1,
                "Open-Sora",
                "dev",
                "b" * 64,
                paired_real="AAAAAAAAAAA_1_0to9.mp4",
            ),
            _row("BBBBBBBBBBB_2_0to9.mp4", 0, "real", "holdout", "c" * 64),
            _row("0001.mp4", 1, "Sora", "holdout", "d" * 64),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = prepare_aigvdbench.prepare(*_write(root, rows), root / "out", "x")
            records = {
                r.sample_id: r for r in load_manifest(root / "out/manifest.jsonl")
            }
        real = records["aigvdb:real:AAAAAAAAAAA_1_0to9"]
        fake = records["aigvdb:Open-Sora:AAAAAAAAAAA_1_0to9"]
        self.assertEqual(real.group_id, fake.group_id)
        self.assertEqual(fake.parent_id, real.sample_id)
        self.assertEqual(fake.path, "fake/Open-Sora/AAAAAAAAAAA_1_0to9.mp4")
        self.assertEqual(records["aigvdb:Sora:0001"].split, "holdout")
        self.assertEqual(summary["isolation"]["leaking_groups"], [])
        self.assertTrue(all(r.status == "available" for r in records.values()))

    def test_identical_bytes_across_splits_are_excluded(self) -> None:
        rows = [
            _row("0001.mp4", 1, "Luma", "dev", "e" * 64),
            _row("0002.mp4", 1, "Luma", "holdout", "e" * 64),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = prepare_aigvdbench.prepare(*_write(root, rows), root / "out", "x")
            records = load_manifest(root / "out/manifest.jsonl")
        self.assertEqual({r.status for r in records}, {"excluded"})
        self.assertEqual(len(summary["isolation"]["byte_duplicate_members"]), 1)

    def test_failed_downloads_stay_as_missing(self) -> None:
        rows = [
            _row(
                "0003.mp4",
                1,
                "Gen3",
                "dev",
                None,
                status="failed",
                error="crc mismatch",
                ffprobe=None,
                local_path=None,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_aigvdbench.prepare(*_write(root, rows), root / "out", "x")
            (record,) = load_manifest(root / "out/manifest.jsonl")
        self.assertEqual(record.status, "missing")
        self.assertEqual(record.failure_reason, "crc mismatch")

    def test_rejects_real_without_youtube_id(self) -> None:
        rows = [
            _row("CCCCCCCCCCC_1_0to9.mp4", 0, "real", "dev", "f" * 64, youtube_id=None)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "youtube_id"):
                prepare_aigvdbench.prepare(*_write(root, rows), root / "out", "x")


if __name__ == "__main__":
    unittest.main()
