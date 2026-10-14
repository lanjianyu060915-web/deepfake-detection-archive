"""Behavior checks for the In-the-Wild import; run with unittest discovery."""

from __future__ import annotations

# ruff: noqa: PT009, PT027 - standalone unittest checks alongside the CLI tool
import json
import tempfile
import unittest
from pathlib import Path

import prepare_in_the_wild

from deepfake_detection_archive.data.manifest import load_manifest


def _release(root: Path, rows: list[tuple[str, str, str]]) -> Path:
    release = root / "release_in_the_wild"
    release.mkdir()
    lines = ["file,speaker,label"]
    for name, speaker, label in rows:
        lines.append(f"{name},{speaker},{label}")
        (release / name).write_bytes(b"RIFF" + name.encode())
    (release / "meta.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return release


class PrepareInTheWildTests(unittest.TestCase):
    def test_writes_speaker_grouped_test_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = _release(
                root,
                [
                    ("0.wav", "Alec Guinness", "spoof"),
                    ("1.wav", "Alec Guinness", "bona-fide"),
                    ("2.wav", "Barack Obama", "bona-fide"),
                ],
            )
            summary = prepare_in_the_wild.materialize(
                release, root / "out", "0" * 64, "2026-10-13"
            )
            records = load_manifest(root / "out" / "manifest.jsonl")
            source = json.loads((root / "out" / "sources.jsonl").read_text())
        self.assertEqual(summary["samples"], 3)
        self.assertEqual((summary["real"], summary["fake"]), (2, 1))
        self.assertEqual(summary["speakers"], 2)
        by_id = {record.sample_id: record for record in records}
        self.assertEqual(by_id["itw-0"].label, 1)
        self.assertEqual(by_id["itw-0"].group_id, by_id["itw-1"].group_id)
        self.assertEqual(by_id["itw-2"].path, "2.wav")
        self.assertTrue(all(record.split == "test" for record in records))
        self.assertEqual(source["license_status"], "unresolved")
        self.assertNotIn("training", source["allowed_uses"])

    def test_rejects_unknown_labels_and_missing_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = _release(root, [("0.wav", "A", "fake")])
            with self.assertRaisesRegex(ValueError, "unknown label"):
                prepare_in_the_wild.materialize(release, root / "a", "0" * 64, "x")
            (release / "meta.csv").write_text(
                "file,speaker,label\n9.wav,A,spoof\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "missing media"):
                prepare_in_the_wild.materialize(release, root / "b", "0" * 64, "x")


if __name__ == "__main__":
    unittest.main()
