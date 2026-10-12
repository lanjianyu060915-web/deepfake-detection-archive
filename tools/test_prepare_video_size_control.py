"""Behavior checks for the size-controlled ViF-Bench view; unittest discovery."""

from __future__ import annotations

# ruff: noqa: PT009 - standalone unittest checks alongside the CLI tool
import unittest
from typing import Any

from prepare_video_size_control import rank, select_pairs, summarise

from deepfake_detection_archive.data.manifest import SampleRecord


def record(
    sid: str, label: int, generator: str = "gen", parent: str = "unknown"
) -> SampleRecord:
    return SampleRecord(
        sample_id=sid,
        path=f"{sid}.mp4",
        modality="video",
        label=label,
        source="fixture",
        generator=generator,
        parent_id=parent,
    )


def metadata(*entries: tuple[str, int, int]) -> dict[str, dict[str, Any]]:
    return {
        sid: {
            "sample_id": sid,
            "sha256": "0" * 64,
            "width": width,
            "height": height,
            "fps": 16.0,
            "codec": "h264",
            "container": "mov",
        }
        for sid, width, height in entries
    }


def row(*entries: tuple[str, int, int]) -> list[SampleRecord]:
    return [record(sid, label) for sid, label, _ in entries]


class SizeControlTests(unittest.TestCase):
    def test_pairs_require_an_identical_size_partner(self) -> None:
        rows = [
            record("r1", 0),
            record("f1", 1, parent="r1"),
            record("r2", 0),
            record("f2", 1, parent="r2"),
            record("f3", 1, generator="orphan"),
        ]
        meta = metadata(
            ("r1", 456, 256),
            ("f1", 456, 256),
            ("r2", 452, 256),
            ("f2", 384, 256),
            ("f3", 456, 256),
        )
        pairs, geometry, reasons = select_pairs(rows, meta, 42)
        self.assertEqual([pair["fake_sample_id"] for pair in pairs], ["f1"])
        self.assertEqual(pairs[0]["real_sample_id"], "r1")
        self.assertEqual(geometry["gen"], [(456, 256, 1)])
        self.assertEqual(reasons["paired_real_size_differs"], 1)
        self.assertEqual(reasons["no_paired_real"], 1)

    def test_bucket_supply_is_balanced_and_order_independent(self) -> None:
        rows = [
            record("r1", 0),
            record("r2", 0),
            record("r3", 0),
            record("f1", 1, parent="r1"),
            record("f2", 1, parent="r2"),
            record("f3", 1, parent="r3"),
        ]
        meta = metadata(
            ("r1", 456, 256),
            ("r2", 456, 256),
            ("r3", 456, 256),
            ("f1", 456, 256),
            ("f2", 456, 256),
            ("f3", 456, 256),
        )
        pairs, _, reasons = select_pairs(rows, meta, 42)
        self.assertEqual(len(pairs), 3)
        shuffled, _, _ = select_pairs(list(reversed(rows)), meta, 42)
        self.assertEqual(pairs, shuffled)
        self.assertEqual(reasons["paired_real_supply_exhausted"], 0)
        # A second generator also matches: the real partners are reused there.
        other = [*rows, record("f4", 1, generator="other", parent="r1")]
        other_meta = {**meta, "f4": {**meta["f1"], "sample_id": "f4"}}
        pairs_other, geometry_other, _ = select_pairs(other, other_meta, 42)
        self.assertEqual(len(pairs_other), 4)
        self.assertEqual(sorted(geometry_other), ["gen", "other"])

    def test_probe_failure_and_summary_counts(self) -> None:
        rows = [
            record("r1", 0),
            record("r2", 0),
            record("f1", 1, parent="r1"),
            record("f2", 1, parent="r2"),
        ]
        meta = metadata(("r1", 456, 256), ("f1", 456, 256), ("r2", 456, 256))
        meta["f2"] = {"sample_id": "f2", "sha256": "0" * 64, "error": "boom"}
        pairs, geometry, reasons = select_pairs(rows, meta, 42)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(reasons["fake_probe_failed"], 1)
        summary = summarise(rows, meta, pairs, geometry, reasons)
        self.assertEqual(summary["pairs"], 1)
        self.assertEqual(summary["unique_samples"], 2)
        self.assertEqual(summary["matched_sizes"]["1"]["count"], 1)
        self.assertEqual(summary["matched_sizes"]["1"]["width_height"], {"456x256": 1})
        self.assertEqual(summary["per_generator"]["gen"]["pairs"], 1)
        self.assertEqual(summary["generators_without_pairs"], [])
        self.assertEqual(summary["probe_failures"], {"f2": "boom"})

    def test_generators_without_size_matched_reals_are_reported(self) -> None:
        rows = [
            record("r1", 0),
            record("f1", 1, parent="r1"),
            record("f9", 1, generator="huge", parent="r1"),
        ]
        meta = metadata(("r1", 456, 256), ("f1", 456, 256), ("f9", 928, 544))
        pairs, geometry, reasons = select_pairs(rows, meta, 42)
        self.assertEqual(len(pairs), 1)
        summary = summarise(rows, meta, pairs, geometry, reasons)
        self.assertEqual(summary["generators_without_pairs"], ["huge"])
        self.assertEqual(summary["input_by_generator"], {"gen": 1, "huge": 1})
        self.assertEqual(summary["unmatched_fakes"]["paired_real_size_differs"], 1)

    def test_rank_is_seeded_and_stable(self) -> None:
        self.assertEqual(rank("x", 42), rank("x", 42))
        self.assertNotEqual(rank("x", 42), rank("x", 43))


if __name__ == "__main__":
    unittest.main()
