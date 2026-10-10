"""Behavior checks for video control selection; run with unittest discovery."""

from __future__ import annotations

# ruff: noqa: PT009 - standalone unittest checks alongside the CLI tool
import unittest
from collections import Counter
from dataclasses import replace
from typing import Any

from prepare_video_control import matched, partition

from deepfake_detection_archive.data.manifest import SampleRecord


def record(sid: str, label: int = 0, group: str = "unknown") -> SampleRecord:
    return SampleRecord(
        sample_id=sid,
        path=f"{sid}.mp4",
        modality="video",
        label=label,
        source="fixture",
        derivation_group_id=group,
    )


class VideoControlTests(unittest.TestCase):
    def test_pilot_assignments_and_entire_holdout_component_are_preserved(self) -> None:
        full = [
            record(sid, group=sid[0])
            for sid in ("t1", "t2", "d1", "d2", "h1", "h2", "n1")
        ]
        pilot = [
            replace(full[index], split=split)
            for index, split in ((0, "train"), (2, "dev"), (4, "holdout"))
        ]
        selected, _, excluded = partition(full, pilot, 42)
        actual = {row.sample_id: row.split for row in selected}
        self.assertEqual(excluded, 2)
        self.assertNotIn("h1", actual)
        self.assertNotIn("h2", actual)
        self.assertEqual(actual["t1"], "train")
        self.assertEqual(actual["t2"], "train")
        self.assertEqual(actual["d1"], "dev")
        self.assertEqual(actual["d2"], "dev")
        self.assertEqual(selected, partition(full[::-1], pilot, 42)[0][::-1])

    def test_pilot_extra_relation_is_retained_in_full_grouping(self) -> None:
        full = [record(sid) for sid in ("a", "b", "c")]
        pilot = [
            replace(row, derivation_group_id="pilot-group", split="holdout")
            for row in full[:2]
        ]
        selected, _, excluded = partition(full, pilot, 42)
        self.assertEqual([row.sample_id for row in selected], ["c"])
        self.assertEqual(excluded, 2)

    def test_matching_balances_cells_separately_in_each_split(self) -> None:
        rows = [
            replace(record(f"s{i}", label), split=split)
            for i, (label, split) in enumerate(
                [
                    (0, "train"),
                    (0, "train"),
                    (1, "train"),
                    (1, "dev"),
                    (0, "dev"),
                    (1, "dev"),
                ]
            )
        ]
        meta: dict[str, dict[str, Any]] = {
            row.sample_id: {
                "width": 456,
                "height": 256,
                "fps": 8.0,
                "duration_seconds": 5.1,
            }
            for row in rows
        }
        meta["s5"]["duration_seconds"] = 6.1
        selected = matched(rows, meta, 42, duration=True)
        self.assertEqual(
            Counter((r.split, r.label) for r in selected),
            {
                ("train", 0): 1,
                ("train", 1): 1,
                ("dev", 0): 1,
                ("dev", 1): 1,
            },
        )
        self.assertNotIn("s5", [r.sample_id for r in selected])
        self.assertEqual(selected, matched(rows[::-1], meta, 42, duration=True))


if __name__ == "__main__":
    unittest.main()
