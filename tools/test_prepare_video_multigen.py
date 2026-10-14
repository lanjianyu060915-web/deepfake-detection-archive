"""Regression checks for fixed splits, exclusion closure and shared timing."""

from __future__ import annotations

# ruff: noqa: PT009, PT027, S603, S607 -- standalone unittest checks
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from prepare_robustness import (
    MULTIGEN_CANONICAL,
    SHORT_CANONICAL,
    transform,
    video_timing,
)
from prepare_video_multigen import select_pairs

from deepfake_detection_archive.data.manifest import SampleRecord


def pair(name: str, split: str = "train") -> list[SampleRecord]:
    real = SampleRecord(
        sample_id=f"real:{name}",
        path=f"real/{name}.mp4",
        modality="video",
        label=0,
        source="fixture",
        split=split,
    )
    return [
        real,
        replace(
            real,
            sample_id=f"fake:{name}",
            path=f"fake/{name}.mp4",
            label=1,
            parent_id=real.sample_id,
            generator="fixture-generator",
        ),
    ]


class MultiGeneratorTests(unittest.TestCase):
    def test_preserves_existing_assignments(self) -> None:
        rows = pair("train") + pair("dev", "dev")
        selected, excluded = select_pairs(rows, rows, [])
        self.assertEqual(selected, rows)
        self.assertEqual(excluded, 0)

    def test_old_holdout_and_external_exclusion_close_relations(self) -> None:
        train = pair("abcdefghijk-1")
        holdout = pair("abcdefghijk-2", "holdout")
        other = pair("other")
        external = replace(other[0], sample_id="external", path="else/other.mp4")
        clean = pair("clean", "dev")
        full = train + holdout + other + clean
        selected, excluded = select_pairs(full, train + other + clean, [external])
        self.assertEqual(selected, clean)
        self.assertEqual(excluded, 4)

    def test_bench_underscore_stem_excludes_panda_hyphen_parent(self) -> None:
        rows = pair("abcdefghijk-1")
        external = replace(
            pair("elsewhere")[0], path="Real/abcdefghijk_000000_000032.mp4"
        )
        selected, _ = select_pairs(rows, rows, [external])
        self.assertEqual(selected, [])

    def test_hash_link_blocks_entire_pair(self) -> None:
        rows = pair("train")
        rows[0] = replace(rows[0], sha256="a" * 64)
        excluded = replace(pair("external")[0], sha256="a" * 64)
        selected, _ = select_pairs(rows, rows, [excluded])
        self.assertEqual(selected, [])

    def test_expanded_derivation_groups_remain_part_of_closure(self) -> None:
        train, dev = pair("train"), pair("dev", "dev")
        expanded = [
            replace(row, derivation_group_id="known-group") for row in train + dev
        ]
        with self.assertRaisesRegex(ValueError, "incompatible"):
            select_pairs(train + dev, expanded, [])

    def test_full_and_exclusion_metadata_are_not_overwritten(self) -> None:
        rows = pair("train")
        full = [replace(rows[0], group_id="registered-external-link"), rows[1]]
        excluded = replace(pair("external")[0], group_id="registered-external-link")
        selected, _ = select_pairs(full, rows, [excluded])
        self.assertEqual(selected, [])
        # A second registration of the same external ID must also contribute.
        registered_external = pair("external")[0]
        selected, _ = select_pairs([*full, registered_external], rows, [excluded])
        self.assertEqual(selected, [])

    def test_changed_full_parent_relation_is_retained(self) -> None:
        rows = pair("train")
        held_out = pair("held-out", "holdout")
        full = [rows[0], replace(rows[1], parent_id=held_out[0].sample_id), *held_out]
        selected, _ = select_pairs(full, rows, [])
        self.assertEqual(selected, [])

    def test_fixed_test_cannot_be_reassigned_to_train(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed split"):
            select_pairs(pair("clip", "test"), pair("clip", "train"), [])

    def test_fixed_holdout_cannot_be_reassigned_to_train(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed split"):
            select_pairs(pair("clip", "holdout"), pair("clip", "train"), [])

    def test_pair_closure_and_split_conflict(self) -> None:
        rows = pair("train")
        with self.assertRaisesRegex(ValueError, "real parent"):
            select_pairs([rows[1]], [rows[1]], [])
        incompatible = [rows[0], replace(rows[1], split="dev")]
        with self.assertRaisesRegex(ValueError, "incompatible"):
            select_pairs(incompatible, incompatible, [])

    def test_two_second_canonical_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            (output / "media").mkdir(parents=True)
            source = root / "clip.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=80x40:rate=12:duration=2.25",
                    "-threads",
                    "1",
                    str(source),
                ],
                check=True,
                capture_output=True,
                timeout=20,
            )
            row = replace(pair("clip")[0], path="clip.mp4")
            result = transform(row, root, output, MULTIGEN_CANONICAL, 20)
            self.assertEqual(result.status, "available", result.failure_reason)
            self.assertEqual(result.split, "train")
            self.assertEqual(result.duration_seconds, 2)
            self.assertEqual(video_timing(output / result.path, 20), (2, 32))
            probe = json.loads(
                subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_streams",
                        "-of",
                        "json",
                        str(output / result.path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=20,
                ).stdout
            )["streams"][0]
            self.assertEqual((probe["width"], probe["height"]), (448, 256))
            self.assertEqual(probe["sample_aspect_ratio"], "1:1")
            self.assertEqual(probe["avg_frame_rate"], "16/1")

    def test_short_canonical_fits_clips_under_two_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            (output / "media").mkdir(parents=True)
            source = root / "clip.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=96x96:rate=8:duration=1.5",
                    "-threads",
                    "1",
                    str(source),
                ],
                check=True,
                capture_output=True,
                timeout=20,
            )
            row = replace(pair("clip")[0], path="clip.mp4")
            short = transform(row, root, output, SHORT_CANONICAL, 20)
            self.assertEqual(short.status, "available", short.failure_reason)
            self.assertEqual(video_timing(output / short.path, 20), (1.25, 20))
            too_short = transform(row, root, root / "other", MULTIGEN_CANONICAL, 20)
            self.assertNotEqual(too_short.status, "available")


if __name__ == "__main__":
    unittest.main()
