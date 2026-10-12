"""Behavior checks for the H.264 augmentation view; unittest discovery."""

from __future__ import annotations

# ruff: noqa: PT009, PT027, S603, S607 - standalone unittest CLI checks
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from prepare_video_augmentation import (
    ARMS,
    CLEAN,
    arm_for,
    plan,
    prepare_augmentation,
)

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.media import VideoConfig, decode_video


TRAIN_CLIPS = ("clip-a", "clip-b")
DEV_CLIP = "clip-c"


def relative_path(clip: str, label: int) -> str:
    """Mirror the real layout: a fake and its source real share one basename."""
    directory = "fake" if label else "real"
    return f"source_videos/data/{directory}/{clip}.mp4"


def record(clip: str, label: int, split: str, digest: str) -> SampleRecord:
    return SampleRecord(
        sample_id=f"{split}/{clip}-{label}",
        path=relative_path(clip, label),
        modality="video",
        label=label,
        source="fixture",
        split=split,
        license_id="CC0-1.0",
        original_id="original",
        parent_id="unknown" if label == 0 else f"{split}/{clip}-0",
        identity_id="identity",
        group_id="unknown",
        derivation_group_id=f"{split}:{clip}",
        sha256=digest,
        container="matroska",
        codec="h264",
        frame_rate=10,
        duration_seconds=0.4,
    )


def encode(path: Path, rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=80x40:rate={rate}:duration=0.4",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    return file_sha256(path)


def fixture(tmp: Path) -> tuple[Path, list[SampleRecord]]:
    """Return a media root whose train and dev clips have registered bytes."""
    root = tmp / "media"
    train_source = root / relative_path(TRAIN_CLIPS[0], 0)
    train_digest = encode(train_source, 10)
    dev_source = root / relative_path(DEV_CLIP, 0)
    dev_digest = encode(dev_source, 5)
    rows = [
        record(clip, label, "train", train_digest)
        for clip in TRAIN_CLIPS
        for label in (0, 1)
    ]
    rows.extend(record(DEV_CLIP, label, "dev", dev_digest) for label in (0, 1))
    for row in rows:
        target = root / row.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            source = dev_source if row.split == "dev" else train_source
            shutil.copy2(source, target)
    return root, rows


class ArmAssignmentTests(unittest.TestCase):
    def test_assignment_is_stable_and_spread_over_all_arms(self) -> None:
        keys = [f"clip-{index:04d}.mp4" for index in range(600)]
        first = {key: arm_for(key, 42, ARMS) for key in keys}
        self.assertEqual(
            first, {key: arm_for(key, 42, tuple(reversed(ARMS))) for key in keys}
        )
        counts = Counter(first.values())
        self.assertEqual(set(counts), set(ARMS))
        for arm in ARMS:
            self.assertGreater(counts[arm], 100)
        self.assertNotEqual(first, {key: arm_for(key, 43, ARMS) for key in keys})
        # Pin the seeded rule: a change here rewrites every derived arm.
        self.assertEqual(first["clip-0000.mp4"], "clean")
        self.assertEqual(first["clip-0001.mp4"], "h264-crf35")

    def test_plan_keeps_dev_clean_and_pairs_source_clips(self) -> None:
        rows = [
            record("pair", 0, "train", "0" * 64),
            record("pair", 1, "train", "1" * 64),
            record(DEV_CLIP, 1, "dev", "2" * 64),
        ]
        assigned = {row.sample_id: arm for arm, row in plan(rows, 42, ARMS)}
        self.assertEqual(assigned["dev/clip-c-1"], CLEAN)
        self.assertEqual(assigned["train/pair-0"], assigned["train/pair-1"])
        self.assertIn(assigned["train/pair-0"], ARMS)


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
class PrepareAugmentationTests(unittest.TestCase):
    def test_builds_one_shared_root_for_clean_and_derived_arms(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            root, rows = fixture(tmp)
            manifest = tmp / "input.jsonl"
            write_manifest(rows, manifest)
            output = tmp / "augmented"
            arms = ("clean", "h264-crf23")
            summary = prepare_augmentation(manifest, root, output, arms, seed=42)
            expected = {row.sample_id: arm for arm, row in plan(rows, 42, arms)}
            original = {row.sample_id: row for row in rows}
            derived = load_manifest(output / "manifest.jsonl")
            self.assertEqual(len(derived), len(rows))
            self.assertEqual(summary["failure_count"], 0)
            self.assertEqual(
                summary["manifest"]["sha256"], file_sha256(output / "manifest.jsonl")
            )
            self.assertEqual(summary["manifest"]["data_root"], str(output.resolve()))
            seen: Counter[str] = Counter()
            for row in derived:
                base_id, _, suffix = row.sample_id.partition("#")
                arm = expected[base_id]
                seen[arm] += 1
                if arm == CLEAN:
                    self.assertEqual(suffix, "")
                    self.assertEqual(row.path, original[base_id].path)
                    self.assertEqual(row.sha256, original[base_id].sha256)
                    self.assertTrue((output / row.path).is_file())
                    continue
                self.assertEqual(suffix, arm)
                self.assertTrue(row.path.startswith("media/"))
                self.assertNotEqual(row.sha256, original[base_id].sha256)
                self.assertEqual(row.status, "available")
                self.assertEqual(row.codec, "h264")
                frames = decode_video(
                    output / row.path, VideoConfig(n_frames=2, height=40, width=80)
                )
                self.assertEqual(frames.frames.shape, (2, 40, 80, 3))
            self.assertEqual(
                seen,
                Counter({arm: summary["per_arm"][arm]["rows"] for arm in arms}),
            )
            self.assertEqual(summary["original_modes"], {"linked": len(rows)})
            self.assertEqual(
                summary["per_arm"][CLEAN]["rows"]
                + summary["per_arm"]["h264-crf23"]["rows"],
                len(rows),
            )
            for label in (0, 1):
                self.assertEqual(
                    summary["per_arm"][CLEAN]["split_label_counts"][f"dev/{label}"], 1
                )
            for row in rows:
                if row.split == "dev":
                    self.assertEqual(expected[row.sample_id], CLEAN)

    def test_rejects_bad_arms_and_an_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            root, rows = fixture(tmp)
            manifest = tmp / "input.jsonl"
            write_manifest(rows, manifest)
            output = tmp / "augmented"
            prepare_augmentation(manifest, root, output, ("clean", "h264-crf23"), 42)
            for arms, error in (
                (("h264-crf23",), ValueError),
                (("clean", "clean"), ValueError),
                (("clean", "h264-crf99"), ValueError),
                (("clean", "h264-crf23"), FileExistsError),
            ):
                with self.assertRaises(error):
                    prepare_augmentation(manifest, root, output, arms, 42)

    def test_dev_only_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            root, rows = fixture(tmp)
            manifest = tmp / "input.jsonl"
            write_manifest([row for row in rows if row.split == "dev"], manifest)
            with self.assertRaises(ValueError):
                prepare_augmentation(
                    manifest, root, tmp / "augmented", ("clean", "h264-crf23"), 42
                )

    def test_transcode_failure_is_recorded_and_not_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            root, rows = fixture(tmp)
            manifest = tmp / "input.jsonl"
            write_manifest(rows, manifest)
            arms = ("clean", "h264-crf35")
            assigned = {row.sample_id: arm for arm, row in plan(rows, 7, arms)}
            derived = [row for row in rows if assigned[row.sample_id] != CLEAN]
            if not derived:
                self.skipTest("seed assigns no train row to the derived arm")
            (root / derived[0].path).write_bytes(b"not a video stream")
            output = tmp / "augmented"
            summary = prepare_augmentation(manifest, root, output, arms, 7)
            failures = summary["per_arm"]["h264-crf35"]["failures"]
            self.assertEqual(summary["failure_count"], len(failures))
            self.assertGreater(summary["failure_count"], 0)
            self.assertTrue(all(value for value in failures.values()))
            statuses = {row.status for row in load_manifest(output / "manifest.jsonl")}
            self.assertIn("decode_failed", statuses)
            self.assertIn("available", statuses)


if __name__ == "__main__":
    unittest.main()
