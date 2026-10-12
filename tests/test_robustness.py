"""Paired dev transformations preserve provenance and failure denominators."""

import json
import runpy
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.evaluation.report import evaluate_records
from deepfake_detection_archive.media import VideoConfig, decode_audio, decode_video
from deepfake_detection_archive.submission.runner import run_inference


@pytest.fixture
def prepare() -> Any:
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/prepare_robustness.py")
    )["prepare"]


def rows(modality: str, path: str) -> list[SampleRecord]:
    return [
        SampleRecord(
            f"label/{label}",
            path,
            modality,
            label,
            "synthetic",
            split="dev",
            license_id="CC0-1.0",
            original_id="original",
            parent_id="parent",
            identity_id="identity",
            group_id="group",
            derivation_group_id="derived",
            duration_seconds=1,
            frame_rate=10 if modality == "video" else None,
        )
        for label in (0, 1)
    ]


@pytest.fixture
def media(tmp_path: Path) -> dict[str, Path]:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required")
    result = {}
    for modality, source, extension in (
        ("audio", "sine=frequency=1000:duration=0.4", "wav"),
        ("video", "testsrc2=size=80x40:rate=10:duration=0.4", "mkv"),
    ):
        path = tmp_path / f"source.{extension}"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                source,
                "-threads",
                "1",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )
        result[modality] = path
    return result


@pytest.mark.parametrize("split", ["train", "holdout"])
def test_rejects_non_dev_before_any_media_or_output(
    prepare: Any,
    tmp_path: Path,
    split: str,
) -> None:
    manifest = tmp_path / "input.jsonl"
    write_manifest(
        [replace(row, split=split) for row in rows("audio", "absent")], manifest
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="dev"):
        prepare(manifest, tmp_path, output, ["audio_aac64k"], 60)
    assert not output.exists()


@pytest.mark.parametrize(
    ("modality", "conditions"),
    [
        ("audio", ["audio_aac64k", "audio_bandlimit8k"]),
        ("video", ["video_h264_crf23", "video_h264_crf35"]),
        ("video", ["video_h264_crf18_448x256", "video_h264_crf35_448x256"]),
    ],
)
def test_real_transformations_are_paired_and_decodable(
    prepare: Any,
    tmp_path: Path,
    media: dict[str, Path],
    modality: str,
    conditions: list[str],
) -> None:
    source = media[modality]
    original_hash = file_sha256(source)
    original = rows(modality, source.name)
    manifest = tmp_path / "input.jsonl"
    write_manifest(original, manifest)
    output = tmp_path / "output"
    prepare(manifest, tmp_path, output, conditions, 60)
    assert load_manifest(output / "baseline.jsonl") == original
    assert file_sha256(source) == original_hash
    for condition in conditions:
        derived = load_manifest(output / condition / "manifest.jsonl")
        assert len(derived) == 2
        for before, after in zip(original, derived, strict=True):
            for field in (
                "sample_id",
                "label",
                "source",
                "license_id",
                "original_id",
                "parent_id",
                "identity_id",
                "group_id",
                "derivation_group_id",
            ):
                assert getattr(before, field) == getattr(after, field)
            assert after.status == "available"
            assert after.decode_status == "unknown"
            assert after.duration_seconds is None
            target = output / condition / after.path
            assert after.sha256 == file_sha256(target)
            if modality == "audio":
                assert decode_audio(target).sample_rate == 16000
            else:
                assert after.frame_rate is None
                decoded = decode_video(
                    target, VideoConfig(n_frames=2, height=40, width=80)
                )
                assert decoded.frames.shape[0] == 2
        # Identical source bytes with opposite labels receive identical processing.
        assert derived[0].sha256 == derived[1].sha256
    summary = json.loads((output / "summary.json").read_text())
    assert summary["baseline"]["data_root"] == str(tmp_path)
    with pytest.raises(FileExistsError):
        prepare(manifest, tmp_path, output, conditions, 60)


@pytest.mark.parametrize(
    "condition",
    ["video_h264_crf18_448x256", "video_h264_crf35_448x256"],
)
def test_size_canonical_condition_shares_one_output_geometry(
    prepare: Any,
    tmp_path: Path,
    media: dict[str, Path],
    condition: str,
) -> None:
    """Both labels must land on one size, pixel format and codec in the view."""
    manifest = tmp_path / "input.jsonl"
    write_manifest(rows("video", media["video"].name), manifest)
    output = tmp_path / "output"
    prepare(manifest, tmp_path, output, [condition], 60)
    derived = load_manifest(output / condition / "manifest.jsonl")
    shapes = []
    for row in derived:
        target = output / condition / row.path
        frames = decode_video(
            target,
            VideoConfig(n_frames=2, height=256, width=448, resize_mode="center_crop"),
        )
        shapes.append(frames.frames.shape)
        probed = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,pix_fmt,codec_name",
                    "-of",
                    "json",
                    str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
        )["streams"][0]
        assert (probed["width"], probed["height"]) == (448, 256)
        assert probed["pix_fmt"] == "yuv420p"
        assert probed["codec_name"] == "h264"
    assert shapes[0] == shapes[1] == (2, 256, 448, 3)


def test_bandlimit_removes_high_frequency_without_muting_low_frequency(
    prepare: Any,
    tmp_path: Path,
    media: dict[str, Path],
) -> None:
    source = media["audio"]
    time = np.arange(16000) / 16000
    waveform = 0.2 * (np.sin(2 * np.pi * 1000 * time) + np.sin(2 * np.pi * 6000 * time))
    sf.write(source, waveform, 16000)
    manifest = tmp_path / "input.jsonl"
    write_manifest(rows("audio", source.name), manifest)
    output = tmp_path / "output"
    prepare(manifest, tmp_path, output, ["audio_bandlimit8k"], 60)
    condition = output / "audio_bandlimit8k"
    transformed = load_manifest(condition / "manifest.jsonl")[0]
    before = np.abs(np.fft.rfft(decode_audio(source).waveform))
    after = np.abs(np.fft.rfft(decode_audio(condition / transformed.path).waveform))
    assert after[1000] / before[1000] > 0.8
    assert after[6000] / before[6000] < 0.05


def test_aac_preserves_stereo_level_under_the_project_decoder(
    prepare: Any,
    tmp_path: Path,
    media: dict[str, Path],
) -> None:
    source = media["audio"]
    wave = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)
    sf.write(source, np.column_stack((wave, wave)), 16000)
    manifest = tmp_path / "input.jsonl"
    write_manifest(rows("audio", source.name), manifest)
    output = tmp_path / "output"
    prepare(manifest, tmp_path, output, ["audio_aac64k"], 60)
    condition = output / "audio_aac64k"
    transformed = load_manifest(condition / "manifest.jsonl")[0]
    before = decode_audio(source).waveform
    after = decode_audio(condition / transformed.path).waveform[: len(before)]
    assert np.linalg.norm(after) / np.linalg.norm(before) == pytest.approx(1, rel=0.05)


def test_bad_and_unavailable_rows_retain_failure_denominator(
    prepare: Any,
    tmp_path: Path,
    media: dict[str, Path],
) -> None:
    records = rows("audio", media["audio"].name)
    records[1] = replace(records[1], path="corrupt.wav", sha256="a" * 64)
    (tmp_path / "corrupt.wav").write_text("invalid media")
    records.append(
        replace(
            records[0],
            sample_id="missing",
            path="absent.wav",
            status="missing",
            failure_reason="not downloaded",
        )
    )
    manifest = tmp_path / "input.jsonl"
    write_manifest(records, manifest)
    output = tmp_path / "output"
    prepare(manifest, tmp_path, output, ["audio_aac64k"], 60)
    derived = load_manifest(output / "audio_aac64k/manifest.jsonl")
    assert len(derived) == 3
    assert derived[1].sha256 == "unknown"
    assert derived[1].status == "decode_failed"
    assert derived[1].failure_reason != "unknown"
    assert not (output / "audio_aac64k" / derived[1].path).exists()
    scores = run_inference(
        derived,
        lambda row: float(row.label or 0),
        tmp_path / "inference",
        identity={"test": "robustness"},
    )
    report = evaluate_records(scores)
    assert report.sample_count == 3
    assert report.failure_count == 2
    assert report.coverage == pytest.approx(1 / 3)
    assert report.fullset_eer is None
    summary = json.loads((output / "summary.json").read_text())
    assert summary["conditions"]["audio_aac64k"]["failure_count"] == 2


def test_corrupt_audio_is_not_repaired_into_a_successful_condition(
    prepare: Any,
    tmp_path: Path,
    media: dict[str, Path],
) -> None:
    source = tmp_path / "corrupt.flac"
    # PCM16 FLAC: 3 x 4096 samples of 0.25 at 16 kHz, no optional metadata.
    # The second frame's reserved subframe padding bit (offset 59) is set to 1.
    # Fixed bytes avoid depending on libsndfile's encoding or tail truncation.
    source.write_bytes(
        bytes.fromhex(
            "664c6143800000221000100000000b00000b03e800f000003000"
            "e7f4f8fb2ab685448cefb8078f14e75e"
            "fff8c508006f010009e574"
            "fff8c5080168810009890c"
            "fff8c50802610100093d84"
        )
    )
    repaired = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-c:a",
            "aac",
            str(tmp_path / "repaired.m4a"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert "invalid subframe padding" in repaired.stderr
    manifest = tmp_path / "input.jsonl"
    write_manifest(rows("audio", source.name), manifest)
    output = tmp_path / "output"
    summary = prepare(manifest, tmp_path, output, ["audio_aac64k"], 60)
    assert summary["conditions"]["audio_aac64k"]["failure_count"] == 2
    assert all(
        row.status == "decode_failed"
        for row in load_manifest(output / "audio_aac64k/manifest.jsonl")
    )
    assert not list((output / "audio_aac64k/media").iterdir())


def test_timeout_removes_partial_file_and_skips_unavailable_input(
    prepare: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def timeout(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "-version" in command:
            return subprocess.CompletedProcess(command, 0, stdout="ffmpeg test\n")
        calls.append(command)
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.TimeoutExpired(command, 0.01)

    monkeypatch.setattr(subprocess, "run", timeout)
    records = rows("audio", "absent.wav")
    records[1] = replace(records[1], status="missing", failure_reason="not downloaded")
    manifest = tmp_path / "input.jsonl"
    write_manifest(records, manifest)
    output = tmp_path / "output"
    prepare(manifest, tmp_path, output, ["audio_aac64k"], 0.01)
    assert len(calls) == 1
    assert not list((output / "audio_aac64k/media").iterdir())
    derived = load_manifest(output / "audio_aac64k/manifest.jsonl")
    assert all(row.status == "decode_failed" for row in derived)
    assert "timed out" in derived[0].failure_reason
    assert derived[1].failure_reason == "not downloaded"


def test_temporal_canonical_rejects_short_input_and_standardizes_long_input(
    prepare: Any, tmp_path: Path, media: dict[str, Path]
) -> None:
    long = tmp_path / "long.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=80x40:rate=24:duration=5",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=5",
            "-threads",
            "1",
            str(long),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    original = rows("video", long.name)
    original.append(
        replace(
            original[0],
            sample_id="short",
            path=media["video"].name,
            duration_seconds=10,
        )
    )
    manifest = tmp_path / "input.jsonl"
    write_manifest(original, manifest)
    condition = "video_h264_crf23_448x256_16fps_3p5s"
    output = tmp_path / "output"
    summary = prepare(manifest, tmp_path, output, [condition], 60)
    derived = load_manifest(output / condition / "manifest.jsonl")
    assert len(derived) == 3
    assert derived[2].status == "decode_failed"
    assert "shorter than 3.5 seconds" in derived[2].failure_reason
    assert not (output / condition / derived[2].path).exists()
    assert summary["conditions"][condition]["failure_count"] == 1
    for row in derived[:2]:
        assert row.status == "available"
        assert row.duration_seconds == 3.5
        assert row.frame_rate == 16
        probe = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-of",
                    "json",
                    str(output / condition / row.path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
        )["streams"]
        assert len(probe) == 1
        assert probe[0]["codec_name"] == "h264"
        assert (probe[0]["width"], probe[0]["height"]) == (448, 256)
        assert probe[0]["avg_frame_rate"] == "16/1"
        assert int(probe[0]["nb_frames"]) == 56
        assert float(probe[0]["duration"]) == 3.5
    assert derived[0].sha256 == derived[1].sha256
