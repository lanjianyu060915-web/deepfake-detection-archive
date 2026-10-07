"""Synthetic duplicate candidates do not establish speaker verification."""

import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")

from deepfake_detection_archive.data.duplicates import (  # noqa: E402
    find_near_duplicates,
    image_phash,
)
from deepfake_detection_archive.data.manifest import SampleRecord  # noqa: E402
from deepfake_detection_archive.media import MediaError  # noqa: E402


def _record(name: str, modality: str = "audio", **kwargs: object) -> SampleRecord:
    return SampleRecord(
        sample_id=name,
        path=name,
        modality=modality,
        label=0,
        source="synthetic",
        **kwargs,  # type: ignore[arg-type]
    )


def _wav(path: Path, values: object, rate: int = 16000) -> None:
    samples = (np.asarray(values) * 20000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())


@pytest.fixture
def ffmpeg_available() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/ffprobe required for synthetic duplicate tests")


def _ffmpeg(arguments: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", *arguments],
        check=True,
        capture_output=True,
        timeout=20,
    )


def test_audio_resampling_clone_and_distinct(
    tmp_path: Path, ffmpeg_available: None
) -> None:
    t = np.arange(32000) / 16000
    signal = 0.6 * np.sin(2 * np.pi * (250 * t + 350 * t**2)) * (
        0.7 + 0.3 * np.sin(2 * np.pi * 3 * t)
    ) + 0.2 * np.sin(2 * np.pi * (2000 * t + 100 * t**2))
    _wav(tmp_path / "original.wav", signal)
    _ffmpeg(
        [
            "-i",
            str(tmp_path / "original.wav"),
            "-ar",
            "22050",
            str(tmp_path / "clone.flac"),
        ]
    )
    other = 0.7 * np.sin(2 * np.pi * (2800 * t - 300 * t**2))
    _wav(tmp_path / "other.wav", other)
    records = [
        _record("original.wav", split="train"),
        _record("clone.flac", split="dev"),
        _record("other.wav"),
    ]
    audit = find_near_duplicates(records, tmp_path)
    assert [(p.left_id, p.right_id) for p in audit.candidate_pairs] == [
        ("original.wav", "clone.flac")
    ]
    assert audit.candidate_pairs[0].cross_split
    assert not audit.failures
    assert len({s.source_sha256 for s in audit.samples}) == 3
    assert all(
        s.identity_id == "unknown" and not s.identity_verified for s in audit.samples
    )
    assert not audit.grouping_applied
    assert not audit.speaker_verification_performed
    serialized = json.loads(json.dumps(audit.to_dict()))
    assert serialized["samples"][0]["identity_id"] == "unknown"
    assert serialized["min_similarity"] == 0.97
    assert records[0].group_id == "unknown"


def test_video_transcode_candidates(tmp_path: Path, ffmpeg_available: None) -> None:
    _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=8:duration=1",
            "-c:v",
            "ffv1",
            str(tmp_path / "original.mkv"),
        ]
    )
    _ffmpeg(
        [
            "-i",
            str(tmp_path / "original.mkv"),
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            str(tmp_path / "clone.mp4"),
        ]
    )
    _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "smptebars=size=96x64:rate=8:duration=1",
            "-c:v",
            "ffv1",
            str(tmp_path / "other.mkv"),
        ]
    )
    audit = find_near_duplicates(
        [_record(name, "video") for name in ("original.mkv", "clone.mp4", "other.mkv")],
        tmp_path,
        min_similarity=0.9,
    )
    assert not audit.failures
    assert [(p.left_id, p.right_id) for p in audit.candidate_pairs] == [
        ("original.mkv", "clone.mp4")
    ]
    assert all(len(sample.timestamps_seconds) == 8 for sample in audit.samples)


def test_low_information_is_recorded(tmp_path: Path, ffmpeg_available: None) -> None:
    _wav(tmp_path / "silence.wav", np.zeros(16000))
    _wav(tmp_path / "tone.wav", np.sin(2 * np.pi * 440 * np.arange(16000) / 16000))
    _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=red:size=96x32:rate=4:duration=1",
            "-c:v",
            "ffv1",
            str(tmp_path / "black.mkv"),
        ]
    )
    audit = find_near_duplicates(
        [_record("silence.wav"), _record("tone.wav"), _record("black.mkv", "video")],
        tmp_path,
    )
    assert not audit.candidate_pairs
    assert len(audit.failures) == 3
    assert all(f.code == "low_information" for f in audit.failures)
    assert all(s.status == "low_information" for s in audit.samples)


def test_failures_and_source_hash_mismatch(
    tmp_path: Path, ffmpeg_available: None
) -> None:
    (tmp_path / "broken.wav").write_bytes(b"corrupt")
    _wav(tmp_path / "mismatch.wav", np.zeros(16000))
    audit = find_near_duplicates(
        [
            _record("absent.wav"),
            _record("broken.wav"),
            _record("mismatch.wav", sha256="0" * 64),
        ],
        tmp_path,
    )
    assert [failure.code for failure in audit.failures] == [
        "missing_file",
        "decode_error",
        "hash_mismatch",
    ]
    assert len(audit.samples) == 3
    assert not audit.candidate_pairs


def test_budget_and_parameters_fail_before_decode(tmp_path: Path) -> None:
    records = [_record(str(i)) for i in range(3)]
    with pytest.raises(ValueError, match="pair budget exceeded"):
        find_near_duplicates(records, tmp_path, max_pairs=2)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        find_near_duplicates([records[0], records[0]], tmp_path)
    for threshold in (0, 1.1, float("nan")):
        with pytest.raises(ValueError, match="min_similarity"):
            find_near_duplicates([], tmp_path, min_similarity=threshold)


def test_image_phash_rejects_pure_color_and_matches_resized_texture() -> None:
    with pytest.raises(MediaError, match="low_information"):
        image_phash(np.zeros((32, 32, 3), dtype=np.uint8))
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    assert len(image_phash(image)) == 63
    enlarged = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    similarity = np.mean(np.array(image_phash(image)) == image_phash(enlarged))
    assert similarity > 0.9
