"""Synthetic media validate decoding semantics, not model performance."""

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")

from deepfake_detection_archive.media import (  # noqa: E402
    AudioConfig,
    MediaError,
    VideoConfig,
    _frame_timestamps,
    _run,
    decode_audio,
    decode_video,
    sample_audio_segments,
)


@pytest.fixture
def ffmpeg_available() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for synthetic media tests")


@pytest.fixture
def audio_file(tmp_path: Path, ffmpeg_available: None) -> Path:
    path = tmp_path / "quiet stereo.wav"
    samples = np.full((8000, 2), 3277, dtype="<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(samples.tobytes())
    return path


@pytest.fixture
def video_file(tmp_path: Path, ffmpeg_available: None) -> Path:
    path = tmp_path / "vfr.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=80x40:r=10:d=0.3",
            "-vf",
            "setpts=if(eq(N\\,2)\\,9\\,N)/(10*TB)",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "ffv1",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    return path


def test_audio_resampling_retains_level(audio_file: Path) -> None:
    decoded = decode_audio(audio_file)
    assert decoded.waveform.dtype == np.float32
    assert decoded.waveform.shape == (16000,)
    assert decoded.original_sample_rate == 8000
    assert decoded.sample_rate == 16000
    assert decoded.original_duration_seconds == pytest.approx(1)
    assert float(np.median(decoded.waveform)) == pytest.approx(0.1, abs=0.001)
    np.testing.assert_array_equal(decoded.waveform, decode_audio(audio_file).waveform)


def test_audio_limits(audio_file: Path) -> None:
    with pytest.raises(MediaError, match="duration_limit"):
        decode_audio(audio_file, AudioConfig(max_duration_seconds=0.5))
    with pytest.raises(MediaError, match="output_limit"):
        decode_audio(audio_file, AudioConfig(max_samples=100))


@pytest.mark.parametrize(
    ("kind", "code"),
    [
        ("empty", "no_samples"),
        ("corrupt", "decode_error"),
        ("valid", "invalid_metadata"),
    ],
)
def test_flac_missing_duration_preserves_sample_failures(
    tmp_path: Path, ffmpeg_available: None, kind: str, code: str
) -> None:
    path = tmp_path / "missing-duration.flac"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=r=44100:d=0.3:seed=42",
            "-t",
            "0" if kind == "empty" else "0.3",
            "-c:a",
            "flac",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    encoded = bytearray(path.read_bytes())
    assert encoded[:8] == b"fLaC\x00\x00\x00\x22"
    # FLAC STREAMINFO permits an unknown total sample count (36 zero bits).
    encoded[21] &= 0xF0
    encoded[22:26] = bytes(4)
    if kind == "corrupt":
        encoded = encoded[: len(encoded) // 2]
    path.write_bytes(encoded)
    with pytest.raises(MediaError, match=code):
        decode_audio(path)


@pytest.mark.parametrize("suffix", ["wav", "flac"])
def test_soundfile_matches_ffmpeg(
    tmp_path: Path, ffmpeg_available: None, suffix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sf = pytest.importorskip("soundfile")
    path = tmp_path / f"native.{suffix}"
    samples = np.array([-32768, -16384, -1, 0, 1, 16384, 32767] * 100, dtype=np.int16)
    sf.write(path, samples, 16000, subtype="PCM_16")
    reference = decode_audio(path)

    def no_subprocess(*args: object) -> bytes:
        pytest.fail("native-rate soundfile decoding must not start subprocesses")

    monkeypatch.setattr("deepfake_detection_archive.media._run", no_subprocess)
    result = decode_audio(path, AudioConfig(backend="soundfile"))
    np.testing.assert_array_equal(result.waveform, reference.waveform)
    assert result.sample_rate == result.original_sample_rate == 16000
    assert result.original_duration_seconds == pytest.approx(samples.size / 16000)
    with pytest.raises(MediaError, match="duration_limit"):
        decode_audio(path, AudioConfig(backend="soundfile", max_duration_seconds=0.01))
    with pytest.raises(MediaError, match="output_limit"):
        decode_audio(path, AudioConfig(backend="soundfile", max_samples=100))


@pytest.mark.parametrize(("rate", "channels"), [(8000, 1), (16000, 2)])
def test_soundfile_requires_native_mono(
    tmp_path: Path, rate: int, channels: int
) -> None:
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "unsupported.wav"
    sf.write(path, np.zeros((400, channels), dtype=np.int16), rate)
    with pytest.raises(MediaError, match="unsupported_audio_format"):
        decode_audio(path, AudioConfig(backend="soundfile"))


def test_soundfile_rejects_nonfinite_and_corrupt_audio(tmp_path: Path) -> None:
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "nonfinite.wav"
    sf.write(
        path, np.array([0, np.nan, np.inf], dtype=np.float32), 16000, subtype="FLOAT"
    )
    with pytest.raises(MediaError, match="nonfinite_audio"):
        decode_audio(path, AudioConfig(backend="soundfile"))
    path.write_bytes(b"not an audio stream")
    with pytest.raises(MediaError, match="decode_error"):
        decode_audio(path, AudioConfig(backend="soundfile"))
    flac = tmp_path / "truncated.flac"
    samples = np.random.default_rng(42).integers(-32768, 32768, 10000, dtype=np.int16)
    sf.write(flac, samples, 16000, subtype="PCM_16")
    encoded = flac.read_bytes()
    flac.write_bytes(encoded[: len(encoded) // 2])
    with pytest.raises(MediaError, match="decode_error"):
        decode_audio(flac, AudioConfig(backend="soundfile"))


def test_video_rgb_padding_and_vfr_timestamps(video_file: Path) -> None:
    decoded = decode_video(video_file, VideoConfig(n_frames=5, height=40, width=40))
    assert decoded.frames.shape == (5, 40, 40, 3)
    assert decoded.frames.dtype == np.uint8
    assert decoded.original_width == 80
    assert decoded.original_height == 40
    assert decoded.frame_indices == (0, 1, 1, 2, 2)
    assert decoded.timestamps_seconds == pytest.approx((0, 0.1, 0.1, 0.9, 0.9))
    assert decoded.requested_timestamps_seconds == pytest.approx(
        (0, 0.225, 0.45, 0.675, 0.9)
    )
    assert decoded.frames[:, :9].max() == 0
    assert decoded.frames[:, 31:].max() == 0
    assert decoded.frames[:, 20, 20, 0].min() > 240
    assert decoded.frames[:, 20, 20, 1:].max() < 10
    np.testing.assert_array_equal(decoded.frames[1], decoded.frames[2])


def test_video_crop_and_dimension_guard(video_file: Path) -> None:
    decoded = decode_video(
        video_file,
        VideoConfig(n_frames=1, width=40, height=40, resize_mode="center_crop"),
    )
    assert decoded.frames[..., 0].min() > 240
    with pytest.raises(MediaError, match="dimension_limit"):
        decode_video(video_file, VideoConfig(max_source_pixels=100))
    with pytest.raises(MediaError, match="frame_limit"):
        decode_video(video_file, VideoConfig(max_source_frames=2))


def test_corrupt_missing_and_missing_stream(tmp_path: Path, audio_file: Path) -> None:
    broken = tmp_path / "corrupt.mp4"
    broken.write_bytes(b"not a media stream")
    for decoder in (decode_audio, decode_video):
        with pytest.raises(MediaError, match="decode_error"):
            decoder(broken)
        with pytest.raises(MediaError, match="missing_file"):
            decoder(tmp_path / "absent")
    with pytest.raises(MediaError, match="missing_stream"):
        decode_video(audio_file)


def test_subprocess_timeout_and_output_guard() -> None:
    with pytest.raises(MediaError, match="timeout"):
        _run([sys.executable, "-c", "import time; time.sleep(10)"], 0.05, 100)
    with pytest.raises(MediaError, match="output_limit"):
        _run([sys.executable, "-c", "print('x' * 10000)"], 3, 100)


def test_audio_segments_uniform_and_loop_pad() -> None:
    waveform = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(
        sample_audio_segments(waveform, 4, 3),
        [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]],
    )
    np.testing.assert_array_equal(
        sample_audio_segments(np.array([0.1, -0.2], dtype=np.float32), 5, 2),
        np.array([[0.1, -0.2, 0.1, -0.2, 0.1]] * 2, dtype=np.float32),
    )
    with pytest.raises(MediaError, match="invalid_audio"):
        sample_audio_segments(np.array([], dtype=np.float32), 4, 1)
    with pytest.raises(MediaError, match="invalid_audio"):
        sample_audio_segments(np.array([np.nan], dtype=np.float32), 4, 1)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_configuration(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        AudioConfig(timeout_seconds=value)
    with pytest.raises(ValueError, match="finite and positive"):
        VideoConfig(max_duration_seconds=value)


def test_output_configuration_guard() -> None:
    with pytest.raises(ValueError, match="max_output_bytes"):
        VideoConfig(n_frames=16, width=4096, height=4096)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b'{"frames": []}', "no_frames"),
        (b'{"frames": [{}]}', "invalid_timestamps"),
        (b'{"frames": [{"best_effort_timestamp_time": "nan"}]}', "invalid_timestamps"),
        (
            b'{"frames": [{"best_effort_timestamp_time": "1"},'
            b'{"best_effort_timestamp_time": "0"}]}',
            "invalid_timestamps",
        ),
    ],
)
def test_invalid_frame_metadata(
    monkeypatch: pytest.MonkeyPatch, raw: bytes, code: str
) -> None:
    monkeypatch.setattr("deepfake_detection_archive.media._run", lambda *args: raw)
    with pytest.raises(MediaError, match=code):
        _frame_timestamps("unused", VideoConfig())
