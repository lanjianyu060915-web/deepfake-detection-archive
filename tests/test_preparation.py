"""Decoded availability and the distinction between leakage and statistical units."""

import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from deepfake_detection_archive.data.manifest import SampleRecord
from deepfake_detection_archive.data.split import statistical_groups
from deepfake_detection_archive.preparation import (
    decode_manifest,
    index_media,
    media_snapshot,
)


def test_statistical_groups_require_registered_evidence() -> None:
    samples = [
        SampleRecord("a", "a.wav", "audio", 0, "s", sha256="a" * 64),
        SampleRecord("b", "b.wav", "audio", 1, "s", sha256="a" * 64),
        SampleRecord("c", "c.wav", "audio", 1, "s", original_id="auto-index-c"),
    ]
    assert statistical_groups(samples) == {
        "a": "unknown",
        "b": "unknown",
        "c": "unknown",
    }
    samples[0] = replace(samples[0], identity_id="registered-speaker")
    assert statistical_groups(samples) == {"a": "a", "b": "a", "c": "unknown"}


def test_index_never_invents_statistical_independence(tmp_path: Path) -> None:
    (tmp_path / "one.wav").write_bytes(b"index only does not decode")
    samples = index_media(tmp_path, "audio", "fixture")
    assert samples[0].label is None
    assert samples[0].group_id == "unknown"
    assert statistical_groups(samples) == {"one.wav": "unknown"}


def test_snapshot_keeps_missing_and_path_escape_without_reading_outside(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    (root / "escape.wav").symlink_to(outside)
    samples = [
        SampleRecord("missing", "missing.wav", "audio", None, "s"),
        SampleRecord("escape", "escape.wav", "audio", None, "s"),
    ]
    snapshot = media_snapshot(samples, root)
    assert [row["state"] for row in snapshot] == ["missing", "path_escape"]
    assert all(row["sha256"] is None for row in snapshot)


@pytest.mark.parametrize("status", ["missing", "decode_failed"])
def test_successful_redecode_restores_availability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    np = pytest.importorskip("numpy")
    from deepfake_detection_archive import media  # noqa: PLC0415

    path = tmp_path / "one.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 160)
    monkeypatch.setattr(
        media,
        "decode_audio",
        lambda path: media.AudioData(
            np.zeros(160, dtype=np.float32), 16000, 16000, 0.01
        ),
    )
    sample = SampleRecord(
        "one",
        "one.wav",
        "audio",
        0,
        "s",
        status=status,
        decode_status="failed",
        failure_reason="old failure",
    )
    result = decode_manifest([sample], tmp_path)[0]
    assert result.status == "available"
    assert result.decode_status == "ok"
    assert result.failure_reason == "unknown"


def test_explicit_exclusion_is_never_redecoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("numpy")
    from deepfake_detection_archive import media  # noqa: PLC0415

    def forbidden(path: Path) -> None:
        pytest.fail("excluded sample must not be touched")

    monkeypatch.setattr(media, "decode_audio", forbidden)
    sample = SampleRecord(
        "one",
        "missing.wav",
        "audio",
        0,
        "s",
        status="excluded",
        failure_reason="license disallows use",
    )
    assert decode_manifest([sample], tmp_path) == [sample]


def test_offline_scorer_checks_status_before_media_or_model(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from deepfake_detection_archive.media import MediaError  # noqa: PLC0415
    from deepfake_detection_archive.scoring import OfflineScorer  # noqa: PLC0415

    scorer = object.__new__(OfflineScorer)
    sample = SampleRecord(
        "one",
        "missing.wav",
        "audio",
        0,
        "s",
        status="excluded",
        failure_reason="license disallows use",
    )
    with pytest.raises(MediaError, match="excluded"):
        scorer(sample)


@pytest.mark.parametrize("mode", ["waveform", "features", "standalone"])
@pytest.mark.parametrize(
    "failure", ["path_escape", "unregistered_hash", "hash_mismatch"]
)
def test_media_boundaries_preserve_scoring_errors(
    tmp_path: Path, mode: str, failure: str
) -> None:
    torch = pytest.importorskip("torch")
    from deepfake_detection_archive.config import InferenceConfig  # noqa: PLC0415
    from deepfake_detection_archive.features import extract_features  # noqa: PLC0415
    from deepfake_detection_archive.media import (  # noqa: PLC0415
        AudioConfig,
        MediaError,
        VideoConfig,
    )
    from deepfake_detection_archive.models.encoders import FrozenAudioEncoder  # noqa: PLC0415
    from deepfake_detection_archive.scoring import OfflineScorer  # noqa: PLC0415

    root = tmp_path / "media"
    root.mkdir()
    path = root / "sample.wav"
    if failure == "path_escape":
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"outside")
        path.symlink_to(outside)
    else:
        path.write_bytes(b"changed media")
    record = SampleRecord(
        "sample",
        path.name,
        "audio",
        0,
        "s",
        sha256="unknown" if failure == "unregistered_hash" else "a" * 64,
    )
    # No encoder weights or model should be used for an invalid media boundary.
    encoder = object.__new__(FrozenAudioEncoder)
    if mode == "standalone":
        with pytest.raises(MediaError) as raised:
            extract_features(
                record, root, encoder, AudioConfig(), VideoConfig(), 400, 1, "cpu", 1
            )
    else:
        scorer = object.__new__(OfflineScorer)
        scorer.execution = Mock()
        scorer.execution.score.side_effect = lambda call: call(scorer.config.batch_size)
        scorer.config = InferenceConfig(
            manifest="manifest.jsonl",
            data_root="media",
            output_dir="scores",
            modality="audio",
            checkpoint="checkpoint.pt",
            checkpoint_sha256="a" * 64,
            device="cpu",
        )
        scorer.root = root
        scorer.audio = AudioConfig()
        scorer.video = VideoConfig()
        scorer.device = torch.device("cpu")
        scorer.encoder = encoder if mode == "features" else None
        scorer.frame_preprocessing = None
        with pytest.raises(MediaError) as raised:
            scorer(record)
    assert raised.value.code == failure
