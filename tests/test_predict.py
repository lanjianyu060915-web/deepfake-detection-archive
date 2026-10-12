"""Unlabeled directory prediction uses real decoding and local model weights."""

from __future__ import annotations

import csv
import json
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest

from deepfake_detection_archive import cli
from deepfake_detection_archive.contracts import load_scores
from deepfake_detection_archive.data.manifest import load_manifest
from deepfake_detection_archive.tracking import atomic_json


@pytest.fixture
def prediction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from deepfake_detection_archive.models import (  # noqa: PLC0415
        LogMelCNN,
        LogMelConfig,
        save_checkpoint,
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(cli, "code_identity", lambda: {"code_sha256": "test"})
    root = tmp_path / "输入"
    for name in ("甲/same.wav", "乙/same.wav"):
        path = root / name
        path.parent.mkdir(parents=True)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes((1000 * np.sin(np.arange(1600) * 0.1)).astype("<i2"))
    descriptor = save_checkpoint(
        tmp_path / "weights" / "tiny.pt",
        LogMelCNN(LogMelConfig(channels=4, hidden_dim=8, n_mels=8)),
    )
    config = tmp_path / "configs" / "audio.json"
    atomic_json(
        config,
        {
            "manifest": "unused.jsonl",
            "data_root": "unused-data",
            "output_dir": "unused-output",
            "checkpoint": "../weights/tiny.pt",
            "checkpoint_sha256": descriptor["sha256"],
            "modality": "audio",
            "device": "cuda",
            "segment_samples": 1600,
        },
    )
    yield [
        "predict",
        "--config",
        str(config),
        "--input",
        str(root),
        "--output",
        str(tmp_path / "prediction"),
        "--device",
        "cpu",
    ]
    torch.set_num_threads(previous_threads)


def test_predict_unlabeled_nested_directory_from_other_cwd(
    prediction: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "other-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert cli.main(prediction) == 0
    output = tmp_path / "prediction"
    scores = load_scores(output / "scores.jsonl")
    assert {row.sample_id for row in scores} == {"甲/same.wav", "乙/same.wav"}
    assert all(row.label is None and row.status == "ok" for row in scores)
    assert all(row.label is None for row in load_manifest(output / "manifest.jsonl"))
    with (output / "scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["sample_id"]: float(row["score"]) for row in rows} == {
        row.sample_id: row.score for row in scores
    }
    metadata = json.loads((output / "scores.csv.metadata.json").read_text())
    assert metadata["score_direction"] == "fake_high"
    assert metadata["official"] is False
    report = json.loads((output / "report.json").read_text())
    assert report["coverage"] == 1
    assert report["fullset_eer"] is None
    saved = json.loads((output / "inference.json").read_text())
    assert saved["checkpoint"] == str(tmp_path / "weights" / "tiny.pt")


def test_predict_resume_preserves_scores_without_rescoring(
    prediction: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepfake_detection_archive.scoring import OfflineScorer  # noqa: PLC0415

    assert cli.main(prediction) == 0
    output = tmp_path / "prediction"
    original = (output / "scores.jsonl").read_bytes()
    csv_bytes = (output / "scores.csv").read_bytes()

    def no_rescore(*args: object) -> float:
        pytest.fail("completed resume must not score media again")

    monkeypatch.setattr(OfflineScorer, "__call__", no_rescore)
    assert cli.main([*prediction, "--resume"]) == 0
    assert (output / "scores.jsonl").read_bytes() == original
    assert (output / "scores.csv").read_bytes() == csv_bytes
    assert cli.main(prediction) == 2
    assert (output / "scores.jsonl").read_bytes() == original
    assert (output / "scores.csv").read_bytes() == csv_bytes


def test_predict_failure_keeps_accounting_without_export(
    prediction: list[str], tmp_path: Path
) -> None:
    (tmp_path / "输入" / "broken.wav").write_bytes(b"invalid media")
    output = tmp_path / "prediction"
    assert cli.main(prediction) == 2
    scores = load_scores(output / "scores.jsonl")
    failed = [row for row in scores if row.status != "ok"]
    assert len(failed) == 1
    assert failed[0].sample_id == "broken.wav"
    assert failed[0].score is None
    report = json.loads((output / "report.json").read_text())
    assert report["sample_count"] == 3
    assert report["failure_count"] == 1
    assert report["coverage"] == pytest.approx(2 / 3)
    assert not (output / "scores.csv").exists()
    assert cli.main([*prediction, "--resume"]) == 2
    assert not (output / "scores.csv").exists()


def test_predict_resume_changed_media_preserves_original_run(
    prediction: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(prediction) == 0
    output = tmp_path / "prediction"
    original = (output / "scores.jsonl").read_bytes()
    csv_bytes = (output / "scores.csv").read_bytes()
    metadata = (output / "scores.csv.metadata.json").read_bytes()
    (tmp_path / "输入" / "甲" / "same.wav").write_bytes(b"changed")
    assert cli.main([*prediction, "--resume"]) == 2
    assert "identity changed" in capsys.readouterr().err
    assert (output / "scores.jsonl").read_bytes() == original
    assert (output / "scores.csv").read_bytes() == csv_bytes
    assert (output / "scores.csv.metadata.json").read_bytes() == metadata


def test_predict_resume_early_failure_or_locked_run_preserves_export(
    prediction: list[str], tmp_path: Path
) -> None:
    from deepfake_detection_archive.submission.runner import _run_lock  # noqa: PLC0415

    assert cli.main(prediction) == 0
    output = tmp_path / "prediction"
    csv_bytes = (output / "scores.csv").read_bytes()
    metadata = (output / "scores.csv.metadata.json").read_bytes()
    with _run_lock(output / ".runner.lock"):
        assert cli.main([*prediction, "--resume"]) == 2
    for path in (tmp_path / "输入").rglob("*.wav"):
        path.unlink()
    assert cli.main([*prediction, "--resume"]) == 2
    assert (output / "scores.csv").read_bytes() == csv_bytes
    assert (output / "scores.csv.metadata.json").read_bytes() == metadata


def test_predict_interrupted_resume_only_scores_remaining_media(
    prediction: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepfake_detection_archive.data.manifest import SampleRecord  # noqa: PLC0415
    from deepfake_detection_archive.scoring import OfflineScorer  # noqa: PLC0415

    original_call = OfflineScorer.__call__
    called: list[str] = []

    def interrupted(self: OfflineScorer, record: SampleRecord) -> float:
        if called:
            raise KeyboardInterrupt
        called.append(record.sample_id)
        return original_call(self, record)

    monkeypatch.setattr(OfflineScorer, "__call__", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main(prediction)
    output = tmp_path / "prediction"
    partial = load_scores(output / "scores.jsonl")
    assert len(partial) == 1
    assert not (output / "scores.csv").exists()

    def resumed(self: OfflineScorer, record: SampleRecord) -> float:
        called.append(record.sample_id)
        return original_call(self, record)

    monkeypatch.setattr(OfflineScorer, "__call__", resumed)
    assert cli.main([*prediction, "--resume"]) == 0
    scores = load_scores(output / "scores.jsonl")
    assert len(scores) == 2
    assert scores[0] == partial[0]
    assert len(called) == len(set(called)) == 2
    assert (output / "scores.csv").exists()
