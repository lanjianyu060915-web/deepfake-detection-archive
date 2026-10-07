"""Synthetic local experiments prove execution contracts, not detection quality."""

import json
import shutil
import socket
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.experiments import (
    _save_tensor,
    load_experiment_config,
    run_experiment,
)


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required")
    monkeypatch.setattr(
        "deepfake_detection_archive.experiments.code_identity",
        lambda: {"code_sha256": "a" * 64},
    )
    media = tmp_path / "media"
    media.mkdir()
    records = []
    for index, split in enumerate(
        ("train", "train", "dev", "dev", "holdout", "holdout")
    ):
        path = media / f"{index}.wav"
        digest = f"{index:064x}"
        # Holdout has deliberately absent files: training may inspect its split
        # metadata, but cannot read, decode, cache or score those media.
        if split != "holdout":
            samples = (5000 * np.sin(np.arange(1600) * (index + 1) / 13)).astype("<i2")
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(samples.tobytes())
            digest = file_sha256(path)
        records.append(
            SampleRecord(
                sample_id=str(index),
                path=path.name,
                modality="audio",
                label=index % 2,
                source="synthetic",
                license_id="self-generated",
                sha256=digest,
                split=split,
                decode_status="ok",
                group_id=f"group-{index}",
            )
        )
    write_manifest(records, tmp_path / "manifest.jsonl")
    source = {
        "source": "synthetic",
        "url": "local:generated-test-fixture",
        "version": "1",
        "license_id": "self-generated",
        "license_url": "local:test-fixture",
        "downloaded_at": "2026-10-07",
        "allowed_uses": ["training"],
        "license_status": "approved",
    }
    (tmp_path / "sources.jsonl").write_text(json.dumps(source) + "\n")
    config = {
        "manifest": "manifest.jsonl",
        "data_root": "media",
        "sources": "sources.jsonl",
        "output_dir": "run",
        "modality": "audio",
        "architecture": "logmel_cnn_v1",
        "model_config": {
            "n_fft": 64,
            "hop_length": 32,
            "n_mels": 8,
            "channels": 2,
            "hidden_dim": 4,
        },
        "training": {"max_epochs": 1, "device": "cpu", "seed": 23},
        "batch_size": 2,
        "segment_samples": 400,
        "feature_cache": "cache",
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config))
    return path


def _change(path: Path, **changes: Any) -> None:
    config = json.loads(path.read_text())
    config.update(changes)
    path.write_text(json.dumps(config))


def test_waveform_experiment_caches_only_train_dev_and_records_provenance(
    experiment: Path,
) -> None:
    result = run_experiment(experiment)
    report = json.loads((experiment.parent / "run" / "experiment.json").read_text())
    assert result.best_checkpoint.is_file()
    assert report["status"] == "complete"
    assert report["train_count"] == report["dev_count"] == 2
    assert report["holdout_accessed"] is False
    assert len(list((experiment.parent / "cache").glob("*.pt"))) == 4
    assert len(report["provenance"]["weights_sha256"]) == 64
    assert (
        json.loads((experiment.parent / "run" / "decode_failures.json").read_text())
        == []
    )


def test_spawned_loader_worker_can_read_prepared_tensors(experiment: Path) -> None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except PermissionError:
        pytest.skip("sandbox blocks the Unix socket IPC needed by PyTorch workers")
    _change(experiment, num_workers=1)
    result = run_experiment(experiment)
    assert result.epochs_completed == 1


@pytest.fixture
def feature_experiment(experiment: Path) -> Path:
    transformers = pytest.importorskip("transformers")
    bundle = experiment.parent / "encoder"
    config = transformers.Wav2Vec2Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        conv_dim=(8, 8),
        conv_stride=(5, 2),
        conv_kernel=(10, 3),
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2,
        feat_extract_norm="layer",
        mask_time_prob=0.0,
    )
    torch.manual_seed(9)
    transformers.Wav2Vec2Model(config).save_pretrained(bundle, safe_serialization=True)
    transformers.Wav2Vec2FeatureExtractor().save_pretrained(bundle)
    hashes = {
        path.name: file_sha256(path) for path in bundle.iterdir() if path.is_file()
    }
    _change(
        experiment,
        architecture="media_feature_head_v1",
        segments=2,
        model_config={"input_dim": 8, "hidden_dim": 4, "dropout": 0},
        encoder={"kind": "audio", "model_dir": "encoder", "expected_hashes": hashes},
    )
    return experiment


def test_tiny_local_encoder_experiment_reuses_verified_cache(
    feature_experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = run_experiment(feature_experiment)
    config = json.loads(feature_experiment.read_text())
    cached = sorted((feature_experiment.parent / "cache").glob("*.pt"))
    assert torch.load(cached[0], weights_only=True).shape == (2, 8)

    def unexpected_extract(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("verified cached features should be reused")

    monkeypatch.setattr(
        "deepfake_detection_archive.experiments.extract_features", unexpected_extract
    )
    _change(feature_experiment, output_dir="repeat")
    second = run_experiment(feature_experiment)
    assert first.best_dev_eer == second.best_dev_eer
    # A changed preprocessing choice has a different content cache identity.
    _change(feature_experiment, output_dir="changed", segments=3)
    with pytest.raises(ValueError, match="sample preparation failed"):
        run_experiment(feature_experiment)
    failures = json.loads(
        (feature_experiment.parent / "changed" / "decode_failures.json").read_text()
    )
    assert len(failures) == 4
    assert config["encoder"]["expected_hashes"]


def test_corrupt_cache_is_reported_and_training_does_not_start(
    experiment: Path,
) -> None:
    run_experiment(experiment)
    cached = next((experiment.parent / "cache").glob("*.pt"))
    cached.write_bytes(b"corrupted")
    _change(experiment, output_dir="corrupt")
    with pytest.raises(ValueError, match="sample preparation failed"):
        run_experiment(experiment)
    output = experiment.parent / "corrupt"
    failures = json.loads((output / "decode_failures.json").read_text())
    assert len(failures) == 1
    assert "SHA-256 mismatch" in failures[0]["error"]
    assert not (output / "training").exists()


def test_cross_split_relation_fails_before_preparation(experiment: Path) -> None:
    path = experiment.parent / "manifest.jsonl"
    records = load_manifest(path)
    records[2] = replace(records[2], group_id=records[0].group_id)
    write_manifest(records, path)
    with pytest.raises(ValueError, match="data audit failed"):
        run_experiment(experiment)
    assert not (experiment.parent / "cache").exists()
    report = json.loads((experiment.parent / "run" / "audit.json").read_text())
    assert any("leakage" in error for error in report["split_errors"])


def test_bad_media_is_not_dropped_even_with_registered_bytes(experiment: Path) -> None:
    media = experiment.parent / "media" / "0.wav"
    media.write_bytes(b"not a wave")
    path = experiment.parent / "manifest.jsonl"
    records = load_manifest(path)
    records[0] = replace(records[0], sha256=file_sha256(media))
    write_manifest(records, path)
    with pytest.raises(ValueError, match="sample preparation failed"):
        run_experiment(experiment)
    failures = json.loads(
        (experiment.parent / "run" / "decode_failures.json").read_text()
    )
    assert len(failures) == 1
    assert failures[0]["sample_id"] == "0"
    assert not (experiment.parent / "run" / "training").exists()


def test_cache_capacity_is_enforced_without_partial_tensor_files(
    experiment: Path,
) -> None:
    _change(experiment, cache_max_bytes=1)
    with pytest.raises(ValueError, match="sample preparation failed"):
        run_experiment(experiment)
    assert not list((experiment.parent / "cache").iterdir())


def test_cache_budget_counts_exact_unicode_sidecar_bytes(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    tensor = torch.arange(4, dtype=torch.float32)
    identity = {"sample_id": "样本音频一"}
    _save_tensor(reference / "sample.pt", tensor, identity, 1_000_000)
    sidecar = reference / "sample.json"
    tensor_bytes = (reference / "sample.pt").stat().st_size
    assert len(sidecar.read_bytes()) > len(sidecar.read_text(encoding="utf-8"))
    total = tensor_bytes + sidecar.stat().st_size
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError, match="cache_max_bytes"):
        _save_tensor(target / "sample.pt", tensor, identity, total - 1)
    assert list(target.iterdir()) == []
    _save_tensor(target / "sample.pt", tensor, identity, total)
    assert sum(path.stat().st_size for path in target.iterdir()) == total


def test_cache_replacement_subtracts_old_pair_but_preserves_other_entries(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.pt"
    tensor = torch.arange(4, dtype=torch.float32)
    _save_tensor(target, tensor, {"sample_id": "旧样本" * 20}, 1_000_000)
    (tmp_path / "unrelated.bin").write_bytes(b"unrelated")
    original_bytes = sum(path.stat().st_size for path in tmp_path.iterdir())
    _save_tensor(target, tensor + 1, {"sample_id": "new"}, original_bytes)
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) < original_bytes
    torch.testing.assert_close(torch.load(target, weights_only=True), tensor + 1)
    assert (tmp_path / "unrelated.bin").read_bytes() == b"unrelated"
    assert len(list(tmp_path.iterdir())) == 3


def test_rejected_cache_replacement_preserves_original_pair(tmp_path: Path) -> None:
    target = tmp_path / "sample.pt"
    tensor = torch.arange(4, dtype=torch.float32)
    _save_tensor(target, tensor, {"sample_id": "old"}, 1_000_000)
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    budget = sum(len(value) for value in original.values())
    with pytest.raises(ValueError, match="cache_max_bytes"):
        _save_tensor(target, tensor + 1, {"sample_id": "新样本" * 30}, budget)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


def test_orphan_sidecar_is_subtracted_from_cache_budget(tmp_path: Path) -> None:
    target = tmp_path / "sample.pt"
    tensor = torch.arange(4, dtype=torch.float32)
    _save_tensor(target, tensor, {"sample_id": "x"}, 1_000_000)
    exact_budget = sum(path.stat().st_size for path in tmp_path.iterdir())
    target.unlink()
    _save_tensor(target, tensor, {"sample_id": "x"}, exact_budget)
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) == exact_budget


@pytest.mark.parametrize(
    "change",
    [
        {"segments": 2},
        {"batch_size": True},
        {"num_workers": -1},
        {"unexpected": 1},
        {"cache_max_bytes": 21 * 1024**3},
        {"audio": {"sample_rate": 8000}},
        {"training": {"max_epochs": 0}},
    ],
)
def test_invalid_experiment_config(experiment: Path, change: dict[str, Any]) -> None:
    _change(experiment, **change)
    with pytest.raises(ValueError, match=r"requires|must|invalid|cannot|agree"):
        load_experiment_config(experiment)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"manifest":"a","manifest":"b"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_experiment_config(path)
