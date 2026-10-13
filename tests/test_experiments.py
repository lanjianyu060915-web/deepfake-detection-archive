"""Synthetic local experiments prove execution contracts, not detection quality."""

import json
import shutil
import socket
import wave
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.contracts import load_scores
from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.experiments import (
    ExperimentConfig,
    _balanced_sampler,
    _prepare_cache,
    _save_tensor,
    _write_dev_scores,
    load_experiment_config,
    run_experiment,
)
from deepfake_detection_archive.scoring import OfflineScorer
from deepfake_detection_archive.training import SelectionOptions


def test_dev_scores_resolve_group_evidence_before_domain_reports(
    tmp_path: Path,
) -> None:
    records = [
        SampleRecord(
            "p0", "p0.wav", "audio", 0, "prompt", group_id="aime-prompt:music"
        ),
        SampleRecord(
            "p1",
            "p1.wav",
            "audio",
            1,
            "prompt",
            group_id="aime-prompt:music",
            identity_id="mixed-speaker",
        ),
        SampleRecord(
            "k0",
            "k0.wav",
            "audio",
            0,
            "registered",
            group_id="known-uploader",
            sha256="b" * 64,
        ),
        SampleRecord("k1", "k1.wav", "audio", 1, "registered", sha256="b" * 64),
    ]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor([0.1, 0.9, 0.2, 0.8]), torch.tensor([0, 1, 0, 1])
        ),
        batch_size=2,
    )
    _write_dev_scores(torch.nn.Identity(), loader, records, "cpu", tmp_path)
    scores = load_scores(tmp_path / "dev-scores.jsonl")
    assert {row.sample_id: row.group_id for row in scores} == {
        "p0": "unknown",
        "p1": "unknown",
        "k0": "k0",
        "k1": "k0",
    }
    assert records[0].group_id == "aime-prompt:music"
    report = json.loads((tmp_path / "dev-evaluation.json").read_text())
    assert report["evaluation"]["group_count"] == 1
    assert report["evaluation"]["missing_group_count"] == 2
    assert report["evaluation"]["fullset_eer"] == 0
    assert report["by_content_type"]["unknown"]["group_count"] == 1


def test_video_frame_cache_reuses_tuple_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(
        manifest="manifest.jsonl",
        data_root=".",
        sources="sources.jsonl",
        output_dir="run",
        modality="video",
        architecture="video_frame_finetune_v1",
        model_config={},
        training={},
    )
    record = SampleRecord(
        sample_id="video",
        path="video.mp4",
        modality="video",
        label=0,
        source="synthetic",
        license_id="self-generated",
        sha256="a" * 64,
        split="train",
    )
    preprocessing = {
        "input_size": (3, 4, 4),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.25, 0.25, 0.25),
    }
    monkeypatch.setattr(
        "deepfake_detection_archive.experiments.extract_video_frames",
        lambda *args: torch.ones(2, 3, 4, 4),
    )
    cache = tmp_path / "cache"
    first = _prepare_cache(
        [record],
        config,
        tmp_path,
        cache,
        None,
        tmp_path,
        frame_preprocessing=preprocessing,
    )
    cached = {path: path.stat().st_mtime_ns for path in cache.iterdir()}

    def unexpected_decode(*args: Any) -> torch.Tensor:
        raise AssertionError("existing frame tensors must be reused")

    monkeypatch.setattr(
        "deepfake_detection_archive.experiments.extract_video_frames",
        unexpected_decode,
    )
    second = _prepare_cache(
        [record],
        config,
        tmp_path,
        cache,
        None,
        tmp_path,
        frame_preprocessing=preprocessing,
    )
    assert second == first
    assert {path: path.stat().st_mtime_ns for path in cache.iterdir()} == cached
    changed = {**preprocessing, "mean": (0.4, 0.5, 0.5)}
    with pytest.raises(ValueError, match="sample preparation failed"):
        _prepare_cache(
            [record],
            config,
            tmp_path,
            cache,
            None,
            tmp_path,
            frame_preprocessing=changed,
        )
    failures = json.loads((tmp_path / "decode_failures.json").read_text())
    assert failures[0]["error_type"] == "AssertionError"
    assert failures[0]["error"] == "existing frame tensors must be reused"


def test_balanced_sampling_corrects_class_imbalance_and_repeats_by_seed() -> None:
    entries = [(Path(str(index)), "unused", int(index >= 100)) for index in range(1000)]
    first = list(_balanced_sampler(entries, torch.Generator().manual_seed(42)))
    second = list(_balanced_sampler(entries, torch.Generator().manual_seed(42)))
    assert first == second
    assert len(first) == len(entries)
    fake_count = sum(entries[index][2] for index in first)
    assert 450 < fake_count < 550


def test_source_balancing_preserves_rare_music_and_both_classes() -> None:
    buckets = [("urban", 0)] * 100 + [("urban", 1)] * 800
    buckets += [("music", 0)] * 10 + [("music", 1)] * 90
    entries = [(Path(str(i)), "unused", label) for i, (_, label) in enumerate(buckets)]
    sources = [source for source, _ in buckets]
    first = list(_balanced_sampler(entries, torch.Generator().manual_seed(42), sources))
    second = list(
        _balanced_sampler(entries, torch.Generator().manual_seed(42), sources)
    )
    assert first == second
    counts = Counter(buckets[index] for index in first)
    assert len(first) == len(entries)
    assert all(200 < count < 300 for count in counts.values())
    assert len(counts) == 4


def test_content_type_weights_scale_domains_and_keep_classes_balanced() -> None:
    buckets = [("speech", 0)] * 400 + [("speech", 1)] * 400
    buckets += [("environment", 0)] * 200 + [("environment", 1)] * 200
    buckets += [("music", 0)] * 20 + [("music", 1)] * 20
    entries = [(Path(str(i)), "unused", label) for i, (_, label) in enumerate(buckets)]
    domains = [domain for domain, _ in buckets]
    weights = {"speech": 0.5, "environment": 0.3, "music": 0.2}
    draws = list(
        _balanced_sampler(
            entries,
            torch.Generator().manual_seed(42),
            domains,
            content_types=domains,
            content_type_weights=weights,
        )
    )
    assert len(draws) == len(entries)
    domain_counts = Counter(domains[index] for index in draws)
    expected = {
        "speech": 0.5 * len(entries),
        "environment": 0.3 * len(entries),
        "music": 0.2 * len(entries),
    }
    for domain, share in expected.items():
        assert abs(domain_counts[domain] - share) < 70
    class_counts = Counter(buckets[index] for index in draws)
    assert len(class_counts) == 6
    for (domain, _), count in class_counts.items():
        # Each domain holds two groups, so each class keeps half its share
        # and the normalized domain mass equals the configured weight.
        assert abs(count - expected[domain] / 2) < 70


def test_content_type_weights_reject_absent_or_unlabeled_entries() -> None:
    entries = [(Path(str(index)), "unused", index % 2) for index in range(8)]
    with pytest.raises(ValueError, match="one content type per entry"):
        _balanced_sampler(
            entries,
            torch.Generator().manual_seed(42),
            content_type_weights={"music": 1.0},
        )
    with pytest.raises(ValueError, match="labeled content types"):
        _balanced_sampler(
            entries,
            torch.Generator().manual_seed(42),
            content_types=[""] * len(entries),
            content_type_weights={"music": 1.0},
        )
    with pytest.raises(ValueError, match="absent content types"):
        _balanced_sampler(
            entries,
            torch.Generator().manual_seed(42),
            content_types=["music"] * len(entries),
            content_type_weights={"speech": 1.0},
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


@pytest.mark.parametrize("mask_width", [0, 16])
@pytest.mark.parametrize(
    "waveform_preparation", ["legacy", "center_pad_64000_peak095_v1"]
)
def test_ast_cached_training_and_offline_checkpoint_roundtrip(
    experiment: Path,
    monkeypatch: pytest.MonkeyPatch,
    mask_width: int,
    waveform_preparation: str,
) -> None:
    from transformers import ASTModel  # noqa: PLC0415 -- optional model dependency

    from deepfake_detection_archive import training  # noqa: PLC0415

    fills: list[float] = []
    original_mask = training._mask_fbank

    def observe_mask(
        inputs: torch.Tensor, time_width: int, frequency_width: int, fill: float
    ) -> torch.Tensor:
        fills.append(fill)
        return original_mask(inputs, time_width, frequency_width, fill)

    monkeypatch.setattr(training, "_mask_fbank", observe_mask)

    def no_pretrained_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("checkpoint restore must be self-contained")

    monkeypatch.setattr(ASTModel, "from_pretrained", no_pretrained_download)
    if waveform_preparation != "legacy":
        records = load_manifest(experiment.parent / "manifest.jsonl")
        for index in (0, 2):
            path = experiment.parent / "media" / records[index].path
            samples = (5000 * np.sin(np.arange(80000) * (index + 1) / 13)).astype("<i2")
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(samples.tobytes())
            records[index] = replace(records[index], sha256=file_sha256(path))
        write_manifest(records, experiment.parent / "manifest.jsonl")
    _change(
        experiment,
        architecture="audio_ast_v1",
        segment_samples=400 if waveform_preparation == "legacy" else 64000,
        training={
            "max_epochs": 1,
            "seed": 23,
            "time_mask_width": mask_width,
            "frequency_mask_width": mask_width,
        },
        model_config={
            "backbone_config": {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 24,
                "max_length": 32,
                "num_mel_bins": 32,
            },
            "trainable_layers": 1,
            "waveform_preparation": waveform_preparation,
        },
    )
    result = run_experiment(experiment)
    root = experiment.parent
    scorer = OfflineScorer(
        InferenceConfig(
            manifest="manifest.jsonl",
            data_root="media",
            output_dir="inference",
            modality="audio",
            device="cpu",
            checkpoint=str(result.best_checkpoint),
            checkpoint_sha256=result.checkpoint_sha256,
            segment_samples=400 if waveform_preparation == "legacy" else 64000,
            audio={"sample_rate": 16000, "backend": "soundfile"},
        ),
        root,
    )
    records = {r.sample_id: r for r in load_manifest(root / "manifest.jsonl")}
    scores = load_scores(root / "run" / "dev-scores.jsonl")
    assert len(scores) == 2
    for score in scores:
        assert scorer(records[score.sample_id]) == pytest.approx(score.score, abs=1e-6)
    # One training batch is masked; checkpoint-selection dev uses clean inputs.
    assert len(fills) == 1
    if mask_width:
        assert fills[0] == pytest.approx(4.2677393 / (2 * 4.5689974))
    history = json.loads(result.history_path.read_text())
    expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([score.score for score in scores]),
        torch.tensor([float(records[score.sample_id].label) for score in scores]),
    )
    assert history[0]["dev_loss"] == pytest.approx(float(expected_loss), abs=1e-6)
    report = json.loads((root / "run" / "experiment.json").read_text())
    assert report["train_count"] == report["dev_count"] == 2
    assert not report["holdout_accessed"]


@pytest.mark.parametrize(
    ("modality", "backend"), [("audio", "ffmpeg"), ("video", "soundfile")]
)
@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_scorer_checks_media_executables_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    backend: str,
    missing: str,
) -> None:
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.shutil.which",
        lambda name: None if name == missing else f"/bin/{name}",
    )
    config = InferenceConfig(
        manifest="manifest.jsonl",
        data_root=".",
        output_dir="out",
        modality=modality,
        model_kind="video_frames" if modality == "video" else "audio_waveform",
        checkpoint="does-not-exist.pt",
        checkpoint_sha256="0" * 64,
        device="cpu",
        audio={"backend": backend},
    )
    with pytest.raises(
        RuntimeError, match=f"required media executables missing: {missing}"
    ):
        OfflineScorer(config, tmp_path)


def test_soundfile_scorer_does_not_require_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.shutil.which", lambda name: None
    )
    config = InferenceConfig(
        manifest="manifest.jsonl",
        data_root=".",
        output_dir="out",
        modality="audio",
        checkpoint="does-not-exist.pt",
        checkpoint_sha256="0" * 64,
        device="cpu",
        audio={"backend": "soundfile"},
    )
    with pytest.raises(FileNotFoundError):
        OfflineScorer(config, tmp_path)


def test_waveform_experiment_caches_only_train_dev_and_records_provenance(
    experiment: Path,
) -> None:
    result = run_experiment(experiment)
    report = json.loads((experiment.parent / "run" / "experiment.json").read_text())
    assert result.best_checkpoint.is_file()
    assert report["status"] == "complete"
    assert report["train_count"] == report["dev_count"] == 2
    assert report["holdout_accessed"] is False
    scores = load_scores(experiment.parent / "run" / "dev-scores.jsonl")
    assert [score.sample_id for score in scores] == ["2", "3"]
    evaluation = json.loads(
        (experiment.parent / "run" / "dev-evaluation.json").read_text()
    )["evaluation"]
    assert evaluation["fullset_eer"] == result.best_dev_eer
    assert evaluation["failure_count"] == 0
    assert len(list((experiment.parent / "cache").glob("*.pt"))) == 4
    assert len(report["provenance"]["weights_sha256"]) == 64
    assert (
        json.loads((experiment.parent / "run" / "decode_failures.json").read_text())
        == []
    )


def test_bandwidth_augmentation_trains_without_changing_the_cache(
    experiment: Path,
) -> None:
    run_experiment(experiment)
    root = experiment.parent
    cached = {path.name: file_sha256(path) for path in (root / "cache").glob("*.pt")}
    assert len(cached) == 4
    augmentation = {"bandwidth": {"probability": 0.5, "target_sample_rate": 8000}}
    _change(experiment, output_dir="augmented", augmentation=augmentation)
    result = run_experiment(experiment)
    report = json.loads((root / "augmented" / "experiment.json").read_text())
    assert result.best_checkpoint.is_file()
    assert report["status"] == "complete"
    assert report["provenance"]["augmentation"] == augmentation
    assert report["holdout_accessed"] is False
    assert len(load_scores(root / "augmented" / "dev-scores.jsonl")) == 2
    assert {
        path.name: file_sha256(path) for path in (root / "cache").glob("*.pt")
    } == cached


def test_spawned_loader_worker_can_read_prepared_tensors(experiment: Path) -> None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except PermissionError:
        pytest.skip("sandbox blocks the Unix socket IPC needed by PyTorch workers")
    _change(experiment, num_workers=1)
    result = run_experiment(experiment)
    assert result.epochs_completed == 1


def test_repeated_epochs_verify_existing_cache_once(
    experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_experiment(experiment)
    verified = []
    from deepfake_detection_archive.models.checkpoint import verify_sha256  # noqa: PLC0415

    def count_verification(path: Path, digest: str) -> None:
        verified.append(path)
        verify_sha256(path, digest)

    monkeypatch.setattr(
        "deepfake_detection_archive.experiments.verify_sha256", count_verification
    )
    _change(
        experiment,
        output_dir="repeat",
        balanced_sampling=True,
        balance_sources=True,
        training={"max_epochs": 2, "device": "cpu", "seed": 23},
    )
    result = run_experiment(experiment)
    assert result.epochs_completed == 2
    assert sorted(verified) == sorted((experiment.parent / "cache").glob("*.pt"))


def test_gated_selection_reports_domain_metrics_and_gate_status(
    experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parses = 0
    original = SelectionOptions.from_options

    def parse(options: Mapping[str, Any] | None) -> SelectionOptions | None:
        nonlocal parses
        parses += 1
        return original(options)

    monkeypatch.setattr(SelectionOptions, "from_options", parse)
    _change(
        experiment,
        output_dir="gated",
        selection={
            "metric": "domain_macro_eer",
            "max_bucket_eer": {"unknown": 1.0},
        },
    )
    result = run_experiment(experiment)
    assert parses == 1
    assert result.selection_gate_passed is True
    report = json.loads((experiment.parent / "gated" / "experiment.json").read_text())
    assert report["selection_gate_passed"] is True
    assert report["best_dev_eer"] == result.best_dev_eer
    assert report["best_selection_metric"] == result.best_selection_metric
    assert report["best_selection_value"] == result.best_selection_value
    assert result.best_selection_metric == "domain_macro_eer"
    history = json.loads(result.history_path.read_text())
    assert history[0]["dev_macro_eer"] == history[0]["dev_domain_eer"]["unknown"]
    evaluation = json.loads(
        (experiment.parent / "gated" / "dev-evaluation.json").read_text()
    )
    assert set(evaluation["by_content_type"]) == {"unknown"}
    assert (
        evaluation["by_content_type"]["unknown"]["sample_count"]
        == evaluation["evaluation"]["sample_count"]
    )


def test_training_prepares_media_without_a_redundant_decode_pass(
    experiment: Path,
) -> None:
    manifest = experiment.parent / "manifest.jsonl"
    records = [
        replace(record, decode_status="unknown") for record in load_manifest(manifest)
    ]
    write_manifest(records, manifest)
    result = run_experiment(experiment)
    assert result.epochs_completed == 1
    assert len(load_scores(experiment.parent / "run" / "dev-scores.jsonl")) == 2


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
    monkeypatch.setattr(
        "deepfake_detection_archive.experiments.extract_audio_feature_batch",
        unexpected_extract,
    )
    # A CLI/reporting change alters whole-package provenance, not the features.
    monkeypatch.setattr(
        "deepfake_detection_archive.experiments.code_identity",
        lambda: {"code_sha256": "b" * 64},
    )
    _change(feature_experiment, output_dir="repeat")
    second = run_experiment(feature_experiment)
    assert first.best_dev_eer == second.best_dev_eer
    assert sorted((feature_experiment.parent / "cache").glob("*.pt")) == cached
    report = json.loads(
        (feature_experiment.parent / "repeat" / "experiment.json").read_text()
    )
    assert report["code"]["code_sha256"] == "b" * 64
    # A changed preprocessing choice has a different content cache identity.
    _change(feature_experiment, output_dir="changed", segments=3)
    with pytest.raises(ValueError, match="sample preparation failed"):
        run_experiment(feature_experiment)
    failures = json.loads(
        (feature_experiment.parent / "changed" / "decode_failures.json").read_text()
    )
    assert failures
    assert all(row["stage"] == "audio_feature_batch" for row in failures)
    assert config["encoder"]["expected_hashes"]


def test_changed_feature_implementation_invalidates_cache(
    feature_experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_experiment(feature_experiment)

    def changed_hash(path: Path) -> str:
        if path.name == "features.py":
            return "c" * 64
        return file_sha256(path)

    def unexpected_extract(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("changed feature implementation requires extraction")

    monkeypatch.setattr(
        "deepfake_detection_archive.experiments.file_sha256", changed_hash
    )
    monkeypatch.setattr(
        "deepfake_detection_archive.experiments.extract_features", unexpected_extract
    )
    monkeypatch.setattr(
        "deepfake_detection_archive.experiments.extract_audio_feature_batch",
        unexpected_extract,
    )
    _change(feature_experiment, output_dir="changed-implementation")
    with pytest.raises(ValueError, match="sample preparation failed"):
        run_experiment(feature_experiment)


def test_feature_scoring_reads_media_hash_once(
    feature_experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = run_experiment(feature_experiment)
    config = json.loads(feature_experiment.read_text())
    scorer = OfflineScorer(
        InferenceConfig(
            manifest="manifest.jsonl",
            data_root="media",
            output_dir="scores",
            modality="audio",
            checkpoint=str(result.best_checkpoint),
            checkpoint_sha256=result.checkpoint_sha256,
            device="cpu",
            model_kind="audio_features",
            encoder=config["encoder"],
            segments=2,
            segment_samples=400,
        ),
        feature_experiment.parent,
    )
    record = load_manifest(feature_experiment.parent / "manifest.jsonl")[0]
    media = feature_experiment.parent / "media" / record.path
    reads = []
    original_open = Path.open

    def count_reads(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == media:
            reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_reads)
    assert np.isfinite(scorer(record))
    assert reads == [media]


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros(400, dtype=torch.float64),
        torch.full((400,), float("nan")),
        torch.zeros(2),
    ],
    ids=["wrong-dtype", "nonfinite", "wrong-shape"],
)
def test_invalid_registered_cache_is_reported_before_training(
    experiment: Path, tensor: torch.Tensor
) -> None:
    run_experiment(experiment)
    cached = next((experiment.parent / "cache").glob("*.pt"))
    torch.save(tensor, cached)
    sidecar = cached.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    metadata["sha256"] = file_sha256(cached)
    sidecar.write_text(json.dumps(metadata))
    _change(experiment, output_dir="invalid-cache")
    with pytest.raises(ValueError, match="sample preparation failed"):
        run_experiment(experiment)
    output = experiment.parent / "invalid-cache"
    assert json.loads((output / "decode_failures.json").read_text())
    assert not (output / "training").exists()


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
        {"balance_sources": "yes"},
        {"balance_sources": True, "balanced_sampling": False},
        {"content_type_weights": {"music": 1.0}},
        {
            "balanced_sampling": True,
            "balance_sources": False,
            "content_type_weights": {"music": 1.0},
        },
        {
            "balanced_sampling": True,
            "balance_sources": True,
            "content_type_weights": {"music": 0.0},
        },
        {
            "balanced_sampling": True,
            "balance_sources": True,
            "content_type_weights": {"music": True},
        },
        {"selection": {"metric": "dev_eer"}},
        {"selection": {"metric": "domain_macro_eer", "unexpected": 1}},
        {"selection": {"metric": "domain_macro_eer", "max_bucket_eer": {"music": 1.5}}},
        {"selection": "domain_macro_eer"},
        {"unexpected": 1},
        {"audio": {"sample_rate": 8000}},
        {"training": {"max_epochs": 0}},
        {"augmentation": {"bandwidth": {"probability": 0.5}}},
        {
            "augmentation": {
                "bandwidth": {"probability": True, "target_sample_rate": 8000}
            }
        },
        {
            "augmentation": {
                "bandwidth": {"probability": 1.5, "target_sample_rate": 8000}
            }
        },
        {
            "augmentation": {
                "bandwidth": {"probability": 0.5, "target_sample_rate": 16000}
            }
        },
        {"augmentation": {"cutout": {"probability": 0.5}}},
        {
            "architecture": "audio_ast_v1",
            "augmentation": {
                "bandwidth": {"probability": 0.5, "target_sample_rate": 8000}
            },
        },
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
