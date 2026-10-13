"""CPU-only launch checks must never turn into decoding, training or downloads."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from deepfake_detection_archive import cli
from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.preflight import check_training_config


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("preflight must not construct models, decode or use CUDA")

    for name in ("create_model", "load_encoder", "decode_audio", "run_experiment"):
        monkeypatch.setattr(f"deepfake_detection_archive.experiments.{name}", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    root = tmp_path / "media"
    root.mkdir()
    records = []
    for index, split in enumerate(("train", "train", "dev", "dev", "holdout")):
        path = root / f"{index}.wav"
        # These are intentionally undecodable. Preflight validates bytes, not media.
        digest = "f" * 64
        if split != "holdout":
            path.write_bytes(f"not audio {index}".encode())
            digest = file_sha256(path)
        records.append(
            SampleRecord(
                str(index),
                path.name,
                "audio",
                index % 2,
                "fixture",
                content_type="speech",
                split=split,
                sha256=digest,
                license_id="self-generated",
                group_id=f"group-{index}",
            )
        )
    write_manifest(records, tmp_path / "manifest.jsonl")
    (tmp_path / "sources.jsonl").write_text(
        json.dumps(
            {
                "source": "fixture",
                "url": "local:generated-test-fixture",
                "version": "1",
                "license_id": "self-generated",
                "license_url": "local:test-fixture",
                "downloaded_at": "2026-10-13",
                "allowed_uses": ["training"],
                "license_status": "approved",
            }
        )
        + "\n"
    )
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(
            {
                "manifest": "manifest.jsonl",
                "data_root": "media",
                "sources": "sources.jsonl",
                "output_dir": "new/run",
                "modality": "audio",
                "architecture": "logmel_cnn_v1",
                "model_config": {},
                "training": {"device": "cuda:77", "amp": True},
                "feature_cache": "new/cache",
                "dev_domain_field": "content_type",
            }
        )
    )
    return path


def change_config(path: Path, **changes: Any) -> None:
    config = json.loads(path.read_text())
    config.update(changes)
    path.write_text(json.dumps(config))


def test_cli_read_only_cpu_preflight_allows_unverified_decode_and_absent_holdout(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = {
        path: path.read_bytes()
        for path in config_path.parent.rglob("*")
        if path.is_file()
    }
    assert cli.main(["train", "--config", str(config_path), "--check-config"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "check-config"
    assert report["passed"] is True
    assert report["train_samples"] == report["dev_samples"] == 2
    assert report["excluded_samples"] == 1
    assert report["requested_device"] == "cuda:77"
    assert report["holdout_media_accessed"] is False
    assert any("decoding" in item for item in report["unchecked"])
    assert any("weight compatibility" in item for item in report["unchecked"])
    assert any("GPU memory" in item for item in report["unchecked"])
    assert report["audit"]["total_bytes"] == sum(
        path.stat().st_size for path in (config_path.parent / "media").iterdir()
    )
    assert not (config_path.parent / "new").exists()
    assert {
        path: path.read_bytes()
        for path in config_path.parent.rglob("*")
        if path.is_file()
    } == before


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_config": {"channels": 0}}, "positive integer"),
        ({"model_config": {"unused": 1}}, "unexpected keyword"),
        ({"audio": {"sample_rate": 8000}}, "sample rates"),
        ({"audio": {"unexpected": 2}}, "invalid experiment"),
        ({"video": {"n_frames": 0}}, "n_frames"),
        ({"training": {"device": "cpu", "amp": True}}, "AMP"),
        ({"training": {"accumulation_steps": 0}}, "accumulation_steps"),
        ({"architecture": "aasist_l_v1", "model_config": {"unused": 1}}, "empty"),
        ({"architecture": "aasist_l_v1", "segment_samples": 64000}, "64600"),
        ({"architecture": "audio_ast_v1", "model_config": {}}, "backbone_config"),
        (
            {
                "architecture": "audio_ast_v1",
                "model_config": {
                    "backbone_config": {},
                    "waveform_preparation": "center_pad_64000_peak095_v1",
                },
            },
            "segment_samples=64000",
        ),
    ],
)
def test_preflight_rejects_bad_schema_before_run_creation(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    changes: dict[str, Any],
    message: str,
) -> None:
    change_config(config_path, **changes)
    assert cli.main(["train", "--config", str(config_path), "--check-config"]) == 2
    assert message in json.loads(capsys.readouterr().err)["message"]
    assert not (config_path.parent / "new").exists()


@pytest.mark.parametrize("split", ["train", "dev"])
def test_preflight_requires_both_classes(config_path: Path, split: str) -> None:
    path = config_path.parent / "manifest.jsonl"
    records = [
        replace(record, label=0) if record.split == split else record
        for record in load_manifest(path)
    ]
    write_manifest(records, path)
    report = check_training_config(config_path)
    assert report["passed"] is False
    assert f"{split} requires both classes with known labels" in report["split_errors"]


@pytest.mark.parametrize("domain", ["speech", "music"])
def test_preflight_rejects_missing_or_one_class_constrained_domain(
    config_path: Path, domain: str
) -> None:
    path = config_path.parent / "manifest.jsonl"
    records = load_manifest(path)
    records[3] = replace(records[3], content_type="environment")
    write_manifest(records, path)
    change_config(
        config_path,
        training={
            "domain_eer_constraints": {
                domain: {"baseline_eer": 0.1, "max_regression": 0.02}
            }
        },
    )
    report = check_training_config(config_path)
    assert report["passed"] is False
    assert any(f"domain {domain!r}" in error for error in report["split_errors"])


def test_preflight_checks_full_metadata_for_holdout_leakage(config_path: Path) -> None:
    path = config_path.parent / "manifest.jsonl"
    records = load_manifest(path)
    records[-1] = replace(records[-1], parent_id=records[0].sample_id)
    write_manifest(records, path)
    report = check_training_config(config_path)
    assert report["passed"] is False
    assert any("leakage" in error for error in report["split_errors"])
    assert report["holdout_media_accessed"] is False


@pytest.mark.parametrize(
    "failure", ["missing_file", "hash_mismatch", "license_not_approved"]
)
def test_cli_reports_data_failure_and_exits_nonzero(
    config_path: Path, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    if failure == "missing_file":
        (config_path.parent / "media" / "0.wav").unlink()
    elif failure == "hash_mismatch":
        (config_path.parent / "media" / "0.wav").write_bytes(b"changed")
    else:
        path = config_path.parent / "sources.jsonl"
        source = json.loads(path.read_text())
        source["license_status"] = "unresolved"
        path.write_text(json.dumps(source))
    assert cli.main(["train", "--config", str(config_path), "--check-config"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert failure in {issue["code"] for issue in report["audit"]["issues"]}
    assert not (config_path.parent / "new").exists()


@pytest.mark.parametrize("is_directory", [False, True])
def test_preflight_refuses_existing_output_without_changing_it(
    config_path: Path, is_directory: bool
) -> None:
    output = config_path.parent / "used"
    if is_directory:
        output.mkdir()
        (output / "history.json").write_text("preserve")
    else:
        output.write_text("preserve")
    change_config(config_path, output_dir="used")
    with pytest.raises(ValueError, match="absent or empty"):
        check_training_config(config_path)
    assert (
        output / "history.json" if is_directory else output
    ).read_text() == "preserve"


def audio_bundle(config_path: Path) -> dict[str, str]:
    root = config_path.parent / "bundle"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "hubert"}))
    (root / "preprocessor_config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"not a model; weights are not loaded")
    hashes = {path.name: file_sha256(path) for path in root.iterdir()}
    change_config(
        config_path,
        architecture="media_feature_head_v1",
        model_config={"input_dim": 768},
        encoder={"kind": "audio", "model_dir": "bundle", "expected_hashes": hashes},
    )
    return hashes


def test_preflight_checks_audio_bundle_bytes_without_loading_model(
    config_path: Path,
) -> None:
    audio_bundle(config_path)
    report = check_training_config(config_path)
    assert report["passed"] is True
    assert len(report["local_assets"]) == 3
    (config_path.parent / "bundle" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        check_training_config(config_path)


@pytest.mark.parametrize(
    "failure", ["missing_hash", "bad_hash_type", "bad_architecture", "bad_index"]
)
def test_preflight_rejects_bad_audio_bundle(config_path: Path, failure: str) -> None:
    audio_bundle(config_path)
    root = config_path.parent / "bundle"
    config = json.loads(config_path.read_text())
    expected = config["encoder"]["expected_hashes"]
    if failure == "missing_hash":
        del expected["preprocessor_config.json"]
        message = "trusted hashes"
    elif failure == "bad_hash_type":
        expected["model.safetensors"] = 5
        message = "SHA-256 strings"
    elif failure == "bad_architecture":
        (root / "config.json").write_text('{"model_type": "untrusted"}')
        expected["config.json"] = file_sha256(root / "config.json")
        message = "unsupported local SSL"
    else:
        index = root / "model.safetensors.index.json"
        index.write_text('{"weight_map": {"layer": "missing.safetensors"}}')
        expected[index.name] = file_sha256(index)
        message = "absent shards"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        check_training_config(config_path)
    assert not (config_path.parent / "new").exists()


@pytest.mark.parametrize(
    "architecture", ["media_feature_head_v1", "video_frame_finetune_v1", "audio_ast_v1"]
)
def test_preflight_hashes_local_video_and_ast_assets(
    config_path: Path, architecture: str
) -> None:
    filename = "model.safetensors" if architecture == "audio_ast_v1" else "weights.pt"
    checkpoint = config_path.parent / filename
    checkpoint.write_bytes(b"hash-verified; tensor compatibility is unchecked")
    digest = file_sha256(checkpoint)
    if architecture == "audio_ast_v1":
        (config_path.parent / "config.json").write_text(
            '{"model_type": "audio-spectrogram-transformer"}'
        )
        change_config(
            config_path,
            architecture=architecture,
            model_config={"backbone_config": {}},
            pretrained={"checkpoint": filename, "sha256": digest},
        )
    else:
        path = config_path.parent / "manifest.jsonl"
        write_manifest(
            [replace(record, modality="video") for record in load_manifest(path)], path
        )
        changes: dict[str, Any] = {
            "architecture": architecture,
            "modality": "video",
            "model_config": {"input_dim": 512},
        }
        asset = {"architecture": "resnet18", "expected_sha256": digest}
        if architecture == "media_feature_head_v1":
            changes["encoder"] = {"kind": "vision", "checkpoint": "weights.pt", **asset}
        else:
            changes["model_config"] = {"encoder_checkpoint": "weights.pt", **asset}
        change_config(config_path, **changes)
    report = check_training_config(config_path)
    assert report["passed"] is True
    assert report["local_assets"] == [{"path": str(checkpoint), "sha256": digest}]
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        check_training_config(config_path)


@pytest.mark.parametrize(
    "index", [{}, {"weight_map": []}, {"weight_map": {"layer": 7}}]
)
def test_cli_reports_malformed_ssl_index_as_structured_error(
    config_path: Path, capsys: pytest.CaptureFixture[str], index: dict[str, Any]
) -> None:
    audio_bundle(config_path)
    path = config_path.parent / "bundle" / "model.safetensors.index.json"
    path.write_text(json.dumps(index))
    config = json.loads(config_path.read_text())
    config["encoder"]["expected_hashes"][path.name] = file_sha256(path)
    config_path.write_text(json.dumps(config))
    assert cli.main(["train", "--config", str(config_path), "--check-config"]) == 2
    assert "weight_map" in json.loads(capsys.readouterr().err)["message"]


@pytest.mark.parametrize(
    "encoder",
    [
        {"kind": "audio", "model_dir": "bundle", "expected_hashes": {}, "unused": 1},
        {"kind": "audio", "model_dir": "", "expected_hashes": {}},
        {"kind": "audio", "model_dir": "bundle", "expected_hashes": []},
    ],
)
def test_preflight_rejects_encoder_schema(
    config_path: Path, encoder: dict[str, Any]
) -> None:
    change_config(
        config_path,
        architecture="media_feature_head_v1",
        model_config={"input_dim": 768},
        encoder=encoder,
    )
    with pytest.raises(ValueError, match="encoder"):
        check_training_config(config_path)


@pytest.mark.parametrize(
    "failure", ["wrong_filename", "absent_config", "invalid_config"]
)
def test_preflight_rejects_incomplete_ast_pretrained_bundle(
    config_path: Path, failure: str
) -> None:
    filename = "weights.pt" if failure == "wrong_filename" else "model.safetensors"
    checkpoint = config_path.parent / filename
    checkpoint.write_bytes(b"not loaded")
    change_config(
        config_path,
        architecture="audio_ast_v1",
        model_config={"backbone_config": {}},
        pretrained={"checkpoint": filename, "sha256": file_sha256(checkpoint)},
    )
    if failure == "invalid_config":
        (config_path.parent / "config.json").write_text("[]")
    if failure == "absent_config":
        with pytest.raises(FileNotFoundError, match=r"config\.json"):
            check_training_config(config_path)
    else:
        with pytest.raises(
            ValueError,
            match=r"named model\.safetensors|configuration must be an object",
        ):
            check_training_config(config_path)


def test_train_without_check_flag_keeps_training_dispatch(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from deepfake_detection_archive.training import TrainResult  # noqa: PLC0415

    called = []

    def train(path: Path) -> TrainResult:
        called.append(path)
        return TrainResult(
            Path("best.pt"),
            "a" * 64,
            1,
            0.5,
            1,
            "max_epochs",
            Path("history.json"),
            Path("run.json"),
        )

    monkeypatch.setattr("deepfake_detection_archive.experiments.run_experiment", train)
    assert cli.main(["train", "--config", str(config_path)]) == 0
    assert called == [config_path]
    report = json.loads(capsys.readouterr().out)
    assert report["best_checkpoint"] == "best.pt"
    assert "mode" not in report
