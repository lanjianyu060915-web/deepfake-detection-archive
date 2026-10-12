"""Behaviour checks for the partially trainable video frame classifier."""

import json
import socket
from dataclasses import asdict
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from deepfake_detection_archive import experiments  # noqa: E402
from deepfake_detection_archive.models.checkpoint import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)
from deepfake_detection_archive.models.finetune import (  # noqa: E402
    VideoFrameClassifier,
    VideoFrameConfig,
)


def _config(tmp_path):
    timm = pytest.importorskip("timm")

    backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
    path = tmp_path / "encoder.pt"
    torch.save(
        {key: value.detach().cpu() for key, value in backbone.state_dict().items()},
        path,
    )
    return VideoFrameConfig(
        architecture="resnet18",
        encoder_checkpoint=str(path),
        expected_sha256=sha256_file(path),
        hidden_dim=8,
    )


def test_only_trainable_prefix_and_head_receive_gradients(tmp_path):
    config = _config(tmp_path)
    model = VideoFrameClassifier(config)
    model.initialize_pretrained(Path(config.encoder_checkpoint), config.expected_sha256)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(name.startswith(("backbone.layer4", "head")) for name in trainable)
    frozen_before = model.backbone.layer1[0].bn1.running_mean.clone()
    trained_before = model.backbone.layer4[0].bn1.running_mean.clone()
    model.train()
    assert model.backbone.layer4.training
    assert not model.backbone.layer1.training
    logits = model(torch.randn(2, 3, 3, 64, 64))
    assert logits.shape == (2,)
    logits.sum().backward()
    grads = {name: p.grad is not None for name, p in model.named_parameters()}
    assert any(v for n, v in grads.items() if n.startswith("backbone.layer4"))
    assert any(v for n, v in grads.items() if n.startswith("head"))
    assert not any(v for n, v in grads.items() if n.startswith("backbone.layer1"))
    assert torch.equal(frozen_before, model.backbone.layer1[0].bn1.running_mean)
    assert not torch.equal(trained_before, model.backbone.layer4[0].bn1.running_mean)


def test_checkpoint_restores_the_same_logits_offline(tmp_path, monkeypatch):
    config = _config(tmp_path)
    model = VideoFrameClassifier(config)
    model.initialize_pretrained(Path(config.encoder_checkpoint), config.expected_sha256)
    descriptor = save_checkpoint(tmp_path / "best.pt", model)
    deployed = tmp_path / "deployment"
    deployed.mkdir()
    checkpoint = (tmp_path / "best.pt").rename(deployed / "best.pt")
    Path(model.config.encoder_checkpoint).unlink()

    def forbid_download(*args, **kwargs):
        raise AssertionError("checkpoint restoration must stay offline")

    monkeypatch.setattr(socket, "socket", forbid_download)
    monkeypatch.setattr("timm.models._builder.load_pretrained", forbid_download)
    loaded, _ = load_checkpoint(checkpoint, expected_sha256=descriptor["sha256"])
    assert isinstance(loaded, VideoFrameClassifier)
    model.eval()
    images = torch.randn(2, 3, 3, 64, 64)
    with torch.inference_mode():
        reference = model(images)
        restored = loaded(images)
    assert torch.equal(reference, restored)


@pytest.mark.parametrize("relative", [True, False])
def test_experiment_initializes_encoder_relative_to_config(
    tmp_path, monkeypatch, relative
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = _config(config_dir)
    expected = torch.load(config.encoder_checkpoint, weights_only=True)
    model_config = asdict(config)
    if relative:
        model_config["encoder_checkpoint"] = "encoder.pt"
    config_path = config_dir / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "manifest": "manifest.jsonl",
                "sources": "sources.jsonl",
                "data_root": ".",
                "output_dir": "output",
                "modality": "video",
                "architecture": "video_frame_finetune_v1",
                "model_config": model_config,
                "training": {},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiments, "code_identity", dict)
    monkeypatch.setattr(experiments, "load_manifest", lambda path: [])
    monkeypatch.setattr(experiments, "load_sources", lambda path: {})
    monkeypatch.setattr(experiments, "_audit_training_records", lambda *args: [])

    class PreparationReachedError(Exception):
        pass

    def stop_before_training(*args, **kwargs):
        raise PreparationReachedError

    monkeypatch.setattr(experiments, "_prepare_cache", stop_before_training)
    with pytest.raises(PreparationReachedError):
        experiments.run_experiment(config_path)
    initial = config_dir / "output" / "initial.pt"
    payload = torch.load(initial, weights_only=True)
    for key, value in expected.items():
        assert torch.equal(payload["state_dict"]["backbone." + key], value)


def test_pretrained_initialization_verifies_digest(tmp_path):
    config = _config(tmp_path)
    model = VideoFrameClassifier(config)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        model.initialize_pretrained(Path(config.encoder_checkpoint), "0" * 64)


def test_full_checkpoint_requires_complete_backbone_state(tmp_path):
    config = _config(tmp_path)
    model = VideoFrameClassifier(config)
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(checkpoint, model)
    Path(config.encoder_checkpoint).unlink()
    payload = torch.load(checkpoint, weights_only=True)
    del payload["state_dict"]["backbone.conv1.weight"]
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(checkpoint, expected_sha256=sha256_file(checkpoint))
