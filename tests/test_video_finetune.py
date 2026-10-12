"""Behaviour checks for the partially trainable video frame classifier."""

import json
import socket
from dataclasses import asdict, replace
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
    TemporalHead,
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


@pytest.mark.parametrize("kernel_size", [1, 3])
def test_only_trainable_prefix_and_head_receive_gradients(tmp_path, kernel_size):
    config = replace(_config(tmp_path), temporal_kernel_size=kernel_size)
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
    assert all(
        not has_grad
        for name, has_grad in grads.items()
        if not name.startswith(("backbone.layer4.", "head."))
    )
    assert torch.equal(frozen_before, model.backbone.layer1[0].bn1.running_mean)
    assert not torch.equal(trained_before, model.backbone.layer4[0].bn1.running_mean)


@pytest.mark.parametrize("kernel_size", [1, 3])
def test_checkpoint_restores_the_same_logits_offline(
    tmp_path, monkeypatch, kernel_size
):
    config = replace(_config(tmp_path), temporal_kernel_size=kernel_size)
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


def test_temporal_head_uses_frame_order():
    head = TemporalHead(feature_dim=3, hidden_dim=1, dropout=0).eval()
    with torch.no_grad():
        head.layers[0].weight.zero_()
        head.layers[0].weight[0, 0, 0] = 1
        head.layers[0].weight[0, 1, 1] = 2
        head.layers[0].bias.zero_()
        head.layers[3].weight.fill_(1)
        head.layers[3].bias.zero_()
    features = torch.tensor([[[3.0, 0, 0], [0, 3, 0], [0, 0, 3], [2, 1, 0]]])
    original = head(features).mean(dim=1)
    shuffled = head(features[:, [0, 2, 1, 3]]).mean(dim=1)
    assert not torch.allclose(original, shuffled)


def test_legacy_checkpoint_keeps_independent_frame_pooling(tmp_path):
    model = VideoFrameClassifier(_config(tmp_path)).eval()
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(checkpoint, model)
    payload = torch.load(checkpoint, weights_only=True)
    del payload["config"]["temporal_kernel_size"]
    torch.save(payload, checkpoint)
    loaded, _ = load_checkpoint(checkpoint, expected_sha256=sha256_file(checkpoint))
    assert loaded.config.temporal_kernel_size == 1
    assert list(model.state_dict()) == list(loaded.state_dict())
    images = torch.randn(2, 4, 3, 64, 64)
    with torch.inference_mode():
        reference = model(images)
        restored = loaded(images)
        shuffled = loaded(images[:, [0, 2, 1, 3]])
    assert torch.equal(reference, restored)
    torch.testing.assert_close(restored, shuffled)


@pytest.mark.parametrize("kernel_size", [0, 2, 5, True, 3.0])
def test_temporal_kernel_size_is_validated(kernel_size):
    with pytest.raises(ValueError, match="temporal_kernel_size"):
        VideoFrameConfig(
            architecture="resnet18",
            encoder_checkpoint="unused.pt",
            expected_sha256="0" * 64,
            temporal_kernel_size=kernel_size,
        )


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
