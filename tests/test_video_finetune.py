"""Behaviour checks for the partially trainable video frame classifier."""

import pytest


torch = pytest.importorskip("torch")

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
    model = VideoFrameClassifier(_config(tmp_path))
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


def test_checkpoint_restores_the_same_logits_offline(tmp_path):
    model = VideoFrameClassifier(_config(tmp_path))
    descriptor = save_checkpoint(tmp_path / "best.pt", model)
    loaded, _ = load_checkpoint(
        tmp_path / "best.pt", expected_sha256=descriptor["sha256"]
    )
    assert isinstance(loaded, VideoFrameClassifier)
    model.eval()
    images = torch.randn(2, 3, 3, 64, 64)
    with torch.inference_mode():
        reference = model(images)
        restored = loaded(images)
    assert torch.equal(reference, restored)
