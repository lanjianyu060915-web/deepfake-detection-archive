"""Real small ViT checks for self-contained DINOv2 partial adaptation."""

import socket
import subprocess

import numpy as np
import pytest
import timm
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.data.manifest import SampleRecord, file_sha256
from deepfake_detection_archive.models import (
    DinoAdaptConfig,
    DinoAdaptVideoClassifier,
    load_checkpoint,
    save_checkpoint,
)
from deepfake_detection_archive.scoring import OfflineScorer


@pytest.fixture(autouse=True)
def tiny_backbone(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)

    def create_model(name, **kwargs):
        assert name == "vit_base_patch14_dinov2.lvd142m"
        assert kwargs == {
            "pretrained": False,
            "num_classes": 0,
            "dynamic_img_size": True,
        }
        return VisionTransformer(
            img_size=518,
            patch_size=14,
            embed_dim=768,
            depth=2,
            num_heads=12,
            mlp_ratio=1,
            num_classes=0,
            dynamic_img_size=True,
            init_values=1e-5,
        )

    monkeypatch.setattr(timm, "create_model", create_model)
    yield
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize("trainable_blocks", [0, 1])
def test_full_frames_prefix_and_frame_chunks_match(trainable_blocks):
    torch.manual_seed(79)
    model = DinoAdaptVideoClassifier(
        DinoAdaptConfig(trainable_blocks=trainable_blocks)
    ).eval()
    images = torch.randn(1, 3, 3, 224, 224)
    with torch.inference_mode():
        tokens = model.prefix(images.flatten(0, 1)).reshape(1, 3, 257, 768)
        features = model.backbone(images.flatten(0, 1)).reshape(1, 3, 768)
        expected = model.head(features)
        torch.testing.assert_close(model(images), expected)
        torch.testing.assert_close(model(tokens), expected)
        torch.testing.assert_close(model(images, frame_batch_size=2), expected)
        torch.testing.assert_close(
            model.forward_tokens(tokens, frame_batch_size=2), expected
        )
    assert model.backbone.pos_embed.shape == (1, 1370, 768)
    assert tokens.dtype == torch.float32
    assert not tokens.requires_grad


@pytest.mark.parametrize("trainable_blocks", [0, 1])
def test_only_last_block_norm_and_head_receive_gradients(trainable_blocks):
    model = DinoAdaptVideoClassifier(DinoAdaptConfig(trainable_blocks=trainable_blocks))
    for mode in (False, True):
        model.train(mode)
        assert not model.backbone.patch_embed.training
        assert not model.backbone.blocks[0].training
        assert model.backbone.blocks[-1].training == (mode and bool(trainable_blocks))
        assert model.backbone.norm.training == (mode and bool(trainable_blocks))
        assert model.head.training == mode
    images = torch.randn(1, 2, 3, 224, 224)
    tokens = model.prefix(images.flatten(0, 1)).reshape(1, 2, 257, 768)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(tokens).square().mean().backward()
    optimizer.step()
    changed = set()
    for name, value in model.named_parameters():
        selected = name.startswith("head.") or (
            trainable_blocks
            and name.startswith(("backbone.blocks.1.", "backbone.norm."))
        )
        assert value.requires_grad == bool(selected)
        if not selected:
            assert value.grad is None
        if not torch.equal(value, before[name]):
            changed.add(name)
            assert selected
    assert any(name.startswith("head.") for name in changed)
    if trainable_blocks:
        assert any(name.startswith("backbone.blocks.1.") for name in changed)
        assert any(name.startswith("backbone.norm.") for name in changed)


def test_checkpoint_roundtrip_and_standard_offline_scorer(tmp_path, monkeypatch):
    model = DinoAdaptVideoClassifier(DinoAdaptConfig()).eval()
    path = tmp_path / "dino.pt"
    descriptor = save_checkpoint(path, model)

    def no_network(*args, **kwargs):
        pytest.fail("checkpoint construction/inference must remain offline")

    monkeypatch.setattr(socket, "socket", no_network)
    restored, metadata = load_checkpoint(path, expected_sha256=descriptor["sha256"])
    assert metadata["architecture"] == "video_dinov2_adapt_v1"
    images = torch.randn(1, 2, 3, 224, 224)
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), model(images), rtol=0, atol=0)
    png = tmp_path / "source.png"
    Image.fromarray(np.zeros((226, 230, 3), dtype=np.uint8)).save(png)
    video = tmp_path / "video.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(png), "-c:v", "ffv1", str(video)],
        capture_output=True,
        check=True,
        timeout=20,
    )
    record = SampleRecord(
        "video", video.name, "video", 0, "dev", sha256=file_sha256(video)
    )
    config = InferenceConfig(
        manifest="manifest.jsonl",
        data_root=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        modality="video",
        model_kind="video_frames",
        checkpoint=str(path),
        checkpoint_sha256=descriptor["sha256"],
        device="cpu",
        video={
            "n_frames": 2,
            "height": 224,
            "width": 224,
            "resize_mode": "native_center_crop",
        },
        batch_size=1,
    )
    scorer = OfflineScorer(config, tmp_path)
    assert scorer.frame_preprocessing == model.preprocessing
    assert np.isfinite(scorer(record))
    scorer.execution.finish()


@pytest.mark.parametrize("trainable_blocks", [-1, 2, True])
def test_invalid_trainable_blocks(trainable_blocks):
    with pytest.raises(ValueError, match="trainable_blocks"):
        DinoAdaptConfig(trainable_blocks=trainable_blocks)
