"""Last-block adaptation preserves offline raw-frame scoring and frozen weights."""

import socket
import subprocess

import numpy as np
import pytest
import torch
from PIL import Image

from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.data.manifest import SampleRecord, file_sha256
from deepfake_detection_archive.models import (
    ClipAdaptConfig,
    ClipAdaptVideoClassifier,
    FrozenVisionEncoder,
    load_checkpoint,
    save_checkpoint,
)
from deepfake_detection_archive.scoring import OfflineScorer


def tiny_config(**kwargs):
    return ClipAdaptConfig(
        backbone_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
            "projection_dim": 6,
            "hidden_act": "quick_gelu",
        },
        hidden_dim=8,
        dropout=0.1,
        **kwargs,
    )


@pytest.mark.parametrize("train_last_block", [True, False])
def test_raw_split_and_frame_batches_preserve_unnormalized_projection(train_last_block):
    torch.manual_seed(83)
    model = ClipAdaptVideoClassifier(
        tiny_config(train_last_block=train_last_block)
    ).eval()
    images = torch.randn(2, 3, 3, 8, 8)
    with torch.inference_mode():
        tokens = model.encode_prefix(images.flatten(0, 1)).reshape(2, 3, 5, 16)
        projection = model.backbone(pixel_values=images.flatten(0, 1)).image_embeds
        expected = model.head(projection).reshape(2, 3).mean(dim=1)
        torch.testing.assert_close(model(images), expected)
        torch.testing.assert_close(model(tokens), expected)
        torch.testing.assert_close(model(images, frame_batch_size=4), expected)
        assert not torch.allclose(projection.norm(dim=1), torch.ones(6))
    assert tokens.dtype == torch.float32
    assert not tokens.requires_grad


@pytest.mark.parametrize("train_last_block", [True, False])
def test_only_selected_parameters_update_and_modes_stay_frozen(train_last_block):
    torch.manual_seed(85)
    model = ClipAdaptVideoClassifier(tiny_config(train_last_block=train_last_block))
    vision = model.backbone.vision_model
    for mode in (False, True):
        model.train(mode)
        assert not vision.embeddings.training
        assert not vision.pre_layrnorm.training
        assert not vision.encoder.layers[0].training
        assert not model.backbone.visual_projection.training
        assert vision.encoder.layers[-1].training == (mode and train_last_block)
        assert vision.post_layernorm.training == (mode and train_last_block)
        assert model.head.training == mode
    before = {name: value.clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    images = torch.randn(2, 3, 3, 8, 8)
    tokens = model.encode_prefix(images.flatten(0, 1)).reshape(2, 3, 5, 16)
    model(tokens).square().mean().backward()
    optimizer.step()
    changed = set()
    for name, value in model.named_parameters():
        selected = name.startswith("head.") or (
            train_last_block
            and name.startswith(
                (
                    "backbone.vision_model.encoder.layers.1.",
                    "backbone.vision_model.post_layernorm.",
                )
            )
        )
        assert value.requires_grad == selected
        if not torch.equal(before[name], value):
            changed.add(name)
            assert selected
        if not selected:
            assert value.grad is None
    assert any(name.startswith("head.") for name in changed)
    if train_last_block:
        assert any("encoder.layers.1." in name for name in changed)
        assert any("post_layernorm." in name for name in changed)


def test_offline_roundtrip_encoder_and_real_media(tmp_path, monkeypatch):
    model = ClipAdaptVideoClassifier(tiny_config()).eval()
    path = tmp_path / "adapt.pt"
    descriptor = save_checkpoint(path, model)

    def no_network(*args, **kwargs):
        pytest.fail("adaptation loading/inference must remain offline")

    monkeypatch.setattr(socket, "socket", no_network)
    loaded, metadata = load_checkpoint(path, expected_sha256=descriptor["sha256"])
    assert metadata["architecture"] == "clip_adapt_video_v1"
    images = torch.randn(1, 2, 3, 8, 8)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(images), model(images), rtol=0, atol=0)
    encoder = FrozenVisionEncoder(
        "ufd_clip_vit_l14", path, expected_sha256=descriptor["sha256"]
    )
    assert encoder.feature_dim == 6
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    with torch.inference_mode():
        torch.testing.assert_close(
            encoder(images.flatten(0, 1)),
            model.backbone(pixel_values=images.flatten(0, 1)).image_embeds,
        )
    png = tmp_path / "source.png"
    Image.fromarray(np.zeros((11, 13, 3), dtype=np.uint8)).save(png)
    video = tmp_path / "video.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(png), "-c:v", "ffv1", str(video)],
        check=True,
        capture_output=True,
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
            "n_frames": 3,
            "height": 8,
            "width": 8,
            "resize_mode": "native_center_crop",
        },
        batch_size=2,
    )
    score = OfflineScorer(config, tmp_path)(record)
    assert np.isfinite(score)


def test_invalid_training_flag():
    with pytest.raises(TypeError, match="train_last_block"):
        tiny_config(train_last_block=1)
