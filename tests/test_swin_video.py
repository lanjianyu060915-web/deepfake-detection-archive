"""Real small Video Swin regressions for the native temporal model."""

import socket
from dataclasses import replace

import pytest
import torch
import torchvision.models.video
from torchvision.models.video.swin_transformer import SwinTransformer3d

from deepfake_detection_archive import scoring
from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.data.manifest import SampleRecord
from deepfake_detection_archive.models import (
    SwinVideoClassifier,
    SwinVideoConfig,
    load_checkpoint,
    save_checkpoint,
)
from deepfake_detection_archive.scoring import OfflineScorer


@pytest.fixture(autouse=True)
def tiny_backbone(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)

    def factory(*, weights):
        assert weights is None
        return SwinTransformer3d(
            patch_size=[2, 4, 4],
            embed_dim=4,
            depths=[1, 1, 1, 1],
            num_heads=[1, 1, 2, 4],
            window_size=[2, 2, 2],
            mlp_ratio=1,
            stochastic_depth_prob=0.5,
        )

    monkeypatch.setattr(torchvision.models.video, "swin3d_t", factory)
    yield
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize("trainable_stages", [0, 1])
def test_raw_cached_and_complete_clip_chunks_agree(trainable_stages):
    model = SwinVideoClassifier(
        SwinVideoConfig(trainable_stages=trainable_stages)
    ).eval()
    images = torch.randn(2, 16, 3, 224, 224)
    with torch.inference_mode():
        tokens = model.prefix(images)
        features = model.backbone(images.permute(0, 2, 1, 3, 4)).unsqueeze(1)
        expected = model.head(features)
        assert tokens.shape == (2, 392, 32)
        assert tokens.dtype == torch.float32
        assert not tokens.requires_grad
        torch.testing.assert_close(model.encode(images), features)
        torch.testing.assert_close(model.encode_tokens(tokens), features)
        torch.testing.assert_close(model(images), expected)
        torch.testing.assert_close(model(tokens), expected)
        observed = []
        hook = model.backbone.patch_embed.register_forward_pre_hook(
            lambda module, args: observed.append(args[0].shape[:3])
        )
        torch.testing.assert_close(model(images, frame_batch_size=1), expected)
        hook.remove()
        assert observed == [(1, 3, 16), (1, 3, 16)]
        torch.testing.assert_close(model(tokens, frame_batch_size=16), expected)


@pytest.mark.parametrize("trainable_stages", [0, 1])
def test_only_last_stage_norm_and_head_update(trainable_stages):
    model = SwinVideoClassifier(SwinVideoConfig(trainable_stages=trainable_stages))
    for mode in (False, True):
        model.train(mode)
        assert all(not module.training for module in model.backbone.modules())
        assert model.head.training == mode
    tokens = model.prefix(torch.randn(1, 16, 3, 224, 224))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(tokens).square().mean().backward()
    optimizer.step()
    changed = set()
    for name, value in model.named_parameters():
        selected = name.startswith("head.") or (
            trainable_stages
            and name.startswith(("backbone.features.6.", "backbone.norm."))
        )
        assert value.requires_grad == bool(selected)
        if not selected:
            assert value.grad is None
        if not torch.equal(value, before[name]):
            changed.add(name)
            assert selected
    assert any(name.startswith("head.") for name in changed)
    if trainable_stages:
        assert any(name.startswith("backbone.features.6.") for name in changed)
        assert any(name.startswith("backbone.norm.") for name in changed)


def test_checkpoint_offline_roundtrip_and_scorer_frontend(tmp_path, monkeypatch):
    model = SwinVideoClassifier(SwinVideoConfig()).eval()
    path = tmp_path / "swin.pt"
    descriptor = save_checkpoint(path, model)

    def no_network(*args, **kwargs):
        pytest.fail("Swin checkpoint construction must remain offline")

    monkeypatch.setattr(socket, "socket", no_network)
    restored, metadata = load_checkpoint(path, expected_sha256=descriptor["sha256"])
    assert metadata["architecture"] == "video_swin3d_v1"
    images = torch.randn(1, 16, 3, 224, 224)
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), model(images), rtol=0, atol=0)
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
            "n_frames": 16,
            "height": 224,
            "width": 224,
            "resize_mode": "native_center_crop",
        },
    )
    scorer = OfflineScorer(config, tmp_path)
    assert scorer.frame_preprocessing == model.preprocessing
    monkeypatch.setattr(scoring, "extract_video_frames", lambda *args: images[0])
    record = SampleRecord("clip", "clip.mp4", "video", 0, "dev")
    with torch.inference_mode():
        assert scorer(record) == float(model(images)[0])
    scorer.execution.finish()
    for patch in (
        {"n_frames": 8},
        {"height": 112, "width": 112},
        {"resize_mode": "center_crop"},
    ):
        with pytest.raises(ValueError, match="Video Swin requires"):
            OfflineScorer(replace(config, video={**config.video, **patch}), tmp_path)


@pytest.mark.parametrize("trainable_stages", [-1, 2, True])
def test_invalid_trainable_stages(trainable_stages):
    with pytest.raises(ValueError, match="trainable_stages"):
        SwinVideoConfig(trainable_stages=trainable_stages)


@pytest.mark.parametrize("n_frames", [8, 32, True])
def test_invalid_frame_count(n_frames):
    with pytest.raises(ValueError, match="n_frames"):
        SwinVideoConfig(n_frames=n_frames)
