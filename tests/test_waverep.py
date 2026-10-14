"""Offline WaveRep checkpoint, raw frame-logit mean and native-crop contract."""

import socket

import pytest
import timm
import torch

from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.models import load_checkpoint, save_checkpoint
from deepfake_detection_archive.models.waverep import (
    WaveRepConfig,
    WaveRepVideoClassifier,
)
from deepfake_detection_archive.scoring import OfflineScorer


TINY = {"img_size": 28, "embed_dim": 16, "depth": 1, "num_heads": 2}


def tiny_model() -> WaveRepVideoClassifier:
    return WaveRepVideoClassifier(WaveRepConfig(backbone_overrides=TINY)).eval()


def test_matches_author_model_frame_logits_and_loads_offline(tmp_path, monkeypatch):
    model = tiny_model()
    # Author demo: timm.create_model(arc, num_classes=1, img_size=...) then [:, -1].
    reference = timm.create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained=False,
        num_classes=1,
        **TINY,
    ).eval()
    reference.load_state_dict(model.backbone.state_dict(), strict=True)
    images = torch.randn(2, 3, 3, 28, 28)
    with torch.inference_mode():
        frames = reference(images.flatten(0, 1))[:, -1].reshape(2, 3)
        expected = frames.mean(dim=1)
        torch.testing.assert_close(model(images), expected, rtol=0, atol=1e-6)
    path = tmp_path / "waverep.pt"
    descriptor = save_checkpoint(path, model)

    def no_network(*args, **kwargs):
        pytest.fail("checkpoint loading must remain offline")

    monkeypatch.setattr(socket, "socket", no_network)
    loaded, metadata = load_checkpoint(path, expected_sha256=descriptor["sha256"])
    assert metadata["architecture"] == "waverep_video_v1"
    assert loaded.preprocessing["input_size"] == (3, 28, 28)
    with torch.inference_mode():
        assert torch.equal(loaded(images), model(images))


def test_config_rejects_non_mean_aggregation_and_missing_size():
    with pytest.raises(ValueError, match="mean"):
        WaveRepConfig(aggregation="median")
    with pytest.raises(ValueError, match="img_size"):
        WaveRepConfig(backbone_overrides={})


def test_scorer_requires_native_crop_at_detector_size(tmp_path):
    path = tmp_path / "waverep.pt"
    descriptor = save_checkpoint(path, tiny_model())
    common = {
        "manifest": "manifest.jsonl",
        "data_root": str(tmp_path),
        "output_dir": str(tmp_path / "out"),
        "modality": "video",
        "model_kind": "video_frames",
        "checkpoint": str(path),
        "checkpoint_sha256": descriptor["sha256"],
        "device": "cpu",
    }
    resized = {"n_frames": 4, "height": 28, "width": 28, "resize_mode": "center_crop"}
    with pytest.raises(ValueError, match="native_center_crop"):
        OfflineScorer(InferenceConfig(**common, video=resized), tmp_path)
    wrong_size = {"n_frames": 4, "height": 14, "width": 14}
    with pytest.raises(ValueError, match="native_center_crop"):
        OfflineScorer(
            InferenceConfig(
                **common, video={**wrong_size, "resize_mode": "native_center_crop"}
            ),
            tmp_path,
        )
    video = {"n_frames": 4, "height": 28, "width": 28}
    scorer = OfflineScorer(
        InferenceConfig(**common, video={**video, "resize_mode": "native_center_crop"}),
        tmp_path,
    )
    assert isinstance(scorer.model, WaveRepVideoClassifier)
