"""Offline UFD checkpoint, raw frame logits and author-native crop contract."""

import socket
import subprocess
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, ToTensor

from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.data.manifest import SampleRecord, file_sha256
from deepfake_detection_archive.features import (
    extract_features,
    extract_video_feature_batch,
    extract_video_frames,
)
from deepfake_detection_archive.media import AudioConfig, VideoConfig
from deepfake_detection_archive.models import load_checkpoint, save_checkpoint
from deepfake_detection_archive.models.encoders import FrozenVisionEncoder
from deepfake_detection_archive.models.ufd import UFDConfig, UFDVideoClassifier
from deepfake_detection_archive.scoring import OfflineScorer


def tiny_config() -> UFDConfig:
    return UFDConfig(
        backbone_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
            "projection_dim": 6,
            "hidden_act": "quick_gelu",
        }
    )


def test_checkpoint_cold_load_and_raw_frame_mean(tmp_path, monkeypatch):
    model = UFDVideoClassifier(tiny_config()).eval()
    images = torch.randn(2, 3, 3, 8, 8)
    with torch.inference_mode():
        features = model.backbone(pixel_values=images.flatten(0, 1)).image_embeds
        expected = model.head(features).reshape(2, 3).mean(dim=1)
        assert torch.allclose(model(images), expected)
        assert not torch.allclose(features.norm(dim=1), torch.ones(6))
    path = tmp_path / "ufd.pt"
    descriptor = save_checkpoint(path, model)

    def no_network(*args, **kwargs):
        pytest.fail("checkpoint loading must remain offline")

    monkeypatch.setattr(socket, "socket", no_network)
    loaded, metadata = load_checkpoint(path, expected_sha256=descriptor["sha256"])
    assert metadata["architecture"] == "ufd_video_v1"
    with torch.inference_mode():
        assert torch.equal(loaded(images), expected)


def test_frozen_clip_features_preserve_projection_and_freeze(tmp_path):
    model = UFDVideoClassifier(tiny_config()).eval()
    checkpoint = tmp_path / "ufd.pt"
    descriptor = save_checkpoint(checkpoint, model)
    encoder = FrozenVisionEncoder(
        "ufd_clip_vit_l14", checkpoint, expected_sha256=descriptor["sha256"]
    )
    encoder.train()
    assert not encoder.training
    assert not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    images = torch.randn(3, 3, 8, 8)
    with torch.inference_mode():
        expected = model.backbone(pixel_values=images).image_embeds
        torch.testing.assert_close(encoder(images), expected, rtol=0, atol=0)
    assert encoder.feature_dim == 6
    assert encoder.preprocessing == model.preprocessing


def test_frame_batches_preserve_projection_head_and_tail(monkeypatch):
    torch.manual_seed(66)
    model = UFDVideoClassifier(tiny_config()).eval()
    images = torch.randn(2, 7, 3, 8, 8)
    with torch.inference_mode():
        expected = model(images)
    backbone = Mock(wraps=model.backbone.forward)
    head = Mock(wraps=model.head.forward)
    monkeypatch.setattr(model.backbone, "forward", backbone)
    monkeypatch.setattr(model.head, "forward", head)
    with torch.inference_mode():
        actual = model(images, frame_batch_size=3)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert [
        call.kwargs["pixel_values"].shape[0] for call in backbone.call_args_list
    ] == [3, 3, 3, 3, 2]
    head.assert_called_once()
    assert head.call_args.args[0].shape == (14, 6)


def test_clip_feature_batch_matches_serial_and_owns_storage(tmp_path):
    model = UFDVideoClassifier(tiny_config()).eval()
    checkpoint = tmp_path / "ufd.pt"
    descriptor = save_checkpoint(checkpoint, model)
    encoder = FrozenVisionEncoder(
        "ufd_clip_vit_l14", checkpoint, expected_sha256=descriptor["sha256"]
    )
    records = []
    for index in range(2):
        png = tmp_path / f"source-{index}.png"
        pixels = np.random.default_rng(index).integers(
            0, 256, (11, 13, 3), dtype=np.uint8
        )
        Image.fromarray(pixels).save(png)
        path = tmp_path / f"video-{index}.mkv"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(png), "-c:v", "ffv1", str(path)],
            check=True,
            capture_output=True,
            timeout=20,
        )
        records.append(
            SampleRecord(
                str(index), path.name, "video", index, "dev", sha256=file_sha256(path)
            )
        )
    config = VideoConfig(
        n_frames=3, height=8, width=8, resize_mode="native_center_crop"
    )
    expected = [
        extract_features(
            record, tmp_path, encoder, AudioConfig(), config, 10, 1, "cpu", 4
        )
        for record in records
    ]
    actual = extract_video_feature_batch(
        records, tmp_path, encoder, config, "cpu", 4, 2
    )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    with pytest.raises(ValueError, match="resize_mode"):
        extract_video_feature_batch(
            records,
            tmp_path,
            encoder,
            VideoConfig(n_frames=3, height=8, width=8),
            "cpu",
            4,
            2,
        )


def test_scorer_requires_native_crop_and_matching_aggregation(tmp_path):
    path = tmp_path / "ufd.pt"
    descriptor = save_checkpoint(path, UFDVideoClassifier(tiny_config()))
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
    with pytest.raises(ValueError, match="native_center_crop"):
        OfflineScorer(InferenceConfig(**common), tmp_path)
    video = {
        "n_frames": 4,
        "height": 8,
        "width": 8,
        "resize_mode": "native_center_crop",
    }
    with pytest.raises(ValueError, match="aggregation"):
        OfflineScorer(
            InferenceConfig(**common, video=video, aggregation="median"), tmp_path
        )
    scorer = OfflineScorer(InferenceConfig(**common, video=video), tmp_path)
    assert isinstance(scorer.model, UFDVideoClassifier)


@pytest.mark.parametrize("size", [(231, 237), (217, 219), (231, 219)])
def test_native_video_preprocessing_matches_pil_center_crop(tmp_path: Path, size):
    width, height = size
    pixels = np.random.default_rng(12).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    image = Image.fromarray(pixels)
    png = tmp_path / "source.png"
    image.save(png)
    video = tmp_path / "source.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(png),
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    record = SampleRecord(
        "native", video.name, "video", 0, "dev", sha256=file_sha256(video)
    )
    model = UFDVideoClassifier(tiny_config())
    preprocessing = {**model.preprocessing, "input_size": (3, 224, 224)}
    actual = extract_video_frames(
        record,
        tmp_path,
        VideoConfig(n_frames=1, resize_mode="native_center_crop"),
        preprocessing,
    )
    expected = Compose(
        [
            CenterCrop(224),
            ToTensor(),
            Normalize(preprocessing["mean"], preprocessing["std"]),
        ]
    )(image)
    assert torch.equal(actual[0], expected)


@pytest.mark.parametrize("rotation", [90, 270])
def test_native_crop_uses_autorotated_display_dimensions(tmp_path: Path, rotation):
    pixels = np.random.default_rng(14).integers(0, 256, (217, 231, 3), dtype=np.uint8)
    png = tmp_path / "source.png"
    Image.fromarray(pixels).save(png)
    encoded = tmp_path / "encoded.mp4"
    video = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(png),
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            str(encoded),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-display_rotation",
            str(rotation),
            "-i",
            str(encoded),
            "-c",
            "copy",
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    displayed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    ).stdout
    # A 90/270-degree display matrix swaps the encoded 231x217 dimensions.
    image = Image.fromarray(
        np.frombuffer(displayed, dtype=np.uint8).reshape(231, 217, 3)
    )
    preprocessing = {
        **UFDVideoClassifier(tiny_config()).preprocessing,
        "input_size": (3, 224, 224),
    }
    record = SampleRecord(
        "rotated", video.name, "video", 0, "dev", sha256=file_sha256(video)
    )
    actual = extract_video_frames(
        record,
        tmp_path,
        VideoConfig(n_frames=1, resize_mode="native_center_crop"),
        preprocessing,
    )
    expected = Compose(
        [
            CenterCrop(224),
            ToTensor(),
            Normalize(preprocessing["mean"], preprocessing["std"]),
        ]
    )(image)
    assert torch.equal(actual[0], expected)
