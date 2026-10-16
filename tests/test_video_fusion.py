"""Fixed raw-logit fusion, offline resources, branch failures and entry resume."""

import json
import shutil
import socket
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.data.manifest import SampleRecord, file_sha256
from deepfake_detection_archive.models import (
    VideoFusionClassifier,
    VideoFusionConfig,
    load_checkpoint,
    save_checkpoint,
)
from deepfake_detection_archive.scoring import OfflineScorer
from deepfake_detection_archive.submission.runner import run_inference
from deepfake_detection_archive.unified_predict import predict_media


def tiny_model():
    config = VideoFusionConfig(
        clip={
            "backbone_config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 8,
                "patch_size": 2,
                "projection_dim": 6,
            }
        },
        clip_head={"input_dim": 6, "hidden_dim": 4, "dropout": 0.0},
        waverep={
            "backbone_overrides": {
                "img_size": 28,
                "embed_dim": 16,
                "depth": 1,
                "num_heads": 2,
            }
        },
        mean=[0.6, -5.7],
        std=[3.3, 2.6],
        coef=[1.4, 0.57],
        intercept=0.04,
    )
    return VideoFusionClassifier(config).eval()


def test_matches_separate_raw_logits_and_checkpoint_loads_offline(
    tmp_path, monkeypatch
):
    torch.manual_seed(72)
    model = tiny_model()
    clip = torch.randn(2, 4, 3, 8, 8)
    wave = torch.randn(2, 4, 3, 28, 28)
    with torch.inference_mode():
        features = model.clip_backbone(pixel_values=clip.flatten(0, 1)).image_embeds
        clip_score = model.clip_head(features.reshape(2, 4, -1))
        wave_score = model.waverep(wave)
        expected = (
            1.4 * (clip_score - 0.6) / 3.3 + 0.57 * (wave_score + 5.7) / 2.6 + 0.04
        )
        for batch in (1, 3, 8):
            torch.testing.assert_close(
                model(clip, wave, frame_batch_size=batch), expected
            )
    path = tmp_path / "fusion.pt"
    descriptor = save_checkpoint(path, model)
    monkeypatch.setattr(socket, "socket", Mock(side_effect=AssertionError("network")))
    loaded, metadata = load_checkpoint(path, expected_sha256=descriptor["sha256"])
    assert metadata["architecture"] == "video_fusion_v1"
    with torch.inference_mode():
        torch.testing.assert_close(loaded(clip, wave, frame_batch_size=3), expected)
    changed = VideoFusionClassifier(replace(model.config, intercept=0.05))
    changed.load_state_dict(model.state_dict())
    assert (
        save_checkpoint(tmp_path / "changed.pt", changed)["sha256"]
        != descriptor["sha256"]
    )


@pytest.mark.parametrize(
    "changes", [{"std": [0.0, 1.0]}, {"coef": [1.0]}, {"mean": [float("nan"), 0.0]}]
)
def test_rejects_invalid_calibration(changes):
    with pytest.raises(ValueError, match="fusion"):
        replace(tiny_model().config, **changes)


@pytest.fixture
def scorer(tmp_path):
    model = tiny_model()
    path = tmp_path / "fusion.pt"
    descriptor = save_checkpoint(path, model)
    config = InferenceConfig(
        manifest="manifest.jsonl",
        data_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        modality="video",
        model_kind="video_frames",
        checkpoint=str(path),
        checkpoint_sha256=descriptor["sha256"],
        device="cpu",
        video={
            "n_frames": 4,
            "height": 8,
            "width": 8,
            "resize_mode": "native_center_crop",
        },
        batch_size=3,
    )
    return OfflineScorer(config, tmp_path)


def test_fusion_preserves_branch_crop_normalization_and_selected_stream(
    scorer, monkeypatch
):
    resolved = Mock(return_value=Path("already-verified.mp4"))
    monkeypatch.setattr("deepfake_detection_archive.scoring.media_path", resolved)
    normalize = Mock(side_effect=[torch.zeros(4, 3, 8, 8), torch.ones(4, 3, 28, 28)])
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.normalized_video_frames", normalize
    )
    record = SampleRecord("sample", "sample.mp4", "video", 1, "dev")
    scorer(record, video_config=replace(scorer.video, stream_index=2))
    resolved.assert_called_once_with(record, scorer.root)
    clip, wave = normalize.call_args_list
    assert (clip.args[1].height, wave.args[1].height) == (8, 28)
    assert clip.args[1].stream_index == wave.args[1].stream_index == 2
    assert clip.args[2]["mean"] != wave.args[2]["mean"]
    with pytest.raises(ValueError, match="fusion requires"):
        OfflineScorer(replace(scorer.config, video={}), scorer.root)


@pytest.mark.parametrize("branch", ["clip_head", "waverep"])
def test_branch_failure_is_one_failed_file_without_partial_score(
    scorer, monkeypatch, tmp_path, branch
):
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.media_path", Mock(return_value=Path("x"))
    )
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.normalized_video_frames",
        Mock(side_effect=[torch.zeros(4, 3, 8, 8), torch.zeros(4, 3, 28, 28)]),
    )
    monkeypatch.setattr(
        getattr(scorer.model, branch), "forward", Mock(side_effect=RuntimeError(branch))
    )
    result = run_inference(
        [SampleRecord("sample", "sample.mp4", "video", 1, "dev")],
        scorer,
        tmp_path / "failure",
        identity={"config": scorer.config.to_dict()},
    )
    assert len(result) == 1
    assert result[0].status == "error"
    assert result[0].score is None
    assert result[0].error_message == branch


def test_predict_relocated_bundle_matches_scorer_and_binds_resume(
    scorer, tmp_path, monkeypatch
):
    video = tmp_path / "视频 space.mkv"
    # Video is stream 1: the selected stream must propagate to both branches.
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=0.3",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=32x32:d=0.3",
            "-map",
            "0:a",
            "-map",
            "1:v",
            "-c:v",
            "ffv1",
            str(video),
        ],
        check=True,
        timeout=20,
    )
    record = SampleRecord(
        "video", video.name, "video", 1, "dev", sha256=file_sha256(video)
    )
    expected = scorer(record, video_config=replace(scorer.video, stream_index=1))
    bundle = tmp_path / "bundle"
    (bundle / "configs").mkdir(parents=True)
    (bundle / "weights").mkdir()
    shutil.copyfile(scorer.config.checkpoint, bundle / "weights/fusion.pt")
    config = replace(scorer.config, checkpoint="../weights/fusion.pt")
    config_path = bundle / "configs/video.json"
    config_path.write_text(json.dumps(config.to_dict()))
    registry = {
        "schema_version": 1,
        "score_direction": "higher_is_fake",
        "score_type": "raw_logit",
        "models": {
            "audio": {},
            "video": {
                "model_id": "fusion-research",
                "status": "research",
                "scope": "visual_only; research_only",
                "preprocessing_version": "timestamp4_native_clip8_wave28_v1",
                "architecture": "video_fusion_v1",
                "checkpoint_schema": 1,
                "config": "video.json",
            },
        },
    }
    (bundle / "configs/models.json").write_text(json.dumps(registry))
    moved = tmp_path / "搬移 package"
    bundle.rename(moved)
    Path(scorer.config.checkpoint).unlink()
    monkeypatch.setenv("DEEPFAKE_MODEL_PACKAGE", str(moved))
    output = tmp_path / "prediction"
    report = predict_media([video], output, "cpu", resume=False)
    assert report["failures"] == 0
    row = json.loads((output / "video/scores.jsonl").read_text())
    assert row["score"] == pytest.approx(expected, abs=1e-5)
    prefix = (output / "video/scores.jsonl").read_bytes()
    assert predict_media([video], output, "cpu", resume=True)["failures"] == 0
    assert (output / "video/scores.jsonl").read_bytes() == prefix
    config_path = moved / "configs/video.json"
    changed = VideoFusionClassifier(replace(scorer.model.config, intercept=0.05))
    changed.load_state_dict(scorer.model.state_dict())
    descriptor = save_checkpoint(moved / "weights/fusion.pt", changed)
    config_path.write_text(
        json.dumps(replace(config, checkpoint_sha256=descriptor["sha256"]).to_dict())
    )
    with pytest.raises(ValueError, match="resume rejected"):
        predict_media([video], output, "cpu", resume=True)
    assert (output / "video/scores.jsonl").read_bytes() == prefix
