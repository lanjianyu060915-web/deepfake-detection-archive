"""Every offline scoring branch attributes decode and forward work separately."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.data.manifest import SampleRecord
from deepfake_detection_archive.features import extract_features
from deepfake_detection_archive.media import AudioConfig, AudioData, VideoConfig
from deepfake_detection_archive.models.ast import ASTClassifier
from deepfake_detection_archive.models.encoders import (
    FrozenAudioEncoder,
    FrozenVisionEncoder,
)
from deepfake_detection_archive.scoring import OfflineScorer
from deepfake_detection_archive.submission.runner import run_inference


@pytest.mark.parametrize(
    ("kind", "ast"),
    [
        ("audio_waveform", False),
        ("audio_waveform", True),
        ("audio_features", False),
        ("video_features", False),
        ("video_frames", False),
    ],
)
def test_scoring_stages_include_forward_and_existing_cpu_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, ast: bool
) -> None:
    now = [0.0]
    monkeypatch.setattr(
        "deepfake_detection_archive.timing.time.perf_counter", lambda: now[0]
    )
    synchronize = Mock(side_effect=AssertionError("extra CUDA synchronization"))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

    def timed_result(seconds, value):
        def call(*args, **kwargs):
            now[0] += seconds
            return value

        return call

    source = tmp_path / "source.media"
    source.write_bytes(b"registered local media")
    monkeypatch.setattr(
        "deepfake_detection_archive.features.file_sha256",
        timed_result(1, "a" * 64),
    )
    modality = "audio" if kind.startswith("audio") else "video"
    record = SampleRecord("sample", source.name, modality, 0, "test", sha256="a" * 64)
    scorer = object.__new__(OfflineScorer)
    scorer.config = InferenceConfig(
        manifest="manifest.jsonl",
        data_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        modality=modality,
        model_kind=kind,
        checkpoint="model.pt",
        checkpoint_sha256="0" * 64,
        device="cpu",
        segment_samples=8,
    )
    scorer.root = tmp_path
    scorer.device = torch.device("cpu")
    scorer.audio = AudioConfig()
    scorer.video = VideoConfig(n_frames=1, height=2, width=2)
    preprocessing = {
        "input_size": (3, 2, 2),
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
    }
    scorer.frame_preprocessing = preprocessing if kind == "video_frames" else None
    scorer.encoder = None
    if kind.endswith("features"):
        scorer.encoder = Mock(
            spec=FrozenAudioEncoder if modality == "audio" else FrozenVisionEncoder,
            side_effect=timed_result(3, torch.ones(1, 4)),
        )
        scorer.encoder.preprocessing = preprocessing
    scorer.model = Mock(
        spec=ASTClassifier if ast else None,
        side_effect=timed_result(5, torch.tensor([0.75])),
    )
    if ast:
        scorer.model.prepare_audio = Mock(
            side_effect=timed_result(4, torch.zeros(1, 2, 2))
        )
    audio = AudioData(np.ones(8, dtype=np.float32), 16000, 16000, 8 / 16000)
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.decode_audio", timed_result(2, audio)
    )
    monkeypatch.setattr(
        "deepfake_detection_archive.features.decode_audio", timed_result(2, audio)
    )
    monkeypatch.setattr(
        "deepfake_detection_archive.features.decode_video",
        timed_result(2, Mock(frames=np.zeros((1, 2, 2, 3), dtype=np.uint8))),
    )

    if scorer.encoder is not None:
        features = extract_features(
            record,
            scorer.root,
            scorer.encoder,
            scorer.audio,
            scorer.video,
            scorer.config.segment_samples,
            scorer.config.segments,
            "cpu",
            scorer.config.batch_size,
        )
        # Preserve the existing standalone extraction result, usable by a head
        # outside inference_mode (e.g. before saving a training feature cache).
        assert not torch.is_inference(features)

    scores = run_inference(
        [record], scorer, tmp_path / "output", identity={"model": "test"}
    )
    assert scores[0].score == 0.75
    report = json.loads((tmp_path / "output" / "report.json").read_text())
    assert report["timing"]["current_session_instrumented_rows"] == 1
    assert report["timing"]["current_session_stage_seconds"] == {
        "input_validation": 1,
        "decode_and_preprocess": 6 if ast else 2,
        "model_and_feature_scoring": 8 if kind.endswith("features") else 5,
        "unattributed_sample_call": 0,
    }
    synchronize.assert_not_called()
