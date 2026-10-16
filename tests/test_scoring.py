"""Torch allocation failures abort inference without swallowing model errors."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.data.manifest import SampleRecord
from deepfake_detection_archive.media import AudioConfig, AudioData, VideoConfig
from deepfake_detection_archive.scoring import OfflineScorer
from deepfake_detection_archive.submission.runner import run_inference


CPU_ALLOCATION_FAILURE = (
    "[enforce fail at alloc_cpu.cpp:127] err == 0. "
    "DefaultCPUAllocator: can't allocate memory: you tried to allocate 4000000000 "
    "bytes. Error code 12 (Cannot allocate memory)"
)


def make_scorer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> OfflineScorer:
    scorer = object.__new__(OfflineScorer)
    scorer.config = InferenceConfig(
        manifest="manifest.jsonl",
        data_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        modality="video",
        model_kind="video_frames",
        checkpoint="model.pt",
        checkpoint_sha256="0" * 64,
        device="cpu",
    )
    scorer.execution = Mock()
    scorer.execution.score.side_effect = lambda call: call(scorer.config.batch_size)
    scorer.root = tmp_path
    scorer.device = torch.device("cpu")
    scorer.frame_preprocessing = {}
    scorer.video = VideoConfig()
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.extract_video_frames",
        Mock(return_value=torch.zeros(1, 3, 2, 2)),
    )
    return scorer


def records() -> list[SampleRecord]:
    return [SampleRecord(str(i), f"{i}.mp4", "video", i % 2, "test") for i in range(3)]


def test_frame_scorer_passes_cpu_frames_and_configured_batch(monkeypatch, tmp_path):
    scorer = make_scorer(monkeypatch, tmp_path)
    scorer.config = replace(scorer.config, batch_size=3)
    # Device transfer belongs to the model's bounded frame batches.
    scorer.device = torch.device("cuda")
    frames = torch.randn(7, 3, 2, 2)
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.extract_video_frames",
        Mock(return_value=frames),
    )
    scorer.model = Mock(return_value=torch.tensor([0.25]))
    assert scorer(records()[0]) == 0.25
    scorer.model.assert_called_once()
    images = scorer.model.call_args.args[0]
    assert images.shape == (1, 7, 3, 2, 2)
    assert images.device.type == "cpu"
    assert images.data_ptr() == frames.data_ptr()
    assert scorer.model.call_args.kwargs == {"frame_batch_size": 3}


def test_cpu_allocator_failure_aborts_and_preserves_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer = make_scorer(monkeypatch, tmp_path)
    error = RuntimeError(CPU_ALLOCATION_FAILURE)
    model = Mock(side_effect=[torch.tensor([0.25]), error, torch.tensor([0.75])])
    monkeypatch.setattr(scorer, "model", model, raising=False)
    output = tmp_path / "output"
    with pytest.raises(MemoryError, match="DefaultCPUAllocator") as caught:
        run_inference(records(), scorer, output, identity={"model": "mock"})
    assert caught.value.__cause__ is error
    assert model.call_count == 2
    assert not (output / "report.json").exists()
    path = output / "scores.jsonl"
    prefix = path.read_bytes()
    saved = [json.loads(line) for line in prefix.splitlines()]
    assert len(saved) == 1
    assert saved[0]["sample_id"] == "0"
    assert saved[0]["status"] == "ok"
    assert saved[0]["score"] == 0.25

    model.side_effect = [torch.tensor([0.5]), torch.tensor([0.75])]
    result = run_inference(
        records(), scorer, output, identity={"model": "mock"}, resume=True
    )
    assert model.call_count == 4
    assert [row.score for row in result] == [0.25, 0.5, 0.75]
    assert path.read_bytes().startswith(prefix)


@pytest.mark.parametrize(
    "message", ["model shape mismatch", "can't allocate memory for this sample"]
)
def test_ordinary_runtime_error_remains_a_sample_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, message: str
) -> None:
    scorer = make_scorer(monkeypatch, tmp_path)
    model = Mock(
        side_effect=[torch.tensor([0.25]), RuntimeError(message), torch.tensor([0.75])]
    )
    monkeypatch.setattr(scorer, "model", model, raising=False)
    result = run_inference(
        records(), scorer, tmp_path / "output", identity={"model": "mock"}
    )
    assert model.call_count == 3
    assert [row.status for row in result] == ["ok", "error", "ok"]
    assert result[1].score is None
    assert result[1].error_code == "inference_error"
    assert result[1].error_message == message
    assert (tmp_path / "output" / "report.json").exists()


def test_feature_scorer_uses_selected_stream_configs(monkeypatch, tmp_path):
    scorer = make_scorer(monkeypatch, tmp_path)
    scorer.frame_preprocessing = None
    scorer.encoder = Mock()
    scorer.audio = AudioConfig()
    scorer.model = Mock(return_value=torch.tensor([0.25]))
    extract = Mock(return_value=torch.zeros(2, 4))
    monkeypatch.setattr("deepfake_detection_archive.scoring.extract_features", extract)
    audio_config = AudioConfig(stream_index=2)
    video_config = VideoConfig(stream_index=3)
    assert (
        scorer(records()[0], audio_config=audio_config, video_config=video_config)
        == 0.25
    )
    assert extract.call_args.args[3] is audio_config
    assert extract.call_args.args[4] is video_config


def test_frame_policy_controls_effective_batch_without_changing_sampling(
    monkeypatch, tmp_path
):
    scorer = make_scorer(monkeypatch, tmp_path)
    scorer.execution.score.side_effect = lambda call: call(2)
    scorer.model = Mock(return_value=torch.tensor([0.25]))
    assert scorer(records()[0]) == 0.25
    assert scorer.config.batch_size == 8
    assert scorer.model.call_args.kwargs == {"frame_batch_size": 2}


def test_audio_policy_batches_all_observations_before_one_pool(monkeypatch, tmp_path):
    scorer = make_scorer(monkeypatch, tmp_path)
    scorer.config = replace(
        scorer.config, modality="audio", model_kind="audio_waveform"
    )
    scorer.execution.score.side_effect = lambda call: call(2)
    scorer.frame_preprocessing = None
    scorer.encoder = None
    scorer.audio = AudioConfig()
    scorer.model = Mock(side_effect=lambda batch: torch.full((len(batch),), 0.25))
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.media_path", lambda *_: tmp_path
    )
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.decode_audio",
        lambda *_: AudioData(np.zeros(8, dtype=np.float32), 16000, 16000, 0.0005),
    )
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.sample_audio_segments",
        lambda *_: np.zeros((5, 8), dtype=np.float32),
    )
    record = SampleRecord("audio", "audio.wav", "audio", 0, "test")
    assert scorer(record) == 0.25
    assert [len(call.args[0]) for call in scorer.model.call_args_list] == [2, 2, 1]
