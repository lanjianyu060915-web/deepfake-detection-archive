"""SSL upstream frontend, score direction and complete failure accounting."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.data.manifest import SampleRecord, write_manifest
from deepfake_detection_archive.media import AudioData
from deepfake_detection_archive.models.ssl_antispoofing import (
    fairseq_to_hf,
    pad_or_tile,
)
from deepfake_detection_archive.scoring import OfflineScorer
from deepfake_detection_archive.submission.runner import run_inference
from deepfake_detection_archive.unified_predict import _resolved_config


def config(tmp_path):
    return InferenceConfig(
        manifest="manifest.jsonl",
        data_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        modality="audio",
        checkpoint="LA_model.pth",
        checkpoint_sha256="0" * 64,
        device="cpu",
        model_kind="audio_ssl_antispoofing",
        audio={"backend": "soundfile"},
        ssl={
            "upstream_model": "model.py",
            "xlsr_config": "xlsr.json",
            "xlsr_config_sha256": "1" * 64,
        },
    )


def test_upstream_crop_tile_and_key_mapping():
    wave = np.array([1, 2, 3], dtype=np.float32)
    assert pad_or_tile(wave, 8).tolist() == [1, 2, 3, 1, 2, 3, 1, 2]
    assert pad_or_tile(wave, 2).tolist() == [1, 2]
    assert fairseq_to_hf("quantizer.weight") is None
    assert fairseq_to_hf("encoder.layers.2.fc1.weight") == (
        "encoder.layers.2.feed_forward.intermediate_dense.weight"
    )


@pytest.mark.parametrize("change", [{"segments": 2}, {"audio": {"sample_rate": 8000}}])
def test_ssl_rejects_frontend_change(tmp_path, change):
    with pytest.raises(ValueError, match="SSL requires"):
        replace(config(tmp_path), **change)


def test_standard_scorer_head_tile_direction_and_failed_row(monkeypatch, tmp_path):
    model = Mock(return_value=torch.tensor([[9.0, 2.0]]))
    monkeypatch.setattr("deepfake_detection_archive.scoring.verify_sha256", Mock())
    monkeypatch.setattr(
        "deepfake_detection_archive.scoring.build_model",
        Mock(return_value=(model, {"upstream_model_sha256": "pinned"})),
    )
    wave = np.array([1, 2, 3], dtype=np.float32)
    decode = Mock(return_value=AudioData(wave, 16000, 16000, 3 / 16000))
    monkeypatch.setattr("deepfake_detection_archive.scoring.decode_audio", decode)
    (tmp_path / "ok.wav").touch()
    scorer = OfflineScorer(config(tmp_path), tmp_path)
    rows = [
        SampleRecord(
            "ok",
            "ok.wav",
            "audio",
            0,
            "test",
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        SampleRecord(
            "bad",
            "bad.wav",
            "audio",
            1,
            "test",
            status="missing",
            failure_reason="not downloaded",
        ),
    ]
    scores = run_inference(rows, scorer, tmp_path / "results", identity={"test": "ssl"})
    assert [row.status for row in scores] == ["ok", "error"]
    assert scores[0].score == -2.0
    assert scores[1].score is None
    assert decode.call_count == 1
    inputs = model.call_args.args[0]
    assert inputs.shape == (1, 64600)
    assert inputs.dtype == torch.float32
    assert inputs[0, :7].tolist() == [1, 2, 3, 1, 2, 3, 1]


@pytest.mark.parametrize("model_error", [False, True])
def test_research_tool_keeps_unavailable_and_model_failure_rows(
    monkeypatch, tmp_path, model_error
):
    path = Path(__file__).parents[1] / "tools/score_ssl_antispoofing.py"
    spec = importlib.util.spec_from_file_location("score_ssl_tool", path)
    assert spec is not None
    assert spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    model = Mock(return_value=torch.tensor([[1.0, 2.0]]))
    if model_error:
        model.side_effect = RuntimeError("model failed")
    monkeypatch.setattr(
        tool, "build_model", Mock(return_value=(model, {"loaded_keys": 1}))
    )
    monkeypatch.setattr(
        tool,
        "decode_audio",
        Mock(
            return_value=AudioData(
                np.ones(3, dtype=np.float32), 16000, 16000, 3 / 16000
            )
        ),
    )
    rows = [
        SampleRecord(
            "ok",
            "ok.wav",
            "audio",
            0,
            "test",
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        SampleRecord(
            "bad",
            "bad.wav",
            "audio",
            1,
            "test",
            status="missing",
            failure_reason="not downloaded",
        ),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(rows, manifest)
    output = tmp_path / "tool"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--weights",
            "unused",
            "--upstream-model",
            "unused",
            "--xlsr-config",
            "unused",
            "--manifest",
            str(manifest),
            "--data-root",
            str(tmp_path),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    tool.main()
    scores = [
        json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()
    ]
    assert {row["sample_id"] for row in scores} == {"ok", "bad"}
    assert sum(row["status"] == "error" for row in scores) == (2 if model_error else 1)


def test_packaged_ssl_resources_resolve_inside_moved_bundle(monkeypatch, tmp_path):
    folder = tmp_path / "configs"
    folder.mkdir()
    resources = tmp_path / "models"
    resources.mkdir()
    for name in ("LA_model.pth", "model.py", "xlsr.json"):
        (resources / name).touch()
    template = replace(
        config(tmp_path),
        checkpoint="../models/LA_model.pth",
        ssl={
            "upstream_model": "../models/model.py",
            "xlsr_config": "../models/xlsr.json",
            "xlsr_config_sha256": "1" * 64,
        },
    )
    (folder / "ssl.json").write_text(json.dumps(template.to_dict()))
    monkeypatch.setattr(
        "deepfake_detection_archive.unified_predict._model_package_root",
        lambda: tmp_path,
    )
    resolved = _resolved_config(
        folder, {"config": "ssl.json"}, tmp_path / "out", "audio", "cpu"
    )
    assert Path(resolved.ssl["upstream_model"]) == resources / "model.py"
    assert Path(resolved.ssl["xlsr_config"]) == resources / "xlsr.json"
    assert Path(resolved.checkpoint) == resources / "LA_model.pth"
