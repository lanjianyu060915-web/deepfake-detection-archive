"""Runtime fingerprints follow the executed pipeline, not unrelated installations."""

import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from deepfake_detection_archive import runtime
from deepfake_detection_archive.media import AudioConfig
from deepfake_detection_archive.tracking import object_hash


def test_raw_soundfile_inputs_ignore_model_packages_and_training_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("raw native-rate audio does not use ffmpeg or model dependencies")

    monkeypatch.setattr(runtime, "_executable_version", forbidden)
    monkeypatch.setattr(runtime, "_torch_runtime", forbidden)
    monkeypatch.setattr(runtime, "version", forbidden)
    identity = runtime.preprocessing_runtime("audio", AudioConfig(backend="soundfile"))
    assert set(identity) == {"schema_version", "numpy", "decoder"}
    assert identity["decoder"]["backend"] == "soundfile"
    assert identity["decoder"]["libsndfile"]
    monkeypatch.setenv("UNRELATED_EXPERIMENT_FLAG", "changed")
    assert identity == runtime.preprocessing_runtime(
        "audio", AudioConfig(backend="soundfile")
    )


def test_decoder_identity_changes_with_linked_library_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = "libavcodec 60.1"
    calls: list[str] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[1:] == ["-version"]
        assert kwargs["timeout"] == 10
        calls.append(command[0])
        return subprocess.CompletedProcess(
            command, 0, f"{command[0]} version 6.1\n{library}\n", ""
        )

    monkeypatch.setattr(runtime.subprocess, "run", run)
    first = runtime.preprocessing_runtime("video", AudioConfig(backend="soundfile"))
    assert calls == ["ffmpeg", "ffprobe"]
    library = "libavcodec 60.2"
    changed = runtime.preprocessing_runtime("video", AudioConfig(backend="soundfile"))
    assert (
        first["decoder"]["ffmpeg"]["version"] == changed["decoder"]["ffmpeg"]["version"]
    )
    assert object_hash(first) != object_hash(changed)


@pytest.mark.parametrize("failure", ["missing", "timeout", "invalid"])
def test_unidentifiable_decoder_fails_closed(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if failure == "missing":
            raise FileNotFoundError(command[0])
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="media backend"):
        runtime.preprocessing_runtime("audio", AudioConfig())


def test_ast_tracks_selected_frontend_and_only_its_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers.models.audio_spectrogram_transformer import (  # noqa: PLC0415
        feature_extraction_audio_spectrogram_transformer as frontend,
    )

    versions = {"transformers": "one", "torchaudio": "one", "timm": "one"}
    looked_up: list[str] = []

    def version(name: str) -> str:
        looked_up.append(name)
        return versions[name]

    monkeypatch.setattr(runtime, "version", version)
    monkeypatch.setattr(frontend, "is_speech_available", lambda: False)
    config = AudioConfig(backend="soundfile")
    first = runtime.preprocessing_runtime("audio", config, ast=True)
    assert first["ast_frontend"]["backend"] == "numpy"
    assert looked_up == ["transformers"]
    versions.update(torchaudio="two", timm="two")
    assert first == runtime.preprocessing_runtime("audio", config, ast=True)
    versions["transformers"] = "two"
    assert first != runtime.preprocessing_runtime("audio", config, ast=True)
    monkeypatch.setattr(frontend, "is_speech_available", lambda: True)
    kaldi = runtime.preprocessing_runtime("audio", config, ast=True)
    assert kaldi["ast_frontend"]["backend"] == "torchaudio.compliance.kaldi"
    assert kaldi["ast_frontend"]["torchaudio"] == "two"
    versions["torchaudio"] = "three"
    assert kaldi != runtime.preprocessing_runtime("audio", config, ast=True)


def test_torch_cpu_runtime_does_not_probe_cuda_and_tracks_numerical_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("CPU runtime must not initialize/probe a CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_name", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    first = runtime.model_runtime(torch.nn.Linear(2, 1), "cpu")
    assert set(first) == {"torch", "tensor_dtypes"}
    monkeypatch.setattr(torch, "get_num_threads", lambda: 123)
    changed = runtime.model_runtime(torch.nn.Linear(2, 1), "cpu")
    assert first != changed
    assert changed["torch"]["num_threads"] == 123


def test_model_fingerprint_tracks_loaded_attention_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = SimpleNamespace(
                config=SimpleNamespace(_attn_implementation="sdpa")
            )

    monkeypatch.setattr(runtime, "FrozenAudioEncoder", Encoder)
    monkeypatch.setattr(runtime, "version", lambda name: "test-transformers")
    encoder = Encoder()
    first = runtime.model_runtime(encoder, "cpu")
    encoder.backbone.config._attn_implementation = "eager"
    assert first != runtime.model_runtime(encoder, "cpu")
    assert first["transformers"] == "test-transformers"
    assert "timm" not in first


def test_effective_model_dtype_is_part_of_runtime() -> None:
    model = torch.nn.Linear(2, 1)
    first = runtime.model_runtime(model, "cpu")
    assert first["tensor_dtypes"] == ["torch.float32"]
    changed = runtime.model_runtime(model.double(), "cpu")
    assert changed["tensor_dtypes"] == ["torch.float64"]
    assert first != changed


def test_simple_frame_normalization_ignores_neural_backend_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime, "_executable_version", lambda name: {"version": "test"}
    )
    first = runtime.preprocessing_runtime(
        "video", AudioConfig(), normalized_frames=True
    )
    monkeypatch.setattr(torch, "get_num_threads", lambda: 123)
    monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "bf16")
    assert first == runtime.preprocessing_runtime(
        "video", AudioConfig(), normalized_frames=True
    )


@pytest.mark.parametrize("backend", ["matmul", "conv", "rnn"])
def test_cpu_fp32_precision_uses_supported_backend_api(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    model = torch.nn.Linear(2, 1)
    first = runtime.model_runtime(model, "cpu")
    target = getattr(torch.backends.mkldnn, backend)
    # monkeypatch restores the prior precision after each backend case.
    monkeypatch.setattr(target, "fp32_precision", "bf16")
    changed = runtime.model_runtime(model, "cpu")
    assert changed["torch"][f"{backend}_fp32_precision"] == "bf16"
    assert first != changed
