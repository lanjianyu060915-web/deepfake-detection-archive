"""Tiny CPU AST behavior tests; no downloaded weights or GPU execution."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from deepfake_detection_archive.models.ast import ASTClassifier, ASTConfig  # noqa: E402
from deepfake_detection_archive.models.checkpoint import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def config():
    return ASTConfig(
        backbone_config={
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 24,
            "max_length": 32,
            "num_mel_bins": 32,
            "patch_size": 16,
            "frequency_stride": 10,
            "time_stride": 10,
            "hidden_dropout_prob": 0.2,
        },
        trainable_layers=1,
    )


def test_constructor_is_offline_and_gradient_scope(config):
    with patch.object(transformers.ASTModel, "from_pretrained") as load:
        model = ASTClassifier(config)
        load.assert_not_called()
    assert model.backbone.config._attn_implementation == "sdpa"
    logits = model(torch.randn(2, 32, 32))
    assert logits.shape == (2,)
    logits.sum().backward()
    for name, parameter in model.named_parameters():
        trainable = name.startswith(
            ("backbone.layers.1.", "backbone.layernorm.", "head.")
        )
        assert parameter.requires_grad == trainable, name
        assert (parameter.grad is not None) == trainable, name
        if trainable:
            assert torch.isfinite(parameter.grad).all(), name
    assert model.head[1].weight.grad.abs().sum() > 0


def test_waveform_and_cached_features_are_equivalent(config):
    model = ASTClassifier(config).eval()
    waveforms = torch.randn(2, 8000)
    features = model.prepare_waveforms(waveforms)
    assert features.shape == (2, 32, 32)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()
    with torch.no_grad():
        torch.testing.assert_close(model(waveforms), model(features))
    preprocessing = json.loads(json.dumps(model.preprocessing))
    assert preprocessing["max_length"] == 32
    assert preprocessing["num_mel_bins"] == 32
    assert preprocessing["mean"] == config.feature_mean
    assert "waveform_preparation" not in preprocessing


@pytest.mark.parametrize("kind", ["long", "short", "silent"])
def test_declared_waveform_policy_center_padding_peak_and_silence(
    config, monkeypatch, kind
):
    model = ASTClassifier(
        replace(config, waveform_preparation="center_pad_64000_peak095_v1")
    )
    monkeypatch.setattr(model, "prepare_waveforms", lambda inputs: inputs)
    expected = torch.zeros(1, 64000)
    if kind == "long":
        waveform = np.zeros(64004, dtype=np.float32)
        waveform[0] = 100  # Outside center crop: must not affect normalization.
        waveform[2:4] = [2, -4]
        expected[0, :2] = torch.tensor([0.475, -0.95])
    elif kind == "short":
        waveform = np.array([2, -4, 1], dtype=np.float32)
        expected[0, 31998:32001] = torch.tensor([0.475, -0.95, 0.2375])
    else:
        waveform = np.zeros(123, dtype=np.float32)
    original = waveform.copy()
    actual = model.prepare_audio(waveform, 64000, 1)
    torch.testing.assert_close(actual, expected)
    np.testing.assert_array_equal(waveform, original)
    assert model.preprocessing["waveform_preparation"] == "center_pad_64000_peak095_v1"


def test_legacy_waveform_policy_keeps_start_and_loop_padding(config, monkeypatch):
    model = ASTClassifier(config)
    monkeypatch.setattr(model, "prepare_waveforms", lambda inputs: inputs)
    torch.testing.assert_close(
        model.prepare_audio(np.arange(8, dtype=np.float32), 4, 1),
        torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
    )
    torch.testing.assert_close(
        model.prepare_audio(np.array([0.1, -0.2], dtype=np.float32), 5, 1),
        torch.tensor([[0.1, -0.2, 0.1, -0.2, 0.1]]),
    )


def test_waveform_policy_validation_and_legacy_checkpoint(config, tmp_path):
    with pytest.raises(ValueError, match="waveform_preparation"):
        replace(config, waveform_preparation="unknown")
    model = ASTClassifier(
        replace(config, waveform_preparation="center_pad_64000_peak095_v1")
    )
    model.validate_audio_sampling(64000, 1)
    for count, segments in [(64001, 1), (64000, 2)]:
        with pytest.raises(ValueError, match="segment_samples=64000 and segments=1"):
            model.validate_audio_sampling(count, segments)
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, ASTClassifier(config))
    payload = torch.load(path, weights_only=True)
    del payload["config"]["waveform_preparation"]
    del payload["config"]["frontend"]
    torch.save(payload, path)
    restored, _ = load_checkpoint(path, expected_sha256=sha256_file(path))
    assert restored.config.waveform_preparation == "legacy"
    assert restored.config.frontend == "numpy_spectrogram_v1"


def test_train_keeps_frozen_blocks_in_eval(config):
    model = ASTClassifier(config)
    for mode in (False, True, False, True):
        model.train(mode)
        assert not model.backbone.embeddings.training
        assert not model.backbone.layers[0].training
        assert model.backbone.layers[1].training == mode
        assert model.backbone.layernorm.training == mode
        assert model.head.training == mode
        assert model.head[0].training == mode


def test_verified_local_pretrained_loading(config, tmp_path: Path):
    upstream = transformers.ASTForAudioClassification(
        transformers.ASTConfig(**config.backbone_config)
    )
    upstream.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    model = ASTClassifier(config)
    head = {key: value.clone() for key, value in model.head.state_dict().items()}
    model.initialize_pretrained(path, sha256_file(path))
    for key, tensor in upstream.audio_spectrogram_transformer.state_dict().items():
        torch.testing.assert_close(model.backbone.state_dict()[key], tensor)
    for key, tensor in head.items():
        torch.testing.assert_close(model.head.state_dict()[key], tensor)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        model.initialize_pretrained(path, "0" * 64)


def test_pretrained_time_crop_preserves_frequency_and_special_tokens(config, tmp_path):
    upstream_config = transformers.ASTConfig(
        **{**config.backbone_config, "max_length": 64, "num_mel_bins": 48}
    )
    upstream = transformers.ASTForAudioClassification(upstream_config)
    original = upstream.audio_spectrogram_transformer.state_dict()
    key = "embeddings.position_embeddings"
    with torch.no_grad():
        original[key].copy_(
            torch.arange(original[key].numel()).reshape_as(original[key])
        )
    upstream.save_pretrained(tmp_path)
    target_config = replace(
        config, backbone_config={**config.backbone_config, "num_mel_bins": 48}
    )
    target = ASTClassifier(target_config).eval()
    path = tmp_path / "model.safetensors"
    target.initialize_pretrained(path, sha256_file(path))
    actual = target.backbone.state_dict()
    # Source is 4 frequencies x 5 times; target keeps central times 1:3 per frequency.
    grid = original[key][:, 2:].reshape(1, 4, 5, 16)
    expected = torch.cat(
        (original[key][:, :2], grid[:, :, 1:3].reshape(1, 8, 16)), dim=1
    )
    torch.testing.assert_close(actual[key], expected)
    for name, tensor in original.items():
        if name != key:
            torch.testing.assert_close(actual[name], tensor)
    with torch.no_grad():
        assert target(torch.randn(2, 32, 48)).shape == (2,)
    larger = ASTClassifier(
        replace(
            target_config,
            backbone_config={**target_config.backbone_config, "max_length": 96},
        )
    )
    with pytest.raises(ValueError, match="can only be shortened"):
        larger.initialize_pretrained(path, sha256_file(path))


def test_numpy_frontend_matches_pre_torchaudio_baseline(config):
    model = ASTClassifier(
        replace(config, backbone_config={**config.backbone_config, "max_length": 48})
    )
    time = np.arange(6400, dtype=np.float32) / 16000
    waveform = (
        0.3 * np.sin(2 * np.pi * 440 * time) + 0.1 * np.cos(2 * np.pi * 1700 * time)
    ).astype(np.float32)
    features = model.prepare_waveforms(torch.from_numpy(waveform[None]))[0].numpy()
    # Recorded before installing torchaudio, using the unchanged HF NumPy path.
    np.testing.assert_allclose(
        features[[0, 1, 10, 37, 38, 47]][:, [0, 5, 17, 31]],
        [
            [-0.86405325, 0.87126446, -0.6507202, -1.2775939],
            [-0.7919193, 0.87126243, -0.65026575, -1.2775939],
            [-0.8641039, 0.87126416, -0.6507184, -1.2662324],
            [-0.7630081, 0.87126195, -0.65034467, -0.99096376],
            [0.46703237] * 4,
            [0.46703237] * 4,
        ],
        rtol=1e-6,
        atol=1e-7,
    )
    assert float(features.mean()) == pytest.approx(-0.4011737, abs=1e-7)
    assert float(features.std()) == pytest.approx(0.7758249, abs=1e-7)
    assert model.preprocessing["frontend"] == "numpy_spectrogram_v1"


def test_explicit_kaldi_frontend_and_checkpoint_identity(config, tmp_path):
    torchaudio = pytest.importorskip("torchaudio")
    with pytest.raises(ValueError, match="frontend"):
        replace(config, frontend="auto")
    numpy_model = ASTClassifier(config)
    kaldi_model = ASTClassifier(replace(config, frontend="torchaudio_kaldi_v1"))
    waveform = torch.sin(torch.arange(6400, dtype=torch.float32) * 0.17)[None]
    numpy_features = numpy_model.prepare_waveforms(waveform)
    kaldi_features = kaldi_model.prepare_waveforms(waveform)
    assert numpy_features.shape == kaldi_features.shape
    assert torch.isfinite(kaldi_features).all()
    expected = torchaudio.compliance.kaldi.fbank(
        waveform, sample_frequency=16000, window_type="hanning", num_mel_bins=32
    )[:32]
    expected = (expected - config.feature_mean) / (config.feature_std * 2)
    torch.testing.assert_close(kaldi_features[0], expected)
    assert numpy_model.preprocessing != kaldi_model.preprocessing
    assert "torchaudio" not in numpy_model.preprocessing["frontend_versions"]
    assert "torchaudio" in kaldi_model.preprocessing["frontend_versions"]
    path = tmp_path / "kaldi.pt"
    save_checkpoint(path, kaldi_model)
    restored, _ = load_checkpoint(path, expected_sha256=sha256_file(path))
    assert restored.config.frontend == "torchaudio_kaldi_v1"
    torch.testing.assert_close(restored.prepare_waveforms(waveform), kaldi_features)
