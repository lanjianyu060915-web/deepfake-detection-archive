"""Tiny CPU AST behavior tests; no downloaded weights or GPU execution."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from deepfake_detection_archive.models.ast import ASTClassifier, ASTConfig  # noqa: E402
from deepfake_detection_archive.models.checkpoint import sha256_file  # noqa: E402


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
