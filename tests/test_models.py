"""CPU behavioral tests; synthetic inputs establish correctness, not detection EER."""

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from deepfake_detection_archive.models import (  # noqa: E402
    AASISTLite,
    FeatureHead,
    FrozenAudioEncoder,
    FrozenVisionEncoder,
    HeadConfig,
    LogMelCNN,
    LogMelConfig,
    MediaFeatureClassifier,
    MediaFeatureConfig,
    create_model,
    load_checkpoint,
    pool_media_scores,
    save_checkpoint,
    sha256_file,
)


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_logmel_masks_padding_and_backpropagates():
    torch.manual_seed(42)
    model = LogMelCNN(LogMelConfig(channels=4, hidden_dim=8, dropout=0)).eval()
    signal = torch.randn(1, 1701)
    reference = model(signal)
    batch = torch.zeros(2, 3107)
    batch[0, :1701] = signal[0]
    batch[0, 1701:] = float("nan")  # Padding cannot contaminate valid frames.
    batch[1] = torch.randn(3107)
    actual = model(batch, torch.tensor([1701, 3107]))
    torch.testing.assert_close(actual[:1], reference, atol=1e-6, rtol=1e-5)
    actual.sum().backward()
    assert model.temporal.weight.grad is not None
    assert torch.isfinite(model.temporal.weight.grad).all()
    assert model.temporal.weight.grad.abs().sum() > 0


def test_logmel_silence_short_input_and_validation():
    model = LogMelCNN().eval()
    assert torch.isfinite(model(torch.zeros(2, 1))).all()
    for value, lengths in [
        (torch.empty(1, 0), None),
        (torch.zeros(1, 100), torch.tensor([0])),
        (torch.zeros(1, 100), torch.tensor([101])),
        (torch.zeros(1, 100), torch.tensor([100.0])),
        (torch.full((1, 100), float("inf")), None),
    ]:
        with pytest.raises(ValueError, match=r"waveforms|lengths|finite"):
            model(value, lengths)
    with pytest.raises(ValueError, match="empty mel"):
        LogMelCNN(LogMelConfig(n_fft=16, hop_length=8, n_mels=64))


@pytest.mark.parametrize("method", ["mean", "median", "logsumexp", "trimmed_mean"])
def test_media_pooling_mask(method):
    scores = torch.tensor([[1.0, 3.0, float("nan")], [4.0, 4.0, 4.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    actual = pool_media_scores(scores, mask, method)
    expected = torch.tensor([2.0, 4.0])
    if method == "logsumexp":
        expected[0] = (
            torch.logsumexp(torch.tensor([1.0, 3.0]), 0) - torch.tensor(2.0).log()
        )
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="at least one"):
        pool_media_scores(scores, torch.zeros_like(mask), method)
    with pytest.raises(ValueError, match="finite"):
        pool_media_scores(scores)


def test_trimmed_mean_and_invalid_pooling():
    scores = torch.tensor([[-100.0, 1.0, 2.0, 3.0, 200.0]])
    assert (
        pool_media_scores(scores, method="trimmed_mean", trim_fraction=0.2).item() == 2
    )
    with pytest.raises(ValueError, match="unsupported"):
        pool_media_scores(scores, method="invented")


@pytest.mark.parametrize("method", ["mean", "logsumexp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_vectorized_pooling_matches_reference_values_and_gradients(method, dtype):
    torch.manual_seed(78)
    mask = torch.tensor([[True, False, True, False], [True, True, True, True]])
    values = torch.randn(2, 4, dtype=dtype).masked_fill(~mask, float("nan"))
    actual_inputs = values.clone().requires_grad_()
    reference_inputs = values.clone().requires_grad_()
    actual = pool_media_scores(actual_inputs, mask, method=method)
    reference_rows = []
    for row, valid in zip(reference_inputs, mask, strict=True):
        selected = row[valid]
        reference_rows.append(
            selected.mean()
            if method == "mean"
            else torch.logsumexp(selected, 0)
            - selected.new_tensor(selected.numel()).log()
        )
    reference = torch.stack(reference_rows)
    torch.testing.assert_close(actual, reference)
    weights = torch.tensor([0.25, -0.7], dtype=dtype)
    (actual * weights).sum().backward()
    (reference * weights).sum().backward()
    torch.testing.assert_close(actual_inputs.grad, reference_inputs.grad)
    assert (actual_inputs.grad[~mask] == 0).all()


@pytest.mark.parametrize("kind", ["head", "logmel", "aasist", "media"])
def test_checkpoint_round_trip(tmp_path, kind):
    torch.manual_seed(12)
    if kind == "head":
        model = FeatureHead(HeadConfig(8, 4, 0)).eval()
        inputs = torch.randn(3, 8)
    elif kind == "logmel":
        model = LogMelCNN(LogMelConfig(channels=4)).eval()
        inputs = torch.randn(2, 1000)
    elif kind == "media":
        model = MediaFeatureClassifier(
            MediaFeatureConfig(8, 4, 0, aggregation="logsumexp")
        ).eval()
        inputs = torch.randn(2, 3, 8)
    else:
        model = AASISTLite().eval()
        inputs = torch.randn(1, 64600)
    with torch.no_grad():
        expected = model(inputs)
    path = tmp_path / "model.pt"
    metadata = save_checkpoint(path, model)
    restored, loaded_metadata = load_checkpoint(
        path, expected_sha256=metadata["sha256"]
    )
    assert loaded_metadata == metadata
    assert path.with_suffix(".pt.json").is_file()
    assert restored.training is False
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint(path, expected_sha256="0" * 64)


def test_checkpoint_rejects_unknown_and_nonfinite(tmp_path):
    model = FeatureHead(HeadConfig(8))
    path = tmp_path / "bad.pt"
    save_checkpoint(path, model)
    payload = torch.load(path, weights_only=True)
    payload["architecture"] = "arbitrary.module.Class"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported architecture"):
        load_checkpoint(path, expected_sha256=sha256_file(path))
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        save_checkpoint(path, model)


@pytest.mark.parametrize("aggregation", ["mean", "median", "logsumexp", "trimmed_mean"])
def test_media_feature_classifier_aggregates_with_media_gradients(aggregation):
    model = MediaFeatureClassifier(
        MediaFeatureConfig(8, 4, 0, aggregation=aggregation, trim_fraction=0.2)
    ).eval()
    features = torch.randn(2, 5, 8, requires_grad=True)
    actual = model(features)
    expected = pool_media_scores(
        model.head(features), method=aggregation, trim_fraction=0.2
    )
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model(features[:, [4, 0, 2, 3, 1]]), actual)
    torch.nn.functional.binary_cross_entropy_with_logits(
        actual, torch.tensor([0.0, 1.0])
    ).backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0


def test_media_feature_classifier_masks_and_validates_observations():
    model = MediaFeatureClassifier(MediaFeatureConfig(8, 4, 0)).eval()
    features = torch.randn(2, 5, 8)
    lengths = torch.tensor([2, 5])
    features[0, 2:] = float("nan")
    actual = model(features, lengths)
    torch.testing.assert_close(actual[:1], model(features[:1, :2]))
    with pytest.raises(ValueError, match="finite"):
        model(features)
    for invalid in (
        torch.tensor([0, 5]),
        torch.tensor([2, 6]),
        torch.tensor([2.0, 5.0]),
    ):
        with pytest.raises(ValueError, match="lengths"):
            model(features, invalid)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 8))
    with pytest.raises(ValueError, match="aggregation"):
        create_model("media_feature_head_v1", {"input_dim": 8, "aggregation": "max"})
    with pytest.raises(TypeError, match="unexpected keyword"):
        create_model("media_feature_head_v1", {"input_dim": 8, "unknown": 42})


def test_legacy_media_checkpoint_keeps_default_mlp(tmp_path):
    model = MediaFeatureClassifier(MediaFeatureConfig(8, 4, 0)).eval()
    reference_head = FeatureHead(HeadConfig(8, 4, 0)).eval()
    reference_head.load_state_dict(model.head.state_dict(), strict=True)
    features = torch.randn(2, 4, 8)
    expected = reference_head(features).mean(dim=1)
    torch.testing.assert_close(model(features), expected, atol=0, rtol=0)
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(checkpoint, model)
    payload = torch.load(checkpoint, weights_only=True)
    del payload["config"]["temporal_kernel_size"]
    torch.save(payload, checkpoint)
    loaded, _ = load_checkpoint(checkpoint, expected_sha256=sha256_file(checkpoint))
    assert loaded.config.temporal_kernel_size == 0
    assert list(model.state_dict()) == list(loaded.state_dict())
    torch.testing.assert_close(loaded(features), expected, atol=0, rtol=0)


def test_temporal_media_head_backpropagates_and_round_trips(tmp_path):
    model = MediaFeatureClassifier(
        MediaFeatureConfig(768, 4, 0, temporal_kernel_size=3)
    ).eval()
    features = torch.randn(2, 4, 768, requires_grad=True)
    logits = model(features, torch.tensor([2, 4]))
    assert logits.shape == (2,)
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 1.0])
    ).backward()
    for parameter in model.head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert features.grad is not None
    assert (features.grad[0, 2:] == 0).all()
    path = tmp_path / "temporal.pt"
    descriptor = save_checkpoint(path, model)
    restored, _ = load_checkpoint(path, expected_sha256=descriptor["sha256"])
    assert restored.config.temporal_kernel_size == 3
    torch.testing.assert_close(
        restored(features, torch.tensor([2, 4])), logits, atol=0, rtol=0
    )


@pytest.mark.parametrize("aggregation", ["mean", "median", "logsumexp", "trimmed_mean"])
@pytest.mark.parametrize("length", [1, 2, 5])
def test_temporal_media_head_ignores_padding_after_layernorm(aggregation, length):
    model = MediaFeatureClassifier(
        MediaFeatureConfig(8, 4, 0, aggregation=aggregation, temporal_kernel_size=3)
    ).eval()
    # Trained LayerNorm bias must not become a neighboring padded observation.
    with torch.no_grad():
        model.head.norm.bias.fill_(2)
    features = torch.randn(2, 7, 8)
    expected = model(features[:1, :length])
    lengths = torch.tensor([length, 7])
    actual = model(features, lengths)
    torch.testing.assert_close(actual[:1], expected)
    features[0, length:] = float("nan")
    torch.testing.assert_close(model(features, lengths), actual)


@pytest.mark.parametrize("kernel_size", [1, 2, 5, True, 3.0])
def test_media_temporal_kernel_size_is_validated(kernel_size):
    with pytest.raises(ValueError, match="temporal_kernel_size"):
        MediaFeatureConfig(8, temporal_kernel_size=kernel_size)


def test_aasist_structure_direction_and_upstream_equivalence():
    torch.manual_seed(3)
    model = AASISTLite().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 85306
    waveform = torch.randn(1, 64600)
    with torch.no_grad():
        _, logits = model.backbone(waveform)
        torch.testing.assert_close(model(waveform), logits[:, 0] - logits[:, 1])
    upstream_path = Path("external/aasist/models/AASIST.py")
    if upstream_path.is_file():
        spec = importlib.util.spec_from_file_location(
            "upstream_aasist_reference", upstream_path
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = module.Model(model.backbone.d_args).eval()
        reference.load_state_dict(model.backbone.state_dict())
        with torch.no_grad():
            _, expected = reference(waveform)
        torch.testing.assert_close(logits, expected, atol=0, rtol=0)


def test_aasist_repeat_pad_ignores_batch_padding():
    model = AASISTLite()
    prepared = model.prepare_input(torch.tensor([[1.0, 2.0, 99.0]]), torch.tensor([2]))
    torch.testing.assert_close(
        prepared[0, :6], torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    )
    assert prepared.shape == (1, 64600)


def test_local_vision_encoder_is_frozen(tmp_path, monkeypatch):
    timm = pytest.importorskip("timm")
    model = timm.create_model("resnet18", pretrained=False, num_classes=0).eval()
    path = tmp_path / "resnet18.pt"
    torch.save(model.state_dict(), path)
    constructor = timm.create_model
    calls = []

    def create_local(name, **kwargs):
        calls.append(kwargs)
        assert kwargs["pretrained"] is False
        return constructor(name, **kwargs)

    monkeypatch.setattr(timm, "create_model", create_local)
    encoder = FrozenVisionEncoder("resnet18", path, expected_sha256=sha256_file(path))
    encoder.train()
    assert calls
    assert not encoder.training
    assert not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    images = torch.randn(2, 3, 32, 32)
    features = encoder(images)
    assert features.shape == (2, 512)
    assert not features.requires_grad
    torch.testing.assert_close(features, model(images))
    with pytest.raises(ValueError, match="unsupported"):
        FrozenVisionEncoder(
            "hf-hub:any/remote", path, expected_sha256=sha256_file(path)
        )


def test_local_dinov2_preserves_position_grid_and_uses_224_crops(tmp_path, monkeypatch):
    timm = pytest.importorskip("timm")
    from timm.models.vision_transformer import VisionTransformer  # noqa: PLC0415

    architecture = "vit_base_patch14_dinov2.lvd142m"
    assert timm.is_model(architecture)
    pretrained_cfg = timm.models.get_pretrained_cfg(architecture)

    def tiny_model():
        model = VisionTransformer(
            img_size=518,
            patch_size=14,
            embed_dim=32,
            depth=1,
            num_heads=2,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.pretrained_cfg = pretrained_cfg.to_dict()
        return model.eval()

    reference = tiny_model()
    checkpoint = tmp_path / "dinov2.pt"
    torch.save(reference.state_dict(), checkpoint)
    calls = []

    def create_local(name, **kwargs):
        calls.append((name, kwargs))
        return tiny_model()

    monkeypatch.setattr(timm, "create_model", create_local)
    encoder = FrozenVisionEncoder(
        architecture, checkpoint, expected_sha256=sha256_file(checkpoint)
    )
    assert calls == [
        (
            architecture,
            {"pretrained": False, "num_classes": 0, "dynamic_img_size": True},
        )
    ]
    assert encoder.backbone.pos_embed.shape == (1, 1370, 32)
    assert encoder.preprocessing["input_size"] == (3, 224, 224)
    assert encoder.preprocessing["resize_mode"] == "native_center_crop"
    assert encoder.preprocessing["mean"] == (0.485, 0.456, 0.406)
    assert encoder.preprocessing["std"] == (0.229, 0.224, 0.225)
    encoder.train()
    assert not encoder.training
    assert not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    images = torch.randn(2, 3, 224, 224)
    features = encoder(images)
    assert features.shape == (2, encoder.feature_dim)
    assert encoder.feature_dim == 32
    assert not features.requires_grad
    torch.testing.assert_close(features, reference(images))
    state = reference.state_dict()
    del state["pos_embed"]
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        FrozenVisionEncoder(
            architecture, checkpoint, expected_sha256=sha256_file(checkpoint)
        )


@pytest.mark.parametrize("feature_norm", ["layer", "group"])
def test_local_ssl_encoder_and_bundle_integrity(tmp_path, monkeypatch, feature_norm):
    transformers = pytest.importorskip("transformers")
    config = transformers.Wav2Vec2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(8, 8),
        conv_stride=(5, 2),
        conv_kernel=(10, 3),
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2,
        feat_extract_norm=feature_norm,
        mask_time_prob=0.0,
    )
    model = transformers.Wav2Vec2Model(config).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    transformers.Wav2Vec2FeatureExtractor().save_pretrained(tmp_path)
    hashes = {
        path.name: sha256_file(path) for path in tmp_path.iterdir() if path.is_file()
    }
    constructor = transformers.AutoModel.from_pretrained

    def load_local(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["use_safetensors"] is True
        return constructor(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_local)
    encoder = FrozenAudioEncoder(tmp_path, expected_hashes=hashes)
    encoder.train()
    assert not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    waveform = torch.randn(1, 1000)
    padded = torch.cat((waveform, torch.full((1, 77), float("nan"))), dim=1)
    features = encoder(waveform)
    assert features.shape == (1, 16)
    assert not features.requires_grad
    torch.testing.assert_close(encoder(padded, torch.tensor([1000])), features)
    batch = torch.cat((waveform, torch.randn_like(waveform) * 7 + 3))
    expected = torch.cat([encoder(row[None, :]) for row in batch])
    batch_sizes = []
    forward = encoder.backbone.forward

    def record_batch(*args, **kwargs):
        batch_sizes.append(kwargs["input_values"].shape[0])
        return forward(*args, **kwargs)

    monkeypatch.setattr(encoder.backbone, "forward", record_batch)
    torch.testing.assert_close(encoder(batch), expected, rtol=1e-4, atol=1e-5)
    assert batch_sizes == [2]
    shorter = encoder(batch[1:2, :768])
    batch[1, 768:] = float("nan")
    batch_sizes.clear()
    torch.testing.assert_close(
        encoder(batch, torch.tensor([1000, 768])),
        torch.cat((features, shorter)),
        rtol=1e-4,
        atol=1e-5,
    )
    assert batch_sizes == [1, 1]
    with pytest.raises(ValueError, match="sample rate"):
        encoder(waveform, sample_rate=22050)
    with pytest.raises(ValueError, match="hashes must cover"):
        FrozenAudioEncoder(tmp_path, expected_hashes={})
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        FrozenAudioEncoder(tmp_path, expected_hashes=hashes)
