"""Synthetic training-contract checks, not detection-quality measurements."""

import copy
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import Tensor

from deepfake_detection_archive.models.checkpoint import load_checkpoint
from deepfake_detection_archive.models.core import FeatureHead, HeadConfig
from deepfake_detection_archive.training import TrainConfig, train_model


PROVENANCE = dict.fromkeys(("data_sha256", "code_sha256", "weights_sha256"), "a" * 64)


def model() -> FeatureHead:
    torch.manual_seed(123)
    return FeatureHead(HeadConfig(input_dim=3, hidden_dim=4, dropout=0))


def batches() -> tuple[Tensor, Tensor]:
    inputs = torch.tensor(
        [
            [1.0, 0.2, -0.3],
            [-1.0, 0.3, 1.0],
            [0.2, 1.0, -0.9],
            [0.9, -1.0, 0.3],
            [-0.3, 0.2, 1.0],
            [0.1, 0.8, -0.4],
        ]
    )
    return inputs, torch.tensor([0, 1, 0, 1, 1, 0])


def test_uneven_accumulation_and_tail_match_concatenated_batches(
    tmp_path: Path,
) -> None:
    inputs, labels = batches()
    accumulated, concatenated = model(), model()
    small = [
        (inputs[:1], labels[:1]),
        (inputs[1:4], labels[1:4]),
        (inputs[4:], labels[4:]),
    ]
    large = [(inputs[:4], labels[:4]), (inputs[4:], labels[4:])]
    dev = [(inputs.clone(), labels.clone())]
    first = train_model(
        accumulated,
        small,
        dev,
        config=TrainConfig(max_epochs=1, accumulation_steps=2, weight_decay=0),
        output_dir=tmp_path / "accumulated",
        provenance=PROVENANCE,
    )
    train_model(
        concatenated,
        large,
        dev,
        config=TrainConfig(max_epochs=1, weight_decay=0),
        output_dir=tmp_path / "concatenated",
        provenance=PROVENANCE,
    )
    for actual, expected in zip(
        accumulated.parameters(), concatenated.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        assert torch.isfinite(actual).all()
    history = json.loads(first.history_path.read_text())
    assert history[0]["optimizer_updates"] == 2
    assert history[0]["train_samples"] == 6


def test_dev_labels_do_not_change_gradient_updates_and_best_is_reloaded(
    tmp_path: Path,
) -> None:
    inputs, labels = batches()
    initial, first, second = model(), model(), model()
    config = TrainConfig(max_epochs=1)
    result = train_model(
        first,
        [(inputs, labels)],
        [(inputs, labels)],
        config=config,
        output_dir=tmp_path / "first",
        provenance=PROVENANCE,
    )
    train_model(
        second,
        [(inputs, labels)],
        [(inputs, 1 - labels)],
        config=config,
        output_dir=tmp_path / "second",
        provenance=PROVENANCE,
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(initial.parameters(), first.parameters(), strict=True)
    )
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    restored, _ = load_checkpoint(
        result.best_checkpoint, expected_sha256=result.checkpoint_sha256
    )
    assert not first.training
    with torch.inference_mode():
        torch.testing.assert_close(first(inputs), restored(inputs), atol=0, rtol=0)
    metadata = json.loads(result.run_path.read_text())
    assert metadata["selection_split"] == "dev"
    assert metadata["provenance"] == PROVENANCE
    assert metadata["config"] == asdict(config)
    assert metadata["peak_cuda_allocated_bytes"] == 0
    assert "holdout" not in metadata


def test_seed_reproducibility_and_early_stopping(tmp_path: Path) -> None:
    inputs, labels = batches()
    results = [
        train_model(
            model(),
            [(inputs, labels)],
            [(inputs, labels)],
            config=TrainConfig(max_epochs=10, patience=1, min_delta=1.0),
            output_dir=tmp_path / name,
            provenance=PROVENANCE,
        )
        for name in ("one", "two")
    ]
    assert results[0].epochs_completed == 2
    assert results[0].stop_reason == "early_stopping"
    histories = [json.loads(result.history_path.read_text()) for result in results]
    for history in histories:
        for row in history:
            row.pop("seconds")
            row.pop("train_seconds")
            row.pop("train_samples_per_second")
    assert histories[0] == histories[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_epochs": True},
        {"accumulation_steps": 0},
        {"patience": -1},
        {"learning_rate": float("nan")},
        {"weight_decay": -1},
        {"min_delta": True},
        {"amp": 1},
        {"amp": True},
        {"amp_dtype": "float32"},
        {"amp_dtype": True},
        {"amp": True, "amp_dtype": "bfloat16", "device": "cpu"},
        {"seed": -1},
        {"device": "mps"},
        {"max_seconds": 0},
        {"max_grad_norm": float("inf")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"must|range|supported"):
        TrainConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_amp_dtype_configuration_and_disabled_cpu_path(
    dtype: str, tmp_path: Path
) -> None:
    config = TrainConfig(max_epochs=1, amp_dtype=dtype)  # type: ignore[arg-type]
    result = train_model(
        model(),
        [batches()],
        [batches()],
        config=config,
        output_dir=tmp_path,
        provenance=PROVENANCE,
    )
    metadata = json.loads(result.run_path.read_text())
    assert metadata["grad_scaler_enabled"] is False
    assert metadata["config"]["amp_dtype"] == dtype
    assert TrainConfig().amp_dtype == "float16"
    assert TrainConfig(device="cuda", amp=True, amp_dtype=dtype).amp is True  # type: ignore[arg-type]


def test_unsupported_native_bfloat16_is_rejected_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    def unsupported(*, including_emulation: bool) -> bool:
        assert including_emulation is False
        return False

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", unsupported)
    with pytest.raises(RuntimeError, match="does not support native bfloat16"):
        train_model(
            model(),
            [batches()],
            [batches()],
            config=TrainConfig(device="cuda", amp=True, amp_dtype="bfloat16"),
            output_dir=tmp_path / "unsupported",
            provenance=PROVENANCE,
        )
    assert not (tmp_path / "unsupported").exists()


def test_failures_are_recorded_without_checkpoint_or_silent_samples(
    tmp_path: Path,
) -> None:
    inputs, labels = batches()
    inputs[1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        train_model(
            model(),
            [(inputs, labels)],
            [batches()],
            config=TrainConfig(max_epochs=1),
            output_dir=tmp_path / "bad",
            provenance=PROVENANCE,
        )
    metadata = json.loads((tmp_path / "bad" / "run.json").read_text())
    assert metadata["status"] == "failed"
    assert not (tmp_path / "bad" / "best.pt").exists()


def test_split_identity_provenance_and_existing_output_are_rejected(
    tmp_path: Path,
) -> None:
    batch = [batches()]
    with pytest.raises(ValueError, match="distinct"):
        train_model(
            model(),
            batch,
            batch,
            config=TrainConfig(),
            output_dir=tmp_path,
            provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="provenance"):
        train_model(
            model(),
            batch,
            copy.copy(batch),
            config=TrainConfig(),
            output_dir=tmp_path,
            provenance={},
        )
    (tmp_path / "existing").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        train_model(
            model(),
            batch,
            copy.copy(batch),
            config=TrainConfig(),
            output_dir=tmp_path,
            provenance=PROVENANCE,
        )
    assert (tmp_path / "existing").read_text() == "keep"


def test_time_budget_and_missing_dev_class(tmp_path: Path) -> None:
    inputs, labels = batches()
    result = train_model(
        model(),
        [(inputs, labels)],
        [(inputs, labels)],
        config=TrainConfig(max_epochs=10, max_seconds=1e-12),
        output_dir=tmp_path / "limited",
        provenance=PROVENANCE,
    )
    assert result.epochs_completed == 1
    assert result.stop_reason == "time_limit"
    with pytest.raises(ValueError, match="both real and fake"):
        train_model(
            model(),
            [(inputs, labels)],
            [(inputs, torch.zeros_like(labels))],
            config=TrainConfig(max_epochs=1),
            output_dir=tmp_path / "missing",
            provenance=PROVENANCE,
        )


def test_development_forward_is_eval_without_gradients(tmp_path: Path) -> None:
    candidate = model()
    modes: list[tuple[bool, bool]] = []

    def observe(module: torch.nn.Module, inputs: tuple[Tensor, ...]) -> None:
        modes.append((module.training, torch.is_grad_enabled()))

    handle = candidate.register_forward_pre_hook(observe)
    train_model(
        candidate,
        [batches()],
        [batches()],
        config=TrainConfig(max_epochs=1),
        output_dir=tmp_path,
        provenance=PROVENANCE,
    )
    handle.remove()
    assert modes == [(True, True), (False, False)]


def test_min_delta_controls_patience_but_saves_lowest_eer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = iter([0.4, 0.3, 0.35])

    def fixed_eer(*args: object) -> tuple[float, float, int]:
        return 0.5, next(values), 6

    monkeypatch.setattr("deepfake_detection_archive.training._evaluate", fixed_eer)
    result = train_model(
        model(),
        [batches()],
        [batches()],
        config=TrainConfig(max_epochs=3, min_delta=0.2),
        output_dir=tmp_path,
        provenance=PROVENANCE,
    )
    assert result.best_epoch == 2
    assert result.best_dev_eer == 0.3


def test_nonfinite_cpu_gradients_fail_and_restore_deterministic_state(
    tmp_path: Path,
) -> None:
    candidate = model()
    previous = torch.are_deterministic_algorithms_enabled()
    handle = next(candidate.parameters()).register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    with pytest.raises(ValueError, match="non-finite training gradient"):
        train_model(
            candidate,
            [batches()],
            [batches()],
            config=TrainConfig(max_epochs=1, deterministic=not previous),
            output_dir=tmp_path,
            provenance=PROVENANCE,
        )
    handle.remove()
    assert torch.are_deterministic_algorithms_enabled() == previous
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(("time_width", "frequency_width"), [(8, 0), (0, 6), (8, 6)])
def test_fbank_masking_is_independent_repeatable_and_nonmutating(
    time_width: int, frequency_width: int
) -> None:
    from deepfake_detection_archive.training import _mask_fbank  # noqa: PLC0415

    inputs = torch.ones(16, 12, 10)
    torch.manual_seed(12)
    actual = _mask_fbank(inputs, time_width, frequency_width, 0.4669)
    torch.manual_seed(12)
    repeated = _mask_fbank(inputs, time_width, frequency_width, 0.4669)
    assert torch.equal(actual, repeated)
    assert torch.equal(inputs, torch.ones_like(inputs))
    masked = actual == 0.4669
    assert masked.any()
    assert not all(torch.equal(masked[0], row) for row in masked[1:])
    if not frequency_width:
        assert torch.equal(masked, masked[:, :, :1].expand_as(masked))
        counts = masked[:, :, 0].sum(dim=1)
        assert (counts < time_width).all()
    if not time_width:
        assert torch.equal(masked, masked[:, :1, :].expand_as(masked))
        counts = masked[:, 0, :].sum(dim=1)
        assert (counts < frequency_width).all()


def test_disabled_fbank_masking_does_not_consume_rng() -> None:
    from deepfake_detection_archive.training import _mask_fbank  # noqa: PLC0415

    inputs = torch.ones(2, 12, 10)
    before = torch.get_rng_state().clone()
    assert _mask_fbank(inputs, 0, 0, 0.0) is inputs
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("name", ["time_mask_width", "frequency_mask_width"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_mask_width_is_rejected(name: str, value: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        TrainConfig(**{name: value})  # type: ignore[arg-type]


def test_fbank_masking_rejects_other_models_before_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only with audio_ast_v1"):
        train_model(
            model(),
            [batches()],
            [batches()],
            config=TrainConfig(time_mask_width=1),
            output_dir=tmp_path / "unsupported",
            provenance=PROVENANCE,
        )
    assert not (tmp_path / "unsupported").exists()


@pytest.mark.parametrize("name", ["time_mask_width", "frequency_mask_width"])
def test_fbank_masking_rejects_oversize_width_before_output(
    tmp_path: Path, name: str
) -> None:
    from deepfake_detection_archive.models.ast import (  # noqa: PLC0415
        ASTClassifier,
        ASTConfig,
    )

    candidate = ASTClassifier(
        ASTConfig(
            backbone_config={
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 24,
                "max_length": 32,
                "num_mel_bins": 32,
            },
            trainable_layers=1,
        )
    )
    with pytest.raises(ValueError, match="must not exceed AST"):
        train_model(
            candidate,
            [],
            [],
            config=TrainConfig(**{name: 33}),  # type: ignore[arg-type]
            output_dir=tmp_path / "oversize",
            provenance=PROVENANCE,
        )
    assert not (tmp_path / "oversize").exists()
