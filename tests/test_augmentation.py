"""Bandwidth augmentation is train-only, symmetric and shape preserving."""

import math
import random
from pathlib import Path

import pytest
import torch

from deepfake_detection_archive.augmentation import (
    BandwidthSpec,
    bandlimit_roundtrip,
    bandlimited_resample,
)
from deepfake_detection_archive.experiments import _CachedDataset


def _tone(frequency: float, samples: int = 64_600, rate: int = 16_000) -> torch.Tensor:
    time = torch.arange(samples, dtype=torch.float64) / rate
    return torch.cos(2.0 * math.pi * frequency * time).float()


@pytest.mark.parametrize("frequency", [100.0, 1000.0, 3000.0, 3900.0])
def test_passband_tone_survives_the_roundtrip(frequency: float) -> None:
    tone = _tone(frequency)
    result = bandlimit_roundtrip(tone, BandwidthSpec(1.0, 8000), 16_000)
    assert result.shape == tone.shape
    assert result.dtype == tone.dtype
    assert float(result.std()) == pytest.approx(float(tone.std()), rel=1e-3)


def test_target_nyquist_tone_is_retained() -> None:
    """A tone exactly at the target Nyquist sits on the passband edge."""
    tone = _tone(4000.0)
    result = bandlimit_roundtrip(tone, BandwidthSpec(1.0, 8000), 16_000)
    assert float(result.std()) == pytest.approx(float(tone.std()), rel=1e-3)


@pytest.mark.parametrize(
    ("frequency", "limit"),
    [(4100.0, 0.02), (5000.0, 0.01), (6000.0, 0.005), (7500.0, 0.01)],
)
def test_stopband_tone_is_suppressed(frequency: float, limit: float) -> None:
    """Above the target Nyquist only spectral leakage survives the transform."""
    tone = _tone(frequency)
    result = bandlimit_roundtrip(tone, BandwidthSpec(1.0, 8000), 16_000)
    assert float(result.std()) < limit * float(tone.std())


def test_resampling_preserves_amplitude_and_length() -> None:
    tone = _tone(1000.0)
    reduced = bandlimited_resample(tone, 8000, 16_000)
    assert reduced.shape == (tone.shape[0] // 2,)
    assert float(reduced.std()) == pytest.approx(float(tone.std()), rel=1e-3)
    restored = bandlimited_resample(reduced, 16_000, 8000)
    assert restored.shape == tone.shape


def test_spec_and_inputs_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="probability"):
        BandwidthSpec(probability=0.0, target_sample_rate=8000)
    with pytest.raises(ValueError, match="probability"):
        BandwidthSpec(probability=1.5, target_sample_rate=8000)
    with pytest.raises(TypeError, match="number"):
        BandwidthSpec(probability=True, target_sample_rate=8000)
    with pytest.raises(ValueError, match="target_sample_rate"):
        BandwidthSpec(probability=0.5, target_sample_rate=0)
    with pytest.raises(ValueError, match="target rate"):
        bandlimit_roundtrip(_tone(1000.0), BandwidthSpec(0.5, 16_000), 16_000)
    with pytest.raises(ValueError, match="nonempty mono"):
        bandlimited_resample(torch.zeros(2, 8), 8000, 16_000)
    with pytest.raises(ValueError, match="floating point"):
        bandlimited_resample(torch.zeros(8, dtype=torch.int64), 8000, 16_000)


def test_cached_dataset_augments_train_only(tmp_path: Path) -> None:
    random.seed(0)
    entries: list[tuple[Path, str, int]] = []
    for index, label in enumerate((0, 1)):
        path = tmp_path / f"{index}.pt"
        torch.save(_tone(6000.0 + index * 500.0, samples=4000), path)
        entries.append((path, "unused", label))
    plain = _CachedDataset(entries)
    augmented = _CachedDataset(entries, BandwidthSpec(1.0, 8000), 16_000)
    for index, (path, _, label) in enumerate(entries):
        unchanged, plain_label = plain[index]
        changed, changed_label = augmented[index]
        cached = torch.load(path, weights_only=True)
        assert plain_label == changed_label == float(label)
        assert torch.equal(unchanged, cached)
        assert changed.shape == unchanged.shape
        assert float(changed.std()) < 0.01 * float(cached.std())
