"""Training-time waveform augmentation that leaves evaluation untouched.

Bandwidth augmentation band-limits a training waveform at a lower Nyquist and
restores the training sample rate; noise augmentation adds white Gaussian noise
at a drawn signal-to-noise ratio. Both are symmetric: every label passes through
the same policy, so the transforms cannot become a label shortcut, and nothing
in this module is used by the offline scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor


@dataclass(frozen=True)
class BandwidthSpec:
    """Probability and target rate of one band-limiting augmentation."""

    probability: float
    target_sample_rate: int

    def __post_init__(self) -> None:
        if isinstance(self.probability, bool) or not isinstance(
            self.probability, (int, float)
        ):
            raise TypeError("bandwidth probability must be a number")
        if not math.isfinite(self.probability) or not 0 < self.probability <= 1:
            raise ValueError("bandwidth probability must lie in (0, 1]")
        if type(self.target_sample_rate) is not int or self.target_sample_rate <= 0:
            raise ValueError("bandwidth target_sample_rate must be a positive integer")


@dataclass(frozen=True)
class NoiseSpec:
    """Probability and uniform SNR range, in dB, of additive white noise."""

    probability: float
    min_snr_db: float
    max_snr_db: float

    def __post_init__(self) -> None:
        values = (self.probability, self.min_snr_db, self.max_snr_db)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        ):
            raise TypeError("noise parameters must be numbers")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("noise parameters must be finite")
        if not 0 < self.probability <= 1:
            raise ValueError("noise probability must lie in (0, 1]")
        if self.min_snr_db > self.max_snr_db:
            raise ValueError("noise min_snr_db must not exceed max_snr_db")


def add_white_noise(
    waveform: Tensor, snr_db: float, generator: torch.Generator
) -> Tensor:
    """Add white Gaussian noise at ``snr_db`` relative to the waveform power.

    A silent waveform has no reference power and is returned unchanged.
    """
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("noise augmentation expects a nonempty mono waveform")
    if not waveform.is_floating_point():
        raise ValueError("noise augmentation expects a floating point waveform")
    signal = waveform.to(torch.float64)
    power = float(signal.square().mean())
    if power == 0.0:
        return waveform
    noise = torch.randn(signal.shape, generator=generator, dtype=torch.float64)
    scale = math.sqrt(power / 10.0 ** (snr_db / 10.0))
    return cast(Tensor, (signal + scale * noise).to(waveform.dtype))


def bandlimited_resample(
    waveform: Tensor, target_rate: int, sample_rate: int
) -> Tensor:
    """Ideal band-limited resampling by spectrum truncation or zero padding.

    Downsampling keeps only the spectrum below the new Nyquist frequency, so no
    energy folds back onto lower frequencies; upsampling restores the length
    without adding content. Keeping the coefficients while changing the inverse
    transform length scales amplitude by ``length / target_length``, so the
    transform rescales to preserve amplitude and stay linear.
    """
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("bandwidth augmentation expects a nonempty mono waveform")
    if not waveform.is_floating_point():
        raise ValueError("bandwidth augmentation expects a floating point waveform")
    for name, value in (
        ("target_rate", target_rate),
        ("sample_rate", sample_rate),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    length = int(waveform.shape[0])
    target_length = max(2, round(length * target_rate / sample_rate))
    return _resample_length(waveform, target_length)


def _resample_length(waveform: Tensor, target_length: int) -> Tensor:
    length = int(waveform.shape[0])
    if target_length == length:
        return waveform
    spectrum = torch.fft.rfft(waveform.to(torch.float64))
    bins = target_length // 2 + 1
    trimmed = spectrum.new_zeros(bins)
    shared = min(bins, int(spectrum.shape[0]))
    trimmed[:shared] = spectrum[:shared]
    resampled = torch.fft.irfft(trimmed, n=target_length)
    resampled *= target_length / length
    return cast(Tensor, resampled.to(waveform.dtype))


def bandlimit_roundtrip(
    waveform: Tensor, spec: BandwidthSpec, sample_rate: int
) -> Tensor:
    """Downsample to the target rate and back to the training rate."""
    if spec.target_sample_rate >= sample_rate:
        raise ValueError("bandwidth augmentation target rate must be below the rate")
    reduced = bandlimited_resample(waveform, spec.target_sample_rate, sample_rate)
    return _resample_length(reduced, waveform.numel())
