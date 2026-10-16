"""Small, auditable classifiers and media-level score aggregation.

All classifier outputs are logits: larger values mean more likely fake. They
are not calibrated probabilities. No constructor fetches pretrained weights.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_waveforms(waveforms: Tensor, lengths: Tensor | None) -> Tensor:
    """Validate a right-padded batch and return its actual sample lengths."""
    if waveforms.ndim != 2 or min(waveforms.shape) == 0:
        raise ValueError("waveforms must have nonempty shape [batch, samples]")
    if not waveforms.is_floating_point():
        raise ValueError("waveforms must be floating point")
    if lengths is None:
        lengths = torch.full(
            (waveforms.shape[0],),
            waveforms.shape[1],
            dtype=torch.long,
            device=waveforms.device,
        )
    if lengths.shape != (waveforms.shape[0],) or lengths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("lengths must be an integer vector matching the batch")
    lengths = lengths.to(waveforms.device)
    if bool(((lengths < 1) | (lengths > waveforms.shape[1])).any()):
        raise ValueError("lengths must lie between 1 and padded sample count")
    valid = torch.arange(waveforms.shape[1], device=waveforms.device)[None, :]
    if not bool(
        torch.isfinite(waveforms.masked_select(valid < lengths[:, None])).all()
    ):
        raise ValueError("valid waveform samples must be finite")
    return lengths


@dataclass(frozen=True)
class HeadConfig:
    input_dim: int
    hidden_dim: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        _positive_integer(self.input_dim, "input_dim")
        _positive_integer(self.hidden_dim, "hidden_dim")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


class FeatureHead(nn.Module):
    """MLP shared by cached audio embeddings and individual video frames."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim < 2 or features.shape[-1] != self.config.input_dim:
            raise ValueError("features must have shape [..., input_dim]")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must be finite")
        result: Tensor = self.layers(features).squeeze(-1)
        return result


class TemporalHead(nn.Module):
    """Contextual frame logits from chronological [batch, frames, features]."""

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.layers = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, features: Tensor, mask: Tensor | None = None) -> Tensor:
        normalized = self.norm(features)
        if mask is not None:
            normalized = normalized.masked_fill(~mask[:, :, None], 0)
        normalized = normalized.transpose(1, 2)
        logits: Tensor = self.layers(normalized).squeeze(1)
        return logits


@dataclass(frozen=True)
class LogMelConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 64
    channels: int = 24
    hidden_dim: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "sample_rate",
            "n_fft",
            "hop_length",
            "n_mels",
            "channels",
            "hidden_dim",
        ):
            _positive_integer(getattr(self, name), name)
        if self.n_fft < 16 or self.n_mels < 4:
            raise ValueError("n_fft >= 16 and n_mels >= 4 are required")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must not exceed n_fft")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


def _mel_filterbank(config: LogMelConfig) -> Tensor:
    # HTK mel scale; triangles are evaluated at exact FFT-bin frequencies.
    upper = 2595.0 * torch.log10(torch.tensor(1.0 + config.sample_rate / 1400.0))
    mels = torch.linspace(0.0, float(upper), config.n_mels + 2)
    hz = 700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0)
    bins = torch.arange(config.n_fft // 2 + 1) * (config.sample_rate / config.n_fft)
    rising = (bins[None, :] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None]
    falling = (hz[2:, None] - bins[None, :]) / (hz[2:] - hz[1:-1])[:, None]
    filters = torch.minimum(rising, falling).clamp_min(0)
    if bool((filters.sum(dim=1) == 0).any()):
        raise ValueError("n_mels is too large for n_fft: empty mel filters")
    return filters


class LogMelCNN(nn.Module):
    """General-audio candidate using time-then-frequency convolutions.

    Each sample is zero-padded to cover its final STFT window (center=False).
    Masks are applied after each convolution, so unrelated batch padding does
    not change a sample's representation. There is no time-spanning batch or
    group normalization. Actual detection quality requires labeled data.
    """

    window: Tensor
    mel_filters: Tensor

    def __init__(self, config: LogMelConfig | None = None) -> None:
        super().__init__()
        config = config or LogMelConfig()
        self.config = config
        self.register_buffer("window", torch.hann_window(config.n_fft))
        self.register_buffer("mel_filters", _mel_filterbank(config))
        self.temporal = nn.Conv2d(1, config.channels, (3, 1), padding=(1, 0))
        self.spectral = nn.Conv2d(
            config.channels, config.channels, (1, 3), padding=(0, 1)
        )
        self.temporal2 = nn.Conv2d(
            config.channels, config.channels * 2, (3, 1), padding=(1, 0)
        )
        self.spectral2 = nn.Conv2d(
            config.channels * 2, config.channels * 2, (1, 3), padding=(0, 1)
        )
        self.head = FeatureHead(
            HeadConfig(config.channels * 4, config.hidden_dim, config.dropout)
        )

    def extract_features(
        self, waveforms: Tensor, lengths: Tensor | None = None
    ) -> Tensor:
        lengths = validate_waveforms(waveforms, lengths)
        cfg = self.config
        positions = torch.arange(waveforms.shape[1], device=waveforms.device)
        waves = waveforms.masked_fill(positions[None, :] >= lengths[:, None], 0)
        frame_lengths = (
            (lengths - cfg.n_fft).clamp_min(0) + cfg.hop_length - 1
        ) // cfg.hop_length + 1
        sample_count = (int(frame_lengths.max()) - 1) * cfg.hop_length + cfg.n_fft
        waves = functional.pad(
            waves[:, :sample_count], (0, max(0, sample_count - waves.shape[1]))
        )
        # STFT remains FP32 under CUDA AMP; subsequent convolutions may use AMP.
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            spectrum = (
                torch.stft(
                    waves.float(),
                    cfg.n_fft,
                    cfg.hop_length,
                    window=self.window.float(),
                    center=False,
                    return_complex=True,
                )
                .abs()
                .square()
            )
            mel = torch.matmul(self.mel_filters.float(), spectrum).clamp_min(1e-7).log()
        frame_mask = (
            torch.arange(mel.shape[-1], device=waveforms.device)[None, :]
            < frame_lengths[:, None]
        )[:, None, :, None]
        x = mel.transpose(1, 2).unsqueeze(1).masked_fill(~frame_mask, 0)
        for layer in (self.temporal, self.spectral, self.temporal2, self.spectral2):
            x = functional.gelu(layer(x)).masked_fill(~frame_mask, 0)
            if layer is self.spectral:
                x = functional.avg_pool2d(x, (1, 2))
        x = x.mean(dim=-1)
        count = frame_lengths[:, None].to(x.dtype)
        mean = x.sum(dim=-1) / count
        variance = ((x - mean[:, :, None]).square() * frame_mask[..., 0]).sum(
            dim=-1
        ) / count
        return torch.cat((mean, (variance + 1e-6).sqrt()), dim=1)

    def forward(self, waveforms: Tensor, lengths: Tensor | None = None) -> Tensor:
        return self.head.forward(self.extract_features(waveforms, lengths))


PoolingMethod = Literal["mean", "median", "logsumexp", "trimmed_mean"]


def pool_media_scores(
    scores: Tensor,
    mask: Tensor | None = None,
    method: PoolingMethod = "mean",
    trim_fraction: float = 0.1,
) -> Tensor:
    """Aggregate actual frame/segment scores without dropping failed media.

    Callers choose whether inputs are logits or sigmoid scores; the function
    never changes their domain or clips them. Invalid/padded positions require
    an explicit mask. No valid observations raises instead of fabricating a score.
    """
    if scores.ndim != 2 or min(scores.shape) == 0 or not scores.is_floating_point():
        raise ValueError("scores must have floating shape [batch, observations]")
    if method not in ("mean", "median", "logsumexp", "trimmed_mean"):
        raise ValueError(f"unsupported pooling method: {method}")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5)")
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    if mask.shape != scores.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be bool and match scores")
    mask = mask.to(scores.device)
    if bool((~mask.any(dim=1)).any()):
        raise ValueError("each media must have at least one valid observation")
    if not bool(torch.isfinite(scores[mask]).all()):
        raise ValueError("valid scores must be finite")
    if method == "mean":
        accumulation_dtype = (
            torch.float64 if scores.dtype == torch.float64 else torch.float32
        )
        total = scores.masked_fill(~mask, 0).sum(dim=1, dtype=accumulation_dtype)
        return (total / mask.sum(dim=1).to(accumulation_dtype)).to(scores.dtype)
    if method == "logsumexp":
        return (
            torch.logsumexp(scores.masked_fill(~mask, -torch.inf), dim=1)
            - mask.sum(dim=1).to(scores.dtype).log()
        )
    results: list[Tensor] = []
    for row, valid in zip(scores, mask, strict=True):
        values = row[valid]
        if method == "median":
            quantile_values = (
                values if values.dtype == torch.float64 else values.float()
            )
            result = torch.quantile(quantile_values, 0.5).to(values.dtype)
        else:
            ordered = values.sort().values
            cut = int(values.numel() * trim_fraction)
            result = ordered[cut : values.numel() - cut].mean()
        results.append(result)
    return torch.stack(results)


@dataclass(frozen=True)
class MediaFeatureConfig:
    input_dim: int
    hidden_dim: int = 64
    dropout: float = 0.1
    aggregation: PoolingMethod = "mean"
    trim_fraction: float = 0.1
    temporal_kernel_size: int = 0

    def __post_init__(self) -> None:
        HeadConfig(self.input_dim, self.hidden_dim, self.dropout)
        if type(
            self.temporal_kernel_size
        ) is not int or self.temporal_kernel_size not in (
            0,
            3,
        ):
            raise ValueError("temporal_kernel_size must be 0 or 3")
        if self.aggregation not in ("mean", "median", "logsumexp", "trimmed_mean"):
            raise ValueError(f"unsupported aggregation: {self.aggregation}")
        if not 0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must lie in [0, 0.5)")


class MediaFeatureClassifier(nn.Module):
    """One media-level fake logit from cached frame or audio-segment embeddings.

    The shared head and its aggregation are checkpointed together. Training,
    validation and inference all call this same forward; labels refer to media,
    never to individual observations. Aggregation acts on logits, not sigmoid
    scores, and the output is suitable for BCEWithLogitsLoss. A sigmoid may be
    applied to the final media logit by a caller needing bounded scores.
    """

    def __init__(self, config: MediaFeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.head: FeatureHead | TemporalHead
        if config.temporal_kernel_size == 0:
            self.head = FeatureHead(
                HeadConfig(config.input_dim, config.hidden_dim, config.dropout)
            )
        else:
            self.head = TemporalHead(
                config.input_dim, config.hidden_dim, config.dropout
            )

    def forward(self, features: Tensor, lengths: Tensor | None = None) -> Tensor:
        if features.ndim != 3 or min(features.shape) == 0:
            raise ValueError(
                "features must have nonempty shape [batch, observations, dim]"
            )
        mask: Tensor | None = None
        if lengths is not None:
            if lengths.shape != (features.shape[0],) or lengths.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("lengths must be an integer vector matching the batch")
            lengths = lengths.to(features.device)
            if bool(((lengths < 1) | (lengths > features.shape[1])).any()):
                raise ValueError("lengths must lie between 1 and observation count")
            positions = torch.arange(features.shape[1], device=features.device)
            mask = positions[None, :] < lengths[:, None]
            features = features.masked_fill(~mask[:, :, None], 0)
        logits = (
            self.head.forward(features, mask)
            if isinstance(self.head, TemporalHead)
            else self.head.forward(features)
        )
        return pool_media_scores(
            logits, mask, self.config.aggregation, self.config.trim_fraction
        )
