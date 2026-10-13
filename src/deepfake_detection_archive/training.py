"""Reproducible binary training with development-only checkpoint selection.

Loaders own decoding, split isolation and worker seeds. They must be re-iterable;
no holdout loader is accepted here. Scores are logits with fake=1. Wall-clock
limits are checked at epoch boundaries to preserve complete development reports.
"""

import hashlib
import json
import math
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .evaluation.metrics import evaluate_binary
from .models.ast import ASTClassifier
from .models.checkpoint import load_checkpoint, model_spec, save_checkpoint
from .tracking import atomic_json


Batch = tuple[Tensor, Tensor] | Mapping[str, Tensor]


@dataclass(frozen=True)
class DomainEERConstraint:
    """A fixed development baseline and absolute EER regression allowance."""

    baseline_eer: float
    max_regression: float

    def __post_init__(self) -> None:
        for name in ("baseline_eer", "max_regression"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class DevDomainMetrics:
    sample_count: int
    real_count: int
    fake_count: int
    eer: float | None
    reason: str | None


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    accumulation_steps: int = 1
    patience: int = 5
    min_delta: float = 0.0
    device: str = "cpu"
    amp: bool = False
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    seed: int = 42
    max_seconds: float | None = None
    max_grad_norm: float | None = None
    deterministic: bool = True
    backbone_lr_scale: float = 1.0
    time_mask_width: int = 0
    frequency_mask_width: int = 0
    domain_eer_constraints: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        for name in ("max_epochs", "accumulation_steps", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("time_mask_width", "frequency_mask_width"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("amp", "deterministic"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError("amp_dtype must be float16 or bfloat16")
        for name in (
            "learning_rate",
            "weight_decay",
            "min_delta",
            "max_seconds",
            "max_grad_norm",
        ):
            value = getattr(self, name)
            if value is None and name in ("max_seconds", "max_grad_norm"):
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            minimum_inclusive = name in ("weight_decay", "min_delta")
            if value < 0 or (value == 0 and not minimum_inclusive):
                raise ValueError(f"{name} is outside its allowed range")
        if not 0 < self.backbone_lr_scale <= 1:
            raise ValueError("backbone_lr_scale must lie in (0, 1]")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        try:
            device = torch.device(self.device)
        except (ValueError, RuntimeError) as exc:
            raise ValueError("invalid device") from exc
        if device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU and CUDA are supported")
        if self.amp and device.type != "cuda":
            raise ValueError("AMP is supported only with CUDA")
        _ = self.domain_constraints

    @cached_property
    def domain_constraints(self) -> dict[str, DomainEERConstraint]:
        if not isinstance(self.domain_eer_constraints, dict):
            raise ValueError("domain_eer_constraints must be an object")  # noqa: TRY004 -- schema
        parsed: dict[str, DomainEERConstraint] = {}
        for domain, options in self.domain_eer_constraints.items():
            if (
                not isinstance(domain, str)
                or not domain.strip()
                or domain.strip().casefold() == "unknown"
            ):
                raise ValueError("domain constraints require explicit known domains")
            if not isinstance(options, dict) or set(options) != {
                "baseline_eer",
                "max_regression",
            }:
                raise ValueError(
                    f"domain constraint {domain!r} requires exactly "
                    "baseline_eer and max_regression"
                )
            parsed[domain] = DomainEERConstraint(**options)
        return parsed


@dataclass(frozen=True)
class TrainResult:
    best_checkpoint: Path
    checkpoint_sha256: str
    best_epoch: int
    best_dev_eer: float
    epochs_completed: int
    stop_reason: Literal["max_epochs", "early_stopping", "time_limit"]
    history_path: Path
    run_path: Path


def _digest_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _prepare_batch(
    batch: Batch, device: torch.device
) -> tuple[Tensor, Tensor, Tensor | None]:
    lengths: Tensor | None = None
    if isinstance(batch, Mapping):
        if not {"inputs", "labels"} <= set(batch) or set(batch) - {
            "inputs",
            "labels",
            "lengths",
        }:
            raise ValueError(
                "batch mapping requires inputs, labels and optional lengths"
            )
        inputs, labels = batch["inputs"], batch["labels"]
        lengths = batch.get("lengths")
    else:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("batch must be (inputs, labels) or a tensor mapping")
        inputs, labels = batch
    if not isinstance(inputs, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("inputs and labels must be tensors")
    if inputs.ndim < 2 or inputs.shape[0] == 0:
        raise ValueError("inputs must be nonempty and batched")
    if (
        labels.shape != (inputs.shape[0],)
        or labels.dtype == torch.bool
        or labels.is_complex()
    ):
        raise ValueError("labels must have shape [batch] and numeric binary values")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must be 0 or 1")
    if not inputs.is_floating_point() or not bool(torch.isfinite(inputs).all()):
        raise ValueError("inputs must be finite floating tensors")
    if lengths is not None:
        if not isinstance(lengths, Tensor):
            raise ValueError("lengths must be a tensor")
        lengths = lengths.to(device, non_blocking=True)
    return (
        inputs.to(device, non_blocking=True),
        labels.to(device, dtype=torch.float32, non_blocking=True),
        lengths,
    )


def _parameter_groups(
    model: nn.Module, parameters: Sequence[nn.Parameter], config: TrainConfig
) -> list[Any]:
    """Split AdamW groups so a partially fine-tuned backbone trains slower.

    Only models that expose a `backbone.` prefix take part; everything else
    keeps the previous single-group behaviour, and a scale of one is a no-op.
    """
    if config.backbone_lr_scale == 1.0:
        return list(parameters)
    slow_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    }
    slow = [value for value in parameters if id(value) in slow_ids]
    fast = [value for value in parameters if id(value) not in slow_ids]
    if not slow or not fast:
        return list(parameters)
    return [
        {"params": slow, "lr": config.learning_rate * config.backbone_lr_scale},
        {"params": fast, "lr": config.learning_rate},
    ]


def _logits(model: nn.Module, inputs: Tensor, lengths: Tensor | None) -> Tensor:
    logits = model(inputs) if lengths is None else model(inputs, lengths=lengths)
    if not isinstance(logits, Tensor) or logits.shape != (inputs.shape[0],):
        raise ValueError("model must return logits with shape [batch]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("model returned non-finite logits")
    return logits


def _step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sample_count: int,
    max_grad_norm: float | None,
) -> bool:
    scaler.unscale_(optimizer)
    finite = True
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)
            if not bool(torch.isfinite(parameter.grad).all()):
                finite = False
    if not finite:
        if not scaler.is_enabled():
            raise ValueError("non-finite training gradient")
        # unscale_ recorded overflow: GradScaler skips this optimizer step and
        # reduces its scale. Record the skip instead of silently counting it.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return False
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True
        )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return True


def _dev_predictions(
    model: nn.Module, loader: Iterable[Batch], device: torch.device
) -> tuple[float, list[int], list[float]]:
    model.eval()
    labels_all: list[int] = []
    scores_all: list[float] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            inputs, labels, lengths = _prepare_batch(batch, device)
            logits = _logits(model, inputs, lengths).float()
            loss_sum += float(
                functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum"
                )
            )
            labels_all.extend(int(value) for value in labels.cpu().tolist())
            scores_all.extend(logits.cpu().tolist())
    return loss_sum, labels_all, scores_all


def _evaluate(
    model: nn.Module, loader: Iterable[Batch], device: torch.device
) -> tuple[float, float, int]:
    loss_sum, labels, scores = _dev_predictions(model, loader, device)
    metrics = evaluate_binary(labels, scores)
    return loss_sum / len(labels), metrics.eer, len(labels)


def _evaluate_domains(
    model: nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
    domains: Sequence[str],
) -> tuple[float, float, int, dict[str, DevDomainMetrics]]:
    loss_sum, labels, scores = _dev_predictions(model, loader, device)
    metrics = evaluate_binary(labels, scores)
    if len(domains) != len(labels):
        raise ValueError("dev_domains must match development sample count and order")
    buckets: dict[str, list[int]] = {}
    for index, domain in enumerate(domains):
        buckets.setdefault(domain, []).append(index)
    reports = {}
    for domain, indices in sorted(buckets.items()):
        domain_labels = [labels[index] for index in indices]
        both_classes = set(domain_labels) == {0, 1}
        reports[domain] = DevDomainMetrics(
            sample_count=len(indices),
            real_count=domain_labels.count(0),
            fake_count=domain_labels.count(1),
            eer=(
                evaluate_binary(domain_labels, [scores[index] for index in indices]).eer
                if both_classes
                else None
            ),
            reason=None if both_classes else "missing_class",
        )
    return loss_sum / len(labels), metrics.eer, len(labels), reports


def _check_domain_constraints(
    reports: Mapping[str, DevDomainMetrics],
    constraints: Mapping[str, DomainEERConstraint],
) -> dict[str, dict[str, float]]:
    regressions = {}
    for domain, constraint in constraints.items():
        report = reports.get(domain)
        if report is None or report.eer is None:
            reason = "missing_domain" if report is None else report.reason
            raise ValueError(
                f"domain constraint {domain!r} lacks development EER evidence: {reason}"
            )
        limit = constraint.baseline_eer + constraint.max_regression
        if report.eer > limit:
            regressions[domain] = {
                "eer": report.eer,
                "baseline_eer": constraint.baseline_eer,
                "max_regression": constraint.max_regression,
                "max_eer": limit,
            }
    return regressions


def _validate_dev_domains(
    domains: Sequence[str] | None, constraints: Mapping[str, DomainEERConstraint]
) -> tuple[str, ...] | None:
    if domains is None:
        if constraints:
            raise ValueError("domain_eer_constraints requires explicit dev_domains")
        return None
    if (
        isinstance(domains, str)
        or not domains
        or any(not isinstance(domain, str) or not domain.strip() for domain in domains)
    ):
        raise ValueError("dev_domains must be a nonempty sequence of domain strings")
    if missing := set(constraints) - set(domains):
        raise ValueError(
            f"constrained development domains are missing: {sorted(missing)}"
        )
    return tuple(domains)


def _mask_fbank(
    inputs: Tensor, time_width: int, frequency_width: int, fill: float
) -> Tensor:
    """Mask each sample independently; widths exclude the configured upper bound."""
    for axis, width in ((1, time_width), (2, frequency_width)):
        if width == 0:
            continue
        size = inputs.shape[axis]
        widths = torch.randint(width, (inputs.shape[0],), device=inputs.device)
        starts = (
            torch.rand(inputs.shape[0], device=inputs.device) * (size - widths + 1)
        ).long()
        positions = torch.arange(size, device=inputs.device)[None, :]
        mask = (positions >= starts[:, None]) & (positions < (starts + widths)[:, None])
        inputs = inputs.masked_fill(mask.unsqueeze(2 if axis == 1 else 1), fill)
    return inputs


def _train_epoch(  # noqa: PLR0913, PLR0917
    model: nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    mask_fill: float,
) -> tuple[float, int, int, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    samples, window_samples, updates, skipped = 0, 0, 0, 0
    loss_sum = 0.0
    for batches, batch in enumerate(loader, start=1):
        inputs, labels, lengths = _prepare_batch(batch, device)
        inputs = _mask_fbank(
            inputs, config.time_mask_width, config.frequency_mask_width, mask_fill
        )
        with torch.autocast(
            device_type=device.type,
            enabled=config.amp,
            dtype=torch.float16 if config.amp_dtype == "float16" else torch.bfloat16,
        ):
            logits = _logits(model, inputs, lengths)
            loss = functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"
            )
        if not bool(torch.isfinite(loss)):
            raise ValueError("non-finite training loss")
        scaled_loss: Tensor = scaler.scale(loss)
        scaled_loss.backward()  # type: ignore[no-untyped-call]
        count = labels.numel()
        samples += count
        window_samples += count
        loss_sum += float(loss.detach())
        if batches % config.accumulation_steps == 0:
            stepped = _step(
                model, optimizer, scaler, window_samples, config.max_grad_norm
            )
            window_samples = 0
            updates += int(stepped)
            skipped += int(not stepped)
    if not samples:
        raise ValueError("training loader is empty or exhausted")
    if window_samples:
        stepped = _step(model, optimizer, scaler, window_samples, config.max_grad_norm)
        updates += int(stepped)
        skipped += int(not stepped)
    return loss_sum / samples, samples, updates, skipped


def _training_device(config: TrainConfig) -> torch.device:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if config.amp and config.amp_dtype == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError("CUDA device does not support native bfloat16 AMP")
    return device


def train_model(  # noqa: C901, PLR0913, PLR0912, PLR0915
    model: nn.Module,
    train_loader: Iterable[Batch],
    dev_loader: Iterable[Batch],
    *,
    config: TrainConfig,
    output_dir: Path,
    provenance: dict[str, Any],
    architecture: str | None = None,
    model_config: dict[str, Any] | None = None,
    dev_domains: Sequence[str] | None = None,
) -> TrainResult:
    """Train AdamW/BCE and restore the checkpoint with the lowest dev EER.

    Summed per-sample losses are backpropagated; gradients are divided by the
    actual accumulation-window sample count, including uneven and tail batches.
    This matches a concatenated batch for models without batch-dependent layers.
    Seed controls training randomness, not caller-side model initialization.
    Provenance hashes must describe data, code and initial weights. The caller
    must enforce train/dev media-group separation before constructing loaders.
    Optional dev_domains must follow the unshuffled, stable dev loader order.
    Domain constraints use fixed baselines and never inspect a holdout split.
    """
    constraints = dict(config.domain_constraints)
    domains = _validate_dev_domains(dev_domains, constraints)
    # Record the validated effective values, not a caller-mutated raw JSON dict.
    effective_config = asdict(config)
    effective_config["domain_eer_constraints"] = {
        domain: asdict(constraint) for domain, constraint in constraints.items()
    }
    actual_architecture, actual_config = model_spec(model)
    if architecture is not None and architecture != actual_architecture:
        raise ValueError("architecture differs from model")
    if model_config is not None and model_config != actual_config:
        raise ValueError("model_config differs from model")
    mask_fill = 0.0
    if config.time_mask_width or config.frequency_mask_width:
        if actual_architecture != "audio_ast_v1":
            raise ValueError("fbank masking is supported only with audio_ast_v1")
        backbone_config = cast(ASTClassifier, model).backbone.config
        if config.time_mask_width > backbone_config.max_length:
            raise ValueError("time_mask_width must not exceed AST max_length")
        if config.frequency_mask_width > backbone_config.num_mel_bins:
            raise ValueError("frequency_mask_width must not exceed AST num_mel_bins")
        mask_fill = -actual_config["feature_mean"] / (2 * actual_config["feature_std"])
    for name in ("data_sha256", "code_sha256", "weights_sha256"):
        value = provenance.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"provenance requires a lowercase SHA-256: {name}")
    provenance_hash = _digest_json(provenance)
    if train_loader is dev_loader:
        raise ValueError("train and development loaders must be distinct")
    device = _training_device(config)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("output_dir must be empty to prevent overwriting experiments")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    model.to(device)
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, parameters, config),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.amp and config.amp_dtype == "float16"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    run_path, history_path = output_dir / "run.json", output_dir / "history.json"
    run: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "config": effective_config,
        "config_sha256": _digest_json(effective_config),
        "architecture": actual_architecture,
        "model_config": actual_config,
        "model_config_sha256": _digest_json(actual_config),
        "provenance": provenance,
        "provenance_sha256": provenance_hash,
        "selection_split": "dev",
        "score_direction": "higher_is_fake",
        "torch_version": str(torch.__version__),
        "device": str(device),
        "grad_scaler_enabled": scaler.is_enabled(),
        "seed_scope": "training; initialize model separately",
        "time_limit_scope": "epoch_boundary",
        "resume_supported": False,
    }
    atomic_json(run_path, run)
    history: list[dict[str, Any]] = []
    best_eer, best_epoch, stale = math.inf, 0, 0
    best_domains: dict[str, DevDomainMetrics] = {}
    checkpoint = output_dir / "best.pt"
    descriptor: dict[str, Any] = {}
    stop_reason: Literal["max_epochs", "early_stopping", "time_limit"] = "max_epochs"
    try:
        torch.use_deterministic_algorithms(config.deterministic)
        for epoch in range(1, config.max_epochs + 1):
            epoch_started = time.monotonic()
            train_loss, samples, updates, skipped = _train_epoch(
                model, train_loader, device, optimizer, scaler, config, mask_fill
            )
            train_seconds = time.monotonic() - epoch_started
            domain_reports: dict[str, DevDomainMetrics] = {}
            if domains is None:
                dev_loss, dev_eer, dev_count = _evaluate(model, dev_loader, device)
            else:
                dev_loss, dev_eer, dev_count, domain_reports = _evaluate_domains(
                    model, dev_loader, device, domains
                )
            regressions = _check_domain_constraints(domain_reports, constraints)
            eligible = not regressions
            stale = (
                0 if eligible and dev_eer < best_eer - config.min_delta else stale + 1
            )
            improved = eligible and dev_eer < best_eer
            if improved:
                descriptor = save_checkpoint(checkpoint, model)
                best_eer, best_epoch = dev_eer, epoch
                best_domains = domain_reports
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_samples": samples,
                    "train_seconds": train_seconds,
                    "train_samples_per_second": samples / max(train_seconds, 1e-9),
                    "optimizer_updates": updates,
                    "skipped_updates": skipped,
                    "dev_loss": dev_loss,
                    "dev_eer": dev_eer,
                    "dev_samples": dev_count,
                    "selected": improved,
                    "seconds": time.monotonic() - epoch_started,
                }
            )
            if domains is not None:
                history[-1].update(
                    dev_domains={
                        domain: asdict(report)
                        for domain, report in domain_reports.items()
                    },
                    selection_eligible=eligible,
                    domain_regressions=regressions,
                )
            atomic_json(history_path, history)
            if stale >= config.patience:
                stop_reason = "early_stopping"
                break
            if (
                config.max_seconds is not None
                and time.monotonic() - started >= config.max_seconds
            ):
                stop_reason = "time_limit"
                break
        if not descriptor:
            raise ValueError(  # noqa: TRY301 -- recorded in failed run metadata
                "no checkpoint met the development domain EER constraints; "
                "inspect history.json domain_regressions and configured baselines"
            )
        restored, _ = load_checkpoint(checkpoint, expected_sha256=descriptor["sha256"])
        model.load_state_dict(restored.state_dict())
        model.eval()
        run.update(
            status="complete",
            best_epoch=best_epoch,
            best_dev_eer=best_eer,
            checkpoint=descriptor,
            stop_reason=stop_reason,
            epochs_completed=len(history),
        )
        if domains is not None:
            run["best_dev_domains"] = {
                domain: asdict(report) for domain, report in best_domains.items()
            }
    except BaseException as exc:
        optimizer.zero_grad(set_to_none=True)
        run.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        run["seconds"] = time.monotonic() - started
        run["peak_cuda_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
        run["peak_cuda_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        )
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )
        atomic_json(run_path, run)
    return TrainResult(
        checkpoint,
        descriptor["sha256"],
        best_epoch,
        best_eer,
        len(history),
        stop_reason,
        history_path,
        run_path,
    )
