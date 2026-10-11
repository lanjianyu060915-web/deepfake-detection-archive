"""Reproducible binary training with development-only checkpoint selection.

Loaders own decoding, split isolation and worker seeds. They must be re-iterable;
no holdout loader is accepted here. Scores are logits with fake=1. Wall-clock
limits are checked at epoch boundaries to preserve complete development reports.
"""

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .evaluation.metrics import evaluate_binary
from .models.checkpoint import load_checkpoint, model_spec, save_checkpoint


Batch = tuple[Tensor, Tensor] | Mapping[str, Tensor]


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

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        for name in ("max_epochs", "accumulation_steps", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
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


def _write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    model: nn.Module, parameters: list[Tensor], config: TrainConfig
) -> list[Any]:
    """Split AdamW groups so a partially fine-tuned backbone trains slower.

    Only models that expose a `backbone.` prefix take part; everything else
    keeps the previous single-group behaviour, and a scale of one is a no-op.
    """
    if config.backbone_lr_scale == 1.0:
        return cast(list[Any], list(parameters))
    slow_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    }
    slow = [value for value in parameters if id(value) in slow_ids]
    fast = [value for value in parameters if id(value) not in slow_ids]
    if not slow or not fast:
        return cast(list[Any], list(parameters))
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


def _evaluate(
    model: nn.Module, loader: Iterable[Batch], device: torch.device
) -> tuple[float, float, int]:
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
    metrics = evaluate_binary(labels_all, scores_all)
    return loss_sum / len(labels_all), metrics.eer, len(labels_all)


def _train_epoch(  # noqa: PLR0913, PLR0917
    model: nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
) -> tuple[float, int, int, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    samples, window_samples, updates, skipped = 0, 0, 0, 0
    loss_sum = 0.0
    for batches, batch in enumerate(loader, start=1):
        inputs, labels, lengths = _prepare_batch(batch, device)
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


def train_model(  # noqa: PLR0913, PLR0912, PLR0915
    model: nn.Module,
    train_loader: Iterable[Batch],
    dev_loader: Iterable[Batch],
    *,
    config: TrainConfig,
    output_dir: Path,
    provenance: dict[str, Any],
    architecture: str | None = None,
    model_config: dict[str, Any] | None = None,
) -> TrainResult:
    """Train AdamW/BCE and restore the checkpoint with the lowest dev EER.

    Summed per-sample losses are backpropagated; gradients are divided by the
    actual accumulation-window sample count, including uneven and tail batches.
    This matches a concatenated batch for models without batch-dependent layers.
    Seed controls training randomness, not caller-side model initialization.
    Provenance hashes must describe data, code and initial weights. The caller
    must enforce train/dev media-group separation before constructing loaders.
    """
    actual_architecture, actual_config = model_spec(model)
    if architecture is not None and architecture != actual_architecture:
        raise ValueError("architecture differs from model")
    if model_config is not None and model_config != actual_config:
        raise ValueError("model_config differs from model")
    for name in ("data_sha256", "code_sha256", "weights_sha256"):
        value = provenance.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"provenance requires a lowercase SHA-256: {name}")
    _digest_json(provenance)
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
        "config": asdict(config),
        "config_sha256": _digest_json(asdict(config)),
        "architecture": actual_architecture,
        "model_config": actual_config,
        "model_config_sha256": _digest_json(actual_config),
        "provenance": provenance,
        "provenance_sha256": _digest_json(provenance),
        "selection_split": "dev",
        "score_direction": "higher_is_fake",
        "torch_version": str(torch.__version__),
        "device": str(device),
        "grad_scaler_enabled": scaler.is_enabled(),
        "seed_scope": "training; initialize model separately",
        "time_limit_scope": "epoch_boundary",
        "resume_supported": False,
    }
    _write_json(run_path, run)
    history: list[dict[str, Any]] = []
    best_eer, best_epoch, stale = math.inf, 0, 0
    checkpoint = output_dir / "best.pt"
    descriptor: dict[str, Any] = {}
    stop_reason: Literal["max_epochs", "early_stopping", "time_limit"] = "max_epochs"
    try:
        torch.use_deterministic_algorithms(config.deterministic)
        for epoch in range(1, config.max_epochs + 1):
            epoch_started = time.monotonic()
            train_loss, samples, updates, skipped = _train_epoch(
                model, train_loader, device, optimizer, scaler, config
            )
            train_seconds = time.monotonic() - epoch_started
            dev_loss, dev_eer, dev_count = _evaluate(model, dev_loader, device)
            stale = 0 if dev_eer < best_eer - config.min_delta else stale + 1
            improved = dev_eer < best_eer
            if improved:
                descriptor = save_checkpoint(checkpoint, model)
                best_eer, best_epoch = dev_eer, epoch
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
            _write_json(history_path, history)
            if stale >= config.patience:
                stop_reason = "early_stopping"
                break
            if (
                config.max_seconds is not None
                and time.monotonic() - started >= config.max_seconds
            ):
                stop_reason = "time_limit"
                break
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
        _write_json(run_path, run)
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
