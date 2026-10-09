"""Audited train/dev experiments backed by content-bound local tensor caches."""

import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import load_object
from .contracts import ScoreRecord
from .data.audit import audit_manifest
from .data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    load_manifest,
    load_sources,
    manifest_hash,
)
from .data.split import validate_splits
from .evaluation.report import evaluate_records
from .features import (
    Encoder,
    extract_audio_feature_batch,
    extract_features,
    load_encoder,
    media_path,
)
from .media import AudioConfig, VideoConfig, decode_audio, sample_audio_segments
from .models.checkpoint import create_model, save_checkpoint, verify_sha256
from .models.encoders import FrozenAudioEncoder
from .tracking import atomic_json, canonical_json, code_identity, object_hash, utc_now
from .training import TrainConfig, TrainResult, train_model


@dataclass(frozen=True)
class ExperimentConfig:
    manifest: str
    data_root: str
    sources: str
    output_dir: str
    modality: str
    architecture: str
    model_config: dict[str, Any]
    training: dict[str, Any]
    batch_size: int = 8
    feature_batch_size: int = 8
    num_workers: int = 0
    balanced_sampling: bool = False
    segment_samples: int = 64600
    segments: int = 1
    audio: dict[str, Any] = field(default_factory=dict)
    video: dict[str, Any] = field(default_factory=dict)
    encoder: dict[str, Any] = field(default_factory=dict)
    feature_cache: str | None = None
    cache_max_bytes: int = 20 * 1024**3
    schema_version: int = 1

    def __post_init__(self) -> None:  # noqa: C901, PLR0912 -- strict configuration schema
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("experiment schema_version must be 1")
        for name in ("manifest", "data_root", "sources", "output_dir"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a nonempty path")
        for name in (
            "batch_size",
            "feature_batch_size",
            "segment_samples",
            "segments",
            "cache_max_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.balanced_sampling) is not bool:
            raise ValueError("balanced_sampling must be boolean")
        if self.modality not in ("audio", "video"):
            raise ValueError("modality must be audio or video")
        for name in ("model_config", "training", "audio", "video", "encoder"):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"{name} must be an object")
        if any(
            type(value) is bool
            for options in (self.model_config, self.audio, self.video)
            for value in options.values()
        ):
            raise ValueError("model and media parameters cannot be boolean")
        if self.feature_cache is not None and (
            not isinstance(self.feature_cache, str) or not self.feature_cache.strip()
        ):
            raise ValueError("feature_cache must be null or a nonempty path")
        if self.architecture not in (
            "logmel_cnn_v1",
            "aasist_l_v1",
            "media_feature_head_v1",
        ):
            raise ValueError("unsupported experiment architecture")
        if self.architecture != "media_feature_head_v1":
            if self.modality != "audio" or self.segments != 1 or self.encoder:
                raise ValueError(
                    "waveform training requires audio, segments=1 and no encoder"
                )
        elif self.encoder.get("kind") != (
            "audio" if self.modality == "audio" else "vision"
        ):
            raise ValueError("encoder kind must match experiment modality")
        _ = self.train_config
        audio, _ = self.media_configs
        if self.architecture == "aasist_l_v1" and (
            audio.sample_rate != 16000 or self.segment_samples != 64600
        ):
            raise ValueError("AASIST-L requires 16000 Hz and segment_samples=64600")
        if (
            self.architecture == "logmel_cnn_v1"
            and audio.sample_rate != self.model_config.get("sample_rate", 16000)
        ):
            raise ValueError("audio and log-mel sample rates must agree")

    @cached_property
    def train_config(self) -> TrainConfig:
        return TrainConfig(**self.training)

    @cached_property
    def media_configs(self) -> tuple[AudioConfig, VideoConfig]:
        return AudioConfig(**self.audio), VideoConfig(**self.video)


def load_experiment_config(path: Path) -> ExperimentConfig:
    try:
        return ExperimentConfig(**load_object(path))  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"invalid experiment configuration: {exc}") from exc


def _cache_pair_bytes(path: Path) -> int:
    return sum(
        entry.stat().st_size
        for entry in (path, path.with_suffix(".json"))
        if entry.is_file()
    )


def _save_tensor(
    path: Path,
    tensor: Tensor,
    identity: dict[str, Any],
    maximum: int,
    *,
    cache_bytes: int | None = None,
) -> str:
    """Publish a tensor/sidecar pair only if the final cache fits its byte cap.

    The runner may supply its tracked committed byte count to avoid rescanning
    the cache for every sample. Staging bytes are temporary disk overhead, while
    `maximum` bounds committed cache contents, including UTF-8 sidecar bytes.
    """
    if cache_bytes is None:
        cache_bytes = sum(
            entry.stat().st_size for entry in path.parent.iterdir() if entry.is_file()
        )
    replaced_bytes = _cache_pair_bytes(path)
    temporary: Path | None = None
    temporary_metadata: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(tensor.detach().cpu().contiguous(), stream)
        digest = file_sha256(temporary)
        metadata = {"identity": identity, "sha256": digest}
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary_metadata = Path(stream.name)
            stream.write((canonical_json(metadata) + "\n").encode("utf-8"))
        new_bytes = temporary.stat().st_size + temporary_metadata.stat().st_size
        if cache_bytes - replaced_bytes + new_bytes > maximum:
            raise ValueError("tensor cache exceeds cache_max_bytes")
        temporary.replace(path)
        temporary_metadata.replace(path.with_suffix(".json"))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if temporary_metadata is not None:
            temporary_metadata.unlink(missing_ok=True)
    return digest


def _load_tensor(path: Path, digest: str) -> Tensor:
    verify_sha256(path, digest)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("invalid cached tensor")
    return value


class _CachedDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, entries: list[tuple[Path, str, int]]) -> None:
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, digest, label = self.entries[index]
        return _load_tensor(path, digest), torch.tensor(label, dtype=torch.float32)


def _worker_seed(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- initialize worker-local legacy RNG users


def _balanced_sampler(
    entries: list[tuple[Path, str, int]], generator: torch.Generator
) -> WeightedRandomSampler:
    labels = torch.tensor([label for _, _, label in entries], dtype=torch.long)
    counts = torch.bincount(labels, minlength=2)
    weights = counts.double().reciprocal()[labels]
    return WeightedRandomSampler(
        weights.tolist(), len(entries), replacement=True, generator=generator
    )


def _write_dev_scores(
    model: torch.nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    records: list[SampleRecord],
    device: str,
    output: Path,
) -> None:
    """Score the selected checkpoint from the prepared dev tensors once."""
    values: list[float] = []
    started = time.monotonic()
    model.eval()
    with torch.inference_mode():
        for inputs, _ in loader:
            logits = cast(Tensor, model(inputs.to(device, non_blocking=True)))
            values.extend(logits.float().cpu().tolist())
    seconds = time.monotonic() - started
    scores = [
        ScoreRecord(
            sample_id=record.sample_id,
            modality=record.modality,
            score=value,
            status="ok",
            elapsed_seconds=seconds / len(records),
            label=record.label,
            group_id=record.group_id,
            source=record.source,
            content_type=record.content_type,
            generator=record.generator,
        )
        for record, value in zip(records, values, strict=True)
    ]
    (output / "dev-scores.jsonl").write_text(
        "".join(canonical_json(score.to_dict()) + "\n" for score in scores),
        encoding="utf-8",
    )
    atomic_json(
        output / "dev-evaluation.json",
        {
            "scope": "selected checkpoint on prepared dev tensors; decoding excluded",
            "seconds": seconds,
            "evaluation": asdict(evaluate_records(scores)),
        },
    )


def _audit_training_records(
    records: list[SampleRecord],
    config: ExperimentConfig,
    root: Path,
    sources: dict[str, SourceRecord],
    output: Path,
) -> list[SampleRecord]:
    # Inspect split metadata globally without reading or decoding holdout bytes.
    errors = validate_splits(records, require_class_coverage=False)
    selected = [
        record
        for record in records
        if record.modality == config.modality and record.split in ("train", "dev")
    ]
    for split in ("train", "dev"):
        if {record.label for record in selected if record.split == split} != {0, 1}:
            errors.append(f"{split} requires both classes with known labels")
    audit = audit_manifest(
        [replace(record, split="unassigned") for record in selected],
        root,
        sources,
    )
    atomic_json(
        output / "audit.json",
        {
            "audit": audit.to_dict(),
            "split_errors": errors,
            "scope": "train/dev bytes only; original full manifest split metadata checked",
        },
    )
    if errors or not audit.passed:
        raise ValueError("experiment data audit failed; inspect audit.json")
    return selected


def _prepare_cache(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917 -- cache and media batching
    records: list[SampleRecord],
    config: ExperimentConfig,
    root: Path,
    cache: Path,
    encoder: Encoder | None,
    output: Path,
) -> dict[str, list[tuple[Path, str, int]]]:
    audio_config, video_config = config.media_configs
    pipeline_files = ["media/__init__.py"]
    if encoder is not None:
        pipeline_files.extend(("features.py", "models/encoders.py"))
    pipeline_hash = object_hash(
        {name: file_sha256(Path(__file__).parent / name) for name in pipeline_files}
    )
    preprocessing: dict[str, Any] = (
        {
            "audio": asdict(audio_config),
            "segment_samples": config.segment_samples,
            "segments": config.segments,
        }
        if config.modality == "audio"
        else {"video": asdict(video_config)}
    )
    cache.mkdir(parents=True, exist_ok=True)
    cache_bytes = sum(path.stat().st_size for path in cache.iterdir() if path.is_file())
    if cache_bytes > config.cache_max_bytes:
        raise ValueError("existing tensor cache exceeds cache_max_bytes")
    entries: dict[str, list[tuple[Path, str, int]]] = {"train": [], "dev": []}
    failures: list[dict[str, str]] = []
    expected_shape: tuple[int, ...] | None = None

    def cache_identity(record: SampleRecord) -> dict[str, Any]:
        return {
            "cache_version": 2,
            "media_sha256": record.sha256,
            "encoder": config.encoder,
            "feature_precision": (
                config.train_config.amp_dtype
                if isinstance(encoder, FrozenAudioEncoder) and config.train_config.amp
                else "float32"
            ),
            "preprocessing": preprocessing,
            "pipeline_sha256": pipeline_hash,
        }

    chunk_size = (
        max(1, config.feature_batch_size // config.segments)
        if isinstance(encoder, FrozenAudioEncoder)
        else 1
    )
    for offset in range(0, len(records), chunk_size):
        chunk = records[offset : offset + chunk_size]
        prepared: dict[str, Tensor] = {}
        if isinstance(encoder, FrozenAudioEncoder):
            missing = [
                record
                for record in chunk
                if not (cache / f"{object_hash(cache_identity(record))}.pt").exists()
            ]
            if missing:
                try:
                    # Training AMP also applies to the frozen audio feature pass.
                    with torch.autocast(
                        device_type=torch.device(config.train_config.device).type,
                        dtype=getattr(torch, config.train_config.amp_dtype),
                        enabled=config.train_config.amp,
                    ):
                        tensors = extract_audio_feature_batch(
                            missing,
                            root,
                            encoder,
                            audio_config,
                            config.segment_samples,
                            config.segments,
                            config.train_config.device,
                            config.feature_batch_size,
                        )
                    prepared = {
                        record.sample_id: tensor
                        for record, tensor in zip(missing, tensors, strict=True)
                    }
                except Exception as exc:
                    atomic_json(
                        output / "decode_failures.json",
                        [
                            {
                                "sample_id": record.sample_id,
                                "stage": "audio_feature_batch",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            for record in missing
                        ],
                    )
                    raise ValueError(
                        "sample preparation failed; inspect decode_failures.json"
                    ) from exc
        for record in chunk:
            identity = cache_identity(record)
            path = cache / f"{object_hash(identity)}.pt"
            try:
                if path.exists():
                    sidecar = json.loads(path.with_suffix(".json").read_text())
                    if sidecar["identity"] != identity:
                        raise ValueError("tensor cache identity mismatch")  # noqa: TRY301 -- per-sample failure record
                    digest = sidecar["sha256"]
                    tensor = _load_tensor(path, digest)
                else:
                    if record.sample_id in prepared:
                        tensor = prepared.pop(record.sample_id)
                    elif encoder is None:
                        decoded = decode_audio(media_path(record, root), audio_config)
                        tensor = torch.from_numpy(
                            sample_audio_segments(
                                decoded.waveform, config.segment_samples, 1
                            )[0]
                        )
                    else:
                        tensor = extract_features(
                            record,
                            root,
                            encoder,
                            audio_config,
                            video_config,
                            config.segment_samples,
                            config.segments,
                            config.train_config.device,
                            config.feature_batch_size,
                        )
                    previous_pair_bytes = _cache_pair_bytes(path)
                    digest = _save_tensor(
                        path,
                        tensor,
                        identity,
                        config.cache_max_bytes,
                        cache_bytes=cache_bytes,
                    )
                    cache_bytes += _cache_pair_bytes(path) - previous_pair_bytes
                shape = tuple(tensor.shape)
                if (
                    not shape
                    or any(size <= 0 for size in shape)
                    or (expected_shape is not None and shape != expected_shape)
                ):
                    raise ValueError(  # noqa: TRY301 -- per-media failure accounting
                        "prepared tensors must have one fixed nonempty shape"
                    )
                expected_shape = shape
                entries[record.split].append((path, digest, cast(int, record.label)))
            except (
                Exception
            ) as exc:  # sample-boundary accounting; every failure is fatal below
                failures.append(
                    {
                        "sample_id": record.sample_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    atomic_json(output / "decode_failures.json", failures)
    if failures:
        raise ValueError("sample preparation failed; inspect decode_failures.json")
    return entries


def run_experiment(config_path: Path) -> TrainResult:
    """Audit bytes and splits, prepare train/dev only, train and record provenance."""
    config = load_experiment_config(config_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    base = config_path.resolve().parent
    output, root = (
        (base / config.output_dir).resolve(),
        (base / config.data_root).resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("experiment output_dir must be empty")
    started = time.monotonic()
    code = code_identity()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "preparing",
        "started_at": utc_now(),
        "config": asdict(config),
        "config_sha256": object_hash(asdict(config)),
        "code": code,
        "holdout_accessed": False,
    }
    atomic_json(output / "experiment.json", report)
    try:
        records = load_manifest(base / config.manifest)
        source_path = base / config.sources
        sources = load_sources(source_path)
        selected = _audit_training_records(records, config, root, sources, output)
        selected_sources = {record.source for record in selected}
        report["sources"] = {
            key: asdict(value)
            for key, value in sources.items()
            if key in selected_sources
        }
        train_config = config.train_config
        torch.manual_seed(train_config.seed)
        model = create_model(config.architecture, config.model_config)
        initial = save_checkpoint(output / "initial.pt", model)
        report["initial_checkpoint"] = initial
        encoder = (
            load_encoder(config.encoder, base, train_config.device)
            if config.encoder
            else None
        )
        cache = (
            (base / config.feature_cache).resolve()
            if config.feature_cache is not None
            else output / "prepared"
        )
        entries = _prepare_cache(selected, config, root, cache, encoder, output)
        # Release the frozen backbone before allocating optimizer state.
        del encoder
        generator = torch.Generator().manual_seed(train_config.seed)
        loader_options: dict[str, Any] = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": train_config.device.startswith("cuda"),
            "worker_init_fn": _worker_seed,
        }
        if config.num_workers:
            loader_options["multiprocessing_context"] = "spawn"
            loader_options["timeout"] = 30
            loader_options["persistent_workers"] = True
        sampler = (
            _balanced_sampler(entries["train"], generator)
            if config.balanced_sampling
            else None
        )
        train_loader = DataLoader(
            _CachedDataset(entries["train"]),
            shuffle=sampler is None,
            sampler=sampler,
            generator=generator,
            **loader_options,
        )
        dev_loader = DataLoader(
            _CachedDataset(entries["dev"]), shuffle=False, **loader_options
        )
        provenance = {
            "data_sha256": manifest_hash(selected),
            "code_sha256": code["code_sha256"],
            "weights_sha256": initial["sha256"],
            "manifest_sha256": manifest_hash(records),
            "sources_sha256": file_sha256(source_path),
            "encoder_sha256": object_hash(config.encoder),
        }
        report.update(
            status="training",
            provenance=provenance,
            train_count=len(entries["train"]),
            dev_count=len(entries["dev"]),
        )
        atomic_json(output / "experiment.json", report)
        result = train_model(
            model,
            cast(Any, train_loader),
            cast(Any, dev_loader),
            config=train_config,
            output_dir=output / "training",
            provenance=provenance,
            architecture=config.architecture,
            model_config=initial["config"],
        )
        _write_dev_scores(
            model,
            dev_loader,
            [record for record in selected if record.split == "dev"],
            train_config.device,
            output,
        )
        report.update(
            status="complete",
            best_epoch=result.best_epoch,
            best_dev_eer=result.best_dev_eer,
            checkpoint_sha256=result.checkpoint_sha256,
            checkpoint=str(result.best_checkpoint.relative_to(output)),
        )
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(finished_at=utc_now(), seconds=time.monotonic() - started)
        atomic_json(output / "experiment.json", report)
    return result
