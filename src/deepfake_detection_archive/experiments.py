"""Audited train/dev experiments backed by content-bound local tensor caches."""

import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augmentation import BandwidthSpec, bandlimit_roundtrip
from .config import load_object
from .contracts import ScoreRecord
from .data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    load_manifest,
    load_sources,
    manifest_hash,
)
from .data.split import statistical_groups
from .evaluation.report import evaluate_records, grouped_reports
from .features import (
    Encoder,
    extract_audio_feature_batch,
    extract_features,
    extract_video_frames,
    load_encoder,
    media_path,
)
from .media import AudioConfig, VideoConfig, decode_audio, sample_audio_segments
from .models.ast import ASTClassifier
from .models.checkpoint import create_model, save_checkpoint, verify_sha256
from .models.encoders import FrozenAudioEncoder
from .models.finetune import VideoFrameClassifier
from .preflight import audit_training_data
from .tracking import atomic_json, canonical_json, code_identity, object_hash, utc_now
from .training import SelectionOptions, TrainConfig, TrainResult, train_model


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
    balance_sources: bool = False
    balance_generators: bool = False
    content_type_weights: dict[str, float] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    segment_samples: int = 64600
    segments: int = 1
    audio: dict[str, Any] = field(default_factory=dict)
    video: dict[str, Any] = field(default_factory=dict)
    encoder: dict[str, Any] = field(default_factory=dict)
    pretrained: dict[str, Any] = field(default_factory=dict)
    augmentation: dict[str, Any] = field(default_factory=dict)
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
        for name in ("balanced_sampling", "balance_sources", "balance_generators"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.balance_generators and (
            self.modality != "video"
            or not self.balanced_sampling
            or self.balance_sources
        ):
            raise ValueError(
                "balance_generators requires video balanced_sampling without "
                "balance_sources"
            )
        if self.balance_sources and not self.balanced_sampling:
            raise ValueError("balance_sources requires balanced_sampling")
        if not isinstance(self.content_type_weights, dict) or any(
            not isinstance(name, str)
            or not name.strip()
            or type(value) is bool
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for name, value in self.content_type_weights.items()
        ):
            raise ValueError(
                "content_type_weights must map content types to finite positive numbers"
            )
        if self.content_type_weights and not (
            self.balanced_sampling and self.balance_sources
        ):
            raise ValueError(
                "content_type_weights requires balanced_sampling and balance_sources"
            )
        if not isinstance(self.selection, dict):
            raise ValueError("selection must be an object")  # noqa: TRY004 -- schema
        _ = self.selection_options
        if self.modality not in ("audio", "video"):
            raise ValueError("modality must be audio or video")
        for name in (
            "model_config",
            "training",
            "audio",
            "video",
            "encoder",
            "pretrained",
            "augmentation",
        ):
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
            "audio_ast_v1",
            "video_frame_finetune_v1",
        ):
            raise ValueError("unsupported experiment architecture")
        if self.architecture == "video_frame_finetune_v1":
            if self.modality != "video" or self.segments != 1 or self.encoder:
                raise ValueError(
                    "video frame fine-tuning requires video, segments=1 and no encoder"
                )
        elif self.architecture != "media_feature_head_v1":
            if self.modality != "audio" or self.segments != 1 or self.encoder:
                raise ValueError(
                    "waveform training requires audio, segments=1 and no encoder"
                )
        elif self.encoder.get("kind") != (
            "audio" if self.modality == "audio" else "vision"
        ):
            raise ValueError("encoder kind must match experiment modality")
        if self.pretrained and (
            self.architecture != "audio_ast_v1"
            or set(self.pretrained) != {"checkpoint", "sha256"}
        ):
            raise ValueError("pretrained requires AST checkpoint and sha256")
        _ = self.bandwidth_spec
        _ = self.train_config
        audio, _ = self.media_configs
        if self.architecture == "aasist_l_v1" and (
            audio.sample_rate != 16000 or self.segment_samples != 64600
        ):
            raise ValueError("AASIST-L requires 16000 Hz and segment_samples=64600")
        if self.architecture in (
            "logmel_cnn_v1",
            "audio_ast_v1",
        ) and audio.sample_rate != self.model_config.get("sample_rate", 16000):
            raise ValueError("audio and model sample rates must agree")

    @cached_property
    def selection_options(self) -> SelectionOptions | None:
        return SelectionOptions.from_options(self.selection or None)

    @cached_property
    def bandwidth_spec(self) -> BandwidthSpec | None:
        """Return the configured train-only bandwidth augmentation, if any."""
        if not self.augmentation:
            return None
        unknown = set(self.augmentation) - {"bandwidth"}
        if unknown:
            raise ValueError(f"invalid augmentation options: {sorted(unknown)}")
        options = self.augmentation["bandwidth"]
        expected = {"probability", "target_sample_rate"}
        if not isinstance(options, dict) or set(options) != expected:
            raise ValueError(
                "bandwidth augmentation requires exactly probability and "
                "target_sample_rate"
            )
        if any(type(value) is bool for value in options.values()):
            raise ValueError("augmentation parameters cannot be boolean")
        spec = BandwidthSpec(**options)
        audio, _ = self.media_configs
        if self.architecture not in ("logmel_cnn_v1", "aasist_l_v1"):
            raise ValueError(
                "bandwidth augmentation requires LogMel or AASIST waveform training"
            )
        if spec.target_sample_rate >= audio.sample_rate:
            raise ValueError(
                "bandwidth augmentation target rate must be below the audio rate"
            )
        return spec

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
    """Verified cached tensors with optional train-only waveform augmentation.

    Augmentation runs after the cache lookup, so cached bytes stay identical to
    an unaugmented run and existing tensor caches remain reusable. The dev copy
    of this dataset is always built without a spec.
    """

    def __init__(
        self,
        entries: list[tuple[Path, str, int]],
        bandwidth: BandwidthSpec | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        self.entries = entries
        self.bandwidth = bandwidth
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # Cache preparation validates bytes and tensors once before training.
        path, _, label = self.entries[index]
        value = torch.load(path, map_location="cpu", weights_only=True)
        bandwidth = self.bandwidth
        if bandwidth is not None and _bandwidth_draw(bandwidth):
            value = bandlimit_roundtrip(value, bandwidth, self.sample_rate)
        return value, torch.tensor(label, dtype=torch.float32)


def _bandwidth_draw(spec: BandwidthSpec) -> bool:
    """Draw one augmentation decision from the seeded worker-local stream."""
    return random.random() < spec.probability  # noqa: S311


def _worker_seed(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- initialize worker-local legacy RNG users


def _balanced_sampler(  # noqa: PLR0913 -- explicit independent sampling options
    entries: list[tuple[Path, str, int]],
    generator: torch.Generator,
    sources: list[str] | None = None,
    content_types: list[str] | None = None,
    content_type_weights: dict[str, float] | None = None,
    *,
    generators: list[str] | None = None,
) -> WeightedRandomSampler:
    """Balance declared groups, optionally scaling whole content domains.

    Every (source, label) or label group receives an equal share unless
    content-type weights multiply the per-entry weight. Unlisted content types
    keep weight 1.0; classes inside a weighted domain stay balanced because the
    domain factor is constant across its groups. Generator balancing instead
    divides each label's equal mass between its generators, so the real/unknown
    bucket receives half the draws even when there are many fake generators.
    """
    keys = (
        [
            (source, label)
            for source, (_, _, label) in zip(sources, entries, strict=True)
        ]
        if sources is not None
        else [("all", label) for _, _, label in entries]
    )
    if generators is not None:
        keys = [
            (name, label)
            for name, (_, _, label) in zip(generators, entries, strict=True)
        ]
    counts = Counter(keys)
    groups_per_label = Counter(label for _, label in counts)
    weights = [
        1.0 / (counts[key] * groups_per_label[key[1]])
        if generators is not None
        else 1.0 / counts[key]
        for key in keys
    ]
    if content_type_weights:
        if content_types is None or len(content_types) != len(entries):
            raise ValueError("content_type_weights requires one content type per entry")
        if any(not isinstance(name, str) or not name.strip() for name in content_types):
            raise ValueError("content_type_weights requires labeled content types")
        absent = set(content_type_weights) - set(content_types)
        if absent:
            raise ValueError(
                f"content_type_weights names absent content types: {sorted(absent)}"
            )
        weights = [
            weight * content_type_weights.get(content_type, 1.0)
            for weight, content_type in zip(weights, content_types, strict=True)
        ]
    return WeightedRandomSampler(
        weights, len(entries), replacement=True, generator=generator
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
    groups = statistical_groups(records)
    scores = [
        ScoreRecord(
            sample_id=record.sample_id,
            modality=record.modality,
            score=value,
            status="ok",
            elapsed_seconds=seconds / len(records),
            label=record.label,
            group_id=groups[record.sample_id],
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
            "by_content_type": {
                name: asdict(report)
                for name, report in grouped_reports(scores, "content_type").items()
            },
        },
    )


def _audit_training_records(
    records: list[SampleRecord],
    config: ExperimentConfig,
    root: Path,
    sources: dict[str, SourceRecord],
    output: Path,
) -> list[SampleRecord]:
    data = audit_training_data(records, config, root, sources)
    atomic_json(output / "audit.json", data.to_dict())
    if not data.passed:
        raise ValueError("experiment data audit failed; inspect audit.json")
    return data.records


def _prepare_cache(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917 -- cache and media batching
    records: list[SampleRecord],
    config: ExperimentConfig,
    root: Path,
    cache: Path,
    encoder: Encoder | None,
    output: Path,
    audio_preprocessor: ASTClassifier | None = None,
    frame_preprocessing: dict[str, Any] | None = None,
) -> dict[str, list[tuple[Path, str, int]]]:
    audio_config, video_config = config.media_configs
    pipeline_files = ["media/__init__.py"]
    if encoder is not None:
        pipeline_files.extend(("features.py", "models/encoders.py"))
    if audio_preprocessor is not None:
        pipeline_files.append("models/ast.py")
    if frame_preprocessing is not None:
        pipeline_files.append("features.py")
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
    if audio_preprocessor is not None:
        preprocessing["ast"] = audio_preprocessor.preprocessing
    if frame_preprocessing is not None:
        preprocessing["frame_preprocessing"] = {
            key: list(frame_preprocessing[key]) for key in ("input_size", "mean", "std")
        }
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
        else max(1, config.num_workers)
        if frame_preprocessing is not None
        else 1
    )
    for offset in range(0, len(records), chunk_size):
        chunk = []
        for record in records[offset : offset + chunk_size]:
            identity = cache_identity(record)
            chunk.append((record, identity, cache / f"{object_hash(identity)}.pt"))
        prepared: dict[str, Tensor] = {}
        if isinstance(encoder, FrozenAudioEncoder):
            missing = [record for record, _, path in chunk if not path.exists()]
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
        frame_futures: dict[str, Future[Tensor]] = {}
        if frame_preprocessing is not None and config.num_workers > 0:
            with ThreadPoolExecutor(max_workers=config.num_workers) as pool:
                frame_futures = {
                    record.sample_id: pool.submit(
                        extract_video_frames,
                        record,
                        root,
                        video_config,
                        frame_preprocessing,
                    )
                    for record, _, path in chunk
                    if not path.exists()
                }
        for record, identity, path in chunk:
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
                    elif record.sample_id in frame_futures:
                        tensor = frame_futures.pop(record.sample_id).result()
                    elif frame_preprocessing is not None:
                        tensor = extract_video_frames(
                            record, root, video_config, frame_preprocessing
                        )
                    elif encoder is None:
                        decoded = decode_audio(media_path(record, root), audio_config)
                        if audio_preprocessor is not None:
                            tensor = audio_preprocessor.prepare_audio(
                                decoded.waveform, config.segment_samples, 1
                            )[0]
                        else:
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


def run_experiment(config_path: Path) -> TrainResult:  # noqa: PLR0915 -- experiment lifecycle
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
        if isinstance(model, ASTClassifier):
            model.validate_audio_sampling(config.segment_samples, config.segments)
        if isinstance(model, ASTClassifier) and config.pretrained:
            model.initialize_pretrained(
                (base / config.pretrained["checkpoint"]).resolve(),
                config.pretrained["sha256"],
            )
        if isinstance(model, VideoFrameClassifier):
            model.initialize_pretrained(
                (base / model.config.encoder_checkpoint).resolve(),
                model.config.expected_sha256,
            )
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
        entries = _prepare_cache(
            selected,
            config,
            root,
            cache,
            encoder,
            output,
            audio_preprocessor=model if isinstance(model, ASTClassifier) else None,
            frame_preprocessing=(
                model.preprocessing if isinstance(model, VideoFrameClassifier) else None
            ),
        )
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
            _balanced_sampler(
                entries["train"],
                generator,
                [record.source for record in selected if record.split == "train"]
                if config.balance_sources
                else None,
                content_types=(
                    [
                        record.content_type
                        for record in selected
                        if record.split == "train"
                    ]
                    if config.content_type_weights
                    else None
                ),
                content_type_weights=config.content_type_weights or None,
                generators=(
                    [record.generator for record in selected if record.split == "train"]
                    if config.balance_generators
                    else None
                ),
            )
            if config.balanced_sampling
            else None
        )
        train_loader = DataLoader(
            _CachedDataset(
                entries["train"],
                bandwidth=config.bandwidth_spec,
                sample_rate=config.media_configs[0].sample_rate,
            ),
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
            "augmentation": config.augmentation,
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
            selection=config.selection_options,
            dev_domains=(
                [record.content_type for record in selected if record.split == "dev"]
                if config.selection
                else None
            ),
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
            best_selection_metric=result.best_selection_metric,
            best_selection_value=result.best_selection_value,
            checkpoint_sha256=result.checkpoint_sha256,
            checkpoint=str(result.best_checkpoint.relative_to(output)),
            selection_gate_passed=result.selection_gate_passed,
        )
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(finished_at=utc_now(), seconds=time.monotonic() - started)
        atomic_json(output / "experiment.json", report)
    return result
