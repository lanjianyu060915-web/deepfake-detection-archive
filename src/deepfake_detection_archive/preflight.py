"""Read-only training checks, shared with the real experiment's data gate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import load_object
from .data.audit import AuditReport, audit_manifest
from .data.manifest import SampleRecord, SourceRecord, load_manifest, load_sources
from .data.split import validate_splits
from .features import validate_encoder_config
from .models.ast import ASTConfig
from .models.checkpoint import verify_sha256
from .models.core import LogMelConfig, MediaFeatureConfig
from .models.encoders import VISION_ARCHITECTURES, verify_audio_bundle
from .models.finetune import VideoFrameConfig


if TYPE_CHECKING:
    from .experiments import ExperimentConfig


@dataclass(frozen=True)
class TrainingDataAudit:
    records: list[SampleRecord]
    audit: AuditReport
    split_errors: list[str]

    @property
    def passed(self) -> bool:
        return self.audit.passed and not self.split_errors

    def to_dict(self) -> dict[str, object]:
        return {
            "audit": self.audit.to_dict(),
            "split_errors": self.split_errors,
            "scope": (
                "train/dev registry and bytes; full split metadata checked; "
                "actual decoding or verified feature cache required by preparation"
            ),
        }


def audit_training_data(
    records: list[SampleRecord],
    config: ExperimentConfig,
    root: Path,
    sources: dict[str, SourceRecord],
) -> TrainingDataAudit:
    """Inspect all split metadata but only consume selected train/dev bytes."""
    errors = validate_splits(records, require_class_coverage=False)
    selected = [
        record
        for record in records
        if record.modality == config.modality and record.split in ("train", "dev")
    ]
    for split in ("train", "dev"):
        if {record.label for record in selected if record.split == split} != {0, 1}:
            errors.append(f"{split} requires both classes with known labels")
    if config.dev_domain_field is not None:
        for domain in config.train_config.domain_constraints:
            labels = {
                record.label
                for record in selected
                if record.split == "dev"
                and getattr(record, config.dev_domain_field) == domain
            }
            if labels != {0, 1}:
                errors.append(
                    f"constrained dev domain {domain!r} requires both classes "
                    f"in {config.dev_domain_field}"
                )
    # Full metadata already passed through the split gate above. The general
    # audit otherwise requires holdout class coverage, irrelevant to training.
    audit = audit_manifest(
        [replace(record, split="unassigned") for record in selected],
        root,
        sources,
        require_decoded=False,
    )
    return TrainingDataAudit(selected, audit, errors)


def _validate_model_config(config: ExperimentConfig) -> None:
    """Parse existing model schemas without allocating model tensors."""
    if config.architecture == "logmel_cnn_v1":
        LogMelConfig(**config.model_config)
    elif config.architecture == "media_feature_head_v1":
        MediaFeatureConfig(**config.model_config)
    elif config.architecture == "video_frame_finetune_v1":
        VideoFrameConfig(**config.model_config)
    elif config.architecture == "audio_ast_v1":
        ast = ASTConfig(**config.model_config)
        if not isinstance(ast.backbone_config, dict):
            raise ValueError("AST backbone_config must be an object")
        if ast.waveform_preparation != "legacy" and config.segment_samples != 64000:
            raise ValueError(
                "AST center_pad_64000_peak095_v1 requires segment_samples=64000"
            )
    elif config.model_config:
        raise ValueError("AASIST-L model_config must be empty")


def _local_assets(config: ExperimentConfig, base: Path) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []

    def checkpoint(path: object, digest: object) -> Path:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("checkpoint must be a nonempty local path")
        if not isinstance(digest, str):
            raise TypeError("checkpoint hash must be a SHA-256 string")
        resolved = (base / path).resolve()
        verify_sha256(resolved, digest)
        assets.append({"path": str(resolved), "sha256": digest})
        return resolved

    if config.pretrained:
        path = config.pretrained["checkpoint"]
        if isinstance(path, str) and Path(path).name != "model.safetensors":
            raise ValueError("AST pretrained weights must be named model.safetensors")
        verified = checkpoint(path, config.pretrained["sha256"])
        # HF loads this sibling metadata, even though the experiment schema
        # currently pins only the weights. Parsing does not certify its provenance.
        load_object(verified.parent / "config.json")
    if config.architecture == "video_frame_finetune_v1":
        checkpoint(
            config.model_config["encoder_checkpoint"],
            config.model_config["expected_sha256"],
        )
    if config.encoder:
        validate_encoder_config(config.encoder)
        if config.encoder["kind"] == "vision":
            if config.encoder["architecture"] not in VISION_ARCHITECTURES:
                raise ValueError("unsupported local vision architecture")
            checkpoint(config.encoder["checkpoint"], config.encoder["expected_sha256"])
        else:
            root = verify_audio_bundle(
                base / config.encoder["model_dir"], config.encoder["expected_hashes"]
            )
            assets.extend(
                {"path": str(root / name), "sha256": digest}
                for name, digest in sorted(config.encoder["expected_hashes"].items())
            )
    return assets


def check_training_config(config_path: Path) -> dict[str, object]:
    """Check a launch on CPU, without model loading, decoding or filesystem writes.

    CUDA configurations remain checkable on CPU-only hosts. Passing certifies
    only the listed checks; it is not a training, decoding or GPU-capacity result.
    """
    from .experiments import load_experiment_config  # noqa: PLC0415 -- shared schema

    config = load_experiment_config(config_path)
    base = config_path.resolve().parent
    _validate_model_config(config)
    output = (base / config.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("experiment output_dir must be absent or empty")
    assets = _local_assets(config, base)
    records = load_manifest(base / config.manifest)
    data = audit_training_data(
        records, config, base / config.data_root, load_sources(base / config.sources)
    )
    return {
        "mode": "check-config",
        "passed": data.passed,
        "architecture": config.architecture,
        "modality": config.modality,
        "requested_device": config.train_config.device,
        "train_samples": sum(record.split == "train" for record in data.records),
        "dev_samples": sum(record.split == "dev" for record in data.records),
        "excluded_samples": len(records) - len(data.records),
        "holdout_media_accessed": False,
        "local_assets": assets,
        "output_dir": str(output),
        **data.to_dict(),
        "unchecked": [
            "media decoding and cached tensor validity",
            "model construction, weight compatibility and encoder feature dimensions",
            "CUDA availability, GPU memory, throughput and training results",
            "output/cache write permissions and free disk capacity",
            "provenance of domain EER baselines",
        ]
        + (
            ["AST pretrained config.json integrity (no trusted digest configured)"]
            if config.pretrained
            else []
        ),
    }
