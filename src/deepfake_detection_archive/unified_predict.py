"""Local media discovery and default-model orchestration over the existing runner."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, cast

from .config import InferenceConfig, load_inference_config, load_object
from .contracts import load_scores
from .data.manifest import SampleRecord, file_sha256, write_manifest
from .media_tools import media_tool
from .tracking import (
    atomic_json,
    canonical_json,
    code_identity,
    object_hash,
    parse_json,
)


NON_MEDIA_SUFFIXES = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
    ".pdf",
    ".zip",
    ".tar",
    ".xz",
    ".py",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
}
MEDIA_SUFFIXES = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".aac",
    ".opus",
    ".mp4",
    ".mkv",
    ".avi",
    ".webm",
    ".mov",
    ".mpeg",
    ".m4v",
    ".wma",
    ".aiff",
    ".aif",
    ".ts",
    ".mts",
}


def _model_package_root() -> Path:
    selected = os.environ.get("DEEPFAKE_MODEL_PACKAGE")
    if selected is not None:
        return Path(selected).expanduser().resolve()
    module = Path(__file__).resolve()
    if module.parent.parent.name != "src":
        raise FileNotFoundError(
            "Set DEEPFAKE_MODEL_PACKAGE to a local model package containing "
            "configs/models.json; the installed wheel does not include weights"
        )
    return module.parents[2]


def _package_resource(root: Path, base: Path, resource: str) -> Path:
    """Resolve default-package resources once, including local symlink targets."""
    path = (base / resource).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"resource escapes model package: {base / resource}")
    return path


def default_models() -> tuple[Path, dict[str, Any]]:
    """Use one model package; only source layouts have an implicit default."""
    root = _model_package_root()
    folder = root / "configs"
    registry_path = _package_resource(root, folder, "models.json")
    if os.environ.get("DEEPFAKE_MODEL_PACKAGE") is None and not registry_path.is_file():
        folder = folder / "release"
        registry_path = _package_resource(root, folder, "models.json")
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Model registry not found: {registry_path}; "
            "DEEPFAKE_MODEL_PACKAGE must point to a package containing configs/models.json"
        )
    registry = load_object(registry_path)
    if registry.get("schema_version") != 1 or set(
        cast(dict[str, Any], registry.get("models", {}))
    ) != {"audio", "video"}:
        raise ValueError("unsupported default model registry")
    return folder, registry


def probe_media(path: Path) -> tuple[str, int]:
    """Use actual local streams, excluding attached pictures from visual scoring."""
    from .media import MediaError, _run  # noqa: PLC0415

    raw = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        15.0,
        1_048_576,
    )
    streams = sorted(
        cast(
            list[dict[str, Any]],
            cast(dict[str, Any], parse_json(raw)).get("streams", []),
        ),
        key=lambda stream: int(stream["index"]),
    )
    for stream in streams:
        if (
            stream.get("codec_type") == "video"
            and not stream.get("disposition", {}).get("attached_pic", 0)
            and int(stream.get("width", 0)) > 0
            and int(stream.get("height", 0)) > 0
        ):
            return "video", int(stream["index"])
    for stream in streams:
        if (
            stream.get("codec_type") == "audio"
            and int(stream.get("sample_rate", 0)) > 0
        ):
            return "audio", int(stream["index"])
    raise MediaError(
        "unsupported_media", "no usable audio or non-attached video stream"
    )


def discover(
    inputs: list[Path], output: Path
) -> tuple[list[SampleRecord], list[dict[str, Any]]]:
    """Deterministic path identities; never collapse distinct byte-identical files."""
    records: list[SampleRecord] = []
    entries: list[dict[str, Any]] = []
    seen: dict[Path, str] = {}

    def add(path: Path, alias: str, explicit: bool) -> None:
        absolute = Path(os.path.abspath(path))  # noqa: PTH100 -- preserve symlink identity
        entry: dict[str, Any] = {"input": alias, "path": str(absolute)}
        if path.is_symlink():
            entries.append({**entry, "status": "error", "reason": "restricted_symlink"})
            return
        try:
            canonical = path.resolve(strict=True)
            if canonical.is_relative_to(output):
                entries.append(
                    {**entry, "status": "excluded", "reason": "output_directory"}
                )
                return
            if canonical in seen:
                entries.append(
                    {**entry, "status": "duplicate", "sample_id": seen[canonical]}
                )
                return
            if not canonical.is_file():
                entries.append({**entry, "status": "error", "reason": "not_file"})
                return
            if not explicit and canonical.suffix.lower() in NON_MEDIA_SUFFIXES:
                entries.append({**entry, "status": "excluded", "reason": "non_media"})
                return
            sample_id = hashlib.sha256(str(canonical).encode()).hexdigest()
            seen[canonical] = sample_id
            entry["sample_id"] = sample_id
            modality, stream_index = probe_media(canonical)
            digest = file_sha256(canonical)
            record = SampleRecord(
                sample_id,
                canonical.as_posix().lstrip("/"),
                modality,
                None,
                "unlabeled-local-input",
                sha256=digest,
                split="test",
            )
            records.append(record)
            entries.append(
                {
                    **entry,
                    "path": str(canonical),
                    "status": "media",
                    "sample_id": sample_id,
                    "modality": modality,
                    "stream_index": stream_index,
                    "sha256": digest,
                }
            )
        except (OSError, ValueError, RuntimeError) as exc:
            # Unknown directory items are exclusions; explicit inputs and media
            # containers that fail bounded probing remain errors.
            status = (
                "error"
                if explicit
                or path.suffix.lower() in MEDIA_SUFFIXES
                or isinstance(exc, OSError)
                else "excluded"
            )
            entries.append(
                {
                    **entry,
                    "status": status,
                    "reason": getattr(exc, "code", type(exc).__name__),
                }
            )

    def visit(path: Path, alias: str, explicit: bool) -> None:
        if path.is_symlink():
            add(path, alias, explicit)
            return
        try:
            if path.resolve().is_relative_to(output):
                entries.append(
                    {
                        "input": alias,
                        "path": str(path.absolute()),
                        "status": "excluded",
                        "reason": "output_directory",
                    }
                )
                return
            if not path.is_dir():
                add(path, alias, explicit)
                return
            children = sorted(path.iterdir(), key=lambda item: item.name)
        except (OSError, RuntimeError) as exc:
            entries.append(
                {
                    "input": alias,
                    "path": str(path.absolute()),
                    "status": "error",
                    "reason": type(exc).__name__,
                }
            )
            return
        for child in children:
            visit(child, f"{alias}/{child.name}", False)

    for index, path in enumerate(inputs):
        visit(path, f"input{index + 1}:{path}", True)
    return records, entries


def _resolved_config(
    folder: Path, entry: dict[str, Any], output: Path, modality: str, device: str
) -> InferenceConfig:
    root = _model_package_root()
    config_path = _package_resource(root, folder, entry["config"])
    template = load_inference_config(config_path)
    if template.modality != modality:
        raise ValueError("registry and inference modality disagree")
    encoder = dict(template.encoder)
    for key in ("checkpoint", "model_dir"):
        if key in encoder:
            encoder[key] = str(
                _package_resource(root, config_path.parent, str(encoder[key]))
            )
    if "model_dir" in encoder:
        model_dir = Path(str(encoder["model_dir"]))
        resources = {path.name for path in model_dir.iterdir()}
        resources.update(cast(dict[str, str], encoder.get("expected_hashes", {})))
        for name in resources:
            _package_resource(root, model_dir, name)
    return replace(
        template,
        manifest=str(output / "manifest.jsonl"),
        data_root="/",
        output_dir=str(output),
        checkpoint=str(
            _package_resource(root, config_path.parent, template.checkpoint)
        ),
        encoder=encoder,
        device=device,
    )


def predict_media(  # noqa: PLR0913 -- media and explicit execution controls
    inputs: list[Path],
    output: Path | None,
    device: str,
    *,
    resume: bool,
    execution_mode: str | None = None,
    batch_size: int | None = None,
) -> dict[str, object]:
    import torch  # noqa: PLC0415

    from .submission.runner import _run_lock  # noqa: PLC0415

    if resume and output is None:
        raise ValueError("--resume requires --output")
    if output is None:
        output = Path(tempfile.mkdtemp(prefix="deepfake-predict-", dir=Path.cwd()))
    else:
        output = output.resolve()
        if resume:
            if not output.is_dir():
                raise ValueError("resume output does not exist")
        else:
            output.mkdir(parents=True, exist_ok=False)
    print(f"Results: {output}", file=sys.stderr)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    with _run_lock(output / ".predict.lock"):
        try:
            return _predict_run(
                inputs,
                output,
                device,
                resume=resume,
                execution_mode=execution_mode,
                batch_size=batch_size,
            )
        except (OSError, ValueError, RuntimeError, ImportError, MemoryError) as exc:
            if not resume:
                atomic_json(
                    output / "report.json",
                    {"failures": 1, "global_error": f"{type(exc).__name__}: {exc}"},
                )
            raise


def _validate_ast_frontend(
    folder: Path, registry: dict[str, Any], runtime: dict[str, str | None]
) -> dict[str, object]:
    """Validate the selected frontend, independent of unused installed packages."""
    release = load_object(
        _package_resource(
            _model_package_root(), folder.parent, registry["runtime_contract"]
        )
    )
    expected = cast(dict[str, str], release["runtime_dependencies"])
    frontend = release.get(
        "ast_frontend",
        "torchaudio_kaldi_v1" if "torchaudio" in expected else "numpy_spectrogram_v1",
    )
    if frontend not in {"numpy_spectrogram_v1", "torchaudio_kaldi_v1"}:
        raise ValueError("unsupported release AST frontend")
    uses_torchaudio = frontend == "torchaudio_kaldi_v1"
    implementation = (
        ("transformers", "numpy", "torch", "torchaudio")
        if uses_torchaudio
        else ("transformers", "numpy")
    )
    for name in implementation:
        if runtime[name] != expected[name]:
            raise ValueError(
                f"incompatible AST frontend: {name}={runtime[name]}, release requires {expected[name]}"
            )
    return {
        "frontend": frontend,
        "versions": {name: expected[name] for name in implementation},
    }


def _score_selected(
    record: SampleRecord,
    *,
    scorer: Any,
    selected: dict[str, int],
    stream_configs: dict[int, Any],
    config_key: str,
) -> float:
    return float(
        scorer(record, **{config_key: stream_configs[selected[record.sample_id]]})
    )


def _predict_run(  # noqa: C901, PLR0912, PLR0913, PLR0915 -- existing runner orchestration
    inputs: list[Path],
    output: Path,
    device: str,
    *,
    resume: bool,
    execution_mode: str | None = None,
    batch_size: int | None = None,
) -> dict[str, object]:
    import torch  # noqa: PLC0415

    from .execution import execution_signature  # noqa: PLC0415
    from .media import _run  # noqa: PLC0415
    from .models.ast import ASTClassifier  # noqa: PLC0415
    from .models.checkpoint import sha256_file  # noqa: PLC0415
    from .scoring import OfflineScorer  # noqa: PLC0415
    from .submission.runner import run_inference  # noqa: PLC0415

    folder, registry = default_models()
    records, entries = discover(inputs, output)
    models = cast(dict[str, dict[str, Any]], registry["models"])
    configs: dict[str, InferenceConfig] = {}
    global_error: str | None = None
    media_tools: dict[str, str] = {}
    media_tools_sha256: dict[str, str] = {}
    execution: dict[str, object] = {}
    try:
        media_tools = {
            name: _run([name, "-version"], 5.0, 65_536).decode("utf-8")
            for name in ("ffmpeg", "ffprobe")
        }
        for name in media_tools:
            executable = shutil.which(media_tool(name))
            if executable is None:
                raise FileNotFoundError(f"{name} unavailable")
            media_tools_sha256[name] = sha256_file(Path(executable))
        for modality in ("audio", "video"):
            if any(row.modality == modality for row in records):
                config = _resolved_config(
                    folder, models[modality], output / modality, modality, device
                )
                configs[modality] = replace(
                    config,
                    execution_mode=execution_mode or config.execution_mode,
                    batch_size=(
                        batch_size if batch_size is not None else config.batch_size
                    ),
                )
                execution[modality] = execution_signature(
                    device, configs[modality].execution_mode
                )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        global_error = f"{type(exc).__name__}: {exc}"
    package = Path(__file__).parent
    inference_paths = [
        package / name
        for name in (
            "scoring.py",
            "media_tools.py",
            "features.py",
            "config.py",
            "execution.py",
            "contracts.py",
            "timing.py",
            "unified_predict.py",
        )
    ]
    for name in ("models", "media", "submission", "evaluation", "data"):
        inference_paths.extend((package / name).rglob("*.py"))
    inference_code = {
        path.relative_to(package).as_posix(): file_sha256(path)
        for path in sorted(inference_paths)
    }
    runtime: dict[str, str | None] = {}
    for name in (
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "numpy",
        "torchaudio",
        "soundfile",
    ):
        try:
            runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime[name] = None
    frontend_contract: dict[str, object] = {}
    if any(models[modality]["architecture"] == "audio_ast_v1" for modality in configs):
        try:
            frontend_contract = _validate_ast_frontend(folder, registry, runtime)
        except (OSError, ValueError, ImportError) as exc:
            global_error = f"{type(exc).__name__}: {exc}"
    identity = {
        "frontend_contract": frontend_contract,
        "execution": execution,
        "media_tools": media_tools,
        "media_tools_sha256": media_tools_sha256,
        "runtime": runtime,
        "protocol": "unified_media_v1",
        "discovery": entries,
        "registry": registry,
        "configs": {key: value.to_dict() for key, value in configs.items()},
        "inference_code": inference_code,
    }
    metadata = output / "input.json"
    if resume:
        if not metadata.exists() or object_hash(load_object(metadata)) != object_hash(
            identity
        ):
            raise ValueError(
                "resume rejected: input, routing, model or configuration changed"
            )
    else:
        atomic_json(metadata, identity)
    write_manifest(records, output / "manifest.jsonl")
    atomic_json(output / "discovery.json", {"entries": entries})
    atomic_json(output / "provenance.json", code_identity())
    selected = {
        entry["sample_id"]: entry["stream_index"]
        for entry in entries
        if entry["status"] == "media"
    }
    for modality, config in configs.items():
        if global_error is not None:
            break
        subset = [row for row in records if row.modality == modality]
        scorer: OfflineScorer | None = None
        try:
            scorer = OfflineScorer(
                config,
                folder,
                resume=resume and (output / modality / "metadata.json").exists(),
            )
            if (
                scorer.metadata["architecture"] != models[modality]["architecture"]
                or scorer.metadata["schema_version"]
                != models[modality]["checkpoint_schema"]
            ):
                raise ValueError(  # noqa: TRY301 -- global failure recorded below
                    "default model architecture/schema differs from registry"
                )
            if (
                scorer.metadata["architecture"] == "audio_ast_v1"
                and frontend_contract
                and cast(ASTClassifier, scorer.model).config.frontend
                != frontend_contract["frontend"]
            ):
                raise ValueError(  # noqa: TRY301 -- recorded as global failure below
                    "checkpoint AST frontend differs from release"
                )
            decoder_config = scorer.audio if modality == "audio" else scorer.video
            stream_configs = {
                index: replace(decoder_config, stream_index=index)
                for index in {selected[row.sample_id] for row in subset}
            }
            run_inference(
                subset,
                partial(
                    _score_selected,
                    scorer=scorer,
                    selected=selected,
                    stream_configs=stream_configs,
                    config_key=f"{modality}_config",
                ),
                output / modality,
                identity={
                    "unified_identity": object_hash(identity),
                    "config": config.to_dict(),
                },
                resume=resume and (output / modality / "metadata.json").exists(),
                fatal_exceptions=(torch.OutOfMemoryError,),
            )
        except (OSError, ValueError, RuntimeError, ImportError, MemoryError) as exc:
            global_error = f"{type(exc).__name__}: {exc}"
            break
        finally:
            if scorer is not None:
                scorer.execution.finish()
                del scorer
    rows: list[dict[str, Any]] = []
    scored = {
        row.sample_id: row
        for modality in configs
        for row in (
            load_scores(output / modality / "scores.jsonl")
            if (output / modality / "scores.jsonl").exists()
            else []
        )
    }
    for record in records:
        score = scored.get(record.sample_id)
        entry = models[record.modality]
        rows.append(
            {
                "sample_id": record.sample_id,
                "path": "/" + record.path,
                "modality": record.modality,
                "stream_index": selected[record.sample_id],
                "model_id": entry["model_id"],
                "model_status": entry["status"],
                "scope": entry["scope"],
                "preprocessing_version": entry["preprocessing_version"],
                "score_direction": registry["score_direction"],
                "score_type": registry["score_type"],
                "score": score.score if score else None,
                "status": score.status if score else "not_run",
                "error": score.error_message if score else global_error,
            }
        )
    (output / "results.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    with (output / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sample_id", "path", "modality", "model_id", "score"]
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames}
            for row in rows
            if row["status"] == "ok"
        )
    counts = {
        status: sum(entry["status"] == status for entry in entries)
        for status in ("media", "error", "excluded", "duplicate")
    }
    by_modality = {}
    for modality in ("audio", "video"):
        result_subset = [row for row in rows if row["modality"] == modality]
        successful = sum(row["status"] == "ok" for row in result_subset)
        by_modality[modality] = {
            "registered": len(result_subset),
            "success": successful,
            "failed": sum(row["status"] == "error" for row in result_subset),
            "not_run": sum(row["status"] == "not_run" for row in result_subset),
            "coverage": successful / len(result_subset) if result_subset else None,
        }
    failures = counts["error"] + sum(row["status"] != "ok" for row in rows)
    if failures == 0 and (not records or global_error is not None):
        failures = 1
        if global_error is None:
            global_error = "no_media: no scorable audio/video input found"
    report: dict[str, object] = {
        "output": str(output),
        "device": device,
        "discovered": len(entries),
        "discovery_counts": counts,
        "by_modality": by_modality,
        "samples": len(records),
        "failures": failures,
        "global_error": global_error,
        "score_type": registry["score_type"],
        "score_direction": registry["score_direction"],
        "official": False,
    }
    atomic_json(output / "report.json", report)
    return report
