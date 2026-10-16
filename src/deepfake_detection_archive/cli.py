"""Explicit command-line entry points for reproducible competition workflows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .config import InferenceConfig, load_inference_config, load_object, reject_unknown
from .contracts import ScoreRecord, load_scores
from .data.audit import audit_manifest
from .data.manifest import (
    SampleRecord,
    load_manifest,
    load_sources,
    manifest_hash,
    write_manifest,
)
from .data.split import assign_splits, connected_groups, statistical_groups
from .evaluation.report import evaluate_records, grouped_reports
from .preparation import decode_manifest, index_media, media_snapshot
from .tracking import atomic_json, canonical_json, code_identity, object_hash, utc_now


def _paths(config: dict[str, object], base: Path, name: str) -> Path:
    value = config.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"configuration requires path {name}")
    return base / value


def _audit(args: argparse.Namespace) -> dict[str, object]:
    config = load_object(args.config)
    reject_unknown(
        config, {"manifest", "data_root", "sources", "output", "decoded_manifest"}
    )
    base = args.config.parent
    root = _paths(config, base, "data_root")
    records = load_manifest(_paths(config, base, "manifest"))
    if args.decode:
        records = decode_manifest(records, root)
        write_manifest(records, _paths(config, base, "decoded_manifest"))
    report = audit_manifest(
        records, root, load_sources(_paths(config, base, "sources"))
    )
    atomic_json(_paths(config, base, "output"), report.to_dict())
    return {
        "passed": report.passed,
        "sample_count": len(records),
        "trainable_count": len(report.trainable_ids),
        "issue_count": len(report.issues),
        "report": str(_paths(config, base, "output")),
    }


def _index(args: argparse.Namespace) -> dict[str, object]:
    records = index_media(args.root, args.modality, args.source)
    digest = write_manifest(records, args.output)
    return {
        "samples": len(records),
        "manifest_sha256": digest,
        "output": str(args.output),
    }


def _split(args: argparse.Namespace) -> dict[str, object]:
    records = assign_splits(load_manifest(args.manifest), seed=args.seed)
    groups = statistical_groups(records)
    digest = write_manifest(records, args.output)
    return {
        "samples": len(records),
        "leakage_components": len(set(connected_groups(records).values())),
        "known_statistical_groups": len(set(groups.values()) - {"unknown"}),
        "sha256": digest,
    }


def _verified_scores(
    scores: list[ScoreRecord], records: list[SampleRecord]
) -> list[ScoreRecord]:
    by_id = {record.sample_id: record for record in records}
    if set(by_id) != {score.sample_id for score in scores}:
        raise ValueError("score records do not match manifest sample IDs")
    groups = statistical_groups(records)
    verified = []
    fields = ("label", "modality", "source", "content_type", "generator")
    for score in scores:
        record = by_id[score.sample_id]
        for field in fields:
            if getattr(score, field) != getattr(record, field):
                raise ValueError(f"score/manifest mismatch: {score.sample_id}.{field}")
        verified.append(replace(score, group_id=groups[score.sample_id]))
    return verified


def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    scores = load_scores(args.scores)
    expected = None
    if args.manifest is not None:
        records = load_manifest(args.manifest)
        scores = _verified_scores(scores, records)
        expected = [record.sample_id for record in records]
    else:
        scores = [replace(score, group_id="unknown") for score in scores]
    report = {
        **asdict(
            evaluate_records(
                scores, frozen_threshold=args.threshold, expected_sample_ids=expected
            )
        ),
        "by_modality": {
            key: asdict(value)
            for key, value in grouped_reports(
                scores, "modality", frozen_threshold=args.threshold
            ).items()
        },
        "by_source": {
            key: asdict(value)
            for key, value in grouped_reports(
                scores, "source", frozen_threshold=args.threshold
            ).items()
        },
        "by_content_type": {
            key: asdict(value)
            for key, value in grouped_reports(
                scores, "content_type", frozen_threshold=args.threshold
            ).items()
        },
        "by_generator": {
            key: asdict(value)
            for key, value in grouped_reports(
                scores, "generator", frozen_threshold=args.threshold
            ).items()
        },
        "expected_id_set_verified": expected is not None,
        "manifest_metadata_verified": expected is not None,
        "statistical_groups_from_manifest": expected is not None,
        "official_evaluation": False,
        "det_endpoint_note": "null threshold is strictly above the maximum score",
    }
    atomic_json(args.output, report)
    return {
        "fullset_eer": report["fullset_eer"],
        "failures": report["failure_count"],
        "output": str(args.output),
    }


def _infer(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    config = load_inference_config(args.config)
    config = replace(
        config,
        execution_mode=args.execution_mode or config.execution_mode,
        batch_size=(
            args.batch_size if args.batch_size is not None else config.batch_size
        ),
    )
    base = args.config.parent
    records = load_manifest(base / config.manifest)
    return _infer_records(
        config,
        base,
        records,
        snapshot=media_snapshot(records, base / config.data_root),
        preflight_started=started,
        resume=args.resume,
        frozen_audit=args.frozen_audit,
    )


def _infer_records(  # noqa: PLR0913 -- parsed inputs and explicit run controls
    config: InferenceConfig,
    base: Path,
    records: list[SampleRecord],
    *,
    snapshot: list[dict[str, object]],
    preflight_started: float,
    resume: bool,
    frozen_audit: bool,
) -> dict[str, object]:
    import torch  # noqa: PLC0415 -- optional inference backend

    from .scoring import OfflineScorer  # noqa: PLC0415
    from .submission.runner import run_inference  # noqa: PLC0415

    if any(record.modality != config.modality for record in records):
        raise ValueError("input manifest has a different modality")
    if any(record.split == "holdout" for record in records) and not frozen_audit:
        raise ValueError(
            "holdout inference requires --frozen-audit after candidate freeze"
        )
    groups = statistical_groups(records)
    records = [replace(record, group_id=groups[record.sample_id]) for record in records]
    preflight = time.perf_counter() - preflight_started
    started = time.perf_counter()
    from .execution import execution_signature  # noqa: PLC0415

    scorer = OfflineScorer(config, base, resume=resume)
    startup = time.perf_counter() - started
    output = base / config.output_dir
    identity = {
        "config": config.to_dict(),
        "manifest_sha256": manifest_hash(records),
        "media_snapshot": snapshot,
        "media_snapshot_sha256": object_hash(snapshot),
        "code": code_identity(),
        "official_adapter": "unconfirmed",
        "frozen_audit": frozen_audit,
        "execution": execution_signature(config.device, config.execution_mode),
    }
    try:
        scores = run_inference(
            records,
            scorer,
            output,
            identity=identity,
            resume=resume,
            fatal_exceptions=(torch.OutOfMemoryError,),
        )
    finally:
        scorer.execution.finish()
    summary = load_object(output / "report.json")
    timing = summary["timing"]
    if not isinstance(timing, dict):
        raise TypeError("invalid runner timing report")
    previous_startup = timing.get("startup_seconds")
    if previous_startup is None:
        timing["startup_seconds"] = startup
        timing["preflight_seconds"] = preflight
        timing["preflight_seconds_per_sample"] = preflight / len(records)
        timing["T10k_raw_seconds"] = (
            float(timing["T10k_raw_seconds"])
            + startup
            + preflight / len(records) * 10000
        )
        timing["T10k_safe_seconds"] = 1.25 * timing["T10k_raw_seconds"]
        timing["projection_scope"] = (
            "model startup, media/config preflight per sample, sample calls, and row writes"
        )
    summary["finished_at"] = utc_now()
    atomic_json(output / "report.json", summary)
    return {
        "samples": len(scores),
        "failures": summary["failure_count"],
        "fullset_eer": summary["fullset_eer"],
        "output": str(output),
        "elapsed_seconds": time.perf_counter() - preflight_started,
    }


def _predict(args: argparse.Namespace) -> dict[str, object]:
    from .submission.adapter import export_scores  # noqa: PLC0415

    if args.paths or args.show_models:
        if args.config is not None or args.input is not None:
            raise ValueError(
                "positional media cannot be combined with --config/--input"
            )
        from .unified_predict import (  # noqa: PLC0415
            _model_package_root,
            _package_resource,
            default_models,
            predict_media,
        )

        if args.show_models:
            if args.paths:
                raise ValueError("--show-models does not take media inputs")
            folder, catalog = default_models()
            return {
                "models": catalog,
                "inference_configs": {
                    modality: load_inference_config(
                        _package_resource(
                            _model_package_root(), folder, entry["config"]
                        )
                    ).to_dict()
                    for modality, entry in catalog["models"].items()
                },
                "failures": 0,
            }
        return predict_media(
            args.paths,
            args.output,
            args.device or "auto",
            resume=args.resume,
            execution_mode=args.execution_mode,
            batch_size=args.batch_size,
        )
    if args.config is None or args.input is None or args.output is None:
        raise ValueError("provide media paths, or --config --input --output")
    if args.device == "auto":
        raise ValueError("--device auto is supported by positional media mode")
    started = time.perf_counter()
    template = load_inference_config(args.config)
    base = args.config.resolve().parent
    root, output = args.input.resolve(), args.output.resolve()
    records = index_media(root, template.modality, "unlabeled-input")
    config = replace(
        template,
        manifest=str(output / "manifest.jsonl"),
        data_root=str(root),
        output_dir=str(output),
        checkpoint=str((base / template.checkpoint).resolve()),
        device=args.device or template.device,
        execution_mode=args.execution_mode or template.execution_mode,
        batch_size=(
            args.batch_size if args.batch_size is not None else template.batch_size
        ),
    )
    # Indexing already hashed every input; reuse those bytes in the resume identity.
    snapshot: list[dict[str, object]] = [
        {
            "sample_id": row.sample_id,
            "path": row.path,
            "sha256": row.sha256,
            "state": "present",
        }
        for row in records
    ]
    csv_path = output / "scores.csv"
    result = _infer_records(
        config,
        base,
        records,
        snapshot=snapshot,
        preflight_started=started,
        resume=args.resume,
        frozen_audit=False,
    )
    write_manifest(records, output / "manifest.jsonl")
    atomic_json(output / "inference.json", config.to_dict())
    export_scores(load_scores(output / "scores.jsonl"), csv_path, allow_partial=True)
    return {
        **result,
        "official": False,
        "scores_csv": str(csv_path),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _train(args: argparse.Namespace) -> dict[str, object]:
    if args.check_config:
        from .preflight import check_training_config  # noqa: PLC0415

        return check_training_config(args.config)
    from .experiments import run_experiment  # noqa: PLC0415

    result = run_experiment(args.config)
    return {
        **asdict(result),
        "best_checkpoint": str(result.best_checkpoint),
        "history_path": str(result.history_path),
        "run_path": str(result.run_path),
    }


def _duplicates(args: argparse.Namespace) -> dict[str, object]:
    from .data.duplicates import find_near_duplicates  # noqa: PLC0415

    report = find_near_duplicates(
        load_manifest(args.manifest), args.root, min_similarity=args.min_similarity
    )
    atomic_json(args.output, report.to_dict())
    return {
        "output": str(args.output),
        "candidate_pairs": len(report.candidate_pairs),
        "automatic_merge": False,
    }


def _export(args: argparse.Namespace) -> dict[str, object]:
    from .submission.adapter import AdapterConfig, export_scores  # noqa: PLC0415

    config = AdapterConfig(**load_object(args.config))  # type: ignore[arg-type]
    path = export_scores(
        load_scores(args.scores), args.output, config, official=args.official
    )
    return {"output": str(path), "official": args.official}


def _package(args: argparse.Namespace) -> dict[str, Any]:
    from .submission.package import package_bundle  # noqa: PLC0415

    config = load_object(args.config)
    reject_unknown(config, {"files", "metadata", "output"})
    files, metadata = config.get("files"), config.get("metadata")
    if not isinstance(files, dict) or not isinstance(metadata, dict):
        raise TypeError("package config requires files and metadata objects")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in files.items()
    ):
        raise ValueError("package files must map bundle paths to local paths")
    output = _paths(config, args.config.parent, "output")
    result = package_bundle(
        output,
        {key: args.config.parent / value for key, value in files.items()},
        metadata=metadata,
    )
    return {"output": str(output), "sha256": result["sha256"]}


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    from .submission.package import verify_bundle  # noqa: PLC0415

    result = verify_bundle(
        args.bundle, expected_sha256=args.sha256, extract_to=args.extract_to
    )
    return {key: value for key, value in result.items() if key != "manifest"}


def parser() -> argparse.ArgumentParser:  # noqa: PLR0915 -- explicit command options
    result = argparse.ArgumentParser(description="Offline AIGC audio/video detection")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("audit", "train", "infer", "package"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "audit":
            command.add_argument("--decode", action="store_true")
        if name == "train":
            command.add_argument(
                "--check-config",
                action="store_true",
                help="read-only schema, train/dev data and local asset checks; no training",
            )
        if name == "infer":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--frozen-audit", action="store_true")
            command.add_argument("--execution-mode", choices=("reference", "adaptive"))
            command.add_argument("--batch-size", type=int)
    command = commands.add_parser("predict")
    command.add_argument("paths", nargs="*", type=Path)
    command.add_argument("--show-models", action="store_true")
    command.add_argument("--config", type=Path)
    command.add_argument("--input", type=Path)
    command.add_argument("--output", type=Path)
    command.add_argument("--device", choices=("auto", "cpu", "cuda"))
    command.add_argument("--resume", action="store_true")
    command.add_argument("--execution-mode", choices=("reference", "adaptive"))
    command.add_argument("--batch-size", type=int)
    command = commands.add_parser("index")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--modality", choices=("audio", "video"), required=True)
    command.add_argument("--source", default="unknown")
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("split")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--seed", type=int, default=42)
    command = commands.add_parser("evaluate")
    command.add_argument("--scores", type=Path, required=True)
    command.add_argument("--manifest", type=Path)
    command.add_argument("--threshold", type=float)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("duplicates")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--min-similarity", type=float, default=0.97)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("export")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--scores", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--official", action="store_true")
    command = commands.add_parser("verify-package")
    command.add_argument("--bundle", type=Path, required=True)
    command.add_argument("--sha256")
    command.add_argument("--extract-to", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    actions = {
        "index": _index,
        "audit": _audit,
        "split": _split,
        "evaluate": _evaluate,
        "infer": _infer,
        "predict": _predict,
        "train": _train,
        "duplicates": _duplicates,
        "export": _export,
        "package": _package,
        "verify-package": _verify,
    }
    try:
        result = actions[args.command](args)
        print(canonical_json(result))
        if (
            args.command == "audit" or (args.command == "train" and args.check_config)
        ) and not result["passed"]:
            return 2
        if args.command in {"infer", "predict"} and result["failures"]:
            return 2
    except (
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        ImportError,
        MemoryError,
    ) as exc:
        print(
            json.dumps(
                {"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 2
    return 0
