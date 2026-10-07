"""Pre-registered ViF Panda local-research pilot; no detector training/scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from deepfake_detection_archive.data.duplicates import image_phash
from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.data.split import (
    connected_groups,
    statistical_groups,
    validate_splits,
)
from deepfake_detection_archive.features import media_path
from deepfake_detection_archive.media import MediaError, VideoConfig, decode_video
from deepfake_detection_archive.models.encoders import FrozenVisionEncoder
from deepfake_detection_archive.tracking import atomic_json, code_identity, object_hash


ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
WORK = ROOT / "outputs/experiment-planning/vif-pilot"
DOC = ROOT / "docs/research/source/vif-pilot"
DATA = ROOT / "data/vif-panda-research"
CACHE = ROOT / "cache/vif-pilot"
WEIGHTS = ROOT / "models/pretrained/resnet18-a1/encoder.pt"
WEIGHTS_SHA = "db6fd0c5cfb2b712d1904bb8b7edcf41a850491bdf42729e4ab485e89f1b6889"
VIDEO = VideoConfig(n_frames=4, height=224, width=224, resize_mode="pad")
SPLITS = ("train", "dev", "holdout")
RATIOS = {"train": 0.7, "dev": 0.15, "holdout": 0.15}


def now() -> str:
    return datetime.now(UTC).isoformat()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def emit(name: str, payload: Any) -> None:
    atomic_json(WORK / name, payload)
    shutil.copyfile(WORK / name, DOC / name)


def rank(value: str) -> str:
    return hashlib.sha256(f"42:{value}".encode()).hexdigest()


def budget(reserve_bytes: int = 0) -> None:
    cache = sum(p.stat().st_size for p in (ROOT / "cache").rglob("*") if p.is_file())
    total = sum(
        p.stat().st_size
        for name in ("cache", "data", "models", "outputs")
        for p in (ROOT / name).rglob("*")
        if p.is_file()
    )
    if cache + reserve_bytes > 20 * 1024**3 or total + reserve_bytes > 100 * 1024**3:
        raise ValueError(f"resource budget exceeded: cache={cache}, total={total}")


def freeze() -> None:
    """Freeze selection and both training configurations before detector scores."""
    if (WORK / "preregistration.json").exists():
        raise ValueError("preregistration already exists; refusing to rewrite")
    for directory in (WORK, DOC, CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    configs: dict[str, str] = {}
    for seed in (42, 43):
        config = {
            "schema_version": 1,
            "manifest": str(WORK / "manifest.jsonl"),
            "data_root": str(DATA / "media"),
            "sources": str(DATA / "sources.jsonl"),
            "output_dir": str(ROOT / f"outputs/vif-pilot-resnet18-seed{seed}"),
            "modality": "video",
            "architecture": "media_feature_head_v1",
            "model_config": {
                "input_dim": 512,
                "hidden_dim": 64,
                "dropout": 0.1,
                "aggregation": "mean",
                "trim_fraction": 0.1,
            },
            "training": {
                "device": "cuda",
                "amp": True,
                "amp_dtype": "bfloat16",
                "max_epochs": 20,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "accumulation_steps": 1,
                "patience": 5,
                "min_delta": 0.0,
                "max_grad_norm": 1.0,
                "seed": seed,
                "deterministic": True,
            },
            "batch_size": 64,
            "num_workers": 0,
            "segments": 1,
            "video": asdict(VIDEO),
            "encoder": {
                "kind": "vision",
                "architecture": "resnet18",
                "checkpoint": str(WEIGHTS),
                "expected_sha256": WEIGHTS_SHA,
            },
            "feature_cache": str(CACHE / "training-features"),
            "cache_max_bytes": 1024**3,
        }
        name = f"train-seed{seed}.json"
        emit(name, config)
        configs[name] = file_sha256(WORK / name)
    emit(
        "preregistration.json",
        {
            "created_at": now(),
            "purpose": "local_research_only; exploratory Panda subset, no formal P1/P2",
            "target_media": 1024,
            "selection_seed": 42,
            "selection": "full known-relation components; proportional fake-generator "
            "quotas; repeatedly choose largest relative generator deficit, then "
            "smallest SHA256(42:component) fitting remaining media budget",
            "split_ratios": RATIOS,
            "split": "component greedy class+paired-generator stratification; seed42; "
            "cover each stratum in all splits first, then quadratic balance cost",
            "near_duplicate_policy": {
                "phash": "all 4 aligned frames valid, minimum bit agreement >=0.97, "
                "duration max/min <=1.03",
                "embedding": "cosine of L2-normalized mean ResNet18 frame features "
                ">=0.995, independent of duration",
                "handling": "union every threshold candidate as conservative leakage "
                "edge; does not establish duplicate truth, identity or source group",
                "nearest_neighbors": "top3 cosine neighbors recorded for every media; "
                "below both candidate thresholds retained as inspection diagnostics",
            },
            "failure_policy": "record every failure; abort split lock on any decode or "
            "feature failure; low-information pHash remains explicitly unaudited",
            "unknown_fields": ["identity_id", "group_id", "original_id"],
            "independent_group_confidence_intervals": False,
            "holdout": "media governance audit only before lock; never training, "
            "early stopping, threshold selection or hyperparameter tuning",
            "training_config_sha256": configs,
            "weights_sha256": WEIGHTS_SHA,
            "video": asdict(VIDEO),
            "code_identity_at_registration": code_identity(),
        },
    )


def strata(records: list[SampleRecord]) -> dict[str, str]:
    paired = {r.parent_id: r.generator for r in records if r.label == 1}
    return {
        r.sample_id: f"{r.label}:{r.generator if r.label == 1 else paired[r.sample_id]}"
        for r in records
    }


def select() -> None:
    budget()
    if not (WORK / "preregistration.json").exists():
        raise ValueError("freeze must run first")
    if (WORK / "selected.jsonl").exists():
        raise ValueError("selection already exists")
    records = load_manifest(DATA / "manifest.jsonl")
    summary = read(DATA / "import-summary.json")
    if summary["manifest_sha256"] != file_sha256(DATA / "manifest.jsonl"):
        raise ValueError("import manifest checksum mismatch")
    groups = connected_groups(records)
    items: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        items[groups[record.sample_id]].append(record)
    totals = Counter(r.generator for r in records if r.label == 1)
    targets = {g: 512 * n / sum(totals.values()) for g, n in totals.items()}
    counts = {
        g: Counter(r.generator for r in rows if r.label == 1)
        for g, rows in items.items()
    }
    remaining = set(items)
    chosen: list[SampleRecord] = []
    current: Counter[str] = Counter()
    while True:
        feasible = {g for g in remaining if len(items[g]) <= 1024 - len(chosen)}
        if not feasible:
            break
        generators = sorted(
            totals, key=lambda g: (-(targets[g] - current[g]) / targets[g], g)
        )
        selected = None
        for generator in generators:
            candidates = [g for g in feasible if counts[g][generator]]
            if candidates:
                selected = min(candidates, key=rank)
                break
        if selected is None:
            raise ValueError("no generator-backed component remains")
        chosen.extend(items[selected])
        current.update(counts[selected])
        remaining.remove(selected)
    chosen.sort(key=lambda r: r.sample_id)
    checksum = write_manifest(chosen, WORK / "selected.jsonl")
    emit(
        "selection.json",
        {
            "created_at": now(),
            "source_manifest_sha256": summary["manifest_sha256"],
            "source_summary_sha256": file_sha256(DATA / "import-summary.json"),
            "source_grouping_sha256": file_sha256(DATA / "grouping-evidence.json"),
            "selected_manifest_sha256": checksum,
            "media_count": len(chosen),
            "component_count": len({groups[r.sample_id] for r in chosen}),
            "generator_fake_counts": dict(current),
            "proportional_targets": targets,
            "all_selected_source_components_complete": all(
                not ({r.sample_id for r in rows} & {r.sample_id for r in chosen})
                or {r.sample_id for r in rows} <= {r.sample_id for r in chosen}
                for rows in items.values()
            ),
        },
    )


def check_selected() -> None:
    selection = read(WORK / "selection.json")
    if file_sha256(WORK / "selected.jsonl") != selection["selected_manifest_sha256"]:
        raise ValueError("selected manifest changed after selection")
    if file_sha256(DATA / "manifest.jsonl") != selection["source_manifest_sha256"]:
        raise ValueError("source manifest changed after selection")


def frame_hash(frame: NDArray[np.uint8], width: int, height: int) -> list[int]:
    """Reject flat original content before hashing any synthetic padding."""
    ratio = min(VIDEO.width / width, VIDEO.height / height)
    content_width, content_height = (
        max(1, round(width * ratio)),
        max(1, round(height * ratio)),
    )
    left, top = (VIDEO.width - content_width) // 2, (VIDEO.height - content_height) // 2
    margin_x, margin_y = content_width // 10, content_height // 10
    interior = frame[
        top + margin_y : top + content_height - margin_y,
        left + margin_x : left + content_width - margin_x,
    ]
    image_phash(interior)
    return list(image_phash(frame))


def decode_one(record: SampleRecord) -> dict[str, Any]:
    start = time.monotonic()
    result: dict[str, Any] = {"sample_id": record.sample_id, "label": record.label}
    try:
        path = media_path(record, DATA / "media")
        probe = subprocess.run(  # noqa: S603
            [
                shutil.which("ffprobe") or "/usr/bin/ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        metadata = json.loads(probe.stdout)
        stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        video = decode_video(path, VIDEO)
        output = CACHE / "frames" / f"{object_hash(record.sample_id)}.npy"
        np.save(output, video.frames, allow_pickle=False)
        hashes: list[list[int] | None] = []
        phash_failures: list[dict[str, Any]] = []
        for index, frame in enumerate(video.frames):
            try:
                hashes.append(
                    frame_hash(frame, video.original_width, video.original_height)
                )
            except MediaError as exc:
                hashes.append(None)
                phash_failures.append(
                    {"frame": index, "code": exc.code, "message": str(exc)}
                )
        result.update(
            {
                "status": "ok",
                "media_sha256": record.sha256,
                "frames": str(output.relative_to(ROOT)),
                "frames_sha256": file_sha256(output),
                "phash": hashes,
                "phash_failures": phash_failures,
                "duration_seconds": video.original_duration_seconds,
                "timestamps_seconds": list(video.timestamps_seconds),
                "frame_indices": list(video.frame_indices),
                "width": video.original_width,
                "height": video.original_height,
                "codec": stream.get("codec_name", "unknown"),
                "container": metadata["format"].get("format_name", "unknown"),
                "frame_rate": stream.get("avg_frame_rate", "unknown"),
                "pixel_format": stream.get("pix_fmt", "unknown"),
                "bit_rate": stream.get("bit_rate", "unknown"),
            }
        )
    except Exception as exc:
        result.update(
            {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}
        )
    result["elapsed_seconds"] = time.monotonic() - start
    return result


def decode() -> None:
    check_selected()
    records = load_manifest(WORK / "selected.jsonl")
    budget(len(records) * (4 * 224 * 224 * 3 + 1024))
    (CACHE / "frames").mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(decode_one, records), 1):
            results.append(result)
            if index % 64 == 0:
                print(f"decoded {index}/{len(records)}", flush=True)
    emit(
        "decode-audit.json",
        {
            "created_at": now(),
            "elapsed_seconds": time.monotonic() - start,
            "selected_manifest_sha256": file_sha256(WORK / "selected.jsonl"),
            "video": asdict(VIDEO),
            "workers": 4,
            "samples": results,
            "failure_count": sum(r["status"] != "ok" for r in results),
            "code_identity": code_identity(),
        },
    )
    budget()


def features() -> None:
    check_selected()
    budget(16 * 1024**2)
    audit = read(WORK / "decode-audit.json")
    if audit["selected_manifest_sha256"] != file_sha256(WORK / "selected.jsonl"):
        raise ValueError("stale decode audit")
    if audit["failure_count"]:
        raise ValueError("decode failures prohibit feature audit")
    atomic_json(
        WORK / "feature-progress.json",
        {
            "completed_sample_ids": [],
            "active_sample_ids": [],
            "unattempted_sample_ids": [s["sample_id"] for s in audit["samples"]],
        },
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for feature audit")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = FrozenVisionEncoder(
        "resnet18", WEIGHTS, expected_sha256=WEIGHTS_SHA
    ).cuda()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    result = []
    with torch.inference_mode():
        for offset in range(0, len(audit["samples"]), 16):
            batch = audit["samples"][offset : offset + 16]
            atomic_json(
                WORK / "feature-progress.json",
                {
                    "completed_sample_ids": [
                        s["sample_id"] for s in audit["samples"][:offset]
                    ],
                    "active_sample_ids": [s["sample_id"] for s in batch],
                    "unattempted_sample_ids": [
                        s["sample_id"] for s in audit["samples"][offset + len(batch) :]
                    ],
                },
            )
            arrays = []
            for sample in batch:
                path = ROOT / sample["frames"]
                if file_sha256(path) != sample["frames_sha256"]:
                    raise ValueError("frame cache checksum mismatch")
                arrays.append(np.load(path, allow_pickle=False))
            inputs = (
                torch.from_numpy(np.concatenate(arrays))
                .permute(0, 3, 1, 2)
                .float()
                .div_(255)
            )
            mean = inputs.new_tensor(encoder.preprocessing["mean"])[None, :, None, None]
            std = inputs.new_tensor(encoder.preprocessing["std"])[None, :, None, None]
            output = encoder(((inputs - mean) / std).cuda()).float().cpu()
            if (
                output.shape != (4 * len(batch), 512)
                or not torch.isfinite(output).all()
            ):
                raise ValueError("invalid encoder features")
            result.append(output.reshape(len(batch), 4, 512).numpy())
            if offset % 128 == 0:
                print(
                    f"features {offset + len(batch)}/{len(audit['samples'])}",
                    flush=True,
                )
    atomic_json(
        WORK / "feature-progress.json",
        {
            "completed_sample_ids": [s["sample_id"] for s in audit["samples"]],
            "active_sample_ids": [],
            "unattempted_sample_ids": [],
        },
    )
    torch.cuda.synchronize()
    path = CACHE / "audit-features.npy"
    np.save(path, np.concatenate(result), allow_pickle=False)
    emit(
        "feature-audit.json",
        {
            "created_at": now(),
            "elapsed_seconds": time.monotonic() - start,
            "weights_sha256": WEIGHTS_SHA,
            "feature_sha256": file_sha256(path),
            "path": str(path.relative_to(ROOT)),
            "dtype": "float32",
            "amp": False,
            "shape": [len(audit["samples"]), 4, 512],
            "gpu": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "decode_audit_sha256": file_sha256(WORK / "decode-audit.json"),
            "code_identity": code_identity(),
            "optimizer_steps": 0,
            "purpose": "pre-split leakage audit only, no detector predictions",
        },
    )
    del encoder
    torch.cuda.empty_cache()


def partition(
    records: list[SampleRecord], edges: tuple[tuple[str, str], ...]
) -> list[SampleRecord]:
    groups = connected_groups(records, edges)
    labels = strata(records)
    items: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        items[groups[record.sample_id]].append(record)
    counts = {
        g: Counter(labels[r.sample_id] for r in rows) for g, rows in items.items()
    }
    totals = Counter(labels.values())
    running: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    assigned: dict[str, str] = {}
    remaining = set(items)
    for split in ("holdout", "dev", "train"):
        while missing := {k for k in totals if not running[split][k]}:
            key = min(
                missing, key=lambda k: (sum(counts[g][k] > 0 for g in remaining), k)
            )
            possible = [g for g in remaining if counts[g][key]]
            if not possible:
                raise ValueError(f"insufficient components for {split}/{key}")
            group = min(
                possible,
                key=lambda g: (
                    -sum(counts[g][k] > 0 for k in missing),
                    len(items[g]),
                    rank(g),
                ),
            )
            assigned[group] = split
            running[split].update(counts[group])
            remaining.remove(group)
    for group in sorted(remaining, key=lambda g: (-len(items[g]), rank(g))):

        def cost(split: str, current_group: str = group) -> float:
            return sum(
                (
                    (running[split][k] + counts[current_group][k]) ** 2
                    - running[split][k] ** 2
                )
                / max(RATIOS[split] * n, 1)
                for k, n in totals.items()
            )

        split = min(SPLITS, key=cost)
        assigned[group] = split
        running[split].update(counts[group])
    return [
        replace(
            r,
            split=assigned[groups[r.sample_id]],
            derivation_group_id=f"vif_pilot_conservative_component:{object_hash(groups[r.sample_id])}",
        )
        for r in records
    ]


def audited_inputs() -> tuple[list[SampleRecord], dict[str, Any], dict[str, Any]]:
    check_selected()
    registration = read(WORK / "preregistration.json")
    for name, expected in registration["training_config_sha256"].items():
        if file_sha256(WORK / name) != expected:
            raise ValueError("training configuration changed after registration")
    records = load_manifest(WORK / "selected.jsonl")
    if not 1 <= len(records) <= 1024:
        raise ValueError("pilot media/pair budget exceeded")
    decoded = read(WORK / "decode-audit.json")
    feature_audit = read(WORK / "feature-audit.json")
    if decoded["selected_manifest_sha256"] != file_sha256(WORK / "selected.jsonl"):
        raise ValueError("stale decode audit")
    if decoded["failure_count"] or feature_audit["decode_audit_sha256"] != file_sha256(
        WORK / "decode-audit.json"
    ):
        raise ValueError("failed or stale audit")
    path = ROOT / feature_audit["path"]
    if file_sha256(path) != feature_audit["feature_sha256"]:
        raise ValueError("feature checksum mismatch")
    samples = decoded["samples"]
    if [r.sample_id for r in records] != [r["sample_id"] for r in samples]:
        raise ValueError("audit ordering mismatch")
    return records, decoded, feature_audit


def lock() -> None:
    if (WORK / "split-lock.json").exists():
        raise ValueError("split already locked")
    records, decoded, feature_audit = audited_inputs()
    samples = decoded["samples"]
    path = ROOT / feature_audit["path"]
    embeddings = np.load(path, allow_pickle=False).mean(axis=1)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    cosine = embeddings @ embeddings.T
    candidates = []
    nearest = []
    edges: list[tuple[str, str]] = []
    known = connected_groups(records)
    for i, left in enumerate(samples):
        neighbors = sorted(
            (j for j in range(len(samples)) if j != i),
            key=lambda j: (-float(cosine[i, j]), samples[j]["sample_id"]),
        )[:3]
        nearest.append(
            {
                "sample_id": left["sample_id"],
                "neighbors": [
                    {
                        "sample_id": samples[j]["sample_id"],
                        "cosine": float(cosine[i, j]),
                    }
                    for j in neighbors
                ],
            }
        )
        for j in range(i + 1, len(samples)):
            right = samples[j]
            agreement = None
            if None not in left["phash"] and None not in right["phash"]:
                agreement = min(
                    sum(a == b for a, b in zip(x, y, strict=True)) / 63
                    for x, y in zip(left["phash"], right["phash"], strict=True)
                )
            duration_ratio = max(
                left["duration_seconds"], right["duration_seconds"]
            ) / min(left["duration_seconds"], right["duration_seconds"])
            phash_hit = (
                agreement is not None and agreement >= 0.97 and duration_ratio <= 1.03
            )
            embedding_hit = float(cosine[i, j]) >= 0.995
            if phash_hit or embedding_hit:
                pair = (left["sample_id"], right["sample_id"])
                edges.append(pair)
                candidates.append(
                    {
                        "left_id": pair[0],
                        "right_id": pair[1],
                        "minimum_phash_agreement": agreement,
                        "cosine": float(cosine[i, j]),
                        "duration_ratio": duration_ratio,
                        "phash_candidate": phash_hit,
                        "embedding_candidate": embedding_hit,
                        "known_component": known[pair[0]] == known[pair[1]],
                        "label_conflict": records[i].label != records[j].label,
                        "action": "merge_as_conservative_leakage_edge",
                        "reason": "predeclared high similarity; false merge sacrifices independence/count, not leakage; no identity or duplicate truth claim",
                    }
                )
    emit(
        "duplicate-audit.json",
        {
            "created_at": now(),
            "candidate_count": len(candidates),
            "new_component_edges": sum(not p["known_component"] for p in candidates),
            "candidates": candidates,
            "nearest_neighbors": nearest,
            "phash_incomplete_sample_count": sum(
                bool(s["phash_failures"]) for s in samples
            ),
            "statistical_identity_known": False,
        },
    )
    merged = connected_groups(records, tuple(edges))
    component_sizes = Counter(merged.values())
    emit(
        "component-audit.json",
        {
            "component_count": len(component_sizes),
            "max_size": max(component_sizes.values()),
            "max_fraction": max(component_sizes.values()) / len(records),
            "size_histogram": dict(Counter(component_sizes.values())),
            "components": dict(component_sizes),
            "pair_comparisons": len(records) * (len(records) - 1) // 2,
        },
    )
    final = partition(records, tuple(edges))
    updated = []
    for record, sample in zip(final, samples, strict=True):
        updated.append(
            replace(
                record,
                decode_status="ok",
                duration_seconds=sample["duration_seconds"],
                codec=sample["codec"],
                container=sample["container"],
            )
        )
    final = updated
    errors = validate_splits(final, tuple(edges))
    if errors or any(g != "unknown" for g in statistical_groups(final).values()):
        raise ValueError(f"split/unknown-group invariant failed: {errors}")
    split_sha = write_manifest(final, WORK / "manifest.jsonl")
    shutil.copyfile(WORK / "manifest.jsonl", DOC / "manifest.jsonl")
    emit(
        "relationship-preservation.json",
        {
            r.sample_id: {
                "original_derivation_group_id": r.derivation_group_id,
                "parent_id": r.parent_id,
                "sha256": r.sha256,
                "pilot_derivation_group_id": f.derivation_group_id,
            }
            for r, f in zip(records, final, strict=True)
        },
    )
    tables: dict[str, Any] = {}
    for field in (
        "codec",
        "container",
        "width",
        "height",
        "frame_rate",
        "pixel_format",
    ):
        table: dict[str, Counter[str]] = defaultdict(Counter)
        for record, sample in zip(final, samples, strict=True):
            table[str(sample[field])][f"{record.split}:label{record.label}"] += 1
        tables[field] = {k: dict(v) for k, v in sorted(table.items())}
    emit(
        "shortcut-diagnostics.json",
        {
            "purpose": "pre-score metadata association; not detector performance",
            "tables": tables,
            "limitation": "same-source pairs can still differ systematically in resolution, codec, duration; this pilot does not prove content-independent forgery detection",
            "duration_by_label": {
                str(label): {
                    "min": float(np.min(values)),
                    "median": float(np.median(values)),
                    "max": float(np.max(values)),
                }
                for label in (0, 1)
                if (
                    values := [
                        s["duration_seconds"]
                        for r, s in zip(records, samples, strict=True)
                        if r.label == label
                    ]
                )
            },
        },
    )
    emit(
        "split-lock.json",
        {
            "created_at": now(),
            "manifest_sha256": split_sha,
            "preregistration_sha256": file_sha256(WORK / "preregistration.json"),
            "split_counts": dict(Counter(r.split for r in final)),
            "split_class_generator_counts": dict(
                Counter(f"{r.split}:{strata(records)[r.sample_id]}" for r in final)
            ),
            "conservative_components": len(set(connected_groups(final).values())),
            "all_statistical_groups_unknown": True,
            "split_validation_errors": errors,
            "script_sha256": file_sha256(Path(__file__)),
            "code_identity": code_identity(),
            "training_started": False,
            "formal_P1_P2_pass": False,
            "purpose": "local_research_only",
        },
    )
    budget()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("freeze", "select", "decode", "features", "lock")
    )
    args = parser.parse_args()
    try:
        {
            "freeze": freeze,
            "select": select,
            "decode": decode,
            "features": features,
            "lock": lock,
        }[args.phase]()
    except Exception as exc:
        emit(
            f"{args.phase}-failure.json",
            {
                "created_at": now(),
                "phase": args.phase,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "phase_completed": False,
                "feature_progress": read(WORK / "feature-progress.json")
                if args.phase == "features"
                and (WORK / "feature-progress.json").exists()
                else None,
                "no_detector_scores_created": True,
            },
        )
        raise


if __name__ == "__main__":
    main()
