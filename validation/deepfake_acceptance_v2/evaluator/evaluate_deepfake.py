"""D-v2 black-box deepfake evaluator.

This program never imports the model implementation. Each manifest row is run
in a separate subprocess and receives a separate artifact directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics import NEGATIVE_LABEL, POSITIVE_LABEL, compute_binary_metrics, compute_roc_auc


SCHEMA_VERSION = "d-v2-evaluator/1.0"
LABEL_MAPPING = {"real": NEGATIVE_LABEL, "deepfake": POSITIVE_LABEL}
REQUIRED_DATASET_FIELDS = ("sample_id", "file_path", "label")
SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_RESULT_BYTES = 5 * 1024 * 1024
CSV_ENCODING = "utf-8-sig"

PREDICTION_FIELDS = (
    "sample_index",
    "sample_id",
    "file_path",
    "ground_truth",
    "raw_label",
    "mapped_label",
    "effective_prediction",
    "score",
    "score_issue",
    "prediction_usable",
    "correct",
    "outcome",
    "failure_code",
    "failure_message",
    "runner_returncode",
    "timed_out",
    "runner_runtime_seconds",
    "source",
    "generation_method",
    "is_original",
    "parent_sample_id",
    "perturbation_type",
    "perturbation_level",
    "sha256",
)

ERROR_SAMPLE_FIELDS = (
    "sample_id",
    "file_path",
    "ground_truth",
    "score",
    "generation_method",
    "perturbation_type",
    "outcome",
    "failure_code",
    "failure_message",
    "raw_label",
    "mapped_label",
    "effective_prediction",
    "runner_returncode",
    "timed_out",
)


class DatasetError(ValueError):
    """Raised when the evaluator cannot establish a valid fixed denominator."""


class ResultJSONError(ValueError):
    """Raised for strict JSON syntax or object-integrity violations."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _decode_process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _reject_constant(token: str) -> None:
    raise ResultJSONError(f"non-standard JSON numeric constant: {token}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResultJSONError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_result_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_RESULT_BYTES:
        raise ResultJSONError(f"result JSON exceeds {MAX_RESULT_BYTES} bytes")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ResultJSONError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ResultJSONError("top-level JSON value is not an object")
    return payload


def _parse_is_original(value: str, perturbation_type: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    if normalized == "":
        return not bool(perturbation_type.strip())
    raise DatasetError(f"invalid is_original value: {value!r}")


def _resolve_dataset_file(dataset_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = dataset_path.parent / candidate
    return candidate.resolve(strict=False)


def _load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.is_file():
        raise DatasetError(f"dataset CSV does not exist: {dataset_path}")
    with dataset_path.open("r", encoding=CSV_ENCODING, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError("dataset CSV has no header")
        missing = [name for name in REQUIRED_DATASET_FIELDS if name not in reader.fieldnames]
        if missing:
            raise DatasetError(f"dataset CSV missing fields: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise DatasetError("dataset CSV contains zero samples")

    seen_ids: set[str] = set()
    loaded: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, start=1):
        sample_id = (raw.get("sample_id") or "").strip()
        if not SAFE_SAMPLE_ID.fullmatch(sample_id) or sample_id in {".", ".."}:
            raise DatasetError(f"row {index}: unsafe sample_id {sample_id!r}")
        if sample_id in seen_ids:
            raise DatasetError(f"row {index}: duplicate sample_id {sample_id!r}")
        seen_ids.add(sample_id)
        truth = (raw.get("label") or "").strip()
        if truth not in {NEGATIVE_LABEL, POSITIVE_LABEL}:
            raise DatasetError(f"row {index}: label must be exactly REAL or FAKE")
        file_value = (raw.get("file_path") or "").strip()
        if not file_value:
            raise DatasetError(f"row {index}: empty file_path")
        perturbation_type = (raw.get("perturbation_type") or "").strip()
        is_original = _parse_is_original(raw.get("is_original") or "", perturbation_type)
        loaded.append(
            {
                **raw,
                "_index": index,
                "_sample_id": sample_id,
                "_truth": truth,
                "_file_path": _resolve_dataset_file(dataset_path, file_value),
                "_is_original": is_original,
                "_perturbation_type": perturbation_type,
                "_generation_method": (raw.get("generation_method") or "").strip() or "unknown",
                "_parent_sample_id": (raw.get("parent_sample_id") or "").strip(),
            }
        )
    return loaded


def _opposite(label: str) -> str:
    return NEGATIVE_LABEL if label == POSITIVE_LABEL else POSITIVE_LABEL


def _score_from_payload(payload: dict[str, Any]) -> tuple[float | None, str]:
    value = payload.get("score")
    if value is None:
        return None, "score is missing or null"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "score is not numeric"
    score = float(value)
    if not math.isfinite(score):
        return None, "score is not finite"
    if score < 0.0 or score > 1.0:
        return None, "score is outside the confirmed probability range [0, 1]"
    return score, ""


def _validate_success_payload(
    payload: dict[str, Any], query_path: Path
) -> tuple[str | None, str | None, float | None, str, str | None, str]:
    """Return raw label, mapped label, score, score issue, failure code/message."""

    if payload.get("status") != "success":
        return None, None, None, "", "RESULT_STATUS_NOT_SUCCESS", (
            f"result status is {payload.get('status')!r}, expected 'success'"
        )
    if "label" not in payload or payload.get("label") is None:
        return None, None, None, "", "RESULT_LABEL_MISSING", "result label is missing or null"
    raw_label = payload.get("label")
    if not isinstance(raw_label, str) or raw_label not in LABEL_MAPPING:
        return str(raw_label), None, None, "", "RESULT_LABEL_INVALID", (
            f"label {raw_label!r} is not in the exact mapping {tuple(LABEL_MAPPING)}"
        )
    mapped_label = LABEL_MAPPING[raw_label]
    expected_boolean = mapped_label == POSITIVE_LABEL
    if "is_deepfake" in payload:
        is_deepfake = payload.get("is_deepfake")
        if not isinstance(is_deepfake, bool) or is_deepfake is not expected_boolean:
            return raw_label, mapped_label, None, "", "RESULT_INCONSISTENT", (
                "is_deepfake is non-boolean or inconsistent with label"
            )
    if "query_image" in payload:
        result_query = Path(str(payload.get("query_image"))).expanduser().resolve(strict=False)
        if os.path.normcase(str(result_query)) != os.path.normcase(str(query_path)):
            return raw_label, mapped_label, None, "", "RESULT_QUERY_MISMATCH", (
                f"result query_image {result_query} does not match {query_path}"
            )
    score, score_issue = _score_from_payload(payload)
    return raw_label, mapped_label, score, score_issue, None, ""


def _run_one_sample(
    row: dict[str, Any],
    *,
    python_executable: Path,
    python_args: list[str],
    runner: Path,
    runner_args: list[str],
    per_sample_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    sample_id = row["_sample_id"]
    truth = row["_truth"]
    query_path = row["_file_path"]
    sample_dir = per_sample_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=False)
    result_path = sample_dir / "result.json"
    stdout_path = sample_dir / "stdout.log"
    stderr_path = sample_dir / "stderr.log"
    status_path = sample_dir / "status.json"
    command = [
        str(python_executable),
        *python_args,
        str(runner),
        "--query",
        str(query_path),
        "--output",
        str(result_path),
        *runner_args,
    ]
    started_at = _utc_now()
    started_clock = time.perf_counter()
    returncode: int | None = None
    timed_out = False
    stdout = ""
    stderr = ""
    failure_code: str | None = None
    failure_message = ""
    raw_label: str | None = None
    mapped_label: str | None = None
    score: float | None = None
    score_issue = ""

    if not query_path.is_file():
        failure_code = "QUERY_FILE_MISSING"
        failure_message = f"query file does not exist: {query_path}"
    else:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _decode_process_text(exc.stdout)
            stderr = _decode_process_text(exc.stderr)
            failure_code = "RUNNER_TIMEOUT"
            failure_message = f"runner exceeded timeout of {timeout_seconds:g} seconds"
        except OSError as exc:
            failure_code = "RUNNER_LAUNCH_ERROR"
            failure_message = f"could not launch runner: {exc}"

    runner_runtime = max(0.0, time.perf_counter() - started_clock)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    if failure_code is None and returncode != 0:
        failure_code = "RUNNER_NONZERO_EXIT"
        failure_message = f"runner exited with code {returncode}"
    if failure_code is None and not result_path.is_file():
        failure_code = "RESULT_MISSING"
        failure_message = "runner returned zero but result.json does not exist"
    payload: dict[str, Any] | None = None
    if failure_code is None:
        try:
            payload = _load_result_json(result_path)
        except (OSError, ResultJSONError) as exc:
            failure_code = "RESULT_INVALID_JSON"
            failure_message = str(exc)
    if failure_code is None and payload is not None:
        (
            raw_label,
            mapped_label,
            score,
            score_issue,
            failure_code,
            failure_message,
        ) = _validate_success_payload(payload, query_path)

    prediction_usable = failure_code is None
    if prediction_usable:
        effective_prediction = mapped_label
        correct = mapped_label == truth
        outcome = "CORRECT" if correct else "MISCLASSIFIED"
    else:
        effective_prediction = _opposite(truth)
        correct = False
        outcome = "SYSTEM_FAILURE_COUNTED_WRONG"
        score = None

    prediction = {
        "sample_index": row["_index"],
        "sample_id": sample_id,
        "file_path": str(query_path),
        "ground_truth": truth,
        "raw_label": raw_label,
        "mapped_label": mapped_label,
        "effective_prediction": effective_prediction,
        "score": score,
        "score_issue": score_issue,
        "prediction_usable": prediction_usable,
        "correct": correct,
        "outcome": outcome,
        "failure_code": failure_code,
        "failure_message": failure_message,
        "runner_returncode": returncode,
        "timed_out": timed_out,
        "runner_runtime_seconds": runner_runtime,
        "source": (row.get("source") or "").strip(),
        "generation_method": row["_generation_method"],
        "is_original": row["_is_original"],
        "parent_sample_id": row["_parent_sample_id"],
        "perturbation_type": row["_perturbation_type"],
        "perturbation_level": (row.get("perturbation_level") or "").strip(),
        "sha256": (row.get("sha256") or "").strip(),
    }
    _write_json(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "ground_truth": truth,
            "command": command,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "timeout_seconds": timeout_seconds,
            "returncode": returncode,
            "timed_out": timed_out,
            "result_json_exists": result_path.is_file(),
            "prediction_usable": prediction_usable,
            "mapped_label": mapped_label,
            "effective_prediction": effective_prediction,
            "correct": correct,
            "outcome": outcome,
            "failure_code": failure_code,
            "failure_message": failure_message,
            "score_issue": score_issue,
            "artifacts": {
                "result_json": str(result_path),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
        },
    )
    return prediction


def _perturbation_group(row: dict[str, Any]) -> str:
    if row["is_original"]:
        return "Original"
    canonical = {
        "jpeg": "JPEG",
        "scale": "Scale",
        "blur": "Blur",
        "brightness": "Brightness",
    }
    raw = str(row.get("perturbation_type") or "").strip()
    return canonical.get(raw.lower(), f"Other:{raw or 'unknown'}")


def _group_metrics(predictions: list[dict[str, Any]], key_function: Any) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(key_function(row))].append(row)
    result: list[dict[str, Any]] = []
    preferred = {name: index for index, name in enumerate(("Original", "JPEG", "Scale", "Blur", "Brightness"))}
    for name in sorted(grouped, key=lambda value: (preferred.get(value, 99), value)):
        rows = grouped[name]
        metrics = compute_binary_metrics(
            (row["ground_truth"], row["effective_prediction"]) for row in rows
        )
        n_real = sum(row["ground_truth"] == NEGATIVE_LABEL for row in rows)
        n_fake = len(rows) - n_real
        result.append(
            {
                "group": name,
                "n_REAL": n_real,
                "n_FAKE": n_fake,
                "descriptive_only": len(rows) < 30 or n_real < 10 or n_fake < 10,
                **metrics,
            }
        )
    return result


def _flatten_group_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for group in groups:
        row = {
            key: value
            for key, value in group.items()
            if not isinstance(value, dict)
        }
        for name in ("Accuracy", "Recall", "Specificity"):
            metric = group[name]
            row[name] = metric["value"]
            row[f"{name}_status"] = metric["status"]
        flattened.append(row)
    return flattened


def _parent_summary(predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        parent = row["sample_id"] if row["is_original"] else row["parent_sample_id"]
        groups[parent or row["sample_id"]].append(row)
    summaries: list[dict[str, Any]] = []
    for parent in sorted(groups):
        rows = groups[parent]
        correct = sum(bool(row["correct"]) for row in rows)
        truths = sorted({str(row["ground_truth"]) for row in rows})
        methods = sorted({str(row["generation_method"]) for row in rows})
        summaries.append(
            {
                "parent_sample_id": parent,
                "ground_truth": truths[0] if len(truths) == 1 else "MIXED",
                "generation_method": methods[0] if len(methods) == 1 else "MIXED",
                "n_variants": len(rows),
                "n_correct": correct,
                "accuracy": correct / len(rows),
                "failure_count": sum(not bool(row["prediction_usable"]) for row in rows),
            }
        )
    if not summaries:
        return summaries, None
    return summaries, sum(row["accuracy"] for row in summaries) / len(summaries)


def _metric_markdown(metric: dict[str, Any]) -> str:
    if metric["value"] is None:
        return f"NOT_COMPUTED ({metric['status']})"
    return f"{metric['value']:.6f}"


def _build_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# D-v2 鉴伪评测报告",
        "",
        f"- 运行状态：`{report['run_status']}`",
        f"- 固定分母 N：{metrics['N']}",
        "- 阳性类：`FAKE`；阴性类：`REAL`",
        f"- 系统失败并固定计错：{report['failures']['count']}",
        "",
        "## 总体指标（含系统失败固定计错）",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
    ]
    for name in ("Accuracy", "Precision", "Recall", "Specificity", "F1"):
        lines.append(f"| {name} | {_metric_markdown(metrics[name])} |")
    lines.extend(
        [
            f"| TP | {metrics['TP']} |",
            f"| FP | {metrics['FP']} |",
            f"| TN | {metrics['TN']} |",
            f"| FN | {metrics['FN']} |",
            "",
            "## ROC-AUC",
            "",
        ]
    )
    auc = metrics["ROC_AUC"]
    if auc["value"] is None:
        lines.append(f"`NOT_COMPUTED`：{auc['reason']}")
    else:
        lines.append(f"ROC-AUC：{auc['value']:.6f}")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "Original、JPEG、Scale、Blur、Brightness 与 generation_method 的结果见 JSON/CSV。",
            "所有小样本分组均通过 `descriptive_only` 标记；含同源扰动的总体微平均仅用于诊断，不能当成独立样本主成绩。",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_hashes(output_dir: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = output_dir / name
        result[name] = {"sha256": _sha256(path), "file_size": path.stat().st_size}
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset).expanduser().resolve(strict=False)
    runner = Path(args.runner).expanduser().resolve(strict=False)
    python_executable = Path(args.python).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    if not runner.is_file():
        raise DatasetError(f"runner does not exist: {runner}")
    if not python_executable.is_file():
        raise DatasetError(f"python executable does not exist: {python_executable}")
    if args.timeout_seconds <= 0:
        raise DatasetError("timeout must be greater than zero")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DatasetError(f"output directory is not empty: {output_dir}")

    dataset_rows = _load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_root = output_dir / "per_sample"
    per_sample_root.mkdir(parents=True, exist_ok=False)
    run_id = str(uuid.uuid4())
    started_at = _utc_now()
    predictions: list[dict[str, Any]] = []
    for row in dataset_rows:
        predictions.append(
            _run_one_sample(
                row,
                python_executable=python_executable,
                python_args=list(args.python_arg),
                runner=runner,
                runner_args=list(args.runner_arg),
                per_sample_root=per_sample_root,
                timeout_seconds=args.timeout_seconds,
            )
        )

    metrics = compute_binary_metrics(
        (row["ground_truth"], row["effective_prediction"]) for row in predictions
    )
    if metrics["N"] != len(dataset_rows):
        raise RuntimeError("fixed-denominator invariant failed")
    auc = compute_roc_auc(
        [row["ground_truth"] for row in predictions],
        [row["score"] for row in predictions],
        score_semantics=args.score_semantics,
        all_predictions_usable=all(bool(row["prediction_usable"]) for row in predictions),
    )
    metrics["ROC_AUC"] = auc

    perturbation_groups = _group_metrics(predictions, _perturbation_group)
    generation_groups = _group_metrics(predictions, lambda row: row["generation_method"])
    parents, parent_macro_accuracy = _parent_summary(predictions)
    failures = [row for row in predictions if not row["prediction_usable"]]
    false_positives = [
        row
        for row in predictions
        if row["ground_truth"] == NEGATIVE_LABEL
        and row["effective_prediction"] == POSITIVE_LABEL
    ]
    false_negatives = [
        row
        for row in predictions
        if row["ground_truth"] == POSITIVE_LABEL
        and row["effective_prediction"] == NEGATIVE_LABEL
    ]

    _write_csv(output_dir / "predictions.csv", list(PREDICTION_FIELDS), predictions)
    _write_csv(
        output_dir / "confusion_matrix.csv",
        ["Actual", "Pred_REAL", "Pred_FAKE"],
        [
            {"Actual": "True_REAL", "Pred_REAL": metrics["TN"], "Pred_FAKE": metrics["FP"]},
            {"Actual": "True_FAKE", "Pred_REAL": metrics["FN"], "Pred_FAKE": metrics["TP"]},
        ],
    )
    _write_csv(output_dir / "failures.csv", list(ERROR_SAMPLE_FIELDS), failures)
    _write_csv(output_dir / "false_positive_samples.csv", list(ERROR_SAMPLE_FIELDS), false_positives)
    _write_csv(output_dir / "false_negative_samples.csv", list(ERROR_SAMPLE_FIELDS), false_negatives)
    _write_csv(
        output_dir / "false_negatives.csv",
        [
            "sample_id",
            "file_path",
            "score",
            "generation_method",
            "perturbation_type",
            "outcome",
            "failure_code",
        ],
        false_negatives,
    )
    group_fields = [
        "group",
        "N",
        "n_REAL",
        "n_FAKE",
        "TP",
        "FP",
        "TN",
        "FN",
        "Accuracy",
        "Accuracy_status",
        "Recall",
        "Recall_status",
        "Specificity",
        "Specificity_status",
        "descriptive_only",
    ]
    _write_csv(
        output_dir / "metrics_by_perturbation.csv",
        group_fields,
        _flatten_group_rows(perturbation_groups),
    )
    _write_csv(
        output_dir / "metrics_by_generation_method.csv",
        group_fields,
        _flatten_group_rows(generation_groups),
    )
    _write_csv(
        output_dir / "parent_summary.csv",
        [
            "parent_sample_id",
            "ground_truth",
            "generation_method",
            "n_variants",
            "n_correct",
            "accuracy",
            "failure_count",
        ],
        parents,
    )

    failure_counts = Counter(str(row["failure_code"]) for row in failures)
    run_status = "COMPLETED_WITH_SAMPLE_FAILURES" if failures else "COMPLETED"
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_status": run_status,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "sample_count": len(dataset_rows),
        },
        "policy": {
            "positive_class": POSITIVE_LABEL,
            "negative_class": NEGATIVE_LABEL,
            "exact_label_mapping": LABEL_MAPPING,
            "fixed_denominator": True,
            "failure_effective_prediction": {
                "True_REAL": POSITIVE_LABEL,
                "True_FAKE": NEGATIVE_LABEL,
            },
            "score_semantics": args.score_semantics,
            "descriptive_only_rule": "N < 30 or n_REAL < 10 or n_FAKE < 10",
            "all_samples_micro_is_diagnostic_when_perturbations_exist": True,
        },
        "metrics": metrics,
        "groups": {
            "perturbation": perturbation_groups,
            "generation_method": generation_groups,
        },
        "parent_aggregation": {
            "parent_count": len(parents),
            "parent_macro_accuracy": parent_macro_accuracy,
        },
        "failures": {"count": len(failures), "by_code": dict(sorted(failure_counts.items()))},
        "artifacts": {
            "predictions": "predictions.csv",
            "confusion_matrix": "confusion_matrix.csv",
            "failures": "failures.csv",
            "false_positives": "false_positive_samples.csv",
            "false_negatives": "false_negatives.csv",
            "per_sample": "per_sample",
        },
    }
    _write_json(output_dir / "evaluation_report.json", report)
    (output_dir / "evaluation_report.md").write_text(_build_markdown(report), encoding="utf-8")

    top_level_artifacts = [
        "predictions.csv",
        "confusion_matrix.csv",
        "evaluation_report.json",
        "evaluation_report.md",
        "failures.csv",
        "false_positive_samples.csv",
        "false_negative_samples.csv",
        "false_negatives.csv",
        "metrics_by_perturbation.csv",
        "metrics_by_generation_method.csv",
        "parent_summary.csv",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_status": run_status,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "python": str(python_executable),
        "python_args": list(args.python_arg),
        "runner": {"path": str(runner), "sha256": _sha256(runner)},
        "runner_args": list(args.runner_arg),
        "dataset": {"path": str(dataset_path), "sha256": _sha256(dataset_path)},
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "metrics_module": {
            "path": str(Path(__file__).with_name("metrics.py").resolve()),
            "sha256": _sha256(Path(__file__).with_name("metrics.py").resolve()),
        },
        "timeout_seconds": args.timeout_seconds,
        "score_semantics": args.score_semantics,
        "sample_count": len(dataset_rows),
        "failure_count": len(failures),
        "artifact_hashes": _artifact_hashes(output_dir, top_level_artifacts),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-v2 frozen black-box deepfake evaluator")
    parser.add_argument("--python", required=True, help="Python executable used to invoke the runner")
    parser.add_argument("--runner", required=True, help="Frozen run_deepfake_detection.py path")
    parser.add_argument("--dataset", required=True, help="Validated D-v2 ground-truth CSV")
    parser.add_argument("--output-dir", required=True, help="New or empty evaluation run directory")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Per-sample timeout")
    parser.add_argument(
        "--score-semantics",
        choices=("unconfirmed", "higher_is_more_fake"),
        default="unconfirmed",
        help="AUC is calculated only for the explicitly confirmed higher_is_more_fake value",
    )
    parser.add_argument(
        "--python-arg",
        action="append",
        default=[],
        help="Repeatable Python argument; use --python-arg=-S for dash-prefixed values",
    )
    parser.add_argument(
        "--runner-arg",
        action="append",
        default=[],
        help="Repeatable runner argument; use --runner-arg=--name=value",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate(args)
    except (DatasetError, OSError, RuntimeError, ValueError) as exc:
        print(f"EVALUATOR_ERROR: {exc}", file=sys.stderr)
        return 2
    metrics = report["metrics"]
    print(
        json.dumps(
            {
                "run_status": report["run_status"],
                "N": metrics["N"],
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "TN": metrics["TN"],
                "FN": metrics["FN"],
                "Accuracy": metrics["Accuracy"]["value"],
                "failure_count": report["failures"]["count"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
