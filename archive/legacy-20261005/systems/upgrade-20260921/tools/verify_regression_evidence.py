#!/usr/bin/env python3
"""Build and verify the v0.2-to-v0.3 real-query regression evidence.

This script deliberately reads the archived, real execution artifacts instead of
calling a mock or recomputing model scores.  It produces a machine-readable JSON
comparison, a short Chinese text comparison, and a validation result for the
three generated reports and the live progress stream.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


BASELINE_EXPECTED_QUERIES = {
    "integrated_query_001.jpg",
    "integrated_query_002.jpg",
    "integrated_query_003.jpg",
    "aigc_fake_image.jpg",
    "aigc_real_image.jpg",
    "no_face_image.jpg",
}
TOLERANCES = {"deepfake": 1e-9, "identity": 1e-4, "source": 1e-6}
REQUIRED_EVENT_FIELDS = {
    "event_id",
    "timestamp",
    "task_id",
    "module",
    "stage",
    "status",
    "message",
    "progress_current",
    "progress_total",
    "progress_percent",
    "elapsed_seconds",
    "details",
}
ALLOWED_MODULES = {"overall", "deepfake", "identity", "source"}
ALLOWED_STATUSES = {
    "pending",
    "running",
    "success",
    "warning",
    "error",
    "skipped",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"JSON 顶层不是对象：{path}")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    )


def _basename(value: Any) -> str:
    # Pure Path on the current OS does not split the other platform's separator.
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]


def _compare_number(
    name: str, baseline: Any, candidate: Any, tolerance: float
) -> dict[str, Any]:
    baseline_number = float(baseline)
    candidate_number = float(candidate)
    difference = candidate_number - baseline_number
    return {
        "metric": name,
        "v0.2": baseline_number,
        "v0.3": candidate_number,
        "absolute_difference": abs(difference),
        "signed_difference": difference,
        "tolerance": tolerance,
        "passed": math.isclose(
            baseline_number, candidate_number, rel_tol=0.0, abs_tol=tolerance
        ),
    }


def _find_baseline_tasks(baseline_root: Path) -> tuple[list[dict[str, Any]], Path]:
    inventory: list[dict[str, Any]] = []
    query001_task: Path | None = None
    for task_dir in sorted(path for path in baseline_root.iterdir() if path.is_dir()):
        status_path = task_dir / "task_status.json"
        if not status_path.is_file():
            continue
        status = _read_json(status_path)
        query_name = _basename(status.get("query_source") or status.get("query_image"))
        selected_modules = status.get("selected_modules") or []
        inventory.append(
            {
                "task_id": status.get("task_id") or task_dir.name,
                "directory": task_dir.name,
                "query_file": query_name,
                "status": status.get("status"),
                "selected_modules": list(selected_modules),
            }
        )
        if query_name == "integrated_query_001.jpg":
            query001_task = task_dir

    observed_queries = {item["query_file"] for item in inventory}
    if observed_queries != BASELINE_EXPECTED_QUERIES:
        raise AssertionError(
            "六个基线样例集合不一致："
            f"expected={sorted(BASELINE_EXPECTED_QUERIES)}, "
            f"observed={sorted(observed_queries)}"
        )
    if len(inventory) != 6:
        raise AssertionError(f"应有 6 个基线任务，实际为 {len(inventory)} 个")
    if query001_task is None:
        raise AssertionError("未找到 integrated_query_001.jpg 的 v0.2 基线任务")
    return inventory, query001_task


def build_query001_comparison(
    root: Path, v3_task_dir: Path
) -> dict[str, Any]:
    baseline_root = root / "docs" / "baseline_test_results"
    inventory, v2_task_dir = _find_baseline_tasks(baseline_root)

    v3_status = _read_json(v3_task_dir / "task_status.json")
    v3_query_name = _basename(
        v3_status.get("query_source") or v3_status.get("query_image")
    )
    if v3_query_name != "integrated_query_001.jpg":
        raise AssertionError(
            f"v0.3 任务不是 integrated_query_001.jpg：{v3_query_name}"
        )

    v2_deepfake = _read_json(v2_task_dir / "detection_result.json")
    v3_deepfake = _read_json(v3_task_dir / "detection_result.json")
    v2_identity = _read_json(v2_task_dir / "identity_result.json")
    v3_identity = _read_json(v3_task_dir / "identity_result.json")
    v2_source = _read_json(v2_task_dir / "source_result.json")
    v3_source = _read_json(v3_task_dir / "source_result.json")

    v2_df_details = v2_deepfake["details"]
    v3_df_metrics = v3_deepfake["metrics"]
    deepfake_metrics = [
        _compare_number(
            "spatial_score",
            v2_df_details["spatial_score"],
            v3_df_metrics["spatial_score"],
            TOLERANCES["deepfake"],
        ),
        _compare_number(
            "frequency_score",
            v2_df_details["frequency_score"],
            v3_df_metrics["frequency_score"],
            TOLERANCES["deepfake"],
        ),
        _compare_number(
            "fusion_score",
            v2_df_details["fusion_score"],
            v3_df_metrics["fusion_score"],
            TOLERANCES["deepfake"],
        ),
        _compare_number(
            "decision_threshold",
            v2_df_details["decision_threshold"],
            v3_df_metrics["decision_threshold"],
            TOLERANCES["deepfake"],
        ),
    ]
    deepfake_classification_equal = (
        bool(v2_deepfake.get("is_deepfake")) == bool(v3_deepfake.get("is_deepfake"))
        and str(v2_deepfake.get("label")) == str(v3_deepfake.get("label"))
    )

    v2_identity_candidates = v2_identity["candidates"]
    v3_identity_candidates = v3_identity["candidates"]
    identity_rows: list[dict[str, Any]] = []
    for old, new in zip(v2_identity_candidates, v3_identity_candidates):
        score = _compare_number(
            f"rank_{old['rank']}_mean_cosine_similarity",
            old["score"],
            new.get("mean_similarity", new["score"]),
            TOLERANCES["identity"],
        )
        identity_rows.append(
            {
                "rank": old["rank"],
                "v0.2_identity_id": old["identity_id"],
                "v0.3_identity_id": new["identity_id"],
                "ranking_equal": (
                    old["rank"] == new["rank"]
                    and old["identity_id"] == new["identity_id"]
                ),
                "score_comparison": score,
            }
        )
    identity_count_equal = len(v2_identity_candidates) == len(v3_identity_candidates)

    v2_source_candidates = v2_source["candidates"]
    v3_source_candidates = v3_source["candidates"]
    source_rows: list[dict[str, Any]] = []
    for old, new in zip(v2_source_candidates, v3_source_candidates):
        score_comparisons = [
            _compare_number(
                f"rank_{old['rank']}_{score_name}",
                old[score_name],
                new[score_name],
                TOLERANCES["source"],
            )
            for score_name in ("global_score", "local_score", "final_score")
        ]
        source_rows.append(
            {
                "rank": old["rank"],
                "v0.2_file": _basename(old["file_path"]),
                "v0.3_file": _basename(new["file_path"]),
                "ranking_equal": (
                    old["rank"] == new["rank"]
                    and _basename(old["file_path"]) == _basename(new["file_path"])
                ),
                "score_comparisons": score_comparisons,
            }
        )
    source_count_equal = len(v2_source_candidates) == len(v3_source_candidates)

    input_sha256_v2 = v2_df_details.get("input_sha256")
    input_sha256_v3 = (
        v3_deepfake.get("provenance", {}).get("input_sha256")
        or v3_deepfake.get("details", {}).get("input_sha256")
    )
    all_checks: list[bool] = [
        input_sha256_v2 == input_sha256_v3,
        deepfake_classification_equal,
        *(item["passed"] for item in deepfake_metrics),
        identity_count_equal,
        *(item["ranking_equal"] for item in identity_rows),
        *(item["score_comparison"]["passed"] for item in identity_rows),
        source_count_equal,
        *(item["ranking_equal"] for item in source_rows),
        *(
            metric["passed"]
            for item in source_rows
            for metric in item["score_comparisons"]
        ),
    ]

    return {
        "schema_version": "1.0",
        "evidence_type": "real_model_regression_comparison",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sample": "integrated_query_001.jpg",
        "uses_mock": False,
        "baseline_inventory": inventory,
        "artifacts": {
            "v0.2_task": str(v2_task_dir),
            "v0.3_task": str(v3_task_dir),
        },
        "input_sha256": {
            "v0.2": input_sha256_v2,
            "v0.3": input_sha256_v3,
            "equal": input_sha256_v2 == input_sha256_v3,
        },
        "tolerances": TOLERANCES,
        "deepfake": {
            "metrics": deepfake_metrics,
            "classification_equal": deepfake_classification_equal,
            "v0.2_label": v2_deepfake.get("label"),
            "v0.3_label": v3_deepfake.get("label"),
            "passed": deepfake_classification_equal
            and all(item["passed"] for item in deepfake_metrics),
        },
        "identity": {
            "candidate_count_equal": identity_count_equal,
            "top_k": identity_rows,
            "maximum_absolute_score_difference": max(
                (item["score_comparison"]["absolute_difference"] for item in identity_rows),
                default=0.0,
            ),
            "passed": identity_count_equal
            and all(item["ranking_equal"] for item in identity_rows)
            and all(item["score_comparison"]["passed"] for item in identity_rows),
        },
        "source": {
            "candidate_count_equal": source_count_equal,
            "top_k": source_rows,
            "maximum_absolute_score_difference": max(
                (
                    metric["absolute_difference"]
                    for item in source_rows
                    for metric in item["score_comparisons"]
                ),
                default=0.0,
            ),
            "passed": source_count_equal
            and all(item["ranking_equal"] for item in source_rows)
            and all(
                metric["passed"]
                for item in source_rows
                for metric in item["score_comparisons"]
            ),
        },
        "overall_passed": all(all_checks),
        "conclusion": (
            "同一真实样例的核心算法分数和候选排名在规定绝对误差范围内保持一致。"
            if all(all_checks)
            else "至少一项核心算法分数或候选排名超出规定误差，详见逐项结果。"
        ),
    }


def _read_progress_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise AssertionError(f"进度 JSONL 第 {line_number} 行为空")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"进度 JSONL 第 {line_number} 行不是完整 JSON：{exc}"
                ) from exc
            if not isinstance(event, dict):
                raise AssertionError(f"进度 JSONL 第 {line_number} 行顶层不是对象")
            missing = REQUIRED_EVENT_FIELDS.difference(event)
            if missing:
                raise AssertionError(
                    f"进度 JSONL 第 {line_number} 行缺少字段：{sorted(missing)}"
                )
            events.append(event)
    if not events:
        raise AssertionError("进度 JSONL 为空")
    return events


def validate_reports_and_progress(v3_task_dir: Path) -> dict[str, Any]:
    task_id = v3_task_dir.name
    report_paths = {
        "json": v3_task_dir / "final_report.json",
        "txt": v3_task_dir / f"检测结果_{task_id}.txt",
        "html": v3_task_dir / f"检测结果_{task_id}.html",
    }
    for report_path in report_paths.values():
        if not report_path.is_file() or report_path.stat().st_size <= 0:
            raise AssertionError(f"报告不存在或为空：{report_path}")

    final_report = _read_json(report_paths["json"])
    text_report = report_paths["txt"].read_text(encoding="utf-8")
    html_report = report_paths["html"].read_text(encoding="utf-8")
    if "\ufffd" in text_report or "\ufffd" in html_report:
        raise AssertionError("TXT 或 HTML 报告包含 Unicode 替换字符，疑似乱码")

    raw_results = {
        "deepfake": _read_json(v3_task_dir / "detection_result.json"),
        "identity": _read_json(v3_task_dir / "identity_result.json"),
        "source": _read_json(v3_task_dir / "source_result.json"),
    }
    report_result_equality = {
        module: final_report.get("modules", {}).get(module, {}).get("result") == result
        for module, result in raw_results.items()
    }
    if not all(report_result_equality.values()):
        raise AssertionError(
            f"final_report.json 与原始模块 JSON 不一致：{report_result_equality}"
        )

    expected_numeric_texts = {
        f"{raw_results['deepfake']['metrics']['spatial_score']:.6f}",
        f"{raw_results['deepfake']['metrics']['frequency_score']:.6f}",
        f"{raw_results['deepfake']['metrics']['fusion_score']:.6f}",
        f"{raw_results['deepfake']['metrics']['decision_threshold']:.6f}",
        f"{raw_results['identity']['candidates'][0]['mean_similarity']:.6f}",
        f"{raw_results['source']['candidates'][0]['global_score']:.6f}",
        f"{raw_results['source']['candidates'][0]['local_score']:.6f}",
        f"{raw_results['source']['candidates'][0]['final_score']:.6f}",
    }
    missing_in_txt = sorted(value for value in expected_numeric_texts if value not in text_report)
    missing_in_html = sorted(value for value in expected_numeric_texts if value not in html_report)
    if missing_in_txt or missing_in_html:
        raise AssertionError(
            "报告未包含原始 JSON 的六位小数值："
            f"txt_missing={missing_in_txt}, html_missing={missing_in_html}"
        )
    required_report_phrases = {
        task_id,
        "不是最终身份确认",
        "当前底图库内",
        "不替代原始介质固定",
    }
    for phrase in required_report_phrases:
        if phrase not in text_report or phrase not in html_report:
            raise AssertionError(f"TXT 或 HTML 报告缺少必要文本：{phrase}")

    progress_path = v3_task_dir / "logs" / "progress_events.jsonl"
    events = _read_progress_events(progress_path)
    event_ids = [str(event["event_id"]) for event in events]
    if len(set(event_ids)) != len(event_ids):
        raise AssertionError("进度事件 event_id 存在重复")

    timestamps = [datetime.fromisoformat(str(event["timestamp"])) for event in events]
    if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
        raise AssertionError("进度事件时间戳发生倒退")
    if any(event["task_id"] != task_id for event in events):
        raise AssertionError("进度事件 task_id 与任务目录不一致")
    invalid_modules = sorted({event["module"] for event in events} - ALLOWED_MODULES)
    invalid_statuses = sorted({event["status"] for event in events} - ALLOWED_STATUSES)
    if invalid_modules or invalid_statuses:
        raise AssertionError(
            f"进度事件枚举非法：modules={invalid_modules}, statuses={invalid_statuses}"
        )

    percent_sequences: dict[str, list[float]] = {}
    for module in sorted(ALLOWED_MODULES):
        values: list[float] = []
        for event in events:
            if event["module"] != module or event["progress_percent"] is None:
                continue
            percent = float(event["progress_percent"])
            if not 0.0 <= percent <= 100.0:
                raise AssertionError(f"{module} 的进度百分比越界：{percent}")
            values.append(percent)
        if any(current < previous for previous, current in zip(values, values[1:])):
            raise AssertionError(f"{module} 的进度百分比发生倒退：{values}")
        percent_sequences[module] = values

    scan_events = [event for event in events if event["stage"] == "scan_source_library"]
    scan_currents = [int(event["progress_current"]) for event in scan_events]
    if (
        not scan_currents
        or scan_currents[0] != 1
        or scan_currents[-1] != 203
        or any(int(event["progress_total"]) != 203 for event in scan_events)
        or any(current <= previous for previous, current in zip(scan_currents, scan_currents[1:]))
    ):
        raise AssertionError(
            f"底图库扫描进度未形成严格递增的 1..203 实际范围：{scan_currents}"
        )

    review_events = [
        event for event in events if event["stage"] == "source_geometric_review"
    ]
    review_currents = [int(event["progress_current"]) for event in review_events]
    if review_currents != list(range(1, 21)) or any(
        int(event["progress_total"]) != 20 for event in review_events
    ):
        raise AssertionError(
            f"底图几何复核进度不是完整的 1..20：{review_currents}"
        )

    last_event = events[-1]
    if not (
        last_event["module"] == "overall"
        and last_event["stage"] == "completed"
        and last_event["status"] == "success"
        and float(last_event["progress_percent"]) == 100.0
    ):
        raise AssertionError(f"任务最终进度事件不正确：{last_event}")
    latest = _read_json(v3_task_dir / "logs" / "progress_latest.json")
    if latest != last_event:
        raise AssertionError("progress_latest.json 与 JSONL 最后一条完整事件不一致")

    return {
        "schema_version": "1.0",
        "evidence_type": "report_and_progress_validation",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task_id": task_id,
        "reports": {
            "passed": True,
            "files": {
                kind: {"path": str(path), "size_bytes": path.stat().st_size}
                for kind, path in report_paths.items()
            },
            "utf8_decoding": True,
            "raw_module_json_embedded_without_change": report_result_equality,
            "checked_numeric_values": sorted(expected_numeric_texts),
        },
        "progress": {
            "passed": True,
            "event_count": len(events),
            "all_json_lines_valid": True,
            "required_fields_present": True,
            "event_ids_unique": True,
            "timestamps_monotonic": True,
            "module_percentages_non_decreasing": {
                module: True for module in percent_sequences
            },
            "scan_source_library": {
                "observed_event_count": len(scan_currents),
                "first_current": scan_currents[0],
                "last_current": scan_currents[-1],
                "progress_total": 203,
                "strictly_increasing": True,
                "note": "按节流策略记录关键数量变化；真实范围从 1/203 到 203/203。",
            },
            "source_geometric_review": {
                "observed_event_count": len(review_currents),
                "currents": review_currents,
                "progress_total": 20,
                "complete_1_through_20": True,
            },
            "latest_matches_last_jsonl_event": True,
            "final_event_success_100_percent": True,
        },
        "overall_passed": True,
    }


def render_comparison_text(comparison: dict[str, Any]) -> str:
    lines = [
        "v0.2 与 v0.3 真实模型回归对比（integrated_query_001.jpg）",
        "=" * 68,
        f"生成时间：{comparison['generated_at']}",
        "数据来源：已归档的真实模块结果 JSON（非 Mock，不重新计算模型分数）",
        f"v0.2 任务：{Path(comparison['artifacts']['v0.2_task']).name}",
        f"v0.3 任务：{Path(comparison['artifacts']['v0.3_task']).name}",
        f"输入 SHA-256 一致：{'是' if comparison['input_sha256']['equal'] else '否'}",
        "",
        "一、AIGC / 深伪风险甄别",
        f"绝对误差容差：{comparison['tolerances']['deepfake']:.0e}",
    ]
    for item in comparison["deepfake"]["metrics"]:
        lines.append(
            f"- {item['metric']}: v0.2={item['v0.2']:.15g}, "
            f"v0.3={item['v0.3']:.15g}, |差值|={item['absolute_difference']:.6g}, "
            f"{'通过' if item['passed'] else '未通过'}"
        )
    lines.extend(
        [
            f"- 标签与判定一致：{'是' if comparison['deepfake']['classification_equal'] else '否'}",
            "",
            "二、身份候选排序",
            f"绝对误差容差：{comparison['tolerances']['identity']:.0e}",
        ]
    )
    for item in comparison["identity"]["top_k"]:
        score = item["score_comparison"]
        lines.append(
            f"- Top-{item['rank']}: {item['v0.2_identity_id']} -> "
            f"{item['v0.3_identity_id']}；v0.2={score['v0.2']:.15g}，"
            f"v0.3={score['v0.3']:.15g}，|差值|={score['absolute_difference']:.6g}，"
            f"{'通过' if item['ranking_equal'] and score['passed'] else '未通过'}"
        )
    lines.extend(
        [
            "说明：比较值是人员级平均余弦相似度，不是身份概率。",
            "",
            "三、底图候选排序",
            f"绝对误差容差：{comparison['tolerances']['source']:.0e}",
        ]
    )
    for item in comparison["source"]["top_k"]:
        metrics = ", ".join(
            f"{metric['metric'].split('_', 2)[-1]} "
            f"v0.2={metric['v0.2']:.6f}/v0.3={metric['v0.3']:.6f} "
            f"|差值|={metric['absolute_difference']:.6g}"
            for metric in item["score_comparisons"]
        )
        lines.append(
            f"- Top-{item['rank']}: {item['v0.2_file']} -> {item['v0.3_file']}；"
            f"{metrics}；{'通过' if item['ranking_equal'] and all(m['passed'] for m in item['score_comparisons']) else '未通过'}"
        )
    lines.extend(
        [
            "说明：全局分、局部分和最终分仅用于当前底图库候选排序，不是来源概率。",
            "",
            "四、结论",
            comparison["conclusion"],
            f"总体验证：{'通过' if comparison['overall_passed'] else '未通过'}",
            "",
        ]
    )
    return "\n".join(lines)


def run(root: Path, v3_task_dir: Path, *, write_outputs: bool = True) -> dict[str, Any]:
    comparison = build_query001_comparison(root, v3_task_dir)
    validation = validate_reports_and_progress(v3_task_dir)
    overall_passed = bool(comparison["overall_passed"] and validation["overall_passed"])
    result = {
        "comparison": comparison,
        "validation": validation,
        "overall_passed": overall_passed,
    }
    if write_outputs:
        output_root = root / "docs" / "regression_test_results"
        _atomic_write_json(
            output_root / "v0.2_v0.3_query001_comparison.json", comparison
        )
        _atomic_write_text(
            output_root / "v0.2_v0.3_query001_comparison.txt",
            render_comparison_text(comparison),
        )
        _atomic_write_json(
            output_root / "regression_evidence_validation.json", validation
        )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument(
        "--v3-task",
        type=Path,
        default=(
            project_root
            / "docs"
            / "regression_test_results"
            / "case_20260826_001607_552e"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run(args.root.resolve(), args.v3_task.resolve(), write_outputs=True)
    print(
        json.dumps(
            {
                "overall_passed": result["overall_passed"],
                "comparison_passed": result["comparison"]["overall_passed"],
                "reports_and_progress_passed": result["validation"]["overall_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
