"""Verify installed v0.4 tasks and emit phase-2 business acceptance evidence.

The script never runs a model and never invents a pass.  It reads tasks already
created by the installed UI, validates their result/report/artifact files and
checks the historical reference case within the agreed 1e-6 tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


MODULES = ("deepfake", "identity", "source")
EXPECTED_QUERY_SHA256 = "5a09a99a2e0b42ed7cfe923a15994f038ca815c03cba972419a25d8ae0e3baed"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON顶层不是对象：{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(actual: Any, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise AssertionError(f"{label}不是数值：{actual!r}")
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError(f"{label}超出1e-6：{actual!r} != {expected!r}")


def _validate_manifest(task_dir: Path) -> dict[str, Any]:
    manifest_path = task_dir / "artifact_manifest.json"
    manifest = _read(manifest_path)
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise AssertionError(f"任务级artifact manifest为空：{manifest_path}")
    verified = 0
    for item in records:
        if not isinstance(item, dict):
            raise AssertionError("artifact记录不是对象")
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise AssertionError("artifact路径为空")
        portable = PurePosixPath(relative)
        if portable.is_absolute() or ".." in portable.parts or "\\" in relative:
            raise AssertionError(f"artifact路径不安全：{relative}")
        path = task_dir.joinpath(*portable.parts).resolve(strict=True)
        path.relative_to(task_dir.resolve(strict=True))
        if path.stat().st_size != item.get("size_bytes"):
            raise AssertionError(f"artifact大小不一致：{relative}")
        if _sha256(path).lower() != str(item.get("sha256") or "").lower():
            raise AssertionError(f"artifact SHA-256不一致：{relative}")
        verified += 1
    return {"path": str(manifest_path), "verified_artifacts": verified}


def _task_record(task_dir: Path) -> tuple[frozenset[str], dict[str, Any]]:
    status = _read(task_dir / "task_status.json")
    selected_raw = status.get("selected_modules")
    if not isinstance(selected_raw, list):
        raise AssertionError(f"selected_modules缺失：{task_dir}")
    selected = frozenset(str(item) for item in selected_raw)
    record: dict[str, Any] = {
        "task_id": str(status.get("task_id") or task_dir.name),
        "task_dir": str(task_dir.resolve()),
        "status": str(status.get("status") or "error"),
        "selected_modules": sorted(selected),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "runtime_seconds": status.get("runtime_seconds"),
    }
    if record["status"] not in {"success", "partial_success"}:
        raise AssertionError(f"安装任务未成功：{record}")
    for module in selected:
        result_name = {
            "deepfake": "detection_result.json",
            "identity": "identity_result.json",
            "source": "source_result.json",
        }[module]
        result = _read(task_dir / result_name)
        if result.get("status") != "success":
            raise AssertionError(f"{module}结果未成功：{task_dir}")
    if selected == frozenset(MODULES):
        detection = _read(task_dir / "detection_result.json")
        identity = _read(task_dir / "identity_result.json")
        source = _read(task_dir / "source_result.json")
        query_sha = _sha256(task_dir / "query.jpg")
        if query_sha != EXPECTED_QUERY_SHA256:
            raise AssertionError(f"综合任务不是指定基准输入：{query_sha}")
        metrics = detection["metrics"]
        _assert_close(metrics["spatial_score"], 0.3240990936756134, "空间域")
        _assert_close(metrics["frequency_score"], 0.01133384222433596, "频域")
        _assert_close(detection["score"], 0.20524829812412798, "融合")
        _assert_close(metrics["decision_threshold"], 0.36360106194654707, "阈值")
        if identity["candidates"][0]["identity_id"] != "person_001":
            raise AssertionError("身份Top1变化")
        _assert_close(identity["candidates"][0]["mean_similarity"], 0.6991182208061218, "身份Top1")
        if Path(source["candidates"][0]["file_path"]).name != "051.jpg":
            raise AssertionError("底图Top1变化")
        _assert_close(source["candidates"][0]["global_score"], 0.534041, "底图全局")
        _assert_close(source["candidates"][0]["local_score"], 0.767547, "底图局部")
        _assert_close(source["candidates"][0]["final_score"], 0.662469, "底图最终")
        record["reference_case"] = {
            "query_sha256": query_sha,
            "deepfake_score": detection["score"],
            "identity_top1": identity["candidates"][0]["identity_id"],
            "identity_top1_score": identity["candidates"][0]["mean_similarity"],
            "source_top1": Path(source["candidates"][0]["file_path"]).name,
            "source_top1_score": source["candidates"][0]["final_score"],
            "tolerance": 1e-6,
        }
        record["artifact_manifest"] = _validate_manifest(task_dir)
    return selected, record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks_root = args.data_root.resolve(strict=True) / "tasks"
    by_selection: dict[frozenset[str], list[dict[str, Any]]] = {}
    for task_dir in sorted(tasks_root.glob("case_*")):
        if not task_dir.is_dir() or not (task_dir / "task_status.json").is_file():
            continue
        selected, record = _task_record(task_dir)
        by_selection.setdefault(selected, []).append(record)
    required = {
        "deepfake": frozenset({"deepfake"}),
        "identity": frozenset({"identity"}),
        "source": frozenset({"source"}),
        "comprehensive": frozenset(MODULES),
    }
    output: dict[str, Any] = {"schema_version": "1.0", "status": "passed"}
    for name, selection in required.items():
        records = by_selection.get(selection, [])
        if not records:
            raise AssertionError(f"缺少真实安装任务：{name}")
        output[name] = records[-1]
    task_dir = Path(output["comprehensive"]["task_dir"])
    reports = {
        "json": task_dir / "final_report.json",
        "txt": task_dir / f"检测结果_{task_dir.name}.txt",
        "html": task_dir / f"检测结果_{task_dir.name}.html",
    }
    for label, path in reports.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise AssertionError(f"报告缺失或为空：{label} {path}")
    output["reports"] = {key: str(value.resolve()) for key, value in reports.items()}
    output["task_count_scanned"] = sum(len(items) for items in by_selection.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
