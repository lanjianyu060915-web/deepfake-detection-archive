from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Collection, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from progress_events import AUTO_PERCENT, ProgressEventWriter
from reporting.report_builder import generate_task_reports


PRODUCT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path(
    os.environ.get("FSTS_CONFIG_PATH", str(PRODUCT_ROOT / "config.json"))
).expanduser().resolve(strict=False)
ProgressCallback = Callable[[dict[str, Any]], None]
MODULE_ORDER = ("deepfake", "identity", "source")
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


class RuntimeServices(Protocol):
    """Optional persistent-worker interface accepted by :func:`run_task`.

    A service manager may implement either method independently. Missing
    methods or infrastructure errors transparently use the subprocess adapter.
    """

    def run_deepfake(
        self,
        query_path: str | Path,
        output_path: str | Path,
        timeout: float,
    ) -> dict[str, Any]: ...

    def run_identity(
        self,
        query_path: str | Path,
        output_path: str | Path,
        library_path: str | Path,
        top_k: int,
        timeout: float,
        task_id: str | None = None,
        progress_events_path: str | Path | None = None,
        progress_latest_path: str | Path | None = None,
    ) -> dict[str, Any]: ...


class OrchestrationError(RuntimeError):
    """A controlled error raised before a task can be executed."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, value: str | bytes | None) -> None:
    if value is None:
        text = ""
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class _ProgressReporter:
    def __init__(
        self,
        callback: ProgressCallback | None,
        logs_dir: Path,
        task_id: str,
    ) -> None:
        self._callback = callback or (lambda _event: None)
        self.task_id = task_id
        self.events_path = logs_dir / "progress_events.jsonl"
        self.latest_path = logs_dir / "progress_latest.json"
        self._legacy_path = logs_dir / "progress.json"
        self._writers = {
            module: ProgressEventWriter(
                task_id,
                module,
                self.events_path,
                self.latest_path,
                min_interval_seconds=0.5,
                min_percent_delta=5.0,
            )
            for module in (*MODULE_ORDER, "overall")
        }
        self.events: list[dict[str, Any]] = []

    def emit(
        self,
        stage: str,
        message: str,
        module: str | None = None,
        *,
        status: str = "running",
        progress_current: int | None = None,
        progress_total: int | None = None,
        progress_percent: Any = AUTO_PERCENT,
        details: Mapping[str, Any] | None = None,
        force: bool = True,
    ) -> dict[str, Any] | None:
        module_name = module or "overall"
        writer = self._writers[module_name]
        event = writer.emit(
            stage,
            status,
            message,
            progress_current=progress_current,
            progress_total=progress_total,
            progress_percent=progress_percent,
            details=details,
            force=force,
        )
        if event is None:
            return None
        self.events.append(event)
        # Retain the v0.2 array for archived-task compatibility while the JSONL
        # file is the authoritative append-only event stream.
        _write_json(self._legacy_path, self.events)
        try:
            self._callback(dict(event))
        except Exception as exc:
            # A display callback is not allowed to terminate forensic work.
            event["callback_error"] = f"{type(exc).__name__}: {exc}"
            _write_json(self._legacy_path, self.events)
        return event


def load_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    path = Path(config_path or DEFAULT_CONFIG).expanduser().resolve(strict=False)
    if not path.is_file():
        raise OrchestrationError(f"配置文件不存在：{path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OrchestrationError(f"无法读取配置文件：{path}；{exc}") from exc
    if not isinstance(config, dict):
        raise OrchestrationError("配置文件顶层必须是 JSON 对象。")
    for section in ("runtime", "deepfake", "identity", "source", "frontend"):
        if not isinstance(config.get(section), dict):
            raise OrchestrationError(f"配置缺少对象：{section}")
    return config, path


def resolve_path(value: str | Path, config_path: Path) -> Path:
    raw_value = os.path.expandvars(str(value))
    configured_data_root = os.environ.get("FSTS_DATA_ROOT")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if configured_data_root:
        data_root = Path(configured_data_root).expanduser()
    elif local_app_data:
        data_root = Path(local_app_data) / "FaceSourceTrace"
    else:
        data_root = Path.home() / "AppData" / "Local" / "FaceSourceTrace"
    replacements = {
        "{APP_ROOT}": str(config_path.parent),
        "{DATA_ROOT}": str(data_root),
        "{CURRENT_PYTHON}": sys.executable,
    }
    for token, replacement in replacements.items():
        raw_value = raw_value.replace(token, replacement)
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve(strict=False)


def _new_task_dir(tasks_root: Path) -> tuple[str, Path]:
    tasks_root.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        now = datetime.now().astimezone()
        task_id = f"case_{now:%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"
        task_dir = tasks_root / task_id
        try:
            task_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return task_id, task_dir
    raise OrchestrationError("无法生成唯一 task_id。")


def _standard_search_result(
    query_path: Path,
    runtime: float,
    message: str | None,
    *,
    status: str = "error",
) -> dict[str, Any]:
    return {
        "status": status,
        "query_image": str(query_path),
        "candidates": [],
        "runtime_seconds": max(0.0, float(runtime)),
        "error_message": (
            (message or "未知模块错误").strip() if status == "error" else None
        ),
        **(
            {"details": {"reason": message or "未选择此模块"}}
            if status == "skipped"
            else {}
        ),
    }


def _standard_deepfake_result(
    query_path: Path,
    runtime: float,
    message: str | None,
    *,
    status: str = "error",
) -> dict[str, Any]:
    return {
        "status": status,
        "query_image": str(query_path),
        "is_deepfake": None,
        "score": None,
        "label": "skipped" if status == "skipped" else None,
        "runtime_seconds": max(0.0, float(runtime)),
        "details": ({"reason": message or "未选择此模块"} if status == "skipped" else {}),
        "error_message": (
            (message or "未知鉴伪模块错误").strip() if status == "error" else None
        ),
    }


def _validate_search_result(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "结果 JSON 顶层不是对象"
    required = {"status", "query_image", "candidates", "runtime_seconds", "error_message"}
    missing = sorted(required - payload.keys())
    if missing:
        return False, f"结果 JSON 缺少字段：{', '.join(missing)}"
    if payload["status"] not in {"success", "error", "skipped"}:
        return False, f"未知结果状态：{payload['status']!r}"
    if not isinstance(payload["candidates"], list):
        return False, "结果 candidates 不是数组"
    runtime = payload["runtime_seconds"]
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(float(runtime))
        or float(runtime) < 0
    ):
        return False, "结果 runtime_seconds 必须为非负有限数值"
    if payload["status"] == "success" and payload["error_message"] is not None:
        return False, "成功结果的 error_message 必须为 null"
    if payload["status"] == "error" and not str(payload["error_message"] or "").strip():
        return False, "失败结果必须包含非空 error_message"
    return True, ""


def _validate_deepfake_result(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "鉴伪结果 JSON 顶层不是对象"
    required = {
        "status",
        "query_image",
        "is_deepfake",
        "score",
        "label",
        "runtime_seconds",
        "details",
        "error_message",
    }
    missing = sorted(required - payload.keys())
    if missing:
        return False, f"鉴伪结果 JSON 缺少字段：{', '.join(missing)}"
    if payload["status"] not in {"success", "error", "skipped"}:
        return False, f"未知鉴伪结果状态：{payload['status']!r}"
    if payload["is_deepfake"] is not None and not isinstance(payload["is_deepfake"], bool):
        return False, "鉴伪结果 is_deepfake 必须为布尔值或 null"
    score = payload["score"]
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return False, "鉴伪结果 score 必须为数值或 null"
        if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            return False, "鉴伪结果 score 必须是 0 到 1 之间的有限数值"
    if payload["label"] is not None and not isinstance(payload["label"], str):
        return False, "鉴伪结果 label 必须为字符串或 null"
    if not isinstance(payload["details"], dict):
        return False, "鉴伪结果 details 必须为对象"
    runtime = payload["runtime_seconds"]
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(float(runtime))
        or float(runtime) < 0
    ):
        return False, "鉴伪结果 runtime_seconds 必须为非负有限数值"
    if payload["status"] == "success" and payload["error_message"] is not None:
        return False, "鉴伪成功结果的 error_message 必须为 null"
    if payload["status"] == "error" and not str(payload["error_message"] or "").strip():
        return False, "鉴伪失败结果必须包含非空 error_message"
    return True, ""


def _service_method(services: Any, name: str) -> Callable[..., Any] | None:
    if services is None:
        return None
    candidate = services.get(name) if isinstance(services, Mapping) else getattr(services, name, None)
    return candidate if callable(candidate) else None


def _invoke_service(
    method: Callable[..., Any],
    *,
    kwargs: dict[str, Any],
    timeout: float,
) -> Any:
    """Bound an in-process service call without making its model non-persistent."""

    outcomes: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            outcomes.put((True, method(**kwargs)))
        except BaseException as exc:  # transported to the orchestration thread
            outcomes.put((False, exc))

    thread = threading.Thread(target=target, name="fsts-service-call", daemon=True)
    thread.start()
    try:
        # ``timeout`` is the caller's remaining module budget, not a fresh
        # per-attempt allowance.  The outer guard must not extend that budget.
        ok, value = outcomes.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError(f"持久服务调用超时（{timeout:g} 秒）") from exc
    if ok:
        return value
    if not isinstance(value, Exception):
        raise RuntimeError(f"持久服务异常退出：{type(value).__name__}") from value
    raise value


def _remaining_timeout(deadline: float, *, module: str) -> float:
    """Return the positive remainder of one module-wide monotonic deadline."""

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{module} 已耗尽模块总超时预算")
    return remaining


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    hide_console: bool,
    executable_worker: bool,
    extra_environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    child_environment = os.environ.copy()
    if extra_environment:
        child_environment.update(extra_environment)
    restore_dll_directory: str | None = None
    if executable_worker:
        # Frozen workers must not inherit the front-end bundle's ABI/DLL state.
        child_environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        if os.name == "nt" and getattr(sys, "frozen", False):
            import ctypes

            restore_dll_directory = str(getattr(sys, "_MEIPASS", "")) or None
            ctypes.windll.kernel32.SetDllDirectoryW(None)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=child_environment,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if os.name == "nt" and hide_console
                else 0
            ),
        )
    finally:
        if restore_dll_directory is not None:
            import ctypes

            ctypes.windll.kernel32.SetDllDirectoryW(restore_dll_directory)


def _finish_module(
    *,
    name: str,
    payload: dict[str, Any],
    output_path: Path,
    logs_dir: Path,
    started_at: str,
    started: float,
    returncode: int | None,
    timed_out: bool,
    stdout: str = "",
    stderr: str = "",
    command_record: dict[str, Any] | None = None,
    execution_mode: str,
    fallback_reason: str | None = None,
    skipped: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = max(0.0, time.perf_counter() - started)
    status = str(payload.get("status", "error"))
    payload.setdefault("schema_version", "3.0")
    payload.setdefault("module", name)
    payload.setdefault("method", {"name": f"{name} module"})
    payload.setdefault("metrics", {})
    payload.setdefault("explanation", {})
    payload.setdefault("warnings", [])
    payload.setdefault(
        "limitations",
        ["本模块仅提供风险筛查或当前库内候选线索，不替代人工复核或司法鉴定。"],
    )
    runtime_details = payload.get("runtime")
    if not isinstance(runtime_details, dict):
        runtime_details = {}
    runtime_details.setdefault("module_elapsed_seconds", payload.get("runtime_seconds"))
    runtime_details["orchestrator_elapsed_seconds"] = runtime
    payload["runtime"] = runtime_details
    payload.setdefault("provenance", {})
    _write_json(output_path, payload)
    stdout_path = logs_dir / f"{name}_stdout.log"
    stderr_path = logs_dir / f"{name}_stderr.log"
    command_path = logs_dir / f"{name}_command.json"
    module_log_path = logs_dir / f"{name}.log"
    _write_text(stdout_path, stdout)
    _write_text(stderr_path, stderr)
    _write_json(
        command_path,
        command_record
        or {"module": name, "mode": execution_mode, "argv": [], "output": str(output_path)},
    )
    metadata = {
        "module": name,
        "status": status,
        "started_at": started_at,
        "finished_at": _now_iso(),
        "runtime_seconds": runtime,
        "returncode": returncode,
        "timed_out": timed_out,
        "skipped": skipped or status == "skipped",
        "execution_mode": execution_mode,
        "fallback_reason": fallback_reason,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command_log": str(command_path),
        "error": payload.get("error_message") if status == "error" else None,
    }
    _write_json(module_log_path.with_suffix(".json"), metadata)
    _write_text(
        module_log_path,
        "\n".join(
            [
                f"module={name}",
                f"status={status}",
                f"started_at={started_at}",
                f"finished_at={metadata['finished_at']}",
                f"runtime_seconds={runtime:.6f}",
                f"returncode={returncode}",
                f"timed_out={str(timed_out).lower()}",
                f"skipped={str(metadata['skipped']).lower()}",
                f"execution_mode={execution_mode}",
                f"fallback_reason={fallback_reason or ''}",
                f"error={metadata['error'] or ''}",
            ]
        )
        + "\n",
    )
    return payload, metadata


def _skip_module(
    *,
    name: str,
    query_path: Path,
    output_path: Path,
    logs_dir: Path,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = _now_iso()
    started = time.perf_counter()
    payload = (
        _standard_deepfake_result(query_path, 0.0, reason, status="skipped")
        if name == "deepfake"
        else _standard_search_result(query_path, 0.0, reason, status="skipped")
    )
    return _finish_module(
        name=name,
        payload=payload,
        output_path=output_path,
        logs_dir=logs_dir,
        started_at=started_at,
        started=started,
        returncode=None,
        timed_out=False,
        command_record={
            "module": name,
            "mode": "skipped",
            "argv": [],
            "reason": reason,
            "output": str(output_path),
        },
        execution_mode="skipped",
        skipped=True,
    )


def _library_has_images(path: Path) -> bool:
    try:
        return any(
            item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
            for item in path.rglob("*")
        )
    except OSError:
        return False


def _library_image_count(path: Path) -> int:
    try:
        return sum(
            1
            for item in path.rglob("*")
            if item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
        )
    except OSError:
        return 0


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _task_artifact_manifest(
    task_dir: Path,
    task_id: str,
    results: Mapping[str, Mapping[str, Any]],
    *,
    report_files: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate only task-contained, actually written evidence artifacts.

    Module manifests are intentionally advisory.  This boundary recomputes
    file size and SHA-256, rejects absolute/traversal paths, and never lets an
    optional visualization failure invalidate the original module result.
    """

    root = task_dir.resolve(strict=False)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append(relative_value: Any, module: str, kind: str, description: str = "") -> None:
        if not isinstance(relative_value, str) or not relative_value.strip():
            return
        portable = relative_value.strip().replace("\\", "/")
        relative = Path(portable)
        if relative.is_absolute() or ".." in relative.parts:
            return
        candidate = (root / relative).resolve(strict=False)
        try:
            normalized = candidate.relative_to(root).as_posix()
        except ValueError:
            return
        if normalized in seen or not candidate.is_file():
            return
        seen.add(normalized)
        records.append(
            {
                "id": f"{module}-{len(records) + 1:03d}",
                "module": module,
                "kind": kind,
                "path": normalized,
                "status": "present",
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256_path(candidate),
                "description": description,
            }
        )

    append("query.jpg", "overall", "query_image", "本次任务固化的查询图")
    result_files = {
        "deepfake": "detection_result.json",
        "identity": "identity_result.json",
        "source": "source_result.json",
    }
    for module in MODULE_ORDER:
        result = results.get(module, {})
        manifest = result.get("artifact_manifest")
        if manifest is None and isinstance(result.get("visual_evidence"), Mapping):
            manifest = result["visual_evidence"].get("artifact_manifest")
        if isinstance(manifest, list):
            for item in manifest:
                if not isinstance(item, Mapping):
                    continue
                append(
                    item.get("path") or item.get("relative_path"),
                    module,
                    str(item.get("kind") or item.get("type") or "visual_evidence"),
                    str(item.get("description") or ""),
                )
        append(result_files[module], module, "module_result_json", "模块结构化结果")
    for format_name, file_name in (report_files or {}).items():
        append(file_name, "overall", f"report_{format_name}", "统一任务报告")
    return {
        "schema_version": "1.0",
        "task_id": task_id,
        "generated_at": _now_iso(),
        "path_policy": "task_relative_posix",
        "artifacts": records,
        "artifact_count": len(records),
    }


def _validate_deepfake_hashes(
    section: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    configured = section.get("known_sha256")
    expected = dict(configured) if isinstance(configured, Mapping) else {}
    if not any(str(value or "").strip() for value in expected.values()):
        return {
            "ok": True,
            "validation_status": "not_configured",
            "actual_sha256": {},
            "missing": [],
            "mismatches": [],
            "checked_resource_count": 0,
            "warning": "当前外部配置没有提供已知SHA-256；正式配置必须提供。",
        }
    install_root = resolve_path(section["install_root"], config_path)
    runner_value = section.get("runner")
    runner = resolve_path(runner_value, config_path) if runner_value else None
    paths: dict[str, Path] = {
        "executable": install_root / "深度伪造识别系统.exe",
        "spatial_model": install_root
        / "models"
        / "spatial_mobilenet_finetuned_gpu"
        / "spatial_mobilenet_finetuned_gpu.pt",
        "frequency_model": install_root
        / "models"
        / "frequency_stat_formal"
        / "frequency_stat_detector.joblib",
        "fusion_config": install_root
        / "models"
        / "fusion_formal"
        / "fusion_config.json",
    }
    if runner is not None:
        paths.update(
            {
                "adapter": runner,
                "adapter_runtime": runner.parent / "pyinstaller_pyz_runtime.py",
                "adapter_inspector": runner.parent / "inspect_installed_archive.py",
            }
        )
    actual: dict[str, str] = {}
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []
    for key, path in paths.items():
        if not path.is_file():
            missing.append(key)
            continue
        digest = _sha256_path(path)
        actual[key] = digest
        expected_digest = str(expected.get(key) or "").strip().lower()
        if expected_digest and digest.lower() != expected_digest:
            mismatches.append(
                {"resource": key, "expected": expected_digest, "actual": digest}
            )
    return {
        "ok": not missing and not mismatches,
        "actual_sha256": actual,
        "missing": missing,
        "mismatches": mismatches,
        "checked_resource_count": len(paths),
    }


def _emit_deepfake_result_progress(
    progress: _ProgressReporter, payload: Mapping[str, Any]
) -> None:
    details = payload.get("details")
    details = details if isinstance(details, Mapping) else {}
    if payload.get("status") != "success":
        progress.emit(
            "deepfake_result_error",
            "AIGC真实模块未能生成可用结论。",
            "deepfake",
            status="error",
            progress_percent=None,
            details={"error": payload.get("error_message")},
        )
        return
    valid_count = details.get("valid_frame_count")
    insufficient_face_evidence = (
        valid_count == 0
        or str(payload.get("label") or "").strip().lower()
        in {"insufficient_evidence", "insufficient", "no_face"}
    )
    progress.emit(
        "face_and_crop_completed",
        f"有效人脸/有效帧处理完成：{valid_count if valid_count is not None else '未返回数量'}。",
        "deepfake",
        status="warning" if insufficient_face_evidence else "running",
        progress_percent=None,
        details={
            "valid_frame_count": valid_count,
            "invalid_frame_count": details.get("invalid_frame_count"),
        },
    )
    if insufficient_face_evidence:
        progress.emit(
            "deepfake_inference_skipped",
            "未获得有效人脸证据，空间域与频域推理未形成可用分数。",
            "deepfake",
            status="warning",
            progress_percent=None,
            details={
                "valid_frame_count": valid_count,
                "label": payload.get("label"),
                "warnings": payload.get("warnings") or details.get("warnings") or [],
            },
        )
        return
    progress.emit(
        "spatial_inference_completed",
        "空间域模型推理完成。",
        "deepfake",
        status="running",
        progress_percent=None,
        details={"spatial_score": details.get("spatial_score")},
    )
    progress.emit(
        "frequency_inference_completed",
        "频域模型推理完成。",
        "deepfake",
        status="running",
        progress_percent=None,
        details={"frequency_score": details.get("frequency_score")},
    )
    fusion = details.get("fusion_score")
    threshold = details.get("decision_threshold")
    margin = (
        float(fusion) - float(threshold)
        if isinstance(fusion, (int, float)) and isinstance(threshold, (int, float))
        else None
    )
    progress.emit(
        "fusion_and_threshold_completed",
        "已按真实权重计算融合分数并与判定阈值比较。",
        "deepfake",
        status="running",
        progress_percent=None,
        details={
            "spatial_weight": details.get("spatial_weight"),
            "frequency_weight": details.get("frequency_weight"),
            "fusion_score": fusion,
            "decision_threshold": threshold,
            "decision_margin": margin,
        },
    )


def _run_deepfake_module(
    *,
    section: dict[str, Any],
    config_path: Path,
    query_path: Path,
    output_path: Path,
    logs_dir: Path,
    progress: _ProgressReporter,
    services: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    name = "deepfake"
    started_at = _now_iso()
    started = time.perf_counter()
    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        message = "deepfake 配置无效：enabled 必须为布尔值"
        payload = _standard_deepfake_result(query_path, 0.0, message)
        return _finish_module(
            name=name,
            payload=payload,
            output_path=output_path,
            logs_dir=logs_dir,
            started_at=started_at,
            started=started,
            returncode=None,
            timed_out=False,
            stderr=message + "\n",
            execution_mode="configuration",
        )
    if not enabled:
        message = "鉴伪模块已由配置关闭，无法执行用户选择的模块"
        payload = _standard_deepfake_result(query_path, 0.0, message)
        return _finish_module(
            name=name,
            payload=payload,
            output_path=output_path,
            logs_dir=logs_dir,
            started_at=started_at,
            started=started,
            returncode=None,
            timed_out=False,
            stderr=message + "\n",
            command_record={
                "module": name,
                "mode": "disabled",
                "argv": [],
                "reason": message,
                "output": str(output_path),
            },
            execution_mode="disabled",
        )

    try:
        timeout = float(section.get("timeout_seconds", 180))
    except (TypeError, ValueError) as exc:
        timeout = 0.0
        config_error = f"deepfake 配置无效：{exc}"
    else:
        config_error = (
            ""
            if math.isfinite(timeout) and timeout > 0
            else f"timeout_seconds 必须是大于 0 的有限数值：{timeout}"
        )
    if config_error:
        payload = _standard_deepfake_result(query_path, 0.0, config_error)
        return _finish_module(
            name=name,
            payload=payload,
            output_path=output_path,
            logs_dir=logs_dir,
            started_at=started_at,
            started=started,
            returncode=None,
            timed_out=False,
            stderr=config_error + "\n",
            execution_mode="configuration",
        )

    deadline = time.monotonic() + timeout
    progress.emit(
        "input_validated",
        "输入图片存在且已复制到独立任务目录。",
        name,
        status="running",
        progress_percent=None,
        details={"file_name": query_path.name, "size_bytes": query_path.stat().st_size},
    )
    try:
        hash_validation = _validate_deepfake_hashes(section, config_path)
    except (KeyError, OSError, ValueError) as exc:
        hash_validation = {
            "ok": False,
            "actual_sha256": {},
            "missing": [],
            "mismatches": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not hash_validation.get("ok"):
        message = "AIGC模型或配置哈希校验未通过"
        details = {
            "hash_validation": hash_validation,
            "reason": message,
        }
        progress.emit(
            "model_hash_validation",
            message,
            name,
            status="error",
            progress_percent=None,
            details=details,
        )
        payload = _standard_deepfake_result(query_path, time.perf_counter() - started, message)
        payload["details"] = details
        payload["provenance"] = {"hash_validation": hash_validation}
        return _finish_module(
            name=name,
            payload=payload,
            output_path=output_path,
            logs_dir=logs_dir,
            started_at=started_at,
            started=started,
            returncode=None,
            timed_out=False,
            stderr=message + "\n",
            execution_mode="preflight",
        )
    progress.emit(
        "model_hash_validation",
        (
            "AIGC模型、融合配置和适配器哈希校验通过。"
            if hash_validation.get("validation_status") != "not_configured"
            else "当前外部配置未提供已知哈希；由所连接的隔离服务负责资源校验。"
        ),
        name,
        status=(
            "running"
            if hash_validation.get("validation_status") != "not_configured"
            else "warning"
        ),
        progress_percent=None,
        details=hash_validation,
    )
    service_error: str | None = None
    method = _service_method(services, "run_deepfake")
    if method is not None:
        # Keep the temporary JSON under the task root so the frozen adapter's
        # native visualizations still land in task_dir/visualizations/deepfake.
        # The distinct filename prevents a late service response racing the
        # canonical subprocess result.
        service_output_path = output_path.with_name(".deepfake_service_result.json")
        try:
            progress.emit(
                "deepfake_service_reuse",
                "正在启动或复用 AIGC 真实模型服务。",
                name,
                status="running",
                progress_percent=None,
            )
            progress.emit(
                "deepfake_inference_running",
                "正在读取图片、检测有效人脸并执行空间域与频域模型。",
                name,
                status="running",
                progress_percent=None,
            )
            service_timeout = _remaining_timeout(deadline, module=name)
            payload = _invoke_service(
                method,
                kwargs={
                    "query_path": query_path,
                    "output_path": service_output_path,
                    "timeout": service_timeout,
                },
                timeout=service_timeout,
            )
            valid, validation_error = _validate_deepfake_result(payload)
            if not valid:
                raise ValueError(validation_error)
            _emit_deepfake_result_progress(progress, payload)
            return _finish_module(
                name=name,
                payload=payload,
                output_path=output_path,
                logs_dir=logs_dir,
                started_at=started_at,
                started=started,
                returncode=None,
                timed_out=False,
                command_record={
                    "module": name,
                    "mode": "service",
                    "method": "run_deepfake",
                    "timeout_seconds": timeout,
                    "attempt_timeout_seconds": service_timeout,
                    "service_output": str(service_output_path),
                    "output": str(output_path),
                },
                execution_mode="service",
            )
        except Exception as exc:
            service_error = f"{type(exc).__name__}: {exc}"
            progress.emit(
                "deepfake_service_fallback",
                "AIGC常驻服务不可用，切换隔离子进程……",
                name,
                status="warning",
                progress_percent=None,
                details={"fallback_reason": service_error},
            )

    errors: list[str] = []
    command: list[str] = []
    python_path: Path | None = None
    runner_path: Path | None = None
    install_root: Path | None = None
    device = "auto"
    executable_worker = False
    try:
        executable_value = section.get("executable")
        if executable_value:
            executable_path = resolve_path(executable_value, config_path)
            command_prefix = [str(executable_path)]
            runner_path = executable_path
            executable_worker = True
        else:
            python_path = resolve_path(section["python"], config_path)
            runner_path = resolve_path(section["runner"], config_path)
            python_args = section.get("python_args", ["-S", "-B"])
            if not isinstance(python_args, list) or not all(isinstance(item, str) for item in python_args):
                raise ValueError("python_args 必须为字符串数组")
            command_prefix = [str(python_path), *python_args, str(runner_path)]
        install_root = resolve_path(section["install_root"], config_path)
        device = str(section.get("device", "auto"))
        if device not in {"cpu", "auto", "cuda"}:
            raise ValueError(f"未知 device：{device}")
        extra_args = section.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
            raise ValueError("extra_args 必须为字符串数组")
        command = [
            *command_prefix,
            "--query",
            str(query_path),
            "--output",
            str(output_path),
            "--install-root",
            str(install_root),
            "--device",
            device,
            *extra_args,
        ]
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"deepfake 配置无效：{exc}")

    if executable_worker:
        if runner_path is None or not runner_path.is_file():
            errors.append(f"模块可执行文件不存在：{runner_path}")
    else:
        if python_path is None or not python_path.is_file():
            errors.append(f"Python 不存在：{python_path}")
        if runner_path is None or not runner_path.is_file():
            errors.append(f"runner 不存在：{runner_path}")
    if install_root is None or not install_root.is_dir():
        errors.append(f"鉴伪安装目录不存在或不是目录：{install_root}")
    command_record = {
        "module": name,
        "mode": "subprocess",
        "argv": command,
        "cwd": str(runner_path.parent) if runner_path is not None else None,
        "install_root": str(install_root) if install_root is not None else None,
        "install_root_read_only": bool(section.get("install_root_read_only", True)),
        "official_base_weight_required": False,
        "output": str(output_path),
        "service_fallback_reason": service_error,
    }
    stdout = ""
    stderr = ""
    returncode: int | None = None
    timed_out = False
    execution_mode = "subprocess"
    if not errors and runner_path is not None:
        try:
            progress.emit(
                "deepfake_subprocess_running",
                "正在隔离子进程中执行真实空间域与频域模型。",
                name,
                status="running",
                progress_percent=None,
            )
            subprocess_timeout = _remaining_timeout(deadline, module=name)
            command_record["attempt_timeout_seconds"] = subprocess_timeout
            completed = _run_subprocess(
                command,
                cwd=runner_path.parent,
                timeout=subprocess_timeout,
                hide_console=bool(section.get("hide_console", False)),
                executable_worker=executable_worker,
                extra_environment={"PYTHONDONTWRITEBYTECODE": "1"},
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            errors.append(f"deepfake 超时（模块总预算 {timeout:g} 秒）")
        except TimeoutError as exc:
            timed_out = True
            execution_mode = "deadline_exhausted"
            command_record["attempted"] = False
            errors.append(f"deepfake 超时（模块总预算 {timeout:g} 秒）：{exc}")
        except OSError as exc:
            errors.append(f"无法启动 deepfake 子进程：{exc}")
            stderr = (stderr + "\n" + str(exc)).strip() + "\n"
    elif errors:
        stderr = "\n".join(errors) + "\n"

    payload: Any = None
    if output_path.is_file():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            valid, validation_error = _validate_deepfake_result(payload)
            if not valid:
                errors.append(validation_error)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"无法解析 deepfake 结果 JSON：{exc}")
    else:
        errors.append("deepfake 未生成结果文件")
    if returncode not in (0, None):
        errors.append(f"deepfake 子进程退出码为 {returncode}")
    if isinstance(payload, dict) and payload.get("status") == "error":
        errors.append(str(payload.get("error_message") or "deepfake 返回 error"))
    if errors:
        if output_path.is_file():
            try:
                shutil.copy2(output_path, logs_dir / "deepfake_raw_result.json")
            except OSError:
                pass
        if service_error:
            errors.insert(0, f"持久服务回退原因：{service_error}")
        message = "; ".join(dict.fromkeys(item.strip() for item in errors if item.strip()))
        payload = _standard_deepfake_result(query_path, time.perf_counter() - started, message)
    if isinstance(payload, Mapping):
        _emit_deepfake_result_progress(progress, payload)
    return _finish_module(
        name=name,
        payload=payload,
        output_path=output_path,
        logs_dir=logs_dir,
        started_at=started_at,
        started=started,
        returncode=returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        command_record=command_record,
        execution_mode=execution_mode,
        fallback_reason=service_error,
    )


def _run_search_module(
    *,
    name: str,
    section: dict[str, Any],
    config_path: Path,
    query_path: Path,
    library_override: str | Path | None,
    output_path: Path,
    logs_dir: Path,
    identity_cache: Path,
    progress: _ProgressReporter,
    services: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = _now_iso()
    started = time.perf_counter()
    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        message = f"{name} 配置无效：enabled 必须为布尔值"
        payload = _standard_search_result(query_path, 0.0, message)
        return _finish_module(
            name=name,
            payload=payload,
            output_path=output_path,
            logs_dir=logs_dir,
            started_at=started_at,
            started=started,
            returncode=None,
            timed_out=False,
            stderr=message + "\n",
            execution_mode="configuration",
        )
    if not enabled:
        message = f"{name} 模块已由配置关闭，无法执行用户选择的模块"
        payload = _standard_search_result(query_path, 0.0, message)
        return _finish_module(
            name=name,
            payload=payload,
            output_path=output_path,
            logs_dir=logs_dir,
            started_at=started_at,
            started=started,
            returncode=None,
            timed_out=False,
            stderr=message + "\n",
            command_record={
                "module": name,
                "mode": "disabled",
                "argv": [],
                "reason": message,
                "output": str(output_path),
            },
            execution_mode="disabled",
        )
    try:
        library_path = resolve_path(library_override or section["library"], config_path)
        timeout = float(section.get("timeout_seconds", 600))
        top_k = int(section.get("top_k", 5))
    except (KeyError, TypeError, ValueError) as exc:
        message = f"{name} 配置无效：{exc}"
        payload = _standard_search_result(query_path, 0.0, message)
        return _finish_module(
            name=name,
            payload=payload,
            output_path=output_path,
            logs_dir=logs_dir,
            started_at=started_at,
            started=started,
            returncode=None,
            timed_out=False,
            stderr=message + "\n",
            execution_mode="configuration",
        )

    timeout_valid = math.isfinite(timeout) and timeout > 0
    deadline = time.monotonic() + timeout if timeout_valid else None
    preflight_errors: list[str] = []
    library_image_count = 0
    if not library_path.is_dir():
        preflight_errors.append(f"图库不存在或不是目录：{library_path}")
    else:
        library_image_count = _library_image_count(library_path)
    if library_path.is_dir() and library_image_count == 0:
        preflight_errors.append(f"图库为空或没有受支持图片：{library_path}")
    if top_k < 1:
        preflight_errors.append(f"top_k 必须大于 0：{top_k}")
    if not timeout_valid:
        preflight_errors.append(
            f"timeout_seconds 必须是大于 0 的有限数值：{timeout}"
        )
    if preflight_errors:
        message = "; ".join(preflight_errors)
        payload = _standard_search_result(query_path, time.perf_counter() - started, message)
        return _finish_module(
            name=name,
            payload=payload,
            output_path=output_path,
            logs_dir=logs_dir,
            started_at=started_at,
            started=started,
            returncode=None,
            timed_out=False,
            stderr=message + "\n",
            command_record={
                "module": name,
                "mode": "preflight",
                "library": str(library_path),
                "output": str(output_path),
            },
            execution_mode="preflight",
        )

    assert deadline is not None
    identity_count = (
        sum(1 for item in library_path.iterdir() if item.is_dir())
        if name == "identity"
        else None
    )
    progress.emit(
        "library_checked",
        (
            f"身份库检查完成：{identity_count} 人、{library_image_count} 张参考图片。"
            if name == "identity"
            else f"底图库检查完成：发现 {library_image_count} 张支持格式图片。"
        ),
        name,
        status="running",
        progress_percent=None,
        details={
            "library_image_count": library_image_count,
            "library_identity_count": identity_count,
            "top_k": top_k,
        },
    )

    service_error: str | None = None
    method = _service_method(services, "run_identity") if name == "identity" else None
    if method is not None:
        service_output_path = output_path.with_name(".identity_service_result.json")
        try:
            progress.emit(
                "identity_service_reuse",
                "正在启动或复用 InsightFace 身份服务。",
                name,
                status="running",
                progress_percent=None,
            )
            service_timeout = _remaining_timeout(deadline, module=name)
            payload = _invoke_service(
                method,
                kwargs={
                    "query_path": query_path,
                    "output_path": service_output_path,
                    "library_path": library_path,
                    "top_k": top_k,
                    "timeout": service_timeout,
                    "task_id": progress.task_id,
                    "progress_events_path": progress.events_path,
                    "progress_latest_path": progress.latest_path,
                },
                timeout=service_timeout,
            )
            valid, validation_error = _validate_search_result(payload)
            if not valid:
                raise ValueError(validation_error)
            return _finish_module(
                name=name,
                payload=payload,
                output_path=output_path,
                logs_dir=logs_dir,
                started_at=started_at,
                started=started,
                returncode=None,
                timed_out=False,
                command_record={
                    "module": name,
                    "mode": "service",
                    "method": "run_identity",
                    "library": str(library_path),
                    "top_k": top_k,
                    "timeout_seconds": timeout,
                    "attempt_timeout_seconds": service_timeout,
                    "service_output": str(service_output_path),
                    "output": str(output_path),
                },
                execution_mode="service",
            )
        except Exception as exc:
            service_error = f"{type(exc).__name__}: {exc}"
            progress.emit(
                "identity_service_fallback",
                "身份常驻服务不可用，切换隔离子进程……",
                name,
                status="warning",
                progress_percent=None,
                details={"fallback_reason": service_error},
            )

    errors: list[str] = []
    executable_worker = False
    executable_path: Path | None = None
    python_path: Path | None = None
    runner_path: Path | None = None
    command: list[str] = []
    module_cwd = config_path.parent
    try:
        executable_value = section.get("executable")
        if executable_value:
            executable_path = resolve_path(executable_value, config_path)
            command_prefix = [str(executable_path)]
            module_cwd = executable_path.parent
            executable_worker = True
        else:
            python_path = resolve_path(section["python"], config_path)
            runner_path = resolve_path(section["runner"], config_path)
            python_args = section.get("python_args", ["-B"])
            if not isinstance(python_args, list) or not all(isinstance(item, str) for item in python_args):
                raise ValueError("python_args 必须为字符串数组")
            command_prefix = [str(python_path), *python_args, str(runner_path)]
            module_cwd = runner_path.parent
        command = [
            *command_prefix,
            "--query",
            str(query_path),
            "--library",
            str(library_path),
            "--top-k",
            str(top_k),
            "--output",
            str(output_path),
            "--task-id",
            progress.task_id,
            "--progress-events",
            str(progress.events_path),
            "--progress-latest",
            str(progress.latest_path),
        ]
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{name} 配置无效：{exc}")

    if executable_worker:
        if executable_path is None or not executable_path.is_file():
            errors.append(f"模块可执行文件不存在：{executable_path}")
    else:
        if python_path is None or not python_path.is_file():
            errors.append(f"Python 不存在：{python_path}")
        if runner_path is None or not runner_path.is_file():
            errors.append(f"runner 不存在：{runner_path}")
    if name == "identity":
        try:
            identity_cache.mkdir(parents=True, exist_ok=True)
            command.extend(["--ctx", str(section.get("ctx", -1))])
            command.extend(["--cache-dir", str(identity_cache)])
            if section.get("model_root"):
                command.extend(["--model-root", str(resolve_path(section["model_root"], config_path))])
            if section.get("offline", True):
                command.append("--offline")
        except OSError as exc:
            errors.append(f"无法创建身份缓存目录：{exc}")

    command_record = {
        "module": name,
        "mode": "subprocess",
        "argv": command,
        "cwd": str(module_cwd),
        "library": str(library_path),
        "output": str(output_path),
        "service_fallback_reason": service_error,
    }
    returncode: int | None = None
    timed_out = False
    execution_mode = "subprocess"
    stdout = ""
    stderr = ""
    if not errors:
        try:
            subprocess_timeout = _remaining_timeout(deadline, module=name)
            command_record["attempt_timeout_seconds"] = subprocess_timeout
            completed = _run_subprocess(
                command,
                cwd=module_cwd,
                timeout=subprocess_timeout,
                hide_console=bool(section.get("hide_console", False)),
                executable_worker=executable_worker,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            errors.append(f"{name} 超时（模块总预算 {timeout:g} 秒）")
        except TimeoutError as exc:
            timed_out = True
            execution_mode = "deadline_exhausted"
            command_record["attempted"] = False
            errors.append(f"{name} 超时（模块总预算 {timeout:g} 秒）：{exc}")
        except OSError as exc:
            errors.append(f"无法启动 {name} 子进程：{exc}")
            stderr = (stderr + "\n" + str(exc)).strip() + "\n"
    else:
        stderr = "\n".join(errors) + "\n"

    payload: Any = None
    if output_path.is_file():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            valid, validation_error = _validate_search_result(payload)
            if not valid:
                errors.append(validation_error)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"无法解析 {name} 结果 JSON：{exc}")
    else:
        errors.append(f"{name} 未生成结果文件")
    if returncode not in (0, None):
        errors.append(f"{name} 子进程退出码为 {returncode}")
    if isinstance(payload, dict) and payload.get("status") == "error":
        errors.append(str(payload.get("error_message") or f"{name} 返回 error"))
    if errors:
        if output_path.is_file():
            try:
                shutil.copy2(output_path, logs_dir / f"{name}_raw_result.json")
            except OSError:
                pass
        if service_error:
            errors.insert(0, f"持久服务回退原因：{service_error}")
        message = "; ".join(dict.fromkeys(item.strip() for item in errors if item.strip()))
        payload = _standard_search_result(query_path, time.perf_counter() - started, message)
    return _finish_module(
        name=name,
        payload=payload,
        output_path=output_path,
        logs_dir=logs_dir,
        started_at=started_at,
        started=started,
        returncode=returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        command_record=command_record,
        execution_mode=execution_mode,
        fallback_reason=service_error,
    )


def _unexpected_module_failure(
    *,
    name: str,
    query_path: Path,
    output_path: Path,
    logs_dir: Path,
    exc: Exception,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = _now_iso()
    started = time.perf_counter()
    message = f"{name} 分支发生受控异常：{type(exc).__name__}: {exc}"
    payload = (
        _standard_deepfake_result(query_path, 0.0, message)
        if name == "deepfake"
        else _standard_search_result(query_path, 0.0, message)
    )
    return _finish_module(
        name=name,
        payload=payload,
        output_path=output_path,
        logs_dir=logs_dir,
        started_at=started_at,
        started=started,
        returncode=None,
        timed_out=False,
        stderr=message + "\n",
        command_record={"module": name, "mode": "orchestrator_exception", "argv": []},
        execution_mode="orchestrator_exception",
    )


def _normalize_selected_modules(selected_modules: Collection[str] | None) -> frozenset[str]:
    if selected_modules is None:
        return frozenset(MODULE_ORDER)
    if isinstance(selected_modules, str):
        values = {selected_modules}
    else:
        try:
            values = {str(value) for value in selected_modules}
        except TypeError as exc:
            raise OrchestrationError("selected_modules 必须是模块名集合。") from exc
    unknown = sorted(values - set(MODULE_ORDER))
    if unknown:
        raise OrchestrationError(
            "selected_modules 包含未知模块："
            + ", ".join(unknown)
            + "；可选值为 deepfake/identity/source。"
        )
    if not values:
        raise OrchestrationError("selected_modules 至少必须包含一个模块。")
    return frozenset(values)


def run_task(
    query: str | Path,
    *,
    identity_library: str | Path | None = None,
    source_library: str | Path | None = None,
    selected_modules: Collection[str] | None = None,
    services: RuntimeServices | Mapping[str, Callable[..., Any]] | Any | None = None,
    config_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    selected = _normalize_selected_modules(selected_modules)
    config, resolved_config = load_config(config_path)
    query_source = Path(query).expanduser().resolve(strict=False)
    if not query_source.is_file():
        raise OrchestrationError(f"查询图片不存在或不是文件：{query_source}")

    try:
        tasks_root = resolve_path(config["runtime"]["tasks_root"], resolved_config)
        identity_cache = resolve_path(config["runtime"]["identity_cache"], resolved_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise OrchestrationError(f"runtime 配置无效：{exc}") from exc
    task_id, task_dir = _new_task_dir(tasks_root)
    logs_dir = task_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    query_path = task_dir / "query.jpg"
    shutil.copy2(query_source, query_path)

    started_at = _now_iso()
    started = time.perf_counter()
    progress = _ProgressReporter(progress_callback, logs_dir, task_id)
    selected_order = [name for name in MODULE_ORDER if name in selected]
    overall_work_total = len(selected_order) + 1  # selected modules + report generation
    module_states: dict[str, dict[str, Any]] = {
        name: {
            "module": name,
            "status": "pending" if name in selected else "skipped",
            "stage": "pending" if name in selected else "not_selected",
            "message": "等待执行" if name in selected else "用户未选择此模块",
            "runtime_seconds": None,
            "error": None,
        }
        for name in MODULE_ORDER
    }

    def write_running_status(current_module: str | None = None) -> None:
        _write_json(
            task_dir / "task_status.json",
            {
                "schema_version": "3.0",
                "task_id": task_id,
                "task_dir": str(task_dir),
                "status": "running",
                "started_at": started_at,
                "query_source": str(query_source),
                "query_image": str(query_path),
                "selected_modules": selected_order,
                "current_module": current_module,
                "modules": module_states,
                "progress_events": str(progress.events_path),
                "progress_latest": str(progress.latest_path),
            },
        )

    write_running_status()
    progress.emit(
        "task_created",
        "已创建独立分析任务。",
        status="running",
        progress_current=0,
        progress_total=overall_work_total,
        details={
            "task_dir": str(task_dir),
            "selected_modules": selected_order,
            "query_file": query_path.name,
        },
    )

    output_paths = {
        "deepfake": task_dir / "detection_result.json",
        "identity": task_dir / "identity_result.json",
        "source": task_dir / "source_result.json",
    }
    results: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    selected_completed = 0
    for name in MODULE_ORDER:
        if name not in selected:
            results[name], metadata[name] = _skip_module(
                name=name,
                query_path=query_path,
                output_path=output_paths[name],
                logs_dir=logs_dir,
                reason="用户未选择此模块",
            )
            module_states[name].update(
                status="skipped",
                stage="not_selected",
                message="用户未选择此模块",
                runtime_seconds=metadata[name]["runtime_seconds"],
            )
            progress.emit(
                f"{name}_skipped",
                f"{name} 模块已跳过。",
                name,
                status="skipped",
                progress_current=1,
                progress_total=1,
            )
            continue
        module_states[name].update(
            status="running",
            stage="starting",
            message="正在进入真实模块",
        )
        write_running_status(name)
        progress.emit(
            f"{name}_started",
            f"开始执行 {name} 真实模块。",
            name,
            status="running",
            progress_current=0,
            progress_total=1,
            details={"execution_order": selected_order.index(name) + 1},
        )
        try:
            if name == "deepfake":
                results[name], metadata[name] = _run_deepfake_module(
                    section=config[name],
                    config_path=resolved_config,
                    query_path=query_path,
                    output_path=output_paths[name],
                    logs_dir=logs_dir,
                    progress=progress,
                    services=services,
                )
            else:
                results[name], metadata[name] = _run_search_module(
                    name=name,
                    section=config[name],
                    config_path=resolved_config,
                    query_path=query_path,
                    library_override=(identity_library if name == "identity" else source_library),
                    output_path=output_paths[name],
                    logs_dir=logs_dir,
                    identity_cache=identity_cache,
                    progress=progress,
                    services=services,
                )
        except Exception as exc:
            results[name], metadata[name] = _unexpected_module_failure(
                name=name,
                query_path=query_path,
                output_path=output_paths[name],
                logs_dir=logs_dir,
                exc=exc,
            )
        module_status = str(results[name].get("status") or "error")
        progress.emit(
            f"{name}_finished",
            f"{name} 模块完成：{module_status}",
            name,
            status=module_status if module_status in {"success", "error", "skipped"} else "warning",
            progress_current=1,
            progress_total=1,
            details={"runtime_seconds": metadata[name]["runtime_seconds"]},
        )
        module_states[name].update(
            status=module_status,
            stage="finished",
            message=f"模块完成：{module_status}",
            runtime_seconds=metadata[name]["runtime_seconds"],
            error=metadata[name].get("error"),
        )
        selected_completed += 1
        write_running_status(None if selected_completed == len(selected_order) else name)
        progress.emit(
            "modules_completed",
            f"已完成所选模块：{selected_completed} / {len(selected_order)}",
            status="running",
            progress_current=selected_completed,
            progress_total=overall_work_total,
            details={"last_completed_module": name},
        )

    selected_statuses = [results[name]["status"] for name in MODULE_ORDER if name in selected]
    successes = selected_statuses.count("success")
    non_successes = sum(status != "success" for status in selected_statuses)
    overall_status = (
        "success"
        if non_successes == 0
        else "partial_success"
        if successes > 0
        else "error"
    )
    total_runtime = time.perf_counter() - started
    finished_at = _now_iso()
    errors = [
        f"{name}: {results[name]['error_message']}"
        for name in MODULE_ORDER
        if name in selected and results[name].get("status") == "error"
    ]
    unified = {
        "schema_version": "3.0",
        "task_id": task_id,
        "status": overall_status,
        "query_image": "query.jpg",
        "selected_modules": [name for name in MODULE_ORDER if name in selected],
        "deepfake_detection": {
            "selected": "deepfake" in selected,
            "status": results["deepfake"]["status"],
            "result_file": "detection_result.json",
        },
        "identity_search": {
            "selected": "identity" in selected,
            "status": results["identity"]["status"],
            "result_file": "identity_result.json",
        },
        "source_search": {
            "selected": "source" in selected,
            "status": results["source"]["status"],
            "result_file": "source_result.json",
        },
        "runtime_seconds": {
            **{name: metadata[name]["runtime_seconds"] for name in MODULE_ORDER},
            "total": total_runtime,
        },
        "errors": errors,
    }
    task_status = {
        "schema_version": "3.0",
        "task_id": task_id,
        "task_dir": str(task_dir),
        "status": overall_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "query_source": str(query_source),
        "query_image": str(query_path),
        "selected_modules": [name for name in MODULE_ORDER if name in selected],
        "modules": module_states,
        "deepfake": metadata["deepfake"],
        "identity": metadata["identity"],
        "source": metadata["source"],
        "runtime_seconds": total_runtime,
        "progress_log": str(progress.events_path),
        "progress_events": str(progress.events_path),
        "progress_latest": str(progress.latest_path),
        "errors": errors,
    }

    artifact_manifest = _task_artifact_manifest(task_dir, task_id, results)
    _write_json(task_dir / "artifact_manifest.json", artifact_manifest)
    artifact_reference = {
        "file": "artifact_manifest.json",
        "artifact_count": artifact_manifest["artifact_count"],
        "path_policy": artifact_manifest["path_policy"],
    }
    unified["artifact_manifest"] = artifact_reference
    task_status["artifact_manifest"] = artifact_reference

    _write_json(task_dir / "unified_result.json", unified)
    progress.emit(
        "report_generation",
        "正在根据三个模块的结构化真实结果生成统一报告。",
        status="running",
        progress_current=selected_completed,
        progress_total=overall_work_total,
        details={
            "report_formats": ["json", "txt", "html"],
            "module_statuses": {
                name: results[name].get("status") for name in MODULE_ORDER
            },
        },
    )
    report_started = time.perf_counter()
    try:
        generated_reports = generate_task_reports(
            task_dir,
            task_id=task_id,
            task_status=task_status,
            deepfake_result=results["deepfake"],
            identity_result=results["identity"],
            source_result=results["source"],
            unified_result=unified,
            product_name=str(
                config.get("product_name")
                or "AI换脸涉案图像智能鉴伪与原始素材溯源系统"
            ),
            product_version=str(config.get("product_version") or "0.3.0"),
            generated_at=finished_at,
        )
        report_runtime = time.perf_counter() - report_started
        report_files = dict(generated_reports["file_names"])
        report_metadata = {
            "status": "success",
            "runtime_seconds": report_runtime,
            "files": report_files,
        }
        task_status["reports"] = report_metadata
        unified["reports"] = report_metadata
        progress.emit(
            "report_generated",
            "检测与检索结果报告已生成。",
            status="success",
            progress_current=overall_work_total,
            progress_total=overall_work_total,
            details={"report_files": report_files},
        )
    except Exception as exc:
        report_runtime = time.perf_counter() - report_started
        report_error = f"报告生成失败：{type(exc).__name__}: {exc}"
        errors.append(report_error)
        if overall_status == "success":
            overall_status = "partial_success"
        unified["status"] = overall_status
        unified["errors"] = errors
        task_status["status"] = overall_status
        task_status["errors"] = errors
        report_metadata = {
            "status": "error",
            "runtime_seconds": report_runtime,
            "error_message": report_error,
            "files": {},
        }
        task_status["reports"] = report_metadata
        unified["reports"] = report_metadata
        progress.emit(
            "report_generation_failed",
            report_error,
            status="error",
            progress_current=overall_work_total,
            progress_total=overall_work_total,
            details={"error": report_error},
        )

    artifact_manifest = _task_artifact_manifest(
        task_dir,
        task_id,
        results,
        report_files=(
            dict(report_metadata.get("files"))
            if isinstance(report_metadata.get("files"), Mapping)
            else {}
        ),
    )
    _write_json(task_dir / "artifact_manifest.json", artifact_manifest)
    artifact_reference = {
        "file": "artifact_manifest.json",
        "artifact_count": artifact_manifest["artifact_count"],
        "path_policy": artifact_manifest["path_policy"],
    }
    unified["artifact_manifest"] = artifact_reference
    task_status["artifact_manifest"] = artifact_reference

    _write_json(task_dir / "unified_result.json", unified)
    _write_json(task_dir / "task_status.json", task_status)
    progress.emit(
        "completed",
        "检测与检索结果已生成。",
        status=(
            "success"
            if overall_status == "success"
            else "warning"
            if overall_status == "partial_success"
            else "error"
        ),
        progress_current=overall_work_total,
        progress_total=overall_work_total,
        details={"task_status": overall_status, "task_dir": str(task_dir)},
    )
    _write_text(
        logs_dir / "orchestrator.log",
        "\n".join(
            [
                f"task_id={task_id}",
                f"status={overall_status}",
                "selected_modules=" + ",".join(name for name in MODULE_ORDER if name in selected),
                *[f"{name}_status={results[name]['status']}" for name in MODULE_ORDER],
                f"runtime_seconds={total_runtime:.6f}",
            ]
        )
        + "\n",
    )
    return task_status


__all__ = [
    "DEFAULT_CONFIG",
    "MODULE_ORDER",
    "OrchestrationError",
    "RuntimeServices",
    "load_config",
    "resolve_path",
    "run_task",
]
