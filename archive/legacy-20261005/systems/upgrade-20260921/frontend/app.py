from __future__ import annotations

from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import secrets
import signal
import sys
import threading
from typing import Any


FRONTEND_ROOT = Path(__file__).resolve().parent
PRODUCT_ROOT = FRONTEND_ROOT.parent
for import_root in (FRONTEND_ROOT, PRODUCT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import streamlit as st

from orchestrator import (
    OrchestrationError,
    load_config,
    resolve_path,
    run_task,
)
from components.data_loader import (
    ResultDataError,
    TaskResults,
    load_detection_result,
    load_identity_result,
    load_source_result,
    load_task_results,
)
from components.path_security import ResourcePathPolicy
from components.live_progress import render_live_progress
from components.report_views import render_task_reports
from components.evidence_dashboard import render_evidence_workspace
from components.status_adapter import (
    ModuleAvailability,
    device_label,
    module_availability,
)


PROJECT_ROOT = FRONTEND_ROOT
APP_CONFIG = Path(
    os.environ.get("FSTS_CONFIG_PATH", str(PRODUCT_ROOT / "config.json"))
).expanduser().resolve(strict=False)
SAMPLE_ROOT = FRONTEND_ROOT / "sample_data"
MOCK_RESULTS_ROOT = FRONTEND_ROOT / "mock_results"
IDENTITY_LIBRARY_ROOT = SAMPLE_ROOT / "identity_libraries"
SOURCE_LIBRARY_ROOT = SAMPLE_ROOT / "source_libraries"
MODULE_KEYS = ("deepfake", "identity", "source")
MODULE_LABELS = {
    "deepfake": "AI换脸/深伪风险甄别",
    "identity": "人脸身份搜索",
    "source": "原始底图搜索",
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_VIDEO_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


@st.cache_resource(show_spinner=False)
def _live_job_runtime() -> dict[str, Any]:
    """One serial background executor plus a process-local job registry."""

    return {
        "executor": ThreadPoolExecutor(max_workers=1, thread_name_prefix="fsts-analysis"),
        "lock": threading.RLock(),
        "jobs": {},
    }


def _job_update(runtime: Mapping[str, Any], job_id: str, **values: Any) -> None:
    lock = runtime["lock"]
    with lock:
        job = runtime["jobs"].get(job_id)
        if isinstance(job, dict):
            job.update(values)


def _job_snapshot(job_id: str | None) -> dict[str, Any] | None:
    if not job_id:
        return None
    runtime = _live_job_runtime()
    with runtime["lock"]:
        job = runtime["jobs"].get(job_id)
        return dict(job) if isinstance(job, dict) else None


def _execute_live_job(
    runtime: Mapping[str, Any],
    job_id: str,
    *,
    inbox_path: Path,
    identity_root: Path | None,
    source_root: Path | None,
    modules: frozenset[str],
    services: Any | None,
) -> None:
    """Run forensic work without calling any Streamlit API from this thread."""

    _job_update(runtime, job_id, status="running")

    def on_progress(event: dict[str, Any]) -> None:
        values: dict[str, Any] = {"latest_event": dict(event)}
        details = event.get("details")
        if isinstance(details, Mapping) and details.get("task_dir"):
            values["task_dir"] = str(details["task_dir"])
            values["task_id"] = str(event.get("task_id") or "")
        _job_update(runtime, job_id, **values)

    try:
        result = run_task(
            inbox_path,
            identity_library=identity_root,
            source_library=source_root,
            selected_modules=modules,
            services=services,
            config_path=APP_CONFIG,
            progress_callback=on_progress,
        )
        _job_update(
            runtime,
            job_id,
            status=str(result.get("status") or "error"),
            task_dir=str(result.get("task_dir") or ""),
            task_id=str(result.get("task_id") or ""),
            result=result,
            error=None,
        )
    except Exception as exc:
        _job_update(
            runtime,
            job_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        try:
            inbox_path.unlink(missing_ok=True)
        except OSError:
            pass


def discover_libraries(library_root: Path) -> list[Path]:
    if not library_root.is_dir():
        return []
    return sorted(
        (path.resolve() for path in library_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )


def _parse_optional_root(path_value: str, label: str) -> tuple[Path, ...]:
    value = path_value.strip()
    if not value:
        return ()
    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_dir():
        raise ResultDataError(f"{label}不存在或不是目录：{path}")
    return (path,)


@st.cache_data(ttl=5, show_spinner=False)
def _cached_system_status(config_path_text: str) -> dict[str, Any]:
    """Inspect real resources while turning import/inspection failures into data."""

    try:
        from system_status import inspect_system_status

        snapshot = inspect_system_status(config_path=config_path_text)
        if not isinstance(snapshot, dict):
            raise TypeError("系统状态检查结果不是 JSON 对象")
        return {"snapshot": snapshot, "error": None}
    except Exception as exc:
        return {
            "snapshot": {
                "overall_status": "unavailable",
                "device": "unknown",
                "deepfake": {
                    "available": False,
                    "message": "无法完成 AIGC 资源检查。",
                },
                "identity": {
                    "available": False,
                    "message": "无法完成身份库检查。",
                },
                "source": {
                    "available": False,
                    "message": "无法完成底图库检查。",
                },
            },
            "error": f"{type(exc).__name__}: {exc}",
        }


@st.cache_resource(show_spinner="正在初始化 AIGC 与身份模型服务……")
def _cached_runtime_services(config_path_text: str) -> dict[str, Any]:
    """Start model-caching workers once; callers can always fall back safely."""

    manager: Any = None
    try:
        from runtime_services import RuntimeServiceManager

        manager = RuntimeServiceManager(config_path_text)
        startup_status = manager.start()
        if not isinstance(startup_status, dict):
            startup_status = manager.get_status()
        return {
            "manager": manager,
            "status": startup_status,
            "error": None,
        }
    except Exception as exc:
        if manager is not None:
            try:
                manager.close()
            except Exception:
                pass
        return {
            "manager": None,
            "status": {"started": False, "ready": False, "health": "error"},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _runtime_service_snapshot(bundle: Mapping[str, Any]) -> dict[str, Any]:
    manager = bundle.get("manager")
    if manager is not None:
        try:
            status = manager.get_status()
            if isinstance(status, dict):
                return status
        except Exception as exc:
            return {
                "started": False,
                "ready": False,
                "health": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    status = bundle.get("status")
    return dict(status) if isinstance(status, Mapping) else {}


def _store_results(
    *,
    mode: str,
    deepfake_result: dict[str, Any],
    identity_result: dict[str, Any],
    source_result: dict[str, Any],
    path_policy: ResourcePathPolicy,
    source_label: str,
    legacy_dual_module: bool = False,
    task_status: dict[str, Any] | None = None,
    unified_result: dict[str, Any] | None = None,
    final_report: dict[str, Any] | None = None,
    artifact_manifest: dict[str, Any] | None = None,
) -> None:
    st.session_state["loaded_results"] = {
        "mode": mode,
        "deepfake": deepfake_result,
        "identity": identity_result,
        "source": source_result,
        "policy": path_policy,
        "source_label": source_label,
        "legacy_dual_module": legacy_dual_module,
        "task_status": task_status,
        "unified_result": unified_result,
        "final_report": final_report,
        "artifact_manifest": artifact_manifest,
    }


def _load_mock(identity_library: Path, source_library: Path) -> None:
    policy = ResourcePathPolicy(
        identity_roots=(identity_library,),
        source_roots=(source_library,),
        sample_roots=(SAMPLE_ROOT,),
    )
    deepfake = load_detection_result(
        MOCK_RESULTS_ROOT / "detection_result.json", policy
    )
    identity = load_identity_result(
        MOCK_RESULTS_ROOT / "identity_result.json", policy
    )
    source = load_source_result(MOCK_RESULTS_ROOT / "source_result.json", policy)
    _store_results(
        mode="Mock Demo",
        deepfake_result=deepfake,
        identity_result=identity,
        source_result=source,
        path_policy=policy,
        source_label="内置、非算法生成的布局测试素材",
    )


def _load_task(
    task_dir_text: str,
    identity_root_text: str,
    source_root_text: str,
    resource_root_text: str,
    *,
    mode: str = "Task Result",
) -> None:
    task_dir = Path(task_dir_text.strip()).expanduser().resolve(strict=False)
    results: TaskResults = load_task_results(
        task_dir,
        identity_roots=_parse_optional_root(identity_root_text, "身份库根"),
        source_roots=_parse_optional_root(source_root_text, "底图库根"),
        sample_roots=_parse_optional_root(resource_root_text, "额外批准资源根"),
    )
    _store_results(
        mode=mode,
        deepfake_result=results.deepfake_result,
        identity_result=results.identity_result,
        source_result=results.source_result,
        path_policy=results.path_policy,
        source_label=str(results.task_dir),
        legacy_dual_module=results.legacy_dual_module,
        task_status=results.task_status,
        unified_result=results.unified_result,
        final_report=results.final_report,
        artifact_manifest=results.artifact_manifest,
    )


def _live_defaults() -> tuple[str, str, Path]:
    config, config_path = load_config(APP_CONFIG)
    identity_root = resolve_path(config["identity"]["library"], config_path)
    source_root = resolve_path(config["source"]["library"], config_path)
    inbox_root = resolve_path(config["runtime"]["inbox_root"], config_path)
    return str(identity_root), str(source_root), inbox_root


def _archived_task_defaults() -> tuple[str, str, str]:
    """Return formal config roots and the newest readable task, if present."""

    config, config_path = load_config(APP_CONFIG)
    tasks_root = resolve_path(config["runtime"]["tasks_root"], config_path)
    identity_root = resolve_path(config["identity"]["library"], config_path)
    source_root = resolve_path(config["source"]["library"], config_path)
    newest_task: Path | None = None
    try:
        candidates = [
            path
            for path in tasks_root.iterdir()
            if path.is_dir() and (path / "task_status.json").is_file()
        ]
        if candidates:
            newest_task = max(
                candidates,
                key=lambda path: (path / "task_status.json").stat().st_mtime_ns,
            )
    except OSError:
        newest_task = None
    return (
        str(newest_task or tasks_root),
        str(identity_root),
        str(source_root),
    )


def _validate_upload(uploaded_query: Any) -> str:
    if uploaded_query is None:
        raise ResultDataError("请先上传查询图片或视频。")
    suffix = Path(str(uploaded_query.name)).suffix.lower()
    size = getattr(uploaded_query, "size", 0)
    if suffix in ALLOWED_VIDEO_SUFFIXES:
        if isinstance(size, (int, float)) and size > MAX_VIDEO_UPLOAD_BYTES:
            raise ResultDataError("查询视频超过 200 MiB 上限。")
        return suffix
    if isinstance(size, (int, float)) and size > MAX_UPLOAD_BYTES:
        raise ResultDataError("查询图片超过 20 MiB 上限。")
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ResultDataError(
            "查询文件扩展名只允许 jpg/jpeg/png/webp（图片）或 mp4/mov/avi/mkv（视频）。"
        )
    return suffix


def _humanize_progress(event: Any) -> str:
    if isinstance(event, Mapping):
        message = event.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        module = str(event.get("module") or event.get("service") or "").lower()
        stage = str(event.get("stage") or event.get("state") or "").lower()
        if module == "deepfake":
            return "正在加载 AIGC 模型并检测图像……"
        if module == "identity":
            return "正在检测人脸、提取身份特征并搜索数据库……"
        if module == "source":
            return "正在提取底图特征并搜索数据库……"
        if stage:
            return f"任务进度：{stage}"
        return "正在执行分析任务……"
    message = str(event).strip()
    replacements = {
        "正在进行人脸身份搜索": "正在检测人脸、提取身份特征并搜索身份数据库",
        "正在进行原始底图搜索": "正在提取底图特征并搜索底图数据库",
        "正在进行换脸鉴伪": "正在加载 AIGC 模型并检测图像",
    }
    for original, friendly in replacements.items():
        if original in message:
            return message.replace(original, friendly)
    return message or "正在执行分析任务……"


def _start_live_analysis(
    uploaded_query: Any,
    identity_root_text: str,
    source_root_text: str,
    selected_modules: Collection[str],
    services: Any | None,
) -> str:
    modules = frozenset(selected_modules)
    if not modules or not modules <= set(MODULE_KEYS):
        raise ResultDataError(f"未知模块选择：{sorted(modules)}")
    suffix = _validate_upload(uploaded_query)

    identity_root = (
        _parse_optional_root(identity_root_text, "身份库")
        if "identity" in modules
        else ()
    )
    source_root = (
        _parse_optional_root(source_root_text, "底图库")
        if "source" in modules
        else ()
    )
    if "identity" in modules and not identity_root:
        raise ResultDataError("人脸身份搜索必须指定身份库。")
    if "source" in modules and not source_root:
        raise ResultDataError("原始底图搜索必须指定底图库。")

    _, _, inbox_root = _live_defaults()
    inbox_root.mkdir(parents=True, exist_ok=True)
    inbox_path = inbox_root / f"upload_{secrets.token_hex(8)}{suffix}"
    inbox_path.write_bytes(uploaded_query.getvalue())
    job_id = f"job_{secrets.token_hex(8)}"
    runtime = _live_job_runtime()
    job = {
        "job_id": job_id,
        "status": "queued",
        "task_id": None,
        "task_dir": None,
        "selected_modules": tuple(key for key in MODULE_KEYS if key in modules),
        "identity_root": str(identity_root[0]) if identity_root else "",
        "source_root": str(source_root[0]) if source_root else "",
        "query_name": str(uploaded_query.name),
        "latest_event": None,
        "result": None,
        "error": None,
    }
    with runtime["lock"]:
        runtime["jobs"][job_id] = job
    try:
        future = runtime["executor"].submit(
            _execute_live_job,
            runtime,
            job_id,
            inbox_path=inbox_path,
            identity_root=identity_root[0] if identity_root else None,
            source_root=source_root[0] if source_root else None,
            modules=modules,
            services=services,
        )
    except Exception:
        try:
            inbox_path.unlink(missing_ok=True)
        except OSError:
            pass
        with runtime["lock"]:
            runtime["jobs"].pop(job_id, None)
        raise
    _job_update(runtime, job_id, future=future)
    return job_id


def _render_loaded_results(mode: str) -> None:
    loaded = st.session_state.get("loaded_results")
    if not loaded or loaded["mode"] != mode:
        return

    st.success(f"{mode} 结果文件格式与路径检查正常。")
    source_name = Path(str(loaded["source_label"])).name or "当前任务"
    st.write(f"**结果来源：** `{source_name}`")
    if loaded.get("legacy_dual_module"):
        st.warning(
            "检测到旧版双模块 task：缺少 detection_result.json，"
            "AIGC 页签明确标记为“未运行”，不伪造鉴伪结果。"
        )

    task_root = (
        MOCK_RESULTS_ROOT.resolve()
        if mode == "Mock Demo"
        else Path(str(loaded["source_label"])).expanduser().resolve(strict=False)
    )
    try:
        config, _ = load_config(APP_CONFIG)
        product_version = str(config.get("product_version") or "0.4.0")
    except Exception:
        product_version = "0.4.0"
    selected_page = render_evidence_workspace(
        deepfake=loaded["deepfake"],
        identity=loaded["identity"],
        source=loaded["source"],
        policy=loaded["policy"],
        task_dir=task_root,
        task_status=loaded.get("task_status"),
        final_report=loaded.get("final_report"),
        task_artifact_manifest=loaded.get("artifact_manifest"),
        product_version=product_version,
        key_prefix=f"{mode.replace(' ', '_').lower()}_{task_root.name}",
    )

    if mode != "Mock Demo" and selected_page == "证据链与完整报告":
        st.markdown("### 完整报告与导出")
        render_task_reports(
            task_root,
            task_id=task_root.name,
            key_prefix=f"{mode.replace(' ', '_').lower()}_{task_root.name}",
        )

    with st.expander("原始 JSON 与批准资源根"):
        st.markdown("#### 完整任务路径（技术详情）")
        st.code(str(loaded["source_label"]))
        st.markdown("#### AI 换脸鉴伪 JSON")
        st.json(loaded["deepfake"])
        st.markdown("#### 人脸身份搜索 JSON")
        st.json(loaded["identity"])
        st.markdown("#### 原始底图搜索 JSON")
        st.json(loaded["source"])
        st.markdown("#### 本次批准的资源根")
        for root in loaded["policy"].display_roots():
            st.code(root)


def _render_status_strip(
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, ModuleAvailability], tuple[str, str]]:
    modules = {key: module_availability(snapshot, key) for key in MODULE_KEYS}
    device = device_label(snapshot)
    columns = st.columns(4)
    for column, key in zip(columns[:3], MODULE_KEYS, strict=True):
        item = modules[key]
        column.metric(item.title, item.value)
        column.caption(item.caption)
    columns[3].metric("运算设备", device[0])
    columns[3].caption(device[1])
    return modules, device


def _render_sidebar_system_status(
    modules: Mapping[str, ModuleAvailability],
    device: tuple[str, str],
    service_bundle: Mapping[str, Any],
    inspection_error: str | None,
) -> None:
    st.header("系统就绪状态")
    for key in MODULE_KEYS:
        item = modules[key]
        icon = "🟢" if item.runnable else "🟠" if item.empty else "🔴"
        st.write(f"{icon} **{item.title}**：{item.value}")
        st.caption(item.caption)
    st.write(f"🖥️ **运算设备**：{device[0]}")
    st.caption(device[1])

    deepfake_raw = modules["deepfake"].raw
    components = deepfake_raw.get("components")
    readiness = deepfake_raw.get("readiness")
    if isinstance(components, Mapping):
        labels = {
            "spatial_model": "空间域模型",
            "frequency_model": "频域模型",
            "fusion_config": "融合配置",
            "decision_threshold": "判定阈值配置",
            "python": "Python解释器",
            "torch": "torch",
            "torchvision": "torchvision",
            "sklearn": "sklearn",
            "opencv": "opencv",
            "cuda": "CUDA",
        }
        with st.expander("AIGC启动前资源检查", expanded=not modules["deepfake"].runnable):
            for key, label in labels.items():
                record = components.get(key)
                if not isinstance(record, Mapping):
                    st.error(f"{label}：未返回检查结果")
                    continue
                status = str(record.get("status") or "unknown")
                available = record.get("available")
                detail = record.get("message") or record.get("version") or record.get("value")
                icon = "✅" if status == "ok" else "⚠️" if key == "cuda" and available is False else "❌"
                st.write(f"{icon} **{label}**：{status}{f'（{detail}）' if detail is not None else ''}")
            ready_status = readiness.get("status") if isinstance(readiness, Mapping) else "unknown"
            if ready_status == "ready":
                st.success("AIGC模块：就绪")
            else:
                missing = readiness.get("missing_resources", []) if isinstance(readiness, Mapping) else []
                st.error(f"AIGC模块：不可用；缺少或异常资源：{'、'.join(str(item) for item in missing) or '未知'}")

    service_status = _runtime_service_snapshot(service_bundle)
    service_ready = {
        key: bool((service_status.get(key) or {}).get("ready"))
        for key in ("deepfake", "identity")
        if isinstance(service_status.get(key), Mapping)
    }
    if service_bundle.get("manager") is None:
        st.warning("模型缓存服务未启动；真实分析将降级为隔离子进程。")
    elif service_ready and all(service_ready.values()):
        st.success("AIGC/身份模型已由持久服务缓存，后续查询可复用。")
    else:
        ready_names = [MODULE_LABELS[key] for key, ready in service_ready.items() if ready]
        ready_text = "、".join(ready_names) if ready_names else "无"
        st.warning(
            f"持久服务部分可用（已就绪：{ready_text}）；"
            "不可用模块将自动降级为隔离子进程。"
        )
    if service_bundle.get("error"):
        st.caption(f"服务降级原因：{service_bundle['error']}")
    if inspection_error:
        st.error(f"系统状态检查失败：{inspection_error}")
    with st.expander("服务运行详情"):
        st.json(service_status)


def _module_meta(task_status: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = task_status.get(key)
    return dict(value) if isinstance(value, Mapping) else {"status": "skipped"}


def _render_live_summary() -> None:
    task_status = st.session_state.get("live_task_status")
    loaded = st.session_state.get("loaded_results")
    if not isinstance(task_status, Mapping) or not loaded:
        return
    if loaded.get("mode") != "Live Analysis":
        return

    columns = st.columns(4)
    columns[0].metric("任务总状态", str(task_status.get("status") or "未知"))
    for column, key in zip(columns[1:], MODULE_KEYS, strict=True):
        meta = _module_meta(task_status, key)
        status = str(meta.get("status") or meta.get("state") or "skipped")
        runtime = meta.get("runtime_seconds")
        delta = f"{float(runtime):.3f} 秒" if isinstance(runtime, (int, float)) else None
        column.metric(MODULE_LABELS[key], status, delta, delta_color="off")
        execution_mode = meta.get("execution_mode")
        fallback_reason = meta.get("fallback_reason")
        if fallback_reason:
            column.caption(f"已降级为隔离子进程：{fallback_reason}")
        elif execution_mode:
            column.caption(f"执行方式：{execution_mode}")
        error = meta.get("error") or meta.get("error_message")
        if status == "error" and error:
            column.caption(str(error))
    if task_status.get("task_id"):
        st.write(f"**task_id：** `{task_status['task_id']}`")
    total_runtime = task_status.get("runtime_seconds")
    if isinstance(total_runtime, (int, float)):
        st.write(f"**总耗时：** `{float(total_runtime):.3f} 秒`")
    errors = task_status.get("errors")
    if isinstance(errors, list):
        for error in errors:
            st.error(str(error))


@st.fragment(run_every=0.5)
def _render_active_job() -> None:
    job_id = st.session_state.get("live_job_id")
    job = _job_snapshot(str(job_id) if job_id else None)
    if not job:
        return
    status = str(job.get("status") or "queued")
    selected = tuple(job.get("selected_modules") or MODULE_KEYS)
    task_dir = str(job.get("task_dir") or "").strip() or None
    render_live_progress(
        task_dir=task_dir,
        selected_modules=selected,
        job_status=status,
        queued_message=(
            "当前任务正在等待前一个任务结束；系统按提交顺序串行执行，避免模型与底图库争用。"
            if status == "queued"
            else "正在创建独立任务目录……"
        ),
    )

    if status in {"queued", "running"}:
        return
    error = job.get("error")
    if error and not task_dir:
        st.error(f"分析任务未能创建结果目录：{error}")
        return
    result = job.get("result")
    if not task_dir or not isinstance(result, Mapping):
        st.error(f"分析任务结束但没有返回可读结果：{error or status}")
        return

    if st.session_state.get("loaded_live_job_id") == job_id:
        return
    try:
        _load_task(
            task_dir,
            str(job.get("identity_root") or ""),
            str(job.get("source_root") or ""),
            "",
            mode="Live Analysis",
        )
    except (ResultDataError, OSError, RuntimeError) as exc:
        st.error(f"任务完成，但结果文件格式检查未通过：{exc}")
        return
    st.session_state["live_task_status"] = dict(result)
    st.session_state["live_selected_modules"] = selected
    st.session_state["loaded_live_job_id"] = job_id
    st.rerun(scope="app")


def _render_shutdown(service_bundle: Mapping[str, Any]) -> None:
    st.divider()
    if st.button("关闭系统", help="停止本地服务并退出程序", width="stretch"):
        manager = service_bundle.get("manager")
        if manager is not None:
            try:
                manager.close()
            except Exception as exc:
                st.warning(f"持久服务关闭时返回警告：{exc}")
        st.success("系统正在关闭，可以关闭此浏览器页面。")
        threading.Timer(
            0.8,
            lambda: os.kill(os.getpid(), signal.SIGTERM),
        ).start()
        st.stop()


def _render_live_mode(
    *,
    modules: Mapping[str, ModuleAvailability],
    service_bundle: Mapping[str, Any],
    identity_root_text: str,
    source_root_text: str,
    live_config_error: str,
) -> None:
    st.subheader("案件图像分析")
    st.caption("上传一张图片或一段视频，然后选择单一模块或一键综合分析。")
    uploaded_query = st.file_uploader(
        "上传涉案图片或视频",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
        help=(
            "上传文件只暂存于 runtime/inbox，调度器为每次分析创建独立 task_dir。"
            "视频输入会先抽帧选关键帧，再逐帧鉴伪/身份检索，底图检索只用首要帧。"
        ),
        key="live_query",
    )
    if uploaded_query is not None:
        uploaded_suffix = Path(str(uploaded_query.name)).suffix.lower()
        if uploaded_suffix in ALLOWED_VIDEO_SUFFIXES:
            st.video(uploaded_query)
            st.caption(f"待分析视频：{uploaded_query.name}")
        else:
            st.image(
                uploaded_query,
                caption=f"待分析：{uploaded_query.name}",
                width=420,
            )
    else:
        st.info(
            "请先上传 JPG/JPEG/PNG/WEBP 图片（最大 20 MiB）或 MP4/MOV/AVI/MKV 视频（最大 200 MiB）。"
        )

    upload_error = ""
    if uploaded_query is not None:
        try:
            _validate_upload(uploaded_query)
        except ResultDataError as exc:
            upload_error = str(exc)
            st.error(upload_error)
    upload_missing = uploaded_query is None
    upload_blocked = upload_missing or bool(upload_error)
    config_blocked = bool(live_config_error)
    active_job = _job_snapshot(st.session_state.get("live_job_id"))
    analysis_running = bool(
        active_job and str(active_job.get("status")) in {"queued", "running"}
    )
    deepfake_disabled = (
        upload_blocked
        or config_blocked
        or analysis_running
        or not modules["deepfake"].runnable
    )
    identity_disabled = (
        upload_blocked
        or config_blocked
        or analysis_running
        or not modules["identity"].runnable
    )
    source_disabled = (
        upload_blocked
        or config_blocked
        or analysis_running
        or not modules["source"].runnable
    )
    comprehensive_disabled = (
        upload_blocked
        or config_blocked
        or analysis_running
        or any(not modules[key].runnable for key in MODULE_KEYS)
    )

    columns = st.columns(4)
    deepfake_clicked = columns[0].button(
        "AI换脸/深伪风险甄别",
        type="primary",
        disabled=deepfake_disabled,
        width="stretch",
    )
    identity_clicked = columns[1].button(
        "人脸身份搜索",
        type="primary",
        disabled=identity_disabled,
        width="stretch",
    )
    source_clicked = columns[2].button(
        "原始底图搜索",
        type="primary",
        disabled=source_disabled,
        width="stretch",
    )
    comprehensive_clicked = columns[3].button(
        "一键综合分析",
        type="primary",
        disabled=comprehensive_disabled,
        width="stretch",
    )

    for key in MODULE_KEYS:
        item = modules[key]
        if item.empty:
            st.error(f"{item.title}为空或未初始化：{item.message}；相应按钮已禁用。")
        elif not item.available:
            st.warning(f"{item.title}不可用：{item.message}；相应按钮已禁用。")
    if live_config_error:
        st.error(f"配置读取失败：{live_config_error}")
    if analysis_running:
        st.info("已有分析任务正在后台运行；当前页面会每 0.5 秒读取真实进度事件。")

    selected: frozenset[str] | None = None
    if deepfake_clicked:
        selected = frozenset({"deepfake"})
    elif identity_clicked:
        selected = frozenset({"identity"})
    elif source_clicked:
        selected = frozenset({"source"})
    elif comprehensive_clicked:
        selected = frozenset(MODULE_KEYS)

    if selected is not None:
        st.session_state.pop("loaded_results", None)
        st.session_state.pop("live_task_status", None)
        st.session_state.pop("loaded_live_job_id", None)
        try:
            job_id = _start_live_analysis(
                uploaded_query,
                identity_root_text,
                source_root_text,
                selected,
                service_bundle.get("manager"),
            )
            st.session_state["live_job_id"] = job_id
            st.rerun()
        except (ResultDataError, OrchestrationError, OSError, RuntimeError) as exc:
            st.error(f"分析失败：{exc}")
        except Exception as exc:
            st.error(f"分析失败：{type(exc).__name__}: {exc}")

    _render_active_job()
    _render_live_summary()
    _render_loaded_results("Live Analysis")


def main() -> None:
    st.set_page_config(
        page_title="AI换脸涉案图像智能鉴伪与原始素材溯源系统",
        page_icon="🔍",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1720px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(49, 95, 134, 0.20);
            border-radius: 0.70rem;
            padding: 0.62rem 0.78rem;
            background: linear-gradient(180deg, rgba(49, 95, 134, 0.035), rgba(49, 95, 134, 0.012));
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(49, 95, 134, 0.22);
            box-shadow: 0 4px 16px rgba(23, 45, 64, 0.055);
        }
        h1, h2, h3, h4 {
            letter-spacing: -0.015em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI换脸涉案图像智能鉴伪与原始素材溯源系统")
    st.caption(
        "单图任务、三模块故障隔离；综合分析只汇总证据，不伪造融合概率。"
    )

    status_result = _cached_system_status(str(APP_CONFIG))
    snapshot = status_result["snapshot"]
    modules, device = _render_status_strip(snapshot)
    service_bundle = _cached_runtime_services(str(APP_CONFIG))

    identity_libraries = discover_libraries(IDENTITY_LIBRARY_ROOT)
    source_libraries = discover_libraries(SOURCE_LIBRARY_ROOT)

    with st.sidebar:
        _render_sidebar_system_status(
            modules,
            device,
            service_bundle,
            status_result.get("error"),
        )
        if st.button("重新检测 / 重试服务", width="stretch"):
            manager = service_bundle.get("manager")
            if manager is not None:
                try:
                    manager.close()
                except Exception:
                    pass
            _cached_runtime_services.clear()
            _cached_system_status.clear()
            st.rerun()

        st.divider()
        st.header("结果来源")
        mode_options = ["Live Analysis", "Task Result"]
        if os.environ.get("FSTS_ENABLE_DEV_MOCK") == "1":
            mode_options.append("Mock Demo")
        mode = st.radio(
            "运行模式",
            mode_options,
            format_func={
                "Live Analysis": "真实分析",
                "Task Result": "读取已归档任务",
                "Mock Demo": "开发模拟数据",
            }.get,
        )

        if mode == "Mock Demo":
            st.warning("模拟数据：仅用于隐藏的开发布局测试，不是算法结果或准确率证据。")
            identity_names = [path.name for path in identity_libraries]
            source_names = [path.name for path in source_libraries]
            selected_identity = st.selectbox(
                "选择演示身份库",
                identity_names or ["(无演示库)"],
                disabled=not identity_names,
            )
            selected_source = st.selectbox(
                "选择演示底图库",
                source_names or ["(无演示库)"],
                disabled=not source_names,
            )
            load_button = st.button(
                "加载 Mock 演示",
                type="primary",
                disabled=not identity_names or not source_names,
                width="stretch",
            )
            task_dir_text = identity_root_text = source_root_text = resource_root_text = ""
            live_config_error = ""
        elif mode == "Task Result":
            try:
                (
                    default_task_dir,
                    default_identity_root,
                    default_source_root,
                ) = _archived_task_defaults()
                archived_config_error = ""
            except (OrchestrationError, KeyError, TypeError) as exc:
                default_task_dir = default_identity_root = default_source_root = ""
                archived_config_error = str(exc)
            task_dir_text = st.text_input("task_dir", value=default_task_dir)
            identity_root_text = st.text_input(
                "批准的身份库根",
                value=default_identity_root,
            )
            source_root_text = st.text_input(
                "批准的底图库根",
                value=default_source_root,
            )
            resource_root_text = st.text_input(
                "额外批准资源根（可留空）",
                value="",
            )
            if archived_config_error:
                st.error(f"归档任务配置读取失败：{archived_config_error}")
            else:
                st.caption("默认定位到当前用户数据目录中最近完成的任务；也可填写其他具体 case 目录。")
            load_button = st.button(
                "读取 Task Result",
                type="primary",
                width="stretch",
                disabled=bool(archived_config_error),
            )
            selected_identity = selected_source = ""
            live_config_error = ""
        else:
            try:
                default_identity, default_source, _ = _live_defaults()
                live_config_error = ""
            except (OrchestrationError, KeyError, TypeError) as exc:
                default_identity, default_source = "", ""
                live_config_error = str(exc)
            identity_root_text = st.text_input(
                "本次任务身份库",
                value=default_identity,
                key="live_identity_root",
            )
            source_root_text = st.text_input(
                "本次任务底图库",
                value=default_source,
                key="live_source_root",
            )
            st.caption("顶部真实计数对应当前 config 指向的数据库。")
            task_dir_text = resource_root_text = ""
            selected_identity = selected_source = ""
            load_button = False

        _render_shutdown(service_bundle)

    if mode == "Mock Demo":
        st.warning("当前页面正在显示模拟数据，禁止将其作为检测或检索结论。")
        st.subheader("开发模拟数据")
        uploaded_preview = st.file_uploader(
            "可选：仅在内存中预览查询图片",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            help="该图片不会写入源码目录。",
            key="mock_preview",
        )
        if uploaded_preview is not None:
            st.image(
                uploaded_preview,
                caption=f"仅内存预览：{uploaded_preview.name}",
                width=420,
            )
        else:
            st.info("无需上传即可加载内置 Mock。")
        if load_button:
            try:
                identity_path = next(
                    path for path in identity_libraries if path.name == selected_identity
                )
                source_path = next(
                    path for path in source_libraries if path.name == selected_source
                )
                _load_mock(identity_path, source_path)
            except (ResultDataError, StopIteration) as exc:
                st.error(f"Mock 结果拒绝加载：{exc}")
        _render_loaded_results("Mock Demo")
    elif mode == "Task Result":
        st.subheader("Task Result")
        st.info(
            "只读 task_dir 下固定命名的三份结果 JSON；"
            "旧双模块 task 如缺少 detection_result.json，AIGC 会显式标记为 skipped。"
        )
        if load_button:
            try:
                _load_task(
                    task_dir_text,
                    identity_root_text,
                    source_root_text,
                    resource_root_text,
                )
            except (ResultDataError, OSError, RuntimeError) as exc:
                st.error(f"Task Result 拒绝加载：{exc}")
        _render_loaded_results("Task Result")
    else:
        _render_live_mode(
            modules=modules,
            service_bundle=service_bundle,
            identity_root_text=identity_root_text,
            source_root_text=source_root_text,
            live_config_error=live_config_error,
        )


if __name__ == "__main__":
    main()
