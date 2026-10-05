from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser


PRODUCT_NAME = "换脸寻源系统 v0.4.0"
PRODUCT_DATA_DIR = "FaceSourceTrace"
PRODUCT_VERSION = "0.4.0"
MAIN_EXE_NAME = "换脸寻源系统_v0.4.0.exe"
HEALTH_PATH = "/_stcore/health"
STATE_PREFIX = "launcher_state_v0.4.0"


def _install_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _bundle_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    return Path(frozen_root).resolve() if frozen_root else _install_root()


def _data_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return (base / PRODUCT_DATA_DIR).resolve(strict=False)


def _canonical_text(path: Path) -> str:
    return str(path.resolve(strict=False)).rstrip("\\/").casefold()


def _instance_key(app_root: Path) -> str:
    return hashlib.sha256(_canonical_text(app_root).encode("utf-8")).hexdigest()[:12]


def _state_path(data_root: Path, app_root: Path) -> Path:
    return data_root / f"{STATE_PREFIX}_{_instance_key(app_root)}.json"


def _show_error(title: str, message: str) -> None:
    if os.name == "nt":
        try:
            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
            return
        except Exception:
            pass
    print(f"{title}: {message}", file=sys.stderr)


def _configure_logging(data_root: Path):
    log_dir = data_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "launcher_v0.4.0.log"
    stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = stream
    sys.stderr = stream
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launcher pid={os.getpid()}")
    return stream


def _read_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取{label}：{path}\n{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label}格式错误：{path}")
    return payload


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _expand_path(value: str, app_root: Path, data_root: Path) -> Path:
    expanded = os.path.expandvars(str(value))
    expanded = expanded.replace("{APP_ROOT}", str(app_root))
    expanded = expanded.replace("{DATA_ROOT}", str(data_root))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = app_root / path
    return path.resolve(strict=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_resource_manifest(config: dict, app_root: Path, data_root: Path) -> list[str]:
    resource_config = config.get("resources", {})
    if not resource_config.get("verify_manifest_on_start", False):
        return []
    value = resource_config.get("manifest")
    if not value:
        return ["resources.manifest 未配置"]
    manifest_path = _expand_path(str(value), app_root, data_root)
    if not manifest_path.is_file():
        return [f"资源哈希清单不存在：{manifest_path}"]
    try:
        manifest = _read_json(manifest_path, "资源哈希清单")
    except RuntimeError as exc:
        return [str(exc)]
    if str(manifest.get("schema_version")) != "1.0":
        return [f"资源哈希清单版本不受支持：{manifest_path}"]
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        return [f"资源哈希清单没有可验证条目：{manifest_path}"]

    errors: list[str] = []
    app_text = _canonical_text(app_root)
    for record in entries:
        if not isinstance(record, dict):
            errors.append("资源哈希清单包含无效记录")
            continue
        relative = str(record.get("path", "")).replace("/", os.sep)
        expected_hash = str(record.get("sha256", "")).lower()
        try:
            expected_size = int(record.get("size"))
        except (TypeError, ValueError):
            errors.append(f"资源大小字段无效：{relative}")
            continue
        candidate = (app_root / relative).resolve(strict=False)
        candidate_text = _canonical_text(candidate)
        if candidate_text != app_text and not candidate_text.startswith(app_text + os.sep.casefold()):
            errors.append(f"资源清单路径越界：{relative}")
            continue
        if not candidate.is_file():
            errors.append(f"资源缺失：{relative}")
            continue
        if candidate.stat().st_size != expected_size:
            errors.append(f"资源大小不一致：{relative}")
            continue
        if len(expected_hash) != 64 or _sha256(candidate) != expected_hash:
            errors.append(f"资源 SHA-256 不一致：{relative}")
    return errors


def _preflight(config: dict, app_root: Path, data_root: Path) -> None:
    features = config.get("features", {})
    resources = config.get("resources", {})
    errors: list[str] = []
    if str(config.get("schema_version")) != "3.0":
        errors.append("安装配置 schema_version 不是 3.0")
    if config.get("build_flavor") != "formal":
        errors.append("安装配置不是 formal 构建")
    if features.get("mock_enabled") is not False:
        errors.append("正式安装配置禁止启用 Mock")
    if features.get("real_modules_default") is not True:
        errors.append("正式安装配置必须默认运行真实模块")
    if resources.get("verify_manifest_on_start") is not True:
        errors.append("正式安装配置必须启用资源 SHA-256 校验")

    required_files = [
        ("AI 换脸甄别 Python", config.get("deepfake", {}).get("python")),
        ("AI 换脸甄别适配器", config.get("deepfake", {}).get("runner")),
        ("身份候选检索引擎", config.get("identity", {}).get("executable")),
        ("底图候选检索引擎", config.get("source", {}).get("executable")),
    ]
    required_dirs = [
        ("AI 换脸甄别运行时", config.get("deepfake", {}).get("install_root")),
        ("身份参考库", config.get("identity", {}).get("library")),
        ("底图候选库", config.get("source", {}).get("library")),
    ]
    for label, value in required_files:
        if not value:
            errors.append(f"{label}未配置")
        elif not _expand_path(str(value), app_root, data_root).is_file():
            errors.append(f"{label}不存在")
    for label, value in required_dirs:
        if not value:
            errors.append(f"{label}未配置")
        elif not _expand_path(str(value), app_root, data_root).is_dir():
            errors.append(f"{label}不存在")

    model_root_value = config.get("identity", {}).get("model_root")
    if not model_root_value:
        errors.append("身份模型根目录未配置")
    else:
        model_dir = _expand_path(str(model_root_value), app_root, data_root) / "models" / "buffalo_l"
        expected_models = {
            "1k3d68.onnx",
            "2d106det.onnx",
            "det_10g.onnx",
            "genderage.onnx",
            "w600k_r50.onnx",
        }
        actual_models = {item.name for item in model_dir.iterdir() if item.is_file()} if model_dir.is_dir() else set()
        if actual_models != expected_models:
            errors.append("InsightFace buffalo_l 模型清单不完整或包含未批准文件")

    errors.extend(_verify_resource_manifest(config, app_root, data_root))
    try:
        for key in ("tasks_root", "inbox_root", "identity_cache", "persistent_services_root"):
            value = config.get("runtime", {}).get(key)
            if not value:
                errors.append(f"runtime.{key} 未配置")
                continue
            _expand_path(str(value), app_root, data_root).mkdir(parents=True, exist_ok=True)
        probe = data_root / ".write_probe_v040"
        probe.write_text("ok", encoding="ascii")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        errors.append(f"用户数据目录不可写：{exc}")

    if errors:
        raise RuntimeError("启动前资源检查失败：\n\n" + "\n".join(f"• {item}" for item in errors))


def _health_ok(port: int, timeout: float = 0.75) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{HEALTH_PATH}", timeout=timeout
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _process_executable(pid: int) -> Path | None:
    if os.name != "nt" or pid <= 0:
        return None
    process_query_limited_information = 0x1000
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(
        process_query_limited_information, False, pid
    )
    if not handle:
        return None
    try:
        capacity = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(capacity.value)
        if not kernel32.QueryFullProcessImageNameW(
            handle, 0, buffer, ctypes.byref(capacity)
        ):
            return None
        return Path(buffer.value).resolve(strict=False)
    finally:
        kernel32.CloseHandle(handle)


def _open_existing(state_path: Path, app_root: Path, expected_executable: Path) -> bool:
    try:
        state = _read_json(state_path, "启动状态")
        pid = int(state["pid"])
        port = int(state["port"])
        state_root = Path(str(state["app_root"]))
        state_executable = Path(str(state["executable"]))
    except (RuntimeError, OSError, ValueError, KeyError, TypeError):
        return False
    if _canonical_text(state_root) != _canonical_text(app_root):
        return False
    if _canonical_text(state_executable) != _canonical_text(expected_executable):
        return False
    running_executable = _process_executable(pid)
    if running_executable is None or _canonical_text(running_executable) != _canonical_text(expected_executable):
        return False
    if not _health_ok(port):
        return False
    webbrowser.open(f"http://127.0.0.1:{port}", new=2)
    return True


def _acquire_mutex(app_root: Path):
    if os.name != "nt":
        return object(), False
    name = f"Local\\FaceSourceTrace_v040_{_instance_key(app_root)}"
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.GetLastError.argtypes = []
    kernel32.GetLastError.restype = wintypes.DWORD
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        raise RuntimeError("无法创建本地启动互斥锁。")
    already_exists = kernel32.GetLastError() == 183
    return handle, already_exists


def _release_mutex(handle) -> None:
    if os.name == "nt" and handle:
        kernel32 = ctypes.windll.kernel32
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(handle)


def _select_port(preferred: int) -> int:
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # Windows treats SO_REUSEADDR differently from POSIX: two unrelated
            # processes may bind the same address and port.  That allowed a
            # newly installed instance to share a legacy instance's
            # Streamlit port, so browser connections could reach either UI.
            # Request exclusive ownership on Windows and otherwise rely on the
            # normal bind conflict instead of enabling address reuse.
            if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_EXCLUSIVEADDRUSE,
                    1,
                )
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return int(probe.getsockname()[1])
    raise RuntimeError("无法找到可用的本地端口。")


def _wait_and_open(port: int, log_stream, enabled: bool) -> None:
    if not enabled:
        return
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if _health_ok(port):
            url = f"http://127.0.0.1:{port}"
            print(f"server ready: {url}")
            try:
                webbrowser.open(url, new=2)
            except Exception:
                traceback.print_exc(file=log_stream)
            return
        time.sleep(0.25)
    _show_error(
        PRODUCT_NAME,
        "本地服务未能在 60 秒内启动。请查看用户数据目录中的 launcher_v0.4.0.log。",
    )


def main() -> int:
    app_root = _install_root()
    bundle_root = _bundle_root()
    data_root = _data_root()
    data_root.mkdir(parents=True, exist_ok=True)
    log_stream = _configure_logging(data_root)
    state_path = _state_path(data_root, app_root)
    expected_executable = (
        Path(sys.executable).resolve()
        if getattr(sys, "frozen", False)
        else (app_root / MAIN_EXE_NAME).resolve(strict=False)
    )
    mutex_handle = None

    try:
        if _open_existing(state_path, app_root, expected_executable):
            print("existing verified server reused")
            return 0

        mutex_handle, already_exists = _acquire_mutex(app_root)
        if already_exists:
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if _open_existing(state_path, app_root, expected_executable):
                    print("concurrent verified server reused")
                    return 0
                time.sleep(0.25)
            raise RuntimeError("另一启动进程仍在初始化，请稍后重试。")

        config_path = (
            app_root / "config.json"
            if getattr(sys, "frozen", False)
            else app_root / "packaging_v0.3" / "config.packaged.json"
        )
        app_script = bundle_root / "frontend" / "app.py"
        config = _read_json(config_path, "安装配置")
        if not app_script.is_file():
            raise RuntimeError(f"前端入口不存在：{app_script}")
        _preflight(config, app_root, data_root)
        preferred_port = int(config.get("frontend", {}).get("port", 8767))
        port = _select_port(preferred_port)

        _atomic_write_json(
            state_path,
            {
                "schema_version": "1.0",
                "product_version": PRODUCT_VERSION,
                "pid": os.getpid(),
                "port": port,
                "url": f"http://127.0.0.1:{port}",
                "app_root": str(app_root),
                "executable": str(expected_executable),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        os.environ["FSTS_CONFIG_PATH"] = str(config_path)
        os.environ["FSTS_DATA_ROOT"] = str(data_root)
        os.environ["FSTS_LAUNCHER_STATE"] = str(state_path)
        os.environ["FSTS_FORMAL_MODE"] = "1"
        os.environ["FSTS_MOCK_ENABLED"] = "0"
        os.environ["FSTS_ENABLE_DEV_MOCK"] = "0"
        os.environ["FSTS_PRODUCT_VERSION"] = PRODUCT_VERSION
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

        opener = threading.Thread(
            target=_wait_and_open,
            args=(
                port,
                log_stream,
                bool(config.get("frontend", {}).get("auto_open_browser", True)),
            ),
            name="browser-opener",
            daemon=True,
        )
        opener.start()

        sys.argv = [
            "streamlit",
            "run",
            str(app_script),
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--server.headless=true",
            "--server.fileWatcherType=none",
            "--browser.gatherUsageStats=false",
            "--global.developmentMode=false",
        ]
        from streamlit.web import cli as streamlit_cli

        return int(streamlit_cli.main() or 0)
    except Exception as exc:
        traceback.print_exc(file=log_stream)
        _show_error(
            f"{PRODUCT_NAME} 启动失败",
            f"{exc}\n\n详细日志位于：\n{data_root / 'logs' / 'launcher_v0.4.0.log'}",
        )
        return 2
    finally:
        try:
            if state_path.is_file():
                state = _read_json(state_path, "启动状态")
                if int(state.get("pid", -1)) == os.getpid():
                    state_path.unlink(missing_ok=True)
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
        _release_mutex(mutex_handle)
        log_stream.flush()


if __name__ == "__main__":
    raise SystemExit(main())
