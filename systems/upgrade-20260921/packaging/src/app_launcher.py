from __future__ import annotations

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


PRODUCT_DATA_DIR = "换脸寻源双搜索系统"
STATE_FILE = "launcher_state.json"
HEALTH_PATH = "/_stcore/health"


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
    return base / PRODUCT_DATA_DIR


def _show_error(title: str, message: str) -> None:
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
            return
        except Exception:
            pass
    print(f"{title}: {message}", file=sys.stderr)


def _configure_logging(data_root: Path):
    log_dir = data_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "launcher.log"
    stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = stream
    sys.stderr = stream
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launcher pid={os.getpid()}")
    return stream


def _read_config(config_path: Path) -> dict:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取安装配置：{config_path}\n{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"安装配置格式错误：{config_path}")
    return payload


def _expand_path(value: str, app_root: Path, data_root: Path) -> Path:
    expanded = os.path.expandvars(str(value))
    expanded = expanded.replace("{APP_ROOT}", str(app_root))
    expanded = expanded.replace("{DATA_ROOT}", str(data_root))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = app_root / path
    return path.resolve(strict=False)


def _preflight(config: dict, app_root: Path, data_root: Path) -> None:
    required_files = [
        ("身份搜索引擎", config.get("identity", {}).get("executable")),
        ("底图搜索引擎", config.get("source", {}).get("executable")),
    ]
    required_dirs = [
        ("身份图库", config.get("identity", {}).get("library")),
        ("底图图库", config.get("source", {}).get("library")),
    ]
    errors: list[str] = []
    for label, value in required_files:
        if not value:
            errors.append(f"{label}未配置")
            continue
        path = _expand_path(value, app_root, data_root)
        if not path.is_file():
            errors.append(f"{label}不存在：{path}")
    for label, value in required_dirs:
        if not value:
            errors.append(f"{label}未配置")
            continue
        path = _expand_path(value, app_root, data_root)
        if not path.is_dir():
            errors.append(f"{label}不存在：{path}")

    model_root_value = config.get("identity", {}).get("model_root")
    if model_root_value:
        model_dir = _expand_path(model_root_value, app_root, data_root) / "models" / "buffalo_l"
        expected_models = {
            "1k3d68.onnx",
            "2d106det.onnx",
            "det_10g.onnx",
            "genderage.onnx",
            "w600k_r50.onnx",
        }
        actual_models = {path.name for path in model_dir.iterdir() if path.is_file()} if model_dir.is_dir() else set()
        if actual_models != expected_models:
            errors.append(f"InsightFace 模型清单不完整：{model_dir}")

    try:
        for value in config.get("runtime", {}).values():
            path = _expand_path(value, app_root, data_root)
            path.mkdir(parents=True, exist_ok=True)
        probe = data_root / ".write_probe"
        probe.write_text("ok", encoding="ascii")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        errors.append(f"用户数据目录不可写：{data_root}；{exc}")

    if errors:
        raise RuntimeError("启动前检查失败：\n\n" + "\n".join(f"• {item}" for item in errors))


def _health_ok(port: int, timeout: float = 0.75) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{HEALTH_PATH}", timeout=timeout
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _open_existing(state_path: Path) -> bool:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        port = int(state["port"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if not _health_ok(port):
        return False
    webbrowser.open(f"http://127.0.0.1:{port}", new=2)
    return True


def _select_port(preferred: int) -> int:
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return int(probe.getsockname()[1])
    raise RuntimeError("无法找到可用的本地端口。")


def _wait_and_open(port: int, log_stream) -> None:
    deadline = time.monotonic() + 45.0
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
        "换脸寻源双搜索系统",
        "本地服务未能在 45 秒内启动。请查看：\n"
        + str(_data_root() / "logs" / "launcher.log"),
    )


def main() -> int:
    app_root = _install_root()
    bundle_root = _bundle_root()
    data_root = _data_root()
    data_root.mkdir(parents=True, exist_ok=True)
    log_stream = _configure_logging(data_root)
    state_path = data_root / STATE_FILE

    if _open_existing(state_path):
        print("existing server reused")
        return 0

    if getattr(sys, "frozen", False):
        config_path = app_root / "config.json"
    else:
        config_path = app_root / "packaging" / "config.packaged.json"
    app_script = bundle_root / "frontend" / "app.py"

    try:
        config = _read_config(config_path)
        if not app_script.is_file():
            raise RuntimeError(f"前端入口不存在：{app_script}")
        _preflight(config, app_root, data_root)
        preferred_port = int(config.get("frontend", {}).get("port", 8766))
        port = _select_port(preferred_port)

        state_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "port": port,
                    "url": f"http://127.0.0.1:{port}",
                    "app_root": str(app_root),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.environ["FSTS_CONFIG_PATH"] = str(config_path)
        os.environ["FSTS_LAUNCHER_STATE"] = str(state_path)
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

        opener = threading.Thread(
            target=_wait_and_open,
            args=(port, log_stream),
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

        result = streamlit_cli.main()
        return int(result or 0)
    except Exception as exc:
        traceback.print_exc(file=log_stream)
        _show_error(
            "换脸寻源双搜索系统启动失败",
            f"{exc}\n\n详细日志：\n{data_root / 'logs' / 'launcher.log'}",
        )
        return 2
    finally:
        try:
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if int(state.get("pid", -1)) == os.getpid():
                    state_path.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        log_stream.flush()


if __name__ == "__main__":
    raise SystemExit(main())
