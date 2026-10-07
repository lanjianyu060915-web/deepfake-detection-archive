from __future__ import annotations

import importlib.util
from pathlib import Path
import socket

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "packaging_v0.3" / "src" / "app_launcher.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "packaging_v03_app_launcher_for_test",
        LAUNCHER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_port_keeps_free_preferred_port() -> None:
    launcher = _load_launcher()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        preferred = int(reservation.getsockname()[1])

    assert launcher._select_port(preferred) == preferred


def test_select_port_never_reuses_an_active_listener() -> None:
    launcher = _load_launcher()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        occupied = int(listener.getsockname()[1])

        selected = launcher._select_port(occupied)

    assert selected > 0
    assert selected != occupied


def test_select_port_zero_returns_a_reusable_nonzero_port() -> None:
    launcher = _load_launcher()

    selected = launcher._select_port(0)

    assert selected > 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", selected))


BUFFALO_L_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)


def _preflight_message(tmp_path: Path, identity_section: dict) -> str:
    """跑一次启动前资源自检并返回失败文本；其它模块故意留空。

    身份模块声明的每个文件都会真的建出来，这样断言「消息里没有身份模块的
    报错」就等于断言该形态的自检通过。
    """

    launcher = _load_launcher()
    app_root = tmp_path / "app"
    data_root = tmp_path / "data"
    app_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    for key in ("executable", "python", "runner"):
        value = identity_section.get(key)
        if isinstance(value, str) and value:
            target = launcher._expand_path(value, app_root, data_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
    for key in ("library",):
        value = identity_section.get(key)
        if isinstance(value, str) and value:
            launcher._expand_path(value, app_root, data_root).mkdir(
                parents=True, exist_ok=True
            )
    model_root = identity_section.get("model_root")
    if isinstance(model_root, str) and model_root:
        model_dir = (
            launcher._expand_path(model_root, app_root, data_root)
            / "models"
            / "buffalo_l"
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in BUFFALO_L_FILES:
            (model_dir / name).write_bytes(b"x")

    config = {
        "schema_version": "3.0",
        "build_flavor": "formal",
        "features": {"mock_enabled": False, "real_modules_default": True},
        "resources": {"verify_manifest_on_start": False},
        "runtime": {
            "tasks_root": "{DATA_ROOT}/tasks",
            "inbox_root": "{DATA_ROOT}/inbox",
            "identity_cache": "{DATA_ROOT}/identity_cache",
            "persistent_services_root": "{DATA_ROOT}/services",
        },
        "deepfake": {},
        "identity": identity_section,
        "source": {},
    }
    with pytest.raises(RuntimeError) as excinfo:
        launcher._preflight(config, app_root, data_root)
    return str(excinfo.value)


def test_preflight_accepts_packaged_identity_executable(tmp_path: Path) -> None:
    message = _preflight_message(
        tmp_path,
        {
            "executable": "{APP_ROOT}/workers/identity/identity_worker.exe",
            "library": "{DATA_ROOT}/identity_library",
            "model_root": "{APP_ROOT}/model_runtime",
        },
    )
    assert "身份候选检索引擎" not in message
    assert "InsightFace buffalo_l" not in message


def test_preflight_accepts_python_only_identity_engine(tmp_path: Path) -> None:
    """2.0 升级后身份模块是 python + runner + 常驻 worker，自检不得再要 exe。"""

    message = _preflight_message(
        tmp_path,
        {
            "python": "{APP_ROOT}/runtime/python.exe",
            "runner": "{APP_ROOT}/modules/identity_search_v2/run_identity_search_v2.py",
            "persistent_worker": "{APP_ROOT}/modules/identity_search_v2/persistent_worker_v2.py",
            "library": "{DATA_ROOT}/identity_library",
            "model_root": "{APP_ROOT}/model_runtime",
        },
    )
    assert "身份候选检索引擎" not in message
    assert "InsightFace buffalo_l" not in message


def test_preflight_reports_missing_identity_engine(tmp_path: Path) -> None:
    message = _preflight_message(
        tmp_path,
        {
            "library": "{DATA_ROOT}/identity_library",
            "model_root": "{APP_ROOT}/model_runtime",
        },
    )
    assert "身份候选检索引擎 Python未配置" in message
    assert "身份候选检索引擎适配器未配置" in message
