from __future__ import annotations

import importlib.util
from pathlib import Path
import socket


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
