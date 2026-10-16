"""Development entrypoints must use this checkout without modifying shared envs."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_env(tmp_path: Path) -> dict[str, str]:
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    shadow = tmp_path / "other-checkout" / "src"
    package = shadow / "deepfake_detection_archive"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import importlib.util, json, os, sys\n"
        "from pathlib import Path\n"
        "spec = importlib.util.find_spec('deepfake_detection_archive')\n"
        "with Path(os.environ['UV_TEST_LOG']).open('a') as log:\n"
        "    log.write(json.dumps({'args': sys.argv[1:], "
        "'environment': os.environ['UV_PROJECT_ENVIRONMENT'], "
        "'source': spec.origin}) + '\\n')\n"
        "sys.exit(int(os.environ.get('UV_TEST_EXIT', '0')))\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env = dict(os.environ)
    env.pop("UV_NO_INSTALL_PROJECT", None)
    env.pop("DEEPFAKE_BACKEND", None)
    env.update(
        UV_PROJECT_ENVIRONMENT=str(venv),
        PYTHONPATH=str(shadow),
        PATH=f"{bin_dir}{os.pathsep}{env['PATH']}",
        UV_TEST_LOG=str(tmp_path / "uv.jsonl"),
    )
    return env


def test_devctl_run_uses_current_checkout_from_external_cwd(
    isolated_env: dict[str, str], tmp_path: Path
) -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/devctl"),
            "run",
            "python",
            "-c",
            "import deepfake_detection_archive as package; print(package.__file__)",
        ],
        cwd=tmp_path,
        env=isolated_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(result.stdout.strip()) == (
        ROOT / "src/deepfake_detection_archive/__init__.py"
    )


def test_devctl_sync_uses_explicit_environment_without_installing_project(
    isolated_env: dict[str, str], tmp_path: Path
) -> None:
    subprocess.run(
        ["bash", str(ROOT / "scripts/devctl"), "sync"],
        cwd=tmp_path,
        env=isolated_env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = Path(isolated_env["UV_TEST_LOG"]).read_text().splitlines()
    assert len(calls) == 1
    call = json.loads(calls[0])
    assert call["environment"] == isolated_env["UV_PROJECT_ENVIRONMENT"]
    assert call["args"] == [
        "sync",
        "--locked",
        "--no-install-project",
        "--python",
        "3.12",
        "--extra",
        "cu132",
        "--group",
        "dev",
        "--group",
        "research",
        "--group",
        "media",
    ]
    assert (
        Path(isolated_env["UV_PROJECT_ENVIRONMENT"]) / ".deepfake-sync.lock"
    ).is_file()


def test_devctl_check_delegates_and_propagates_failure(
    isolated_env: dict[str, str], tmp_path: Path
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "tools").mkdir()
    shutil.copyfile(ROOT / "scripts/devctl", checkout / "scripts/devctl")
    (checkout / "tools/check_project.sh").write_text(
        "#!/usr/bin/env bash\nprintf 'delegated\\n'\nexit 19\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(checkout / "scripts/devctl"), "check"],
        check=False,
        cwd=tmp_path,
        env=isolated_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 19
    assert result.stdout == "delegated\n"


def test_standard_check_uses_current_checkout_and_stops_on_failure(
    isolated_env: dict[str, str], tmp_path: Path
) -> None:
    isolated_env["UV_TEST_EXIT"] = "17"
    result = subprocess.run(
        ["bash", str(ROOT / "tools/check_project.sh")],
        check=False,
        cwd=tmp_path,
        env=isolated_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 17
    calls = Path(isolated_env["UV_TEST_LOG"]).read_text().splitlines()
    assert len(calls) == 1
    assert json.loads(calls[0])["source"] == str(
        ROOT / "src/deepfake_detection_archive/__init__.py"
    )


@pytest.mark.parametrize("configured", [True, False])
def test_devctl_preserves_caller_toolchain_environment(
    isolated_env: dict[str, str], tmp_path: Path, configured: bool
) -> None:
    values = {
        "CUDA_HOME": "/custom/cuda",
        "CUDACXX": "/custom/compiler/nvcc",
        "UV_PYTHON_INSTALL_DIR": str(tmp_path / "python"),
        "UV_NO_MANAGED_PYTHON": "1",
    }
    for key, value in values.items():
        isolated_env.pop(key, None)
        if configured:
            isolated_env[key] = value
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/devctl"),
            "run",
            "python",
            "-c",
            "import json, os; print(json.dumps(dict(os.environ)))",
        ],
        env=isolated_env,
        capture_output=True,
        text=True,
        check=True,
    )
    actual = json.loads(result.stdout)
    assert {key: actual.get(key) for key in values} == {
        key: value if configured else None for key, value in values.items()
    }


@pytest.mark.parametrize("backend", ["cpu", "cu126"])
def test_devctl_local_sync_can_initialize_explicit_environment(
    isolated_env: dict[str, str], tmp_path: Path, backend: str
) -> None:
    target = tmp_path / "new-env"
    isolated_env.update(DEEPFAKE_BACKEND=backend, UV_PROJECT_ENVIRONMENT=str(target))
    subprocess.run(
        ["bash", str(ROOT / "scripts/devctl"), "sync"],
        env=isolated_env,
        capture_output=True,
        text=True,
        check=True,
    )
    call = json.loads(Path(isolated_env["UV_TEST_LOG"]).read_text())
    assert call["environment"] == str(target)
    assert call["args"][call["args"].index("--extra") + 1] == backend


@pytest.mark.parametrize("backend", ["cpu", "cu126"])
def test_devctl_local_backend_defaults_to_checkout_environment(
    isolated_env: dict[str, str], tmp_path: Path, backend: str
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/devctl", checkout / "scripts/devctl")
    isolated_env.pop("UV_PROJECT_ENVIRONMENT")
    isolated_env["DEEPFAKE_BACKEND"] = backend
    subprocess.run(
        ["bash", str(checkout / "scripts/devctl"), "sync"],
        cwd=tmp_path,
        env=isolated_env,
        capture_output=True,
        text=True,
        check=True,
    )
    call = json.loads(Path(isolated_env["UV_TEST_LOG"]).read_text())
    assert call["environment"] == str(checkout / f".venv-{backend}")


def test_devctl_cpu_doctor_without_cuda(
    isolated_env: dict[str, str], tmp_path: Path
) -> None:
    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    (fake_modules / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        "__version__ = 'test+cpu'\n"
        "version = SimpleNamespace(cuda=None)\n"
        "cuda = SimpleNamespace(is_available=lambda: False)\n",
        encoding="utf-8",
    )
    isolated_env.update(DEEPFAKE_BACKEND="cpu", PYTHONPATH=str(fake_modules))
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/devctl"), "doctor"],
        env=isolated_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "CUDA available: False" in result.stdout
    assert "PyTorch: test+cpu" in result.stdout


def test_envrc_leaves_toolchain_unset_without_local_configuration(
    isolated_env: dict[str, str], tmp_path: Path
) -> None:
    keys = (
        "CUDA_HOME",
        "CUDACXX",
        "UV_PYTHON_INSTALL_DIR",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
    )
    for key in keys:
        isolated_env.pop(key, None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'PATH_add() { export PATH="$1:$PATH"; }; '
            'source "$1"; python -c '
            "'import json, os; print(json.dumps(dict(os.environ)))'",
            "bash",
            str(ROOT / ".envrc"),
        ],
        cwd=tmp_path,
        env=isolated_env,
        capture_output=True,
        text=True,
        check=True,
    )
    actual = json.loads(result.stdout)
    assert all(key not in actual for key in keys)


def test_envrc_activates_explicit_local_environment(
    isolated_env: dict[str, str], tmp_path: Path
) -> None:
    target = isolated_env["UV_PROJECT_ENVIRONMENT"]
    (tmp_path / ".envrc.local").write_text(
        f'export UV_PROJECT_ENVIRONMENT="{target}"\n', encoding="utf-8"
    )
    isolated_env.pop("UV_PROJECT_ENVIRONMENT")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source_env() { source "$1"; }; '
            'PATH_add() { export PATH="$1:$PATH"; }; '
            'source "$1"; printf "%s\\n" "$VIRTUAL_ENV"; command -v python',
            "bash",
            str(ROOT / ".envrc"),
        ],
        cwd=tmp_path,
        env=isolated_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [target, str(Path(target) / "bin/python")]


@pytest.mark.parametrize("backend", ["cpu", "cu126"])
@pytest.mark.parametrize("target_kind", ["direct", "trailing_slash", "symlink"])
def test_local_sync_rejects_shared_cuda_environment(
    isolated_env: dict[str, str], tmp_path: Path, target_kind: str, backend: str
) -> None:
    shared = "/opt/venvs/deepfake-cu132"
    target = shared
    if target_kind == "trailing_slash":
        target += "/"
    elif target_kind == "symlink":
        link = tmp_path / "shared-alias"
        link.symlink_to(shared, target_is_directory=True)
        target = str(link)
    isolated_env.update(DEEPFAKE_BACKEND=backend, UV_PROJECT_ENVIRONMENT=target)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/devctl"), "sync"],
        env=isolated_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3
    assert "独立" in result.stderr
    assert not Path(isolated_env["UV_TEST_LOG"]).exists()
