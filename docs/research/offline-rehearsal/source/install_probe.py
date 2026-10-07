# ruff: noqa: S101, S603
# Verification assertions are deliberate; subprocess argv is constructed locally.
"""Run only installation and basic CLI probes inside a caller-created net namespace."""

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


if not __debug__:
    raise RuntimeError("Run this verification script without Python optimization")

stage = Path(sys.argv[1]).resolve()
uv = str(Path(sys.argv[2]).resolve())
python = str(Path(sys.argv[3]).resolve())
report_path = stage / "results/install-probe.json"
report = {
    "scope": "Local Linux x86_64 Python 3.12 CPU clean offline installation only",
    "model_inference_executed": False,
    "official_target_verified": False,
    "probe_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "network_namespace": str(Path("/proc/self/ns/net").readlink()),
    "routes": Path("/proc/net/route").read_text(),
    "commands": [],
}
env = dict(os.environ)
for name in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
    env.pop(name, None)
env.update(
    {
        "UV_OFFLINE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(stage / "empty-hf-cache"),
        "XDG_CACHE_HOME": str(stage / "empty-xdg-cache"),
        "PYTHONNOUSERSITE": "1",
    }
)
for directory in ("empty-hf-cache", "empty-xdg-cache", "temporary-cache"):
    (stage / directory).mkdir(exist_ok=False)
    assert not list((stage / directory).iterdir())
env["TMPDIR"] = str(stage / "temporary-cache")
report["explicit_environment"] = {
    k: env[k]
    for k in (
        "UV_OFFLINE",
        "UV_PYTHON_DOWNLOADS",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HOME",
        "XDG_CACHE_HOME",
        "PYTHONNOUSERSITE",
        "TMPDIR",
    )
}
report["main_namespace"] = sys.argv[4]
assert report["network_namespace"] != report["main_namespace"], (
    "Network namespace unchanged"
)
assert not (stage / "env").exists(), "Target environment must be new"


def run(command):
    started = datetime.datetime.now(datetime.UTC).isoformat()
    tick = time.monotonic()
    result = subprocess.run(
        command, cwd=stage, env=env, capture_output=True, text=True, check=False
    )
    record = {
        "command": command,
        "started_at_utc": started,
        "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - tick,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    report["commands"].append(record)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({"command": command[:4], "exit_code": result.returncode}), flush=True
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    return result


run(
    [
        python,
        "-I",
        "-c",
        """import json,socket
try:
    socket.create_connection(('1.1.1.1',443),timeout=2).close()
except OSError as exc:
    print(json.dumps({'external_tcp_failed':True,'exception':type(exc).__name__,'errno':exc.errno,'message':str(exc)}))
else:
    raise SystemExit('Network isolation failed: external TCP succeeded')
""",
    ]
)
run(
    [
        uv,
        "--no-config",
        "--offline",
        "--no-cache",
        "venv",
        "--no-python-downloads",
        "--python",
        python,
        str(stage / "env"),
    ]
)
target_python = str(stage / "env/bin/python")
run(
    [
        uv,
        "--no-config",
        "--offline",
        "--no-cache",
        "pip",
        "install",
        "--python",
        target_python,
        "--no-index",
        "--find-links",
        str(stage / "wheelhouse"),
        "--only-binary=:all:",
        "--require-hashes",
        "--strict",
        "-r",
        str(stage / "install.txt"),
    ]
)
run(
    [
        uv,
        "--no-config",
        "--offline",
        "--no-cache",
        "pip",
        "check",
        "--python",
        target_python,
    ]
)
run(
    [
        target_python,
        "-I",
        "-c",
        """import importlib.metadata,json,pathlib,sys,torch,deepfake_detection_archive
stage=pathlib.Path(sys.argv[1]).resolve()
package=pathlib.Path(deepfake_detection_archive.__file__).resolve()
assert package.is_relative_to(stage/'env'), package
assert 'site-packages' in package.parts, package
assert torch.__version__=='2.14.1+cpu', torch.__version__
assert torch.version.cuda is None
assert not torch.cuda.is_available()
print(json.dumps({'python':sys.version,'python_executable':sys.executable,'package_path':str(package),'torch_version':torch.__version__,'torch_cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'distribution_count':len(list(importlib.metadata.distributions()))}))
""",
        str(stage),
    ]
)
run([target_python, "-I", "-m", "deepfake_detection_archive", "--help"])
run(
    [
        uv,
        "--no-config",
        "--offline",
        "--no-cache",
        "pip",
        "list",
        "--python",
        target_python,
        "--format",
        "json",
    ]
)
report["passed"] = True
report["completed_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
report["remaining_temporary_cache_entries"] = [
    p.name for p in (stage / "temporary-cache").iterdir()
]
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
