# ruff: noqa: S101, S603
# Verification assertions are deliberate; subprocess argv is constructed locally.
"""Prepare verified wheels and run a clean CPU install in a network namespace."""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


if not __debug__:
    raise RuntimeError("Run this verification script without Python optimization")

ROOT = Path(sys.argv[1]).resolve()
DOCS = ROOT / "docs/research/offline-rehearsal"
STAGE = ROOT / "outputs/offline-rehearsal-cpu"
assert not STAGE.exists(), "Fresh rehearsal directory required"
for name in ("wheelhouse", "results", "scripts"):
    (STAGE / name).mkdir(parents=True, exist_ok=False)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_snapshot():
    paths = [ROOT / p for p in ("pyproject.toml", "uv.lock", "README.md")]
    paths += [
        p
        for p in (ROOT / "src").rglob("*")
        if p.is_file() and p.suffix in (".py", ".txt")
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def main_environment_snapshot():
    paths = [ROOT / ".venv/pyvenv.cfg"]
    site = ROOT / ".venv/lib/python3.12/site-packages"
    for pattern in ("*.dist-info/METADATA", "*.dist-info/RECORD"):
        paths.extend(site.glob(pattern))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


uv = shutil.which("uv")
python = shutil.which("python3")
assert uv is not None
assert python is not None
inventory = json.loads((DOCS / "cpu-wheelhouse.json").read_text())
assert sha256(ROOT / "uv.lock") == inventory["uv_lock_sha256"]
report = {
    "schema_version": 1,
    "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "scope": "Local Linux x86_64 Python 3.12 CPU clean offline install and CLI help only",
    "model_inference_executed": False,
    "official_target_verified": False,
    "source_before_build": source_snapshot(),
    "main_environment_before": main_environment_snapshot(),
    "prepare_script_sha256": sha256(Path(__file__)),
    "uv_path": uv,
    "uv_sha256": sha256(Path(uv).resolve()),
    "python_path": python,
    "python_binary_sha256": sha256(Path(python).resolve()),
    "main_namespace": str(Path("/proc/self/ns/net").readlink()),
    "commands": [],
}
report_path = STAGE / "results/preparation.json"


def run(command, *, environment=None):
    start = datetime.datetime.now(datetime.UTC).isoformat()
    tick = time.monotonic()
    result = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False
    )
    report["commands"].append(
        {
            "command": command,
            "started_at_utc": start,
            "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - tick,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({"command": command[:5], "exit_code": result.returncode}), flush=True
    )
    if result.returncode:
        print(result.stderr, flush=True)
        raise SystemExit(result.returncode)
    return result


for wheel in inventory["wheels"]:
    source = ROOT / "cache/wheelhouse/cpu-py312" / wheel["filename"]
    target = STAGE / "wheelhouse" / wheel["filename"]
    shutil.copyfile(source, target)
    assert target.stat().st_size == wheel["size"]
    assert sha256(target) == wheel["sha256"]
report["third_party_copied_count"] = len(inventory["wheels"])
report["third_party_copied_bytes"] = sum(w["size"] for w in inventory["wheels"])
env = dict(os.environ)
env["UV_CACHE_DIR"] = "/tmp/aigc-uv-cache"  # noqa: S108 -- existing project build cache
run(
    [
        uv,
        "--offline",
        "build",
        "--no-sources",
        "--wheel",
        "--out-dir",
        str(STAGE / "wheelhouse"),
    ],
    environment=env,
)
report["source_after_build"] = source_snapshot()
assert report["source_after_build"] == report["source_before_build"], (
    "Sources changed during build"
)
project_wheels = list((STAGE / "wheelhouse").glob("deepfake_detection_archive-*.whl"))
assert len(project_wheels) == 1
project = project_wheels[0]
assert project.name == "deepfake_detection_archive-0.1.0-py3-none-any.whl"
report["project_wheel"] = {
    "filename": project.name,
    "size": project.stat().st_size,
    "sha256": sha256(project),
}
requirements = (DOCS / "requirements-cpu-py312.txt").read_text()
requirements += (
    "deepfake-detection-archive==0.1.0 --hash=sha256:" + sha256(project) + "\n"
)
(STAGE / "install.txt").write_text(requirements)
report["install_requirements_sha256"] = sha256(STAGE / "install.txt")
shutil.copyfile(DOCS / "source/install_probe.py", STAGE / "scripts/install_probe.py")
report["probe_script_sha256"] = sha256(STAGE / "scripts/install_probe.py")
for executable in ("ffmpeg", "ffprobe"):
    path = shutil.which(executable)
    assert path is not None
    report[executable] = {
        "path": path,
        "resolved_path": str(Path(path).resolve()),
        "sha256": sha256(Path(path).resolve()),
    }
    run([path, "-version"])
    run(["ldd", path])
run(
    [
        "unshare",
        "--user",
        "--map-root-user",
        "--net",
        python,
        "-I",
        str(STAGE / "scripts/install_probe.py"),
        str(STAGE),
        uv,
        python,
        report["main_namespace"],
    ]
)
probe = json.loads((STAGE / "results/install-probe.json").read_text())
assert probe["passed"]
report["main_environment_after"] = main_environment_snapshot()
assert report["main_environment_after"] == report["main_environment_before"], (
    "Main environment metadata changed"
)
report["passed"] = True
report["finished_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print("CLEAN_OFFLINE_CPU_INSTALL_PASSED", flush=True)
