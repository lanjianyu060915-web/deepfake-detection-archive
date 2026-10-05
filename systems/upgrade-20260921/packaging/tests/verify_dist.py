from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


EXPECTED_MODELS = {
    "1k3d68.onnx": (143607619, "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc"),
    "2d106det.onnx": (5030888, "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"),
    "det_10g.onnx": (16923827, "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
    "genderage.onnx": (1322532, "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"),
    "w600k_r50.onnx": (174383860, "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_help(executable: Path) -> None:
    completed = subprocess.run(
        [str(executable), "--help"],
        cwd=executable.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        env={**dict(__import__("os").environ), "PYINSTALLER_RESET_ENVIRONMENT": "1"},
    )
    if completed.returncode != 0 or "--query" not in completed.stdout:
        raise RuntimeError(
            f"worker --help failed: {executable}\n"
            f"exit={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-root", required=True)
    args = parser.parse_args()
    root = Path(args.app_root).resolve()

    required_files = [
        root / "换脸寻源双搜索系统.exe",
        root / "config.json",
        root / "workers" / "identity" / "identity_search.exe",
        root / "workers" / "source" / "source_search.exe",
        root / "samples" / "queries" / "query_001.jpg",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError("distribution files missing:\n" + "\n".join(missing))

    config_text = (root / "config.json").read_text(encoding="utf-8")
    config = json.loads(config_text)
    if "D:\\" in config_text or "C:\\Users\\" in config_text:
        raise RuntimeError("packaged config contains a development-machine absolute path")
    if "{DATA_ROOT}" not in config_text or "{APP_ROOT}" not in config_text:
        raise RuntimeError("packaged config does not separate app and data roots")
    if config["identity"].get("python") or config["source"].get("python"):
        raise RuntimeError("packaged workers still depend on Python configuration")

    identity_files = [path for path in (root / "data" / "identity_library").rglob("*") if path.is_file()]
    source_files = [path for path in (root / "data" / "source_library").rglob("*") if path.is_file()]
    sample_files = [
        path for path in (root / "samples" / "queries").glob("query_*.jpg") if path.is_file()
    ]
    if len(identity_files) != 45:
        raise RuntimeError(f"identity library count mismatch: {len(identity_files)}")
    if len(source_files) != 203:
        raise RuntimeError(f"source library count mismatch: {len(source_files)}")
    if len(sample_files) != 10:
        raise RuntimeError(f"sample query count mismatch: {len(sample_files)}")

    model_dir = root / "model_runtime" / "models" / "buffalo_l"
    actual_names = {path.name for path in model_dir.iterdir() if path.is_file()}
    if actual_names != set(EXPECTED_MODELS):
        raise RuntimeError(f"model inventory mismatch: {sorted(actual_names)}")
    for name, (size, sha256) in EXPECTED_MODELS.items():
        path = model_dir / name
        if path.stat().st_size != size or _sha256(path) != sha256:
            raise RuntimeError(f"model size/hash mismatch: {path}")

    _run_help(required_files[2])
    _run_help(required_files[3])
    print(
        json.dumps(
            {
                "status": "PASS",
                "app_root": str(root),
                "identity_images": len(identity_files),
                "source_images": len(source_files),
                "samples": len(sample_files),
                "models": len(EXPECTED_MODELS),
                "worker_help": "PASS",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
