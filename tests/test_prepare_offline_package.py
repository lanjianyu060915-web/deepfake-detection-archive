"""Exercise the package builder without copying the multi-GB GPU runtime."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from deepfake_detection_archive.data.manifest import SampleRecord, load_manifest


@pytest.fixture
def builder_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools/prepare_offline_package.py"
    spec = importlib.util.spec_from_file_location("prepare_offline_package", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", ["../escape.wav", "/escape.wav", "a\\b.wav"])
def test_read_manifest_rejects_unsafe_paths(
    tmp_path: Path, builder_module: ModuleType, path: str
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    row = SampleRecord("r", "r.wav", "audio", 0, "test", split="dev").to_dict()
    row["path"] = path
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="relative POSIX"):
        builder_module.read_manifest(manifest)


def test_holdout_rejected_before_publish(
    tmp_path: Path, builder_module: ModuleType
) -> None:
    row = SampleRecord("r", "r.wav", "audio", 0, "test", split="holdout")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row.to_dict()) + "\n")
    with pytest.raises(ValueError, match="holdout"):
        builder_module.read_manifest(manifest)
    builder = builder_module.Builder(tmp_path / "package", tmp_path, 1)
    with pytest.raises(ValueError, match="holdout"):
        builder.publish("audio", [row], tmp_path)
    assert not (builder.package / "media").exists()


def test_builder_relocates_isolated_runtime_and_source(
    tmp_path: Path, builder_module: ModuleType
) -> None:
    build = tmp_path / "build"
    build.mkdir()
    runtime = build / "installed"
    runtime.mkdir()
    (runtime / "tiny_model.py").write_text("def score(value): return value / 10\n")
    for name in ("poison.pth", "__editable___mapping.py", "unused.pyc"):
        (runtime / name).write_text("raise RuntimeError('external environment')\n")
    source = build / "src" / "deepfake_detection_archive"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "__main__.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "import tiny_model\n"
        "assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "assert sys.flags.isolated and sys.flags.no_site\n"
        "config = json.loads(Path(sys.argv[2]).read_text())\n"
        "print(json.dumps({'score': tiny_model.score(config['value']), "
        "'model': tiny_model.__file__, 'source': __file__}))\n"
    )
    lock = build / "uv.lock"
    lock.write_text("version = 1\n")
    media = build / "shared"
    media.mkdir()
    (media / "r.wav").write_bytes(b"fixture")
    builder = builder_module.Builder(build / "package", media, 1)
    builder.publish(
        "audio", [SampleRecord("r", "r.wav", "audio", 0, "test", split="dev")], media
    )
    builder.copy_dependencies(lock, runtime, source.parent)
    configs = build / "configs"
    configs.mkdir()
    (configs / "sample.json").write_text('{"value": 7}')
    builder.copy_configs(configs)
    moved = tmp_path / "moved"
    shutil.move(builder.package, moved)
    shutil.rmtree(build)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(moved / "run.py"),
            "infer",
            str(moved / "configs/sample.json"),
        ],
        cwd="/",
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={"PYTHONPATH": "/missing/original/source"},
    )
    output = json.loads(result.stdout)
    assert output["score"] == 0.7
    assert Path(output["model"]).is_relative_to(moved / "runtime")
    assert Path(output["source"]).is_relative_to(moved / "src")
    assert not list(moved.rglob("*.pth"))
    assert not list(moved.rglob("__editable__*"))
    assert (
        load_manifest(moved / "manifests/sample-audio.jsonl")[0].path == "audio/r.wav"
    )

    # The same generated launcher also loads the real bundled CLI from cwd=/,
    # with no installed project or third-party packages on the Python path.
    shutil.rmtree(moved / "src")
    repository = Path(__file__).resolve().parents[1]
    shutil.copytree(repository / "src", moved / "src")
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(moved / "run.py"), "--help"],
        cwd="/",
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "infer" in result.stdout


def _distribution(site: Path, name: str, requirements: tuple[str, ...] = ()) -> None:
    info = site / f"{name}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Name: {name}\nVersion: 1.0\n"
        + "".join(f"Requires-Dist: {value}\n" for value in requirements)
    )
    (info / "LICENSE").write_text(f"{name} license")
    (site / f"{name}.py").write_text(f"# {name}\n")
    files = [
        f"{name}.py",
        f"{info.name}/METADATA",
        f"{info.name}/LICENSE",
        f"{info.name}/RECORD",
    ]
    (info / "RECORD").write_text("".join(f"{file},,\n" for file in files))


def test_runtime_closure_markers_extras_and_wheel_files(
    tmp_path: Path, builder_module: ModuleType
) -> None:
    site = tmp_path / "installed"
    _distribution(
        site,
        "model",
        (
            "backend[fast]>=1",
            "native; sys_platform == 'linux'",
            "missing; python_version < '3'",
            "devtools; extra == 'dev'",
        ),
    )
    _distribution(site, "backend", ("accelerator; extra == 'fast'", "model"))
    for name in ("native", "accelerator", "devtools", "unrelated"):
        _distribution(site, name)
    native_lib = site / "native.libs/libnative.so"
    native_lib.parent.mkdir()
    native_lib.write_bytes(b"native binary")
    record = site / "native-1.0.dist-info/RECORD"
    with record.open("a") as stream:
        stream.write("native.libs/libnative.so,,\n../../../bin/native-cli,,\n")
    (site / "poison.pth").write_text("raise Exception()")
    target = tmp_path / "runtime"
    builder_module.copy_runtime_dependencies(site, target, ("model",))
    assert {path.stem for path in target.glob("*.py")} == {
        "model",
        "backend",
        "accelerator",
        "native",
    }
    assert (target / "native.libs/libnative.so").read_bytes() == b"native binary"
    assert (target / "backend-1.0.dist-info/LICENSE").read_text() == "backend license"
    assert not (target / "poison.pth").exists()
    assert "devtools" in builder_module.runtime_dependencies(site, ("model[dev]",))


@pytest.mark.parametrize("requirement", ["missing", "model>=2"])
def test_runtime_closure_rejects_incomplete_environment(
    tmp_path: Path, builder_module: ModuleType, requirement: str
) -> None:
    _distribution(tmp_path, "model")
    with pytest.raises(ValueError, match=r"missing installed|violates"):
        builder_module.runtime_dependencies(tmp_path, (requirement,))
