"""Release layout and relocatable launcher without copying the GPU environment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from deepfake_detection_archive.data.manifest import load_manifest


@pytest.fixture
def release_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "build_release", tools / "build_release.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bundled_media", [False, True])
def test_build_layout_and_launcher_after_relocation(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_module: ModuleType,
    bundled_media: bool,
) -> None:
    repo = tmp_path / "repo"
    for directory in ("docs", "configs/release", "src/demo"):
        (repo / directory).mkdir(parents=True)
    for filename in (
        "uv.lock",
        "docs/release.md",
        "docs/release-licenses.md",
        "docs/license.txt",
        "configs/release/models.json",
    ):
        (repo / filename).write_text(filename)
    (repo / "src/demo/__init__.py").write_text("value = 1\n")
    (repo / "configs/release/licenses.json").write_text(
        json.dumps({"upstream/license.txt": "docs/license.txt"})
    )
    runtime = tmp_path / "site-packages"
    (runtime / "torch-2.0.dist-info").mkdir(parents=True)
    (runtime / "torch-2.0.dist-info/METADATA").write_text("Name: torch\nVersion: 2.0\n")
    (runtime / "torch-2.0.dist-info/LICENSE").write_text("license")
    (runtime / "torch-2.0.dist-info/RECORD").write_text(
        "torch-2.0.dist-info/METADATA,,\n"
        "torch-2.0.dist-info/LICENSE,,\n"
        "torch-2.0.dist-info/RECORD,,\n"
    )
    monkeypatch.setattr(release_module, "RUNTIME_ROOTS", ("torch",))
    python_home = tmp_path / "python"
    (python_home / "bin").mkdir(parents=True)
    interpreter = python_home / "bin/python3.12"
    interpreter.write_text(
        '#!/bin/sh\nif [ "$3" = "-c" ]; then\n'
        '  echo \'{"version":"fixture","platform":"linux","machine":"x86_64"}\'\n'
        'else\n  printf "%s\\n" "$PWD" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$@"\nfi\n'
    )
    interpreter.chmod(0o755)
    (python_home / "bin/python").symlink_to("python3.12")
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fixture weights")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    weights.with_suffix(".pt.json").write_text(json.dumps({"sha256": digest}))
    for modality, filename in (("audio", "audio-environment"), ("video", "video")):
        (repo / f"configs/release/{filename}.json").write_text(
            json.dumps({"modality": modality, "checkpoint_sha256": digest})
        )
    monkeypatch.setattr(release_module, "REPOSITORY", repo)
    monkeypatch.setattr(release_module, "generate_examples", lambda output: None)
    monkeypatch.setattr(
        release_module, "checkpoint_ast_frontend", lambda path: "numpy_spectrogram_v1"
    )
    output = tmp_path / "build"
    args = (output, runtime, python_home, weights, weights, repo / "src")
    media = make_media_tools(tmp_path / "media-tools") if bundled_media else None
    release_module.build_release(*args, version="0.1.1", media_tools=media)
    assert (output / "configs/models.json").read_text() == "configs/release/models.json"
    with pytest.raises(FileExistsError):
        release_module.build_release(*args, version="0.1.1")
    assert (output / "configs/models.json").read_text() == "configs/release/models.json"
    assert (output / "licenses/upstream/license.txt").read_text() == "docs/license.txt"
    assert (output / "runtime/site-packages/torch-2.0.dist-info/LICENSE").is_file()
    assert (output / "runtime/python/bin/python").readlink() == Path("python3.12")
    metadata = json.loads((output / "release.json").read_text())
    if media is not None:
        assert metadata["media_tools"] == json.loads((media / "build.json").read_text())
        assert (output / "runtime/media-tools/build.sh").is_file()
        assert "runtime/media-tools/bin" in (output / "deps/README.md").read_text()
    else:
        assert "media_tools" not in metadata
        assert "由目标机提供" in (output / "deps/README.md").read_text()
    assert metadata["ast_frontend"] == "numpy_spectrogram_v1"
    assert metadata["torch"] == "2.0"
    assert metadata["version"] == "0.1.1"
    assert metadata["runtime_dependencies"] == {"torch": "2.0"}
    assert metadata["model_scope"]["audio"] == "environmental_audio_research"
    assert metadata["general_audio_ready"] is False
    assert metadata["competition_use"] == "unconfirmed"
    assert not (output / "configs/audio.json").exists()
    assert (output / "configs/video.json").is_file()
    assert metadata["checkpoint_sha256"] == {"audio": digest, "video": digest}
    assert (
        json.loads((output / "configs/audio-environment.json").read_text())[
            "checkpoint_sha256"
        ]
        == digest
    )
    moved = tmp_path / "moved package"
    shutil.move(output, moved)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    result = subprocess.run(
        [str(moved / "run.sh"), "predict", "--input", "space in path"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        str(tmp_path),
        "4",
        "4",
        "-I",
        "-S",
        str(moved / "run.py"),
        "predict",
        "--input",
        "space in path",
    ]


def test_generated_examples_decode_with_expected_media_properties(
    tmp_path: Path, release_module: ModuleType
) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required for generated release examples")
    release_module.generate_examples(tmp_path)
    for modality in ("audio", "video"):
        rows = load_manifest(tmp_path / f"manifests/input-{modality}.jsonl")
        assert len(rows) == 2
        assert all(row.label is None and row.modality == modality for row in rows)
        assert all((tmp_path / "media" / modality / row.path).is_file() for row in rows)
    expected = {
        "audio/stereo-44100.wav": ("audio", "44100", 2),
        "audio/mono-16000.flac": ("audio", "16000", 1),
        "video/testsrc.mp4": ("video", None, None),
        "video/blue.mp4": ("video", None, None),
    }
    for name, (kind, rate, channels) in expected.items():
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(tmp_path / "media" / name),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        metadata = json.loads(result.stdout)
        stream = metadata["streams"][0]
        assert stream["codec_type"] == kind
        assert float(metadata["format"]["duration"]) == pytest.approx(2, abs=0.01)
        if kind == "audio":
            assert stream["sample_rate"] == rate
            assert stream["channels"] == channels


@pytest.mark.parametrize(
    "frontend", [None, "numpy_spectrogram_v1", "torchaudio_kaldi_v1"]
)
def test_release_frontend_comes_from_checkpoint(
    tmp_path: Path, release_module: ModuleType, frontend: str | None
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "ast.pt"
    config: dict[str, object] = {"backbone_config": {}}
    if frontend is not None:
        config["frontend"] = frontend
    torch.save({"config": config}, path)
    assert release_module.checkpoint_ast_frontend(path) == (
        frontend or "numpy_spectrogram_v1"
    )


def make_media_tools(root: Path) -> Path:
    for directory in ("bin", "source", "licenses"):
        (root / directory).mkdir(parents=True)
    tools = {}
    for name in ("ffmpeg", "ffprobe"):
        path = root / "bin" / name
        path.write_bytes(b"#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        tools[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    for filename in (
        "source/ffmpeg-n8.1.3-git.tar.gz",
        "licenses/LICENSE.md",
        "licenses/COPYING.LGPLv2.1",
        "build.sh",
    ):
        (root / filename).write_text("fixture")
    (root / "build.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "version": "8.1.3",
                "tools": tools,
                "source": {
                    "url": "https://api.github.com/repos/FFmpeg/FFmpeg/tarball/n8.1.3",
                    "archive": "source/ffmpeg-n8.1.3-git.tar.gz",
                    "sha256": hashlib.sha256(b"fixture").hexdigest(),
                },
                "configure": [],
                "platform": "linux-x86_64",
                "license": "LGPL-2.1-or-later",
                "limitations": [],
            }
        )
    )
    return root


@pytest.mark.parametrize("failure", ["missing", "hash", "executable"])
@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
def test_release_rejects_invalid_media_tools_before_build(
    tmp_path: Path, release_module: ModuleType, failure: str, tool: str
) -> None:
    media = make_media_tools(tmp_path / "media-tools")
    binary = media / "bin" / tool
    if failure == "missing":
        binary.unlink()
    elif failure == "hash":
        binary.write_bytes(b"x" * binary.stat().st_size)
    else:
        binary.chmod(0o644)
    output = tmp_path / "release"
    with pytest.raises(ValueError, match=tool):
        release_module.build_release(
            output,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            version="test",
            media_tools=media,
        )
    assert not output.exists()


@pytest.mark.parametrize("media_mode", ["host", "bundle", "explicit"])
def test_real_launcher_resolves_media_after_relocation(
    tmp_path: Path, release_module: ModuleType, media_mode: str
) -> None:
    import prepare_offline_package  # noqa: PLC0415

    package = tmp_path / "package"
    project = package / "src/deepfake_detection_archive"
    project.mkdir(parents=True)
    (project / "__init__.py").write_text("")
    shutil.copyfile(
        Path(__file__).resolve().parents[1]
        / "src/deepfake_detection_archive/media_tools.py",
        project / "media_tools.py",
    )
    (project / "__main__.py").write_text(
        "import json, shutil, sys\n"
        "print(json.dumps({'tools': [shutil.which('ffmpeg'), "
        "shutil.which('ffprobe')], 'args': sys.argv[1:]}))\n"
    )
    (package / "run.py").write_text(prepare_offline_package.LAUNCHER)
    (package / "run.sh").write_text(release_module.RUN_SH)
    interpreter = package / "runtime/python/bin/python3.12"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    if media_mode != "host":
        make_media_tools(package / "runtime/media-tools")
    moved = tmp_path / "moved package"
    shutil.move(package, moved)
    host_bin = tmp_path / "host-bin"
    host_bin.mkdir()
    host_bin.joinpath("dirname").symlink_to("/usr/bin/dirname")
    env = {"PATH": str(host_bin)}
    expected: list[str | None] = [None, None]
    if media_mode != "host":
        selected = moved / "runtime/media-tools/bin"
        if media_mode == "explicit":
            selected = make_media_tools(tmp_path / "explicit tools") / "bin"
            env["DEEPFAKE_MEDIA_TOOLS"] = str(selected)
        expected = [str(selected / name) for name in ("ffmpeg", "ffprobe")]
    args = ["predict", "--input", "space in path"]
    for command in (
        [
            str(moved / "runtime/python/bin/python3.12"),
            "-I",
            "-S",
            str(moved / "run.py"),
        ],
        ["/bin/bash", str(moved / "run.sh")],
    ):
        result = subprocess.run(
            [*command, *args],
            cwd="/",
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(result.stdout) == {"tools": expected, "args": args}


@pytest.mark.parametrize(
    "filename",
    [
        "build.json",
        "source/ffmpeg-n8.1.3-git.tar.gz",
        "licenses/LICENSE.md",
        "licenses/COPYING.LGPLv2.1",
        "build.sh",
    ],
)
def test_release_requires_media_bundle_assets(
    tmp_path: Path, release_module: ModuleType, filename: str
) -> None:
    media = make_media_tools(tmp_path / "media-tools")
    (media / filename).unlink()
    with pytest.raises(ValueError, match="media bundle missing"):
        release_module.validate_media_tools(media)


@pytest.mark.parametrize("failure", ["missing", "executable"])
@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
def test_launcher_rejects_invalid_explicit_media_tools(
    tmp_path: Path, release_module: ModuleType, failure: str, tool: str
) -> None:
    import prepare_offline_package  # noqa: PLC0415

    (tmp_path / "run.py").write_text(prepare_offline_package.LAUNCHER)
    source = tmp_path / "src/deepfake_detection_archive"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    shutil.copyfile(
        Path(__file__).resolve().parents[1]
        / "src/deepfake_detection_archive/media_tools.py",
        source / "media_tools.py",
    )
    make_media_tools(tmp_path / "runtime/media-tools")
    explicit = make_media_tools(tmp_path / "explicit") / "bin"
    binary = explicit / tool
    if failure == "missing":
        binary.unlink()
    else:
        binary.chmod(0o644)
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(tmp_path / "run.py")],
        cwd="/",
        env={"DEEPFAKE_MEDIA_TOOLS": str(explicit)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "DEEPFAKE_MEDIA_TOOLS requires executable" in result.stderr
    assert tool in result.stderr


@pytest.mark.parametrize(
    "archive",
    ["../outside.tar.gz", "/source/outside.tar.gz", "source/../outside.tar.gz"],
)
def test_media_bundle_source_must_stay_in_source(
    tmp_path: Path, release_module: ModuleType, archive: str
) -> None:
    media = make_media_tools(tmp_path / "media-tools")
    metadata = json.loads((media / "build.json").read_text())
    metadata["source"]["archive"] = archive
    (media / "build.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="relative source/ file"):
        release_module.validate_media_tools(media)


def test_media_bundle_source_hash_must_match(
    tmp_path: Path, release_module: ModuleType
) -> None:
    media = make_media_tools(tmp_path / "media-tools")
    (media / "source/ffmpeg-n8.1.3-git.tar.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source differs"):
        release_module.validate_media_tools(media)
