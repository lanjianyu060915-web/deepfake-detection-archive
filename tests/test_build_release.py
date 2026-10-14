"""Release layout and relocatable launcher without copying the GPU environment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
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


def test_build_layout_and_launcher_after_relocation(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_module: ModuleType
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
    output = tmp_path / "build"
    args = (output, runtime, python_home, weights, weights, repo / "src")
    release_module.build_release(*args, version="0.1.1")
    assert (output / "configs/models.json").read_text() == "configs/release/models.json"
    with pytest.raises(FileExistsError):
        release_module.build_release(*args, version="0.1.1")
    assert (output / "configs/models.json").read_text() == "configs/release/models.json"
    assert (output / "licenses/upstream/license.txt").read_text() == "docs/license.txt"
    assert (output / "runtime/site-packages/torch-2.0.dist-info/LICENSE").is_file()
    assert (output / "runtime/python/bin/python").readlink() == Path("python3.12")
    metadata = json.loads((output / "release.json").read_text())
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
