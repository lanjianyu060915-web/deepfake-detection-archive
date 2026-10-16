"""Shared executable selection for direct API calls and offline launchers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from deepfake_detection_archive import media_tools
from deepfake_detection_archive.config import InferenceConfig
from deepfake_detection_archive.media import AudioConfig, _run
from deepfake_detection_archive.scoring import _validate_media_executables
from deepfake_detection_archive.unified_predict import predict_media


def make_tools(directory: Path, identity: str) -> Path:
    directory.mkdir(parents=True)
    for name in ("ffmpeg", "ffprobe"):
        binary = directory / name
        binary.write_text(f"#!/bin/sh\nprintf '%s\\n' '{identity} {name}' \"$@\"\n")
        binary.chmod(0o755)
    return directory


@pytest.fixture(autouse=True)
def isolate_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPFAKE_MEDIA_TOOLS", raising=False)
    monkeypatch.setattr(
        media_tools, "_BUNDLE_BIN", tmp_path / "runtime/media-tools/bin"
    )
    media_tools._selected_bin.cache_clear()


@pytest.mark.parametrize("mode", ["explicit", "bundle"])
def test_direct_api_selects_tools_and_preserves_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    assert media_tools._BUNDLE_BIN is not None
    bundle = make_tools(media_tools._BUNDLE_BIN, "bundle")
    selected = bundle
    if mode == "explicit":
        selected = make_tools(tmp_path / "explicit tools", "explicit")
        monkeypatch.setenv("DEEPFAKE_MEDIA_TOOLS", str(selected))
    monkeypatch.setenv("PATH", "/no-host-tools")
    for name in ("ffmpeg", "ffprobe"):
        assert media_tools.media_tool(name) == str(selected / name)
        assert _run(
            [name, "-version", "space in path"], 5, 4096
        ).decode().splitlines() == [
            f"{mode} {name}",
            "-version",
            "space in path",
        ]


@pytest.mark.parametrize("mode", ["explicit", "bundle"])
@pytest.mark.parametrize("failure", ["missing", "executable"])
def test_selected_directory_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, failure: str
) -> None:
    assert media_tools._BUNDLE_BIN is not None
    directory = make_tools(media_tools._BUNDLE_BIN, "bundle")
    if mode == "explicit":
        directory = make_tools(tmp_path / "explicit tools", "explicit")
        monkeypatch.setenv("DEEPFAKE_MEDIA_TOOLS", str(directory))
    if failure == "missing":
        (directory / "ffprobe").unlink()
    else:
        (directory / "ffprobe").chmod(0o644)
    with pytest.raises(RuntimeError, match="ffprobe"):
        _run(["ffmpeg", "-version"], 5, 4096)


def test_host_commands_and_path_remain_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/host/bin")
    assert media_tools.media_tool("ffmpeg") == "ffmpeg"
    assert media_tools.media_tool("ffprobe") == "ffprobe"
    assert media_tools.media_tool("other") == "other"
    media_tools.prepare_media_tools_path()
    assert os.environ["PATH"] == "/host/bin"


def test_selection_validates_only_once(tmp_path: Path) -> None:
    assert media_tools._BUNDLE_BIN is not None
    make_tools(media_tools._BUNDLE_BIN, "bundle")
    assert media_tools.media_tool("ffmpeg").endswith("/ffmpeg")
    (media_tools._BUNDLE_BIN / "ffprobe").unlink()
    assert media_tools.media_tool("ffmpeg").endswith("/ffmpeg")
    assert media_tools._selected_bin.cache_info().hits == 1


def test_predict_resume_identity_uses_selected_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Empty discovery exercises identity generation without loading a model.
    registry = {
        "schema_version": 1,
        "models": {"audio": {}, "video": {}},
        "score_type": "sigmoid",
        "score_direction": "higher_is_fake",
    }
    monkeypatch.setattr(
        "deepfake_detection_archive.unified_predict.default_models",
        lambda: (tmp_path, registry),
    )
    first = make_tools(tmp_path / "first tools", "first")
    second = make_tools(tmp_path / "second tools", "second")
    monkeypatch.setenv("DEEPFAKE_MEDIA_TOOLS", str(first))
    output = tmp_path / "output"
    predict_media([], output, "cpu", resume=False)
    identity = json.loads((output / "input.json").read_text())
    assert identity["media_tools"] == {
        name: f"first {name}\n-version\n" for name in ("ffmpeg", "ffprobe")
    }
    monkeypatch.setenv("DEEPFAKE_MEDIA_TOOLS", str(second))
    with pytest.raises(ValueError, match="resume rejected"):
        predict_media([], output, "cpu", resume=True)


@pytest.mark.parametrize("mode", ["explicit", "bundle", "host"])
def test_scorer_precheck_uses_same_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = InferenceConfig(
        manifest="input.jsonl",
        data_root="media",
        output_dir="output",
        modality="audio",
        checkpoint="model.pt",
        checkpoint_sha256="0" * 64,
        device="cpu",
    )
    monkeypatch.setenv("PATH", "/no-host-tools")
    if mode == "host":
        with pytest.raises(RuntimeError, match="required media executables missing"):
            _validate_media_executables(config, AudioConfig())
    else:
        directory = media_tools._BUNDLE_BIN
        assert directory is not None
        if mode == "explicit":
            directory = tmp_path / "explicit"
            monkeypatch.setenv("DEEPFAKE_MEDIA_TOOLS", str(directory))
        make_tools(directory, mode)
        _validate_media_executables(config, AudioConfig())
