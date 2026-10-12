"""Dual-branch offline package: relocatable layout and no implicit reads.

These pin the delivered behaviour of the movable research package: both branches
run from relative paths after the directory is moved, and the video branch never
reads the training encoder path that its checkpoint config still records.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from deepfake_detection_archive import cli  # noqa: E402
from deepfake_detection_archive.contracts import load_scores  # noqa: E402
from deepfake_detection_archive.data.manifest import (  # noqa: E402
    SampleRecord,
    write_manifest,
)
from deepfake_detection_archive.models.checkpoint import (  # noqa: E402
    save_checkpoint,
    sha256_file,
)
from deepfake_detection_archive.tracking import atomic_json  # noqa: E402


TINY_AST_CONFIG: dict[str, Any] = {
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 24,
    "max_length": 32,
    "num_mel_bins": 32,
    "patch_size": 16,
    "frequency_stride": 10,
    "time_stride": 10,
}


def _package(root: Path) -> Path:
    package = root / "offline-package"
    for name in ("configs", "manifests", "media", "weights"):
        (package / name).mkdir(parents=True)
    return package


def _wav(path: Path, frequency: float) -> None:
    samples = (2000 * np.sin(np.arange(16000) * frequency)).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(samples.tobytes())


def _mp4(path: Path, pattern: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"{pattern}=size=224x224:rate=8:duration=1",
            "-frames:v",
            "8",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )


def _block_external_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline package must not reach the network")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(transformers.ASTModel, "from_pretrained", blocked)
    monkeypatch.setattr("timm.models._builder.load_pretrained", blocked)
    monkeypatch.setattr(cli, "code_identity", lambda: {"code_sha256": "test"})


def _relocate(package: Path, tmp_path: Path) -> Path:
    moved = tmp_path / "relocated" / "deeper" / "offline-package"
    moved.parent.mkdir(parents=True)
    shutil.move(str(package), str(moved))
    return moved


def _assert_relocated_output(moved: Path, expected_ids: list[str]) -> None:
    scores = load_scores(moved / "runs" / "dev" / "scores.jsonl")
    assert [score.sample_id for score in scores] == expected_ids
    failed = [
        f"{score.sample_id}:{score.error_code}:{score.error_message}"
        for score in scores
        if score.status != "ok" or score.score is None
    ]
    assert not failed, failed
    report = json.loads((moved / "runs" / "dev" / "report.json").read_text())
    assert report["failure_count"] == 0
    assert report["coverage"] == 1.0


def test_audio_branch_relocates_without_downloading_pretrained_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepfake_detection_archive.models.ast import (  # noqa: PLC0415
        ASTClassifier,
        ASTConfig,
    )

    build = tmp_path / "build"
    package = _package(build)
    _wav(package / "media" / "real.wav", 0.03)
    _wav(package / "media" / "fake.wav", 0.09)
    write_manifest(
        [
            SampleRecord(
                "real",
                "real.wav",
                "audio",
                0,
                "synthetic",
                content_type="music",
                sha256=sha256_file(package / "media" / "real.wav"),
            ),
            SampleRecord(
                "fake",
                "fake.wav",
                "audio",
                1,
                "synthetic",
                content_type="music",
                sha256=sha256_file(package / "media" / "fake.wav"),
            ),
        ],
        package / "manifests" / "dev.jsonl",
    )
    descriptor = save_checkpoint(
        package / "weights" / "best.pt",
        ASTClassifier(
            ASTConfig(backbone_config=dict(TINY_AST_CONFIG), trainable_layers=1)
        ),
    )
    atomic_json(
        package / "configs" / "dev.json",
        {
            "schema_version": 1,
            "manifest": "../manifests/dev.jsonl",
            "data_root": "../media",
            "output_dir": "../runs/dev",
            "modality": "audio",
            "model_kind": "audio_waveform",
            "checkpoint": "../weights/best.pt",
            "checkpoint_sha256": descriptor["sha256"],
            "device": "cpu",
            "batch_size": 2,
            "segment_samples": 1600,
            "segments": 1,
            "aggregation": "mean",
            "audio": {"sample_rate": 16000, "backend": "soundfile"},
            "seed": 42,
        },
    )
    moved = _relocate(package, tmp_path)
    assert not any(
        str(build) in path.read_text(encoding="utf-8")
        for path in moved.rglob("*")
        if path.suffix in {".json", ".jsonl"}
    )
    _block_external_reads(monkeypatch)
    assert cli.main(["infer", "--config", str(moved / "configs" / "dev.json")]) == 0
    _assert_relocated_output(moved, ["fake", "real"])


def test_video_branch_relocates_and_ignores_training_encoder_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for the video branch test")
    from deepfake_detection_archive.models.finetune import (  # noqa: PLC0415
        VideoFrameClassifier,
        VideoFrameConfig,
    )

    build = tmp_path / "build"
    package = _package(build)
    _mp4(package / "media" / "real.mp4", "testsrc")
    _mp4(package / "media" / "fake.mp4", "testsrc2")
    write_manifest(
        [
            SampleRecord(
                "real",
                "real.mp4",
                "video",
                0,
                "synthetic",
                content_type="general_generated",
                sha256=sha256_file(package / "media" / "real.mp4"),
            ),
            SampleRecord(
                "fake",
                "fake.mp4",
                "video",
                1,
                "synthetic",
                content_type="general_generated",
                sha256=sha256_file(package / "media" / "fake.mp4"),
            ),
        ],
        package / "manifests" / "dev.jsonl",
    )
    training_encoder = build / "training" / "encoder.pt"
    descriptor = save_checkpoint(
        package / "weights" / "best.pt",
        VideoFrameClassifier(
            VideoFrameConfig(
                architecture="resnet18",
                encoder_checkpoint=str(training_encoder),
                expected_sha256="ab" * 32,
                hidden_dim=8,
            )
        ),
    )
    assert not training_encoder.exists()
    atomic_json(
        package / "configs" / "dev.json",
        {
            "schema_version": 1,
            "manifest": "../manifests/dev.jsonl",
            "data_root": "../media",
            "output_dir": "../runs/dev",
            "modality": "video",
            "model_kind": "video_frames",
            "checkpoint": "../weights/best.pt",
            "checkpoint_sha256": descriptor["sha256"],
            "device": "cpu",
            "batch_size": 2,
            "aggregation": "mean",
            "video": {
                "n_frames": 4,
                "height": 224,
                "width": 224,
                "resize_mode": "center_crop",
            },
            "seed": 42,
        },
    )
    moved = _relocate(package, tmp_path)
    sidecar = json.loads((moved / "weights" / "best.pt.json").read_text())
    assert sidecar["config"]["encoder_checkpoint"] == str(training_encoder)
    _block_external_reads(monkeypatch)
    assert cli.main(["infer", "--config", str(moved / "configs" / "dev.json")]) == 0
    _assert_relocated_output(moved, ["fake", "real"])
