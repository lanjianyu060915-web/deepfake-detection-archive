"""Build a relocatable Linux offline baseline from installed local assets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

from prepare_offline_package import Builder

from deepfake_detection_archive.data.manifest import SampleRecord, write_manifest


REPOSITORY = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (
    "torch",
    "torchvision",
    "timm",
    "transformers",
    "numpy",
    "scipy",
    "soundfile",
    "safetensors",
)
RUN_SH = """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$ROOT/runtime/python/bin/python3.12" -I -S "$ROOT/run.py" "$@"
"""


def generate_examples(output: Path) -> None:
    """Self-generated signals exercise decoding; they carry no authenticity labels."""
    audio, video = output / "media/audio", output / "media/video"
    audio.mkdir(parents=True)
    video.mkdir(parents=True)
    examples = [
        (
            "sine=frequency=440:sample_rate=44100:duration=2",
            ["-ac", "2"],
            audio / "stereo-44100.wav",
        ),
        (
            "sine=frequency=880:sample_rate=16000:duration=2",
            ["-ac", "1"],
            audio / "mono-16000.flac",
        ),
        (
            "testsrc=size=224x224:rate=8:duration=2",
            ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
            video / "testsrc.mp4",
        ),
        (
            "color=c=blue:size=224x224:rate=8:duration=2",
            ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
            video / "blue.mp4",
        ),
    ]
    for signal, options, path in examples:
        subprocess.run(  # noqa: S603 -- local assets passed as separate arguments
            [  # noqa: S607 -- configured system FFmpeg
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                signal,
                *options,
                "-threads",
                "1",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

    for modality in ("audio", "video"):
        rows = [
            SampleRecord(
                sample_id=f"example-{modality}-{path.stem}",
                path=path.name,
                modality=modality,
                label=None,
                source="self-generated-decoding-example",
                split="test",
            )
            for path in sorted((output / "media" / modality).iterdir())
        ]
        write_manifest(rows, output / "manifests" / f"input-{modality}.jsonl")


def source_identity(source: Path) -> dict[str, object]:
    head = subprocess.run(  # noqa: S603 -- local assets passed as separate arguments
        ["git", "-C", str(source), "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode:
        return {"git_head": None, "dirty": None}
    status = subprocess.run(  # noqa: S603 -- local assets passed as separate arguments
        ["git", "-C", str(source), "status", "--porcelain"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return {"git_head": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def checkpoint_ast_frontend(path: Path) -> str:
    """Read the checkpoint's explicit frontend; legacy weights use NumPy."""
    import torch  # noqa: PLC0415

    from deepfake_detection_archive.models.ast import ASTConfig  # noqa: PLC0415

    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return ASTConfig(**payload["config"]).frontend


def build_release(  # noqa: PLR0913, PLR0917 -- explicit local build inputs
    output: Path,
    site_packages: Path,
    python_home: Path,
    audio_weight: Path,
    video_weight: Path,
    source: Path,
    *,
    version: str,
) -> None:
    if not version.strip():
        raise ValueError("release version must not be empty")
    output.mkdir(parents=True, exist_ok=False)
    builder = Builder(output, output, 1)
    roots: tuple[str, ...] = RUNTIME_ROOTS
    ast_frontend = checkpoint_ast_frontend(audio_weight)
    if ast_frontend == "torchaudio_kaldi_v1":
        roots += ("torchaudio",)
    builder.copy_dependencies(
        REPOSITORY / "uv.lock", site_packages.resolve(), source, runtime_roots=roots
    )
    hashes = {
        "audio": builder.copy_weight(audio_weight, "audio-ast-urban"),
        "video": builder.copy_weight(video_weight, "video-resnet18"),
    }
    shutil.copytree(python_home, output / "runtime/python", symlinks=True)
    (output / "run.sh").write_text(RUN_SH, encoding="utf-8")
    (output / "run.sh").chmod(0o755)
    (output / "configs").mkdir()
    for name, filename in (
        ("audio", "audio-environment.json"),
        ("video", "video.json"),
    ):
        path = REPOSITORY / "configs/release" / filename
        config = json.loads(path.read_text(encoding="utf-8"))
        if config["checkpoint_sha256"] != hashes[name]:
            raise ValueError(
                f"{name} checkpoint differs from the frozen release config"
            )
        (output / "configs" / path.name).write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    shutil.copyfile(
        REPOSITORY / "configs/release/models.json", output / "configs/models.json"
    )
    shutil.copyfile(REPOSITORY / "docs/release.md", output / "README.md")
    licenses = output / "licenses"
    licenses.mkdir()
    shutil.copyfile(REPOSITORY / "docs/release-licenses.md", licenses / "README.md")
    license_files = json.loads(
        (REPOSITORY / "configs/release/licenses.json").read_text(encoding="utf-8")
    )
    for destination, original in license_files.items():
        target = licenses / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / original, target)
    # Override the reused research-package note: this release includes CPython.
    (output / "deps/README.md").write_text(
        "运行依赖按已安装 wheel 的 Requires-Dist 闭包和 RECORD 复制，\n"
        "许可证保留在 dist-info。\n"
        "CPython 位于 runtime/python，使用根目录 run.sh 启动；"
        "FFmpeg/ffprobe、系统动态库与 NVIDIA 驱动由目标机提供。\n"
        "uv.lock 记录构建环境来源，实际运行依赖见 release.json。\n",
        encoding="utf-8",
    )
    generate_examples(output)
    python_info = subprocess.run(  # noqa: S603 -- local assets passed as separate arguments
        [
            str(output / "runtime/python/bin/python3.12"),
            "-I",
            "-S",
            "-c",
            "import json,platform,sys; print(json.dumps({"
            "'version':sys.version,'platform':platform.platform(),"
            "'machine':platform.machine()}))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    distributions = {
        dist.metadata["Name"].lower(): dist.version
        for dist in importlib.metadata.distributions(
            path=[str(output / "runtime/site-packages")]
        )
    }
    metadata = {
        "version": version,
        "runtime_dependencies": distributions,
        "ast_frontend": ast_frontend,
        "python": json.loads(python_info.stdout),
        "torch": distributions["torch"],
        "checkpoint_sha256": hashes,
        "source": source_identity(source),
        "official_adapter_confirmed": False,
        "model_scope": {"audio": "environmental_audio_research"},
        "general_audio_ready": False,
        "competition_use": "unconfirmed",
        "examples": "self-generated decoding demonstrations, no authenticity labels",
    }
    (output / "release.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--python-home", type=Path, required=True)
    parser.add_argument("--audio-weight", type=Path, required=True)
    parser.add_argument("--video-weight", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=REPOSITORY / "src")
    args = parser.parse_args()
    build_release(
        args.output,
        args.site_packages,
        args.python_home,
        args.audio_weight,
        args.video_weight,
        args.source,
        version=args.version,
    )
    print(args.output)


if __name__ == "__main__":
    main()
