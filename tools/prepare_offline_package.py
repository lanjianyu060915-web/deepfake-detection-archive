"""Materialise the dual-branch offline package for issue #31.

The package is the movable research bundle: portable configs, batch/sample
manifests, a few tiny sample media files, the two research checkpoints
(sha256-verified), project code, installed dependencies and the dependency lock.
Shared media stays in the read-only shared asset tree and is only reached through the package-local
``shared-data`` mount point, so nothing inside the package hard-codes an
absolute source path.

Every manifest row keeps ``split == "dev"``; no holdout row is read, selected,
or copied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    load_manifest,
    write_manifest,
)


AUDIO_BUCKETS = ("audio-env", "audio-music", "audio-speech")
VIDEO_BUCKETS = ("video-control", "video-vifbench")
Row = SampleRecord


def read_manifest(path: Path) -> list[Row]:
    rows = load_manifest(path)
    check_no_holdout(rows)
    return rows


def sort_key(row: Row) -> str:
    return str(row.sample_id)


def split_labels(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    real = sorted((r for r in rows if r.label == 0), key=sort_key)
    fake = sorted((r for r in rows if r.label == 1), key=sort_key)
    return real, fake


def select(rows: list[Row], rule: str) -> list[Row]:
    real, fake = split_labels(rows)
    if rule == "all":
        return real + fake
    if rule == "balanced100":
        return real[:100] + fake[:100]
    if rule == "balanced100-attack":
        by_attack: dict[str, list[Row]] = {}
        for row in fake:
            by_attack.setdefault(str(row.generator), []).append(row)
        queues = [by_attack[key] for key in sorted(by_attack)]
        chosen: list[Row] = []
        while len(chosen) < 100 and any(queues):
            for queue in queues:
                if queue and len(chosen) < 100:
                    chosen.append(queue.pop(0))
        return real[:100] + sorted(chosen, key=sort_key)
    raise ValueError(rule)


def check_no_holdout(rows: list[Row]) -> None:
    if any(row.split != "dev" for row in rows):
        raise ValueError("refusing to publish non-dev rows (including holdout)")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vifbench_subset(manifest: Path, diagnostics: Path) -> list[Row]:
    by_id = {str(r.sample_id): r for r in read_manifest(manifest)}
    used_real: set[str] = set()
    selected: list[Row] = []
    for diagnostic in sorted(diagnostics.glob("*.jsonl")):
        rows = read_manifest(diagnostic)
        real, fake = split_labels(rows)
        picked = [r for r in real if str(r.sample_id) not in used_real][:5]
        used_real.update(str(r.sample_id) for r in picked)
        selected.extend(by_id[str(r.sample_id)] for r in fake[:5] + picked)
    return selected


def runtime_dependencies(
    site_packages: Path, roots: tuple[str, ...]
) -> dict[str, importlib.metadata.Distribution]:
    """Resolve the installed runtime closure, including requested extras and markers."""
    from packaging.requirements import Requirement  # noqa: PLC0415 -- release only
    from packaging.utils import canonicalize_name  # noqa: PLC0415 -- release only

    installed = {
        canonicalize_name(dist.metadata["Name"]): dist
        for dist in importlib.metadata.distributions(path=[str(site_packages)])
    }
    selected: dict[str, importlib.metadata.Distribution] = {}
    visited: set[tuple[str, frozenset[str]]] = set()
    pending = [Requirement(root) for root in roots]
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        state = (name, frozenset(requirement.extras))
        if name not in installed:
            raise ValueError(f"missing installed runtime dependency: {requirement}")
        dist = installed[name]
        if requirement.specifier and not requirement.specifier.contains(
            dist.version, prereleases=True
        ):
            raise ValueError(f"installed {name}=={dist.version} violates {requirement}")
        if state in visited:
            continue
        visited.add(state)
        selected[name] = dist
        for raw in dist.requires or ():
            dependency = Requirement(raw)
            if dependency.marker is None or any(
                dependency.marker.evaluate({"extra": extra})
                for extra in ("", *requirement.extras)
            ):
                pending.append(dependency)
    return selected


def copy_runtime_dependencies(
    site_packages: Path, target: Path, roots: tuple[str, ...]
) -> None:
    """Copy wheel-owned runtime files and metadata; omit venv scripts and hooks."""
    site_packages = site_packages.resolve()
    selected = runtime_dependencies(site_packages, roots)
    target.mkdir(parents=True)
    for name, dist in sorted(selected.items()):
        files = dist.files
        if files is None:
            raise ValueError(f"runtime dependency {name} has no installed RECORD")
        for entry in files:
            relative = Path(entry)
            # RECORD also lists executable scripts outside site-packages. The
            # release calls library APIs and supplies its own launchers.
            if relative.is_absolute() or ".." in relative.parts:
                continue
            if "__pycache__" in relative.parts or relative.suffix in (
                ".pyc",
                ".pyo",
                ".pth",
            ):
                continue
            original = site_packages / relative
            if not original.resolve().is_relative_to(site_packages):
                raise ValueError(f"runtime file escapes site-packages: {original}")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)


class Builder:
    """Write one package layout; every stored path is package-relative."""

    def __init__(self, package: Path, shared_root: Path, sample_per_class: int) -> None:
        self.package = package
        self.shared_root = shared_root
        self.sample_per_class = sample_per_class
        self.manifests = package / "manifests"
        self.media = package / "media"
        self.manifests.mkdir(parents=True, exist_ok=True)
        self.summary: dict[str, object] = {}

    def relative(self, absolute: Path) -> str:
        return str(absolute.resolve().relative_to(self.shared_root.resolve()))

    def publish(self, bucket: str, rows: list[Row], root: Path) -> None:
        """Write one shared-media manifest and copy a tiny sample into the package."""
        check_no_holdout(rows)
        readable = [r for r in rows if (root / str(r.path)).is_file()]
        if len(readable) != len(rows):
            raise FileNotFoundError(
                f"{bucket}: {len(rows) - len(readable)} media missing"
            )
        write_manifest(
            [replace(r, path=self.relative(root / r.path)) for r in readable],
            self.manifests / f"batch-{bucket}.jsonl",
        )
        self.summary[f"batch-{bucket}"] = dict(Counter(str(r.label) for r in readable))
        real, fake = split_labels(readable)
        chosen = real[: self.sample_per_class] + fake[: self.sample_per_class]
        copied = []
        for row in chosen:
            relative = str(row.path)
            target = self.media / bucket / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
            copied.append(replace(row, path=f"{bucket}/{relative}"))
        write_manifest(copied, self.manifests / f"sample-{bucket}.jsonl")
        self.summary[f"sample-{bucket}"] = dict(Counter(str(r.label) for r in copied))

    def dev_rows(self, manifest: Path, rule: str) -> list[Row]:
        rows = [r for r in load_manifest(manifest) if r.split == "dev"]
        return select(rows, rule)

    def combine(self, name: str, buckets: tuple[str, ...], family: str) -> None:
        merged: list[Row] = []
        for bucket in buckets:
            merged.extend(read_manifest(self.manifests / f"{family}-{bucket}.jsonl"))
        write_manifest(merged, self.manifests / f"{family}-{name}.jsonl")
        self.summary[f"{family}-{name}"] = dict(Counter(str(r.label) for r in merged))

    def copy_configs(self, config_dir: Path) -> None:
        target = self.package / "configs"
        target.mkdir(parents=True, exist_ok=True)
        names = sorted(path.name for path in config_dir.glob("*.json"))
        if not names:
            raise FileNotFoundError(f"no configs in {config_dir}")
        for name in names:
            shutil.copyfile(config_dir / name, target / name)
        self.summary["configs"] = names

    def copy_weight(self, source: Path, name: str) -> str:
        target = self.package / "weights"
        target.mkdir(parents=True, exist_ok=True)
        sidecar = json.loads(source.with_suffix(".pt.json").read_text(encoding="utf-8"))
        expected = str(sidecar["sha256"])
        actual = sha256_of(source)
        if actual != expected:
            raise ValueError(f"{source}: sha256 {actual} != sidecar {expected}")
        shutil.copyfile(source, target / f"{name}.pt")
        shutil.copyfile(source.with_suffix(".pt.json"), target / f"{name}.pt.json")
        for path in (target / f"{name}.pt", target / f"{name}.pt.json"):
            path.chmod(0o644)
        return actual

    def link_shared_data(self) -> None:
        link = self.package / "shared-data"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(self.shared_root, self.package))
        (self.package / "SHARED-DATA.md").write_text(
            "# shared-data\n\n"
            "共享只读素材根的挂载点：`configs/batch-*.json` 的 `data_root` 指向\n"
            "`../shared-data`。本目录是包内相对软链接，指向团队共享的只读素材根；\n"
            "迁移到新机器后请重新指向本机的素材根，或改用 `configs/sample-*.json`\n"
            "（自带极小样例媒体，可完全离线独立运行）。\n",
            encoding="utf-8",
        )

    def copy_dependencies(
        self,
        lock: Path,
        site_packages: Path,
        source: Path,
        *,
        runtime_roots: tuple[str, ...] | None = None,
    ) -> None:
        """Snapshot installed wheels and project code, without venv path hooks."""
        target = self.package / "deps"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lock, target / "uv.lock")
        runtime = self.package / "runtime" / "site-packages"

        def ignore(directory: str, names: list[str]) -> set[str]:
            excluded = {
                name
                for name in names
                if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
            }
            if Path(directory) == site_packages:
                excluded.update(
                    name
                    for name in names
                    if name.endswith(".pth")
                    or name.startswith("__editable__")
                    or name == "deepfake_detection_archive"
                )
            return excluded

        if runtime_roots is None:
            shutil.copytree(site_packages, runtime, ignore=ignore)
        else:
            copy_runtime_dependencies(site_packages, runtime, runtime_roots)
        shutil.copytree(
            source,
            self.package / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        (self.package / "run.py").write_text(LAUNCHER, encoding="utf-8")
        (target / "README.md").write_text(
            "# 运行环境\n\n"
            "包内 runtime/site-packages 为构建环境已安装依赖的实体快照，"
            "src 为项目代码；不依赖原仓库或共享虚拟环境，不自动安装或下载。\n"
            "目标机需兼容构建机的 Linux 架构/系统动态库、CPython 3.12、"
            "ffmpeg/ffprobe（PATH 可用）；CUDA 配置另需兼容的 NVIDIA 驱动。"
            "解释器、系统动态库及驱动不在包内；该包不是跨平台发行版。\n\n"
            "从包目录运行：`python3.12 -I -S run.py infer "
            "--config configs/sample-audio-dev.json`。"
            "依赖许可保留在各 dist-info，uv.lock 仅用于追溯；"
            "快照包含构建环境已有工具，不能视为最小依赖清单。\n",
            encoding="utf-8",
        )


LAUNCHER = '''"""Run only bundled project code and installed dependencies."""
import os
from pathlib import Path
import runpy
import sys

if (sys.version_info[:2] != (3, 12) or sys.implementation.name != "cpython"
        or not sys.platform.startswith("linux")):
    raise SystemExit("requires Linux CPython 3.12")
if not sys.flags.isolated or not sys.flags.no_site:
    raise SystemExit("run with python3.12 -I -S run.py ...")
root = Path(__file__).resolve().parent
sys.path[:0] = [str(root / "src"), str(root / "runtime" / "site-packages")]
for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[name] = "1"
runpy.run_module("deepfake_detection_archive", run_name="__main__")
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--audio-weight", type=Path, required=True)
    parser.add_argument(
        "--audio-weight-name", default="ast-mixed12-class-balanced-seed42"
    )
    parser.add_argument("--video-weight", type=Path, required=True)
    parser.add_argument("--video-weight-name", default="video-l4head-seed42")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1] / "src"
    )
    parser.add_argument("--env-manifest", type=Path, required=True)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--music-manifest", type=Path, required=True)
    parser.add_argument("--music-root", type=Path, required=True)
    parser.add_argument("--speech-manifest", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--video-control-manifest", type=Path, required=True)
    parser.add_argument("--video-control-root", type=Path, required=True)
    parser.add_argument("--vifbench-manifest", type=Path, required=True)
    parser.add_argument("--vifbench-root", type=Path, required=True)
    parser.add_argument("--vifbench-diagnostics", type=Path, required=True)
    parser.add_argument("--sample-per-class", type=int, default=3)
    args = parser.parse_args()

    if args.package.exists():
        parser.error("--package must be a new directory")
    if args.sample_per_class < 1:
        parser.error("--sample-per-class must be positive")
    builder = Builder(args.package, args.shared_root, args.sample_per_class)
    selections = [
        (
            "audio-env",
            builder.dev_rows(args.env_manifest, "balanced100"),
            args.env_root,
        ),
        ("audio-music", builder.dev_rows(args.music_manifest, "all"), args.music_root),
        (
            "audio-speech",
            builder.dev_rows(args.speech_manifest, "balanced100-attack"),
            args.speech_root,
        ),
        (
            "video-control",
            builder.dev_rows(args.video_control_manifest, "all"),
            args.video_control_root,
        ),
        (
            "video-vifbench",
            vifbench_subset(args.vifbench_manifest, args.vifbench_diagnostics),
            args.vifbench_root,
        ),
    ]
    for bucket, rows, root in selections:
        builder.publish(bucket, rows, root)

    for name, members in (("audio-dev", AUDIO_BUCKETS), ("video-dev", VIDEO_BUCKETS)):
        for family in ("batch", "sample"):
            builder.combine(name, members, family)

    builder.copy_configs(args.config_dir)
    builder.summary["audio_weight_sha256"] = builder.copy_weight(
        args.audio_weight, args.audio_weight_name
    )
    builder.summary["video_weight_sha256"] = builder.copy_weight(
        args.video_weight, args.video_weight_name
    )
    builder.copy_dependencies(args.lock, args.site_packages, args.source)
    builder.link_shared_data()
    print(json.dumps(builder.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
