"""Materialise the dual-branch offline package for issue #31.

The package is the movable research bundle: portable configs, batch/sample
manifests, a few tiny sample media files, the two research checkpoints
(sha256-verified) and the dependency lock. Shared media stays in the read-only
shared asset tree and is only reached through the package-local
``shared-data`` mount point, so nothing inside the package hard-codes an
absolute source path.

Every manifest row keeps ``split == "dev"``; no holdout row is read, selected,
or copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


AUDIO_BUCKETS = ("audio-env", "audio-music", "audio-speech")
VIDEO_BUCKETS = ("video-control", "video-vifbench")
Row = dict[str, object]


def read_manifest(path: Path) -> list[Row]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sort_key(row: Row) -> str:
    return str(row["sample_id"])


def split_labels(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    real = sorted((r for r in rows if r["label"] == 0), key=sort_key)
    fake = sorted((r for r in rows if r["label"] == 1), key=sort_key)
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
            by_attack.setdefault(str(row["generator"]), []).append(row)
        queues = [by_attack[key] for key in sorted(by_attack)]
        chosen: list[Row] = []
        while len(chosen) < 100 and any(queues):
            for queue in queues:
                if queue and len(chosen) < 100:
                    chosen.append(queue.pop(0))
        return real[:100] + sorted(chosen, key=sort_key)
    raise ValueError(rule)


def write_manifest(path: Path, rows: list[Row]) -> None:
    ordered = sorted(rows, key=sort_key)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in ordered
        ),
        encoding="utf-8",
    )


def check_no_holdout(rows: list[Row]) -> None:
    if any(row["split"] == "holdout" for row in rows):
        raise ValueError("refusing to select holdout rows")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vifbench_subset(manifest: Path, diagnostics: Path) -> list[Row]:
    by_id = {str(r["sample_id"]): r for r in read_manifest(manifest)}
    used_real: set[str] = set()
    selected: list[Row] = []
    for diagnostic in sorted(diagnostics.glob("*.jsonl")):
        rows = read_manifest(diagnostic)
        real, fake = split_labels(rows)
        picked = [r for r in real if str(r["sample_id"]) not in used_real][:5]
        used_real.update(str(r["sample_id"]) for r in picked)
        selected.extend(by_id[str(r["sample_id"])] for r in fake[:5] + picked)
    return selected


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
        readable = [r for r in rows if (root / str(r["path"])).is_file()]
        if len(readable) != len(rows):
            raise FileNotFoundError(
                f"{bucket}: {len(rows) - len(readable)} media missing"
            )
        write_manifest(
            self.manifests / f"batch-{bucket}.jsonl",
            [{**r, "path": self.relative(root / str(r["path"]))} for r in readable],
        )
        self.summary[f"batch-{bucket}"] = dict(
            Counter(str(r["label"]) for r in readable)
        )
        real, fake = split_labels(readable)
        chosen = real[: self.sample_per_class] + fake[: self.sample_per_class]
        copied = []
        for row in chosen:
            relative = str(row["path"])
            target = self.media / bucket / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
            copied.append({**row, "path": f"{bucket}/{relative}"})
        write_manifest(self.manifests / f"sample-{bucket}.jsonl", copied)
        self.summary[f"sample-{bucket}"] = dict(
            Counter(str(r["label"]) for r in copied)
        )

    def dev_rows(self, manifest: Path, rule: str) -> list[Row]:
        rows = [r for r in read_manifest(manifest) if r["split"] == "dev"]
        return select(rows, rule)

    def combine(self, name: str, buckets: tuple[str, ...], family: str) -> None:
        merged: list[Row] = []
        for bucket in buckets:
            merged.extend(read_manifest(self.manifests / f"{family}-{bucket}.jsonl"))
        write_manifest(self.manifests / f"{family}-{name}.jsonl", merged)
        self.summary[f"{family}-{name}"] = dict(
            Counter(str(r["label"]) for r in merged)
        )

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

    def copy_dependencies(self, lock: Path) -> None:
        target = self.package / "deps"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lock, target / "uv.lock")
        (target / "README.md").write_text(
            "# deps\n\n"
            "`uv.lock` 是本包对应的依赖锁定文件。运行时使用仓库根共享虚拟环境\n"
            "（Python 3.12 + CUDA 13.2 wheel），不自动安装或下载任何依赖。\n\n"
            "```bash\n"
            "scripts/devctl run python -m deepfake_detection_archive infer \\\n"
            "  --config configs/sample-audio-dev.json\n"
            "```\n",
            encoding="utf-8",
        )


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
    builder.copy_dependencies(args.lock)
    builder.link_shared_data()
    print(json.dumps(builder.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
