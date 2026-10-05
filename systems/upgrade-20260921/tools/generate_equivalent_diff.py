"""Generate a reproducible Git-like report for two non-Git directory trees."""

from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path
from typing import Iterable


EXCLUDED_PREFIXES = (
    "runtime/tasks/",
    "runtime/inbox/",
    "runtime/validation/",
    "packaging/",
    "release_v1.0/",
    "docs/EQUIVALENT_DIFF.md",
)
TEXT_SUFFIXES = {
    ".bat",
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    parts = normalized.split("/")
    return (
        "__pycache__" in parts
        or ".pytest_cache" in parts
        or any(normalized.startswith(prefix) for prefix in EXCLUDED_PREFIXES)
    )


def _inventory(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if _excluded(relative):
            continue
        result[relative] = path
    return result


def _safe_text(path: Path) -> list[str] | None:
    if path.suffix.casefold() not in TEXT_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeDecodeError):
        return None


def _item_line(relative: str, path: Path, digest: str) -> str:
    return f"- `{relative}` — {path.stat().st_size:,} bytes — `{digest}`"


def _section(lines: list[str], title: str, values: Iterable[str]) -> None:
    lines.extend([f"## {title}", ""])
    collected = list(values)
    if collected:
        lines.extend(collected)
    else:
        lines.append("- 无")
    lines.append("")


def generate(base: Path, target: Path, output: Path) -> None:
    base_files = _inventory(base)
    target_files = _inventory(target)
    added_names = sorted(target_files.keys() - base_files.keys())
    deleted_names = sorted(base_files.keys() - target_files.keys())
    common_names = sorted(base_files.keys() & target_files.keys())

    hashes: dict[tuple[str, str], str] = {}

    def digest(side: str, relative: str, path: Path) -> str:
        key = (side, relative)
        if key not in hashes:
            hashes[key] = _sha256(path)
        return hashes[key]

    modified_names: list[str] = []
    for relative in common_names:
        old = base_files[relative]
        new = target_files[relative]
        if old.stat().st_size != new.stat().st_size:
            modified_names.append(relative)
            continue
        if digest("base", relative, old) != digest("target", relative, new):
            modified_names.append(relative)

    lines = [
        "# 非 Git 等价差异报告",
        "",
        f"- 基线：`{base}`",
        f"- 目标：`{target}`",
        "- 算法：相对路径集合 + 文件 SHA-256；文本文件附 unified diff。",
        "- 排除：任意层级 `__pycache__`/`.pytest_cache`，以及 `"
        + "`、`".join(EXCLUDED_PREFIXES)
        + "`。",
        "  排除项是运行时任务/缓存噪音和继承的 v0.1 大型打包产物；正式测试证据另归档在 `test_results/`。",
        "",
        "## 汇总",
        "",
        f"- 新增：{len(added_names)} 个文件",
        f"- 修改：{len(modified_names)} 个文件",
        f"- 删除：{len(deleted_names)} 个文件",
        "",
    ]

    _section(
        lines,
        "新增文件",
        (
            _item_line(
                relative,
                target_files[relative],
                digest("target", relative, target_files[relative]),
            )
            for relative in added_names
        ),
    )
    _section(
        lines,
        "修改文件",
        (
            f"- `{relative}` — "
            f"{base_files[relative].stat().st_size:,} → "
            f"{target_files[relative].stat().st_size:,} bytes — "
            f"`{digest('base', relative, base_files[relative])}` → "
            f"`{digest('target', relative, target_files[relative])}`"
            for relative in modified_names
        ),
    )
    _section(
        lines,
        "删除文件",
        (
            _item_line(
                relative,
                base_files[relative],
                digest("base", relative, base_files[relative]),
            )
            for relative in deleted_names
        ),
    )

    lines.extend(["## 文本 unified diff", ""])
    diff_count = 0
    for relative in modified_names:
        old_lines = _safe_text(base_files[relative])
        new_lines = _safe_text(target_files[relative])
        if old_lines is None or new_lines is None:
            continue
        delta = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"base/{relative}",
                tofile=f"target/{relative}",
                n=3,
            )
        )
        if not delta:
            continue
        diff_count += 1
        lines.extend([f"### `{relative}`", "", "```diff"])
        lines.extend(line.rstrip("\n") for line in delta)
        lines.extend(["```", ""])
    if diff_count == 0:
        lines.append("无可解码的已修改文本文件。")
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(
        args.base.expanduser().resolve(strict=True),
        args.target.expanduser().resolve(strict=True),
        args.output.expanduser().resolve(strict=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
