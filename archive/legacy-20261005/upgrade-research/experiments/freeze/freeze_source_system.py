# -*- coding: utf-8 -*-
"""把 v0.4.0 源系统完整冻结成单个归档，并生成逐文件 SHA-256 清单。
只读源目录（D 盘），只写归档目录（C 盘桌面），源目录零改动。
用法：
    python freeze_source_system.py
"""
import csv
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

SOURCE = Path(r"D:\刑侦查竞赛\face_source_trace_three_module_v0.4.0_visualization_dev")
DESKTOP = Path(os.path.expanduser("~")) / "Desktop"
DEST = DESKTOP / "换脸寻源_v0.4.0_源系统冻结_20260920"
ARCHIVE = DEST / "源系统_v0.4.0_完整冻结_20260920.zip"
MANIFEST = DEST / "SHA256SUMS_源系统_v0.4.0.csv"
PROGRESS = DEST / "_freeze_progress.json"
STORED_ABOVE = 4 * 1024 * 1024
CHUNK = 4 * 1024 * 1024


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def collect():
    files = []
    for root, _dirs, names in os.walk(SOURCE):
        for name in names:
            path = Path(root) / name
            if path.is_file():
                files.append(path)
    files.sort(key=lambda item: str(item.relative_to(SOURCE)).lower())
    return files


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    started = time.time()
    files = collect()
    total_bytes = sum(path.stat().st_size for path in files)
    log(f"源目录 {SOURCE}")
    log(f"文件数 {len(files)}，总大小 {total_bytes / 1e9:.2f} GB")
    log(f"归档目标 {ARCHIVE}")

    manifest_rows = []
    hashed_bytes = 0
    for index, path in enumerate(files, 1):
        digest = hashlib.sha256()
        size = 0
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK), b""):
                digest.update(chunk)
                size += len(chunk)
        hashed_bytes += size
        manifest_rows.append({
            "relative_path": str(path.relative_to(SOURCE)).replace("\\", "/"),
            "size_bytes": size,
            "sha256": digest.hexdigest(),
        })
        if index % 2000 == 0 or index == len(files):
            elapsed = time.time() - started
            log(f"  校验 {index}/{len(files)} "
                f"({hashed_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB) {elapsed:.0f}s")

    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    log(f"清单写入 {MANIFEST}")

    zip_started = time.time()
    stored = 0
    with zipfile.ZipFile(ARCHIVE, "w", allowZip64=True) as archive:
        for index, path in enumerate(files, 1):
            relative = ("face_source_trace_three_module_v0.4.0_visualization_dev/"
                        f"{path.relative_to(SOURCE)}").replace("\\", "/")
            if path.stat().st_size > STORED_ABOVE:
                archive.write(path, relative, compress_type=zipfile.ZIP_STORED)
                stored += 1
            else:
                archive.write(path, relative, compress_type=zipfile.ZIP_DEFLATED,
                              compresslevel=1)
            if index % 2000 == 0 or index == len(files):
                log(f"  打包 {index}/{len(files)} 已用 {time.time() - zip_started:.0f}s")

    archive_size = ARCHIVE.stat().st_size
    summary = {
        "source": str(SOURCE),
        "archive": str(ARCHIVE),
        "file_count": len(files),
        "source_bytes": total_bytes,
        "archive_bytes": archive_size,
        "compression_ratio": round(archive_size / total_bytes, 4),
        "stored_files_over_4mb": stored,
        "manifest": str(MANIFEST),
        "hash_seconds": round(zip_started - started, 1),
        "zip_seconds": round(time.time() - zip_started, 1),
        "total_seconds": round(time.time() - started, 1),
    }
    PROGRESS.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
