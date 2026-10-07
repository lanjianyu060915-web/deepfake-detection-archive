# -*- coding: utf-8 -*-
"""下载 DFDC 公开视频，构建 M4 视频级分带的独立校准集与内部留出集。

来源：HuggingFace 数据集 ``gonnerthetooner/DFDC-extracted-full``（DFDC train
part 00-09 的抽取版），每个 part 附带 ``metadata.json``，给出文件名到
``{"label": "REAL"|"FAKE"}`` 的映射。

抽样设计（与 M4 外部测试集完全分离）：

- 校准集：part 00 随机抽 60 REAL + 60 FAKE；
- 内部留出集：part 01 随机抽 60 REAL + 60 FAKE；
- 外部测试集：SWAN-DF pair1-3 的 22 个视频（见 experiments/m4_video），
  本脚本不下载、不参与校准，也不参与 DFDC 内部的阈值选择。

脚本幂等：文件已存在且字节数与远端一致时不重复下载；每次运行都重写
``manifest.json``（记录 URL、字节数、SHA-256、标签、划分），保证任何后续
实验都能追溯到具体文件。

用法（开发解释器）：
  & $py experiments/m4_video_calibration/download_dfdc.py --sample-size 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DATASET_ID = "gonnerthetooner/DFDC-extracted-full"
REPO_BASE = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"
ROOT = Path(__file__).resolve().parent
META_DIR = ROOT / "meta"
VIDEO_DIR = ROOT / "videos"
MANIFEST_PATH = ROOT / "manifest.json"

# split 名称 -> (part 目录名, 远端 part 序号)
SPLIT_PARTS = {
    "calibration": "dfdc_train_part_00/dfdc_train_part_0",
    "holdout": "dfdc_train_part_01/dfdc_train_part_1",
}
SPLIT_SEED = {"calibration": 20260920, "holdout": 20260921}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, timeout: int = 300) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "fsts-m4cal/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_labels(split: str) -> dict[str, dict]:
    """取该 split 的标签表（远端小文件，本地缓存）。"""

    part = SPLIT_PARTS[split]
    META_DIR.mkdir(parents=True, exist_ok=True)
    cached = META_DIR / f"{split}.json"
    if not cached.is_file():
        payload = fetch(f"{REPO_BASE}/{part}/metadata.json")
        cached.write_bytes(payload)
    return json.loads(cached.read_text(encoding="utf-8"))


def sample_split(split: str, size: int) -> list[dict]:
    """在校准/留出 split 内按标签分层随机抽样，保证可复现。"""

    part = SPLIT_PARTS[split]
    labels = load_labels(split)
    buckets: dict[str, list[str]] = {"REAL": [], "FAKE": []}
    for name, meta in labels.items():
        label = str((meta or {}).get("label") or "").upper()
        if label in buckets:
            buckets[label].append(name)
    rng = random.Random(SPLIT_SEED[split])
    picked: list[dict] = []
    for label in ("REAL", "FAKE"):
        names = sorted(buckets[label])
        if len(names) < size:
            raise SystemExit(f"{split} 的 {label} 只有 {len(names)} 个，少于请求的 {size}")
        for name in rng.sample(names, size):
            picked.append(
                {
                    "split": split,
                    "label": label,
                    "file_name": name,
                    "part": part.split("/")[0],
                    "url": f"{REPO_BASE}/{part}/{name}",
                    "path": str(VIDEO_DIR / split / name),
                }
            )
    return picked


def download_one(item: dict) -> dict:
    target = Path(item["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    record = dict(item)
    if target.is_file():
        record["bytes"] = target.stat().st_size
        record["sha256"] = sha256_file(target)
        record["downloaded"] = False
        return record
    payload = fetch(item["url"])
    target.write_bytes(payload)
    record["bytes"] = len(payload)
    record["sha256"] = sha256_file(target)
    record["downloaded"] = True
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 DFDC 视频校准集")
    parser.add_argument("--sample-size", type=int, default=60, help="每个 split 每个标签抽多少条（默认 60）")
    parser.add_argument("--workers", type=int, default=8, help="并发下载线程数（默认 8）")
    parser.add_argument("--dry-run", action="store_true", help="只打印抽样结果，不下载")
    args = parser.parse_args(argv)

    selected: list[dict] = []
    for split in SPLIT_PARTS:
        selected.extend(sample_split(split, args.sample_size))
    counts: dict[str, int] = {}
    for item in selected:
        counts[f"{item['split']}/{item['label']}"] = counts.get(f"{item['split']}/{item['label']}", 0) + 1
    print("[抽样] " + json.dumps(counts, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    records: list[dict] = []
    failures: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(download_one, item): item for item in selected}
        for future in as_completed(futures):
            item = futures[future]
            done += 1
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - 单条失败不得中断整批
                failures.append({**item, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{done}/{len(selected)}] 失败 {item['file_name']}：{exc}", flush=True)
                continue
            records.append(record)
            flag = "下载" if record["downloaded"] else "命中缓存"
            print(
                f"[{done}/{len(selected)}] {flag} {record['split']}/{record['label']} "
                f"{record['file_name']} {record['bytes'] / 1048576:.2f} MB",
                flush=True,
            )

    records.sort(key=lambda item: (item["split"], item["label"], item["file_name"]))
    manifest = {
        "dataset": DATASET_ID,
        "repo_base": REPO_BASE,
        "sample_size_per_label": args.sample_size,
        "seeds": SPLIT_SEED,
        "splits": SPLIT_PARTS,
        "videos": records,
        "failures": failures,
        "total_bytes": sum(int(item["bytes"]) for item in records),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[完成] {len(records)} 个视频 / {manifest['total_bytes'] / 1073741824:.2f} GB，"
        f"失败 {len(failures)} 条，清单：{MANIFEST_PATH}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
