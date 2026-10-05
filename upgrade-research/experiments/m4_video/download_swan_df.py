# -*- coding: utf-8 -*-
"""下载 SWAN-DF 公开视频（原始底图 / 换入身份 / 换脸结果），仅用于 M4 真实视频验证。

数据来源与「案例1」数据集完全一致（`来源总表.csv` 中的官方公开链接，
https://swan-df.github.io/videos/...）。脚本是幂等的：已存在且 SHA-256 一致的
文件不会重复下载；每次运行都会重写 manifest.json（含 URL、字节数、SHA-256），
保证任何后续实验都能追溯到具体文件。
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


SOURCE_TABLE = Path(r"D:\刑侦查竞赛\数据\案例1\案例1\来源总表.csv")
OUTPUT_ROOT = Path(r"D:\刑科技项目升级2.0\experiments\m4_video\videos")
MANIFEST = Path(r"D:\刑科技项目升级2.0\experiments\m4_video\videos\manifest.json")

ROLE_COLUMNS = {
    "原始底图视频URL": "original_source",
    "身份来源视频URL": "identity_source",
    "换脸结果视频URL": "deepfake_result",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "fsts-m4-download/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    target.write_bytes(payload)
    return {"bytes": len(payload), "sha256": sha256_file(target)}


def main() -> int:
    if not SOURCE_TABLE.is_file():
        print(f"缺少来源总表：{SOURCE_TABLE}", file=sys.stderr)
        return 2
    with SOURCE_TABLE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"source_table": str(SOURCE_TABLE), "videos": [], "failures": []}
    seen: dict[str, dict] = {}
    for row in rows:
        case = str(row["案例编号"]).strip()
        pair = str(row["SWAN-DF配对"]).strip()
        for column, role in ROLE_COLUMNS.items():
            url = str(row.get(column) or "").strip()
            if not url:
                continue
            name = url.rsplit("/", 1)[-1]
            key = f"{pair}/{name}"
            if key in seen:
                record = dict(seen[key])
                record.setdefault("cases", []).append({"case": case, "role": role})
                manifest["videos"].append(record)
                continue
            target = OUTPUT_ROOT / pair / name
            record = {
                "pair": pair,
                "file_name": name,
                "url": url,
                "role": role,
                "path": str(target),
                "cases": [{"case": case, "role": role}],
            }
            try:
                if target.is_file() and target.stat().st_size > 0:
                    record["downloaded"] = False
                    record["bytes"] = target.stat().st_size
                    record["sha256"] = sha256_file(target)
                else:
                    info = download(url, target)
                    record["downloaded"] = True
                    record.update(info)
            except Exception as exc:  # noqa: BLE001 - 单条失败不终止整体
                record["error"] = f"{type(exc).__name__}: {exc}"
                manifest["failures"].append(record)
                print(f"FAILED {key}: {record['error']}")
                continue
            seen[key] = record
            manifest["videos"].append(record)
            print(f"OK {key} bytes={record['bytes']} sha={record['sha256'][:12]}")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"videos={len(manifest['videos'])} failures={len(manifest['failures'])}")
    return 0 if not manifest["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

