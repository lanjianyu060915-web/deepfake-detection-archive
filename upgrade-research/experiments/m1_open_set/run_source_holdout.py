# -*- coding: utf-8 -*-
"""M1 留出实验：用 SWAN-DF 案例 11-20 的 01 底图查 203 图库。
这些查询与 v0.4.0 论文评测的 OOT 实验（案例 01-10）不重叠，用作独立验证负样本。
只读冻结资产，结果全部写入升级工作区。"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SWAN = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
LIB = Path(r"D:\刑侦查竞赛\first_week\data\source_library")
EXE = Path(r"D:\刑侦查竞赛\face_source_trace_three_module_v0.4.0_visualization_dev"
           r"\packaging_v0.3\work\dist\app\workers\source\source_search.exe")
OUT = Path(r"D:\刑科技项目升级2.0\experiments\m1_open_set\source_holdout")
OUT.mkdir(parents=True, exist_ok=True)

CASES = [f"案例{n:02d}" for n in range(11, 21)]

def run_one(case: str) -> dict:
    query = SWAN / case / "01_原始底图.jpg"
    out_json = OUT / case / "source_result.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(
        [str(EXE), "--query", str(query), "--library", str(LIB),
         "--top-k", "5", "--output", str(out_json)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900,
    )
    wall = time.time() - started
    if not out_json.is_file():
        return {"case": case, "status": "no_output", "wall_seconds": wall,
                "stderr": (proc.stderr or "")[-400:]}
    data = json.loads(out_json.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    candidates = data.get("candidates") or []
    top = candidates[0] if candidates else {}
    return {
        "case": case,
        "status": data.get("status"),
        "wall_seconds": round(wall, 2),
        "runtime_seconds": data.get("runtime_seconds"),
        "top1_file": Path(str(top.get("file_path") or "")).name,
        "final_score": top.get("final_score"),
        "global_score": top.get("global_score"),
        "local_score": top.get("local_score"),
        "good_match_count": top.get("good_match_count"),
        "ransac_inlier_count": top.get("ransac_inlier_count"),
        "margin_top1_top2": metrics.get("top1_top2_margin"),
    }

rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(run_one, case): case for case in CASES}
    for future in as_completed(futures):
        row = future.result()
        rows.append(row)
        print(f"[{len(rows)}/{len(CASES)}] {row['case']} status={row['status']} "
              f"global={row.get('global_score')} final={row.get('final_score')} "
              f"inliers={row.get('ransac_inlier_count')} {row['wall_seconds']}s", flush=True)

rows.sort(key=lambda r: r["case"])
(OUT / "holdout_rows.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n完成：", OUT / "holdout_rows.json")
