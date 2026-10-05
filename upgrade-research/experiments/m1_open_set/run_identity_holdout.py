# -*- coding: utf-8 -*-
"""M1 留出实验：SWAN-DF 案例 01-20 的 05 换脸图查 v0.4.0 正式 9 人身份库。
这些人物不属于该库，属于开放集（库外）样本，用于验证身份检索的拒识规则。
只读冻结资产，结果写入升级工作区。"""
import json
import subprocess
import time
from pathlib import Path

SWAN = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
LIB = Path(r"D:\刑侦查竞赛\first_week\data\identity_library")
EXE = Path(r"D:\刑侦查竞赛\face_source_trace_three_module_v0.4.0_visualization_dev"
           r"\packaging_v0.3\work\dist\app\workers\identity\identity_worker.exe")
MODEL_ROOT = Path(r"D:\刑侦查竞赛\face_source_trace_three_module_v0.4.0_visualization_dev"
                  r"\packaging_v0.3\resources")
OUT = Path(r"D:\刑科技项目升级2.0\experiments\m1_open_set\identity_holdout")
CACHE = Path(r"D:\刑科技项目升级2.0\experiments\m1_open_set\_identity_cache")
OUT.mkdir(parents=True, exist_ok=True)

CASES = [f"案例{n:02d}" for n in range(1, 21)]

def run_one(case: str) -> dict:
    query = SWAN / case / "05_换脸后图片.jpg"
    out_json = OUT / case / "identity_result.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    subprocess.run(
        [str(EXE), "--query", str(query), "--library", str(LIB),
         "--top-k", "5", "--output", str(out_json), "--device", "cpu",
         "--cache-dir", str(CACHE), "--model-root", str(MODEL_ROOT), "--offline"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    wall = time.time() - started
    if not out_json.is_file():
        return {"case": case, "status": "no_output", "wall_seconds": round(wall, 2)}
    data = json.loads(out_json.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    candidates = data.get("candidates") or []
    top = candidates[0] if candidates else {}
    return {
        "case": case,
        "status": data.get("status"),
        "error_message": data.get("error_message"),
        "wall_seconds": round(wall, 2),
        "query_face_count": metrics.get("query_face_count"),
        "face_detection_score": metrics.get("face_detection_score"),
        "valid_embedding_count": metrics.get("valid_embedding_count"),
        "library_identity_count": metrics.get("library_identity_count"),
        "margin_top1_top2": metrics.get("top1_top2_margin"),
        "top1_id": top.get("identity_id"),
        "top1_score": top.get("score"),
        "best_reference_score": top.get("best_reference_score"),
        "valid_reference_count": top.get("valid_reference_count"),
    }

rows = []
for index, case in enumerate(CASES, 1):
    row = run_one(case)
    rows.append(row)
    print(f"[{index}/{len(CASES)}] {case} status={row['status']} "
          f"top1={row.get('top1_id')} score={row.get('top1_score')} "
          f"margin={row.get('margin_top1_top2')} {row['wall_seconds']}s", flush=True)

(OUT / "holdout_rows.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
ok = [r for r in rows if r["status"] == "success"]
scores = [r["top1_score"] for r in ok if isinstance(r.get("top1_score"), (int, float))]
print(f"\n完成：success={len(ok)}/{len(rows)}")
if scores:
    print(f"库外 top1_score 区间: {min(scores):.6f} - {max(scores):.6f}")
