r"""M2 批量验证：用 v2 worker（尺度回退）跑 30 例，与冻结结果逐例对照。

库外 20 例：SWAN-DF 案例01-20 的 05 换脸图；
库内 10 例：主实验 query_001-010。

冻结 worker 的结论来自既有结果文件（experiments/m1_open_set/identity_holdout、
_paper_eval_20260919/identity），本脚本不重新运行冻结 worker。

用法：
    D:\python\.venv\Scripts\python.exe run_identity_v2_batch.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

UPGRADE = Path(r"D:\刑科技项目升级2.0")
WS = UPGRADE / "workspace" / "fsts"
PY = Path(r"D:\python\.venv\Scripts\python.exe")
WORKER = WS / "modules" / "identity_search_v2" / "run_identity_search_v2.py"
SWAN = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
EV = Path(r"D:\刑侦查竞赛\_paper_eval_20260919")
LIBRARY = Path(r"D:\刑侦查竞赛\first_week\data\identity_library")
MODEL_ROOT = WS / "packaging_v0.3" / "resources"
OUT = UPGRADE / "experiments" / "m2_preprocess" / "identity_v2_results"
CACHE = UPGRADE / "experiments" / "m1_open_set" / "_identity_cache"
FROZEN_ROWS = UPGRADE / "experiments" / "m1_open_set" / "identity_holdout" / "holdout_rows.json"

OUT.mkdir(parents=True, exist_ok=True)


def frozen_reference() -> dict[str, dict]:
    reference: dict[str, dict] = {}
    if FROZEN_ROWS.is_file():
        for row in json.loads(FROZEN_ROWS.read_text(encoding="utf-8")):
            reference[f"outside/{row['case']}"] = row
    for index in range(1, 11):
        query = f"query_{index:03d}"
        payload = json.loads(
            (EV / "identity" / query / "identity_result.json").read_text(encoding="utf-8")
        )
        top = (payload.get("candidates") or [{}])[0]
        reference[f"inside/{query}"] = {
            "status": payload.get("status"),
            "query_face_count": (payload.get("metrics") or {}).get("query_face_count"),
            "top1_id": top.get("identity_id"),
            "top1_score": top.get("score"),
        }
    return reference


def invocations() -> list[dict]:
    rows = []
    for index in range(1, 21):
        case = f"案例{index:02d}"
        rows.append(
            {
                "group": "outside",
                "id": case,
                "key": f"outside/{case}",
                "query": SWAN / case / "05_换脸后图片.jpg",
            }
        )
    for index in range(1, 11):
        query = f"query_{index:03d}"
        payload = json.loads(
            (EV / "identity" / query / "identity_result.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "group": "inside",
                "id": query,
                "key": f"inside/{query}",
                "query": Path(payload["query_image"]),
            }
        )
    return rows


reference = frozen_reference()
rows = []
for position, item in enumerate(invocations(), 1):
    out_json = OUT / f"{item['key'].replace('/', '_')}.json"
    started = time.time()
    subprocess.run(
        [
            str(PY), "-B", str(WORKER),
            "--query", str(item["query"]),
            "--library", str(LIBRARY),
            "--top-k", "5",
            "--output", str(out_json),
            "--cache-dir", str(CACHE),
            "--model-root", str(MODEL_ROOT),
            "--offline",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
    )
    wall = round(time.time() - started, 1)
    payload = json.loads(out_json.read_text(encoding="utf-8")) if out_json.is_file() else {}
    preprocessing = ((payload.get("method") or {}).get("preprocessing") or {})
    top = (payload.get("candidates") or [{}])[0]
    row = {
        "key": item["key"],
        "status": payload.get("status"),
        "wall_seconds": wall,
        "detection_scale_used": preprocessing.get("detection_scale_used"),
        "fallback_applied": preprocessing.get("fallback_applied"),
        "attempt_count": len(preprocessing.get("attempts") or []),
        "top1_id": top.get("identity_id"),
        "top1_score": top.get("mean_similarity"),
        "margin": (payload.get("metrics") or {}).get("top1_top2_margin"),
        "frozen_status": (reference.get(item["key"]) or {}).get("status"),
        "frozen_top1_id": (reference.get(item["key"]) or {}).get("top1_id"),
        "frozen_top1_score": (reference.get(item["key"]) or {}).get("top1_score"),
    }
    rows.append(row)
    print(
        f"[{position}/30] {item['key']}: {row['status']} scale={row['detection_scale_used']} "
        f"top1={row['top1_id']} score={row['top1_score']} "
        f"(冻结 {row['frozen_status']}/{row['frozen_top1_id']}) {wall}s",
        flush=True,
    )

(OUT / "batch_rows.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("\n汇总（v2 worker）")
for group in ("outside", "inside"):
    subset = [row for row in rows if row["key"].startswith(group)]
    ok = [row for row in subset if row["status"] == "success"]
    scores = [row["top1_score"] for row in ok if isinstance(row["top1_score"], (int, float))]
    span = f"{min(scores):.4f}-{max(scores):.4f}" if scores else "—"
    fallbacks = [row for row in ok if row["fallback_applied"]]
    print(
        f"  {group:8} success {len(ok)}/{len(subset)}  回退生效 {len(fallbacks)}  "
        f"score {span}"
    )

print("\n与冻结结果对照")
rescued = [
    row for row in rows
    if row["frozen_status"] == "error" and row["status"] == "success"
]
regressed = [
    row for row in rows
    if row["frozen_status"] == "success" and row["status"] != "success"
]
print(f"  由 error 转为可判定: {len(rescued)} 例")
for row in rescued:
    print(f"    {row['key']} -> top1={row['top1_id']} score={row['top1_score']:.6f} "
          f"scale={row['detection_scale_used']}")
print(f"  由成功转为失败: {len(regressed)} 例")
sys.exit(1 if regressed else 0)
