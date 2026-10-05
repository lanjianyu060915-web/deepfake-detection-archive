# -*- coding: utf-8 -*-
"""M1 标定数据准备：从 v0.4.0 论文评测结果中提取可用的开放集判别信号。"""
import csv
import json
from pathlib import Path

EV = Path(r"D:\刑侦查竞赛\_paper_eval_20260919")
OUT = Path(r"D:\刑科技项目升级2.0\experiments\m1_open_set")

FIELDS = [
    "group", "sample", "label", "status", "rank1_file",
    "final_score", "global_score", "local_score",
    "good_match_count", "ransac_inlier_count", "ransac_inlier_ratio",
    "query_keypoint_count", "candidate_keypoint_count", "ransac_applied",
    "margin_top1_top2", "local_mode",
]

def row_from(path: Path, group: str, sample: str, label: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    candidates = data.get("candidates") or []
    top = candidates[0] if candidates else {}
    comp = top.get("local_score_components") or {}
    return {
        "group": group, "sample": sample, "label": label,
        "status": data.get("status"),
        "rank1_file": Path(str(top.get("file_path") or "")).name,
        "final_score": top.get("final_score"),
        "global_score": top.get("global_score"),
        "local_score": top.get("local_score"),
        "good_match_count": top.get("good_match_count"),
        "ransac_inlier_count": top.get("ransac_inlier_count"),
        "ransac_inlier_ratio": top.get("ransac_inlier_ratio"),
        "query_keypoint_count": top.get("query_keypoint_count"),
        "candidate_keypoint_count": top.get("candidate_keypoint_count"),
        "ransac_applied": top.get("ransac_applied"),
        "margin_top1_top2": metrics.get("top1_top2_margin"),
        "local_mode": comp.get("mode"),
    }

rows = []
for group, root, label in (
    ("oot_out_of_library", EV / "supp_source" / "out_of_library", "negative"),
    ("main_library_hit", EV / "source", "positive"),
    ("same_frame_hit", EV / "supp_source" / "same_frame", "positive"),
):
    if not root.is_dir():
        continue
    for case_dir in sorted(root.iterdir()):
        p = case_dir / "source_result.json"
        if p.is_file():
            rows.append(row_from(p, group, case_dir.name, label))

csv_path = OUT / "source_signals.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print("rows:", len(rows), "->", csv_path)
for group in ("oot_out_of_library", "main_library_hit", "same_frame_hit"):
    subset = [r for r in rows if r["group"] == group]
    if not subset:
        print(f"\n[{group}] 无数据"); continue
    print(f"\n[{group}] n={len(subset)}")
    for key in ("final_score", "global_score", "local_score", "good_match_count",
                "ransac_inlier_count", "margin_top1_top2"):
        values = [r[key] for r in subset if isinstance(r[key], (int, float))]
        if values:
            print(f"  {key:<20} min={min(values):>12.6f} max={max(values):>12.6f} mean={sum(values)/len(values):>12.6f}")
