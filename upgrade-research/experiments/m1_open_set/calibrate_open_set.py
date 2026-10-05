# -*- coding: utf-8 -*-
"""M1 开放集阈值标定：只用校准集选阈值，在留出集上评估。

划分（按案例号/查询号预先固定，不事后挑选）：
  校准集 —— 底图：OOT 案例01-05（库外） + 主实验 query_001-005（库内命中）
           身份：supp_identity 案例15/16/20（库内无该身份） + 主实验 query_001-005（命中）
  测试集 —— 底图：OOT 案例06-10 + 本轮新增 holdout 案例11-20（库外）
                  主实验 query_006-010 + 同帧实验 20 例（命中）
           身份：本轮新增 identity holdout 的成功样本（库外）
                  主实验 query_006-010 + supp_identity 命中样本

阈值规则（先验固定，不在数据上扫描）：
    gap = 校准集正类下界 - 校准集负类上界
    T_low  = 负类上界 + 0.25 * gap
    T_high = 正类下界 - 0.25 * gap
即把观测间隙的中间 50% 作为不确定带，两侧各留 25% 作为安全边际。
"""

import csv
import json
from pathlib import Path

EV = Path(r"D:\刑侦查竞赛\_paper_eval_20260919")
WS = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
WORK = Path(r"D:\刑科技项目升级2.0\experiments\m1_open_set")
MARGIN_RATIO = 0.25

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

def _source_signals(path: Path) -> dict | None:
    data = _read_json(path)
    if data.get("status") != "success":
        return None
    metrics = data.get("metrics") or {}
    candidates = data.get("candidates") or []
    top = candidates[0] if candidates else {}
    return {
        "global_score": top.get("global_score"),
        "inlier_count": top.get("ransac_inlier_count"),
        "margin": metrics.get("top1_top2_margin"),
    }

def _identity_signals(path: Path) -> dict | None:
    data = _read_json(path)
    if data.get("status") != "success":
        return None
    metrics = data.get("metrics") or {}
    candidates = data.get("candidates") or []
    top = candidates[0] if candidates else {}
    score = top.get("mean_similarity")
    if score is None:
        score = top.get("score")
    return {"score": score, "margin": metrics.get("top1_top2_margin")}

def collect_source():
    groups = {"cal_neg": [], "cal_pos": [], "test_neg": [], "test_pos": []}
    for index in range(1, 11):
        case = f"案例{index:02d}"
        row = _source_signals(EV / "supp_source" / "out_of_library" / case / "source_result.json")
        if row is None:
            continue
        groups["cal_neg" if index <= 5 else "test_neg"].append({"id": f"oot/{case}", **row})
    holdout = WORK / "source_holdout"
    for index in range(11, 21):
        case = f"案例{index:02d}"
        row = _source_signals(holdout / case / "source_result.json")
        if row is not None:
            groups["test_neg"].append({"id": f"holdout/{case}", **row})
    for index in range(1, 11):
        query = f"query_{index:03d}"
        row = _source_signals(EV / "source" / query / "source_result.json")
        if row is None:
            continue
        groups["cal_pos" if index <= 5 else "test_pos"].append({"id": f"main/{query}", **row})
    for index in range(1, 21):
        case = f"案例{index:02d}"
        row = _source_signals(EV / "supp_source" / "same_frame" / case / "source_result.json")
        if row is not None:
            groups["test_pos"].append({"id": f"same_frame/{case}", **row})
    return groups

def collect_identity():
    groups = {"cal_neg": [], "cal_pos": [], "test_neg": [], "test_pos": [], "unavailable": []}
    with open(EV / "tables" / "supp_identity_swandf.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            case = row["case"]
            if row["status"] != "success":
                groups["unavailable"].append({"id": f"supp/{case}", "reason": "query_face_count!=1"})
                continue
            entry = {
                "id": f"supp/{case}",
                "score": float(row["top1_score"]),
                "margin": None,
                "hit": row["top1_hit"] == "1",
            }
            path = EV / "supp_identity" / "results" / case / "identity_result.json"
            data = _read_json(path)
            entry["margin"] = (data.get("metrics") or {}).get("top1_top2_margin")
            if entry["hit"]:
                groups["test_pos"].append(entry)
            else:
                groups["cal_neg"].append(entry)
    for index in range(1, 11):
        query = f"query_{index:03d}"
        row = _identity_signals(EV / "identity" / query / "identity_result.json")
        if row is None:
            continue
        groups["cal_pos" if index <= 5 else "test_pos"].append({"id": f"main/{query}", **row})
    holdout = WORK / "identity_holdout"
    for index in range(1, 21):
        case = f"案例{index:02d}"
        path = holdout / case / "identity_result.json"
        row = _identity_signals(path)
        if row is None:
            data = _read_json(path)
            if data:
                groups["unavailable"].append({"id": f"holdout/{case}", "reason": data.get("status")})
            continue
        groups["test_neg"].append({"id": f"holdout/{case}", **row})
    return groups

def calibrate(values_neg, values_pos, label):
    if not values_neg or not values_pos:
        raise SystemExit(f"[{label}] 校准集不完整：neg={len(values_neg)} pos={len(values_pos)}")
    neg_max = max(values_neg)
    pos_min = min(values_pos)
    gap = pos_min - neg_max
    if gap <= 0:
        raise SystemExit(f"[{label}] 校准集不可分：neg_max={neg_max} pos_min={pos_min}")
    return {
        "negative_max": neg_max,
        "positive_min": pos_min,
        "gap": gap,
        "low": neg_max + MARGIN_RATIO * gap,
        "high": pos_min - MARGIN_RATIO * gap,
        "n_negative": len(values_neg),
        "n_positive": len(values_pos),
    }

def evaluate(rows, low, high, key, *, invert=False):
    """invert=True 表示分数越低越像命中（本系统不适用，保留参数以显式说明）。"""
    out = []
    for row in rows:
        value = row[key]
        if value is None:
            state = "unavailable"
        elif value >= high:
            state = "high_confidence"
        elif value >= low:
            state = "insufficient_evidence"
        else:
            state = "no_reliable_candidate"
        out.append({**row, "state": state})
    return out

def summarise(rows, expected):
    counts = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {"n": len(rows), "expected": expected, "state_counts": counts,
            "detail": [{"id": r["id"], "value": r.get("global_score", r.get("score")), "state": r["state"]} for r in rows]}

source = collect_source()
identity = collect_identity()

source_cal = calibrate(
    [r["global_score"] for r in source["cal_neg"]],
    [r["global_score"] for r in source["cal_pos"]],
    "source",
)
identity_cal = calibrate(
    [r["score"] for r in identity["cal_neg"]],
    [r["score"] for r in identity["cal_pos"]],
    "identity",
)
identity_margin_min_neg = max(
    (r["margin"] for r in identity["cal_neg"] if r["margin"] is not None), default=0.0)
identity_margin_min_pos = min(
    (r["margin"] for r in identity["cal_pos"] if r["margin"] is not None), default=1.0)
identity_margin_gap = identity_margin_min_pos - identity_margin_min_neg
identity_margin_threshold = (
    identity_margin_min_neg + MARGIN_RATIO * identity_margin_gap
    if identity_margin_gap > 0 else 0.0
)

report = {
    "margin_ratio": MARGIN_RATIO,
    "source": {
        "calibration": source_cal,
        "thresholds": {"high_global_score": source_cal["high"], "low_global_score": source_cal["low"]},
        "test_negative": summarise(
            evaluate(source["test_neg"], source_cal["low"], source_cal["high"], "global_score"),
            "no_reliable_candidate"),
        "test_positive": summarise(
            evaluate(source["test_pos"], source_cal["low"], source_cal["high"], "global_score"),
            "high_confidence"),
    },
    "identity": {
        "calibration": identity_cal,
        "margin_calibration": {
            "negative_max": identity_margin_min_neg,
            "positive_min": identity_margin_min_pos,
            "gap": identity_margin_gap,
            "threshold": identity_margin_threshold,
        },
        "thresholds": {
            "high_similarity": identity_cal["high"],
            "low_similarity": identity_cal["low"],
            "min_top1_top2_margin": identity_margin_threshold,
        },
        "test_negative": summarise(
            evaluate(identity["test_neg"], identity_cal["low"], identity_cal["high"], "score"),
            "no_reliable_candidate"),
        "test_positive": summarise(
            evaluate(identity["test_pos"], identity_cal["low"], identity_cal["high"], "score"),
            "high_confidence"),
        "unavailable": identity["unavailable"],
    },
}

(WORK / "calibration_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

thresholds = {
    "schema_version": "1.0",
    "source_version": "m1-calibrated-20260920",
    "calibration_note": (
        "阈值由校准集（底图：OOT 案例01-05 + 主实验 query_001-005；"
        "身份：库内无该身份样本 + 主实验 query_001-005）按间隙中间 50% 规则确定，"
        "测试集未参与选阈值。"
    ),
    "modules": {
        "source": report["source"]["thresholds"],
        "identity": report["identity"]["thresholds"],
    },
}
target = WS / "packaging_v0.4" / "resources" / "decision" / "open_set_thresholds.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("=== 底图（判据 global_score）===")
print(f"  校准集: 负 n={source_cal['n_negative']} max={source_cal['negative_max']:.6f} | "
      f"正 n={source_cal['n_positive']} min={source_cal['positive_min']:.6f} | gap={source_cal['gap']:.6f}")
print(f"  阈值  : low={source_cal['low']:.6f} high={source_cal['high']:.6f}")
print(f"  测试负类 n={report['source']['test_negative']['n']} -> {report['source']['test_negative']['state_counts']}")
print(f"  测试正类 n={report['source']['test_positive']['n']} -> {report['source']['test_positive']['state_counts']}")
print()
print("=== 身份（判据 mean_similarity）===")
print(f"  校准集: 负 n={identity_cal['n_negative']} max={identity_cal['negative_max']:.6f} | "
      f"正 n={identity_cal['n_positive']} min={identity_cal['positive_min']:.6f} | gap={identity_cal['gap']:.6f}")
print(f"  阈值  : low={identity_cal['low']:.6f} high={identity_cal['high']:.6f}")
print(f"  间隔阈值: {identity_margin_threshold:.6f} (负上界 {identity_margin_min_neg:.4f} / 正下界 {identity_margin_min_pos:.4f})")
print(f"  测试负类 n={report['identity']['test_negative']['n']} -> {report['identity']['test_negative']['state_counts']}")
print(f"  测试正类 n={report['identity']['test_positive']['n']} -> {report['identity']['test_positive']['state_counts']}")
print(f"  无法比对: {len(identity['unavailable'])} 例")
print()
print("阈值配置 ->", target)
