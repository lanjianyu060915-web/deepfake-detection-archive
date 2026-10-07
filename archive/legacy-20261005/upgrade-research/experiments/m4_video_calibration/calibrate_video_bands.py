# -*- coding: utf-8 -*-
"""用独立 DFDC 视频集标定视频级分带，并在未见过的 SWAN-DF 视频上评估。

三层数据，互不重叠：

1. 校准集   = DFDC part 00 抽样（60 REAL + 60 FAKE）——只用于选择统计量与阈值；
2. 内部留出 = DFDC part 01 抽样（60 REAL + 60 FAKE）——只用于验证，不回头改阈值；
3. 外部测试 = SWAN-DF pair1-3 真实视频（6 REAL + 12 FAKE，来自 experiments/m4_video），
   即 M4 的测试集，全程未参与任何校准决策。

脚本只读结果文件，不重跑模型。输出：

- results/<tag>/analysis.json：统计量对比、AUC 与 bootstrap 置信区间、阈值、
  四种候选规则在三个集合上的混淆矩阵；
- 控制台打印同一份表格，便于直接抄进文档。

用法：
  & $py experiments/m4_video_calibration/calibrate_video_bands.py \
      --runs results/m4cal_dfdc/runs.jsonl --tag video_bands_v1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = ROOT.parent
SWAN_RESULTS = EXPERIMENTS_ROOT / "m4_video" / "results"

REAL_ROLES = {"original_source", "identity_source"}
SWAN_SOURCES = ("m4_swaps_all", "m4_controls", "m4_identity_sources")

# 帧级分带边界（M3 整图标定，落在 detection_result.json 的 band_thresholds）；
# 这里的默认值只用于在缺字段时给出提示，正式计算一律读结果文件。
FALLBACK_FRAME_BANDS = {"low_max": 2.9245e-04, "medium_max": 4.448e-04}


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def frame_scores(record: dict) -> list[float]:
    scores: list[float] = []
    for frame in (record.get("deepfake") or {}).get("per_frame") or []:
        value = frame.get("score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                scores.append(float(value))
    return scores


def video_statistics(scores: list[float]) -> dict:
    """同一个视频的多种聚合口径；用于比较哪种口径更能分开真人与深伪。"""

    if not scores:
        return {}
    ordered = sorted(scores)
    return {
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "maximum": ordered[-1],
        "minimum": ordered[0],
        "frames": len(ordered),
    }


def auc(positive: list[float], negative: list[float]) -> float | None:
    """Mann-Whitney U 统计量对应的 AUC（正类=深伪）。"""

    if not positive or not negative:
        return None
    combined = sorted([(value, 1) for value in positive] + [(value, 0) for value in negative])
    ranks: dict[int, float] = {}
    index = 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average_rank
        index = end + 1
    rank_sum = sum(ranks[position] for position, (_, label) in enumerate(combined) if label == 1)
    n_pos = len(positive)
    n_neg = len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_auc_ci(
    positive: list[float], negative: list[float], iterations: int = 2000, seed: int = 20260920
) -> tuple[float, float] | None:
    if not positive or not negative:
        return None
    import random

    rng = random.Random(seed)
    values: list[float] = []
    n_pos, n_neg = len(positive), len(negative)
    for _ in range(iterations):
        sample_pos = [positive[rng.randrange(n_pos)] for _ in range(n_pos)]
        sample_neg = [negative[rng.randrange(n_neg)] for _ in range(n_neg)]
        value = auc(sample_pos, sample_neg)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return (low, high)


def order_statistic_threshold(real_scores: list[float], target_fpr: float) -> tuple[float, float]:
    """按顺序统计量取阈值：允许 k 个真人样本超过它，k 至少为 1。

    返回 ``(阈值, 校准集上实现的 FPR)``。n 较小时 1% 目标点无法分辨，这里
    用 k>=1 保证阈值不落在真值之上，并把实现 FPR 如实报出来。
    """

    ordered = sorted(real_scores)
    n = len(ordered)
    if n == 0:
        raise SystemExit("校准集里没有真人视频分数，无法标定阈值")
    k = max(1, min(n, int(round(target_fpr * n))))
    return ordered[n - k], k / n


def band_from_thresholds(score: float, low_max: float, medium_max: float) -> str:
    if score > medium_max:
        return "high"
    if score > low_max:
        return "medium"
    return "low"


def load_dfdc_originals(meta_dir: Path, splits: tuple[str, ...]) -> dict[str, str]:
    """DFDC 元数据里每个换脸视频对应的原视频文件名。"""

    mapping: dict[str, str] = {}
    for split in splits:
        path = meta_dir / f"{split}.json"
        if not path.is_file():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        for name, info in meta.items():
            original = (info or {}).get("original")
            if isinstance(original, str) and original:
                mapping[name] = original
    return mapping


def paired_analysis(
    records: list[dict], originals: dict[str, str]
) -> dict:
    """同一原视频内的配对比较：换脸版中位分是否高于原视频中位分。

    跨视频比较会被场景/光照/编码等视频级因素主导；配对比较把这一层消掉，
    直接回答「检测器对换脸本身是否敏感」。这是判断分数是否可用的关键证据。
    """

    by_name = {record["file_name"]: record for record in records}
    pairs: list[dict] = []
    for name, original in originals.items():
        fake = by_name.get(name)
        real = by_name.get(original)
        if not fake or not real:
            continue
        fake_median = fake["stats"].get("median")
        real_median = real["stats"].get("median")
        if fake_median is None or real_median is None:
            continue
        pairs.append(
            {
                "fake": name,
                "real": original,
                "fake_median": fake_median,
                "real_median": real_median,
                "fake_higher": bool(fake_median > real_median),
            }
        )
    n = len(pairs)
    higher = sum(1 for item in pairs if item["fake_higher"])
    if n:
        from math import comb

        p_value = sum(comb(n, k) for k in range(higher, n + 1)) / (2**n)
    else:
        p_value = None
    return {
        "pairs": n,
        "fake_higher": higher,
        "fake_higher_rate": round(higher / n, 4) if n else None,
        "sign_test_one_sided_p": round(p_value, 5) if p_value is not None else None,
        "details": sorted(pairs, key=lambda item: item["fake_median"] - item["real_median"]),
    }


def confusion(records: list[dict], decide) -> dict:
    """按给定判定函数统计三态结果；REAL 为正类之外，用误报数刻画风险。"""

    table = {"high": {"REAL": 0, "FAKE": 0}, "medium": {"REAL": 0, "FAKE": 0}, "low": {"REAL": 0, "FAKE": 0}}
    for record in records:
        band = decide(record)
        label = record["label"]
        table.setdefault(band, {"REAL": 0, "FAKE": 0})[label] += 1
    real_total = sum(1 for record in records if record["label"] == "REAL")
    fake_total = sum(1 for record in records if record["label"] == "FAKE")
    flagged_real = table["high"]["REAL"] + table["medium"]["REAL"]
    flagged_fake = table["high"]["FAKE"] + table["medium"]["FAKE"]
    return {
        "table": table,
        "real_total": real_total,
        "fake_total": fake_total,
        "fpr_medium_or_high": round(flagged_real / real_total, 4) if real_total else None,
        "recall_medium_or_high": round(flagged_fake / fake_total, 4) if fake_total else None,
        "fpr_high": round(table["high"]["REAL"] / real_total, 4) if real_total else None,
        "recall_high": round(table["high"]["FAKE"] / fake_total, 4) if fake_total else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="视频级分带标定")
    parser.add_argument("--runs", required=True, help="DFDC 校准/留出 runs.jsonl 路径")
    parser.add_argument("--tag", default="video_bands_v1")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--meta-dir", default=str(ROOT / "meta"))
    args = parser.parse_args(argv)

    runs_path = Path(args.runs)
    if not runs_path.is_absolute():
        runs_path = (ROOT / runs_path).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else runs_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    dfdc = [record for record in read_jsonl(runs_path) if record.get("status") == "success"]
    calibration = [record for record in dfdc if record.get("split") == "calibration"]
    holdout = [record for record in dfdc if record.get("split") == "holdout"]

    swan: list[dict] = []
    seen: set[str] = set()
    for source in SWAN_SOURCES:
        for record in read_jsonl(SWAN_RESULTS / source / "runs.jsonl"):
            name = record.get("file_name")
            if name in seen or record.get("status") != "success":
                continue
            seen.add(name)
            swan.append(
                {
                    "label": "REAL" if record.get("role") in REAL_ROLES else "FAKE",
                    "file_name": name,
                    "source": source,
                    "role": record.get("role"),
                    "deepfake": record.get("deepfake") or {},
                }
            )

    for group in (calibration, holdout, swan):
        for record in group:
            record["stats"] = video_statistics(frame_scores(record))
            record["frame_bands"] = {
                "high": ((record.get("deepfake") or {}).get("frame_band_counts") or {}).get("high", 0),
                "medium": ((record.get("deepfake") or {}).get("frame_band_counts") or {}).get("medium", 0),
            }

    statistic_names = ("median", "mean", "maximum")
    statistic_comparison = {}
    for name in statistic_names:
        entry = {}
        for split_name, group in (("calibration", calibration), ("holdout", holdout), ("swan_df_test", swan)):
            pos = [record["stats"][name] for record in group if record["label"] == "FAKE" and name in record["stats"]]
            neg = [record["stats"][name] for record in group if record["label"] == "REAL" and name in record["stats"]]
            value = auc(pos, neg)
            entry[split_name] = {
                "auc": round(value, 4) if value is not None else None,
                "auc_ci95": [round(item, 4) for item in (bootstrap_auc_ci(pos, neg) or [])] or None,
                "n_real": len(neg),
                "n_fake": len(pos),
            }
        statistic_comparison[name] = entry

    paired = paired_analysis(
        calibration + holdout,
        load_dfdc_originals(Path(args.meta_dir), ("calibration", "holdout")),
    )

    calibration_real = [record["stats"]["median"] for record in calibration if record["label"] == "REAL"]
    calibration_fake = [record["stats"]["median"] for record in calibration if record["label"] == "FAKE"]
    low_max, realized_low = order_statistic_threshold(calibration_real, 0.05)
    medium_max, realized_medium = order_statistic_threshold(calibration_real, 0.01)

    def decide_score(record: dict) -> str:
        if "median" not in record["stats"]:
            return "low"
        return band_from_thresholds(record["stats"]["median"], low_max, medium_max)

    def decide_score_plus_one_high(record: dict) -> str:
        band = decide_score(record)
        if record["frame_bands"]["high"] >= 1 and band == "low":
            return "medium"
        return band

    def decide_score_plus_two_high(record: dict) -> str:
        band = decide_score(record)
        if record["frame_bands"]["high"] >= 2:
            return "high"
        if record["frame_bands"]["high"] >= 1 and band == "low":
            return "medium"
        return band

    def decide_majority(record: dict) -> str:
        return str((record.get("deepfake") or {}).get("band") or "low")

    rules = {
        "R0_band_majority": decide_majority,
        "R1_median_thresholds": decide_score,
        "R2_median_plus_one_high_frame": decide_score_plus_one_high,
        "R3_median_plus_two_high_frames": decide_score_plus_two_high,
    }
    rule_evaluation = {
        name: {
            "calibration": confusion(calibration, decide),
            "holdout": confusion(holdout, decide),
            "swan_df_test": confusion(swan, decide),
        }
        for name, decide in rules.items()
    }

    analysis = {
        "tag": args.tag,
        "runs": str(runs_path),
        "counts": {
            "dfdc_calibration": len(calibration),
            "dfdc_holdout": len(holdout),
            "swan_df_test": len(swan),
        },
        "statistic_comparison": statistic_comparison,
        "paired_within_source": {key: value for key, value in paired.items() if key != "details"},
        "paired_details": paired["details"],
        "video_band_thresholds": {
            "statistic": "median",
            "low_max": low_max,
            "low_max_target_fpr": 0.05,
            "low_max_realized_fpr": round(realized_low, 4),
            "medium_max": medium_max,
            "medium_max_target_fpr": 0.01,
            "medium_max_realized_fpr": round(realized_medium, 4),
            "calibration_real_n": len(calibration_real),
            "calibration_fake_n": len(calibration_fake),
        },
        "rule_evaluation": rule_evaluation,
        "swan_df_records": [
            {
                "file_name": record["file_name"],
                "label": record["label"],
                "role": record.get("role"),
                "median": record["stats"].get("median"),
                "frames_high": record["frame_bands"]["high"],
                "majority_band": (record.get("deepfake") or {}).get("band"),
            }
            for record in swan
        ],
    }
    (out_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis["video_band_thresholds"], ensure_ascii=False, indent=2))
    for name, entry in statistic_comparison.items():
        print(f"[统计量] {name}: " + json.dumps(entry, ensure_ascii=False))
    print("[配对] " + json.dumps({k: v for k, v in paired.items() if k != "details"}, ensure_ascii=False))
    for name, entry in rule_evaluation.items():
        print(f"[规则] {name}: " + json.dumps(entry, ensure_ascii=False))
    print(f"[完成] 写出 {out_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
