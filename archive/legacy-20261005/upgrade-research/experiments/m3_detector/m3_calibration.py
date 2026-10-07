# -*- coding: utf-8 -*-
"""M3：SWAN-DF 上「案例对半拆分」的阈值标定与折外评估。
把案例01-10 当校准集、案例11-20 当测试集，阈值只在校准集上选，成绩只在测试集上报。
另给出全量结果仅作参考（明确标注为不可用于论文）。
用法：
    python m3_calibration.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import bootstrap, UPGRADE_ROOT  # noqa: E402
from fusion_eval import bootstrap_auc_ci, roc_auc  # noqa: E402

bootstrap()

M3 = UPGRADE_ROOT / "experiments" / "m3_detector"
LOG = UPGRADE_ROOT / "experiments" / "logs" / "m3_calibration.log"
CALIB_CASES = {f"案例{index:02d}" for index in range(1, 11)}


def log(message: str) -> None:
    print(f"[calib] {message}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def load_scores():
    rows = []
    with open(M3 / "communityforensics_vit" / "scores.csv", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["case"], float(row["score"]), int(row["y"])))
    return rows


def threshold_at_fpr(y, scores, target_fpr):
    negatives = np.sort(np.asarray(scores)[np.asarray(y) == 0])
    if negatives.size == 0:
        raise ValueError("校准集没有负样本")
    index = int(np.ceil((1 - target_fpr) * negatives.size)) - 1
    return float(negatives[max(index, 0)])


def evaluate(y, scores, threshold):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"threshold": threshold, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "recall": round(tp / max(tp + fn, 1), 4),
            "fpr": round(fp / max(fp + tn, 1), 4),
            "accuracy": round((tp + tn) / len(y), 4)}


def main() -> None:
    rows = load_scores()
    calib = [(c, s, y) for c, s, y in rows if c in CALIB_CASES]
    test = [(c, s, y) for c, s, y in rows if c not in CALIB_CASES]
    log(f"校验集={len(calib)} 张（案例01-10） 测试集={len(test)} 张（案例11-20）")

    y_calib = [item[2] for item in calib]
    s_calib = [item[1] for item in calib]
    y_test = np.array([item[2] for item in test])
    s_test = np.array([item[1] for item in test])

    result = {"calibration_split": "案例01-10", "test_split": "案例11-20",
              "calibration_auc": round(roc_auc(y_calib, s_calib), 4),
              "test_auc": round(roc_auc(y_test, s_test), 4),
              "test_auc_ci95": [round(v, 4) for v in bootstrap_auc_ci(y_test, s_test)],
              "operating_points": {}}
    log(f"校准集 AUC={result['calibration_auc']} 测试集 AUC={result['test_auc']} "
        f"CI95={result['test_auc_ci95']}")

    for target_fpr in (0.01, 0.05, 0.10, 0.20):
        threshold = threshold_at_fpr(y_calib, s_calib, target_fpr)
        entry = evaluate(y_test, s_test, threshold)
        entry["calibration_realized_fpr"] = evaluate(y_calib, s_calib, threshold)["fpr"]
        entry["target_fpr"] = target_fpr
        result["operating_points"][f"fpr_{target_fpr}"] = entry
        log(f"  目标FPR={target_fpr:.0%} 阈值={threshold:.3e} → 测试集 召回={entry['recall']} "
            f"实测FPR={entry['fpr']} 准确率={entry['accuracy']} {entry['confusion']}"
            if "confusion" in entry else
            f"  目标FPR={target_fpr:.0%} 阈值={threshold:.3e} → 测试集 召回={entry['recall']} "
            f"实测FPR={entry['fpr']} 准确率={entry['accuracy']}")

    result["full_set_reference_only"] = {
        "note": "全量 100 张上的数值仅作参考，属于在同一集合上选阈值，不可用于正式报告",
        "auc": round(roc_auc([row[2] for row in rows], [row[1] for row in rows]), 6),
    }
    (M3 / "cross_domain").mkdir(parents=True, exist_ok=True)
    (M3 / "cross_domain" / "calibration_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log("写入 cross_domain/calibration_metrics.json")


if __name__ == "__main__":
    main()
