# -*- coding: utf-8 -*-
"""M3：跨域校准迁移实验——回答「固定/学习到的融合权重能不能跨域」。
协议：
  校准集 = data_formal/val（150 张，112px 对齐裁剪，独立数据源）
  测试集 = data_formal/test（同域，150 张）+ SWAN-DF 案例1（跨域，100 张整图）
  校准集上拟合融合权重与阈值，全部指标只在测试集上报告。
用法：
    python cross_domain_eval.py
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
BASELINE = UPGRADE_ROOT / "experiments" / "baseline" / "aigc"
LOG = UPGRADE_ROOT / "experiments" / "logs" / "m3_cross_domain.log"


def log(message: str) -> None:
    print(f"[cross-domain] {message}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def logit(values, eps=1e-12):
    values = np.clip(np.asarray(values, dtype=float), eps, 1 - eps)
    return np.log(values / (1 - values))


def load_formal(folder: Path):
    """读 v0.4.0 与 CF 在同一 data_formal 划分上的分数，按文件名对齐。"""
    v040 = {}
    with open(folder / "scores.csv", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            v040[row["file"]] = (int(row["y"]), float(row["score"]), float(row["spatial"]),
                                 float(row["frequency"]))
    cf = {}
    with open(M3 / ("cf_" + folder.name.replace("v040_formal_", "")) / "scores.csv",
              encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            cf[row["file"]] = float(row["score"])
    keys = sorted(set(v040) & set(cf))
    return {
        "keys": keys,
        "y": np.array([v040[key][0] for key in keys]),
        "v040": np.array([v040[key][1] for key in keys]),
        "spatial": np.array([v040[key][2] for key in keys]),
        "frequency": np.array([v040[key][3] for key in keys]),
        "cf": np.array([cf[key] for key in keys]),
    }


def load_swan():
    v040, spatial, frequency, y_map = {}, {}, {}, {}
    for path in sorted(BASELINE.rglob("detection_result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        details = data.get("details") or {}
        key = f"{path.parent.parent.name}/{path.parent.name}"
        v040[key] = float(details.get("fusion_score", data.get("score")) or 0.0)
        spatial[key] = float(details.get("spatial_score") or 0.0)
        frequency[key] = float(details.get("frequency_score") or 0.0)
        y_map[key] = 1 if path.parent.name.startswith("05") else 0
    cf = {}
    with open(M3 / "communityforensics_vit" / "scores.csv", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            cf[f"{row['case']}/{Path(row['file']).stem}"] = float(row["score"])
    keys = sorted(set(v040) & set(cf))
    return {
        "keys": keys,
        "y": np.array([y_map[key] for key in keys]),
        "v040": np.array([v040[key] for key in keys]),
        "spatial": np.array([spatial[key] for key in keys]),
        "frequency": np.array([frequency[key] for key in keys]),
        "cf": np.array([cf[key] for key in keys]),
    }


def fit(calibration, use_logit=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = np.column_stack([
        logit(calibration["v040"]) if use_logit else calibration["v040"],
        logit(calibration["cf"]) if use_logit else calibration["cf"],
    ])
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(max_iter=5000, C=1.0)
    model.fit(scaler.transform(features), calibration["y"])
    return model, scaler


def apply_fusion(model, scaler, data, use_logit=True):
    features = np.column_stack([
        logit(data["v040"]) if use_logit else data["v040"],
        logit(data["cf"]) if use_logit else data["cf"],
    ])
    return model.predict_proba(scaler.transform(features))[:, 1]


def pick_threshold(y, scores, target_fpr=0.05):
    """在校准集上按目标假阳性率取阈值（工程上比 Youden 更贴合研判场景）。"""
    negatives = np.sort(scores[y == 0])
    if negatives.size == 0:
        return 0.5
    index = int(np.ceil((1 - target_fpr) * negatives.size)) - 1
    return float(negatives[max(index, 0)])


def report(y, scores, threshold):
    auc = roc_auc(y, scores)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auc": round(auc, 4),
        "threshold": round(float(threshold), 6),
        "recall": round(tp / max(tp + fn, 1), 4),
        "fpr": round(fp / max(fp + tn, 1), 4),
        "accuracy": round((tp + tn) / len(y), 4),
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def main() -> None:
    calibration = load_formal(M3 / "v040_formal_val")
    in_domain = load_formal(M3 / "v040_formal_test")
    out_domain = load_swan()
    log(f"校准集 n={len(calibration['y'])}（data_formal/val）")
    log(f"同域测试 n={len(in_domain['y'])}（data_formal/test）"
        f" 跨域测试 n={len(out_domain['y'])}（SWAN-DF 案例1）")

    model, scaler = fit(calibration)
    weights = model.coef_[0].tolist()
    log(f"校准集拟合权重（标准化后）: v0.4.0={weights[0]:+.3f}  CF={weights[1]:+.3f}  "
        f"截距={model.intercept_[0]:+.3f}")

    result = {"calibration_set": "data_formal/val", "fusion_weights": [round(w, 4) for w in weights],
              "tests": {}}
    for name, data in (("in_domain_formal_test", in_domain), ("out_domain_swan_df", out_domain)):
        fused = apply_fusion(model, scaler, data)
        calibrated_threshold = pick_threshold(calibration["y"],
                                              apply_fusion(model, scaler, calibration),
                                              target_fpr=0.05)
        v040_threshold = pick_threshold(calibration["y"], calibration["v040"], target_fpr=0.05)
        cf_threshold = pick_threshold(calibration["y"], calibration["cf"], target_fpr=0.05)
        entry = {
            "n": len(data["y"]), "n_fake": int(data["y"].sum()),
            "fused": report(data["y"], fused, calibrated_threshold),
            "v040_alone": report(data["y"], data["v040"], v040_threshold),
            "cf_alone": report(data["y"], data["cf"], cf_threshold),
        }
        entry["fused"]["auc_ci95"] = [round(v, 4) for v in bootstrap_auc_ci(data["y"], fused)]
        result["tests"][name] = entry
        log(f"--- {name} ---")
        for key in ("v040_alone", "cf_alone", "fused"):
            row = entry[key]
            log(f"  {key:12s} AUC={row['auc']:.4f} 阈值={row['threshold']} "
                f"召回={row['recall']} FPR={row['fpr']} {row['confusion']}")

    (M3 / "cross_domain").mkdir(parents=True, exist_ok=True)
    (M3 / "cross_domain" / "cross_domain_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log("写入 cross_domain/cross_domain_metrics.json")


if __name__ == "__main__":
    main()
