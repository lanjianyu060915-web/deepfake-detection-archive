# -*- coding: utf-8 -*-
"""M3：把 v0.4.0 冻结鉴伪分与二代检测器分放到同一张表，做分组交叉验证评估。
评估纪律：
  * 100 张图来自同一批 20 个案例，正负样本同源，因此一律按「案例」分组做
    留一案例交叉验证（LOCO-CV），避免同案例信息泄漏；
  * 所有融合权重/阈值都在训练折内拟合，报告的是折外（OOF）成绩；
  * AUC 对单调变换不变，故各信号可直接比较。
用法：
    python fusion_eval.py --v2 <tag> [--v2 <tag2> ...] [--out 结果目录名]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import bootstrap, UPGRADE_ROOT  # noqa: E402

bootstrap()

BASELINE_DIR = UPGRADE_ROOT / "experiments" / "baseline" / "aigc"
M3_DIR = UPGRADE_ROOT / "experiments" / "m3_detector"
LOG = UPGRADE_ROOT / "experiments" / "logs" / "m3_fusion.log"


def log(message: str) -> None:
    print(f"[fusion] {message}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def load_v040():
    table = {}
    for path in sorted(BASELINE_DIR.rglob("detection_result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        details = data.get("details") or {}
        case = path.parent.parent.name
        image = path.parent.name
        table[(case, image)] = {
            "y": 1 if image.startswith("05") else 0,
            "v040_fusion": float(details.get("fusion_score", data.get("score")) or 0.0),
            "v040_spatial": float(details.get("spatial_score") or 0.0),
            "v040_frequency": float(details.get("frequency_score") or 0.0),
            "bbox": (details.get("frame_results") or [{}])[0].get("raw_face_bbox_xywh"),
            "image_size": ((details.get("frame_results") or [{}])[0].get("original_size")
                           or [details.get("media_metadata", {}).get("width"),
                               details.get("media_metadata", {}).get("height")]),
        }
    return table


def load_v2(tag):
    path = M3_DIR / tag / "scores.csv"
    table = {}
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row["file"]).stem
            table[(row["case"], stem)] = float(row["score"])
    return table


def roc_auc(y_true, scores) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = ranks[y_true == 1].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def logit(values, eps=1e-9):
    values = np.clip(np.asarray(values, dtype=float), eps, 1 - eps)
    return np.log(values / (1 - values))


def rank_normalize(train_values, test_values):
    """把 test 值按其落在 train 分布中的分位映射到 (0,1)，对极端尺度鲁棒。"""
    ordered = np.sort(np.asarray(train_values, dtype=float))
    ranks = np.searchsorted(ordered, np.asarray(test_values, dtype=float), side="left")
    return (ranks + 0.5) / (len(ordered) + 1.0)


def loco_cv(features, y, groups, transform="rank", seed=20260920):
    """留一案例交叉验证：折内做特征变换 + 逻辑回归，返回 OOF 分数与平均系数。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(features, dtype=float)
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    coefs = []
    for case in sorted(set(groups)):
        test = np.array([group == case for group in groups])
        train = ~test
        if transform == "rank":
            train_x = np.column_stack([
                rank_normalize(features[train, col], features[train, col])
                for col in range(features.shape[1])
            ])
            test_x = np.column_stack([
                rank_normalize(features[train, col], features[test, col])
                for col in range(features.shape[1])
            ])
        elif transform == "logit":
            train_x = np.column_stack([
                logit(features[train, col]) for col in range(features.shape[1])
            ])
            test_x = np.column_stack([
                logit(features[test, col]) for col in range(features.shape[1])
            ])
        else:
            train_x, test_x = features[train], features[test]
        scaler = StandardScaler().fit(train_x)
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(scaler.transform(train_x), y[train])
        oof[test] = model.predict_proba(scaler.transform(test_x))[:, 1]
        coefs.append(model.coef_[0].tolist())
    return oof, np.mean(coefs, axis=0).tolist()


def bootstrap_auc_ci(y, scores, rounds=4000, seed=20260920):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    stats = []
    n = len(y)
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        stats.append(roc_auc(y[idx], scores[idx]))
    if not stats:
        return (float("nan"), float("nan"))
    stats = np.sort(np.asarray(stats))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def paired_bootstrap_p(y, scores_a, scores_b, rounds=4000, seed=20260920):
    """H0: 两个打分器 AUC 相同。返回双侧 bootstrap p 值。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    observed = roc_auc(y, scores_a) - roc_auc(y, scores_b)
    n = len(y)
    diffs = []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc(y[idx], scores_a[idx]) - roc_auc(y[idx], scores_b[idx]))
    diffs = np.asarray(diffs)
    centered = diffs - diffs.mean()
    p = float((np.abs(centered) >= abs(observed)).mean())
    return observed, p


def threshold_report(y, scores):
    """在给定分数上取最佳阈值，仅作可分性刻画，不作为部署阈值。"""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(scores)
    best = (float(np.median(scores)), -1.0, None)
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        acc = (tp + tn) / len(y)
        if acc > best[1]:
            best = (float(threshold), acc, {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
                                            "recall": round(tp / max(tp + fn, 1), 4),
                                            "fpr": round(fp / max(fp + tn, 1), 4)})
    return {"threshold": best[0], "accuracy": round(best[1], 4), "confusion": best[2]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="append", default=[])
    parser.add_argument("--out", default="fusion_v040_v2")
    parser.add_argument("--transform", choices=("rank", "logit", "none"), default="rank")
    args = parser.parse_args()

    base = load_v040()
    log(f"v0.4.0 基线样本数={len(base)}")

    keys = sorted(base)
    y = np.array([base[key]["y"] for key in keys])
    groups = [key[0] for key in keys]
    signals = {
        "v040_fusion": np.array([base[key]["v040_fusion"] for key in keys]),
        "v040_spatial": np.array([base[key]["v040_spatial"] for key in keys]),
        "v040_frequency": np.array([base[key]["v040_frequency"] for key in keys]),
    }
    for tag in args.v2:
        table = load_v2(tag)
        signals[f"v2_{tag}"] = np.array([
            table.get(key, float("nan")) for key in keys
        ])
        missing = int(np.isnan(signals[f"v2_{tag}"]).sum())
        log(f"载入 v2 检测器 {tag}: 缺失={missing}")
        if missing:
            raise SystemExit(f"{tag} 分数与基线样本未完全对齐，缺失 {missing}")

    log("== 单信号 AUC（全量，仅作参考） ==")
    single = {}
    for name, values in signals.items():
        single[name] = round(roc_auc(y, values), 6)
        log(f"  {name}: AUC={single[name]}")

    result = {"n": len(keys), "n_fake": int(y.sum()), "single_auc": single, "combinations": {}}
    v2_tags = [tag for tag in args.v2]

    combos = {
        "v2_only": [f"v2_{tag}" for tag in v2_tags],
        "v2_plus_frequency": [f"v2_{tag}" for tag in v2_tags] + ["v040_frequency"],
        "v2_plus_v040fusion": [f"v2_{tag}" for tag in v2_tags] + ["v040_fusion"],
        "v2_plus_frequency_plus_spatial": [f"v2_{tag}" for tag in v2_tags]
        + ["v040_frequency", "v040_spatial"],
    }
    log(f"== 分组留一案例交叉验证（变换={args.transform}，融合权重折内拟合，报折外成绩） ==")
    oof_store = {}
    for name, feature_names in combos.items():
        if not feature_names:
            continue
        raw = np.column_stack([signals[feature] for feature in feature_names])
        oof, coefs = loco_cv(raw, y, groups, transform=args.transform)
        oof_store[name] = oof
        auc = roc_auc(y, oof)
        low, high = bootstrap_auc_ci(y, oof)
        report = threshold_report(y, oof)
        result["combinations"][name] = {
            "features": feature_names,
            "oof_auc": round(auc, 6),
            "oof_auc_ci95": [round(low, 6), round(high, 6)],
            "mean_coef": [round(c, 4) for c in coefs],
            "threshold_report": report,
        }
        log(f"  {name}: OOF AUC={auc:.6f} CI=[{low:.3f},{high:.3f}] "
            f"均值系数={[round(c, 3) for c in coefs]} 最佳阈值准确率={report['accuracy']} "
            f"{report['confusion']}")

    reference = "v2_only"
    if reference in oof_store:
        log(f"== 与 {reference} 的成对 bootstrap 检验（H0: AUC 相同） ==")
        for name, oof in oof_store.items():
            if name == reference:
                continue
            diff, p_value = paired_bootstrap_p(y, oof, oof_store[reference])
            result["combinations"][name]["vs_v2_only"] = {
                "auc_diff": round(diff, 6), "p_value": round(p_value, 4),
            }
            log(f"  {name} - {reference}: ΔAUC={diff:+.4f} p={p_value:.3f}")

    out_dir = M3_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fusion_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"写入 {out_dir / 'fusion_metrics.json'}")


if __name__ == "__main__":
    main()
