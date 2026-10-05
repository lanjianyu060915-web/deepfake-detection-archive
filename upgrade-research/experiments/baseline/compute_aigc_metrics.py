# -*- coding: utf-8 -*-
"""汇总 AIGC 批量结果，计算混淆矩阵、AUC、分组统计。"""
import argparse
import json
from pathlib import Path

THRESHOLD = 0.36360106194654707
VERSION_BY_CASE = {
    "320px": ["案例01", "案例05", "案例08", "案例12", "案例15", "案例19"],
    "256px": ["案例02", "案例06", "案例09", "案例13", "案例16", "案例20"],
    "160px-A": ["案例03", "案例07", "案例10", "案例14", "案例17"],
    "160px-B": ["案例04", "案例11", "案例18"],
}


def auc(scores, labels) -> float:
    pairs = sorted(zip(scores, labels))
    ranks = {}
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        ranks[pairs[index][0]] = average_rank
        index = end + 1
    positives = [ranks[score] for score, label in pairs if label == 1]
    n_pos = len(positives)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (sum(positives) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def stats(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    def quantile(q):
        position = q * (n - 1)
        lower = int(position)
        upper = min(lower + 1, n - 1)
        fraction = position - lower
        return values[lower] * (1 - fraction) + values[upper] * fraction
    return {
        "n": n, "mean": sum(values) / n, "median": quantile(0.5),
        "min": values[0], "max": values[-1],
        "p25": quantile(0.25), "p75": quantile(0.75),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "aigc"))
    parser.add_argument("--json", default=str(Path(__file__).parent / "aigc_metrics_baseline.json"))
    args = parser.parse_args()
    out_root = Path(args.out)

    rows = []
    for path in sorted(out_root.glob("案例*/*/detection_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case = path.parts[-3]
        image = path.parts[-2]
        label = 1 if image.startswith("05") else 0
        rows.append({
            "case": case, "image": image, "label": label,
            "score": payload.get("score"),
            "spatial": (payload.get("details") or {}).get("spatial_score"),
            "frequency": (payload.get("details") or {}).get("frequency_score"),
            "status": payload.get("status"),
            "predicted": payload.get("label"),
        })

    scored = [row for row in rows if isinstance(row["score"], (int, float))]
    scores = [float(row["score"]) for row in scored]
    labels = [int(row["label"]) for row in scored]
    tp = sum(1 for s, y in zip(scores, labels) if s >= THRESHOLD and y == 1)
    fp = sum(1 for s, y in zip(scores, labels) if s >= THRESHOLD and y == 0)
    tn = sum(1 for s, y in zip(scores, labels) if s < THRESHOLD and y == 0)
    fn = sum(1 for s, y in zip(scores, labels) if s < THRESHOLD and y == 1)
    accuracy = (tp + tn) / max(len(scored), 1)
    metrics = {
        "total_images": len(rows),
        "scored_images": len(scored),
        "threshold": THRESHOLD,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "auc": auc(scores, labels),
        "real_stats": stats([s for s, y in zip(scores, labels) if y == 0]),
        "fake_stats": stats([s for s, y in zip(scores, labels) if y == 1]),
        "per_version": {},
    }
    for version, cases in VERSION_BY_CASE.items():
        subset = [row for row in scored if row["case"] in cases]
        metrics["per_version"][version] = {
            "n_fake": sum(1 for row in subset if row["label"] == 1),
            "n_real": sum(1 for row in subset if row["label"] == 0),
            "auc": auc([float(row["score"]) for row in subset],
                       [int(row["label"]) for row in subset]),
        }
    Path(args.json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
