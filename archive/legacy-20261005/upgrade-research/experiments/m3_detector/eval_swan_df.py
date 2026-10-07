# -*- coding: utf-8 -*-
"""M3：二代深伪检测器在 SWAN-DF 案例1（100 张：20 FAKE + 80 REAL）上的评测。
评估纪律：
  * 标签方向只依据模型卡 / config.id2label 判定，不根据 SWAN-DF 结果翻转；
  * 同时输出原始方向与反向两套 AUC，反向仅作诊断，不作为成绩；
  * AUC 附带 bootstrap 95% 置信区间。
用法：
    python eval_swan_df.py --model <repo_id 或本地路径> [--device cuda] [--tag run1]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import bootstrap, UPGRADE_ROOT  # noqa: E402

bootstrap()

DATASET = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
OUT_ROOT = UPGRADE_ROOT / "experiments" / "m3_detector"
LOG = UPGRADE_ROOT / "experiments" / "logs" / "m3_eval.log"

FAKE_TOKENS = ("fake", "deepfake", "ai", "synthetic", "generated", "artificial", "manipulated")
REAL_TOKENS = ("real", "realism", "authentic", "pristine", "genuine", "natural")

SOURCE_TABLE = {
    "320px": ["案例01", "案例05", "案例08", "案例12", "案例15", "案例19"],
    "256px": ["案例02", "案例06", "案例09", "案例13", "案例16", "案例20"],
    "160px-A": ["案例03", "案例07", "案例10", "案例14", "案例17"],
    "160px-B": ["案例04", "案例11", "案例18"],
}


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def collect_samples():
    samples = []
    for case_dir in sorted(DATASET.glob("案例*")):
        for image in sorted(case_dir.glob("*.jpg")):
            label = 1 if image.name.startswith("05") else 0
            samples.append((case_dir.name, image, label))
    return samples


def version_of(case_name: str) -> str:
    for version, cases in SOURCE_TABLE.items():
        if case_name in cases:
            return version
    return "unknown"


def resolve_fake_index(id2label: dict) -> tuple:
    """按 config.id2label 的标签文本判定 fake 类别下标，返回 (index, 依据说明)。"""
    mapping = {int(k): str(v) for k, v in id2label.items()}
    for index, text in sorted(mapping.items()):
        lowered = text.lower()
        if any(token in lowered for token in FAKE_TOKENS) and not any(
            token in lowered for token in REAL_TOKENS
        ):
            return index, f"id2label匹配: {index}->{text}"
    for index, text in sorted(mapping.items()):
        if any(token in text.lower() for token in REAL_TOKENS):
            if len(mapping) == 2:
                return 1 - index, f"由real标签反推: {index}->{text}"
            return 0, f"由real标签反推但类别数不为2: {index}->{text}"
    return (max(mapping) if mapping else 0), "无标签信息，取最后一个下标"


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


def bootstrap_auc(y_true, scores, rounds=2000, seed=20260920):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    stats = []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        stats.append(roc_auc(yt, scores[idx]))
    if not stats:
        return (float("nan"), float("nan"))
    stats = np.sort(np.asarray(stats))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def best_threshold(y_true, scores):
    """在 SWAN-DF 上求最佳阈值，仅刻画可分性上限，不作为部署阈值（属于事后统计）。"""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best = (0.5, -1.0)
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        acc = float((pred == y_true).mean())
        if acc > best[1]:
            best = (float(threshold), acc)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    tag = args.tag or args.model.replace("/", "__")
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    log(f"===== 评测开始 model={args.model} device={device} tag={tag} =====")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model)
    model.eval().to(device)

    config = model.config
    id2label = getattr(config, "id2label", None) or {0: "LABEL_0", 1: "LABEL_1"}
    num_labels = int(getattr(config, "num_labels", len(id2label)) or len(id2label))
    fake_index, reason = resolve_fake_index(id2label)
    log(f"config: num_labels={num_labels} id2label={id2label}")
    log(f"fake 类别下标={fake_index}（{reason}）")
    total_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数量={total_params / 1e6:.2f}M 架构={config.architectures}")

    samples = collect_samples()
    log(f"样本数={len(samples)}（FAKE={sum(item[2] for item in samples)}）")

    rows = []
    started = time.time()
    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start:start + args.batch_size]
        images = [Image.open(item[1]).convert("RGB") for item in chunk]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits.float()
        if logits.shape[-1] == 1:
            scores = torch.sigmoid(logits[:, 0])
        else:
            scores = torch.softmax(logits, dim=-1)[:, fake_index]
        for (case_name, image, label), score in zip(chunk, scores.tolist()):
            rows.append({
                "case": case_name,
                "file": image.name,
                "version": version_of(case_name) if label == 1 else "",
                "expected": "FAKE" if label == 1 else "REAL",
                "y": label,
                "score": round(float(score), 6),
            })
        done = min(start + args.batch_size, len(samples))
        log(f"  进度 {done}/{len(samples)} 已用 {time.time() - started:.1f}s")

    csv_path = out_dir / "scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "file", "version", "expected", "y", "score"])
        writer.writeheader()
        writer.writerows(rows)

    y = np.array([row["y"] for row in rows])
    scores = np.array([row["score"] for row in rows])
    auc = roc_auc(y, scores)
    auc_inv = roc_auc(y, -scores)
    low, high = bootstrap_auc(y, scores)
    thr, acc = best_threshold(y, scores)

    def at_threshold(threshold):
        pred = (scores >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        return {"threshold": threshold, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "accuracy": round((tp + tn) / len(y), 6),
                "recall": round(tp / max(tp + fn, 1), 6),
                "fpr": round(fp / max(fp + tn, 1), 6)}

    by_version = {}
    for version in list(SOURCE_TABLE):
        mask_fake = np.array([row["version"] == version for row in rows])
        if mask_fake.sum() == 0:
            continue
        sel = mask_fake | (y == 0)
        by_version[version] = {
            "n_fake": int(mask_fake.sum()),
            "auc": round(roc_auc(y[sel], scores[sel]), 6),
            "fake_mean": round(float(scores[mask_fake].mean()), 6),
            "fake_min": round(float(scores[mask_fake].min()), 6),
            "real_mean": round(float(scores[y == 0].mean()), 6),
            "real_max": round(float(scores[y == 0].max()), 6),
        }

    summary = {
        "model": args.model,
        "tag": tag,
        "device": device,
        "params_millions": round(total_params / 1e6, 3),
        "architectures": [str(a) for a in (config.architectures or [])],
        "id2label": {str(key): str(value) for key, value in id2label.items()},
        "fake_index": fake_index,
        "fake_index_reason": reason,
        "n_total": len(rows),
        "n_fake": int(y.sum()),
        "auc": round(auc, 6),
        "auc_ci95": [round(low, 6), round(high, 6)],
        "auc_inverted_diagnostic_only": round(auc_inv, 6),
        "best_threshold_on_this_set": thr,
        "best_accuracy_on_this_set": round(acc, 6),
        "at_0.5": at_threshold(0.5),
        "by_version": by_version,
        "real_mean": round(float(scores[y == 0].mean()), 6),
        "real_max": round(float(scores[y == 0].max()), 6),
        "fake_mean": round(float(scores[y == 1].mean()), 6),
        "fake_min": round(float(scores[y == 1].min()), 6),
        "elapsed_seconds": round(time.time() - started, 1),
        "baseline_v040_auc": 0.459375,
    }
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"结果: AUC={auc:.6f} CI95=[{low:.4f},{high:.4f}] 反向诊断AUC={auc_inv:.6f}")
    log(f"分数分布: REAL均值={summary['real_mean']} 最大={summary['real_max']} | "
        f"FAKE均值={summary['fake_mean']} 最小={summary['fake_min']}")
    log(f"写入 {csv_path} 与 {json_path}")


if __name__ == "__main__":
    main()
