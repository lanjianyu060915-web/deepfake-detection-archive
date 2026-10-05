# -*- coding: utf-8 -*-
"""M3：在任意「fake/ + real/」结构的图片目录上跑深伪检测器，输出分数与指标。
用于 data_formal 的 val（校准集）与 test（独立测试集），与 SWAN-DF（外部集）分开。
用法：
    python eval_dataset.py --model <repo_id> --root <含 fake/real 的目录> --tag <名字>
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
from eval_swan_df import bootstrap_auc as bootstrap_auc_ci  # noqa: E402
from eval_swan_df import resolve_fake_index, roc_auc  # noqa: E402

bootstrap()

OUT_ROOT = UPGRADE_ROOT / "experiments" / "m3_detector"
LOG = UPGRADE_ROOT / "experiments" / "logs" / "m3_dataset_eval.log"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def collect(root: Path):
    samples = []
    for label_name, label in (("fake", 1), ("real", 0)):
        folder = root / label_name
        if not folder.is_dir():
            raise SystemExit(f"缺少目录 {folder}")
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((folder.name, image, label))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    root = Path(args.root)
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model)
    model.eval().to(device)
    id2label = getattr(model.config, "id2label", None) or {0: "LABEL_0", 1: "LABEL_1"}
    fake_index, reason = resolve_fake_index(id2label)
    log(f"===== {args.tag} root={root} model={args.model} device={device} fake_index={fake_index} =====")

    samples = collect(root)
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
        for (folder, image, label), score in zip(chunk, scores.tolist()):
            rows.append({"folder": folder, "file": image.name, "y": label,
                         "score": round(float(score), 8)})
        log(f"  进度 {min(start + args.batch_size, len(samples))}/{len(samples)}")

    csv_path = out_dir / "scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["folder", "file", "y", "score"])
        writer.writeheader()
        writer.writerows(rows)

    y = np.array([row["y"] for row in rows])
    scores = np.array([row["score"] for row in rows])
    auc = roc_auc(y, scores)
    low, high = bootstrap_auc_ci(y, scores)
    summary = {
        "tag": args.tag, "root": str(root), "model": args.model, "device": device,
        "fake_index": fake_index, "fake_index_reason": reason,
        "n_total": len(rows), "n_fake": int(y.sum()),
        "auc": round(auc, 6), "auc_ci95": [round(low, 6), round(high, 6)],
        "real_mean": round(float(scores[y == 0].mean()), 8),
        "real_max": round(float(scores[y == 0].max()), 8),
        "real_p95": round(float(np.percentile(scores[y == 0], 95)), 8),
        "real_p99": round(float(np.percentile(scores[y == 0], 99)), 8),
        "fake_mean": round(float(scores[y == 1].mean()), 8),
        "fake_median": round(float(np.median(scores[y == 1])), 8),
        "fake_min": round(float(scores[y == 1].min()), 8),
        "fake_p05": round(float(np.percentile(scores[y == 1], 5)), 8),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"结果: AUC={auc:.6f} CI95=[{low:.4f},{high:.4f}] REAL均值={summary['real_mean']} "
        f"FAKE中位数={summary['fake_median']}")
    log(f"写入 {csv_path}")


if __name__ == "__main__":
    main()
