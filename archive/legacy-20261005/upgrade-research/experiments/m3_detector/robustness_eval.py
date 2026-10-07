# -*- coding: utf-8 -*-
r"""M3.1：二代整图检测器在真实退化条件下的鲁棒性评测。

动机（改进提案第 1 条未完成项）：M3 只证明了整图 ViT 分支在**原始**外部数据上有效
（SWAN-DF 案例11-20 留出集 AUC 0.9925）。但涉案素材在进入系统之前通常已经过一次或多次
有损处理——微信/抖音二次编码、平台缩放、截图、翻拍。这些处理会不会把检测器依赖的信号抹掉，
必须实测，不能假设。

评测纪律：
  * 退化函数是确定性的（噪声用固定种子），原始图像只读；
  * 主指标是**留出集（案例11-20）AUC**，全量 100 张 AUC 一并给出但与定标案例混在一起，仅作参考；
  * 分带是上线口径，退化后真人图分布会漂移，因此额外报告「沿用冻结分带时的真人误报率」；
  * 不翻转分数方向，不做任何按测试集调参。

用法（必须用二代运行时，即随包 Python）：
    & 'D:\刑科技项目升级2.0\workspace\fsts\packaging_v0.4\resources\portable_python_v2\python.exe' `
      robustness_eval.py [--variants original jpeg_q85 ...] [--keep-inputs]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


UPGRADE_ROOT = Path(r"D:\刑科技项目升级2.0")
WORKSPACE = UPGRADE_ROOT / "workspace" / "fsts"
DATASET = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
OUT_DIR = UPGRADE_ROOT / "experiments" / "m3_detector" / "robustness"
NOISE_SEED = 20260920
HOLDOUT_CASES = {f"案例{index:02d}" for index in range(11, 21)}

sys.path.insert(0, str(WORKSPACE / "modules" / "deepfake_detection_v2"))


def as_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return handle.convert("RGB")


def resampled(image: Image.Image, factor: float, *, restore: bool) -> Image.Image:
    width, height = image.size
    small = (max(1, round(width * factor)), max(1, round(height * factor)))
    resized = image.resize(small, Image.BILINEAR)
    if restore:
        resized = resized.resize((width, height), Image.BICUBIC)
    return resized


def rephotographed(image: Image.Image) -> Image.Image:
    """翻拍近似：失焦 + 传感器噪声 + 有损保存。明确是近似，不是真实翻拍采集。"""

    blurred = image.filter(ImageFilter.GaussianBlur(1.2))
    array = np.asarray(blurred, dtype=np.float32)
    rng = np.random.default_rng(NOISE_SEED)
    array = array + rng.normal(0.0, 4.0, array.shape).astype(np.float32)
    array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return as_jpeg(Image.fromarray(array, mode="RGB"), 70)


def double_jpeg(image: Image.Image) -> Image.Image:
    """视频二次编码近似：JPEG 60 -> 0.6 倍缩放 -> JPEG 85。真实视频编码需原始视频。"""

    first = as_jpeg(image, 60)
    scaled = resampled(first, 0.6, restore=False)
    return as_jpeg(scaled, 85)


VARIANTS = {
    "original": (lambda image: image.copy(), "JPEG", {}),
    "jpeg_q95": (lambda image: as_jpeg(image, 95), "JPEG", {}),
    "jpeg_q85": (lambda image: as_jpeg(image, 85), "JPEG", {}),
    "jpeg_q70": (lambda image: as_jpeg(image, 70), "JPEG", {}),
    "jpeg_q50": (lambda image: as_jpeg(image, 50), "JPEG", {}),
    "downup_50": (lambda image: resampled(image, 0.50, restore=True), "PNG", {}),
    "downup_25": (lambda image: resampled(image, 0.25, restore=True), "PNG", {}),
    "downscale_50": (lambda image: resampled(image, 0.50, restore=False), "PNG", {}),
    "downscale_25": (lambda image: resampled(image, 0.25, restore=False), "PNG", {}),
    "rephoto_proxy": (rephotographed, "JPEG", {}),
    "double_jpeg_proxy": (double_jpeg, "JPEG", {}),
}

VARIANT_NOTES = {
    "original": "对照组：原图直接送检",
    "jpeg_q95": "平台轻度重压缩",
    "jpeg_q85": "常见社交平台默认质量",
    "jpeg_q70": "多次转发后的典型质量",
    "jpeg_q50": "强压缩，接近缩略图质量",
    "downup_50": "缩小一半再还原（像素被破坏，尺寸不变）",
    "downup_25": "缩小到四分之一再还原（像素破坏更重，尺寸不变）",
    "downscale_50": "只缩小到一半（短边 360，仍在整图分支路由内）",
    "downscale_25": "只缩小到四分之一（短边 180，按现行路由会切到冻结人脸裁剪分支）",
    "rephoto_proxy": "翻拍近似：高斯模糊 1.2 + 噪声 sigma=4 + JPEG 70",
    "double_jpeg_proxy": "视频二次编码近似：JPEG 60 -> 0.6 倍 -> JPEG 85",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def collect_samples() -> list[tuple[str, Path, int]]:
    samples: list[tuple[str, Path, int]] = []
    for case_dir in sorted(DATASET.glob("案例*")):
        for image in sorted(case_dir.glob("*.jpg")):
            samples.append((case_dir.name, image, 1 if image.name.startswith("05") else 0))
    return samples


def roc_auc(y_true, scores) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    index = 0
    while index < len(sorted_scores):
        end = index
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        ranks[order[index:end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    rank_sum = ranks[y_true == 1].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def bootstrap_auc(y_true, scores, rounds: int = 2000, seed: int = NOISE_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    stats = []
    for _ in range(rounds):
        idx = rng.integers(0, len(y_true), len(y_true))
        labels = y_true[idx]
        if labels.min() == labels.max():
            continue
        stats.append(roc_auc(labels, scores[idx]))
    if not stats:
        return (float("nan"), float("nan"))
    stats = np.sort(np.asarray(stats))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def main() -> int:
    parser = argparse.ArgumentParser(description="二代检测器鲁棒性评测")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--keep-inputs", action="store_true", help="保留退化后的图片")
    args = parser.parse_args()

    from vit_detector import CommunityForensicsDetector  # noqa: E402

    detector = CommunityForensicsDetector.from_app_root(WORKSPACE)
    log(
        f"检测器已加载：短边 {detector.shortest_edge} 裁剪 {detector.crop_size}，"
        f"分带 low<={detector.low_max:.9f} medium<={detector.medium_max:.9f}，"
        f"模型 sha256={detector.model_sha256[:16]}..."
    )

    samples = collect_samples()
    out_dir = args.out.expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_root = out_dir / "_inputs"
    if args.keep_inputs:
        inputs_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    summaries: list[dict] = []
    for variant in args.variants:
        transform, fmt, _meta = VARIANTS[variant]
        work_dir = inputs_root / variant
        work_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        scores = []
        short_edges = []
        for case, source, label in samples:
            with Image.open(source) as handle:
                image = handle.convert("RGB")
            if variant == "original":
                # 对照组直接用原文件，避免任何重编码把基线本身也污染掉。
                target = source
            else:
                degraded = transform(image)
                target = work_dir / f"{case}_{source.stem}.{fmt.lower()}"
                save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
                degraded.save(target, format=fmt, **save_kwargs)
            result = detector.predict(target)
            scores.append(float(result["score"]))
            short_edges.append(int(result["short_edge"]))
            rows.append(
                {
                    "variant": variant,
                    "case": case,
                    "file": source.name,
                    "expected": "FAKE" if label == 1 else "REAL",
                    "y": label,
                    "split": "holdout" if case in HOLDOUT_CASES else "calibration",
                    "short_edge": int(result["short_edge"]),
                    "score": round(float(result["score"]), 9),
                }
            )
        if not args.keep_inputs:
            shutil.rmtree(work_dir, ignore_errors=True)

        y = np.array([row["y"] for row in rows[-len(samples):]])
        score_array = np.array(scores)
        holdout = np.array([row["split"] == "holdout" for row in rows[-len(samples):]])
        fake = y == 1
        real = y == 0

        holdout_auc = roc_auc(y[holdout], score_array[holdout])
        holdout_ci = bootstrap_auc(y[holdout], score_array[holdout])
        full_auc = roc_auc(y, score_array)
        min_short_edge = int(min(short_edges))

        summary = {
            "variant": variant,
            "note": VARIANT_NOTES[variant],
            "n": len(samples),
            "short_edge_min": min_short_edge,
            "short_edge_max": int(max(short_edges)),
            "routing_branch": "vit_full_image" if min_short_edge >= 256 else "mixed(部分短边<256将切到冻结分支)",
            "auc_holdout_cases_11_20": round(holdout_auc, 6),
            "auc_holdout_ci95": [round(value, 6) for value in holdout_ci],
            "auc_full_100": round(full_auc, 6),
            "holdout_auc_delta_vs_original": None,
            "real_mean": round(float(score_array[real].mean()), 9),
            "real_p95": round(float(np.percentile(score_array[real], 95)), 9),
            "real_max": round(float(score_array[real].max()), 9),
            "fake_median": round(float(np.median(score_array[fake])), 9),
            "holdout_fake_min": round(float(score_array[holdout & fake].min()), 9),
            "real_above_frozen_low_max": int((score_array[real] > detector.low_max).sum()),
            "real_above_frozen_medium_max": int((score_array[real] > detector.medium_max).sum()),
            "real_false_alarm_rate_frozen_bands": round(
                float((score_array[real] > detector.medium_max).mean()), 6
            ),
            "fake_flagged_high_frozen": int((score_array[fake] > detector.medium_max).sum()),
            "fake_recall_frozen_high": round(
                float((score_array[fake] > detector.medium_max).mean()), 6
            ),
            "holdout_fake_flagged_high_frozen": int(
                (score_array[holdout & fake] > detector.medium_max).sum()
            ),
            "holdout_fake_n": int((holdout & fake).sum()),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        summaries.append(summary)
        log(
            f"{variant:>18}  留出AUC={summary['auc_holdout_cases_11_20']:.4f} "
            f"全量AUC={full_auc:.4f} 短边min={min_short_edge} "
            f"真人误报(冻结分带)={summary['real_false_alarm_rate_frozen_bands']:.3f} "
            f"{summary['elapsed_seconds']}s"
        )

    baseline = next(
        (item for item in summaries if item["variant"] == "original"),
        summaries[0] if summaries else None,
    )
    if baseline is not None:
        for item in summaries:
            item["holdout_auc_delta_vs_original"] = round(
                item["auc_holdout_cases_11_20"] - baseline["auc_holdout_cases_11_20"], 6
            )

    with open(out_dir / "robustness_scores.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "case", "file", "expected", "y", "split", "short_edge", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": "1.0",
        "module": "deepfake_detection_v2",
        "branch": "vit_full_image",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(DATASET),
        "noise_seed": NOISE_SEED,
        "detector": {
            "shortest_edge": detector.shortest_edge,
            "crop_size": detector.crop_size,
            "model_sha256": detector.model_sha256,
            "bands": {"low_max": detector.low_max, "medium_max": detector.medium_max},
        },
        "primary_metric": "auc_holdout_cases_11_20",
        "variants": summaries,
        "elapsed_seconds": round(sum(item["elapsed_seconds"] for item in summaries), 1),
    }
    (out_dir / "robustness_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"写入 {out_dir / 'robustness_summary.json'} 与 robustness_scores.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
