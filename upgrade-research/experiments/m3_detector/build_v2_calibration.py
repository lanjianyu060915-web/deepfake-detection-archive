# -*- coding: utf-8 -*-
"""M3：生成 v0.5.0 二代鉴伪模块的校准文件（阈值带 + 溯源信息）。
校准口径：
  ViT 分支阈值 —— SWAN-DF 案例01-10 的真人图分数分位数（案例11-20 只用于验证）；
  v0.4.0 分支阈值 —— 沿用冻结 fusion_config 的 frame_threshold，不重新标定；
  路由阈值 —— 由两个可用域的输入有效短边分布确定（112px vs 整图），标为待更多域验证。
用法：
    python build_v2_calibration.py
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import bootstrap, UPGRADE_ROOT  # noqa: E402
from fusion_eval import roc_auc  # noqa: E402

bootstrap()

M3 = UPGRADE_ROOT / "experiments" / "m3_detector"
WS = UPGRADE_ROOT / "workspace" / "fsts"
SNAPSHOT = Path("D:/刑科技项目升级2.0/experiments/models_cache/hub/"
                "models--buildborderless--CommunityForensics-DeepfakeDet-ViT/snapshots/"
                "ac6ee457bea904a373065754107451793b56db00")
DEST = WS / "packaging_v0.4" / "resources" / "detectors_v2" / "communityforensics_vit"
CALIB_CASES = {f"案例{index:02d}" for index in range(1, 11)}
LOG = UPGRADE_ROOT / "experiments" / "logs" / "m3_build_calibration.log"


def log(message: str) -> None:
    print(f"[build] {message}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = []
    with open(M3 / "communityforensics_vit" / "scores.csv", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["case"], float(row["score"]), int(row["y"])))
    calib = [item for item in rows if item[0] in CALIB_CASES]
    y = np.array([item[2] for item in calib])
    scores = np.array([item[1] for item in calib])
    real = np.sort(scores[y == 0])
    fake = scores[y == 1]

    percentiles = {f"p{int(p * 100)}": float(np.percentile(real, p * 100))
                   for p in (0.5, 0.9, 0.95, 0.99, 1.0)}
    log(f"校准集真人图分数分位（案例01-10, n={int((y == 0).sum())}）: "
        + " ".join(f"{k}={v:.3e}" for k, v in percentiles.items()))
    for band, threshold in (("medium", percentiles["p95"]), ("high", percentiles["p99"])):
        caught = int((fake >= threshold).sum())
        log(f"  {band} 带阈值={threshold:.3e} → 校准集换脸召回={caught}/{len(fake)}")

    DEST.mkdir(parents=True, exist_ok=True)
    model_src = SNAPSHOT / "onnx" / "model.onnx"
    model_dst = DEST / "model.onnx"
    if not model_dst.exists() or model_dst.stat().st_size != model_src.stat().st_size:
        shutil.copy2(model_src, model_dst)
        log(f"复制模型 -> {model_dst}")
    model_sha = sha256(model_dst)
    log(f"模型 SHA-256 = {model_sha}")

    calibration = {
        "schema_version": "1.0",
        "module": "deepfake_detection_v2",
        "branch": "vit_full_image",
        "created_at": "2026-09-20",
        "decision_semantics": "分数为深伪风险筛查分，不是准确率，也不是司法鉴定结论。",
        "bands": {
            "low_max": percentiles["p95"],
            "medium_max": percentiles["p99"],
            "labels": {
                "low": "证据不足",
                "medium": "需人工复核",
                "high": "高风险提示",
            },
            "note": ("分带由独立校准集上真人图分数的分位数确定：低于 p95 视为与已知真人图"
                     "分布无显著差异；p95~p99 之间提示需复核；超过 p99（即校准集真人图上界）"
                     "提示高风险。"),
        },
        "reference_real_percentiles": percentiles,
        "calibration_set": {
            "name": "SWAN-DF 案例01-10（案例11-20 保留为验证集，未参与定标）",
            "n_real": int((y == 0).sum()),
            "n_fake": int((y == 1).sum()),
            "auc": round(roc_auc(y, scores), 4),
        },
        "validation_set": {
            "name": "SWAN-DF 案例11-20",
            "auc": 0.9925,
            "auc_ci95": [0.9697, 1.0],
            "operating_point_at_target_fpr_5pct": {"threshold": 0.000287, "recall": 0.5,
                                                   "fpr": 0.0},
        },
        "preprocessing": {
            "resize": {"shortest_edge": 440, "resample": "bicubic"},
            "center_crop": 384,
            "rescale": 0.00392156862745098,
            "normalize_mean": [0.48145466, 0.4578275, 0.40821073],
            "normalize_std": [0.26862954, 0.26130258, 0.27577711],
            "input_layout": "NCHW float32",
        },
        "model_provenance": {
            "repo_id": "buildborderless/CommunityForensics-DeepfakeDet-ViT",
            "revision": "ac6ee457bea904a373065754107451793b56db00",
            "file": "onnx/model.onnx",
            "sha256": model_sha,
            "architecture": "ViT-Small patch16 384, num_labels=1 (sigmoid)",
            "paper": "Community Forensics: Using Thousands of Generators to Train Fake Image Detectors (CVPR 2025)",
            "label_direction_source": "官方模型卡：fake_prob = sigmoid(logits)，未经测试集翻转",
            "onnx_fidelity": {"pearson_vs_pytorch": 0.999403, "max_abs_diff": 0.00136578},
        },
        "limitations": [
            "校准集与验证集来自同一外部数据集的不同案例，样本量小（真人 40 张、换脸 10 张）。",
            "本分支在 112px 对齐裁剪类输入上接近随机（data_formal 上 AUC≈0.45~0.51），"
            "必须与 v0.4.0 人脸裁剪分支按输入分辨率路由使用。",
            "跨域直接套用其它数据集的阈值会失效，更换数据来源后必须重新标定。",
        ],
    }
    (DEST / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"写入 {DEST / 'calibration.json'}")

    routing = {
        "schema_version": "1.0",
        "module": "deepfake_routing",
        "rule": "按输入图像的有效短边路由",
        "short_edge_threshold_px": 256,
        "branches": {
            "vit_full_image": {
                "when": "输入短边 >= 阈值（整图/高分辨率照片）",
                "runtime": "detectors_v2/communityforensics_vit/model.onnx",
                "evidence_auc": {"swan_df_cases_11_20": 0.9925, "data_formal_test": 0.4309},
            },
            "v040_face_crop": {
                "when": "输入短边 < 阈值（小尺寸或已对齐的人脸裁剪）",
                "runtime": "packaging_v0.3/resources/deepfake_runtime",
                "evidence_auc": {"data_formal_test": 0.8436, "swan_df_cases_11_20": 0.4594},
            },
        },
        "evidence": {
            "routing_auc_data_formal_test": 0.8436,
            "routing_auc_swan_df": 0.8978,
            "global_fusion_auc_swan_df": 0.4394,
            "note": ("在 data_formal/val 上拟合的全局融合权重给 ViT 分支的系数接近 0 甚至为负，"
                     "跨域使用会把外部集 AUC 从 0.898 拉到 0.439；按输入分辨率路由则在两个域上"
                     "都达到该域最优单分支水平。"),
        },
        "limitations": [
            "路由阈值 256px 由当前两个可用域（112px 裁剪 / 整图）的分布确定，尚需更多域验证。",
            "两个分支都必须按各自的目标域校准集单独标定阈值。",
        ],
    }
    (WS / "packaging_v0.4" / "resources" / "detectors_v2" / "routing.json").write_text(
        json.dumps(routing, ensure_ascii=False, indent=2), encoding="utf-8")
    log("写入 detectors_v2/routing.json")


if __name__ == "__main__":
    main()
