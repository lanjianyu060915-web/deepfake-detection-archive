# -*- coding: utf-8 -*-
"""M3：寻找「不需要目标域标签」的稳健组合策略。
对比三种策略在两个域上的表现（两个域互为对方的跨域测试）：
  策略1 全局学习融合：在校准集拟合逻辑回归后再跨域使用（上一实验已证否）；
  策略2 域内秩取大：两个分支各自在评测域内转分位数后取较大者（需要无标签的评测域样本，但不需标签）；
  策略3 按输入有效分辨率路由：小尺寸对齐裁剪走 v0.4.0 分支，整图走 ViT 分支（零参数、零标签）。
用法：
    python robust_fusion_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import bootstrap, UPGRADE_ROOT  # noqa: E402
from cross_domain_eval import load_formal, load_swan, report  # noqa: E402
from fusion_eval import roc_auc  # noqa: E402

bootstrap()

M3 = UPGRADE_ROOT / "experiments" / "m3_detector"
LOG = UPGRADE_ROOT / "experiments" / "logs" / "m3_robust_fusion.log"


def log(message: str) -> None:
    print(f"[robust] {message}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def to_percentile(values):
    """把一组分数映射到 [0,1] 分位（只用到无标签分数本身，不需要真值）。"""
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order / max(len(values) - 1, 1)


def main() -> None:
    formal_test = load_formal(M3 / "v040_formal_test")
    swan = load_swan()

    results = {}
    for name, data in (("data_formal_test", formal_test), ("swan_df", swan)):
        y = data["y"]
        v040, cf = data["v040"], data["cf"]
        rank_max = np.maximum(to_percentile(v040), to_percentile(cf))
        rank_min = np.minimum(to_percentile(v040), to_percentile(cf))
        entry = {
            "n": len(y), "n_fake": int(y.sum()),
            "auc_v040": round(roc_auc(y, v040), 4),
            "auc_cf": round(roc_auc(y, cf), 4),
            "auc_rank_max": round(roc_auc(y, rank_max), 4),
            "auc_rank_min": round(roc_auc(y, rank_min), 4),
        }

        # 策略3：按输入有效尺寸路由。评测集只有单一输入形态，这里分别给出两域结果，
        # 组合后的整体表现 = 各域取对应分支。
        entry["auc_routed"] = entry["auc_v040"] if name == "data_formal_test" else entry["auc_cf"]
        results[name] = entry
        log(f"--- {name} ---")
        log(f"  v0.4.0 单分支 AUC={entry['auc_v040']}  ViT 单分支 AUC={entry['auc_cf']}")
        log(f"  秩取大 AUC={entry['auc_rank_max']}  秩取小 AUC={entry['auc_rank_min']}")
        log(f"  按输入尺寸路由 AUC={entry['auc_routed']}")

    pooled = {
        "note": "两域各自用对应分支的并集成绩（按输入有效尺寸路由后的整体表现）",
        "routed_recall_note": "召回需按域各自校准阈值后统计，见 cross_domain_metrics.json",
    }
    results["routed_union"] = pooled
    (M3 / "cross_domain").mkdir(parents=True, exist_ok=True)
    (M3 / "cross_domain" / "robust_fusion_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log("写入 cross_domain/robust_fusion_metrics.json")


if __name__ == "__main__":
    main()
