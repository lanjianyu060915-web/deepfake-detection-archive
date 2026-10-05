r"""M2 交叉尺度实验：库索引仍用冻结尺度，查询改用 640 回退能否直接可用。

上一轮（eval_identity_multiscale.py）中查询与库都在 640 重建，库内命中分数几乎不变。
本实验固定**库索引为冻结的 det_size=(1280,1280) 缓存**，只把查询侧检测改为 640，
用于判断新增回退层是否必须重建身份库缓存。

用法：
    D:\python\.venv\Scripts\python.exe eval_cross_scale_fallback.py
"""

import json
import sys
from pathlib import Path

WS = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
sys.path.insert(0, str(WS / "modules" / "identity_search"))

from identity_search.errors import IdentitySearchError  # noqa: E402
from identity_search.extractor import FeatureExtractor  # noqa: E402
from identity_search.searcher import IdentitySearcher  # noqa: E402

SWAN = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
EV = Path(r"D:\刑侦查竞赛\_paper_eval_20260919")
LIBRARY = Path(r"D:\刑侦查竞赛\first_week\data\identity_library")
MODEL_ROOT = WS / "packaging_v0.3" / "resources"
OUT = Path(r"D:\刑科技项目升级2.0\experiments\m2_preprocess")

FROZEN_CACHE = OUT / "_identity_cache_frozen_1280"
FAILED = [1, 2, 3, 4, 5, 6, 7, 17, 18, 19]


def main() -> int:
    if not (FROZEN_CACHE / "identity_index_plan.json").exists():
        plan_files = list(FROZEN_CACHE.glob("*.json"))
        print(f"冻结尺度缓存目录内容：{[item.name for item in plan_files]}")

    searcher = IdentitySearcher(
        LIBRARY,
        cache_dir=FROZEN_CACHE,
        model_root=MODEL_ROOT,
        device="cpu",
        det_size=(1280, 1280),
        model_fingerprint="buffalo_l-det1280",
        extractor=FeatureExtractor(
            model_root=MODEL_ROOT, device="cpu", ctx_id=-1, det_size=(640, 640)
        ),
    )
    searcher._load_or_build_index()
    print(f"库索引缓存状态：{searcher.diagnostics.get('cache_status')}")

    rows = []
    print("=== 库外失败样例（查询 640 / 库 1280）===")
    for index in FAILED:
        case = f"案例{index:02d}"
        try:
            result = searcher.search(SWAN / case / "05_换脸后图片.jpg", top_k=5)
        except IdentitySearchError as exc:
            rows.append({"group": "outside", "id": case, "status": "error",
                         "error_type": type(exc).__name__})
            print(f"  {case}: error {type(exc).__name__}")
            continue
        top = (result.get("candidates") or [{}])[0]
        rows.append(
            {
                "group": "outside",
                "id": case,
                "status": result.get("status"),
                "top1_id": top.get("identity_id"),
                "top1_score": top.get("mean_similarity"),
                "margin": (result.get("metrics") or {}).get("top1_top2_margin"),
            }
        )
        print(
            f"  {case}: {result.get('status')} top1={top.get('identity_id')} "
            f"score={top.get('mean_similarity'):.6f} "
            f"margin={(result.get('metrics') or {}).get('top1_top2_margin'):.6f}"
        )

    print("=== 库内命中（查询 640 / 库 1280）===")
    for index in range(1, 11):
        query = f"query_{index:03d}"
        payload = json.loads(
            (EV / "identity" / query / "identity_result.json").read_text(encoding="utf-8")
        )
        try:
            result = searcher.search(payload["query_image"], top_k=5)
        except IdentitySearchError as exc:
            rows.append({"group": "inside", "id": query, "status": "error",
                         "error_type": type(exc).__name__})
            print(f"  {query}: error {type(exc).__name__}")
            continue
        top = (result.get("candidates") or [{}])[0]
        rows.append(
            {
                "group": "inside",
                "id": query,
                "status": result.get("status"),
                "top1_id": top.get("identity_id"),
                "top1_score": top.get("mean_similarity"),
                "margin": (result.get("metrics") or {}).get("top1_top2_margin"),
            }
        )
        print(
            f"  {query}: {result.get('status')} top1={top.get('identity_id')} "
            f"score={top.get('mean_similarity'):.6f} "
            f"margin={(result.get('metrics') or {}).get('top1_top2_margin'):.6f}"
        )

    (OUT / "eval_cross_scale_fallback.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for group in ("outside", "inside"):
        subset = [row for row in rows if row["group"] == group]
        ok = [row for row in subset if row["status"] == "success"]
        scores = [row["top1_score"] for row in ok]
        span = f"{min(scores):.4f}-{max(scores):.4f}" if scores else "—"
        print(f"汇总 {group:8} {len(ok)}/{len(subset)}  score {span}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
