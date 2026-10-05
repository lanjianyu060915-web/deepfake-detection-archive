r"""M2 对照实验：只改检测输入尺度，测量身份检索端到端变化。

对照两组输入（库外 20 例 SWAN-DF 查询 + 库内 10 例主实验查询）：

- 基线：det_size=(1280,1280)，与冻结 worker 一致；
- 候选：det_size=(640,640)，检测输入不再把 720x1280 竖屏帧横向拉伸 1.78 倍。

两组都在同一进程内重建身份库索引（缓存指纹含 det_size，会自动分开），
保证查询与库的预处理口径一致。冻结资产只读，索引缓存写入升级工作区。

用法：
    D:\python\.venv\Scripts\python.exe eval_identity_multiscale.py
"""

import json
import sys
from pathlib import Path

WS = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
sys.path.insert(0, str(WS / "modules" / "identity_search"))

from identity_search.searcher import IdentitySearcher  # noqa: E402
from identity_search.errors import IdentitySearchError  # noqa: E402

SWAN = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
EV = Path(r"D:\刑侦查竞赛\_paper_eval_20260919")
LIBRARY = Path(r"D:\刑侦查竞赛\first_week\data\identity_library")
MODEL_ROOT = WS / "packaging_v0.3" / "resources"
OUT = Path(r"D:\刑科技项目升级2.0\experiments\m2_preprocess")

DET_SIZES = {"frozen_1280": (1280, 1280), "scale_640": (640, 640)}


def queries() -> list[dict]:
    rows = []
    for index in range(1, 21):
        case = f"案例{index:02d}"
        rows.append(
            {
                "group": "outside",
                "id": case,
                "path": SWAN / case / "05_换脸后图片.jpg",
            }
        )
    for index in range(1, 11):
        query = f"query_{index:03d}"
        payload = json.loads(
            (EV / "identity" / query / "identity_result.json").read_text(encoding="utf-8")
        )
        rows.append({"group": "inside", "id": query, "path": Path(payload["query_image"])})
    return rows


def run_variant(name: str, det_size: tuple[int, int], items: list[dict]) -> list[dict]:
    cache = OUT / f"_identity_cache_{name}"
    searcher = IdentitySearcher(
        LIBRARY,
        cache_dir=cache,
        model_root=MODEL_ROOT,
        device="cpu",
        det_size=det_size,
        model_fingerprint=f"buffalo_l-det{det_size[0]}",
    )
    searcher._load_or_build_index()
    rows = []
    for item in items:
        record = {
            "variant": name,
            "group": item["group"],
            "id": item["id"],
            "path": str(item["path"]),
        }
        try:
            result = searcher.search(item["path"], top_k=5)
        except IdentitySearchError as exc:
            record.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "face_count": (exc.diagnostics or {}).get("face_count"),
                }
            )
        else:
            metrics = result.get("metrics") or {}
            candidates = result.get("candidates") or []
            top = candidates[0] if candidates else {}
            record.update(
                {
                    "status": result.get("status"),
                    "face_count": metrics.get("query_face_count"),
                    "detection_score": metrics.get("face_detection_score"),
                    "top1_id": top.get("identity_id"),
                    "top1_score": top.get("mean_similarity"),
                    "margin": metrics.get("top1_top2_margin"),
                }
            )
        rows.append(record)
        print(
            f"  [{name}] {item['group']}/{item['id']}: {record['status']} "
            f"faces={record.get('face_count')} top1={record.get('top1_id')} "
            f"score={record.get('top1_score')} margin={record.get('margin')}",
            flush=True,
        )
    return rows


items = queries()
all_rows: list[dict] = []
for name, det_size in DET_SIZES.items():
    print(f"=== {name} det_size={det_size} ===", flush=True)
    all_rows.extend(run_variant(name, det_size, items))

(OUT / "eval_identity_multiscale.json").write_text(
    json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("\n汇总（success / error）")
for name in DET_SIZES:
    for group in ("outside", "inside"):
        subset = [r for r in all_rows if r["variant"] == name and r["group"] == group]
        ok = [r for r in subset if r["status"] == "success"]
        scores = [r["top1_score"] for r in ok if isinstance(r.get("top1_score"), (int, float))]
        span = f"{min(scores):.4f}-{max(scores):.4f}" if scores else "—"
        print(f"  {name:12} {group:8} {len(ok):2}/{len(subset):2}  score {span}")
sys.exit(0)
