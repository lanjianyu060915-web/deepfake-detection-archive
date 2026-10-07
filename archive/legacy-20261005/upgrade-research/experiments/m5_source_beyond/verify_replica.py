# -*- coding: utf-8 -*-
r"""复刻正确性自检：把 frozen_source.search 的输出与冻结 EXE 的实际结果逐案例比对。

冻结结果：experiments\dataset_review\results\source\caseNN.json
（由 packaging_v0.3 的 source_search.exe 在 20 张案例(3)原始图片库上跑出）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_source import FrozenLibrary, read_query, search  # noqa: E402

ROOT = Path(r"D:\刑科技项目升级2.0")
DATASET = ROOT / "experiments" / "dataset_review"
LIB = DATASET / "libraries" / "source_library_case3"
RECORDED = DATASET / "results" / "source"
OUT = Path(__file__).resolve().parent / "results"


def query_of(case: int) -> Path:
    folder = DATASET / "案例" / str(case) / "AI图片"
    return sorted(p for p in folder.iterdir() if p.is_file())[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(LIB.iterdir(), key=lambda p: p.name.lower())
    print(f"库：{len(paths)} 张；冻结结果：{RECORDED}")
    library = FrozenLibrary(paths, FrozenLibrary.extract(paths, progress_every=0), name="case3")

    rows = []
    max_final_diff = 0.0
    max_global_diff = 0.0
    max_local_diff = 0.0
    max_good_diff = 0
    max_inlier_diff = 0
    order_mismatch = []
    for case in range(1, 21):
        recorded = json.loads((RECORDED / f"case{case:02d}.json").read_text(encoding="utf-8"))
        frozen_top = recorded["candidates"][0]
        query_image, query_mask, query_features = read_query(query_of(case))
        started = time.perf_counter()
        mine = search(
            library,
            query_of(case),
            query_image=query_image,
            query_mask=query_mask,
            query_features=query_features,
        )
        elapsed = time.perf_counter() - started
        top = mine["candidates"][0]
        final_diff = abs(top["final_score"] - frozen_top["final_score"])
        global_diff = abs(top["global_score"] - frozen_top["global_score"])
        local_diff = abs(top["local_score"] - frozen_top["local_score"])
        good_diff = top["good_match_count"] - int(frozen_top["good_match_count"])
        inlier_diff = top["ransac_inlier_count"] - int(frozen_top["ransac_inlier_count"])
        max_final_diff = max(max_final_diff, final_diff)
        max_global_diff = max(max_global_diff, global_diff)
        max_local_diff = max(max_local_diff, local_diff)
        max_good_diff = max(max_good_diff, abs(good_diff))
        max_inlier_diff = max(max_inlier_diff, abs(inlier_diff))
        same_top = Path(frozen_top["file_path"]).name == top["name"]
        if not same_top:
            order_mismatch.append({"case": case, "frozen": Path(frozen_top["file_path"]).name, "mine": top["name"]})
        rows.append(
            {
                "case": case,
                "frozen_top": Path(frozen_top["file_path"]).name,
                "mine_top": top["name"],
                "same_top1": same_top,
                "frozen_final": frozen_top["final_score"],
                "mine_final": top["final_score"],
                "final_diff": final_diff,
                "global_diff": global_diff,
                "local_diff": local_diff,
                "good_match_diff": good_diff,
                "inlier_diff": inlier_diff,
                "seconds": round(elapsed, 2),
            }
        )
        print(
            f"案例{case:>2} 冻结={Path(frozen_top['file_path']).name} 复刻={top['name']} "
            f"Δfinal={final_diff:.2e} Δglobal={global_diff:.2e} Δlocal={local_diff:.2e} "
            f"Δgood={good_diff} Δ内点={inlier_diff} {elapsed:.1f}s",
            flush=True,
        )

    summary = {
        "library": str(LIB),
        "library_size": len(paths),
        "recorded": str(RECORDED),
        "max_final_diff": max_final_diff,
        "max_global_diff": max_global_diff,
        "max_local_diff": max_local_diff,
        "max_good_match_diff": max_good_diff,
        "max_inlier_diff": max_inlier_diff,
        "top1_mismatch": order_mismatch,
        "rows": rows,
    }
    (OUT / "replica_verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    print(f"\n最大 final 差 {max_final_diff:.3e}；最大 global 差 {max_global_diff:.3e}；"
          f"最大 local 差 {max_local_diff:.3e}；最大有效匹配数差 {max_good_diff}；"
          f"最大内点数差 {max_inlier_diff}；Top-1 不一致 {len(order_mismatch)} 例")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
