# -*- coding: utf-8 -*-
"""传播链退化实验：同源素材经过平台转码、二次编码、缩放、截图近似后，还能不能串并上。

查询 = 案例(3) 的 20 张 AI 图片按档位退化后的版本；库 = 案例(3) 的 20 张原始图片。
退化只作用在查询侧（库侧是"保存下来的原始素材"），检索算法为冻结实现。
每一档都记录：真值是否仍为 Top-1、真值的全局/局部/最终分、四态结论、库外最高分。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_source import FrozenLibrary, hashes_of, read_query, search  # noqa: E402

ROOT = Path(r"D:\刑科技项目升级2.0")
HERE = Path(__file__).resolve().parent
DATASET = ROOT / "experiments" / "dataset_review"
LIBRARY = DATASET / "libraries" / "source_library_case3"
DEGRADED = HERE / "degraded"
OUT = HERE / "results"

sys.path.insert(0, str(ROOT / "workspace"))
from fsts.decision.open_set import decide_source_open_set, load_thresholds  # noqa: E402

THRESHOLDS_FILE = (
    ROOT / "workspace" / "fsts" / "packaging_v0.4" / "resources" / "decision" / "open_set_thresholds.json"
)


def query_of(case: int) -> Path:
    folder = DATASET / "案例" / str(case) / "AI图片"
    return sorted(p for p in folder.iterdir() if p.is_file())[0]


def jpeg(image: np.ndarray, quality: int, path: Path) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imread(str(path))


def scaled(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(
        image, (max(1, round(width * factor)), max(1, round(height * factor))),
        interpolation=cv2.INTER_AREA,
    )


def with_border(image: np.ndarray, px: int) -> np.ndarray:
    return cv2.copyMakeBorder(image, px, px, px, px, cv2.BORDER_CONSTANT, value=(32, 32, 32))


def build_levels(image: np.ndarray, case: int, folder: Path, original: Path) -> list[tuple[str, str, Path]]:
    """返回 (档位, 说明, 落盘路径)。退化图写到磁盘后再读回，保证解码路径与真实取证一致。"""
    folder.mkdir(parents=True, exist_ok=True)

    def write(name: str, degraded: np.ndarray, quality: int) -> Path:
        path = folder / f"case{case:02d}_{name}.jpg"
        cv2.imwrite(str(path), degraded, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return path

    levels: list[tuple[str, str, Path]] = [
        ("原图", "入库时的原始查询文件，未做任何重新编码", original)
    ]
    for quality in (90, 70, 50, 30):
        levels.append(
            (f"JPEG q{quality}", f"二次编码，JPEG 质量 {quality}",
             write(f"jpeg{quality}", image, quality))
        )
    levels.append(
        ("缩放 0.5x", "长宽各缩到 50%（传播中的降分辨率），JPEG q95",
         write("scale50", scaled(image, 0.5), 95))
    )
    levels.append(
        ("缩放 0.25x", "长宽各缩到 25%，JPEG q95",
         write("scale25", scaled(image, 0.25), 95))
    )
    levels.append(
        ("平台转码", "缩放 0.5x 后 JPEG q60（即时通信平台二次编码近似）",
         write("platform", scaled(image, 0.5), 60))
    )
    levels.append(
        ("截图近似", "缩放 0.6x + 24 px 边框后 JPEG q85（屏幕截图近似）",
         write("screenshot", with_border(scaled(image, 0.6), 24), 85))
    )
    return levels


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    truths = sorted(LIBRARY.iterdir(), key=lambda p: p.name.lower())
    library = FrozenLibrary(
        truths, FrozenLibrary.extract(truths, progress_every=0), name="case3_originals"
    )
    thresholds = load_thresholds(THRESHOLDS_FILE)
    truth_by_case = {int(p.stem[4:]): i for i, p in enumerate(library.paths)}
    library_hashes = hashes_of(library.paths)

    records: dict[str, list[dict]] = {}
    for case in range(1, 21):
        query_path = query_of(case)
        source = cv2.imread(str(query_path))
        if source is None:
            raise ValueError(f"cannot decode {query_path}")
        truth_index = truth_by_case[case]
        truth_digest = library_hashes[truth_index]
        folder = DEGRADED / f"case{case:02d}"
        for level, note, temp in build_levels(source, case, folder, query_path):
            q_image, q_mask, q_features = read_query(temp)
            outcome = search(
                library, temp, query_image=q_image, query_mask=q_mask, query_features=q_features
            )
            top = outcome["candidates"][0]
            same_source = [
                c for c in outcome["candidates"] if library_hashes[c["index"]] == truth_digest
            ]
            truth_candidate = same_source[0] if same_source else None
            distractors = [
                c for c in outcome["candidates"] if library_hashes[c["index"]] != truth_digest
            ]
            best_distractor = max(
                distractors, key=lambda c: c["final_score"]
            ) if distractors else None
            verdict = decide_source_open_set(
                {
                    "status": "success",
                    "metrics": {
                        "top1_top2_margin": round(
                            top["final_score"] - outcome["candidates"][1]["final_score"], 6
                        )
                        if len(outcome["candidates"]) > 1
                        else None
                    },
                    "candidates": [top],
                },
                thresholds,
            )
            record = {
                "case": case,
                "level": level,
                "note": note,
                "truth_global": float(truth_candidate["global_score"]) if truth_candidate else None,
                "truth_local": float(truth_candidate["local_score"]) if truth_candidate else None,
                "truth_final": float(truth_candidate["final_score"]) if truth_candidate else None,
                "truth_rank": (
                    outcome["candidates"].index(truth_candidate) + 1 if truth_candidate else None
                ),
                "top1_is_truth": library_hashes[top["index"]] == truth_digest,
                "top1_name": top["name"],
                "best_distractor_final": float(best_distractor["final_score"]) if best_distractor else None,
                "best_distractor_global": float(best_distractor["global_score"]) if best_distractor else None,
                "margin": float(top["final_score"] - outcome["candidates"][1]["final_score"])
                if len(outcome["candidates"]) > 1
                else None,
                "state": verdict["state"],
                "state_label": verdict["state_label"],
            }
            records.setdefault(level, []).append(record)
        print(f"案例{case:>2} 完成", flush=True)

    order = ["原图", "JPEG q90", "JPEG q70", "JPEG q50", "JPEG q30",
             "缩放 0.5x", "缩放 0.25x", "平台转码", "截图近似"]
    print("\n档位             Top-1 命中   真值 global 中位   真值 final 中位   库外最高 final 中位")
    for level in order:
        rows = records[level]
        hits = sum(1 for r in rows if r["top1_is_truth"])
        globals_sorted = sorted(r["truth_global"] for r in rows if r["truth_global"] is not None)
        finals = sorted(r["truth_final"] for r in rows if r["truth_final"] is not None)
        tails = sorted(r["best_distractor_final"] for r in rows if r["best_distractor_final"] is not None)
        print(f"{level:<14} {hits:>2}/20        {globals_sorted[10]:.4f}            "
              f"{finals[10]:.4f}            {tails[10]:.4f}")

    payload = {
        "library": str(LIBRARY),
        "library_size": len(library.paths),
        "degraded_root": str(DEGRADED),
        "levels": order,
        "thresholds": {
            "low_global_score": thresholds["modules"]["source"]["low_global_score"],
            "high_global_score": thresholds["modules"]["source"]["high_global_score"],
        },
        "records": records,
    }
    (OUT / "propagation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    print(f"\n写入 {OUT / 'propagation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
