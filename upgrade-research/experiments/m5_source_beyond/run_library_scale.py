# -*- coding: utf-8 -*-
"""库规模敏感性初测：把底图库从 20 张逐级扩到千张级，看库外高分尾部与判决怎么变。

阶梯（嵌套，真值始终在库内）：
    L20   = 案例(3) 的 20 张原始图片（论文第 7 节现有口径）
    L223  = L20 + 随包默认底图库 203 张
    L274  = L223 + 案例(3) 身份图片 51 张
    L474  = L274 + 视频关键帧 200 张
    L1274 = L274 + 视频关键帧 1000 张

查询 = 案例(3) 的 20 张 AI 图片；检索算法为冻结实现（frozen_source.search）。
命中判定按内容 SHA-256（案例(3) 原始图片存在跨案例重复，按文件名判会误判）。
只用已有素材，不新增采集；库特征一次提取后跨查询复用，不改变任何分数。
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_source import (  # noqa: E402
    FrozenLibrary,
    hashes_of,
    library_fingerprint,
    read_query,
    sha256_of,
    search,
)

ROOT = Path(r"D:\刑科技项目升级2.0")
HERE = Path(__file__).resolve().parent
DATASET = ROOT / "experiments" / "dataset_review"
CASE3_ORIGINALS = DATASET / "libraries" / "source_library_case3"
CASE3_IDENTITY = DATASET / "libraries" / "identity_library_case3"
DEFAULT_LIBRARY = ROOT / "workspace" / "fsts" / "packaging_v0.3" / "resources" / "data" / "source_library"
TASKS = ROOT / "experiments" / "data_root" / "runtime" / "tasks"
CACHE = HERE / "libraries" / "case3_scale_union.npz"
OUT = HERE / "results"

sys.path.insert(0, str(ROOT / "workspace"))
from fsts.decision.open_set import decide_source_open_set, load_thresholds  # noqa: E402

THRESHOLDS_FILE = (
    ROOT / "workspace" / "fsts" / "packaging_v0.4" / "resources" / "decision" / "open_set_thresholds.json"
)


def query_of(case: int) -> Path:
    folder = DATASET / "案例" / str(case) / "AI图片"
    return sorted(p for p in folder.iterdir() if p.is_file())[0]


def collect_video_frames(exclude_hashes: set[str]) -> list[Path]:
    frames = sorted(
        (
            p
            for p in TASKS.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} and "key_frames" in p.parts
        ),
        key=lambda p: str(p).lower(),
    )
    picked: list[Path] = []
    seen: set[str] = set()
    for path in frames:
        digest = sha256_of(path)
        if digest in exclude_hashes or digest in seen:
            continue
        seen.add(digest)
        picked.append(path)
    return picked


def load_feature_cache(union: list[Path]) -> tuple[FrozenLibrary, bool]:
    fingerprint = library_fingerprint(union)
    if CACHE.is_file():
        cached = FrozenLibrary.load(CACHE)
        if library_fingerprint(cached.paths) == fingerprint:
            return cached, True
    started = time.perf_counter()
    features = FrozenLibrary.extract(union, progress_every=100)
    library = FrozenLibrary(union, features, name="case3_scale_union")
    library.save(CACHE)
    print(f"库特征提取完成：{len(union)} 张，{time.perf_counter() - started:.1f}s", flush=True)
    return library, False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    truths = sorted(CASE3_ORIGINALS.iterdir(), key=lambda p: p.name.lower())
    identity = sorted(
        (p for p in CASE3_IDENTITY.rglob("*") if p.is_file()), key=lambda p: str(p).lower()
    )
    default = sorted(DEFAULT_LIBRARY.iterdir(), key=lambda p: p.name.lower())
    queries = [query_of(case) for case in range(1, 21)]

    reserved = {sha256_of(p) for p in truths} | {sha256_of(p) for p in queries}
    frames = collect_video_frames(reserved)
    print(
        f"素材：真值 {len(truths)}、默认库 {len(default)}、身份 {len(identity)}、"
        f"可用视频关键帧 {len(frames)}",
        flush=True,
    )

    union = truths + default + identity + frames[:1000]
    library, reused = load_feature_cache(union)
    print(f"库：{len(library.paths)} 张（{'复用缓存' if reused else '新提取'}）", flush=True)
    library_hashes = hashes_of(library.paths, progress_every=400)
    truth_digest = {case: sha256_of(truths[case - 1]) for case in range(1, 21)}

    levels = [
        ("L20", truths),
        ("L223", truths + default),
        ("L274", truths + default + identity),
        ("L474", truths + default + identity + frames[:200]),
        ("L1274", truths + default + identity + frames[:1000]),
    ]
    thresholds = load_thresholds(THRESHOLDS_FILE)
    low = thresholds["modules"]["source"]["low_global_score"]

    results: dict[str, list[dict]] = {}
    for name, member_paths in levels:
        member_set = {str(Path(p).resolve()) for p in member_paths}
        indices = [i for i, p in enumerate(library.paths) if str(p.resolve()) in member_set]
        rows = []
        for case, query_path in enumerate(queries, 1):
            digest = truth_digest[case]
            image, mask, features_q = read_query(query_path)
            outcome = search(
                library, query_path, query_image=image, query_mask=mask, query_features=features_q
            )
            shortlist = outcome["global_order"][: outcome["shortlist_size"]]
            member_indices = [i for i in indices]
            truth_entries = [i for i in member_indices if library_hashes[i] == digest]
            distractors = [i for i in member_indices if library_hashes[i] != digest]
            scores = outcome["global_scores"]
            truth_global = max(float(scores[i]) for i in truth_entries)
            best_distractor_index = max(distractors, key=lambda i: float(scores[i]))
            best_distractor_global = float(scores[best_distractor_index])
            first_truth_rank = min(
                (shortlist.index(i) + 1 for i in truth_entries if i in shortlist), default=None
            )
            top = outcome["candidates"][0]
            same_source_top = [c for c in outcome["candidates"] if library_hashes[c["index"]] == digest]
            truth_candidate = same_source_top[0] if same_source_top else None
            distractor_candidates = [
                c for c in outcome["candidates"] if library_hashes[c["index"]] != digest
            ]
            best_distractor_candidate = max(
                distractor_candidates, key=lambda c: c["final_score"]
            ) if distractor_candidates else None
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
            if best_distractor_candidate is not None:
                absent_verdict = decide_source_open_set(
                    {
                        "status": "success",
                        "metrics": {"top1_top2_margin": None},
                        "candidates": [best_distractor_candidate],
                    },
                    thresholds,
                )
            else:
                absent_verdict = {"state": None, "state_label": None}
            rows.append(
                {
                    "case": case,
                    "query": str(query_path),
                    "library_size": len(member_indices),
                    "truth_twin_count": len(truth_entries),
                    "truth_global": truth_global,
                    "truth_global_rank": first_truth_rank,
                    "truth_in_shortlist": first_truth_rank is not None,
                    "truth_final": truth_candidate["final_score"] if truth_candidate else None,
                    "best_distractor_global": best_distractor_global,
                    "best_distractor_name": library.paths[best_distractor_index].name,
                    "best_distractor_final": (
                        best_distractor_candidate["final_score"] if best_distractor_candidate else None
                    ),
                    "distractor_global_exceeds_truth": best_distractor_global > truth_global,
                    "top1_is_truth": library_hashes[top["index"]] == digest,
                    "state": verdict["state"],
                    "state_label": verdict["state_label"],
                    "truth_absent_top1_name": (
                        best_distractor_candidate["name"] if best_distractor_candidate else None
                    ),
                    "truth_absent_top1_global": (
                        best_distractor_candidate["global_score"] if best_distractor_candidate else None
                    ),
                    "truth_absent_top1_final": (
                        best_distractor_candidate["final_score"] if best_distractor_candidate else None
                    ),
                    "truth_absent_state": absent_verdict["state"],
                    "truth_absent_state_label": absent_verdict["state_label"],
                }
            )
        results[name] = rows
        hits = sum(1 for r in rows if r["top1_is_truth"])
        tails = sorted(r["best_distractor_global"] for r in rows)
        exceed = sum(1 for r in rows if r["distractor_global_exceeds_truth"])
        print(
            f"{name:>6} 库={len(member_indices):>5} 张  Top-1 命中 {hits}/20  "
            f"库外最高 global 中位 {tails[10]:.4f} 最大 {tails[-1]:.4f}  "
            f"超过真值 {exceed}/20  超下阈 {sum(1 for t in tails if t >= low)}/20",
            flush=True,
        )
        absent = Counter(r["truth_absent_state"] for r in rows)
        print(
            f"       反事实（真值缺席）判定：高可信 {absent['high_confidence']}/20  "
            f"证据不足 {absent['insufficient_evidence']}/20  "
            f"未发现可靠候选 {absent['no_reliable_candidate']}/20",
            flush=True,
        )

    payload = {
        "ladder": [{"level": name, "size": len(members)} for name, members in levels],
        "feature_cache": CACHE.name,
        "sha256_hit_rule": "命中与非同源干扰项均按内容 SHA-256 判定",
        "thresholds": {
            "low_global_score": low,
            "high_global_score": thresholds["modules"]["source"]["high_global_score"],
        },
        "results": results,
    }
    (OUT / "library_scale.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    print(f"写入 {OUT / 'library_scale.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
