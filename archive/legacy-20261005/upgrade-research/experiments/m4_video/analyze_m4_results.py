# -*- coding: utf-8 -*-
"""汇总 M4 视频实验记录（只读：不改动任何 task 结果，不重跑模块）。

用法（开发解释器）：
  & $py experiments/m4_video/analyze_m4_results.py --tags m4_swaps_primary m4_controls

输出：
  - 控制台 Markdown 表格（可整段贴进文档）；
  - results/<tag>/analysis.json（逐条明细 + 角色级汇总）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

M4_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = M4_ROOT.parent
UPGRADE_ROOT = EXPERIMENTS_ROOT.parent
FSTS_ROOT = UPGRADE_ROOT / "workspace" / "fsts"

os.environ.setdefault("FSTS_DATA_ROOT", str(EXPERIMENTS_ROOT / "data_root"))
sys.path.insert(0, str(FSTS_ROOT))

from decision.open_set import decide_identity_open_set, decide_source_open_set  # noqa: E402

RESULTS_ROOT = M4_ROOT / "results"
STATES = {
    "high_confidence": "高可信",
    "insufficient_evidence": "证据不足",
    "no_reliable_candidate": "无可靠候选",
    "unavailable": "不可判定",
}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def open_set_states(record: dict) -> tuple[str | None, str | None]:
    """优先用记录内已保存的三态；老记录回落到直接读 task_dir 的结果文件。"""

    saved = record.get("open_set") or {}
    if saved.get("identity") and saved.get("source"):
        return (
            saved["identity"].get("state"),
            saved["source"].get("state"),
        )
    task_dir = Path(str(record.get("task_dir") or ""))
    identity = read_json(task_dir / "identity_result.json")
    source = read_json(task_dir / "source_result.json")
    return (
        decide_identity_open_set(identity).get("state") if identity else None,
        decide_source_open_set(source).get("state") if source else None,
    )


def short_name(state: str | None) -> str:
    return STATES.get(str(state), "未记录")


def per_record(record: dict) -> dict:
    deepfake = record.get("deepfake") or {}
    counts = deepfake.get("frame_band_counts") or {}
    identity = record.get("identity") or {}
    source = record.get("source") or {}
    selection = record.get("frame_selection") or {}
    runtimes = record.get("module_runtime_seconds") or {}
    identity_state, source_state = open_set_states(record)
    return {
        "pair": record.get("pair"),
        "role": record.get("role"),
        "file_name": record.get("file_name"),
        "status": record.get("status"),
        "band": deepfake.get("band"),
        "band_label": deepfake.get("band_label"),
        "score": deepfake.get("score"),
        "band_rule": deepfake.get("band_rule"),
        "frames_scored": deepfake.get("frames_scored"),
        "frames_high": counts.get("high"),
        "frames_medium": counts.get("medium"),
        "frames_low": counts.get("low"),
        "branch": ",".join(sorted((deepfake.get("frame_branches") or {}).keys())) or None,
        "key_frames": selection.get("key_frames_selected"),
        "candidates_decoded": selection.get("candidates_decoded"),
        "frames_without_face": selection.get("frames_without_face"),
        "extraction_seconds": (
            round(float(selection["extraction_seconds"]), 2)
            if isinstance(selection.get("extraction_seconds"), (int, float))
            else None
        ),
        "wall_seconds": record.get("wall_seconds"),
        "deepfake_seconds": runtimes.get("deepfake"),
        "identity_seconds": runtimes.get("identity"),
        "source_seconds": runtimes.get("source"),
        "identity_top1": identity.get("top1_identity"),
        "identity_similarity": identity.get("top1_mean_similarity"),
        "identity_state": identity_state,
        "identity_state_label": short_name(identity_state),
        "source_top1": source.get("top1_file"),
        "source_global": source.get("top1_global_score"),
        "source_final": source.get("top1_final_score"),
        "source_inliers": source.get("inlier_count"),
        "source_state": source_state,
        "source_state_label": short_name(source_state),
    }


def role_summary(rows: list[dict]) -> dict:
    bands = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    frames_high = 0
    videos_with_high_frame = 0
    for row in rows:
        band = str(row.get("band") or "unknown")
        bands[band if band in bands else "unknown"] += 1
        if int(row.get("frames_high") or 0) > 0:
            frames_high += 1
            videos_with_high_frame += 1
    walls = [float(row["wall_seconds"]) for row in rows if isinstance(row.get("wall_seconds"), (int, float))]
    extractions = [
        float(row["extraction_seconds"])
        for row in rows
        if isinstance(row.get("extraction_seconds"), (int, float))
    ]
    return {
        "videos": len(rows),
        "video_band_counts": bands,
        "videos_with_high_band_frame": videos_with_high_frame,
        "mean_wall_seconds": round(sum(walls) / len(walls), 1) if walls else None,
        "mean_extraction_seconds": round(sum(extractions) / len(extractions), 1) if extractions else None,
    }


def analyze_tag(tag: str) -> dict:
    out_dir = RESULTS_ROOT / tag
    runs_path = out_dir / "runs.jsonl"
    if not runs_path.is_file():
        return {"tag": tag, "error": "缺少 runs.jsonl"}
    records = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [per_record(record) for record in records]
    by_role: dict[str, list[dict]] = {}
    for row in rows:
        by_role.setdefault(str(row.get("role")), []).append(row)
    payload = {
        "tag": tag,
        "runs_path": str(runs_path),
        "videos": len(rows),
        "rows": rows,
        "role_summary": {role: role_summary(items) for role, items in sorted(by_role.items())},
    }
    (out_dir / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def markdown_table(payload: dict) -> str:
    lines = [
        f"### {payload.get('tag')}（{payload.get('videos', 0)} 条）",
        "",
        "| 案例 | 角色 | 视频级分带 | 分数 | 帧(高/中/低) | 关键帧/候选 | 抽帧s | 总耗时s | 身份三态 | 底图三态 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("rows", []):
        if row.get("status") != "success":
            lines.append(
                f"| {row.get('pair')} | {row.get('role')} | 失败 | - | - | - | - | "
                f"{row.get('wall_seconds')} | - | - |"
            )
            continue
        counts = f"{row.get('frames_high')}/{row.get('frames_medium')}/{row.get('frames_low')}"
        coverage = f"{row.get('key_frames')}/{row.get('candidates_decoded')}"
        score = row.get("score")
        score_text = f"{score:.6g}" if isinstance(score, (int, float)) else " - "
        lines.append(
            f"| {row.get('pair')} | {row.get('role')} | {row.get('band')} | {score_text} | {counts} | "
            f"{coverage} | {row.get('extraction_seconds')} | {row.get('wall_seconds')} | "
            f"{row.get('identity_state_label')} | {row.get('source_state_label')} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总 M4 视频实验记录")
    parser.add_argument("--tags", nargs="+", required=True)
    args = parser.parse_args(argv)

    for tag in args.tags:
        payload = analyze_tag(tag)
        if payload.get("error"):
            print(f"[跳过] {tag}: {payload['error']}")
            continue
        print(markdown_table(payload))
        print()
        print(f"角色级汇总：{json.dumps(payload['role_summary'], ensure_ascii=False)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
