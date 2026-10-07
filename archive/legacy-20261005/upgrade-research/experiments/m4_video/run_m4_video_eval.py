# -*- coding: utf-8 -*-
"""M4 视频入口正式实验：在真实 SWAN-DF 视频上跑通三分支视频任务。

用法（开发解释器）：
  & $py experiments/m4_video/run_m4_video_eval.py --set controls --tag m4_controls
  & $py experiments/m4_video/run_m4_video_eval.py --set swaps_primary --tag m4_swaps_primary

脚本默认通过 RuntimeServiceManager 启动持久模型服务（与前端产品路径一致），
因此关键帧鉴伪走的是二代 ViT 内核；加 --no-services 则回退到冻结子进程路径，
用于对照“未启用持久服务时会退化成什么结果”。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

M4_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = M4_ROOT.parent
UPGRADE_ROOT = EXPERIMENTS_ROOT.parent
FSTS_ROOT = UPGRADE_ROOT / "workspace" / "fsts"
MANIFEST_PATH = M4_ROOT / "videos" / "manifest.json"

DEFAULT_DATA_ROOT = EXPERIMENTS_ROOT / "data_root"
os.environ.setdefault("FSTS_DATA_ROOT", str(DEFAULT_DATA_ROOT))
sys.path.insert(0, str(FSTS_ROOT))

from decision.open_set import decide_identity_open_set, decide_source_open_set  # noqa: E402

PRIMARY_SWAP_TOKEN = "-d320m4o4e15b150r7-"


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["videos"]


def select_videos(videos: list[dict], name: str) -> list[dict]:
    """按预设挑出要跑的视频；同一文件在多个案例里重复出现，只保留一条。"""

    unique: dict[str, dict] = {}
    for item in videos:
        unique.setdefault(item["file_name"], item)
    items = list(unique.values())
    if name == "all":
        return items
    if name == "controls":
        return [item for item in items if item["role"] == "original_source"]
    if name == "swaps_primary":
        return [
            item
            for item in items
            if item["role"] == "deepfake_result" and PRIMARY_SWAP_TOKEN in item["file_name"]
        ]
    if name == "swaps_all":
        return [item for item in items if item["role"] == "deepfake_result"]
    if name == "identity_sources":
        return [item for item in items if item["role"] == "identity_source"]
    raise SystemExit(f"未知集合：{name}")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _decision_state(decision: dict) -> dict:
    """只保留开放集判定的结论与主判据，便于跨视频汇总。"""

    signal = decision.get("primary_signal") or {}
    thresholds = decision.get("thresholds") or {}
    return {
        "state": decision.get("state"),
        "state_label": decision.get("state_label"),
        "decidable": decision.get("decidable"),
        "primary_signal": signal.get("name"),
        "primary_value": signal.get("value"),
        "thresholds_config_version": thresholds.get("config_version"),
    }


def collect_run(item: dict, status: dict, task_dir: Path, wall_seconds: float) -> dict:
    selection = read_json(task_dir / "frame_selection.json")
    video_summary = read_json(task_dir / "video_summary.json")
    unified = read_json(task_dir / "unified_result.json")
    deepfake = read_json(task_dir / "detection_result.json")
    identity = read_json(task_dir / "identity_result.json")
    source = read_json(task_dir / "source_result.json")

    deepfake_details = deepfake.get("details") or {}
    identity_metrics = identity.get("metrics") or {}
    identity_candidates = identity.get("candidates") or []
    identity_top = identity_candidates[0] if identity_candidates else {}
    source_candidates = source.get("candidates") or []
    source_top = source_candidates[0] if source_candidates else {}
    modules = video_summary.get("modules") or {}

    return {
        "pair": item.get("pair"),
        "role": item.get("role"),
        "file_name": item.get("file_name"),
        "video_sha256_manifest": item.get("sha256"),
        "video_sha256_task": selection.get("video_sha256"),
        "task_id": status.get("task_id"),
        "task_dir": str(task_dir),
        "status": status.get("status"),
        "wall_seconds": round(wall_seconds, 3),
        "module_runtime_seconds": {
            name: (status.get(name) or {}).get("runtime_seconds")
            for name in ("deepfake", "identity", "source")
        },
        "input_kind": ((unified.get("input") or {}).get("type")) if unified else None,
        "video": selection.get("video"),
        "frame_selection": {
            "candidates_decoded": (selection.get("sampling") or {}).get("candidate_frames_decoded"),
            "key_frames_selected": len(selection.get("key_frames") or []),
            "extraction_seconds": selection.get("runtime_seconds"),
            "frames_with_single_face": (selection.get("face_detection") or {}).get(
                "frames_with_single_face"
            ),
            "frames_without_face": (selection.get("face_detection") or {}).get(
                "frames_without_face"
            ),
        },
        "deepfake": {
            "status": deepfake.get("status"),
            "score": deepfake.get("score"),
            "band": deepfake_details.get("band"),
            "band_label": deepfake_details.get("band_label"),
            "band_rule": deepfake_details.get("band_rule"),
            "frame_band_counts": deepfake_details.get("frame_band_counts"),
            "frame_band_sources": deepfake_details.get("frame_band_sources"),
            "frame_branches": deepfake_details.get("frame_branches"),
            "frames_scored": deepfake_details.get("frames_scored"),
            "frames_total": deepfake_details.get("frames_total"),
            "per_frame": deepfake_details.get("per_frame"),
            "consistency_statement": deepfake_details.get("consistency_statement"),
        },
        "identity": {
            "status": identity.get("status"),
            "frames_analyzed": identity_metrics.get("frames_analyzed"),
            "frames_with_faces": identity_metrics.get("frames_with_faces"),
            "top1_identity": identity_top.get("identity_id"),
            "top1_mean_similarity": identity_top.get("mean_across_frames"),
            "top1_frame_support": identity_top.get("frame_support"),
            "top1_top2_margin": identity_metrics.get("top1_top2_margin"),
        },
        "source": {
            "status": source.get("status"),
            "top1_file": source_top.get("file_name") or Path(str(source_top.get("file_path") or "")).name,
            "top1_final_score": source_top.get("final_score"),
            "top1_global_score": source_top.get("global_score"),
            "top1_local_score": source_top.get("local_score"),
            "inlier_count": source_top.get("ransac_inlier_count", source_top.get("inlier_count")),
            "inlier_ratio": source_top.get("ransac_inlier_ratio", source_top.get("inlier_ratio")),
        },
        "open_set": {
            "identity": _decision_state(decide_identity_open_set(identity)),
            "source": _decision_state(decide_source_open_set(source)),
        },
        "video_summary_modules": modules,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M4 视频入口实验（真实 SWAN-DF 视频）")
    parser.add_argument("--set", dest="video_set", default="controls")
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["deepfake", "identity", "source"],
    )
    parser.add_argument("--no-services", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--config", default=str(FSTS_ROOT / "config.json"))
    parser.add_argument("--identity-library", default=None)
    parser.add_argument("--source-library", default=None)
    args = parser.parse_args(argv)

    from orchestrator import run_task

    videos = select_videos(load_manifest(), args.video_set)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit("没有选中任何视频")

    out_dir = M4_ROOT / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"

    services = None
    manager = None
    if not args.no_services:
        from runtime_services import RuntimeServiceManager

        manager = RuntimeServiceManager(args.config)
        startup = manager.start()
        services = manager
        print("[服务] " + json.dumps(startup, ensure_ascii=False)[:400], flush=True)

    records: list[dict] = []
    try:
        for index, item in enumerate(videos, start=1):
            video_path = Path(item["path"])
            print(f"[{index}/{len(videos)}] {item['pair']} {item['role']} {item['file_name']}", flush=True)
            started = time.perf_counter()
            try:
                status = run_task(
                    video_path,
                    identity_library=args.identity_library,
                    source_library=args.source_library,
                    selected_modules=args.modules,
                    services=services,
                    config_path=args.config,
                )
                wall = time.perf_counter() - started
                record = collect_run(item, status, Path(status["task_dir"]), wall)
            except Exception as exc:  # noqa: BLE001 - 单条失败不得中断整批
                record = {
                    "pair": item.get("pair"),
                    "role": item.get("role"),
                    "file_name": item.get("file_name"),
                    "status": "exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": round(time.perf_counter() - started, 3),
                }
            records.append(record)
            with runs_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            deepfake = record.get("deepfake") or {}
            print(
                f"    -> status={record.get('status')} band={deepfake.get('band')} "
                f"score={deepfake.get('score')} wall={record.get('wall_seconds')}s",
                flush=True,
            )
    finally:
        if manager is not None:
            try:
                manager.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[服务] 关闭失败：{exc}", flush=True)

    summary = {
        "tag": args.tag,
        "video_set": args.video_set,
        "modules": list(args.modules),
        "used_services": not args.no_services,
        "videos": len(records),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(args.config),
        "records": records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out_dir / "frame_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pair",
                "role",
                "file_name",
                "video_status",
                "video_band",
                "video_score",
                "band_rule",
                "frame_rank",
                "timestamp_seconds",
                "frame_score",
                "frame_band",
                "frame_band_source",
                "frame_branch",
                "is_deepfake",
            ]
        )
        for record in records:
            deepfake = record.get("deepfake") or {}
            for frame in deepfake.get("per_frame") or []:
                writer.writerow(
                    [
                        record.get("pair"),
                        record.get("role"),
                        record.get("file_name"),
                        record.get("status"),
                        deepfake.get("band"),
                        deepfake.get("score"),
                        deepfake.get("band_rule"),
                        frame.get("frame_rank"),
                        frame.get("timestamp_seconds"),
                        frame.get("score"),
                        frame.get("band"),
                        frame.get("band_source"),
                        frame.get("branch"),
                        frame.get("is_deepfake"),
                    ]
                )

    print(f"[完成] {len(records)} 条记录已写入 {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
