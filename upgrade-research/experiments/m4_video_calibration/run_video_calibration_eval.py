# -*- coding: utf-8 -*-
"""在 DFDC 校准集/留出集上跑视频任务的鉴伪分支，产出帧级与视频级分数。

视频任务路径完全复用产品编排（抽帧 → 关键帧选择 → 逐帧鉴伪 → video_pipeline
聚合），只把模块裁剪到 ``deepfake``：视频级分带校准只涉及鉴伪分支，身份与底图
在 M4 已确认是「正确拒识」，不参与标定，跑它们只会把单条耗时从约 15 s 抬到
约 60 s。

用法（开发解释器）：
  & $py experiments/m4_video_calibration/run_video_calibration_eval.py --split calibration --tag m4cal_calibration
  & $py experiments/m4_video_calibration/run_video_calibration_eval.py --split holdout --tag m4cal_holdout
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = ROOT.parent
UPGRADE_ROOT = EXPERIMENTS_ROOT.parent
FSTS_ROOT = UPGRADE_ROOT / "workspace" / "fsts"
DEFAULT_DATA_ROOT = EXPERIMENTS_ROOT / "data_root"
MANIFEST_PATH = ROOT / "manifest.json"
os.environ.setdefault("FSTS_DATA_ROOT", str(DEFAULT_DATA_ROOT))
sys.path.insert(0, str(FSTS_ROOT))


def load_manifest(split: str) -> list[dict]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    videos = list(data.get("videos") or [])
    if split == "both":
        return videos
    return [item for item in videos if item.get("split") == split]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def collect_run(item: dict, status: dict, task_dir: Path, wall_seconds: float) -> dict:
    selection = read_json(task_dir / "frame_selection.json")
    deepfake = read_json(task_dir / "detection_result.json")
    unified = read_json(task_dir / "unified_result.json")
    details = deepfake.get("details") or {}
    return {
        "split": item.get("split"),
        "label": item.get("label"),
        "part": item.get("part"),
        "file_name": item.get("file_name"),
        "video_sha256_manifest": item.get("sha256"),
        "video_sha256_task": selection.get("video_sha256"),
        "task_id": status.get("task_id"),
        "task_dir": str(task_dir),
        "status": status.get("status"),
        "wall_seconds": round(wall_seconds, 3),
        "module_runtime_seconds": {
            "deepfake": (status.get("deepfake") or {}).get("runtime_seconds"),
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
            "band": details.get("band"),
            "band_label": details.get("band_label"),
            "band_rule": details.get("band_rule"),
            "frame_band_counts": details.get("frame_band_counts"),
            "frame_band_sources": details.get("frame_band_sources"),
            "frame_branches": details.get("frame_branches"),
            "frames_scored": details.get("frames_scored"),
            "frames_total": details.get("frames_total"),
            "per_frame": details.get("per_frame"),
            "consistency_statement": details.get("consistency_statement"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DFDC 视频校准集评测（鉴伪分支）")
    parser.add_argument("--split", choices=("calibration", "holdout", "both"), default="calibration")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-services", action="store_true")
    parser.add_argument("--config", default=str(FSTS_ROOT / "config.json"))
    args = parser.parse_args(argv)

    from orchestrator import run_task

    videos = load_manifest(args.split)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit("没有选中任何视频，请先运行 download_dfdc.py")

    out_dir = ROOT / "results" / args.tag
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
            if not video_path.is_file():
                print(f"[{index}/{len(videos)}] 缺少文件：{video_path}", flush=True)
                continue
            print(
                f"[{index}/{len(videos)}] {item['split']}/{item['label']} {item['file_name']}",
                flush=True,
            )
            started = time.perf_counter()
            try:
                status = run_task(
                    video_path,
                    selected_modules=["deepfake"],
                    services=services,
                    config_path=args.config,
                )
                wall = time.perf_counter() - started
                record = collect_run(item, status, Path(status["task_dir"]), wall)
            except Exception as exc:  # noqa: BLE001 - 单条失败不得中断整批
                record = {
                    "split": item.get("split"),
                    "label": item.get("label"),
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
        "split": args.split,
        "modules": ["deepfake"],
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
                "split",
                "label",
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
                        record.get("split"),
                        record.get("label"),
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
