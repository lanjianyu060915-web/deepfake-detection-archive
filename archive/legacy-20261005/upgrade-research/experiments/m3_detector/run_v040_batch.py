# -*- coding: utf-8 -*-
"""用 v0.4.0 冻结鉴伪工人对任意「fake/real」图片目录批量推理（只读数据集，只写升级工作区）。
用法：
    python run_v040_batch.py --root <含 fake/real 的目录> --tag <名字>
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import UPGRADE_ROOT  # noqa: E402

WORKSPACE = UPGRADE_ROOT / "workspace" / "fsts"
PY = WORKSPACE / "packaging_v0.3" / "resources" / "portable_python" / "python.exe"
WORKER = WORKSPACE / "modules" / "deepfake_detection" / "persistent_worker.py"
INSTALL_ROOT = WORKSPACE / "packaging_v0.3" / "resources" / "deepfake_runtime"
LOG = UPGRADE_ROOT / "experiments" / "logs" / "m3_v040_batch.log"
PROTOCOL = "face-source-trace-runtime-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_message(process, timeout_seconds: float):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                raise RuntimeError(f"worker 提前退出 returncode={process.returncode}")
            continue
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise TimeoutError("等待 worker 响应超时")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = UPGRADE_ROOT / "experiments" / "m3_detector" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    writable = out_dir / "_worker_runtime"
    tasks = []
    for label_name, label in (("fake", 1), ("real", 0)):
        folder = root / label_name
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                tasks.append((label_name, image, label))
    log(f"{args.tag}: 任务数={len(tasks)} root={root}")

    proc = subprocess.Popen(
        [str(PY), "-E", "-S", "-B", str(WORKER), "--install-root", str(INSTALL_ROOT),
         "--device", "cpu", "--writable-root", str(writable)],
        cwd=str(WORKER.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    rows = []
    started_all = time.time()
    try:
        ready = read_message(proc, 900)
        log(f"worker ready: ok={ready.get('ok')} health={ready.get('health')}")
        if not ready.get("ok"):
            raise RuntimeError(f"worker 未就绪：{ready}")
        for index, (folder, image, label) in enumerate(tasks, 1):
            request_id = f"{args.tag}_{index:04d}"
            output_path = out_dir / "_native" / f"{index:04d}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            request = {"protocol": PROTOCOL, "type": "request", "request_id": request_id,
                       "action": "run", "query_path": str(image),
                       "output_path": str(output_path)}
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            while True:
                message = read_message(proc, args.timeout)
                if message.get("request_id") == request_id and message.get("type") == "response":
                    break
            payload = message.get("payload") or {}
            details = payload.get("details") or {}
            rows.append({
                "folder": folder, "file": image.name, "y": label,
                "label": payload.get("label"),
                "score": float(payload.get("score") or 0.0),
                "spatial": float(details.get("spatial_score") or 0.0),
                "frequency": float(details.get("frequency_score") or 0.0),
            })
            if index % 20 == 0 or index == len(tasks):
                elapsed = time.time() - started_all
                log(f"  {index}/{len(tasks)} 已用 {elapsed:.0f}s 预计剩余 "
                    f"{elapsed / index * (len(tasks) - index):.0f}s")
    finally:
        try:
            proc.stdin.write(json.dumps({"protocol": PROTOCOL, "type": "request",
                                         "request_id": "shutdown",
                                         "action": "shutdown"}) + "\n")
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()

    import csv
    csv_path = out_dir / "scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["folder", "file", "y", "label",
                                                    "score", "spatial", "frequency"])
        writer.writeheader()
        writer.writerows(rows)

    import numpy as np
    from eval_swan_df import roc_auc
    y = np.array([row["y"] for row in rows])
    summary = {"tag": args.tag, "root": str(root), "n": len(rows), "n_fake": int(y.sum())}
    for key in ("score", "spatial", "frequency"):
        values = np.array([row[key] for row in rows])
        summary[key] = {"auc": round(roc_auc(y, values), 6),
                        "real_mean": round(float(values[y == 0].mean()), 8),
                        "fake_mean": round(float(values[y == 1].mean()), 8)}
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"{args.tag} 完成 用时 {time.time() - started_all:.0f}s -> {csv_path}")
    log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
