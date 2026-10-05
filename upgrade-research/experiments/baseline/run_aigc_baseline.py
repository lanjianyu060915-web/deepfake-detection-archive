# -*- coding: utf-8 -*-
"""基线复现：用 v0.4.0 冻结 AIGC 工人进程对 SWAN-DF 案例1 的 100 张图跑鉴伪。

与 _paper_eval_20260919/run_aigc_batch.py 同一协议；路径全部指向升级工作区副本，
输出只写入 D:\刑科技项目升级2.0\experiments。数据集只读。
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

UPGRADE_ROOT = Path(r"D:\刑科技项目升级2.0")
WORKSPACE = UPGRADE_ROOT / "workspace" / "fsts"
EXPERIMENTS = UPGRADE_ROOT / "experiments" / "baseline"
DATASET = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")

PY = WORKSPACE / "packaging_v0.3" / "resources" / "portable_python" / "python.exe"
WORKER = WORKSPACE / "modules" / "deepfake_detection" / "persistent_worker.py"
INSTALL_ROOT = WORKSPACE / "packaging_v0.3" / "resources" / "deepfake_runtime"
OUT = EXPERIMENTS / "aigc"
WRITABLE = OUT / "_worker_runtime"
LOG = EXPERIMENTS / "logs" / "aigc_baseline_driver.log"
PROTOCOL = "face-source-trace-runtime-v1"


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def collect_tasks(limit=None):
    tasks = []
    for case_dir in sorted(DATASET.glob("案例*")):
        for image in sorted(case_dir.glob("*.jpg")):
            expected = "FAKE" if image.name.startswith("05") else "REAL"
            out_json = OUT / case_dir.name / image.stem / "detection_result.json"
            tasks.append((case_dir.name, image, expected, out_json))
    return tasks[:limit] if limit else tasks


def read_message(process, timeout_seconds: float):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                raise RuntimeError(f"worker 提前退出，returncode={process.returncode}")
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    tasks = collect_tasks(args.limit)
    log(f"任务总数：{len(tasks)}；数据集={DATASET}")
    proc = subprocess.Popen(
        [str(PY), "-E", "-S", "-B", str(WORKER),
         "--install-root", str(INSTALL_ROOT), "--device", "cpu",
         "--writable-root", str(WRITABLE)],
        cwd=str(WORKER.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    try:
        ready = read_message(proc, 600)
        log(f"worker ready: ok={ready.get('ok')} health={ready.get('health')}")
        if not ready.get("ok"):
            raise RuntimeError(f"worker 未就绪：{ready}")
        started_all = time.time()
        for index, (case_name, image, expected, out_json) in enumerate(tasks, 1):
            out_json.parent.mkdir(parents=True, exist_ok=True)
            request_id = f"{case_name}_{image.stem}"
            request = {"protocol": PROTOCOL, "type": "request", "request_id": request_id,
                       "action": "run", "query_path": str(image),
                       "output_path": str(out_json)}
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            started = time.time()
            while True:
                message = read_message(proc, args.timeout)
                if (message.get("request_id") == request_id
                        and message.get("type") == "response"):
                    break
            elapsed = time.time() - started
            payload = message.get("payload") or {}
            details = payload.get("details") or {}
            log(f"[{index}/{len(tasks)}] {case_name}/{image.name} expected={expected} "
                f"label={payload.get('label')} score={payload.get('score')} "
                f"spatial={details.get('spatial_score')} "
                f"freq={details.get('frequency_score')} "
                f"status={payload.get('status')} {elapsed:.1f}s")
        log(f"全部完成，总耗时 {time.time() - started_all:.1f}s")
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


if __name__ == "__main__":
    main()
