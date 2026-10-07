# -*- coding: utf-8 -*-
"""M3 回归测试：直接驱动 v2 便携运行时+worker，验证协议、分带与结果自洽。"""
import json
import subprocess
import sys
from pathlib import Path

WS = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
PY = WS / "packaging_v0.4" / "resources" / "portable_python_v2" / "python.exe"
WORKER = WS / "modules" / "deepfake_detection_v2" / "persistent_worker_v2.py"
OUT = Path(r"D:\刑科技项目升级2.0\experiments\m3_detector\v2_worker_smoke")
PROTOCOL = "face-source-trace-runtime-v1"
CASES = {
    "案例01/05_换脸后图片.jpg": ("FAKE", "vit_full_image"),
    "案例01/01_原始底图.jpg": ("REAL", "vit_full_image"),
    "案例09/05_换脸后图片.jpg": ("FAKE", "vit_full_image"),
}
DATASET = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
SMALL_CROP = Path(r"D:\刑科技项目升级2.0\experiments\data_root\detectors\data_all"
                  r"\data_formal\images\test")
SMALL_CASES = {
    "fake/094_111_f00_idx00000.jpg": ("FAKE", "v040_face_crop"),
    "real/094_f00_idx00000.jpg": ("REAL", "v040_face_crop"),
}


def read_message(proc, timeout=180.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(f"worker 退出 {proc.returncode}")
            continue
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise TimeoutError("等待超时")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([str(PY), "-E", "-S", "-B", str(WORKER),
                             "--writable-root", str(OUT / "_runtime")],
                            cwd=str(WORKER.parent), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    failures = []
    try:
        ready = read_message(proc, 300)
        print("ready:", json.dumps(ready.get("details"), ensure_ascii=False))
        if not ready.get("ok"):
            raise SystemExit(f"worker 未就绪: {ready}")
        tasks = [(DATASET, name, value) for name, value in CASES.items()]
        tasks += [(SMALL_CROP, name, value) for name, value in SMALL_CASES.items()]
        for index, (base, relative, (expected, expected_branch)) in enumerate(tasks, 1):
            request_id = f"smoke_{index}"
            output = OUT / f"{index:02d}.json"
            proc.stdin.write(json.dumps({
                "protocol": PROTOCOL, "type": "request", "request_id": request_id,
                "action": "run", "query_path": str(base / relative),
                "output_path": str(output)}) + "\n")
            proc.stdin.flush()
            while True:
                message = read_message(proc)
                if message.get("request_id") == request_id and message.get("type") == "response":
                    break
            payload = message.get("payload") or {}
            details = payload.get("details") or {}
            routed = details.get("routed_branch") or details.get("branch")
            score = payload.get("score")
            print(f"{relative} 期望={expected}/{expected_branch} status={payload.get('status')} "
                  f"score={'None' if score is None else f'{score:.3e}'} "
                  f"band={details.get('band')} 路由={routed} label={payload.get('label')}")
            if payload.get("status") != "success":
                failures.append(f"{relative}: status={payload.get('status')} "
                                f"error={payload.get('error_message')}")
            if routed != expected_branch:
                failures.append(f"{relative}: 路由分支={routed} 期望={expected_branch}")
            written = json.loads(output.read_text(encoding="utf-8"))
            if written.get("status") != payload.get("status"):
                failures.append(f"{relative}: 落盘结果与协议返回不一致")
            if expected_branch == "vit_full_image" and not written.get(
                    "provenance", {}).get("model_sha256"):
                failures.append(f"{relative}: 落盘结果缺少模型哈希")
            if payload.get("score") is None:
                failures.append(f"{relative}: 分数为空")
        # 错误路径：不存在的文件
        request_id = "smoke_missing"
        proc.stdin.write(json.dumps({
            "protocol": PROTOCOL, "type": "request", "request_id": request_id,
            "action": "run", "query_path": str(OUT / "不存在.jpg"),
            "output_path": str(OUT / "missing.json")}) + "\n")
        proc.stdin.flush()
        while True:
            message = read_message(proc)
            if message.get("request_id") == request_id and message.get("type") == "response":
                break
        if (message.get("payload") or {}).get("status") != "error":
            failures.append("缺失文件未返回结构化错误")
        else:
            print("缺失文件 ->", (message["payload"]).get("error_message"))
    finally:
        try:
            proc.stdin.write(json.dumps({"protocol": PROTOCOL, "type": "request",
                                         "request_id": "shutdown",
                                         "action": "shutdown"}) + "\n")
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
    print("失败项:", failures if failures else "无")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
