r"""M2 常驻 worker v2 协议探针：握手 / 健康检查 / 尺度回退 / 优雅退出。

本脚本验证的是 orchestrator 与 runtime_services 实际会走的那条路径，
即「把 modules/identity_search_v2/persistent_worker_v2.py 当常驻子进程拉起」，
而不是 CLI worker 的一次性调用。检查项：

1. ready 握手：ready=True、health=ok、details.worker="identity_v2"，
   回退策略与检测尺度链与实现一致；
2. health：ok=True，并汇报当前检测尺度；
3. run 案例01（冻结 1280 档漏检）：status=success 且回退到 640，
   method.preprocessing 与 metrics 都写入证据字段；
4. run 案例11（冻结 1280 档可用）：不回退，停在 1280，只尝试一次；
5. shutdown：返回 shutting_down=True，且进程以 0 退出。

stdout 上除协议 JSON 外还有 InsightFace/onnxruntime 的日志行，本脚本只取
protocol == "face-source-trace-runtime-v1" 的消息，其余计入
non_protocol_stdout_lines 以便事后核对丢帧情况。

用法：
    D:\python\.venv\Scripts\python.exe probe_persistent_worker_v2.py
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


UPGRADE_ROOT = Path(r"D:\刑科技项目升级2.0")
WORKSPACE = UPGRADE_ROOT / "workspace" / "fsts"
PYTHON = Path(r"D:\python\.venv\Scripts\python.exe")
WORKER = WORKSPACE / "modules" / "identity_search_v2" / "persistent_worker_v2.py"
SWAN_ROOT = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
LIBRARY = Path(r"D:\刑侦查竞赛\first_week\data\identity_library")
MODEL_ROOT = WORKSPACE / "packaging_v0.3" / "resources"
CACHE_DIR = UPGRADE_ROOT / "experiments" / "m1_open_set" / "_identity_cache"
OUT_DIR = UPGRADE_ROOT / "experiments" / "m2_preprocess" / "persistent_probe"

PROTOCOL_NAME = "face-source-trace-runtime-v1"
EXPECTED_CHAIN = [1280, 640, 320]
EXPECTED_POLICY = "scale_fallback_on_no_face"
FALLBACK_CASE = "案例01"
STABLE_CASE = "案例11"
TOP_K = 5

checks: list[dict[str, Any]] = []


def record(name: str, ok: bool, detail: Any = None) -> bool:
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    suffix = "" if detail is None else f" -> {detail}"
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}", flush=True)
    return bool(ok)


def as_list(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return None


class ProtocolClient:
    """读取子进程 stdout，只保留协议 JSON 行。"""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self.messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.dropped_lines: list[str] = []
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        stdout = self.process.stdout
        if stdout is None:
            return
        for line in stdout:
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                self.dropped_lines.append(text[:200])
                continue
            if isinstance(message, dict) and message.get("protocol") == PROTOCOL_NAME:
                self.messages.put(message)
            else:
                self.dropped_lines.append(text[:200])

    def expect(self, message_type: str, timeout: float = 600.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"等待 {message_type} 消息超时（{timeout:.0f}s）")
            try:
                message = self.messages.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"worker 已退出（code={self.process.returncode}），"
                        f"未收到 {message_type} 消息"
                    )
                continue
            if message.get("type") == message_type:
                return message

    def send(self, action: str, request_id: str, **fields: Any) -> None:
        stdin = self.process.stdin
        if stdin is None:
            raise RuntimeError("worker stdin 不可用")
        payload = {
            "protocol": PROTOCOL_NAME,
            "type": "request",
            "request_id": request_id,
            "action": action,
            **fields,
        }
        stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stdin.flush()


def request_run(
    client: ProtocolClient,
    request_id: str,
    case: str,
    output_path: Path,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = None
    client.send(
        "run",
        request_id,
        query_path=str(SWAN_ROOT / case / "05_换脸后图片.jpg"),
        output_path=str(output_path),
        library_path=str(LIBRARY),
        top_k=TOP_K,
    )
    response = client.expect("response", timeout=timeout)
    return response, (response.get("payload") or {})


def preprocessing_of(payload: dict[str, Any]) -> dict[str, Any]:
    method = payload.get("method")
    if not isinstance(method, dict):
        return {}
    block = method.get("preprocessing")
    return block if isinstance(block, dict) else {}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for required in (WORKER, LIBRARY, MODEL_ROOT, CACHE_DIR):
        if not required.exists():
            print(f"缺少前置资产：{required}", file=sys.stderr)
            return 2

    stderr_log = OUT_DIR / "worker_stderr.log"
    outputs = {
        FALLBACK_CASE: OUT_DIR / f"{FALLBACK_CASE}_identity_result.json",
        STABLE_CASE: OUT_DIR / f"{STABLE_CASE}_identity_result.json",
    }

    client: ProtocolClient | None = None
    process: subprocess.Popen | None = None
    started = time.time()
    try:
        with stderr_log.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                [
                    str(PYTHON),
                    "-B",
                    str(WORKER),
                    "--library",
                    str(LIBRARY),
                    "--cache-dir",
                    str(CACHE_DIR),
                    "--model-root",
                    str(MODEL_ROOT),
                    "--device",
                    "cpu",
                    "--ctx",
                    "-1",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            client = ProtocolClient(process)

            ready = client.expect("ready", timeout=900.0)
            details = ready.get("details") or {}
            record(
                "ready 握手：ready=True 且 health=ok",
                ready.get("ready") is True and ready.get("health") == "ok",
                {"ready": ready.get("ready"), "health": ready.get("health")},
            )
            record(
                "ready 握手：worker 标识为 identity_v2",
                details.get("worker") == "identity_v2",
                details.get("worker"),
            )
            record(
                "ready 握手：回退策略已声明",
                details.get("detection_policy") == EXPECTED_POLICY,
                details.get("detection_policy"),
            )
            record(
                "ready 握手：检测尺度链为 1280,640,320",
                as_list(details.get("det_size_chain")) == EXPECTED_CHAIN,
                details.get("det_size_chain"),
            )
            record(
                "ready 握手：初始检测尺度为主档 1280",
                as_list(details.get("detection_scale_current")) == [1280, 1280],
                details.get("detection_scale_current"),
            )

            client.send("health", "probe-health")
            health = client.expect("health", timeout=300.0)
            health_details = health.get("details") or {}
            record("health：ok=True", health.get("ok") is True, health.get("health"))
            record(
                "health：沿用冻结身份库字段",
                health_details.get("model") == "buffalo_l"
                and health_details.get("model_instances") == 1,
                {
                    "model": health_details.get("model"),
                    "model_instances": health_details.get("model_instances"),
                    "cached_libraries": health_details.get("cached_libraries"),
                },
            )

            response, payload = request_run(
                client, "probe-run-fallback", FALLBACK_CASE, outputs[FALLBACK_CASE], 1800.0
            )
            block = preprocessing_of(payload)
            attempts = block.get("attempts") or []
            metrics = payload.get("metrics") or {}
            record(
                f"{FALLBACK_CASE}（冻结漏检）：响应 ok",
                response.get("ok") is True,
                response.get("error"),
            )
            record(
                f"{FALLBACK_CASE}：status=success",
                payload.get("status") == "success",
                payload.get("status"),
            )
            record(
                f"{FALLBACK_CASE}：回退到 640 档",
                block.get("detection_scale_used") == 640
                and block.get("fallback_applied") is True,
                {
                    "detection_scale_used": block.get("detection_scale_used"),
                    "fallback_applied": block.get("fallback_applied"),
                },
            )
            record(
                f"{FALLBACK_CASE}：库索引仍按主档 1280 口径",
                block.get("library_det_size") == 1280,
                block.get("library_det_size"),
            )
            record(
                f"{FALLBACK_CASE}：逐档记录为 1280 失败 -> 640 成功",
                [entry.get("scale") for entry in attempts] == [1280, 640]
                and [entry.get("status") for entry in attempts] == ["error", "success"],
                attempts,
            )
            record(
                f"{FALLBACK_CASE}：metrics 同步写明实际尺度",
                metrics.get("detection_scale_used") == 640
                and metrics.get("detection_attempt_count") == 2,
                {
                    "detection_scale_used": metrics.get("detection_scale_used"),
                    "detection_attempt_count": metrics.get("detection_attempt_count"),
                },
            )
            record(
                f"{FALLBACK_CASE}：候选可用（top1={((payload.get('candidates') or [{}])[0].get('identity_id'))}）",
                bool(payload.get("candidates")),
                len(payload.get("candidates") or []),
            )

            response, stable_payload = request_run(
                client, "probe-run-stable", STABLE_CASE, outputs[STABLE_CASE], 1800.0
            )
            stable_block = preprocessing_of(stable_payload)
            stable_attempts = stable_block.get("attempts") or []
            record(
                f"{STABLE_CASE}（冻结可用）：响应 ok 且 status=success",
                response.get("ok") is True and stable_payload.get("status") == "success",
                stable_payload.get("status"),
            )
            record(
                f"{STABLE_CASE}：不回退，停在主档 1280",
                stable_block.get("detection_scale_used") == 1280
                and stable_block.get("fallback_applied") is False
                and len(stable_attempts) == 1,
                {
                    "detection_scale_used": stable_block.get("detection_scale_used"),
                    "fallback_applied": stable_block.get("fallback_applied"),
                    "attempt_count": len(stable_attempts),
                },
            )

            expected_scales = {FALLBACK_CASE: 640, STABLE_CASE: 1280}
            for case, output_path in outputs.items():
                on_disk = (
                    json.loads(output_path.read_text(encoding="utf-8"))
                    if output_path.is_file()
                    else {}
                )
                on_disk_block = preprocessing_of(on_disk)
                record(
                    f"{case}：结果 JSON 落盘且带预处理证据块",
                    output_path.is_file()
                    and on_disk.get("status") == "success"
                    and on_disk_block.get("detection_scale_used")
                    == expected_scales[case],
                    {
                        "output": str(output_path),
                        "scale_on_disk": on_disk_block.get("detection_scale_used"),
                        "expected": expected_scales[case],
                    },
                )

            client.send("shutdown", "probe-shutdown")
            shutdown = client.expect("response", timeout=300.0)
            record(
                "shutdown：返回 shutting_down=True",
                shutdown.get("shutting_down") is True,
                shutdown.get("shutting_down"),
            )
            exit_code = process.wait(timeout=300.0)
            record("shutdown：进程以 0 退出", exit_code == 0, exit_code)
    except (TimeoutError, RuntimeError) as exc:
        record("协议探针完整跑完", False, str(exc))
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=30.0)
            except Exception:
                pass

    failed = [entry for entry in checks if not entry["ok"]]
    report = {
        "probe": "persistent_worker_v2",
        "worker": str(WORKER),
        "protocol": PROTOCOL_NAME,
        "expected_chain": EXPECTED_CHAIN,
        "fallback_case": FALLBACK_CASE,
        "stable_case": STABLE_CASE,
        "wall_seconds": round(time.time() - started, 1),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "checks": checks,
        "non_protocol_stdout_lines": len(client.dropped_lines) if client else None,
        "non_protocol_sample": (client.dropped_lines[:5] if client else []),
        "stderr_log": str(stderr_log),
    }
    (OUT_DIR / "probe_persistent_worker_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\n协议探针：{report['passed']} 通过 / {report['failed']} 失败，"
        f"耗时 {report['wall_seconds']}s；非协议 stdout 行 {report['non_protocol_stdout_lines']} 条"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
