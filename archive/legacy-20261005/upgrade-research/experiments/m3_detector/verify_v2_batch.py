# -*- coding: utf-8 -*-
"""M3 端到端验收：按 config.json 的 deepfake_v2 路径驱动 worker，跑完 SWAN-DF 100 张，
核对 AUC 是否与离线实验结果一致，并统计分带分布与路由分支。"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import UPGRADE_ROOT  # noqa: E402
from fusion_eval import roc_auc  # noqa: E402

WS = UPGRADE_ROOT / "workspace" / "fsts"
DATASET = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
OUT = UPGRADE_ROOT / "experiments" / "m3_detector" / "v2_acceptance"
PROTOCOL = "face-source-trace-runtime-v1"


def read_message(proc, timeout=300.0):
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


def main() -> int:
    config = json.loads((WS / "config.json").read_text(encoding="utf-8"))
    section = config["deepfake"]
    python = Path(section["python"].replace("{APP_ROOT}", str(WS)))
    runner = Path(section["persistent_worker"].replace("{APP_ROOT}", str(WS)))
    detector_v2 = section["detector_v2"]
    print(f"config.json -> python={python.parent.parent.name} runner={runner.name} "
          f"内核={section.get('kernel')} "
          f"路由阈值={detector_v2['routing_short_edge_threshold_px']}px")
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([str(python), *section["python_args"], str(runner),
                             "--writable-root", str(OUT / "_runtime")],
                            cwd=str(runner.parent), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    rows = []
    started = time.time()
    try:
        ready = read_message(proc, 600)
        if not ready.get("ok"):
            raise SystemExit(f"worker 未就绪: {ready}")
        tasks = []
        for case_dir in sorted(DATASET.glob("案例*")):
            for image in sorted(case_dir.glob("*.jpg")):
                tasks.append((case_dir.name, image, 1 if image.name.startswith("05") else 0))
        for index, (case, image, label) in enumerate(tasks, 1):
            request_id = f"accept_{index:04d}"
            output = OUT / "results" / f"{index:04d}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            proc.stdin.write(json.dumps({
                "protocol": PROTOCOL, "type": "request", "request_id": request_id,
                "action": "run", "query_path": str(image),
                "output_path": str(output)}) + "\n")
            proc.stdin.flush()
            while True:
                message = read_message(proc)
                if message.get("request_id") == request_id and message.get("type") == "response":
                    break
            payload = message.get("payload") or {}
            details = payload.get("details") or {}
            rows.append({"case": case, "file": image.name, "y": label,
                         "score": payload.get("score"), "status": payload.get("status"),
                         "band": details.get("band"),
                         "routed_branch": details.get("routed_branch") or details.get("branch")})
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

    csv_path = OUT / "acceptance_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "file", "y", "score", "status",
                                                    "band", "routed_branch"])
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["status"] != "success"]
    succeeded = [row for row in rows if row["status"] == "success"]
    y = np.array([row["y"] for row in succeeded])
    scores = np.array([row["score"] for row in succeeded])
    auc = roc_auc(y, scores)
    bands = {}
    for row in rows:
        key = f"{'FAKE' if row['y'] else 'REAL'}/{row['band']}"
        bands[key] = bands.get(key, 0) + 1
    flagged = [row for row in rows if row["band"] in ("high", "medium")]
    high = [row for row in rows if row["band"] == "high"]
    summary = {
        "elapsed_seconds": round(time.time() - started, 1),
        "n": len(rows), "failures": len(failures), "auc": round(auc, 6),
        "expected_auc_onnx": 0.896875, "reference_auc_pytorch": 0.897813,
        "band_distribution": bands,
        "operating_profile": {
            "flagged_high_or_medium": {"n": len(flagged),
                                       "recall": round(sum(r["y"] for r in flagged) / 20, 4),
                                       "fpr": round(sum(1 - r["y"] for r in flagged) / 80, 4)},
            "flagged_high_only": {"n": len(high),
                                  "recall": round(sum(r["y"] for r in high) / 20, 4),
                                  "fpr": round(sum(1 - r["y"] for r in high) / 80, 4)},
            "note": "全量含定标案例，仅作上线自检，不作为论文指标",
        },
        "routed_branches": sorted({str(row["routed_branch"]) for row in rows}),
    }
    (OUT / "acceptance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = abs(auc - 0.896875) < 5e-4 and not failures
    print("验收结果:", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
