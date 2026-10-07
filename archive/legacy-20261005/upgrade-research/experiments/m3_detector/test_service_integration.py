# -*- coding: utf-8 -*-
"""M3 集成验收：按正式 config.json 启动 RuntimeServiceManager，
验证系统持久路径确实跑在二代内核上，且编排层的结果校验能通过。"""
import json
import os
import sys
from pathlib import Path

WS = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
os.environ["FSTS_DATA_ROOT"] = r"D:\刑科技项目升级2.0\experiments\data_root"
sys.path.insert(0, str(WS))

from orchestrator import _validate_deepfake_result  # noqa: E402
from runtime_services import RuntimeServiceManager  # noqa: E402

OUT = Path(r"D:\刑科技项目升级2.0\experiments\m3_detector\service_integration")
SWAN = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
SMALL = Path(r"D:\刑科技项目升级2.0\experiments\data_root\detectors\data_all"
             r"\data_formal\images\test")

CASES = [
    (SWAN / "案例01" / "05_换脸后图片.jpg", "vit_full_image", "FAKE"),
    (SWAN / "案例01" / "01_原始底图.jpg", "vit_full_image", "REAL"),
    (SMALL / "fake" / "094_111_f00_idx00000.jpg", "v040_face_crop", "FAKE"),
    (SMALL / "real" / "094_f00_idx00000.jpg", "v040_face_crop", "REAL"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    events = []
    manager = RuntimeServiceManager(config_path=WS / "config.json",
                                    progress_callback=events.append)
    failures = []
    try:
        status = manager.start()
        deepfake_status = status.get("deepfake", {})
        print(f"deepfake 服务: state={deepfake_status.get('state')} "
              f"ready={deepfake_status.get('ready')} pid={deepfake_status.get('pid')}")
        print(f"  健康详情: {json.dumps(deepfake_status.get('health_details'), ensure_ascii=False)}")
        if deepfake_status.get("ready") is not True:
            failures.append(f"deepfake 服务未就绪: {deepfake_status.get('error')}")

        for index, (path, expected_branch, expected_label) in enumerate(CASES, 1):
            output = OUT / f"{index:02d}.json"
            payload = manager.run_deepfake(query_path=path, output_path=output, timeout=300)
            valid, error = _validate_deepfake_result(payload)
            details = payload.get("details") or {}
            branch = details.get("routed_branch") or details.get("branch")
            print(f"[{index}] {path.name} status={payload.get('status')} "
                  f"score={payload.get('score')} 路由={branch} "
                  f"band={details.get('band')} 编排校验={'OK' if valid else error}")
            if not valid:
                failures.append(f"{path.name}: 编排层校验失败 {error}")
            if branch != expected_branch:
                failures.append(f"{path.name}: 路由={branch} 期望={expected_branch}")
            if payload.get("status") != "success":
                failures.append(f"{path.name}: status={payload.get('status')} "
                                f"error={payload.get('error_message')}")
            if not output.is_file():
                failures.append(f"{path.name}: 结果文件未落盘")

        service_events = [event for event in events if event.get("type") == "runtime_service"]
        print(f"进度事件 {len(events)} 条，其中运行时服务事件 {len(service_events)} 条")
    finally:
        manager.close()

    print("失败项:", failures if failures else "无")
    (OUT / "integration_result.json").write_text(
        json.dumps({"failures": failures, "cases": len(CASES)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
