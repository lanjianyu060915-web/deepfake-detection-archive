# -*- coding: utf-8 -*-
r"""用 streamlit AppTest 跑 M4.1 渲染探针：视频级分带口径在「研判总览」卡片上的显示。

三种口径：
  1. band_majority —— 真实端到端视频任务的未标定口径；
  2. median_video_thresholds —— 合成校准口径，验证新增可读名与说明；
  3. 未知口径 —— 回退路径（可读名回显原值，说明回退到未标定口径）。

真实任务目录只读；合成口径的 JSON 写在本目录 _probe_payloads 下，不修改任务产物。

用法：
    D:\python\.venv\Scripts\python.exe -B -X utf8 test_render_video_bands.py
"""

import copy
import json
import os
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent / "m4_video" / "probe_overview_app.py"
TASK_DIR = Path(
    r"D:\刑科技项目升级2.0\experiments\data_root\runtime\tasks\case_20260920_170309_9ccb"
)
TASK_RESULT = TASK_DIR / "detection_result.json"
OUT_DIR = HERE / "_probe_payloads"

LABEL_MAJORITY = "帧级分带多数一致性（未标定口径）"
NOTE_MAJORITY = "帧级分数取中位数、分带按帧级一致性导出；不是概率，也不是对整段视频的鉴定结论。"
LABEL_CALIBRATED = "视频级中位分对照独立视频校准阈值"
NOTE_CALIBRATED = (
    "帧级分数取中位数，再与独立视频校准集真人视频分布的分位点比较定带；"
    "不是概率，也不是对整段视频的鉴定结论。"
)
SINGLE_IMAGE_HINT = "二代整图检测器 · 输入短边"

FAILURES: list[str] = []


def _load_task_result() -> dict:
    return json.loads(TASK_RESULT.read_text(encoding="utf-8"))


def _write_variant(name: str, rule: str, thresholds: dict) -> Path:
    payload = copy.deepcopy(_load_task_result())
    payload["details"]["video_band_rule"] = rule
    payload["details"]["video_band_thresholds"] = thresholds
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def run_probe(deepfake_path: Path) -> AppTest:
    os.environ["FSTS_PROBE_DEEPFAKE"] = str(deepfake_path)
    os.environ["FSTS_PROBE_TASK_DIR"] = str(TASK_DIR)
    os.environ.setdefault(
        "FSTS_DATA_ROOT", r"D:\刑科技项目升级2.0\experiments\data_root"
    )
    app = AppTest.from_file(str(PROBE), default_timeout=120)
    app.run(timeout=120)
    return app


def check(label: str, condition: bool, detail: str = "") -> None:
    suffix = f"（{detail}）" if detail else ""
    if condition:
        print(f"    [PASS] {label}{suffix}")
    else:
        print(f"    [FAIL] {label}{suffix}")
        FAILURES.append(f"{label} {detail}")


def render(name: str, deepfake_path: Path, expect_label: str, expect_note: str) -> None:
    print(f"### {name}")
    app = run_probe(deepfake_path)
    check(
        "页面无异常",
        not app.exception,
        str([item.message for item in app.exception]),
    )
    check("无错误横幅", not app.error, str([item.value for item in app.error]))
    metric_labels = [element.label for element in app.metric]
    check(
        "AIGC 卡片走视频口径",
        "视频级分带" in metric_labels and "视频级深伪风险分" in metric_labels,
        f"{metric_labels}",
    )
    captions = [element.value for element in app.caption]
    check(
        "显示聚合口径可读名",
        any(expect_label in text for text in captions),
        f"期望包含「{expect_label}」；实际 {captions}",
    )
    check(
        "显示对应口径说明",
        any(expect_note in text for text in captions),
        f"期望包含「{expect_note}」",
    )
    check(
        "未误用单图文案",
        not any(SINGLE_IMAGE_HINT in text for text in captions),
    )
    counts = [
        text
        for text in captions
        if "关键帧" in text and "高风险" in text and "聚合口径" in text
    ]
    check("关键帧统计行存在", len(counts) == 1, f"{counts}")
    if counts:
        check(
            "统计行与任务实际一致",
            "关键帧 4/4" in counts[0] and "高风险 2 帧" in counts[0],
            counts[0],
        )
    print()


render("未标定口径（真实端到端视频任务）", TASK_RESULT, LABEL_MAJORITY, NOTE_MAJORITY)

calibrated = _write_variant(
    "calibrated_rule",
    "median_video_thresholds",
    {
        "low_max": 0.0069995,
        "medium_max": 0.0136095,
        "statistic": "median",
    },
)
render("校准口径（合成，仅验证渲染）", calibrated, LABEL_CALIBRATED, NOTE_CALIBRATED)

unknown = _write_variant("unknown_rule", "future_rule_v9", {})
render("未知口径回退（合成）", unknown, "future_rule_v9", NOTE_MAJORITY)

print("失败项:", FAILURES if FAILURES else "无")
sys.exit(1 if FAILURES else 0)
