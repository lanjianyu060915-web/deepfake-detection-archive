# -*- coding: utf-8 -*-
"""验证报告文本层对二代/回退结果的输出。"""
import json
import sys
from pathlib import Path

WS = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "frontend"))

from reporting.report_builder import build_final_report, render_text_report  # noqa: E402

SAMPLES = Path(r"D:\刑科技项目升级2.0\experiments\m3_detector\service_integration")
for name in ("01.json", "03.json"):
    payload = json.loads((SAMPLES / name).read_text(encoding="utf-8"))
    report = build_final_report(f"probe_{name.split('.')[0]}", deepfake_result=payload)
    text = render_text_report(report)
    start = text.index("二、AI换脸")
    end = text.find("三、", start)
    print("=" * 78)
    print(f"### {name}")
    print(text[start:end if end > 0 else len(text)].rstrip())
