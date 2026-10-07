# -*- coding: utf-8 -*-
"""用 streamlit AppTest 跑渲染探针，确认二代/回退两种结果都不报错。"""
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
PROBE = HERE / "_render_probe.py"
SAMPLES = HERE / "service_integration"

failures = []
for name in ("01.json", "02.json", "03.json", "04.json"):
    os.environ["FSTS_PROBE_PAYLOAD"] = str(SAMPLES / name)
    app = AppTest.from_file(str(PROBE), default_timeout=60)
    app.run()
    if app.exception:
        failures.append(f"{name}: {app.exception}")
        print(f"[{name}] EXCEPTION: {app.exception}")
        continue
    metrics = [(item.label, item.value) for item in app.metric]
    captions = [item.value for item in app.caption]
    print(f"[{name}] OK  metrics={metrics}")
    highlights = [text for text in captions if "二代" in text or "路由" in text]
    for text in highlights:
        print(f"        caption: {text}")

print("失败项:", failures if failures else "无")
sys.exit(1 if failures else 0)
