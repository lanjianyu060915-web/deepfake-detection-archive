import json
import sys
from pathlib import Path

ROOT = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT))

from components.evidence_dashboard import resolve_deepfake_kernel_view
from reporting.explainability import build_deepfake_explanation

SAMPLES = [
    Path(r"D:\刑科技项目升级2.0\experiments\m3_detector\service_integration\01.json"),
    Path(r"D:\刑科技项目升级2.0\experiments\m3_detector\service_integration\02.json"),
    Path(r"D:\刑科技项目升级2.0\experiments\m3_detector\service_integration\03.json"),
    Path(r"D:\刑科技项目升级2.0\experiments\m3_detector\service_integration\04.json"),
]

for sample in SAMPLES:
    payload = json.loads(sample.read_text(encoding="utf-8"))
    view = resolve_deepfake_kernel_view(payload)
    print("=" * 78)
    print(f"[{sample.name}] status={payload.get('status')} score={payload.get('score')}")
    print("  kernel_view:", json.dumps(view, ensure_ascii=False) if view else None)
    explanation = build_deepfake_explanation(payload)
    print("  headline  :", explanation["headline"])
    print("  summary   :", explanation["summary"])
    for item in explanation["basis"]:
        print("  basis     :", item)
    for item in explanation["warnings"]:
        print("  warning   :", item)
    for item in explanation["limitations"]:
        print("  limitation:", item)
    technical = explanation["technical"]
    print("  technical keys:", sorted(technical.keys()))
    print("  model_name:", technical.get("model_name"), "| branch:", technical.get("branch"), "| band:", technical.get("band"))
