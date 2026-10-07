# -*- coding: utf-8 -*-
"""渲染探针：由 AppTest 加载，验证二代内核结果能通过真实 streamlit 渲染。"""
import json
import os
import sys
from pathlib import Path

WS = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
for entry in (str(WS), str(WS / "frontend")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from components.evidence_dashboard import _render_deepfake_page  # noqa: E402
from components.path_security import ResourcePathPolicy  # noqa: E402

payload_path = Path(os.environ["FSTS_PROBE_PAYLOAD"])
payload = json.loads(payload_path.read_text(encoding="utf-8"))
query = Path(str(payload.get("query_image") or ""))
roots = (query.parent,) if query.parent.is_dir() else ()
policy = ResourcePathPolicy(task_roots=roots)
_render_deepfake_page(payload, policy, "probe")
