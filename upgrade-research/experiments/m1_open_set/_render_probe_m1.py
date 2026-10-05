# -*- coding: utf-8 -*-
"""渲染探针：由 AppTest 加载，验证开放集徽章与证据矩阵能通过真实 streamlit 渲染。"""

import json
import os
import sys
from pathlib import Path

WS = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
for entry in (str(WS), str(WS / "frontend")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from components.evidence_dashboard import (  # noqa: E402
    _render_evidence_chain_page,
    _render_overview_cards,
)
from components.path_security import ResourcePathPolicy  # noqa: E402

spec = json.loads(Path(os.environ["FSTS_PROBE_PAYLOADS"]).read_text(encoding="utf-8"))
deepfake = json.loads(Path(spec["deepfake"]).read_text(encoding="utf-8"))
identity = json.loads(Path(spec["identity"]).read_text(encoding="utf-8"))
source = json.loads(Path(spec["source"]).read_text(encoding="utf-8"))
task_dir = Path(spec["task_dir"])
roots = [Path(task_dir)]
roots.extend(Path(item) for item in spec.get("geometry_roots") or [])
for item in (identity, source):
    query = Path(str(item.get("query_image") or ""))
    if query.parent.is_dir():
        roots.append(query.parent)
for item in (identity.get("candidates") or []):
    path = Path(str(item.get("reference_image") or ""))
    if path.parent.is_dir():
        roots.append(path.parent)
for item in (source.get("candidates") or []):
    path = Path(str(item.get("file_path") or ""))
    if path.parent.is_dir():
        roots.append(path.parent)
policy = ResourcePathPolicy(task_roots=tuple(dict.fromkeys(roots)))

if os.environ.get("FSTS_PROBE_MODE", "overview") == "overview":
    _render_overview_cards(
        deepfake, identity, source, request_key="probe_request", key_prefix="probe"
    )
else:
    _render_evidence_chain_page(
        deepfake,
        identity,
        source,
        task_dir=task_dir,
        task_status=None,
        policy=policy,
    )
