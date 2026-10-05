# -*- coding: utf-8 -*-
"""渲染探针：验证「研判总览」头部与三张模块卡片在单图/视频两种输入下的口径。

由 AppTest 加载执行，环境变量：
  FSTS_PROBE_DEEPFAKE  必填，鉴伪结果 JSON 路径（单图或视频任务的 detection_result.json）
  FSTS_PROBE_IDENTITY  选填，身份结果 JSON；缺省用内置最小单图样例
  FSTS_PROBE_SOURCE    选填，底图结果 JSON；缺省用内置最小单图样例
  FSTS_PROBE_TASK_DIR  选填，任务目录；目录内有 video_summary.json 时按视频口径渲染
"""

import json
import os
import sys
from pathlib import Path

WS = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
for entry in (str(WS), str(WS / "frontend")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import streamlit as st  # noqa: E402

from components.evidence_dashboard import (  # noqa: E402
    _render_overview_cards,
    _render_task_header,
    build_task_overview,
)
from components.path_security import ResourcePathPolicy  # noqa: E402

DEFAULT_IDENTITY = {
    "status": "success",
    "candidates": [
        {"identity_id": "person_001", "identity_name": "person_001", "mean_similarity": 0.71, "rank": 1},
        {"identity_id": "person_002", "identity_name": "person_002", "mean_similarity": 0.29, "rank": 2},
    ],
    "metrics": {"top1_top2_margin": 0.42, "library_identity_count": 9, "valid_embedding_count": 43},
    "details": {},
    "runtime_seconds": 12.5,
}

DEFAULT_SOURCE = {
    "status": "success",
    "candidates": [
        {
            "rank": 1,
            "file_name": "sample.jpg",
            "global_score": 0.97,
            "local_score": 0.66,
            "final_score": 0.83,
            "ransac_inlier_count": 21,
            "ransac_inlier_ratio": 0.91,
        }
    ],
    "metrics": {"top1_top2_margin": 0.18},
    "details": {},
    "runtime_seconds": 30.0,
}


def load(env_name: str, default: dict) -> dict:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return dict(default)
    return json.loads(Path(value).read_text(encoding="utf-8"))


deepfake = load("FSTS_PROBE_DEEPFAKE", {})
identity = load("FSTS_PROBE_IDENTITY", DEFAULT_IDENTITY)
source = load("FSTS_PROBE_SOURCE", DEFAULT_SOURCE)
task_dir_value = os.environ.get("FSTS_PROBE_TASK_DIR", "").strip()
task_dir = Path(task_dir_value) if task_dir_value else Path(os.environ.get("TEMP", "."))
video_path = task_dir / "video_summary.json"
video_payload = json.loads(video_path.read_text(encoding="utf-8")) if video_path.is_file() else {}

policy = ResourcePathPolicy(task_roots=(task_dir,))
overview = build_task_overview(deepfake, identity, source, task_dir=task_dir)
st.set_page_config(page_title="总览口径探针", layout="wide")
_render_task_header(overview, video=video_payload)
_render_overview_cards(
    deepfake,
    identity,
    source,
    request_key="probe_request",
    key_prefix="probe",
)
