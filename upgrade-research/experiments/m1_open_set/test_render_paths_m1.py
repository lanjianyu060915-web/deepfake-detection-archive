r"""用 streamlit AppTest 跑 M1 渲染探针：开放集徽章与证据矩阵章节。

对三类真实结果分别渲染「研判总览」和「证据链与完整报告」两个页面，
确认新增的三态徽章与证据矩阵不会让页面抛异常，且结论文字与实际判定一致。

用法：
    D:\python\.venv\Scripts\python.exe test_render_paths_m1.py
"""

import json
import os
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
PROBE = HERE / "_render_probe_m1.py"
EV = Path(r"D:\刑侦查竞赛\_paper_eval_20260919")
M3 = Path(r"D:\刑科技项目升级2.0\experiments\m3_detector\service_integration")
LIBRARY_ROOTS = [
    r"D:\刑侦查竞赛\first_week\data\source_library",
    r"D:\刑侦查竞赛\first_week\data\identity_library",
]

CASES = [
    {
        "name": "库外查询",
        "deepfake": M3 / "01.json",
        "identity": HERE / "identity_holdout" / "案例11" / "identity_result.json",
        "source": HERE / "source_holdout" / "案例11" / "source_result.json",
        "task_dir": HERE / "source_holdout" / "案例11",
        "expect_badge": "当前库中未发现可靠候选",
        "expect_quality": "良好",
    },
    {
        "name": "库内命中",
        "deepfake": M3 / "03.json",
        "identity": EV / "identity" / "query_001" / "identity_result.json",
        "source": EV / "source" / "query_001" / "source_result.json",
        "task_dir": EV / "source" / "query_001",
        "expect_badge": "高可信候选",
        "expect_quality": "良好",
    },
    {
        "name": "无法比对",
        "deepfake": M3 / "01.json",
        "identity": HERE / "identity_holdout" / "案例01" / "identity_result.json",
        "source": HERE / "source_holdout" / "案例11" / "source_result.json",
        "task_dir": HERE / "identity_holdout" / "案例01",
        "expect_badge": "本次无法完成可比对检索",
        "expect_quality": "一般",
    },
]

CHAIN_METRICS = ("查询图像质量", "身份库覆盖", "几何一致性", "开放集风险")

FAILURES: list[str] = []


def run_probe(case: dict, mode: str) -> AppTest:
    spec = HERE / f"_probe_payloads_{mode}.json"
    spec.write_text(
        json.dumps(
            {
                "deepfake": str(case["deepfake"]),
                "identity": str(case["identity"]),
                "source": str(case["source"]),
                "task_dir": str(case["task_dir"]),
                "geometry_roots": LIBRARY_ROOTS,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.environ["FSTS_PROBE_PAYLOADS"] = str(spec)
    os.environ["FSTS_PROBE_MODE"] = mode
    app = AppTest.from_file(str(PROBE), default_timeout=90)
    app.run()
    return app


def check(label: str, condition: bool, detail: str = "") -> None:
    suffix = f"（{detail}）" if detail else ""
    if condition:
        print(f"    [PASS] {label}{suffix}")
    else:
        print(f"    [FAIL] {label}{suffix}")
        FAILURES.append(f"{label} {detail}")


for case in CASES:
    print(f"### {case['name']}")
    overview = run_probe(case, "overview")
    check("总览页无异常", not overview.exception, str(overview.exception))
    banner_texts = [
        element.value
        for group in (overview.success, overview.warning, overview.error, overview.info)
        for element in group
    ]
    badges = [text for text in banner_texts if "开放集判定" in text]
    check("总览页出现两处开放集徽章", len(badges) == 2, f"{len(badges)} 处")
    check(
        "徽章文字与判定一致",
        any(case["expect_badge"] in text for text in badges),
        f"期望包含「{case['expect_badge']}」",
    )
    chain = run_probe(case, "chain")
    check("证据链页无异常", not chain.exception, str(chain.exception))
    markdown = [element.value for element in chain.markdown]
    check("证据矩阵标题存在", any("证据矩阵（研判卡）" in text for text in markdown))
    metric_labels = [element.label for element in chain.metric]
    missing = [label for label in CHAIN_METRICS if label not in metric_labels]
    check("四项矩阵指标齐全", not missing, f"缺失 {missing}" if missing else "")
    quality = [element.value for element in chain.metric if element.label == "查询图像质量"]
    check(
        "查询图像质量与预期一致",
        bool(quality) and quality[0] == case["expect_quality"],
        f"{quality}",
    )
    json_blocks = [element.value for element in chain.json]

    def _as_mapping(block: object) -> dict:
        if isinstance(block, dict):
            return block
        if isinstance(block, str):
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    matrix_blocks = [_as_mapping(block) for block in json_blocks]
    matrix_ok = any(
        "modules" in block and "open_set_risk" in block for block in matrix_blocks
    )
    check("可展开的完整矩阵结构存在", matrix_ok)
    if matrix_ok:
        matrix_block = next(
            block for block in matrix_blocks if "modules" in block and "open_set_risk" in block
        )
        states = {
            key: value.get("grade_label")
            for key, value in matrix_block["modules"].items()
            if isinstance(value, dict)
        }
        check(
            "矩阵 JSON 逐模块带结论标签",
            all(isinstance(label, str) and label for label in states.values()),
            f"{states}",
        )
    print()

print("失败项:", FAILURES if FAILURES else "无")
sys.exit(1 if FAILURES else 0)
