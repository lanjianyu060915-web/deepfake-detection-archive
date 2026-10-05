# -*- coding: utf-8 -*-
r"""M1 端到端验证：用真实结果文件驱动开放集判定、证据矩阵与文本报告。

覆盖三类真实样本（结果文件全部取自既有实验，不重新推理、不改动数据）：

- 库外查询：v0.4.0 闭集排序一定会给出 Top-1，正确结论应为「当前库中未发现可靠候选」；
- 库内命中：应判为高可信候选；
- 无法比对：查询未检出人脸，应判为 unavailable，同时确认报告仍能完整渲染。

用法：
    D:\python\.venv\Scripts\python.exe verify_open_set_e2e.py
"""

import json
import sys
from pathlib import Path

WS = Path(r"D:\刑科技项目升级2.0\workspace\fsts")
sys.path.insert(0, str(WS))

from decision.evidence_matrix import build_evidence_matrix  # noqa: E402
from decision.open_set import (  # noqa: E402
    STATE_HIGH,
    STATE_NONE,
    STATE_UNAVAILABLE,
    active_thresholds,
    load_thresholds,
    resolve_thresholds_path,
)
from reporting.report_builder import build_final_report, render_text_report  # noqa: E402

EV = Path(r"D:\刑侦查竞赛\_paper_eval_20260919")
M1 = Path(r"D:\刑科技项目升级2.0\experiments\m1_open_set")
M3 = Path(r"D:\刑科技项目升级2.0\experiments\m3_detector\service_integration")
THRESHOLDS = WS / "packaging_v0.4" / "resources" / "decision" / "open_set_thresholds.json"

CASES = [
    {
        "name": "库外查询",
        "detail": "底图 holdout/案例11（库外） + 身份 holdout/案例11（库内无该身份）",
        "deepfake": M3 / "01.json",
        "identity": M1 / "identity_holdout" / "案例11" / "identity_result.json",
        "source": M1 / "source_holdout" / "案例11" / "source_result.json",
        "expect": {"identity": STATE_NONE, "source": STATE_NONE},
    },
    {
        "name": "库内命中",
        "detail": "底图 query_001（库内命中） + 身份 query_001（库内命中）",
        "deepfake": M3 / "03.json",
        "identity": EV / "identity" / "query_001" / "identity_result.json",
        "source": EV / "source" / "query_001" / "source_result.json",
        "expect": {"identity": STATE_HIGH, "source": STATE_HIGH},
    },
    {
        "name": "无法比对",
        "detail": "身份 holdout/案例01 未检出人脸；底图 holdout/案例11（库外）",
        "deepfake": M3 / "01.json",
        "identity": M1 / "identity_holdout" / "案例01" / "identity_result.json",
        "source": M1 / "source_holdout" / "案例11" / "source_result.json",
        "expect": {"identity": STATE_UNAVAILABLE, "source": STATE_NONE},
    },
]

MATRIX_LABELS = (
    "AIGC鉴伪",
    "人脸身份检索",
    "原始底图溯源",
    "查询图像质量",
    "身份库覆盖",
    "几何一致性",
    "开放集风险",
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    suffix = f"（{detail}）" if detail else ""
    if condition:
        print(f"  [PASS] {label}{suffix}")
    else:
        print(f"  [FAIL] {label}{suffix}")
        FAILURES.append(f"{label}: {detail}")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    thresholds = load_thresholds(THRESHOLDS)
    print(f"阈值配置：{THRESHOLDS}（source_version={thresholds.get('source_version')}）")
    resolved = resolve_thresholds_path()
    check(
        "默认解析命中正式标定配置",
        resolved is not None and resolved.name == THRESHOLDS.name,
        f"{resolved}",
    )
    check(
        "默认配置版本与标定文件一致",
        active_thresholds().get("source_version") == thresholds.get("source_version"),
        f"{active_thresholds().get('source_version')}",
    )
    for case in CASES:
        print("=" * 78)
        print(f"### {case['name']}：{case['detail']}")
        payloads = {
            "deepfake": read(case["deepfake"]),
            "identity": read(case["identity"]),
            "source": read(case["source"]),
        }
        matrix = build_evidence_matrix(
            payloads["deepfake"],
            payloads["identity"],
            payloads["source"],
            thresholds=thresholds,
        )
        default_matrix = build_evidence_matrix(
            payloads["deepfake"], payloads["identity"], payloads["source"]
        )
        check(
            "默认阈值与显式标定阈值判定一致",
            all(
                default_matrix["modules"][module]["decision"]["state"]
                == matrix["modules"][module]["decision"]["state"]
                for module in ("identity", "source")
            ),
            "",
        )
        for module in ("identity", "source"):
            decision = matrix["modules"][module]["decision"]
            expected = case["expect"][module]
            check(
                f"{module} 三态判定",
                decision["state"] == expected,
                f"实际 {decision['state']} / 期望 {expected}｜{decision['state_label']}",
            )
            primary = decision["primary_signal"]
            value = primary.get("value")
            rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
            print(
                f"       主判据 {primary.get('name')}={rendered}"
                "｜阈值 "
                + " / ".join(f"{key}={value}" for key, value in decision["thresholds"].items())
            )

        report = build_final_report(
            f"e2e_{case['name']}",
            deepfake_result=payloads["deepfake"],
            identity_result=payloads["identity"],
            source_result=payloads["source"],
        )
        text = render_text_report(report)
        start = text.find("五、证据质量矩阵（研判卡）")
        end = text.find("六、并列证据总结")
        check("证据矩阵章节存在", start > 0, f"位置 {start}")
        check("章节顺序 五 → 六", 0 < start < end, f"五={start} 六={end}")
        section = text[start:end] if 0 <= start < end else ""
        missing = [label for label in MATRIX_LABELS if label not in section]
        check("矩阵字段齐全", not missing, f"缺失 {missing}" if missing else "7 项全在")
        check("不含跨模块综合可信度", "综合可信度" not in text, "")
        check(
            "结论标签进入报告",
            all(
                matrix["modules"][module]["decision"]["state_label"] in section
                for module in ("identity", "source")
            ),
            "",
        )
        print("-" * 78)
        print(section.strip())
    print("=" * 78)
    if FAILURES:
        print(f"验证未通过：{len(FAILURES)} 项")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("M1 端到端验证全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
