from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = ROOT / "test_results" / "regression_matrix"
DOC_DIR = ROOT / "docs" / "upgrade_v0.4.0"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(
    test_id: str,
    title: str,
    evidence_level: str,
    observed: str,
    evidence: list[str],
    commands: list[str],
    *,
    status: str = "passed",
    limitation: str | None = None,
) -> dict:
    item = {
        "id": test_id,
        "title": title,
        "status": status,
        "evidence_level": evidence_level,
        "observed": observed,
        "evidence": evidence,
        "commands": commands,
    }
    if limitation:
        item["limitation"] = limitation
    return item


def main() -> int:
    verification = _read(MATRIX_DIR / "verification.json")
    normal = verification["normal_case"]
    reports = verification["reports"]
    tests = [
        _record(
            "T01",
            "正常图片，三个模块全部运行",
            "三套正式运行环境、同一基准输入、真实模型/真实库",
            (
                "AIGC、身份、底图均 status=success；输入 SHA-256 为 "
                f"{normal['query_sha256']}；升级前后所有关键数值最大绝对差 "
                f"{verification['maximum_observed_absolute_error']:.1f}。身份 Top-5 "
                "person_001/person_004/person_006/person_009/person_003，底图 Top-5 "
                "051/017/020/173/144，顺序与基准一致。"
            ),
            [
                "test_results/aigc_visual_evidence/case_20260831_114829_c45e_final/detection_result.json",
                "test_results/aigc_visual_evidence/case_20260831_114829_c45e_final/regression_comparison.json",
                "test_results/regression_matrix/normal_identity/identity_result.json",
                "test_results/regression_matrix/normal_source/source_result.json",
                "test_results/regression_matrix/verification.json",
            ],
            [
                "packaged portable Python -> modules/deepfake_detection/run_deepfake_detection.py",
                "A_env/python.exe -> modules/identity_search/run_identity_search.py --offline",
                "B_env/python.exe -> modules/source_search/run_source_search.py",
            ],
        ),
        _record(
            "T02",
            "未检测到人脸的图片",
            "正式 AIGC portable Python + 真实人脸检测",
            "status=success、label=no_face、score=null、有效人脸数=0；生成明确无人脸状态图，未伪造模型分数。",
            [
                "test_results/aigc_visual_evidence/no_face/detection_result.json",
                "test_results/regression_matrix/aigc_offline_evidence.junit.xml",
            ],
            [
                "A_env/python.exe -m pytest tests/test_deepfake_visual_evidence.py tests/test_aigc_model_resource_contract.py",
            ],
        ),
        _record(
            "T03",
            "检测到多张人脸的图片",
            "A_env + buffalo_l 真实检测",
            "真实检测到 2 张人脸；模块返回结构化 error、空候选及两个真实框/关键点，不误选身份。",
            [
                "test_results/regression_matrix/multi_face_identity/identity_result.json",
                "test_results/regression_matrix/multi_face_identity/logs/progress_events.jsonl",
            ],
            [
                "A_env/python.exe modules/identity_search/run_identity_search.py --query runtime/validation/identity_multi_face.jpg --offline",
            ],
        ),
        _record(
            "T04",
            "身份库为空",
            "A_env + buffalo_l 真实 CLI 边界",
            "空库返回结构化 error；library_identity_count=0、valid_embedding_count=0、candidates=[]。",
            [
                "test_results/regression_matrix/empty_identity/identity_result.json",
                "test_results/regression_matrix/empty_identity/logs/progress_events.jsonl",
            ],
            [
                "A_env/python.exe modules/identity_search/run_identity_search.py --library <empty-directory> --offline",
            ],
        ),
        _record(
            "T05",
            "身份库部分图片损坏",
            "A_env + buffalo_l 真实模型、真实损坏 JPEG",
            "1 张有效参考图完成检索，1 张损坏图以 ImageDecodeError 跳过；任务仍 success，候选和证据正常生成。",
            [
                "test_results/regression_matrix/corrupt_identity/identity_result.json",
                "test_results/regression_matrix/corrupt_identity/logs/progress_events.jsonl",
                "test_results/regression_matrix/identity_evidence.junit.xml",
            ],
            [
                "A_env/python.exe modules/identity_search/run_identity_search.py --library test_results/regression_matrix/corrupt_identity/身份库 --offline",
            ],
        ),
        _record(
            "T06",
            "底图库为空",
            "B_env 真实 CLI 边界",
            "返回结构化 error：library contains no decodable supported images；候选为空，未产生伪排名。",
            [
                "test_results/regression_matrix/empty_source/source_result.json",
                "test_results/regression_matrix/empty_source/logs/progress_events.jsonl",
            ],
            [
                "B_env/python.exe modules/source_search/run_source_search.py --library <empty-directory>",
            ],
        ),
        _record(
            "T07",
            "底图库部分图片损坏",
            "B_env 真实全局/SIFT/RANSAC，真实损坏 JPEG",
            "库文件 3、成功解码 2、跳过 1；任务 success，Top-2、融合分和 11 个真实证据文件正常生成。",
            [
                "test_results/regression_matrix/corrupt_source/source_result.json",
                "test_results/regression_matrix/corrupt_source/logs/progress_events.jsonl",
                "test_results/regression_matrix/source_edges.junit.xml",
            ],
            [
                "B_env/python.exe modules/source_search/run_source_search.py --library 'test_results/regression_matrix/corrupt_source/中文 底图库'",
            ],
        ),
        _record(
            "T08",
            "SIFT 匹配点不足",
            "OpenCV 真实 SIFT/描述子边界",
            "无纹理图真实产生 0 个描述子/匹配；RANSAC 未启动，按冻结不足匹配公式返回 0，流程不崩溃。",
            ["test_results/regression_matrix/source_edges.junit.xml"],
            ["B_env/python.exe -m pytest tests/test_v040_regression_edges.py"],
        ),
        _record(
            "T09",
            "RANSAC 无法稳定估计",
            "真实 SIFT/Lowe + 对 RANSAC 返回值的确定性故障注入",
            "真实 SIFT/Lowe 匹配数 >=4；仅将 findHomography 注入为 (None,None)，结果记录 attempted=true/applied=false、内点=0，未崩溃。",
            ["test_results/regression_matrix/source_edges.junit.xml"],
            ["B_env/python.exe -m pytest tests/test_v040_regression_edges.py"],
            limitation="该稀有失败分支通过可控故障注入稳定覆盖；SIFT 和 Lowe 仍由 OpenCV 对真实数组计算。",
        ),
        _record(
            "T10",
            "解释图生成失败但原检测成功",
            "解释层确定性故障注入与结果隔离",
            "AIGC 注入解释生成异常后，原 status/label/score/threshold 保持；底图注入关键点证据异常后，原候选与 final_score 保持。",
            [
                "test_results/regression_matrix/aigc_offline_evidence.junit.xml",
                "test_results/regression_matrix/source_edges.junit.xml",
            ],
            [
                "A_env/python.exe -m pytest tests/test_deepfake_visual_evidence.py::test_visual_explanation_failure_does_not_change_authoritative_result",
                "B_env/python.exe -m pytest tests/test_source_visual_evidence.py::test_optional_evidence_failure_does_not_change_successful_search",
            ],
        ),
        _record(
            "T11",
            "用户切换页面不重复执行模型",
            "Streamlit AppTest 五页真实 rerun + 结果哈希/mtime/任务目录快照",
            "依次切换五页，异常=0；六个权威结果文件 SHA-256、大小、mtime 全部不变，任务根目录未新增任务。",
            ["test_results/regression_matrix/frontend_state.json"],
            ["D:/python/.venv/Scripts/python.exe frontend/tests/verify_evidence_state_regression.py"],
        ),
        _record(
            "T12",
            "刷新页面后已有任务可重新加载",
            "全新 Streamlit AppTest 会话（无旧 session_state）",
            "新会话从磁盘重新加载 case_20260831_114829_c45e；五页入口完整，异常=0、错误=0，权威结果文件未改动。",
            ["test_results/regression_matrix/frontend_state.json"],
            ["D:/python/.venv/Scripts/python.exe frontend/tests/verify_evidence_state_regression.py"],
        ),
        _record(
            "T13",
            "TXT 报告仍可导出",
            "当前报告生成器单元/集成测试 + 已完成任务实际文件",
            f"当前报告测试通过；实际 TXT 为 {reports['txt']['size_bytes']} 字节，SHA-256={reports['txt']['sha256']}。",
            [
                reports["txt"]["path"],
                "test_results/regression_matrix/frontend_reports.log.json",
            ],
            ["D:/python/.venv/Scripts/python.exe -m unittest frontend.tests.test_explainability_reporting"],
        ),
        _record(
            "T14",
            "HTML 报告仍可导出",
            "当前报告生成器单元/集成测试 + 已完成任务实际文件",
            f"当前报告测试通过；实际 HTML 为 {reports['html']['size_bytes']} 字节，SHA-256={reports['html']['sha256']}。",
            [
                reports["html"]["path"],
                "test_results/regression_matrix/frontend_reports.log.json",
            ],
            ["D:/python/.venv/Scripts/python.exe -m unittest frontend.tests.test_explainability_reporting"],
        ),
        _record(
            "T15",
            "新增可视化证据随任务保存",
            "三模块真实输出逐文件 SHA-256/大小/路径校验",
            "基准任务 AIGC 12、身份 12、底图 17，共 41 个 artifact；均为任务相对 POSIX 路径，实际文件存在且哈希和大小逐项匹配。",
            [
                "test_results/regression_matrix/verification.json",
                "test_results/regression_matrix/aigc_offline_evidence.junit.xml",
                "test_results/regression_matrix/identity_evidence.junit.xml",
                "test_results/regression_matrix/source_edges.junit.xml",
                "test_results/regression_matrix/orchestrator_launcher.junit.xml",
            ],
            ["D:/python/.venv/Scripts/python.exe tests/verify_v040_regression_matrix.py"],
        ),
        _record(
            "T16",
            "中文目录和中文文件名",
            "A_env/B_env/portable Python 真实读写",
            "身份库含 person_测试/有效参考图.jpg/损坏参考图.jpg；底图库为“中文 底图库”、候选甲/乙；AIGC 测试输出为“中文 任务目录”，均完成 JSON 和证据写入。",
            [
                "test_results/regression_matrix/corrupt_identity/identity_result.json",
                "test_results/regression_matrix/corrupt_source/source_result.json",
                "test_results/regression_matrix/aigc_offline_evidence.junit.xml",
            ],
            [
                "A_env/B_env 真实 CLI 中文路径回归",
                "A_env/python.exe -m pytest tests/test_deepfake_visual_evidence.py",
            ],
        ),
        _record(
            "T17",
            "安装包环境离线运行合同",
            "正式资源运行 + 空 Torch Hub 缓存 + 打包静态合同",
            "AIGC portable Python 在隔离空 TORCH_HOME 下真实推理；身份以 --offline 使用随包 buffalo_l；正式配置 real_modules_default=true、mock=false、外部语言模型关闭，静态合同通过。",
            [
                "test_results/regression_matrix/aigc_offline_evidence.junit.xml",
                "test_results/regression_matrix/normal_identity/identity_result.json",
                "test_results/regression_matrix/packaging_static.log.json",
                "packaging_v0.3/static_validation.json",
            ],
            [
                "A_env/python.exe -m pytest tests/test_aigc_model_resource_contract.py",
                "pwsh -NoProfile -File packaging_v0.3/scripts/Test-PackagingStatic.ps1",
            ],
            limitation="本项验证安装资源与离线合同；最终 Setup 安装到非开发目录后的断网整机复跑由正式安装验收阶段记录。",
        ),
        _record(
            "T18",
            "原有启动器和快捷方式合同",
            "启动器单元测试 + Inno Setup 静态合同",
            "启动器端口选择/冲突回退 3 项通过；安装脚本含勾选的 desktopicon 和 {autodesktop} 快捷方式，v0.4.0 AppId 隔离及卸载停止钩子静态检查通过。",
            [
                "test_results/regression_matrix/orchestrator_launcher.junit.xml",
                "test_results/regression_matrix/packaging_static.log.json",
                "packaging_v0.3/static_validation.json",
                "packaging_v0.3/installer/换脸寻源系统_v0.3.iss",
            ],
            [
                "A_env/python.exe -m pytest tests/test_packaging_launcher.py",
                "pwsh -NoProfile -File packaging_v0.3/scripts/Test-PackagingStatic.ps1",
            ],
            limitation="本项是构建前静态/单元合同；桌面快捷方式的实际双击由正式安装验收阶段记录。",
        ),
    ]

    allowed_status = {"passed", "failed", "not_applicable"}
    assert len(tests) == 18
    assert all(item["status"] in allowed_status for item in tests)
    counts = {
        status: sum(item["status"] == status for item in tests)
        for status in ("passed", "failed", "not_applicable")
    }
    payload = {
        "schema_version": "1.0",
        "product_version": "0.4.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "baseline_task_id": verification["baseline_task_id"],
        "status_counts": counts,
        "automated_testcase_count": sum(
            item["testcases"] for item in verification["junit"].values()
        )
        + verification["reports"]["current_builder_tests"],
        "maximum_observed_absolute_error": verification[
            "maximum_observed_absolute_error"
        ],
        "tests": tests,
        "scope_note": (
            "本矩阵不执行安装/卸载；T17/T18覆盖安装资源、离线、启动器和快捷方式的"
            "构建前合同，正式安装后的动态验收由主交付流程另行记录。"
        ),
    }
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DOC_DIR / "回归测试矩阵.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# v0.4.0 回归测试矩阵",
        "",
        f"生成时间：`{payload['generated_at']}`  ",
        f"重点基准：`{payload['baseline_task_id']}`  ",
        (
            f"结果：通过 {counts['passed']} / 18，失败 {counts['failed']}，"
            f"不适用 {counts['not_applicable']}；本轮可复现自动化用例 "
            f"{payload['automated_testcase_count']} 项全部通过。"
        ),
        "",
        "基准任务三模块在相同输入、模型、配置和数据库下的最大数值绝对差为 "
        f"`{payload['maximum_observed_absolute_error']}`（上限 `1e-6`）。",
        "",
        "## 18 项结果",
        "",
        "| ID | 测试项 | 状态 | 证据级别 | 实际观察 |",
        "|---|---|---|---|---|",
    ]
    for item in tests:
        observed = item["observed"].replace("|", "\\|")
        level = item["evidence_level"].replace("|", "\\|")
        lines.append(
            f"| {item['id']} | {item['title']} | {item['status']} | {level} | {observed} |"
        )
    lines.extend(
        [
            "",
            "## 证据索引与命令",
            "",
        ]
    )
    for item in tests:
        lines.extend([f"### {item['id']} {item['title']}", ""])
        lines.append(f"状态：`{item['status']}`；证据级别：{item['evidence_level']}。")
        lines.extend(["", "证据：", ""])
        lines.extend(f"- `{value}`" for value in item["evidence"])
        lines.extend(["", "命令：", ""])
        lines.extend(f"- `{value}`" for value in item["commands"])
        if item.get("limitation"):
            lines.extend(["", f"限制：{item['limitation']}"])
        lines.append("")
    lines.extend(
        [
            "## 证据解释",
            "",
            "- “真实模型/真实库”表示调用了正式 AIGC portable Python、A_env buffalo_l 或 B_env 底图算法，不是 Mock。",
            "- “故障注入”仅用于稳定触达解释失败和 RANSAC 失败分支；权威检测分数、排序、权重和阈值没有修改。",
            "- 五页切换测试比较了结果文件 SHA-256、大小和 mtime，并检查任务目录集合，证明 UI rerun 没有再次提交模型任务。",
            "- TXT/HTML 同时由当前报告生成器自动化测试和已完成任务实际文件验证。",
            "- T17/T18 是本代理被授权范围内的构建前合同验证；不冒充最终安装后双击快捷方式或断网整机验收。",
            "",
            "机器可读版本：`docs/upgrade_v0.4.0/回归测试矩阵.json`。聚合校验明细：`test_results/regression_matrix/verification.json`。",
            "",
        ]
    )
    markdown_path = DOC_DIR / "回归测试矩阵.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), **counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
