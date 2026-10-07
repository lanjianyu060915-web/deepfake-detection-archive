from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


DISCLAIMER = (
    "本结果由自动化模型生成，仅用于深度伪造风险筛查和辅助研判，不应单独作为司法鉴定或案件定性依据。"
    "应结合原始文件、传播来源、设备信息及其他证据进行综合判断。"
)


FRAME_COLUMNS = [
    "candidate_index",
    "source_frame_index",
    "timestamp_seconds",
    "status",
    "invalid_reason",
    "spatial_score",
    "frequency_score",
    "fusion_score",
    "frame_threshold",
    "predicted_label",
    "risk_level",
    "inference_ms",
]


class RuntimeReportService:
    """将一次运行结果稳定导出为JSON、CSV和无外部依赖的HTML。"""

    def write_reports(self, result: dict[str, Any], run_dir: Path) -> dict[str, str]:
        run_dir = run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "result_json": "result.json",
            "frame_predictions_csv": "frame_predictions.csv",
            "report_html": "report.html",
            "runtime_log": "runtime.log",
        }
        result["report_paths"] = dict(paths)
        result["disclaimer"] = DISCLAIMER
        self._write_csv(run_dir / paths["frame_predictions_csv"], result.get("frame_results", []))
        (run_dir / paths["report_html"]).write_text(self._render_html(result), encoding="utf-8")
        (run_dir / paths["result_json"]).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return paths

    @staticmethod
    def _write_csv(path: Path, frame_results: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FRAME_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in frame_results:
                writer.writerow({column: row.get(column, "") for column in FRAME_COLUMNS})

    @staticmethod
    def _format_score(value: Any) -> str:
        if value is None or value == "":
            return "不可用"
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return html.escape(str(value))

    def _render_html(self, result: dict[str, Any]) -> str:
        escape = lambda value: html.escape(str(value if value is not None else ""))
        metadata = result.get("media_metadata", {})
        warnings = result.get("warnings", [])
        keyframes = result.get("keyframe_paths", [])
        frame_rows = []
        for row in result.get("frame_results", []):
            frame_rows.append(
                "<tr>"
                f"<td>{escape(row.get('candidate_index', ''))}</td>"
                f"<td>{escape(row.get('source_frame_index', ''))}</td>"
                f"<td>{self._format_score(row.get('timestamp_seconds'))}</td>"
                f"<td>{escape(row.get('status', ''))}</td>"
                f"<td>{self._format_score(row.get('spatial_score'))}</td>"
                f"<td>{self._format_score(row.get('frequency_score'))}</td>"
                f"<td>{self._format_score(row.get('fusion_score'))}</td>"
                f"<td>{escape(row.get('predicted_label', ''))}</td>"
                f"<td>{escape(row.get('invalid_reason', ''))}</td>"
                "</tr>"
            )
        keyframe_html = "".join(
            f'<figure><img src="{escape(path)}" alt="关键帧"><figcaption>{escape(Path(path).name)}</figcaption></figure>'
            for path in keyframes
        ) or "<p>无可展示关键帧。</p>"
        warning_html = "".join(f"<li>{escape(item)}</li>" for item in warnings) or "<li>无</li>"
        model_paths = result.get("model_paths", {})
        spatial_summary = result.get("spatial_score_summary", {})
        frequency_summary = result.get("frequency_score_summary", {})
        style = """
body{font-family:'Microsoft YaHei','Segoe UI',sans-serif;margin:0;background:#f4f6f8;color:#17202a}
main{max-width:1100px;margin:24px auto;background:white;padding:28px;border-radius:10px;box-shadow:0 2px 12px #ccd3da}
h1,h2{color:#123b5d}.verdict{padding:16px;border-left:5px solid #2471a3;background:#edf6fc}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}.card{background:#f7f9fa;padding:12px;border-radius:6px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{border:1px solid #d5d8dc;padding:7px;text-align:left}th{background:#eaf2f8}
.keyframes{display:flex;gap:12px;flex-wrap:wrap}.keyframes img{max-width:300px;max-height:220px;border:1px solid #ccd3da}
.notice{margin-top:24px;padding:14px;background:#fff4e5;border:1px solid #f5c16c}.mono{word-break:break-all;font-family:Consolas,monospace}
"""
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>深度伪造风险辅助研判报告</title><style>{style}</style></head><body><main>
<h1>深度伪造图片/视频检测辅助研判报告</h1>
<section class="verdict"><strong>最终状态：</strong>{escape(result.get('status'))}<br>
<strong>最终判断：</strong>{escape(result.get('final_label'))}<br>
<strong>伪造风险评分：</strong>{self._format_score(result.get('final_fake_score'))}<br>
<strong>判定阈值：</strong>{self._format_score(result.get('decision_threshold'))}</section>
<h2>文件基本信息</h2><div class="grid">
<div class="card"><strong>文件名</strong><br>{escape(result.get('input_file_name'))}</div>
<div class="card"><strong>媒体类型</strong><br>{escape(result.get('media_type'))}</div>
<div class="card"><strong>文件大小</strong><br>{escape(result.get('file_size'))} bytes</div>
<div class="card"><strong>检测时间</strong><br>{escape(result.get('created_at'))}</div>
<div class="card"><strong>实际设备</strong><br>{escape(result.get('actual_device'))}</div>
<div class="card"><strong>有效/无效帧</strong><br>{escape(result.get('valid_frame_count'))} / {escape(result.get('invalid_frame_count'))}</div>
</div><p><strong>SHA-256：</strong><span class="mono">{escape(result.get('input_sha256'))}</span></p>
<p><strong>媒体元数据：</strong>{escape(json.dumps(metadata, ensure_ascii=False))}</p>
<h2>模型与融合配置</h2>
<p>空间模型：<span class="mono">{escape(model_paths.get('spatial'))}</span><br>
频域模型：<span class="mono">{escape(model_paths.get('frequency'))}</span><br>
融合配置SHA-256：<span class="mono">{escape(result.get('fusion_config_hash'))}</span></p>
<p>空间权重={self._format_score(result.get('spatial_weight'))}，频域权重={self._format_score(result.get('frequency_weight'))}，
帧阈值={self._format_score(result.get('frame_threshold'))}，视频阈值={self._format_score(result.get('video_threshold'))}，
视频聚合={escape(result.get('video_aggregation'))}</p>
<h2>评分汇总</h2><p>空间域中位数：{self._format_score(spatial_summary.get('median'))}；
频域中位数：{self._format_score(frequency_summary.get('median'))}；
最终融合评分：{self._format_score(result.get('final_fake_score'))}</p>
<h2>关键帧</h2><div class="keyframes">{keyframe_html}</div>
<h2>逐帧结果</h2><table><thead><tr><th>候选</th><th>源帧</th><th>时间(s)</th><th>状态</th><th>空间</th><th>频域</th><th>融合</th><th>判断</th><th>说明</th></tr></thead>
<tbody>{''.join(frame_rows)}</tbody></table>
<h2>警告信息</h2><ul>{warning_html}</ul>
<h2>适用边界</h2><p>模型结果是风险评分，不是系统总体准确率。无人脸、有效帧不足、严重压缩、遮挡、极端角度或未知编码可能降低可靠性。</p>
<div class="notice">{escape(DISCLAIMER)}</div>
</main></body></html>"""
