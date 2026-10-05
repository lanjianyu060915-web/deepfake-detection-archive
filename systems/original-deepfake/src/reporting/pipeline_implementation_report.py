from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _result_line(name: str, result: dict[str, Any]) -> str:
    score = "不可用" if result["final_fake_score"] is None else f"{result['final_fake_score']:.6f}"
    return (
        f"- {name}：状态={result['status']}，设备={result['actual_device']}，"
        f"有效/无效帧={result['valid_frame_count']}/{result['invalid_frame_count']}，"
        f"评分={score}，结论={result['final_label']}，耗时={result['total_elapsed_seconds']:.3f}s，"
        f"报告=`{result['report_html']}`"
    )


def write_pipeline_implementation_report(
    acceptance_summary_path: Path,
    test_summary_path: Path,
    output_path: Path,
) -> Path:
    acceptance = json.loads(acceptance_summary_path.read_text(encoding="utf-8"))
    tests = json.loads(test_summary_path.read_text(encoding="utf-8"))
    content = f"""# 正式端到端检测管线实现与验收报告

## 1. 模块结构

- `src/pipeline/deepfake_detection_pipeline.py`：统一图片、视频和自动媒体分派管线。
- `src/reporting/runtime_report_service.py`：生成result.json、frame_predictions.csv和report.html。
- `scripts/detect_media.py`：普通Python命令行入口。
- `config/runtime_pipeline.json`：目标有效帧、最低有效帧、候选数量、关键帧和清理策略。
- `tests/test_runtime_pipeline.py`：输入、推理调度、报告和安全测试。

## 2. 输入处理

输入使用`pathlib`解析，检查文件是否存在、是否为普通文件、扩展名是否受支持及OpenCV实际解码结果；计算SHA-256、文件大小和媒体元数据。运行目录只使用时间戳与随机UUID，不信任用户文件名，也拒绝把输出目录设置到项目目录之外。

## 3. 图片流程

JPG/JPEG/PNG/WEBP经中文路径兼容的`numpy.fromfile + cv2.imdecode`读取；使用正式OpenCV Haar规则检测最大人脸、边缘补边重试、25%扩框并缩放到224×224。确认有人脸后调用正式FusionDetector，使用`fusion_config.json`中的帧阈值，保存人脸预览、结构化结果和报告。

## 4. 视频流程

MP4/AVI/MOV/MKV读取时长、FPS、总帧数、分辨率和FOURCC。先在全时长均匀选取5个主候选点；不足5张有效人脸时，再从最多20个均匀候选点补充。只将正在推理的候选帧写入唯一临时目录，不保存全部视频帧，结束后默认清理。

## 5. 人脸失败处理

图片无人脸时返回`INSUFFICIENT_EVIDENCE`，不调用模型、不生成伪造评分。视频0张有效人脸时返回`INSUFFICIENT_EVIDENCE`；1至2张有效人脸低于配置的最低3张时返回`LOW_CONFIDENCE`，保留暂定评分但最终标签为“证据不足”。所有失败候选均保留时间戳和中文原因。

## 6. 模型调用与设备

管线直接读取`models/fusion_formal/fusion_config.json`，不重新训练、不调整0.62/0.38权重或阈值。FusionDetector在管线实例内通过锁保护的延迟加载只初始化一次，之后所有帧复用。支持auto、cpu、cuda；auto在本机选择了CUDA，GPU为`{acceptance['gpu_name']}`。

## 7. 视频聚合

逐帧调用正式融合检测器后，对有效帧fusion_score按正式配置取中位数，再使用视频阈值0.458342549364判断。按融合评分降序保存最高3张有效帧作为关键帧。

## 8. 输出结构与报告

每次运行生成`outputs/runtime/<run_id>/`，包含`result.json`、`frame_predictions.csv`、`report.html`、`runtime.log`和`keyframes/`。HTML包含文件哈希、元数据、模型与配置、最终评分、阈值、分支汇总、逐帧结果、关键帧、警告、适用边界及司法使用免责声明。

## 9. 真实素材验收结果

{_result_line('真实视频', acceptance['real_video'])}
{_result_line('伪造视频', acceptance['fake_video'])}
{_result_line('伪造图片', acceptance['fake_image'])}
{_result_line('CPU真实图片', acceptance['cpu_real_image'])}
{_result_line('无人脸图片', acceptance['no_face_image'])}

- auto管线连续处理4个媒体，模型加载次数：{acceptance['auto_pipeline_model_load_count']}。
- CPU管线模型加载次数：{acceptance['cpu_pipeline_model_load_count']}。
- 受保护正式文件变化数量：{len(acceptance['protected_files_changed'])}。

## 10. 测试结果

- 新增管线测试：{tests['runtime_tests']}。
- 项目全量pytest：{tests['full_pytest']}。
- CUDA端到端：{tests['cuda_acceptance']}。
- CPU端到端：{tests['cpu_acceptance']}。
- FusionDetector重复加载：{tests['fusion_detector_reload']}。
- 关键原文件SHA-256：{tests['hash_integrity']}。

## 11. 已知局限

- 人脸检测使用轻量OpenCV Haar方案，极端角度、遮挡、低照度和小尺寸人脸可能检出失败。
- 当前每个候选帧只使用最大人脸，不执行多人脸分别研判。
- 视频采用稀疏均匀抽帧，可能遗漏持续时间很短的局部伪造片段。
- 输出是风险筛查结果，不是司法鉴定意见，也不能把单样本评分解释为系统总体准确率。

## 12. Streamlit接入说明

下一阶段创建并缓存一个管线实例：

```python
from src.pipeline import DeepfakeDetectionPipeline

pipeline = DeepfakeDetectionPipeline(project_root, device="auto")
result = pipeline.detect_media(uploaded_file_path)
```

同一个缓存实例应跨请求复用，以避免重复加载模型。界面从返回字典读取状态、评分、关键帧和报告路径；对于`INSUFFICIENT_EVIDENCE`或`LOW_CONFIDENCE`必须突出展示警告，不得包装成确定性结论。
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
