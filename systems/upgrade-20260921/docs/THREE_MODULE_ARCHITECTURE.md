# v0.3 三模块系统架构

## 产品定位

产品名称：**AI换脸涉案图像智能鉴伪与原始素材溯源系统**  
版本：**0.3.0**  
基线：`face_source_trace_three_module_v0.2_dev`

系统处理单张图片，独立执行深伪风险甄别、身份候选检索和底图候选检索。编排层只并列汇总三类结果，不改变核心算法分数，也不生成跨模块融合分值。

## 正式安装运行结构

```text
换脸寻源系统_v0.3.exe
  ├─ 启动前检查
  │    ├─ formal/schema/mock 策略
  │    ├─ 三个真实模块和模型资源
  │    ├─ 默认身份库、默认底图库
  │    ├─ resources_manifest.sha256.json
  │    └─ LocalAppData 可写性
  ├─ 自动选择 127.0.0.1 本地端口并打开浏览器
  └─ Streamlit UI
       └─ orchestrator
            ├─ AIGC：受控便携 Python 常驻服务 / 真实子进程回退
            ├─ Identity：identity_worker.exe 常驻服务 / 真实子进程回退
            ├─ Source：source_search.exe 单次隔离进程
            └─ reporting：结构化解释、TXT/HTML/JSON 确定性模板
```

依赖环境继续隔离，避免 PyTorch、InsightFace/ONNX Runtime、不同 NumPy/OpenCV 版本相互破坏。AIGC 与身份进程可以复用模型；底图算法实时扫描完整候选库。

## 用户数据与只读资源

正式配置用占位路径解析：

```text
{APP_ROOT}  = 当前安装目录
{DATA_ROOT} = %LOCALAPPDATA%\FaceSourceTrace
```

安装目录保存程序、经批准的模型、运行库和默认图库，并按只读资源使用。用户可写数据位于：

```text
%LOCALAPPDATA%\FaceSourceTrace\
  ├─ tasks\case_*\
  ├─ inbox\
  ├─ cache\identity\
  ├─ persistent_services\
  └─ logs\launcher_v0.3.log
```

源码开发配置和正式安装配置分离；正式包不得依赖开发者用户名、开发目录或固定盘符。

## 任务生命周期

```text
上传校验
  → 创建 task_dir 与查询副本
  → 写入总体/模块开始事件
  → 按选择执行三个模块
  → 每个模块独立保存结果或错误
  → 生成 unified_result.json
  → 从原始结果生成 final_report 对象
  → 原子写入 JSON/TXT/HTML 报告
  → 写入最终任务状态
```

未选择模块为 `skipped`。一个模块失败不会中止其他模块；至少一个成功且至少一个失败时为 `partial_success`。报告层支持成功、部分成功、失败和跳过模块，不用缺失值补造结论。

## 统一进度事件

权威事件流：`logs/progress_events.jsonl`  
最新事件：`logs/progress_latest.json`  
v0.2 兼容快照：`logs/progress.json`

每条 v0.3 事件包含：

```text
event_id
timestamp
task_id
module
stage
status
message
progress_current
progress_total
progress_percent
elapsed_seconds
details
```

`module` 为 `overall`、`deepfake`、`identity` 或 `source`；`status` 为 `pending`、`running`、`success`、`warning`、`error` 或 `skipped`。精确百分比只来自可度量阶段；无法度量时为 `null`。事件按阶段切换、约 0.5 秒或约 5% 变化节流，最新 JSON 通过临时文件替换发布。前端忽略被中断的最后一条不完整 JSONL，不让显示读取错误影响算法。

## 模块数据契约

三个结果均使用 `schema_version: 3.0`，保留 v0.2 核心字段，并提供：

```text
method
metrics
explanation
warnings
limitations
runtime
provenance
```

### AIGC

核心结果包括 `label`、`is_deepfake`、`score`、两个分支分数、融合权重、阈值、输入与模型哈希、设备、有效帧或人脸计数和原生报告路径。证据不足时 `score` 可以为 `null`。

### 身份搜索

核心结果包括查询人脸数、选择框、检测分、512 维特征、库人数/图片数/有效 embedding 数、人员均值聚合方法，以及候选的逐参考图相似度、最佳参考图、人员均值和前两名差值。查询图严格要求恰好一张人脸。

### 底图搜索

核心结果包括查询人脸屏蔽信息和屏蔽图、图库读取/跳过数、粗排数量、全局/局部/最终分、关键点数、Lowe 有效匹配数、RANSAC 内点数/内点率、局部分量、权重、前两名差值和匹配可视化。

## 报告单一数据源

`reporting.report_builder` 先建立一个 `final_report.json` 对象，再由该对象生成 TXT 和 HTML。报告包含原始模块结果、解释、任务元数据和来源信息。报告生成器为 `deterministic_template_v3`，正式配置明确关闭外部语言模型。

普通视图只显示文件名和必要指标；完整路径限于技术详情、原始 JSON 和日志。报告文件名经过任务号安全化，写入采用同目录临时文件加原子替换。

## 路径与故障边界

- 上传文件限制扩展名、大小和单文件模式。
- UI 只读取任务目录、已批准图库根和样例根内资源，拒绝越界路径。
- 子进程分别记录命令、stdout、stderr、超时、退出码和执行方式。
- 持久服务失败只回退到同一真实 runner；正式模式不会回退到模拟结果。
- 启动器只监听 `127.0.0.1`，验证已有进程的安装目录和可执行文件后才复用。
- 安装器卸载前执行停止脚本，停止脚本验证 PID 对应可执行文件位于当前安装目录。

## 发布状态

正式构建链的静态检查已记录 `fixed_drive_path_findings=0`、`formal_mock_enabled=false`，但安装、真实业务运行、重启、卸载和数据保留的最终结论仍为：`【待最终验收回填】`。
