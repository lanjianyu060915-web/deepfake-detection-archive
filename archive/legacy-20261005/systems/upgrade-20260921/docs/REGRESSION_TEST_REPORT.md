# 三模块正式回归测试报告

测试日期：2026-08-25（Asia/Shanghai）  
测试目录：`D:\刑侦查竞赛\face_source_trace_three_module_v0.2_dev`  
结论：**三模块在当前机器、当前真实模型和真实图库上均可用；联合流程与 UI 成功。**

## 1. 工程与环境检查

| 检查 | 结果 |
|---|---:|
| 后端/协议/超时/故障隔离 pytest | 24 passed |
| 前端 schema/兼容/AppTest unittest | 9 passed |
| `check_environment.py` | 19 PASS，0 WARN，0 FAIL |
| 关键 Python 静态编译 | PASS |
| config JSON 解析 | PASS |
| AIGC EXE、两模型、融合配置、基础权重、三份适配器哈希 | 全部 match |
| InsightFace buffalo_l 五个 ONNX 的冻结大小和 SHA-256 | 全部 match |
| 身份缓存 fingerprint、NPZ SHA、shape、dtype、有限值 | ready，`43×512` |

轻量测试使用 mock worker 验证协议和异常，但不把 mock 计作模型可用证据；下列各节均为真实模型/算法结果。

## 2. AIGC 鉴伪真实测试

调用路径：`D:\python\python.exe -S -B` + 冻结适配器 + 安装版 PyInstaller 归档。实际设备为 CPU。

| 用例 | 状态 | 结果 | score / 错误 | 耗时 |
|---|---|---|---|---:|
| 安装版真实样例 `aigc_real_image.jpg` | success | real | 0.252206 | 9.498 s |
| 安装版伪造样例 `aigc_fake_image.jpg` | success | deepfake | 0.745166 | 8.750 s |
| 无人脸样例 | success | insufficient_evidence | score=null | 6.461 s |
| `.txt` 错误格式 | error（预期） | 无结果 | 明确列出支持扩展名 | 5.411 s |
| 不存在路径 | error（预期） | 无结果 | `query 不存在或不是文件` | 0.002 s |

无人脸没有被强行编造成 real/deepfake；错误用例均生成结构化 JSON。

## 3. 身份搜索真实测试

模型：InsightFace buffalo_l，CPUExecutionProvider，默认库 9 人/45 图，缓存 43 个 embedding。

| 用例 | 状态 | 结果 | 说明 |
|---|---|---|---|
| 库内 `integrated_query_001.jpg` | success | Top1 `person_001`，0.593203 | 与冻结正式结果一致 |
| 外部单人脸图片 | success | Top1 `person_005`，0.086966 | 图片不在默认库；闭集系统返回低相似度相对候选，不伪造拒识 |
| 无人脸 | error（预期） | `未检测到人脸` | 快速、可理解错误 |
| 多人脸合成测试图 | error（预期） | `检测到 2 张人脸，当前要求恰好 1 张` | 无任意选脸 |
| 空身份库 | error（预期） | `没有至少一个可用身份` | 无无限 loading |

另外实测 45 张原始参考图均可解码；当前两张图未形成有效单人脸 embedding，因此缓存为 43 条。系统没有开放集拒识阈值，外部图片测试不作为身份正确率证据。

## 4. 底图搜索真实测试

图库：完整 203 张。所有成功用例均走冻结全局粗排 + SIFT/RANSAC 局部复核算法。

| 用例 | 状态 | Top1 | final_score | 耗时 |
|---|---|---|---:|---:|
| 原始底图 `122.jpg` | success | `122.jpg` | 1.000000 | 163.949 s |
| JPEG quality=25 压缩图 | success | `122.jpg` | 0.986738 | 246.746 s |
| 中心保留 76% 裁剪图 | success | `122.jpg` | 0.909719 | 210.184 s |
| 图库外历史单人脸图 | success | `134.jpg`（相对候选） | 0.443019 | 100.628 s |
| 空底图库 | error（预期） | 无 | `library contains no decodable supported images` | 0.957 s |

压缩与裁剪用例并发运行，耗时受 CPU 竞争影响，只用于 timeout 边界参考。图库外图仍会产生闭集相对排序，不等于真实来源确认。

## 5. 两次完整三模块联合回归

同一个 `RuntimeServiceManager` 连续执行两次，AIGC/身份模型各只加载一次；底图仍为独立 B_env 子进程。

| 查询 | task_id | AIGC | 身份 Top1 | 底图 Top1 | 总耗时 |
|---|---|---|---|---|---:|
| query_001 | `case_20260825_145014_4989` | deepfake，0.378594 | person_001，0.593203 | 122.jpg，0.962237 | 83.437 s |
| query_003 | `case_20260825_145138_6847` | real，0.058860 | person_003，0.687556 | 019.jpg，0.991742 | 51.790 s |

服务最终状态：deepfake requests=2、model_load_count=1；identity requests=2、model_instances=1。两次任务均为 `success`，三份结果和 `unified_result.json` 均真实落盘。

## 6. 最终代码的持久服务与超时修复复测

在统一 deadline、全局禁用硬开关和非有限 timeout 修复后，再次真实启动两 worker 并执行 query_001：

- deepfake：ready，model_load_count=1，任务 `execution_mode=service`，success；
- identity：ready，model_instances=1，任务 `execution_mode=service`，success；
- manager 关闭后所有测试 worker PID 均退出。

单元测试还覆盖：全局持久服务禁用、NaN/Infinity timeout、空模块选择、三个模块 selected+disabled、服务失败后的剩余 deadline 预算。

## 7. 真实故障隔离

task `case_20260825_150230_d0b9` 使用正常查询图、真实 AIGC 和空身份库，仅选择 AIGC+身份：

- AIGC：success；
- 身份：error，0.001 秒内返回空库错误；
- 底图：skipped；
- 总任务：`partial_success`，进程退出码 1。

说明一个模块失败不会阻断另一模块，也不会把 partial 伪报为 success。

## 8. 实际 UI 端到端验收

用应用内浏览器在 `http://127.0.0.1:8767` 完成：

1. 首页真实显示 AIGC 可用、身份 9 人/45 图/43 embedding、底图 203 图/no persistent index、设备 CPU；
2. AIGC/身份常驻服务显示已缓存；
3. 上传 `integrated_query_001.jpg` 后四个按钮均启用；
4. 点击“一键综合分析”，页面持续展示规定进度；
5. task `case_20260825_150652_5edf` 三模块均 success，总耗时 53.256 秒；
6. 证据汇总显示 AIGC 0.3786、person_001 0.5932、122.jpg 0.9622，并明确“不计算综合概率”；
7. 身份 Top-5 参考图与底图可视化通过路径安全校验并实际渲染。

截图位于 `screenshots/three_module_*.png` 和 `test_results/ui/`。

## 9. 正式发布目录复验

正式目录生成后，没有沿用 dev 工作目录的结论，而是在 `D:\刑侦查竞赛\releases\face_source_trace_three_module_v0.2_official` 重新执行：

- 发布自带 `.venv_frontend` 可导入 Streamlit 1.58.0；
- 环境检查 19 PASS，系统状态三模块 available；adapter/模型哈希、43×512 缓存和绝对图库路径全部通过；
- 后端 24 passed、前端 9 tests OK；
- 真实 task `case_20260825_152531_95be`：deepfake 0.378594、person_001 0.593203、122.jpg 0.962237，三模块 `success`，总耗时 47.578 秒；
- AIGC/身份仍为 service 模式且各只加载一个模型实例；底图为真实 B_env subprocess；关闭后 worker PID 全部退出；
- 双击入口等价执行 `start.bat` 后，`127.0.0.1:8767/_stcore/health` 返回 HTTP 200 / `ok`；停止后端口关闭。

这项复验专门确认了开发目录复制到 `releases` 后不会因相对路径变化而失效。

## 10. 未扩大解释的边界

- 本报告证明当前资源和测试集上的工程可运行性，不代表现实案件总体准确率。
- AIGC score 是 FAKE 风险分数；身份/底图分数是各自相似度，不能融合。
- 身份与底图都是封闭候选库排序，没有经过验证的开放集自动拒识阈值。
- 底图没有持久 index，真实长任务依赖 600 秒 timeout。

原始 JSON、日志、成功/失败核心 task 与 UI 截图见 `test_results/`。
