# 三模块回归测试计划

测试原则：结果必须来自真实冻结模型/真实数据库或明确的错误/空库夹具；不把文件名、人工观感或模型输出当作数据集真值，不伪造准确率。

## AIGC 鉴伪

| 用例 | 输入 | 预期契约 |
| --- | --- | --- |
| 已有 REAL 演示图 | 安装包 `samples/demo/real_image.jpg` | 真实模型完成，保留原生 label/score/threshold；不以文件名代替模型结论 |
| 已有 FAKE 演示图 | 安装包 `samples/demo/fake_image.jpg` | 同上 |
| 无人脸 | 安装包 `samples/demo/no_face_image.jpg` | `INSUFFICIENT_EVIDENCE`，is_deepfake=null，不折叠为 REAL |
| 错误格式 | 文本/损坏图片夹具 | 受控 error JSON，UI 可理解提示 |
| 无效路径 | 不存在路径 | 受控 error JSON，非零 CLI 返回码 |
| 模型复用 | 同一 UI 生命周期连续两次 | 持久 worker 可用时模型加载计数不增加；不可用时明确显示 fallback |

## 人脸身份搜索

| 用例 | 输入 | 预期契约 |
| --- | --- | --- |
| 库内人物 | 9 人库内参考图/官方成功 case | 成功、Top-5、Rank-1 可核对 |
| 库外人物 | AIGC demo 人脸图 | 成功返回候选或明确拒识边界；不得声称候选即真实身份 |
| 无人脸 | no_face_image.jpg | 受控 no-face error |
| 多人脸 | 复用已确认多人脸素材 | 受控 multiple-faces error |
| 空数据库 | 独立空目录 | 立即返回空库错误，不长时间 loading |
| 缓存命中 | 当前默认库 | 43×512 NPZ fingerprint 命中，不重建 |

## 原始底图搜索

| 用例 | 输入 | 预期契约 |
| --- | --- | --- |
| 原始底图 | 203 图库内原图 | Rank-1 为自身，final score 可核对 |
| 压缩图 | 从库内图派生的 JPEG 压缩副本 | 成功返回候选；记录派生方式 |
| 裁剪图 | 从库内图派生的裁剪副本 | 成功返回候选；记录派生方式 |
| 库外图 | AIGC demo 图 | 成功返回最相似候选，但不得声称存在真实同源 |
| 空数据库 | 独立空目录 | 立即返回空库错误，不长时间 loading |
| timeout | 受控短 timeout/慢 runner | 受控 timeout，其他分支继续 |

## 综合分析

至少选择两个真实单图执行：上传 → AIGC → 身份 → 底图 → 证据汇总。检查：

- 三个模块均生成独立 JSON 和日志；
- 一个分支失败时另两个继续；
- 总状态 success/partial_success/error 正确；
- 页面只汇总 AIGC score、身份 Top-1 相似度、底图 Top-1 相似度，不生成跨任务加权概率；
- 任务进度、timeout 和错误提示可见；
- task 资源路径仍受 containment policy 限制。

## UI / 启动 / 环境

- `start.bat` 先执行环境、模型、数据库检查，再启动 Streamlit；
- UI health 返回 HTTP 200；
- 首页标题、四按钮、三库/设备状态正确；
- 缺库时对应按钮禁用；
- Task Result 可读取新三模块 task，并兼容旧双模块 task；
- 关闭系统按钮只结束本产品 UI，不触碰已安装 AIGC 服务。

实际执行结果写入 `docs/REGRESSION_TEST_REPORT.md` 和 `test_results/`。
