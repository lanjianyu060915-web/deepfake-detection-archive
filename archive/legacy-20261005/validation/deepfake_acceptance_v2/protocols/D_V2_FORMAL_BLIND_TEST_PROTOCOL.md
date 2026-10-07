# D-v2 正式盲测执行协议

## 适用范围

本协议只适用于从未被当前 baseline 推理、训练、调参或结果观察过的 D-v2 Core Blind。当前轮次仅构建体系，不执行正式模型测试。

## 一、测试前冻结

负责人和独立验收人共同完成以下动作，并在看到任何预测前签署：

1. 对鉴伪代码、实际 PyInstaller EXE 代码载体、模型、sidecar、融合配置、阈值、预处理和运行环境生成 SHA-256。
2. 完成 `DATA_INDEPENDENCE_STATEMENT_TEMPLATE.md`；逐样本证据只允许 REAL/FAKE，不允许 UNKNOWN。
3. 运行 dataset validator；任何错误必须在开考前解决，形成 ground truth manifest、manifest SHA-256 和数据文件 SHA-256。
4. 冻结 evaluator 代码、Python 解释器、runner 命令、timeout、严格标签映射和 score 语义声明。
5. ground truth 由独立保管人封存；模型开发人员不得在开考前看到样本、标签或预测。

## 二、一次性正式运行

1. 在隔离输出目录一次完整跑完，不并行覆盖，不复用旧 `result.json`。
2. 不得中途查看总体成绩后修改代码、模型、阈值、融合、预处理、数据或 evaluator。
3. 不得删除失败样本或改变分母。崩溃、timeout、结果缺失/非法、非 success、label 缺失/非法全部固定计错。
4. `FAKE` 是阳性类。原图主成绩与 JPEG/Scale/Blur/Brightness 分组分开；扰动结果是相关衍生诊断，并按 parent 聚合。
5. 只有当运行清单显式确认 score 为“数值越大越趋 FAKE”、所有样本均有合法有限 score 且两类真值齐全时才计算 ROC-AUC。

## 三、首次成绩封存

原始 stdout/stderr、逐样本目录、predictions、失败表、混淆矩阵、报告、run manifest、冻结清单和全部输出 SHA-256 永久保存。首次结果不得覆盖或删除。

## 四、后续规则

在看过 D-v2 结果后进行的任何模型修改，再跑同一数据只能称为 `regression test`。再次声称真正独立测试必须准备并封存新的 D-v3。

## 五、停止条件

若发现 ground truth 争议、泄漏、hash 改变、协议未签署或 Core 数据曾被模型运行，正式测试立即停止，不输出“独立成绩”；修复工程问题不能恢复已消耗的盲测独立性。
