# D-v2 鉴伪自动评测器

本目录只实现“阅卷系统”。评测器不会导入、修改或训练鉴伪模型，而是逐样本通过 `subprocess` 调用已冻结的黑盒 runner。当前测试只使用 `tests/stub_deepfake_runner.py`，不得借此提前运行 D-v2 Core Blind。

## 标签与固定分母契约

`FAKE` 是阳性类，`REAL` 是阴性类。runner 标签只做下面两项精确、区分大小写的映射：

| runner `label` | 评测标签 |
| --- | --- |
| `deepfake` | `FAKE` |
| `real` | `REAL` |

不执行去空格、大小写折叠、子串或同义词匹配。`FAKE`、`fake`、`DeepFake`、`authentic`、`low_confidence` 等都属于非法标签。评测器也不会按照 score 重新设阈值。

ground truth 的每一行始终属于分母。runner 无法启动、超时、非零退出、结果缺失、非法 JSON、`status != success`、label 缺失或非法等情况全部计错。为了保证 `TP + FP + TN + FN = N`：

- 真值 FAKE 的系统失败记为 FN，其 `effective_prediction=REAL`；
- 真值 REAL 的系统失败记为 FP，其 `effective_prediction=FAKE`。

这只是固定计错所需的有效预测，不伪造模型输出。`raw_label`、`mapped_label`、`outcome` 和 `failure_code` 会保留模型误判与系统失败的区别。

## 指标契约

指标定义为：

```text
Accuracy    = (TP + TN) / N
Precision   = TP / (TP + FP)
Recall      = TP / (TP + FN)
Specificity = TN / (TN + FP)
F1          = 2TP / (2TP + FP + FN)
```

分母为零时，JSON 的 `value` 为 `null`，状态为 `NOT_COMPUTED_ZERO_DENOMINATOR`；不把未定义指标伪写为 0，也不输出非标准 JSON `NaN`。

ROC-AUC 只有在命令行显式传入 `--score-semantics higher_is_more_fake`，ground truth 同时含两类，全部 N 个样本均成功产生可用预测，且全部 score 都是 `[0,1]` 内有限数时才计算。否则输出 `NOT_COMPUTED` 及原因；绝不静默删掉失败或缺 score 的样本来计算子集 AUC。

## CLI

```powershell
python evaluate_deepfake.py `
  --python "<鉴伪环境python>" `
  --runner "<冻结的run_deepfake_detection.py>" `
  --dataset "<deepfake_ground_truth.csv>" `
  --output-dir "<新的evaluation_run>" `
  --timeout-seconds 120
```

可重复使用 `--python-arg` 与 `--runner-arg` 传递冻结环境所需参数。以连字符开头的参数要用等号形式，例如 `--python-arg=-S`、`--runner-arg=--device=cpu`。输出目录必须不存在或为空，防止覆盖第一次盲测结果。

样本执行失败是考试结果，而不是阅卷器崩溃：只要固定分母报告成功生成，CLI 仍返回 0。无效 manifest、输出目录非空或阅卷器内部错误返回 2。

## 输出结构

```text
evaluation_run/
├── per_sample/<sample_id>/
│   ├── result.json          # runner 原始结果；runner 未生成时不存在
│   ├── stdout.log
│   ├── stderr.log
│   └── status.json          # 命令、时间、退出码、失败码、固定计错结果
├── predictions.csv          # 恰好 N 行，保持 manifest 顺序
├── confusion_matrix.csv
├── evaluation_report.json
├── evaluation_report.md
├── failures.csv
├── false_positive_samples.csv
├── false_negative_samples.csv
├── false_negatives.csv
├── metrics_by_perturbation.csv
├── metrics_by_generation_method.csv
├── parent_summary.csv
└── run_manifest.json
```

`evaluation_report.json` 保存指标的值、状态、分子和分母，以及标签、固定分母和 AUC 策略。`run_manifest.json` 冻结 dataset、runner、evaluator、metrics 模块及顶层结果文件哈希。

## 分组和同源样本

`is_original=true` 归入 Original；扰动类型的规范值 JPEG、Scale、Blur、Brightness 进入对应组，其余进入 `Other:<原值>`。`generation_method` 空值归 `unknown`，其他值按完整字符串分组。每组均应用相同的失败固定计错规则并报告 N、真实类数量、TP/FP/TN/FN、Accuracy、Recall、Specificity。

当 `N < 30`、`n_REAL < 10` 或 `n_FAKE < 10` 任一成立时，分组写明 `descriptive_only=true`。同一原图及其扰动通过 `parent_sample_id` 汇总到 `parent_summary.csv`。含同源扰动的全样本微平均只作为诊断，不可当成完全独立样本的正式主成绩。

## Stub 模式与测试

任务文字称“七种”，但实际逐项列出了八种模式。本实现按不漏项原则支持：`all_correct`、`partial_wrong`、`all_real`、`all_fake`、`crash`、`invalid_json`、`status_error`、`timeout`；另有 `no_output` 和 `missing_label` 用于隔离测试缺结果与缺标签。

10 样本 fixture 的已知混淆矩阵为 TP=3、FP=1、TN=4、FN=2，对应 Accuracy=0.7、Precision=0.75、Recall=0.6、Specificity=0.8、F1=2/3。确认 score 语义后，fixture ROC-AUC=0.76。

只运行 stub 测试：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

