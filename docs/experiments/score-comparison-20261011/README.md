# 历史真实分数的配对比较

日期：2026-10-11。目的：验证新分数比较入口能够读取现有真实评分，按同一批ID比较EER，
为队友候选交付后的压缩/重采样评测提供可直接使用的报告。没有启动训练或重新推理；
这不是新的媒体鲁棒性、全量训练或比赛成绩。

输入来自[ASV保守隔离pilot](../asv-pilot-quarantine-20261008/README.md)已归档的四份
`dev-scores.jsonl`，全部为同一批512条dev：256真、256假。原评分文件保持不变。
本轮复算未设置阈值，不调参、不融合、不重分数据。各份报告保存输入SHA-256和当前源码身份；
分数行中的组字段没有重新用清单推导，不能用于声明独立样本数。

## 总体结果

以LogMel seed42为基准（EER 21.484375%）：

| 候选 | EER | 候选−基准 | 完整/共同成功覆盖率 |
| --- | ---: | ---: | --- |
| LogMel seed43 | 24.218750% | +2.734375个百分点 | 100% / 100% |
| AASIST-L seed42 | 25.781250% | +4.296875个百分点 | 100% / 100% |
| AASIST-L seed43 retry1 | 29.687500% | +8.203125个百分点 | 100% / 100% |

四份输入各512条、0失败，复算与历史记录一致。两侧全量和共同成功集合相同；没有用删除
失败样本的方式改变比较集合。失败集合不同的情况由合成行为回归覆盖，不将其伪装成真实故障实验。

## 生成器诊断

每行使用该生成器的fake及**同一份256条real对照**；各行共享real，不是独立实验。
下表是描述性EER，不是宏平均或显著性检验：

| 生成器 / fake数 | LogMel42 | LogMel43 | AASIST42 | AASIST43 retry1 |
| --- | ---: | ---: | ---: | ---: |
| A01 / 45 | 1.9531% | 4.6875% | 1.5625% | 2.2222% |
| A02 / 56 | 21.4286% | 25.0000% | 21.4286% | 35.7143% |
| A03 / 35 | 0.3906% | 0.0000% | 11.4286% | 9.3750% |
| A04 / 45 | 0.3906% | 2.2222% | 8.8889% | 5.8594% |
| A05 / 34 | 17.6471% | 24.6094% | 46.0938% | 44.5313% |
| A06 / 41 | 65.8537% | 65.8537% | 58.5366% | 62.8906% |

这个旧小样本pilot中，A06对四个候选均困难；AASIST在A06的点估计较低，总体EER仍较高。
后续候选应保留分生成器诊断。这些差异不能外推到全量LogMel42的0.510204%结果；
pilot dev曾用于早停，仅有19个保守统计组，不作正式置信区间、holdout或泛化结论。

## 复现与产物

在仓库根目录运行，不需要原媒体、权重或GPU：

```bash
for candidate in logmel-seed43 aasist-l-seed42 aasist-l-seed43-retry1; do
  scripts/devctl run python tools/compare_scores.py \
    --baseline-scores docs/experiments/asv-pilot-quarantine-20261008/training/logmel-seed42/dev-scores.jsonl \
    --candidate-scores "docs/experiments/asv-pilot-quarantine-20261008/training/$candidate/dev-scores.jsonl" \
    --output "outputs/score-comparison-20261011/$candidate.json"
done
scripts/devctl run python docs/experiments/score-comparison-20261011/by_generator.py
```

三份完整配对报告、生成器摘要及标准检查日志保存在本机
`outputs/score-comparison-20261011/`，不重复提交历史分数或完整DET曲线。
运行时没有核对模型身份或重验清单/原媒体；工具报告保留这两项未核验标记。
18项新回归覆盖配对、失败交集、缺类、错误元数据及固定阈值；独立评审通过。
本地标准检查431项pytest通过（25.33秒，无跳过），3项视频工具测试、Ruff、严格mypy和
构建通过；研究复算脚本的Ruff、格式及mypy单独通过。远端验证按本轮PR实际结果报告。
