# 实验结果

这里只保存能支持算法选择的结论、配置和核心摘要。
权重、逐样本大分数、缓存及日志放 `outputs/<任务>/<run>/`，实验卡注明实际路径与必要标识。
每份卡说明数据/划分、方法、结果、失败数、资源和下一步；不复制训练器或保存个人工作流水。

## 当前主线

| 实验 | 当前结论 | 下一步 |
| --- | --- | --- |
| [ASV 全量语音](asv-main-20261009/README.md) | LogMel42 dev EER 0.510204%；尚非通用音频/未知攻击结论 | #9 完成云端 LogMel/HuBERT 双种子可比实验 |
| [ViF 受控视频](vif-control-20261009/README.md) | Hunyuan train618/dev146；元数据 EER 约50%；视觉结果未交付 | #10 实际训练受控空间基线 |
| [配对环境音](nonspeech-20261011/README.md) | 462条可训练环境音，已完成数据交付 | 训练首个非语音候选并补生成器/来源 |
| [鲁棒性媒体试跑](robustness-smoke-20261011/README.md) | 媒体条件工具可用，没有模型评分 | 使用已选择模型得到实际 ΔEER |

## 历史探索

| 实验 | 保留理由 |
| --- | --- |
| [ASV 小样本对照](asv-pilot-quarantine-20261008/README.md) | LogMel/AASIST-L 两种子比较，解释为何优先 LogMel/SSL |
| [ViF pad](vif-pilot-20261007/README.md) / [crop](vif-crop-control-20261007/README.md) | 记录元数据捷径和中心裁剪无稳定收益 |
| [历史分数比较](score-comparison-20261011/README.md) | 既有四模型分数复算，说明生成器差异；不计为新模型结果 |
| [CPU 离线演练](../research/offline-rehearsal/README.md) | 24条语音离线执行/恢复事实，不能代替最终双分支验收 |

不同数据/划分的数字不直接比较。dev 用于选择，holdout 只在冻结后验收。
历史完整评分、脚本副本和执行流水在整理前提交 `4bbd292`，需要追溯时使用 Git 历史，
不再作为当前开发入口或训练前置检查。
