# 实验结果

这里只保存能支持算法选择的结论、配置和核心摘要。
权重、逐样本大分数、缓存及日志放 `outputs/<任务>/<run>/`，实验卡注明实际路径与必要标识。
每份卡说明数据/划分、方法、结果、失败数、资源和下一步；不复制训练器或保存个人工作流水。

## 当前主线

| 实验 | 当前结论 | 下一步 |
| --- | --- | --- |
| [ASV 云端双种子](asv-cloud-20261011/README.md) | LogMel EER0.7849%/0.9419%；冻结HuBERT未改善同域dev | 未知攻击、跨来源及压缩评分 |
| [语音压缩与未知攻击](audio-generalization-20261011/README.md) | LogMel42未知攻击EER2.97%；AAC退化较小，8kHz采样率带限EER13.10% | 带宽增强、跨来源及未知攻击叠加退化 |
| [ViF 云端受控视频](vif-cloud-20261011/README.md) | Hunyuan train618/dev146；视觉EER20.55%/21.92%，优于同集元数据 | 跨来源/生成器与压缩评分，再决定微调/时序 |
| [配对环境音](nonspeech-20261011/README.md) | 462条可训练环境音，已供首轮模型对照使用 | 补生成器/来源 |
| [环境音模型对照](nonspeech-logmel-20261011/README.md) | HuBERT本地FP32 EER28.57%/34.29%，优于本轮LogMel | 已开展AST微调，见下一行 |
| [环境音 AST 首轮微调](audio-ast-20261011/README.md) | 小型单生成器双种子dev EER2.86%/5.71%；冻结研究候选holdout EER11.43% | 后续扩展见下一行 |
| [多生成器环境音 AST](audio-envsdd-20261011/README.md) | 19,215条配对媒体；12层dev EER1.39%；冻结holdout已见1.43%/未见AudioLCM/TangoFlux6.51% | 音乐联合训练见下一行；继续扩大来源 |
| [音乐与环境音 AST 混合训练](audio-music-20261011/README.md) | 630条AIME音乐；单独/两种混合采样，五桶dev评分；补Urban12层seed43 | 扩大未训练音乐路径与真实原曲，补真正混合内容音频 |
| [鲁棒性媒体试跑](robustness-smoke-20261011/README.md) | 媒体条件工具可用，没有模型评分 | 使用已选择模型得到实际 ΔEER |

## 历史探索

| 实验 | 保留理由 |
| --- | --- |
| [ASV 旧主机全量](asv-main-20261009/README.md) | LogMel42 EER0.510204%；云端旧权重未恢复，不计入本轮统计 |
| [ViF 受控数据准备](vif-control-20261009/README.md) | 保留匹配与元数据基线方法；实际模型结果见云端实验 |
| [ASV 小样本对照](asv-pilot-quarantine-20261008/README.md) | LogMel/AASIST-L 两种子比较，解释为何优先 LogMel/SSL |
| [ViF pad](vif-pilot-20261007/README.md) / [crop](vif-crop-control-20261007/README.md) | 记录元数据捷径和中心裁剪无稳定收益 |
| [历史分数比较](score-comparison-20261011/README.md) | 既有四模型分数复算，说明生成器差异；不计为新模型结果 |
| [CPU 离线演练](../research/offline-rehearsal/README.md) | 24条语音离线执行/恢复事实，不能代替最终双分支验收 |

不同数据/划分的数字不直接比较。dev 用于选择，holdout 只在冻结后验收。
历史完整评分、脚本副本和执行流水在整理前提交 `4bbd292`，需要追溯时使用 Git 历史，
不再作为当前开发入口或训练前置检查。
