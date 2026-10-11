# ASV 云端音频双种子对照

2026-10-11，RTX4090，Issue #9 / PR #17。
同一 ASVspoof LA train25,379/dev24,844 上比较 LogMel 与冻结 HuBERT。
两个 LogMel 种子均优于冻结 HuBERT，本轮保留 LogMel42 为较优语音研究候选。

## 方法与结果

输入为64,600采样点单片段，逆类别频次有放回采样；batch256、BF16 AMP、
AdamW lr0.001、weight_decay0.0001，最多30轮、patience5。
LogMel随机初始化；HuBERT mean pooling为768维，编码器冻结，仅训练64维隐藏层头。
最佳轮次按dev EER选择，每次对全部24,844条dev保存连续logits，高分更假。

| 模型/种子 | dev EER | 最佳/完成轮次 | 完整实验秒数 | 评分失败 |
| --- | ---: | ---: | ---: | ---: |
| LogMel42 | 0.7849% | 23/28 | 888.69 | 0 |
| LogMel43 | 0.9419% | 20/25 | 756.32 | 0 |
| 冻结HuBERT42 | 1.4980% | 26/30 | 350.52 | 0 |
| 冻结HuBERT43 | 1.5306% | 20/25 | 189.19 | 0 |

四次覆盖率均为100%，训练无跳过更新。HuBERT42包含首次编码，43复用特征缓存；
训练器耗时及allocated显存峰值见[结果摘要](summary.json)，不包含完整媒体推理开销。

每个攻击子组使用全部2,548条real和该组3,716条fake：

| 模型/种子 | A01 | A02 | A03 | A04 | A05 | A06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LogMel42 | 0.1884% | 0.1962% | 0.1076% | 0.7849% | 0.8881% | 1.1572% |
| LogMel43 | 0.2960% | 0.1177% | 0.4710% | 1.3455% | 0.7804% | 1.5699% |
| 冻结HuBERT42 | 0.4037% | 0.3532% | 0.4306% | 0.8242% | 0.3498% | 3.3359% |
| 冻结HuBERT43 | 0.6279% | 0.9812% | 0.6997% | 0.9812% | 0.4317% | 3.2575% |

## 数据与复现

共享输入为 `Bisher/ASVspoof_2019_LA@aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c`。
`tools/prepare_asv_parquet.py` 按官方train/dev协议物化原FLAC，再由现有导入器生成清单；
排除 `LA_T_1023667`，与旧全量实验清单一致。
HuBERT固定资源、来源与许可见[数据说明](../../data.md)。

资产根目录 `/var/tmp/lanjianyu-issue9-20261011`，团队可读：

| 相对资产根目录 | 内容 |
| --- | --- |
| `data/LA/`、`data/main/` | 原媒体/协议、训练清单和来源 |
| `hubert/models/hubert-base-ls960/` | 本地HuBERT bundle |
| `runs/<模型>-seed<种子>/` | `training/best.pt`、dev分数、训练历史和日志 |

训练配置：[LogMel42](logmel-seed42.json) / [43](logmel-seed43.json)、
[HuBERT42](hubert-seed42.json) / [43](hubert-seed43.json)。配置中的
`data/issue9-asv` 指向资产根目录的 `data/`，
`models/pretrained/hubert-base-ls960` 指向上述bundle。
复训在配置副本中将 `output_dir`、`feature_cache` 改为自己的新任务目录，
数据/模型路径按配置所在目录解析，再使用项目 `train --config` 入口。
代码版本、清单/来源及checkpoint/分数哈希统一记录在摘要，路径相对资产根目录。

## 结论与下一步

冻结HuBERT在本设置下未改善EER，A06误差尤其明显。下一轮优先LogMel的压缩/重采样、
未知攻击和跨来源表现，并训练已有环境音，随后决定是否继续SSL微调。

以上只覆盖已知A01–A06语音dev，dev用于选模，未使用ASV eval或最终holdout。
HuBERT结果来自BF16编码缓存与FP32头，不能代替离线FP32编码验收；
两模型均为缓存输入评分，不是端到端离线吞吐。
旧主机0.510204%没有可恢复权重，本轮重新训练，不计入双种子统计。
数据按来源条款用于研究，正式比赛使用尚未确认；HuBERT权重为Apache-2.0。
