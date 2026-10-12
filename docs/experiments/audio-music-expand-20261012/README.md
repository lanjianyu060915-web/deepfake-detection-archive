# 音乐数据扩展与 AST 联合训练

2026-10-12，RTX4090，Issue #32。延续[音乐混合训练](../audio-music-20261011/README.md)，
补充最佳类别平衡方案的第二种子，并扩展 AIME 的真实音乐和预留生成路径。
旧、新 holdout 均不参与本轮训练、选模或评分。

## 原数据第二种子

保持原 train13,888、音乐 dev96、AST12层、学习率1e-5、batch32、梯度累积2步、
BF16 AMP与类别平衡，只将 seed42 改为43。全部从相同 AudioSet 预训练初始化。
训练8轮，最佳第3轮，耗时578.20秒；峰值 allocated8.45 / reserved9.21 GiB。
以下为本地FP32端到端评分，各桶均0失败。

| 候选 | 音乐96条 | 未训练音乐25条 | Urban2,885条 | 未训练环境生成器1,731条 | FSD70条 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原混合 seed42 | 0.0000% | 11.1111% | 2.2097% | 5.3726% | 8.5714% |
| 原混合 seed43 | 0.0000% | 11.1111% | 2.9463% | 7.5390% | 5.7143% |

音乐小样本的两个数字复现，但环境音存在种子波动，不能因音乐 dev 为零就认定稳定泛化。
本轮不据已经使用的 holdout 改选种子。
seed43 权重 SHA-256：`55d9340989e748618c619138ecc2d5b0f14a54d7969f017b3aff2dd51df7afa9`。

## 扩展协议

固定 AIME revision `b84d4be5eda830b6eb714998569dba73530f2601`，复用原90个分片，
新增29个，共119个分片、29,012,183,201字节。
路径、大小和官方LFS SHA-256见[资源清单](../../research/source/aime/expanded-resources-20261012.json)。
真实音乐和生成音乐继续统一为单声道16kHz、居中4秒、峰值0.95、PCM16。

导入使用原 `inventory.jsonl` 锁定旧样本、prompt与已确认重复关联的划分。
新增组沿用现有组级划分；若新增精确重复连接旧不同split，整个冲突分量排除并计数。
三标签组合只是语义匹配，AIME仍无可靠原曲/artist桥接，正式赛用许可尚未解决。

导入3,686条输入，主清单为319条real与每个训练生成路径319条fake，共1,914条，
train1,338/dev288/holdout288，319个描述连通组，0解码失败、0旧split冲突。
旧inventory中697条已分配split的记录全部保持归属。
预留生成路径dev为48条real和106条fake，共154条；这些real也在已见音乐dev中，
两桶不能相加当成独立样本。预留路径包括Suno、Udio、Stable Audio，及同家族的
AudioLDM 2 Music/MusicGen Large；后两者不能当作全新家族。

扩展训练继续使用原五条生成路径及全部Urban train，按类别平衡采样；
只改变可用音乐数据，不同时换结构或增强。
早停和epoch选择只用扩展后的已见音乐dev，预留生成路径只作固定候选诊断。
旧seed42模型也在同一扩展dev评分，数据数量改变前后的EER不直接比较。
联合训练共14,788条，其中音乐1,338条，类别平衡下音乐期望采样占比约8.52%，
高于首轮的2.96%；新增数据与自然采样占比同时改变，不能将收益单独归因于其中一项。

## 扩展训练结果

模型和超参数沿用首轮，seed42，从同一AudioSet预训练初始化。
训练10轮，第5轮最佳；后续dev反弹，早停保留该权重。
训练764.00秒，峰值allocated8.45 / reserved9.21 GiB，稳定阶段约200条/秒。
训练使用固定源码副本，来自提交`395a7d8`；导入工具来自`47e2e54`。

下表各条件均为同一批样本、本地FP32端到端评分、0失败、覆盖率1.0。
新音乐两桶重测旧seed42；三个环境音桶复用前轮同模型的完整分数。
摘要与权重/清单标识见[summary.json](summary.json)。

| dev桶 | 样本数 | 旧混合seed42 | 扩展混合seed42 |
| --- | ---: | ---: | ---: |
| 已见音乐 | 288 | 2.9167% | 0.4167% |
| 未训练音乐路径 | 154 | 25.0000% | 18.7500% |
| Urban | 2,885 | 2.2097% | 0.6932% |
| 未训练环境生成器 | 1,731 | 5.3726% | 3.3795% |
| FSD独立来源 | 70 | 8.5714% | 11.4286% |

未训练音乐总体改善6.25个百分点，但并非各生成器都改善。
以下每行复用同一48条real，不能相加当独立评测集：

| 未训练路径 | fake数量 | 旧混合seed42 | 扩展混合seed42 |
| --- | ---: | ---: | ---: |
| Stable Audio v1 | 23 | 22.9167% | 16.6667% |
| Stable Audio v2 | 20 | 25.0000% | 16.6667% |
| Suno v3 | 24 | 25.0000% | 12.5000% |
| Suno v3.5 | 14 | 25.0000% | 29.1667% |
| Udio | 23 | 39.5833% | 21.7391% |

另有AudioLDM 2 Music、MusicGen Large各1条fake，其零EER不支持泛化结论。
从扩展分数取回原96条已见/25条未训练音乐dev子集，新旧模型仍为0%/11.1111%，
说明新扩展集揭示了旧小样本没有覆盖的困难。

保留扩展模型作为后续音乐/环境音研究候选，checkpoint SHA-256：
`07b0b24dd94a850198d946902298fddae1d6a87d52f9c9654ce20fff233d32c7`。
FSD退化2.86个百分点，且未训练音乐仍为18.75%、Suno v3.5为29.17%；
本轮不宣布通用音频系统完成，也不替换已冻结验收的环境音专用模型。
扩展训练仅seed42。下一轮优先补跨来源真实音乐与真正混合内容，
针对困难音乐路径及环境音迁移做训练改进，保留新的独立数据作冻结后验收。

## 资源与复现

共享任务根为 `/var/tmp/deepfake-audio-music-20261012/`，组 `dlteam`。
原始分片在 `upstream/aime/`，音乐在 `aime/`，混合清单在 `mixed/`；
`fbanks512/` 复用原输入缓存，运行目录保存完整模型、配置、逐样本分数及必要日志。
`data/audio-music-20261012` 和 `outputs/audio-music-20261012` 链接到该任务根。

```bash
scripts/devctl run python tools/prepare_aime.py \
  --parquet-dir /var/tmp/deepfake-audio-music-20261012/upstream/aime \
  --resources docs/research/source/aime/expanded-resources-20261012.json \
  --previous-manifest data/audio-music-20261011/aime/inventory.jsonl \
  --output /path/to/new-aime --seed 42
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/audio-music-expand-20261012/expanded12-class-balanced-seed42.json
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/audio-music-expand-20261012/infer-candidate-music-untrained-dev.json
```

已完成运行不能原目录重启；复现时调整输出目录。
本轮仍是内部研究：没有补齐逐曲许可、艺术家隔离或真正混合内容的覆盖。
