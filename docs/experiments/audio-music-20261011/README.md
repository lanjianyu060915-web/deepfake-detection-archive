# 音乐与环境音 AST 混合训练

2026-10-11，RTX4090。在既有 Urban 环境音候选上补音乐迁移诊断，
比较音乐单独训练、类别平衡混合训练、来源与类别联合平衡混合训练。
本轮所有模型只使用 train/dev；没有重新评分旧 holdout，也没有评分音乐 holdout。

## 数据与方法

取得固定 [AIME](../../research/source/aime/README.md) 的90个 Parquet，9,087,224,372字节，
扫描2,790条输入；选择与真实音乐具有相同三标签描述的生成音乐。
主清单为105条 real、525条 fake，按描述组合和已确认精确重复形成105个连通组，
train438/dev96/holdout96，0解码失败。真实与生成媒体统一为单声道16kHz、居中4秒、
峰值0.95、PCM16。训练包含 MusicGen Small/Medium、AudioLDM 2 Large、Riffusion、
Mustango 五条路径、四个家族；标签描述相同不等于原曲波形派生，artist/album未知。

未训练路径 dev 复用16条 real，另有9条 fake；与主 dev 的 real 重合，不能相加当独立样本。
全体未训练路径 dev/holdout 中 AudioLDM 2 Music/Suno v3/Udio 分别1/14/3条；
MusicGen Large 没有进入本轮选中描述的 dev/holdout。
AudioLDM 2 Music 与训练模型同家族，9条 fake 的 dev 只能用来发现薄弱点，
不能支持广泛未知家族泛化结论。AIME 原曲逐曲许可映射缺失，目前仅用于内部非商业研究。

混合清单只取 Urban train13,450条，加入全部音乐主清单630条，共14,080条。
训练13,888条；早停仍只用96条音乐 dev，不让大规模 Urban dev 淹没音乐信号。
Urban dev2,885、AudioLCM/TangoFlux dev1,731、FSD独立来源 dev70 均单独做迁移诊断。
Urban数据、旧权重与划分见[前轮实验](../audio-envsdd-20261011/README.md)。
FSD旧媒体使用推理采样的首64,000点，其原录音/上传者关联已从 Urban train 排除。

三组音乐相关训练从相同本地 AudioSet 预训练初始化，seed42；不从已微调 Urban 权重续训。
另补 Urban-only seed43。全部使用 AST 12个 Transformer block、末层LayerNorm与头微调，
patch/token/位置嵌入冻结；512帧 filterbank、BF16 AMP、batch32、累积2步、lr1e-5、
AdamW weight_decay1e-4、梯度裁剪1、最多20轮、patience5。推理使用完整本地权重、FP32。

混合对照仅改变采样权重：类别平衡时，每个类别等权，音乐期望采样占比约2.96%；
开启 `balance_sources` 时四个实际 `(source,label)` 桶等权，音乐期望占比50%。
每轮有放回采样13,888次；来源平衡不是复制媒体或更改评测权重。
音乐单独训练沿用类别平衡，三个实验不融合语音模型或音视频分数。

## 结果与取舍

结果见下表及 [summary.json](summary.json)。逐样本分数留在共享任务目录。

所有下列本地 FP32 评分均0失败；五桶样本数依次为96、25、2,885、1,731、70。

| 训练 | 音乐 dev | 未训练音乐路径 dev | Urban dev | 未训练环境生成器 dev | FSD dev |
| --- | ---: | ---: | ---: | ---: | ---: |
| Urban-only seed42（前轮） | 22.5000% | 22.2222% | 1.3865% | 5.0260% | 8.5714% |
| Urban-only seed43 | 18.7500% | 22.2222% | 1.6464% | 3.7262% | 8.5714% |
| Music-only seed42 | 3.7500% | 44.4444% | 27.3830% | 25.3033% | 20.0000% |
| 混合：来源+类别平衡 | 0.0000% | 18.7500% | 4.5494% | 11.4385% | 11.4286% |
| 混合：仅类别平衡 | 0.0000% | 11.1111% | 2.2097% | 5.3726% | 8.5714% |

| 本轮训练 | 最佳/完成轮次 | 训练秒数 | allocated / reserved GiB |
| --- | --- | ---: | ---: |
| urban12-seed43 | 2/7 | 572.51 | 8.45 / 9.21 |
| music12-seed42 | 10/15 | 83.95 | 8.45 / 9.21 |
| mixed12-source-balanced-seed42 | 1/6 | 900.93 | 8.45 / 9.21 |
| mixed12-class-balanced-seed42 | 2/7 | 967.00 | 8.45 / 9.21 |

音乐单独训练将已见音乐 EER 从原 Urban seed42 的22.50%降到3.75%，
但未训练音乐路径升到44.44%、Urban升到27.38%，因此不用于替换环境音候选。
两种混合采样在已见音乐 dev 上均为零 EER；类别平衡在其余四个诊断桶均优于来源平衡。
本轮保留 `mixed12-class-balanced-seed42` 为后续混合研究候选，最佳第2轮。
checkpoint SHA-256：`4b1da5d98caebd7d47eb86033c312a4c9699a7e6f8ec13e7ddca72efbcc884f7`。

相对 Urban-only seed42，该模型的已见音乐/未训练音乐路径为0%/11.11%，
Urban/未训练环境生成器由1.3865%/5.0260%变为2.2097%/5.3726%，FSD仍为8.5714%。
这是扩大内容覆盖的开发集取舍，尚不能替换已验收的环境音专用候选或作为比赛最终模型。
零 EER 来自16条 real 的已选模音乐 dev；未训练音乐只有7条 Suno v3、2条 Udio fake，
结果不足以断言商业音乐普遍可检测。两组混合实验仅seed42，未声称多种子稳定性。
50%音乐占比的来源平衡未在本轮取得收益，配置作为已完成对照保留，不默认开启。

Urban-only seed43 的已见/未训练环境生成器 dev 为1.6464%/3.7262%，
补足前轮12层双种子开发诊断；不依据已使用的 holdout 改选种子。

## 资源与复现

团队可读根目录 `/var/tmp/deepfake-audio-music-20261011/`，组 `dlteam`。
`upstream/aime/` 为固定原包，`aime/` 为只读音乐与清单，`mixed/` 只保存组合清单，
不复制 Urban 媒体。`fbanks512/` 复用既有 Urban 输入缓存并补音乐输入。
各 run 的 `experiment.json`、`training/run.json`、`training/history.json`、`training/best.pt`
保留配置、源码内容标识、数据标识、训练耗时与完整模型。
`<run>-<bucket>/scores.jsonl`、`evaluation.json` 为本地媒体评分与失败记账。

训练耗时不含缓存准备；两组混合训练并发，设备快照100%、19,931MiB，
不是单任务或全程平均值。输入缓存及已完成权重团队只读，运行摘要/逐样本分数团队可读。

本实例 `data/audio-music-20261011`、`outputs/audio-music-20261011` 链接到任务根目录，
`data/audio-shared` 链接到 `/var/tmp`；模型 bundle 链接同前轮。已准备数据直接复用。
音乐导入命令见[数据说明](../../data.md)；混合清单保持原 ID、group、label、split，
只把路径改为相对共享媒体根目录，并合并两份来源登记。
四份训练配置保存在本目录，新训练应换输出目录；不覆盖已交付权重与分数。

```bash
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/audio-music-20261011/mixed12-class-balanced-seed42.json
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/audio-music-20261011/infer-candidate-music-dev.json
scripts/devctl run python -m deepfake_detection_archive evaluate \
  --scores outputs/audio-music-20261011/mixed12-class-balanced-seed42-music-dev/scores.jsonl \
  --manifest data/audio-music-20261011/aime/dev-manifest.jsonl \
  --output outputs/audio-music-20261011/mixed12-class-balanced-seed42-music-dev/evaluation.json
```

复现推理不需要预训练目录，配置中的 checkpoint SHA 固定本轮权重。
推理输入均为 dev-only 清单。音乐 holdout 保留未使用；候选和数据覆盖充分后再冻结验收。
下一轮重点扩充真实原曲与未训练商业生成音乐、加入真正混合内容音频；
不因小开发集零 EER 就宣布通用音频系统完成。
