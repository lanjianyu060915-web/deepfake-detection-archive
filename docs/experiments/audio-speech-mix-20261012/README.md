# 音乐/语音对称混合训练

本轮针对音乐混入真实语音后的识别弱点，比较纯音乐重采样与波形混合训练。
两组使用相同样本量、类别采样、初始化、优化器及选模dev，不读取holdout媒体。
前轮已交付候选见[时频遮挡与广播诊断](../audio-ast-regularization-20261012/README.md)。

## 结果与取舍

**对称混合改善合成混合内容，但没有解决真实广播泛化，并损失部分环境音能力。**
保留混合研究候选，不替换原无增强/等权融合研究方案，也不替换冻结环境音候选。
下一步补独立音乐来源和真实混合数据，不继续围绕当前合成dev扩展超参数搜索。

以下为本地FP32端到端EER；四列分别为前轮无增强42、已固定的事后等权logit融合、
本轮纯音乐重采样与本轮混合训练。每个新模型九桶共6,258条评分记录、0失败，覆盖率1.0。
桶间存在共享样本，这不是6,258个独立媒体；完整数字与权重标识见[摘要](summary.json)。

| dev桶 | 条数 | 前轮无增强42 | 前轮等权融合 | pure-replay | speech-mix |
| --- | ---: | ---: | ---: | ---: | ---: |
| 已见纯音乐 | 288 | 0.42% | 0.00% | 0.83% | 1.25% |
| 未训练音乐路径 | 154 | 18.75% | 14.58% | 21.70% | 20.75% |
| Urban | 2,885 | 0.69% | 0.78% | 1.60% | 2.47% |
| 未训练环境生成器 | 1,731 | 3.38% | 4.16% | 6.41% | 7.11% |
| FSD外部来源 | 70 | 11.43% | 5.71% | 11.43% | 5.71% |
| 广播音乐/语音 | 152 | 57.89% | 57.89% | 53.95% | 50.00% |
| 广播纯音乐 | 94 | 38.30% | 31.91% | 36.17% | 29.79% |
| 已见合成混合 | 576 | 7.29% | 6.04% | 6.04% | 1.46% |
| 未训练路径合成混合 | 308 | 35.85% | 32.55% | 32.55% | 26.89% |

匹配对照中，已见合成混合下降4.58个百分点，未训练路径混合下降5.66个百分点。
后者在music/speech -5dB为31.13%→29.17%，+5dB为33.96%→25.00%；
纯音乐Suno v3.5为35.71%→21.43%，但只有14条fake且仍不如前轮融合14.29%。
Udio混合仍39.13%，真实广播混合仍50.00%，不能据合成结果宣称真实混合泛化。
广播两桶pair集合不同、AI窗口生成标签不完全确认，不能单独归因于混入语音。

前轮模型仅作已有参照，本轮选模dev及物理batch/累积方式与前轮不同；
只有pure-replay与speech-mix严格共享本轮训练设置和选模输入。
原等权融合配方不调权，新增两桶仅复用两个旧模型的分数计算；仍未接入离线双模型推理。
本轮为单种子探索，holdout媒体未读取、未评分，暂无冻结后的独立验收。

| run | 最佳轮/完成轮 | 共同选模dev EER | 训练秒数 | 训练吞吐范围 |
| --- | ---: | ---: | ---: | ---: |
| pure-replay-seed42 | 3/8 | 4.86% | 749.25 | 176.1–202.3条/秒 |
| speech-mix-seed42 | 5/10 | 1.39% | 991.07 | 154.2–201.7条/秒 |

两次训练实际峰值allocated14.95/reserved16.11GiB，BF16、有效batch64。
吞吐含与评分并发阶段，不作为独占GPU benchmark。训练和评分均已退出，GPU可供其他任务使用。

## 数据与对照

复用扩展AIME的train1,338/dev288及Urban train13,450。
真实语音来自共享ASV原train/dev，只取可用、已登记解码成功且至少4秒的real，
得到train606条、dev768条素材池。两池原speaker、原utterance及已知文件哈希不跨划分；
不重新划分，不读取语音或音乐holdout媒体。

对每个音乐group和SNR，以 `sha256(seed:group:snr)` 选择相同split的一条真实语音，
所有真假音乐及未训练生成路径共用该语音；音乐和语音均中心裁剪4秒。
音乐已是16kHz、单声道4秒；语音不补齐、不重采样。
整段RMS定义music/speech为-5、+5dB，不是active speech SNR：

```text
mixed = music / RMS(music) * 10**(snr_db/20) + speech / RMS(speech)
output = mixed * 0.95 / max(abs(mixed))
```

同一整体峰值缩放保持成分比例，输出16kHz PCM16；生成标签仅来自音乐，语音均为真实。
来源、原音乐parent/group、实际speech speaker、输入SHA、裁剪偏移及增益保留。
已知原曲、派生和speaker不跨split；同一背景可能在一个split内复用，不将片段数视为独立组数。
AIME原曲/artist关系仍未知，不能以描述组代表已确认artist独立性。

生成3,464个混合变体，0失败：train2,676、dev788。
已见音乐混合dev576条（96real/480fake），未训练路径混合dev308条（96real/212fake），
两桶共享96条真实混合样本；不同SNR也共享原曲，不能相加当作独立样本。
未训练生成路径仅进入dev，广播外部数据仅作诊断。

| 项目 | pure-replay | speech-mix |
| --- | --- | --- |
| 原始train | Urban13,450 + AIME1,338 | 相同 |
| 附加train2,676条 | 原AIME音乐各重复两次，引用原媒体/SHA | 对应音乐各加-5/+5dB真实语音 |
| train总数 | 17,464，real3,359/fake14,105 | 相同，父音乐和类别逐项对应 |
| 选模dev | 纯音乐288 + 已见混合576 =864 | 完全相同的输入与标签 |

按real/fake平衡采样，不平衡来源。音乐及派生期望占比约21.82%，前轮为8.52%；
纯音乐重采样对照用于区分这项变化与混合内容本身的收益。
选模dev含已见合成混合内容，结果不能当作独立验收。
未训练生成路径、FSD和广播dev已用于多轮方向选择，也不是未触及的最终测试集。
本轮不据这些dev宣称通用泛化；最终候选仍需冻结后的独立来源验收。

AST为同一AudioSet初始化、12层Transformer/最终LayerNorm/头微调，embeddings冻结。
batch64、BF16 AMP、累积1，有效batch仍64；AdamW lr1e-5/wd1e-4、clip1、
max20/patience5、seed42。本轮不启用时频遮挡。
batch64小profile峰值allocated14.90/reserved15.19GiB；复用35,922个已有fbank缓存文件。
两次训练顺序执行，评分可按剩余显存并行；没有个人GPU独占。

## 准备与复现

本轮训练和评分使用固定源码 `0b6b1d5`，任务根为
`/var/tmp/deepfake-audio-speech-mix-20261012/`。
派生用途取实际父来源的许可交集。来源登记随后收窄为research/training，
原登记保存在各目录的 `sources-run.jsonl`；原训练权限、媒体、权重及运行provenance未改变。
原资源与研究许可见[AIME](../../research/source/aime/README.md)及
[ASV来源](../../research/source/asvspoof2019/provenance.json)。
派生使用两者许可交集；AIME逐曲权利、artist独立性及正式赛用仍未确认，不进入正式交付。
ASV语音来源不能代表所有广播语音条件；合成混合结果也不能代替真实成品泛化。

```bash
scripts/devctl run python tools/prepare_audio_mixtures.py \
  --music-manifest /var/tmp/deepfake-audio-music-20261012/aime/manifest.jsonl \
  --music-root /var/tmp/deepfake-audio-music-20261012/aime \
  --music-sources /var/tmp/deepfake-audio-music-20261012/aime/sources.jsonl \
  --speech-manifest /var/tmp/lanjianyu-issue9-20261011/data/main/manifest-training.jsonl \
  --speech-root /var/tmp/lanjianyu-issue9-20261011/data/LA \
  --speech-sources /var/tmp/lanjianyu-issue9-20261011/data/main/sources.jsonl \
  --untrained-manifest /var/tmp/deepfake-audio-music-20261012/aime/unseen-dev-manifest.jsonl \
  --output /path/to/new-mixtures
```

已发布资产直接读取，不覆盖或重复物化。`mixtures/` 保存媒体、四份用途清单、来源、
provenance及摘要；`pure-replay/`、`speech-mix/` 为联合Urban/AIME的训练与共同dev清单。
联合清单将工具输出路径加上相对于 `/var/tmp` 的任务前缀；
**训练共同媒体根目录为 `/var/tmp`，不能直接用工具输出目录作control的data_root**，
因为 `music/` 链接引用已有AIME。当前 `data/audio-shared` 指向 `/var/tmp`，
`data/audio-speech-mix-20261012` 与 `outputs/audio-speech-mix-20261012` 指向任务根。

```bash
# 复制配置并将output_dir改为新目录，保留已完成run
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/audio-speech-mix-20261012/speech-mix-seed42.json

# 固定本轮混合权重，在新目录重新执行未训练路径混合dev推理
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/audio-speech-mix-20261012/infer-speech-mix-untrained-dev.json
```

完整最佳权重、逐样本分数、运行配置、固定源码及日志留在任务根，按 `dlteam` 共享读取。
