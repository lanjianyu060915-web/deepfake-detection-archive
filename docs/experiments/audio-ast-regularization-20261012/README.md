# AST 时频遮挡与广播混合内容诊断

本轮固定扩展 AIME/Urban 数据、划分、AST结构、优化器和选模dev，补无增强seed43，
比较两档训练时频遮挡，并引入外部广播纯音乐及音乐/语音片段。
等权分数融合将未训练音乐dev EER从18.75%降至14.58%，但广播混合内容仍弱。
结果摘要见 [summary.json](summary.json)；不读取holdout。

## 结果与选择

所有数字为EER（%），连续高分表示更伪造；每桶全部样本均有成功评分，覆盖率1.0、0失败。

| dev桶 | real/fake | 无增强42（前轮） | 无增强43 | F16/T64，42 | F32/T96，42 | 等权logit融合 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 已见音乐 | 48/240 | 0.4167 | 0.4167 | 0.4167 | 0.0000 | 0.0000 |
| 未训练音乐 | 48/106 | 18.7500 | 22.9167 | 25.0000 | 18.7500 | 14.5833 |
| Urban | 577/2308 | 0.6932 | 1.6898 | 2.5997 | 2.9029 | 0.7799 |
| 未训练环境生成器 | 577/1154 | 3.3795 | 4.4194 | 6.2392 | 9.0121 | 4.1594 |
| FSD独立来源 | 35/35 | 11.4286 | 5.7143 | 8.5714 | 8.5714 | 5.7143 |
| 广播纯音乐 | 47/47 | 38.2979 | 34.0426 | 29.7872 | 31.9149 | 31.9149 |
| 广播音乐/语音 | 76/76 | 57.8947 | 53.9474 | 52.6316 | 57.8947 | 57.8947 |
| AIME Suno v3.5 | 48/14 | 29.1667 | 28.5714 | 27.0833 | 10.4167 | 14.2857 |

两档遮挡单模型均未改善整体取舍，不作为默认训练配置；无增强seed42保留为单模型研究基线。
较强遮挡对Suno v3.5有收益，因此在看过单模型dev结果后，额外复用既有分数探索一次
**无增强seed42与F32/T96的等权logit平均**，公式为 `(score_a + score_b) / 2`，不搜索权重。
该融合未训练音乐18.75%→14.5833%、Suno v3.5为29.1667%→14.2857%、
FSD11.4286%→5.7143%；Urban0.6932%→0.7799%、未训练环境3.3795%→4.1594%，仍有取舍。
融合是事后dev研究候选，尚未接入双模型离线推理，额外推理成本未实测，不替换冻结环境音候选。
[融合配方](fusion.json)固定两份权重及SHA；逐样本融合分数在任务根
`expanded42-mask32-equal-logit-fusion-<bucket>/scores.jsonl`。

广播两桶共有28个pair；仅用共同pair各56段复算，原无增强42纯音乐39.2857%、混合50.00%，
F32/T96为28.5714%/46.4286%。共同pair也复用上游素材且标签纯度未知，
不足以把差异解释为语音的因果效应；总体混合内容依旧很弱。
不反转分数方向，也不以外部标签不确定为由删除难样本。

AIME的原曲/artist独立性、逐曲许可仍未知；dev参与早停与上述探索，不能充当冻结后验收。
广播数据保留外部探索用途，不作为训练标签来源。下一轮优先在已知train音乐与独立语音上
构造真假对称混合，再与纯音乐、环境音及本轮广播dev比较，不继续扩大遮挡参数搜索。

## 训练资源

| run | 最佳/完成epoch | 秒 | 峰值allocated/reserved GiB |
| --- | ---: | ---: | ---: |
| 无增强42（前轮） | 5/10 | 764.00 | 8.45/9.21 |
| 无增强43 | 8/13 | 1744.33 | 8.45/9.21 |
| F16/T64，42 | 2/7 | 1141.14 | 8.45/9.23 |
| F32/T96，42 | 3/8 | 802.52 | 8.45/9.23 |

前轮无增强42耗时列为历史参考；本轮三次训练部分并发，墙钟时间不可直接用于方法速度比较。
最佳权重位于各run的 `training/best.pt`，SHA-256见摘要；全部训练与评分已结束。
共享目录组为 `dlteam`、组可读不可写；缓存和原媒体直接复用，任务输出独立。

## 数据与方法

训练仍为14,788条：Urban13,450、AIME1,338；已见音乐dev288条选epoch。
MusicGen Small/Medium、AudioLDM 2 Large、Riffusion、Mustango参与音乐训练。
按类别平衡采样，音乐期望占比约8.52%；没有改变来源采样或加入外部广播数据。
同一份本地AudioSet预训练初始化，12层Transformer、最终LayerNorm和鉴伪头参与训练，
embeddings冻结。batch32、BF16 AMP、梯度累积2、AdamW lr1e-5/wd1e-4、
梯度裁剪1、最多20轮、patience5；复用35,922个已有fbank缓存文件。

新配置 `training.frequency_mask_width` / `time_mask_width` 默认为0。
训练时每样本分别抽取一个频率和时间遮挡，宽度为 `[0, 配置值)` 内整数，
起点从所有合法位置等概率抽取；F16/T64和F32/T96为本轮两档。
填充值为实际AST配置的 `-feature_mean/(2*feature_std)`，对应原始fbank零值。
只在训练循环应用，dev、推理和缓存均不遮挡。
4秒输入的fbank含补齐部分，512个时间位置不代表5.12秒真实内容。

处理顺序参考[官方AST dataloader](https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/src/dataloader.py#L186-L202)
与[SpecAugment方法](https://arxiv.org/html/1904.08779v3#S2)。
本实现使用独立整数抽样，不声称复现历史torchaudio的具体随机分布；
两档强度为探索配置，不照搬长音频AudioSet配置。

外部[AI-OpenBMAT来源与准备规则](../../research/source/ai-openbmat/README.md)：
固定128对60秒成品，取最长共同纯音乐或音乐/语音区间的中心4秒，
重建时间轴时扣除50ms交叉淡化，真假采用同一起点与处理。
得到246段：混合152段（76对）、纯音乐94段（47对），0准备失败；
分别52对和81对没有足够长的对应区间，计为规则排除。
两桶pair集合不同，不能把总体EER差异单独归因于语音混合。
所有外部样本仅作探索dev；上游Extend提示前缀是否移除未知，
AI侧4秒窗口未逐段确认全为生成音乐。底层BAF素材许可衔接未解决，不进入正式交付。

## 复现与资产

三份训练配置见本目录；相对路径以配置目录为准。已发布资产直接复用。
当前工作副本的 `data/audio-music-20261012` 指向前轮任务，
`data/audio-shared` 指向 `/var/tmp`，`data/audio-regularize-20261012` 与
`outputs/audio-regularize-20261012` 指向本轮任务根。
新训练使用新 `output_dir`，保留已完成run。

```bash
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/audio-ast-regularization-20261012/expanded12-mask32-96-seed42.json

scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/audio-ast-regularization-20261012/infer-baseline-openbmat-mixed-dev.json
scripts/devctl run python -m deepfake_detection_archive evaluate \
  --scores outputs/audio-regularize-20261012/reproduce-baseline-openbmat-mixed-dev/scores.jsonl \
  --manifest data/audio-regularize-20261012/ai-openbmat/mixed-dev-manifest.jsonl \
  --output outputs/audio-regularize-20261012/reproduce-baseline-openbmat-mixed-dev/evaluation.json
```

任务根 `/var/tmp/deepfake-audio-regularize-20261012/` 保存固定源码、配置、
本地原始资源SHA、导入provenance、训练日志、最佳权重及逐样本分数。
无增强训练与统一FP32推理源码固定于 `d4985a1`，遮挡训练于 `01b627d`，
广播片段导入于 `b80e9e9`。完整标识见摘要。
基线seed42沿用前轮 `/var/tmp/deepfake-audio-music-20261012/` 的已完成权重与五桶分数，
本轮只新增其广播外部评分，没有重训。

每次训练在固定代码副本上运行；两训练并发时RTX4090占用约20GiB、利用率约98%–100%。
耗时包含并发竞争，不用不同阶段的墙钟时间判断遮挡本身的开销。
各端到端评分固定batch8、FP32、本地媒体解码，不用训练BF16缓存分数替代。
