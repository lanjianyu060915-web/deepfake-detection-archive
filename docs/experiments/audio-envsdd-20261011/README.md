# 多生成器环境音 AST 微调

2026-10-11，RTX4090。将环境音研究数据从462条单生成器研究集扩展到19,215条
UrbanSound8K 配对媒体，比较 AST 微调深度与学习率，并用 AudioLCM/TangoFlux
开展未见生成器评测。模型与数据当前用于内部非商业研究。

## 数据与划分

固定 `EnvSDD/EnvSDD` revision `b0832726deb55484dbce869c6f09c41f7e9e5fbb`，
八个 train/validation Parquet 共3,003,079,622字节；另取四个 `remain` Parquet
共1,604,943,386字节作评测输入。版本、逐文件哈希和原录音署名见
[来源登记](../../research/source/envsdd/README.md)。

仅使用 UrbanSound8K。每个原切片有 real、AudioGen TTA、AudioLDM1 TTA、
AudioLDM2 TTA、AudioLDM1 ATA 五个成员；AudioLDM1/2 属同一方法家族。
原录音、切片、生成派生及已知上传者按连通组划分，处理后文件精确重复亦不跨 split。
上游 train/validation 存在同录音交叉，因此重新划分，不继承其 split 名称。

20,355条 Urban 输入中，1,140条因与旧研究集共享原录音或上传者而排除；
可用19,215条、3,843个配对家庭、238个连通组，解码失败0。
同批991条 TUT 输入不进入本轮。划分如下：

| split | real | fake | 合计 |
| --- | ---: | ---: | ---: |
| train | 2,690 | 10,760 | 13,450 |
| dev | 577 | 2,308 | 2,885 |
| holdout | 576 | 2,304 | 2,880 |

新清单 SHA-256：`eebe2feb8b9fe424ee03eedd7286a434d01b6420b6068667b23306240e0e3a22`。
全部训练输入对称处理为单声道16kHz、居中裁剪/补齐64,000点、峰值0.95、PCM16。
短原片、旧关联和不完整家庭的训练排除规则保留在库存；本轮实际排除均为旧关联。

AudioLCM/TangoFlux 未进入训练。按既有 dev 原文件名关联，各取得577条 fake，
加577条原样复用的 real，合计1,731条，0失败。未见生成器评测仍使用同一城市环境音语料，
不能当成跨来源结果。holdout 仅在候选冻结后物化和评分。

## 模型与调优

复用[AST 接入实验](../audio-ast-20261011/README.md)的本地 AudioSet 预训练资源，
所有新训练从相同预训练权重初始化，不继承旧鉴伪头。比较：

- 最后4个 Transformer block，lr1e-5，seed42/43。
- 全部12个 Transformer block，seed42，lr1e-5/3e-6。

末层 LayerNorm 和分类头均训练；patch、特殊 token 和位置嵌入仍冻结，
“12层”不表示所有 AST 参数可训练。AdamW/BCE、BF16 AMP、有效 batch64、
weight_decay1e-4、梯度裁剪1、平衡采样、最多20轮、patience5、4 workers。
末四层用batch64；12层用batch32、累积2步。dev 早停选最佳 checkpoint。

4秒波形只有约398个 filterbank 帧；原 AudioSet 配置补到1024帧。
本轮改为512帧，将 token 从1214减到602。初始化依照
[AST 作者的时间位置裁剪方法](https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/src/models/ast_models.py#L141-L154)，
逐频率位置居中裁剪时间位置，保留两个特殊 token。这保留4秒输入内容，
但改变注意力序列，需要重新训练，不能视为旧模型的无损推理优化。
缓存仅保存输入 filterbank；512帧缓存为16,335个张量、约4.32GB，不缓存主干输出。

旧1024帧末四层训练保留一轮结果：dev EER7.6256%，单轮209.22秒。
随后停止该试跑并以新目录开始512帧实验；旧1024帧12层任务没有完成轮次，不列入模型效果。
同batch32的12层 GPU 训练 step profile 从0.3595秒降为0.1339秒，约2.69倍；
此为独立小 profile，不能当成端到端训练总耗时提升。

## 评测与选择

四组训练均完成；下表为本地 FP32 媒体推理 EER，所有评测均0失败。
seen dev 为2,885条，未见生成器 dev 为1,731条，FSD 独立来源 dev 为70条。

| 512帧微调 | seen dev | 未见生成器 dev | FSD 独立来源 dev | 最佳/完成轮次 |
| --- | ---: | ---: | ---: | --- |
| 末四层，lr1e-5，seed42 | 5.5459% | 11.6118% | 17.1429% | 4/9 |
| 末四层，lr1e-5，seed43 | 6.2392% | 12.6516% | 17.1429% | 5/10 |
| **12层，lr1e-5，seed42** | **1.3865%** | **5.0260%** | **8.5714%** | **1/6** |
| 12层，lr3e-6，seed42 | 2.2097% | 5.9792% | 8.5714% | 7/12 |

12层微调在三个开发诊断上均优于末四层；降低学习率没有改善本轮结果。
12层仅运行seed42，尚不能声称其结果具备双种子稳定性。
FSD 诊断沿用旧研究集 dev 的35条 real、35条 Stable Audio Open fake；
所有旧集原录音/上传者关联已从新训练数据排除，不使用旧 holdout。
旧媒体按推理采样取首64,000点。该诊断支持跨来源迁移，但70条不足以代表广泛来源。

旧 Stable Audio Open 研究候选直接迁移至新 dev，EER为33.6222%，
2,885条0失败。其原小样本 dev 的2.86%未延续到新来源/生成器。
新候选均用本地 FP32 主干从媒体端到端评分；HF 离线模式禁用隐式下载。
逐生成方法 EER 使用同一批 real 与该方法 fake，分数越高越伪造。
新旧候选同时改变数据、微调深度和时间窗口，迁移差值不能归因于单一改动；
本轮层数和学习率对照则使用相同数据与512帧输入。

本轮以 seen-generator dev EER 选择候选；AudioLCM/TangoFlux dev 为迁移诊断。
于 `2026-10-11T08:02:10.905418+00:00` 固定12层 lr1e-5/seed42，
checkpoint SHA-256 为 `8fa61daf42054b29fd61906c8c2fcf679ba4558e80472806197958b013c6e95c`。
冻结后验收两个 holdout 条件，不据 holdout 回调学习率、层数或种子。
结果和权重标识保存于 `summary.json`。

| 冻结候选验收 | real / fake | EER | 失败 |
| --- | ---: | ---: | ---: |
| 已见四条生成路径 holdout | 576 / 2,304 | **1.4323%** | 0 |
| 未见 AudioLCM/TangoFlux holdout | 576 / 1,152 | **6.5104%** | 0 |

两个条件复用相同576条 real，合计4,032条独立媒体，不能把评分次数相加当独立样本数。
未见生成器 holdout 比其 dev 更难；本轮保留结果，不据此调参。

| 生成方法（与同 split 的 real 配对） | dev EER | holdout EER |
| --- | ---: | ---: |
| AudioGen TTA | 2.5997% | 2.6042% |
| AudioLDM1 ATA | 1.0399% | 1.0417% |
| AudioLDM1 TTA | 0.6932% | 0.6944% |
| AudioLDM2 TTA | 0.8666% | 0.6944% |
| AudioLCM TTA（未训练） | 4.8527% | 6.0764% |
| TangoFlux TTA（未训练） | 5.1993% | 6.5972% |

## 资源与复现

团队可读根目录 `/var/tmp/deepfake-audio-expand-20261011/`，组 `dlteam`。
`envsdd-urban/` 是只读共享训练数据，`unseen-dev/`、`unseen-holdout/` 是评测数据；
`upstream/` 保存固定原包与 credits，`fbanks512/` 是共享只读训练输入缓存。
各 run 下保存配置、必要资产标识、训练历史、最佳完整 checkpoint 和 dev 分数；
`*-offline-dev/`、`*-unseen-dev/`、`*-fsd-external-dev/` 保存开发诊断；
`*-seen-holdout/`、`*-unseen-holdout/` 保存冻结后验收。
checkpoint 包含完整主干、头及前端，推理无需预训练目录。
末四层训练循环峰值 allocated/reserved 为5.57/6.66GiB，12层为8.45/9.21GiB，
包含dev，不含特征准备。四次训练约855、1,058、1,074、1,720秒，
两任务并行，设备快照利用率100%；整卡快照包含其他GPU进程，不能当成单任务或全程平均值。

本实例的 `data/audio-expand-20261011`、`outputs/audio-expand-20261011` 和
`models/pretrained/ast-audioset` 分别链接到共享数据、任务根目录和只读预训练 bundle。
新实验换输出目录；重训后按实际 checkpoint SHA 更新推理配置。

```bash
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/audio-envsdd-20261011/full12-512-lr1e-5-seed42.json
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/audio-envsdd-20261011/infer-candidate-dev.json
```

`infer-candidate-unseen-dev.json` 固定未见生成器 dev 输入。
`infer-candidate-holdout.json`、`infer-candidate-unseen-holdout.json` 保留本轮复现用途，
holdout 推理需 `--frozen-audit`；已使用的 holdout 不作为后续调参集。
数据准备命令见[数据说明](../../data.md)。原 UrbanSound8K 的 CC BY-NC 3.0
仍适用；集合 CC BY 4.0 不覆盖原媒体限制，当前权重未批准正式比赛交付。
本轮覆盖城市环境音和多个生成路径；音乐、混合音频及跨来源仍需独立数据。

后续已补12层 seed43，并开展音乐与环境音混合训练，见
[音乐混合实验](../audio-music-20261011/README.md)。本卡保留前轮选模与冻结验收事实。
