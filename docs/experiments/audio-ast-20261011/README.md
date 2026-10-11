# 环境音 AST 主干微调

2026-10-11，RTX4090。在同一环境音划分上接入 AudioSet 预训练 AST，
完成四次训练、双种子本地推理和一次冻结候选 holdout 评估。

## 数据与方法

复用[环境音数据卡](../nonspeech-20261011/README.md)的 train322/dev70/holdout70，
真假各半，同原片、派生及上传者组不跨划分。输入为单声道16kHz、64600点，
真实与生成媒体对称处理；当前只有 Stable Audio Open 一种生成路径。

预训练为 `MIT/ast-finetuned-audioset-10-10-0.4593`，固定 revision
`f826b80d28226b62986cc218e5cec390b1096902`，来源及 BSD 许可见
[资源登记](../../research/pretrained-resources/README.md)。
使用官方 ASTFeatureExtractor、AudioSet 归一化和 SDPA；缓存输入 filterbank，
不缓存主干输出。训练最后四个 Transformer block、末层 LayerNorm 和二分类头，
冻结层保持 eval。对照只训练 LayerNorm 和分类头，不是完全冻结的 AST 线性探针。

复用项目训练器：AdamW/BCE、BF16 AMP、batch32、累积2步、有效batch64、
weight_decay1e-4、梯度裁剪1、最多40轮、patience8、平衡采样、2 workers。
仅比较 seed42 的 lr1e-5/1e-4，并在 lr1e-5 复跑 seed43。
早停和选模均用 dev；训练与缓存只处理 train/dev 波形。

## 结果与选择

每次 dev 均70条、0失败、覆盖率100%；EER 使用项目插值协议，分数越高越伪造。
缓存 dev 和本地离线推理均使用 FP32 主干。

| 方法 | seed42 dev EER | seed43 dev EER | 最佳/完成轮次42；43 |
| --- | ---: | ---: | --- |
| 末四层微调，lr1e-5 | **2.8571%** | **5.7143%** | 7/15；11/19 |
| 末四层微调，lr1e-4 | 2.8571% | 未运行 | 4/12；— |
| 仅 LayerNorm＋分类头，lr1e-4 | 17.1429% | 未运行 | 18/26；— |
| 既有 HuBERT，本地 FP32 端到端 | 28.5714% | 34.2857% | 见上一轮实验卡 |

lr1e-4 没有优于 lr1e-5；相同 lr1e-4/seed42 下，开放末四层优于 LayerNorm＋头对照。
AST 两种子的本地媒体推理分别复现2.8571%/5.7143%，各70条0失败，
不依赖预训练目录或网络。HuBERT 数字来自[同划分对照](../nonspeech-logmel-20261011/README.md)。

按 dev 选择 lr1e-5/seed42；学习率打平时取较小值。于
`2026-10-11T05:59:01.577245+00:00` 固定本轮环境音研究候选及权重哈希，
随后只运行一次 holdout：**EER11.4286%，70条，0失败，覆盖率100%**。
本轮不根据 holdout 继续调参；这不是整场比赛的最终候选冻结。

dev 仅35对，且经过多次选模；holdout 较差说明不能以2.86%代表泛化。
holdout 的 FOOT 八对 EER25%、FIRE 六对16.67%，仅用于指出当前薄弱内容，
不用于本轮选择。其余小桶和完整必要标识见 `summary.json`。
当前结果支持保留 AST 部分微调；未证明跨生成器、音乐或混合音频能力。
数据仍仅批准内部非商业研究，正式赛用按来源登记处理。

## 资源与产物

先在真实输入上 profile batch16/32/64；最终 batch32 微调训练峰值 allocated 约5.73GiB、
reserved 约6.89GiB。四次完整实验约49–110秒，包含初始化/缓存准备和最终评分，
单次训练约40–101秒。两组训练分别并发，实测 GPU 利用率快照达94%；不是全程平均值。
离线70条每次模型调用合计约1.7秒，另有约6.4秒加载，小样本不外推正式吞吐。

团队可读根目录：`/var/tmp/deepfake-audio-ast-20261011/`，组 `dlteam`。
`pretrained/ast-audioset/` 为只读输入；各 run 保存训练器原始报告、
`training/best.pt` 和逐样本分数；`*-offline-dev/`、`*-offline-holdout/` 保存实际推理结果。
checkpoint 包含完整主干、头和前端配置。Git 只保留本卡、实值配置和紧凑摘要。

## 复现与下一步

本实例将 `data/nonspeech-logmel-20261011` 指向既有环境音目录，
`models/pretrained/ast-audioset` 指向上述本地 bundle。
其他机器通过 `tools/prepare_pretrained.py prepare-ast` 获取同版本；
使用新输出目录，重训后按实际权重 SHA 更新推理配置。

```bash
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/audio-ast-20261011/last4-lr1e-5-seed42.json
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/audio-ast-20261011/infer-last4-seed42.json
```

推理用同一原清单中的 dev 行保存为输出根目录 `dev-manifest.jsonl`；
本轮已固定的 holdout 配置是 `infer-last4-seed42-holdout.json`，配套
`holdout-manifest.jsonl`，执行需 `--frozen-audit`。它保留复现用途，不作为下一轮调参集。
下一轮补独立生成器/来源，再逐步增加音乐和混合音频；不继续围绕当前小 dev 堆学习率搜索。
