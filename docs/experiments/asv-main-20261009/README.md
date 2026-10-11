# ASVspoof 全量开发对照

2026-10-09启动，2026-10-10归档。全量LogMel42已完成；冻结HuBERT轻量头对照
将在相同数据、片段采样和开发集上于云端执行。它是语音开发实验，尚不代表通用音频或未知生成器泛化。

## 数据和协议

原始官方 train/dev 共50,224条，完整原生16kHz单声道FLAC解码约30.77秒，0失败。
保留官方说话人划分：train/dev各20名说话人，生成样本均来自A01–A06。
官方eval没有取得或访问；dev用于早停和选择，不能当最终留出。

旧谱相似候选补了实际波形诊断：

| 配对 | 全信号能量归一化最大绝对相关 | 最佳偏移 | 处理 |
| --- | ---: | ---: | --- |
| dev LA_D_1032020 / LA_D_1073474 | 0.92342 | 30样本 | 保留同一dev；两条PCM不完全相同 |
| dev LA_D_1062379 / train LA_T_1023667 | 0.93315 | 58样本 | 保守排除这一条train音频；整个说话人保留 |

高相关支持进一步隔离该跨划分媒体，但不证明原始语句或身份关系已经全面确认；
本轮没有再做全量近重复扫描。与旧pilot不同，这里不因未确认候选排除整个说话人。
最终train25,379条（real2,580/fake22,799），dev24,844条（real2,548/fake22,296）。

原完整清单、训练清单、来源和解码摘要保存在 `data/asvspoof2019-la-main/`。
训练清单哈希：`8b204f61a679a2789280a731aae1cefb63a29bd6fe009e66b0b34347aa148257`。
数据使用官方ASVspoof/VCTK来源许可进行本地研究训练；比赛外部数据细则仍未明确。

## 实验选择

- 两模型均使用确定性单片段64,600采样点、同一train/dev；真/假采用相同预处理。
- 训练采用逆类别频次加权、有放回采样，每epoch抽25,379条，期望真假各半。
  因此每轮不保证遍历全部fake；dev保留全部原分布，不做平衡抽样。
- LogMel随机初始化；HuBERT固定本地 `hubert-base-ls960`，mean pooling得到768维，
  冻结编码器，仅训练64维隐藏层的轻量二分类头。不是同参数量模型，也不是HuBERT主干微调。
- batch256、workers4、BF16 AMP、AdamW lr0.001、最多30轮、patience5。
  HuBERT特征提取batch32，编码也使用BF16；开发分类头评测使用FP32。
- 一次解码登记后复用缓存；两种子配置共享相关缓存，不把无关文档或CLI修改作为缓存输入。

实测GPU profile：LogMel batch256训练3步0.774秒、峰值allocated3.19GiB/reserved4.81GiB；
HuBERT batch32编码3批0.423秒、allocated2.35GiB/reserved2.86GiB。它们是短探针，
实际训练吞吐和峰值以各run为准；推理时SF后端不重采样、混声道或提供墙钟超时。

## 复现

数据准备（已有已验证ZIP，不重复下载或归档校验）：

```bash
scripts/devctl run python tools/prepare_asvspoof2019.py \
  --root data/asvspoof2019-la/LA --output-dir data/asvspoof2019-la-main
scripts/devctl run python tools/prepare_asv_main.py \
  --manifest data/asvspoof2019-la-main/manifest.jsonl \
  --root data/asvspoof2019-la/LA \
  --sources data/asvspoof2019-la-pilot/sources.jsonl \
  --output-dir data/asvspoof2019-la-main --exclude-train LA_T_1023667
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  scripts/devctl run python -m deepfake_detection_archive train --config \
  docs/experiments/asv-main-20261009/logmel-seed42.json
```

将配置换为`hubert-seed42.json`运行同划分对照，seed43也已保存。
输出目录须是新的；当前不支持optimizer/RNG训练续跑。配置从本目录解析相对路径。

每个run保存训练历史、best checkpoint、连续`dev-scores.jsonl`和`dev-evaluation.json`。
这些分数来自best checkpoint及已准备dev张量，评分时间排除解码和编码器提取；
不是端到端离线吞吐。HuBERT缓存使用BF16，而当前离线编码入口使用FP32，后续须报告
实际离线结果，不能直接声称两条路径分数完全相同。

## 已完成结果与云端接续

2026-10-10核对：**LogMel seed42已完成；LogMel43与HuBERT42/43尚未训练**。
后续运行移交云端，任务见[音频任务 #9](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/9)。

| 指标 | 实际值 |
| --- | ---: |
| 最佳dev EER | **0.510204%** |
| 最佳轮次 / 已完成轮次 | 30 / 30 |
| dev成功 / 失败 | 24,844 / 0 |
| 完整实验耗时 / 训练器耗时 | 1,614.62 / 1,519.13秒 |
| 训练吞吐中位数 | 863.85条/秒 |
| 峰值allocated / reserved | 3.24 / 5.37GiB |
| skipped optimizer updates | 0 |

开发集按fake生成器分组、每组使用全部2,548条真实音频及3,716条对应fake：

| A01 | A02 | A03 | A04 | A05 | A06 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2422% | 0.0538% | 0.0269% | 1.0204% | 0.7266% | 0.5102% |

这不是未知生成器留出。第30轮仍是最佳，因max_epochs结束，不能说已经收敛；
没有第二种子稳定性或最终holdout结论。与旧dev512划分不同，不能直接把21.48%到0.51%
解释成同一评测上的提升幅度。

原run记录源码commit `9eb78ee`、`dirty=true`及源码哈希，保留该事实；
后续提交包含批量缓存和入口修正，不能把本次运行改写为使用后续commit完成。
小型结果与完整30轮曲线数据纳入Git：`summary.json`、`training-history.json`。
原模型和连续分数仍在Git外，迁移成员及SHA见[资源说明](../../data.md#旧实验资源恢复)。
