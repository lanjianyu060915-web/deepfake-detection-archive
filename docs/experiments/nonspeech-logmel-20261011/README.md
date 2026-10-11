# 环境音 LogMel / 冻结 HuBERT 对照

2026-10-11，RTX4090。首次在已交付 Doppelganger/FSD50K 环境音上训练检测器。
数据准备与许可见[环境音数据卡](../nonspeech-20261011/README.md)。

## 数据与方法

固定 train322/dev70/holdout70，真假各半；同原片、派生及上传者组不跨划分。
本轮只训练 train、用 dev 早停与选模，没有读取 holdout 波形或评分。
输入为对称处理的单声道16kHz、64600点环境音，只有 Stable Audio Open 一种生成路径。

复用现有训练器，batch64、2 workers、BF16 AMP、累积1步，AdamW/BCE。
LogMel 默认 lr1e-3、最多30轮、patience5；补充探索将 lr降为1e-4，最多50轮、patience10，
因此不是只改变学习率的严格消融。HuBERT 固定本地 base-ls960，仅训练现有768维特征头，
lr1e-3、最多50轮、patience10。HuBERT 编码缓存使用 BF16，两个种子复用同一特征缓存。

## 结果

各次 dev 均70条、0失败、覆盖率100%；分数越高越伪造，EER使用项目插值协议。

| 方法 | seed42 dev EER | seed43 dev EER | 最佳/完成轮次42；43 |
| --- | ---: | ---: | --- |
| LogMel，lr1e-3 | 45.7143% | 37.1429% | 1/6；3/8 |
| LogMel，lr1e-4与较长早停 | 45.7143% | 40.0000% | 6/16；4/14 |
| HuBERT冻结特征头，BF16编码缓存 | 25.7143% | 31.4286% | 16/26；5/15 |
| 同一HuBERT权重，本地媒体→FP32编码→头 | 28.5714% | 34.2857% | 使用上一行已选权重 |

HuBERT 的端到端结果优于本轮 LogMel，但缓存与端到端各差2.8571个百分点，
不能拿缓存 EER 直接作为离线交付数字。这里还同时改变了编码精度及批处理方式，
没有单独归因于 BF16；后续候选应统一实际推理条件后选择。

FP32诊断中，RAIN六对两种子EER均0%，WATR八对均37.5%；
WIND只有一对，两个种子为100%/0%，说明小桶数字不稳定。
各事件分数汇总见 `summary.json`，这些结果不证明跨生成器、音乐或混合音频能力。

## 资源与产物

六次完整实验各约9–15秒；LogMel训练峰值allocated约0.86GiB。
HuBERT头训练峰值约0.08GiB，不含预编码主干，不能当作整个HuBERT任务显存。
LogMel的seed43与低学习率seed42并行，两个HuBERT端到端评分也并行。
FP32每次模型调用合计约1秒，另有约6秒加载；70条不足以外推正式吞吐。

团队可读根目录：`/var/tmp/deepfake-nonspeech-logmel-20261011/`。
六个run目录各含 `training/best.pt`、`dev-scores.jsonl` 和现有训练器报告；
`hubert-seed42-offline-dev/`、`hubert-seed43-offline-dev/` 保存实际推理分数与失败覆盖率。
`summary.json`集中记录必要数据/代码/权重/分数标识，不复制原始训练历史或评测报告。

## 复现与下一步

在新工作副本中将 `data/nonspeech-logmel-20261011` 指向已交付的
`/var/tmp/lanjianyu-issue11-20261011/doppelganger-v2/`，
将 `models/nonspeech-hubert-20261011` 指向
`/var/tmp/lanjianyu-issue9-20261011/hubert/models/hubert-base-ls960/`。
使用新的 `outputs/nonspeech-logmel-20261011/`，不要覆盖本轮共享产物。

```bash
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/nonspeech-logmel-20261011/logmel-seed42.json
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/nonspeech-logmel-20261011/hubert-seed42.json
```

seed43及低学习率配置同目录可直接使用；本地推理配置为 `infer-hubert-seed42.json`、
`infer-hubert-seed43.json`。将同一输入清单的dev行保存到新产物目录的 `dev-manifest.jsonl`；
重训后用新checkpoint的实际SHA更新推理配置，再以 `infer --config` 评分。

下一轮优先补独立生成器/来源和通用音频表征，保持同一dev比较；不继续堆LogMel学习率搜索。
当前只作为内部非商业研究候选；数据和HuBERT权重的正式赛用许可仍按资源登记处理。
