# [音频] 接入共享 ASV 数据并完成 LogMel/HuBERT 同划分对照

## 目标与依据

完成 LogMel seed43 与冻结 HuBERT seed42/43 的真实训练和同划分比较，判断是否值得进入后续泛化验证。旧 LogMel42 的 train25,379/dev24,844、dev EER0.510204% 是已知生成器语音开发结果，不是本届比赛成绩。

依据：[全量语音实验](https://github.com/lanjianyu060915-web/deepfake-detection-archive/blob/main/docs/experiments/asv-main-20261009/README.md)、[已有配置](https://github.com/lanjianyu060915-web/deepfake-detection-archive/blob/main/docs/experiments/asv-main-20261009/)、[预训练资源与固定版本](https://github.com/lanjianyu060915-web/deepfake-detection-archive/blob/main/docs/research/pretrained-resources/README.md)。本任务自行完成音频输入接入，无需等待一个统一环境 Issue。

## 工作范围

- 复用 `/srv/shared-data/datasets/asvspoof2019-la/` 中两个 Parquet：train25,380/dev24,844行；完整 SHA-256 已与同目录 `provenance.json` 一致。读取实际 schema，将原音频 bytes 物化到任务目录，复用 `tools/prepare_asvspoof2019.py` 和 `tools/prepare_asv_main.py`，必要时补一个薄的 Parquet 导入工具及行为测试。
- 核对官方 train/dev protocol、speaker、generator 和真假标签；保持原媒体字节，不以重编码或改哈希伪装同一资产。排除 train `LA_T_1023667`，目标 train25,379/dev24,844，`real=0`、`fake=1`。若与原清单/来源字节不同，明确记录差异；不能直接称作原协议复现，也不能拿旧0.51%当同条件结果。
- 将本地 HuBERT bundle 接入当前任务：优先恢复已转换资源；没有时按登记的固定上游 revision/转换过程取得，随附来源与许可，不修改历史源码快照或在训练/推理时隐式下载。仅需要 HuBERT，不与视频任务争用 ResNet18 获取工作。
- 复用 `logmel-seed43.json`、`hubert-seed42.json`、`hubert-seed43.json`，保留64,600采样点单片段、相同 train/dev、平衡训练采样。LogMel43保持旧42的batch、梯度累积和优化配置；若调整这些参数，在新设置下重跑42后再做双种子比较。HuBERT先小profile确定配置，42/43保持相同训练设置，feature_batch_size可单独profile；不把profile当训练。
- 新 run 目录保存 best checkpoint、连续 dev 分数、总体及A01–A06分组EER、失败率、耗时/吞吐/显存、种子及最佳轮次。HuBERT缓存BF16与实际离线FP32路径可能不同，结论按实际评分说明。

主要负责音频导入工具与相应测试、音频实值配置和新的实验卡。公共训练器仅在遇到真实问题时做最小改动。可先提交音频数据接入PR，再在本Issue跟进训练PR。

## 验收

- [ ] 共享Parquet可通过可复现命令接入，协议、样本数、来源和必要哈希清楚。
- [ ] LogMel43、HuBERT42、HuBERT43三次真实运行均有终态、逐样本分数和可比结果；资源或资产阻塞单独报告，不能当作实验完成。
- [ ] 需要比较旧LogMel42时，其原checkpoint/配置/分数已定位；否则如实限制结论。若重建协议有实质差异，先重跑seed42作为新协议基线。
- [ ] ASV eval及最终holdout未用于训练、早停或阈值选择；decode/model失败明确记账。
- [ ] PR包含必要代码、配置、简洁实验卡、产物位置和验证。没有收益也如实交付，不要求承诺更低EER。

不在本任务扩建训练恢复、遥测或融合框架；不中断或复制服务器现有官方原包下载。

## 环境与协作约定

当前服务器为 RTX 4090 24GiB，Python 3.12.15、torch 2.14.1+cu132；共享环境 `/opt/venvs/deepfake-cu132` 已可用，FFmpeg/ffprobe 已全局安装到 `/usr/local/bin`。复用现有依赖，不重新安装环境或修改驱动。

在自己的 clone/worktree 中，基于迁移修复分支 [`chore/cloud-migration-cleanup`](https://github.com/lanjianyu060915-web/deepfake-detection-archive/tree/chore/cloud-migration-cleanup) 建立 `issue/<本Issue编号>-<主题>`；该分支合并后改用最新 `main`。在仓库根目录使用统一入口：

```bash
scripts/devctl run python -m deepfake_detection_archive --help
# 代码变更先跑相关测试，PR前完成一次标准检查
scripts/devctl check
```

`/srv/shared-data/datasets` 为共享只读输入，目录/文件权限允许队友读取。派生媒体、清单、outputs/cache 使用自己可写的任务目录；需要把产物发布到 root 管理的共享目录时交由维护者处理，不因权限问题重下原包或放宽共享原媒体权限。大媒体、权重和完整分数不入 Git，PR 给出团队可读取的路径和必要哈希。

三条任务可独立进行数据准备和开发；只有一张 GPU，运行 GPU profile、特征提取或训练前在 Issue 记录预计占用时段，并与另一个模型任务错开。只修改本方向需要的代码；涉及公共训练器/编码器或依赖变更时先协调，避免同时修改。提交 LoRA 风格的小 PR，关联本 Issue，由负责人审查合并；不自行合并、扩容或正式参赛提交。

领取时在本 Issue 留下负责人、分支和第一步；未分派姓名，按方向领取。
