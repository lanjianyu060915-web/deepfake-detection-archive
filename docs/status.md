# 项目状态

更新：2026-10-11。**音频与受控视频双种子基线已交付**，尚未完成双分支比赛系统。
本页只记录已经交付的结果与正在推进的任务；计划见[路线图](roadmap.md)。

## 已完成与未完成

| 模块 | 已完成 | 未完成 |
| --- | --- | --- |
| 数据/评测 | 标准 manifest、来源登记、组级划分、EER 与失败覆盖率 | 比赛数据及官方评测程序尚未提供 |
| 语音模型 | 同一 ASV train/dev 上 LogMel/HuBERT 各双种子；每次24,844条评分0失败 | 未知攻击、跨来源、压缩及端到端离线路径验收 |
| 通用音频 | Doppelganger/FSD50K 环境音 231 对、462 条；train322/dev70/holdout70；实际训练解码可用 | 检测器尚未训练，音乐/混合音频及多生成器覆盖不足 |
| 视频模型 | 冻结 ResNet18 帧头双种子 EER20.55%/21.92%；Hunyuan train618/dev146 | 跨来源/生成器验收；主干微调和轻量时序尚未实现 |
| 鲁棒性 | AAC、带宽退化、H.264 的配对媒体工具与分数比较入口 | 只有小样本转码试跑，没有模型 EER 退化结果 |
| 离线交付 | 音视频研究权重、本地推理/恢复、导出打包；24条语音CPU断网演练 | 候选选择、双分支代表性批量验收、正式平台 I/O 适配 |

**现阶段最重要的成果缺口是通用音频、候选泛化及双分支离线验收。**
检查通过、准备工具完成和数据可解码，均不计为算法效果完成。

## 当前协作任务

| 任务 | 状态 | 下一份交付 |
| --- | --- | --- |
| [音频 #9](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/9) | 已由 PR #17 合并；四次训练、完整dev分数及权重已交付 | 研究候选泛化与离线路径评分 |
| [视频 #10](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/10) | 已由 PR #18 合并；受控视觉双种子及权重已交付 | 跨来源/生成器与压缩评分，再决定模型改进 |
| [非语音数据 #11](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/11) | 已由 PR #14 合并 | 研究集交给音频模型使用；当前单来源/单生成器 |
| 鲁棒性与分数比较 | PR #13、#15 已合并 | 用已交付候选实际评分，报告 ΔEER 和覆盖率 |

音频产物在 `/var/tmp/lanjianyu-issue9-20261011/`，
视频产物在 `/var/tmp/lanjianyu-issue10-20261011/`；两轮基线均已结束。
大权重、逐样本分数及缓存留在团队可读任务目录，后续 GPU 使用在新任务中协调。

## 已有模型结论

| 本轮模型 | seed42 dev EER | seed43 dev EER |
| --- | ---: | ---: |
| LogMel | 0.784929% | 0.941915% |
| 冻结 HuBERT | 1.498027% | 1.530612% |
| 受控 ResNet18 帧头 | 20.547945% | 21.917808% |

音频使用 train25,379/dev24,844，两种子 LogMel 均优于冻结 HuBERT，
本轮保留 LogMel42 为较优研究候选。HuBERT 的 BF16 编码缓存评分不等于离线 FP32 编码评测。
这是已知 A01–A06 语音开发集；未知攻击、非语音和最终 holdout 未验收。
详见[云端音频实验](experiments/asv-cloud-20261011/README.md)。

视频使用受控 train618/dev146，每次146条评分0失败，均最佳第4轮、第9轮早停；
同集宽度/联合元数据 EER 为50.00%/53.42%。这是同来源 Hunyuan 单生成器小开发集，
尚不能证明跨来源/生成器泛化。旧154条holdout在probe前排除，未解码、评分或调参。
详见[云端视频实验](experiments/vif-cloud-20261011/README.md)。

旧主机全量 LogMel42 dev EER **0.510204%**，checkpoint未在云端恢复，
不计入本轮双种子统计；详见[历史全量语音实验](experiments/asv-main-20261009/README.md)。

历史小样本语音 LogMel42/43 为 21.48%/24.22%，AASIST-L 为 25.78%/29.69%；
与全量划分不同，不能直接当作提升幅度。视频冻结帧模型 pad 为 32.47%/32.47%，
crop 为 33.77%/31.17%；同集宽度基线为24.75%，暴露了元数据捷径，未选为正式候选。

## 环境与资产

RTX 4090 24 GiB；Python 3.12.15、uv 0.13.0、torch 2.14.1+cu132。
共享环境 `/opt/venvs/deepfake-cu132` 可用，FFmpeg/ffprobe 全局入口在 `/usr/local/bin`。
环境已具备开发条件；无需再建设环境证明。数据、模型及许可位置见[数据说明](data.md)。

原始 ASV Parquet、ViF 和 FakeSound ZIP 位于 `/srv/shared-data/datasets/`。
非语音研究集在 `/var/tmp/lanjianyu-issue11-20261011/doppelganger-v2/`，团队可读；
FakeSound 缺少 real、许可 unresolved，保持单类诊断，不用于二分类 EER。
`/var/tmp` 不是已确认的持久卷，长期保留由维护者迁入平台持久化存储。

## 下一步

先用已经可用的环境音训练通用音频对照，并对新音视频研究候选做压缩和未知来源/生成器评分；
根据这些结果决定视频主干微调与时序投入。
候选冻结后完成双分支离线验收。具体依赖、交付物和取舍见[路线图](roadmap.md)。
