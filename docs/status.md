# 项目状态

更新：2026-10-11。**音视频基线及多生成器环境音 AST 已取得实际结果**，尚未完成双分支比赛系统。
本页只记录已经交付的结果与正在推进的任务；计划见[路线图](roadmap.md)。

## 已完成与未完成

| 模块 | 已完成 | 未完成 |
| --- | --- | --- |
| 数据/评测 | 标准 manifest、来源登记、组级划分、EER 与失败覆盖率 | 比赛数据及官方评测程序尚未提供 |
| 语音模型 | 同一 ASV train/dev 上 LogMel/HuBERT 各双种子；每次24,844条评分0失败 | 未知攻击、跨来源、压缩及端到端离线路径验收 |
| 通用音频 | 19,215条Urban配对媒体；AST12层dev EER1.39%；冻结holdout已见1.43%/未见6.51%，分别2,880/1,728条0失败 | 跨来源覆盖有限，音乐及混合音频待补 |
| 视频模型 | 冻结 ResNet18 帧头双种子 EER20.55%/21.92%；Hunyuan train618/dev146 | 跨来源/生成器验收；主干微调和轻量时序尚未实现 |
| 鲁棒性 | AAC、带宽退化、H.264 的配对媒体工具与分数比较入口 | 只有小样本转码试跑，没有模型 EER 退化结果 |
| 离线交付 | 音视频研究权重、本地推理/恢复、导出打包；24条语音CPU断网演练 | 候选选择、双分支代表性批量验收、正式平台 I/O 适配 |

**现阶段最重要的成果缺口是通用音频、候选泛化及双分支离线验收。**
检查通过、准备工具完成和数据可解码，均不计为算法效果完成。

## 当前协作任务

| 任务 | 状态 | 下一份交付 |
| --- | --- | --- |
| [音频泛化 #19](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/19) | 分配给 lanjianyu060915-web | 固定语音候选的压缩退化与未知攻击实际分数 |
| [视频泛化 #20](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/20) | 分配给 rrx59 | 新生成器开发诊断和H.264配对退化 |
| [视频微调 #21](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/21) | 分配给 lanjianyu060915-web | 同划分ResNet18部分主干微调、实际EER及离线推理 |
| 环境音 AST 扩展 | PR #22 已合并；新增四组训练、本地开发诊断与两条件冻结holdout验收完成，独立分支交付 | 跨来源混合训练、音乐及混合音频 |

音频产物在 `/var/tmp/lanjianyu-issue9-20261011/`，
视频产物在 `/var/tmp/lanjianyu-issue10-20261011/`；两轮基线均已结束。
大权重、逐样本分数及缓存留在团队可读任务目录。
AST 预训练资源、微调权重及分数在 `/var/tmp/deepfake-audio-ast-20261011/`。
多生成器数据、微调与分数在 `/var/tmp/deepfake-audio-expand-20261011/`。
RTX4090按剩余显存与负载多人并发，任务在Issue记录占用，不设个人独占。

## 已有模型结论

| 本轮模型 | seed42 dev EER | seed43 dev EER |
| --- | ---: | ---: |
| LogMel | 0.784929% | 0.941915% |
| 冻结 HuBERT | 1.498027% | 1.530612% |
| 受控 ResNet18 帧头 | 20.547945% | 21.917808% |
| 环境音 LogMel（独立小型研究集） | 45.714286% | 37.142857% |
| 环境音 HuBERT（本地FP32端到端） | 28.571429% | 34.285714% |
| 环境音 AST 末四层微调（本地FP32端到端） | 2.857143% | 5.714286% |
| Urban 多生成器 AST 末四层，lr1e-5 | 5.545927% | 6.239168% |
| Urban 多生成器 AST 12层，lr1e-5 | 1.386482% | 未运行 |
| Urban 多生成器 AST 12层，lr3e-6 | 2.209705% | 未运行 |

音频使用 train25,379/dev24,844，两种子 LogMel 均优于冻结 HuBERT，
本轮保留 LogMel42 为较优研究候选。HuBERT 的 BF16 编码缓存评分不等于离线 FP32 编码评测。
这是已知 A01–A06 语音开发集；未知攻击、非语音和最终 holdout 未验收。
详见[云端音频实验](experiments/asv-cloud-20261011/README.md)。

环境音仅train322/dev70，HuBERT的BF16缓存EER25.71%/31.43%，不能替代上表FP32端到端数字。
两个种子在相同研究dev上均优于本轮LogMel，但没有证明跨生成器或音乐能力。
降低LogMel学习率并延长早停没有改善本轮结果。
详见[环境音模型对照](experiments/nonspeech-logmel-20261011/README.md)。

AST 比较两档学习率，seed42 均2.8571%；仅训练末层LayerNorm和头的对照为17.1429%。
本轮保留 lr1e-5/seed42，先固定权重，再对此前未使用的70条holdout评估一次：EER11.4286%，0失败。
这只冻结环境音研究候选，不是比赛最终候选；本轮不据holdout调参。
dev经过选模且样本少，不能以2.86%代表泛化；后续扩展已开展，见下段。
详见[AST微调实验](experiments/audio-ast-20261011/README.md)。

新 Urban 数据为train13,450/dev2,885/holdout2,880，238个原录音/上传者连通组，
排除与旧研究集的已知关联；三种生成器、四条生成路径参与训练。
旧 AST 直接迁移此 dev EER33.62%；新12层 lr1e-5 候选为1.39%。
未进入训练的 AudioLCM/TangoFlux dev 合计1,731条，EER5.03%，分别4.85%/5.20%；
末四层双种子未见生成器为11.61%/12.65%。所有上述本地FP32评分均0失败。
四组训练已完成；12层仅运行seed42，未声称双种子稳定性。低学习率对照没有改善。
固定12层 lr1e-5/seed42后验收：已见holdout EER1.4323%（2,880条），
未见生成器holdout EER6.5104%（1,728条），均0失败，本轮不据holdout调参。
该候选迁移旧FSD50K/Stable Audio Open dev EER8.5714%（70条0失败），
旧集原录音/上传者关系已从新训练排除，旧holdout不参与本轮。
这支持环境音的多生成器及有限跨来源迁移，仍不能代表音乐或混合音频能力。
详见[多生成器实验](experiments/audio-envsdd-20261011/README.md)。

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

环境音已完成多生成器数据扩展并取得迁移收益；下一轮优先补跨来源、音乐与混合音频。
三项队友任务并行推进语音泛化、视频泛化和同划分部分微调，再决定增强与时序投入。
候选冻结后完成双分支离线验收。具体依赖、交付物和取舍见[路线图](roadmap.md)。
