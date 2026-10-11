# 项目状态

更新：2026-10-11。**音视频基线和环境音 AST 部分微调已交付**，尚未完成双分支比赛系统。
本页只记录已经交付的结果与正在推进的任务；计划见[路线图](roadmap.md)。

## 已完成与未完成

| 模块 | 已完成 | 未完成 |
| --- | --- | --- |
| 数据/评测 | 标准 manifest、来源登记、组级划分、EER 与失败覆盖率 | 比赛数据及官方评测程序尚未提供 |
| 语音模型 | 同一 ASV train/dev 上 LogMel/HuBERT 各双种子；每次24,844条评分0失败 | A07–A19研究子集与AAC/带宽诊断已完成；跨来源及最终离线验收待做 |
| 通用音频 | 环境音AST末四层微调：离线dev EER2.86%/5.71%；冻结研究候选holdout EER11.43%，各70条0失败 | 音乐/混合音频、多生成器及跨来源覆盖不足 |
| 视频模型 | 冻结 ResNet18 帧头双种子 EER20.55%/21.92%；Hunyuan train618/dev146 | 跨来源/生成器验收；主干微调和轻量时序尚未实现 |
| 鲁棒性 | AAC、带宽退化、H.264 的配对媒体工具与分数比较入口 | 语音LogMel全dev：AAC退化+0.20/+0.99点，8kHz采样率带限约+12点；视频评分待做 |
| 离线交付 | 音视频研究权重、本地推理/恢复、导出打包；24条语音CPU断网演练 | 候选选择、双分支代表性批量验收、正式平台 I/O 适配 |

**现阶段最重要的成果缺口是通用音频、候选泛化及双分支离线验收。**
检查通过、准备工具完成和数据可解码，均不计为算法效果完成。

## 当前协作任务

| 任务 | 状态 | 下一份交付 |
| --- | --- | --- |
| [音频泛化 #19](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/19) | #24交付AAC/带宽退化及A07–A19研究子集，原权重固定 | 针对带宽弱点训练；跨来源及未知攻击叠加退化待做 |
| [视频泛化 #20](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/20) | 分配给 rrx59 | 新生成器开发诊断和H.264配对退化 |
| [视频微调 #21](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/21) | 分配给 lanjianyu060915-web | 同划分ResNet18部分主干微调、实际EER及离线推理 |
| 环境音 AST 微调 | 四次训练、双种子本地dev推理、一次冻结候选holdout评估完成，更新PR #22 | 补独立生成器/来源，保留AST部分微调候选 |

音频产物在 `/var/tmp/lanjianyu-issue9-20261011/`，
视频产物在 `/var/tmp/lanjianyu-issue10-20261011/`；两轮基线均已结束。
大权重、逐样本分数及缓存留在团队可读任务目录。
AST 预训练资源、微调权重及分数在 `/var/tmp/deepfake-audio-ast-20261011/`。
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

音频使用 train25,379/dev24,844，两种子 LogMel 均优于冻结 HuBERT，
本轮保留 LogMel42 为较优研究候选。HuBERT 的 BF16 编码缓存评分不等于离线 FP32 编码评测。
这是已知 A01–A06 语音开发集。后续固定权重诊断覆盖A07–A19研究子集12,000条，
LogMel42/43 EER2.9712%/4.1442%，HuBERT均6.0096%；全dev AAC及8kHz采样率带限均0失败。
带限后LogMel EER13.10%/13.18%，是下一训练弱点；攻击与说话人同时改变，不能分离归因。
跨来源及最终holdout仍未验收，详见[语音泛化实验](experiments/audio-generalization-20261011/README.md)。
详见[云端音频实验](experiments/asv-cloud-20261011/README.md)。

环境音仅train322/dev70，HuBERT的BF16缓存EER25.71%/31.43%，不能替代上表FP32端到端数字。
两个种子在相同研究dev上均优于本轮LogMel，但没有证明跨生成器或音乐能力。
降低LogMel学习率并延长早停没有改善本轮结果。
详见[环境音模型对照](experiments/nonspeech-logmel-20261011/README.md)。

AST 比较两档学习率，seed42 均2.8571%；仅训练末层LayerNorm和头的对照为17.1429%。
本轮保留 lr1e-5/seed42，先固定权重，再对此前未使用的70条holdout评估一次：EER11.4286%，0失败。
这只冻结环境音研究候选，不是比赛最终候选；本轮不据holdout调参。
dev经过选模且样本少，不能以2.86%代表泛化；下一轮补独立来源和生成器。
详见[AST微调实验](experiments/audio-ast-20261011/README.md)。

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

环境音AST部分微调已取得低于既有HuBERT的同集dev EER，下一轮补独立生成器/来源。
语音泛化诊断已交付；视频泛化和同划分部分微调继续并行，语音下一轮优先带宽增强。
候选冻结后完成双分支离线验收。具体依赖、交付物和取舍见[路线图](roadmap.md)。
