# 项目状态

更新：2026-10-11。当前处于**数据接入与基线模型开发**，尚未完成双分支比赛系统。
本页只记录已经交付的结果与正在推进的任务；计划见[路线图](roadmap.md)。

## 已完成与未完成

| 模块 | 已完成 | 未完成 |
| --- | --- | --- |
| 数据/评测 | 标准 manifest、来源登记、组级划分、EER 与失败覆盖率 | 比赛数据及官方评测程序尚未提供 |
| 语音模型 | 全量 ASVspoof LA LogMel seed42，dev EER 0.510204%，24,844 条评分 0 失败 | 云端可比双种子、SSL 对照、未知攻击与跨来源验收 |
| 通用音频 | Doppelganger/FSD50K 环境音 231 对、462 条；train322/dev70/holdout70；实际训练解码可用 | 检测器尚未训练，音乐/混合音频及多生成器覆盖不足 |
| 视频模型 | 冻结 ResNet18 帧特征与头可训练；Hunyuan 受控划分 train618/dev146 | 受控模型结果未交付；主干微调和轻量时序尚未实现 |
| 鲁棒性 | AAC、带宽退化、H.264 的配对媒体工具与分数比较入口 | 只有小样本转码试跑，没有模型 EER 退化结果 |
| 离线交付 | 本地权重推理、恢复、导出、打包；24 条真实语音 CPU 断网演练 | 有效视频候选、双分支代表性批量验收、正式平台 I/O 适配 |

**现阶段最重要的成果缺口是有效的音视频候选及泛化结果。**
检查通过、准备工具完成和数据可解码，均不计为算法效果完成。

## 当前协作任务

| 任务 | 状态 | 下一份交付 |
| --- | --- | --- |
| [音频 #9](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/9) | lanjianyu060915-web 进行中；官方协议/媒体及 HuBERT 已接入，GPU 训练已启动 | 同一清单上 LogMel/HuBERT 各两种子结果、权重、dev 分数 |
| [视频 #10](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/10) | rrx59 负责；尚无结果 PR | 受控视觉双种子基线及后续模型选择依据 |
| [非语音数据 #11](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/11) | 已由 PR #14 合并 | 研究集交给音频模型使用；当前单来源/单生成器 |
| 鲁棒性与分数比较 | PR #13、#15 已合并 | 用已交付候选实际评分，报告 ΔEER 和覆盖率 |

#9 的 `/var/tmp/lanjianyu-issue9-20261011/` 已有媒体、配置和 HuBERT 资源。
在制 run 尚未交付最终结果；正式 EER 与权重以任务 PR 为准。
单 GPU 先完成 #9，#10 与其错开。

## 已有模型结论

全量 LogMel42 使用 train25,379/dev24,844，最佳第30轮，dev EER **0.510204%**；
这是 ASVspoof LA 已知 A01–A06 攻击开发集，尚无最终 holdout 或本届比赛成绩。
旧 checkpoint 尚未在云端恢复，#9 会在同条件下补跑 seed42，不能直接将旧数字当云端对照。
详见[全量语音实验](experiments/asv-main-20261009/README.md)。

历史小样本语音 LogMel42/43 为 21.48%/24.22%，AASIST-L 为 25.78%/29.69%；
与全量划分不同，不能直接当作提升幅度。视频冻结帧模型 pad 为 32.47%/32.47%，
crop 为 33.77%/31.17%；同集宽度基线为24.75%，暴露了元数据捷径，未选为正式候选。
受控 Hunyuan dev 的宽度/联合元数据 EER 为50.00%/53.42%，视觉内容模型仍待训练。

## 环境与资产

RTX 4090 24 GiB；Python 3.12.15、uv 0.13.0、torch 2.14.1+cu132。
共享环境 `/opt/venvs/deepfake-cu132` 可用，FFmpeg/ffprobe 全局入口在 `/usr/local/bin`。
环境已具备开发条件；无需再建设环境证明。数据、模型及许可位置见[数据说明](data.md)。

原始 ASV Parquet、ViF 和 FakeSound ZIP 位于 `/srv/shared-data/datasets/`。
非语音研究集在 `/var/tmp/lanjianyu-issue11-20261011/doppelganger-v2/`，团队可读；
FakeSound 缺少 real、许可 unresolved，保持单类诊断，不用于二分类 EER。
`/var/tmp` 不是已确认的持久卷，长期保留由维护者迁入平台持久化存储。

## 下一步

先收齐 #9、#10 两条基线，再用已经可用的环境音补通用音频；
根据受控视觉结果决定主干微调与时序，随后做未知生成器/来源和压缩留出。
候选冻结后完成双分支离线验收。具体依赖、交付物和取舍见[路线图](roadmap.md)。
