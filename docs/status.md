# 项目状态

更新：2026-10-12。训练、推理、EER评测和打包入口已具备，当前在改进候选泛化，
**尚未完成可交付的双分支比赛系统**。后续顺序见[路线图](roadmap.md)，
全部实验见[实验索引](experiments/README.md)，资产位置见[数据说明](data.md)。

## 当前候选与缺口

| 方向 | 已有结果 | 当前选择与缺口 |
| --- | --- | --- |
| 语音 | ASV LogMel双种子dev EER0.7849%/0.9419%；seed42在A07–A19子集2.97%、8kHz带限13.10% | 保留LogMel42研究候选；带宽退化和跨来源泛化待改进；[实验](experiments/audio-generalization-20261011/README.md) |
| 环境音 | Urban AST12层seed42 dev1.39%；冻结研究候选holdout已见1.43%、未见生成器6.51% | 保留冻结环境音候选；不能代表音乐或通用音频；[实验](experiments/audio-envsdd-20261011/README.md) |
| 音乐/通用音频 | 扩展AIME无增强候选未训练音乐18.75%；事后等权分数融合14.58% | 融合尚未接离线推理，跨来源和真实混合内容仍弱；[实验](experiments/audio-ast-regularization-20261012/README.md) |
| 音乐/语音混合 | 匹配对照中未训练路径合成混合32.55%→26.89%；真实广播混合仍50%，环境音退化 | 只保留研究结果，不替换主候选；优先独立来源与真实混合数据；[实验](experiments/audio-speech-mix-20261012/README.md) |
| 视频 | ResNet18解冻layer4与头，双种子受控dev均16.44%（146条） | 仅同来源结果；旧冻结头ViF-Bench为33.28%且存在尺寸混杂，微调候选泛化待验收；[实验](experiments/video-finetune-20261011/README.md) |
| 离线系统 | 本地逐样本评分、恢复、导出与打包已实现；24条语音CPU断网演练完成 | 双分支代表性批量验收、最终候选选择和官方I/O适配未完成 |

以上不是比赛成绩。合成混合单种子、音乐原曲/artist关系未知，广播片段标签纯度及底层许可有限；
多轮使用的dev不能充当独立验收。正式交付资产仍须满足[比赛要求](competition/requirements.md)。

## 正在推进的主线

| 任务 | 本次已知状态 | 下一交付 |
| --- | --- | --- |
| [语音带宽增强 #29](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/29) | 已认领，尚无结果PR合入main | clean/带限/未知攻击对照与候选取舍 |
| [视频尺寸控制与压缩增强 #30](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/30) | 已认领，尚无结果PR合入main | 尺寸受控的跨生成器结果及压缩增强训练 |
| [双分支离线运行包 #31](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/31) | 尚无交付结果 | AST与微调视频接入、代表性断网批量运行 |
| [通用音频 #32](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/32) | 扩展、时频遮挡及匹配混合训练已结束 | 独立音乐来源与真实混合内容，保留环境音能力 |

具体认领和进程状态在Issue更新。本页不逐轮追加运行日志或历史结果。
RTX4090按实际负载共享；数据、权重与分数按dlteam读取，任务输出各自独立。
开发环境可用，日常无需重建安装或验证记录。
