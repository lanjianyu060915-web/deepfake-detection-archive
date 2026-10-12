# 项目状态

更新：2026-10-12。训练、推理、EER评测和打包入口已具备，当前在改进候选泛化，
**研究双分支已接通，尚未完成可交付的比赛系统**。后续顺序见[路线图](roadmap.md)，
全部实验见[实验索引](experiments/README.md)，资产位置见[数据说明](data.md)。

## 当前候选与缺口

| 方向 | 已有结果 | 当前选择与缺口 |
| --- | --- | --- |
| 语音 | 带宽增强seed42 clean EER0.90%、8kHz带限1.39%、A07–A19子集2.70%；原基线clean0.78%、带限13.10% | 保留增强研究候选；第二种子与跨来源尚未完成；[实验](experiments/bandwidth-augmentation-20261012/README.md) |
| 环境音 | Urban AST12层seed42 dev1.39%；冻结研究候选holdout已见1.43%、未见生成器6.51% | 保留冻结环境音候选；不能代表音乐或通用音频；[实验](experiments/audio-envsdd-20261011/README.md) |
| 音乐/通用音频 | 扩展AIME无增强候选未训练音乐18.75%；事后等权分数融合14.58% | 融合尚未接离线推理，跨来源和真实混合内容仍弱；[实验](experiments/audio-ast-regularization-20261012/README.md) |
| 视频 | layer4+头同域dev16.44%；H.264增强clean15.07%/CRF35 17.81%，尺寸匹配跨生成器30.77%→32.58% | 保留原微调候选；增强仅保留同域压缩用途，尺寸/时长混杂仍有限制；[实验](experiments/video-size-control-20261012/README.md) |
| 离线系统 | AST/微调视频496/336条批量0失败；源码与依赖随包，30条真实样例隔离迁移启动 | 研究包已接通；最终候选、独立验收、许可与官方I/O仍未完成；[使用](experiments/offline-dual-branch-20261012/README.md) |

以上不是比赛成绩。音乐原曲/artist关系未知，广播片段标签纯度及底层许可有限；
多轮使用的dev不能充当独立验收。正式交付资产仍须满足[比赛要求](competition/requirements.md)。

## 正在推进的主线

| 任务 | 本次已知状态 | 下一交付 |
| --- | --- | --- |
| [语音带宽增强 #29](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/29) | 已交付单种子增强训练与未知攻击诊断 | 第二种子及独立跨来源泛化 |
| [视频尺寸控制与压缩增强 #30](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/30) | 已交付尺寸匹配与压缩增强；跨生成器未改善 | 更丰富训练来源，继续控制尺寸/时长捷径 |
| [双分支离线运行包 #31](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/31) | 已交付可搬移研究包及双分支批量结果 | 随最终候选更新；官方I/O仍待适配 |
| [通用音频 #32](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/32) | 扩展与时频遮挡训练结束；混合结果随#35待合入 | 独立音乐来源与真实混合内容，保留环境音能力 |

具体认领和进程状态在Issue更新。本页不逐轮追加运行日志或历史结果。
RTX4090按实际负载共享；数据、权重与分数按dlteam读取，任务输出各自独立。
开发环境可用，日常无需重建安装或验证记录。
