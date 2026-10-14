# 项目状态

更新：2026-10-12。**v0.1.1 修复环境音/视频离线研究基线**：
带本地权重、CPython 与依赖，无标签目录→推理→连续分数导出，支持失败记账与恢复。
这是可运行系统，尚无覆盖三域的通用音频模型；官方格式与两项权重的赛用许可仍待确认。[下载与使用](release.md)。
后续顺序见[路线图](roadmap.md)，
预处理修正后的环境音模型在音乐/语音固定dev为22.50%/44.00%，通用音频仍是主线缺口。
全部实验见[实验索引](experiments/README.md)，资产位置见[数据说明](data.md)。

## 当前候选与缺口

| 方向 | 已有结果 | 当前选择与缺口 |
| --- | --- | --- |
| 语音 | 带宽增强seed42 clean EER0.90%、8kHz带限1.39%、A07–A19子集2.70%；原基线clean0.78%、带限13.10% | 保留增强研究候选；第二种子与跨来源尚未完成；[实验](experiments/bandwidth-augmentation-20261012/README.md) |
| 环境音 | Urban AST12层seed42 dev1.39%；冻结研究候选holdout已见1.43%、未见生成器6.51% | 保留冻结环境音候选；不能代表音乐或通用音频；[实验](experiments/audio-envsdd-20261011/README.md) |
| 音乐/通用音频 | 扩展AIME无增强候选未训练音乐18.75%；事后等权分数融合14.58% | 融合尚未接离线推理，跨来源和真实混合内容仍弱；[实验](experiments/audio-ast-regularization-20261012/README.md) |
| 通用音频（三域） | 1:1:1联合AST双种子dev三域平均0.70%/0.81%；music-aime96、speech-asv200均0%（发布版22.50%/44.00%）；holdout fsd8.57%、music2.08%；In-the-Wild 24.86% | 冻结seed42为主候选、seed43备选，可直接替换音频配置；许可未决仍为研究用途；[实验](experiments/universal-audio-20261012/README.md) |
| 音乐/语音混合 | 匹配对照中未训练路径合成混合32.55%→26.89%；真实广播混合仍50%，环境音退化 | 只保留研究结果，不替换主候选；优先独立来源与真实混合数据；[实验](experiments/audio-speech-mix-20261012/README.md) |
| 视频 | layer4+头同域dev16.44%；全1949条统一尺寸/帧率/3.5秒后EER33.12%，冻结头33.65% | 微调受控优势仅0.54点，仍缺稳定跨来源能力；[受控诊断](experiments/release-candidates-20261012/README.md) |
| 离线系统 | 既有研究包496/336条批量0失败；v0.1.1运行包CPU/GPU演示和18音频+12视频0失败；部分失败仍导出成功子集；OOM/缺依赖立即中止 | 预处理已对齐，依赖已按运行闭包裁剪；通用音频、官方I/O、赛用资源及万级真实容量验收仍未完成；[使用](release.md) |

以上不是比赛成绩。合成混合仅单种子、音乐原曲/artist关系未知，广播片段标签纯度及底层许可有限；
多轮使用的dev不能充当独立验收。正式交付资产仍须满足[比赛要求](competition/requirements.md)。

## 正在推进的主线

| 任务 | 本次已知状态 | 下一交付 |
| --- | --- | --- |
| [语音稳健训练 #39](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/39) | 已发布，lanjianyu060915-web负责 | 第二种子、编码/噪声训练与独立跨来源诊断 |
| [多生成器视频训练 #40](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/40) | 已发布，rrx59负责 | 扩展训练来源，保留固定跨生成器诊断与原候选对照 |
| [三域通用音频 #41](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/41) | zhl-zx负责；两档比例双种子完成，1:1:1已冻结并完成holdout | 加入真实场景语音与真实混合内容，替换发布包音频配置 |

具体认领和进程状态在Issue更新。本页不逐轮追加运行日志或历史结果。
RTX4090按实际负载共享；数据、权重与分数按dlteam读取，任务输出各自独立。
开发环境可用，日常无需重建安装或验证记录。
