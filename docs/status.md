# 项目状态

更新：2026-10-14。**v0.1.1 是可运行的环境音/视频离线研究基线**：
带本地权重、CPython 与依赖，无标签目录→推理→连续分数导出，支持失败记账与恢复。
[下载与使用](release.md)。三域联合音频研究候选现已完成，尚未替换发布包；
独立来源泛化、官方格式、赛用许可及万级真实容量验收仍有缺口。
后续见[路线图](roadmap.md)，全部实验见[实验索引](experiments/README.md)，资产见[数据说明](data.md)。

## 当前候选与缺口

| 方向 | 已有结果 | 当前选择与缺口 |
| --- | --- | --- |
| 语音 | 带宽seed42/43 clean 0.90%/1.08%、8kHz带限1.39%/1.04%；In-the-Wild 30.67%/31.19% | 保留带宽seed42；白噪声双种子不稳定已停止；真实网络语音仍弱；[实验](experiments/speech-robust-20261013/README.md) |
| 环境音 | Urban AST12层seed42 dev1.39%；冻结研究候选holdout已见1.43%、未见生成器6.51% | 保留环境音对照；不能代表音乐或通用音频；[实验](experiments/audio-envsdd-20261011/README.md) |
| 音乐 | 扩展AIME无增强候选未训练路径18.75%；事后等权分数融合14.58% | 融合仅为dev探索，未接离线推理；跨来源仍弱；[实验](experiments/audio-ast-regularization-20261012/README.md) |
| 音乐/语音混合 | 匹配对照中未训练路径合成混合32.55%→26.89%；真实广播混合仍50%、环境音退化 | 保留负结果，不替换主候选；[实验](experiments/audio-speech-mix-20261012/README.md) |
| 三域联合音频 | 等权双种子dev macro 0.70%/0.81%、pooled 1.14%/0.87%；主候选冻结holdout fsd8.57%、音乐2.08%；In-the-Wild24.86%；未训练音乐25.00%、广播混合51.32% | 冻结seed42为主、seed43备选，权重/配置/完整分数团队可读；泛化与许可尚不足以替换release；[实验](experiments/universal-audio-20261012/README.md) |
| 视频 | 七生成器train3,614/dev844，四轮训练；统一1949条EER33.12%→31.73%，未训练标签33.47%→33.14% | 原同域dev16.44%→19.18%，保留研究候选、不替换v0.1.1；仍缺独立来源泛化与更强视觉表征；[实验](experiments/video-multigen-20261013/README.md) |
| 强视频公开权重对照 | UFD完整1949条47.75%、未训练标签50.00%、旧dev49.32%、CRF35 58.43%；共2405条全覆盖、零失败 | 作者数值已对齐；停止直接采用公开检测头，不替换现候选，继续有限适配与独立来源比较；[实验](experiments/video-ufd-20261014/README.md) |
| 冻结CLIP视频头 | 双种子dev24.88%/24.64%；冻结seed43统一1949条26.56%、未训练标签27.88%、CRF35 25.84% | 优于七生成器ResNet的31.73%/33.14%/32.13%，原同域dev20.55%仍退化；保留下一轮研究候选，独立来源与许可未确认，不替换release；[实验](experiments/video-clip-head-20261014/README.md) |
| 离线系统 | 既有研究包496/336条批量0失败；v0.1.1 CPU/GPU演示及18音频+12视频0失败；联合音频新增两候选各400条标准CLI评分0失败 | 新候选可本地加载，发布包仍为旧候选；官方I/O、赛用资源及两项万级容量尚待完成；[使用](release.md) |

以上不是比赛成绩。music-aime96、speech-asv200来自当前dev；未训练音乐视图共享既有dev
真样本与prompt组，原曲/artist独立性未知。AI-OpenBMAT有标签纯度与底层许可限制。
In-the-Wild及多轮使用的dev已见，不能再作为未来盲测；正式交付仍须满足[比赛要求](competition/requirements.md)。

## 后续主线

#39 的第二种子/噪声对照、#40 的多生成器训练、#41 的双种子联合训练及冻结诊断已交付。
新的比赛主线由 [#49](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/49) 汇总，
具体认领、进程和PR在Issue维护：

- 独立来源数据与评测隔离：[#51](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/51)。
- 更强视频基线：[#52](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/52)；语音SSL对照：[#53](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/53)。
- 统一媒体预测入口：[#57](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/57)，由maochuanjun在视频首轮对照后推进。
- 共享资产持久保存：[#50](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/50)；离线兼容与容量：[#54](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/54)。

RTX4090按实际负载共享，任务输出独立。联合音频候选位于
`/srv/shared-data/candidates/audio-universal-20261014/`，文件只读、团队可读取；
该路径仍在当前实例文件系统，跨实例持久备份尚未确认。日常开发无需重建环境或安装记录。
