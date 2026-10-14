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
| 冻结CLIP视频头 | 双种子dev24.88%/24.64%；冻结seed43统一1949条26.56%、未训练标签27.88%、CRF35 25.84% | 优于七生成器ResNet，原同域dev20.55%仍退化；新来源受控dev1880条31.33%（ResNet39.00%），8帧31.80%无收益、保留4帧，作为后续视频研究主候选；母媒体独立性及许可未确认，不替换release；[新来源实验](experiments/video-aigvdbench-20261014/README.md) |
| 离线系统 | 统一入口RC包搬移、CPU/GPU演示及440条profile成功；真实唯一语音10,000/10,000、零失败，端到端1,800.67秒；中断恢复20/20 | 仍用原研究默认，未重发Release；可用已登记视频不足万条，新环境/完整容量/官方I/O及赛用资源仍待完成；[实验](experiments/offline-entry-20261014/README.md) |

以上不是比赛成绩。music-aime96、speech-asv200来自当前dev；未训练音乐视图共享既有dev
真样本与prompt组，原曲/artist独立性未知。AI-OpenBMAT有标签纯度与底层许可限制。
In-the-Wild及多轮使用的dev已见，不能再作为未来盲测；正式交付仍须满足[比赛要求](competition/requirements.md)。

## 候选状态（2026-10-14 核对）

按产物核对，不以旧评论或 `running` 字段为准；权重哈希与持久副本见[数据说明](data.md#持久归档)。
“复核”指他人重跑结果，不等于独立来源泛化。#51已交首批视频dev，受控组四候选完整比较已完成；
与旧Panda训练/选择集的YouTube母媒体交集未知，最终保留评价尚未进行。

| 候选 | 训练/评分 | 选定 | 复核 | 合并 | 发布 | 权重可恢复 | 标准入口加载 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0.1.1 环境音AST + 视频ResNet18 | 完成 | 发布默认 | release复核、CPU/GPU演示 | #42、#44 | v0.1.1 | 是 | 发布包可用 |
| 三域联合AST seed42主/seed43备 | 完成 | 研究主候选 | #56复核 | #45、#56 | 否 | 是 | 发布包`infer`可加载，归档副本80条分数逐位一致 |
| LogMel带宽 seed42冻结/seed43 | 完成 | 语音对照 | seed43经#55复核，seed42仅自评 | #37、#55 | 否 | 是 | 仓库源码入口 |
| 七生成器视频 paired-seed43 | 完成 | 研究候选 | #48复核 | #48 | 否 | 是 | 发布包不兼容（`temporal_kernel_size`），仓库源码入口20条最大差4.9e-3 |
| 视频时序头 seed43 | 完成 | 研究候选 | 复核目录团队不可读 | #43 | 否 | 是（原权重root 0600，归档副本可读） | 未演练 |
| 冻结CLIP视频头 seed43 | 完成 | 后续研究主候选 | 未独立重跑完整集 | #61、#64 | 否 | 是（C6） | 标准infer可加载；C6两条恢复分数一致 |

所有候选的正式赛用许可均未确认。三域AST两种子与时序头为脏工作区产物，LogMel seed43未记录commit
（实际为#55分支`2fabb61`），代码以 `code_sha256` 追溯。

## 后续主线

#39 的第二种子/噪声对照、#40 的多生成器训练、#41 的双种子联合训练及冻结诊断已交付。
新的比赛主线由 [#49](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/49) 汇总，
具体认领、进程和PR在Issue维护：

- 独立来源数据与评测隔离：[#51](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/51)。
- 更强视频基线：[#52](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/52)；语音SSL对照：[#53](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/53)。
- 统一媒体预测入口：[#57](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/57)已完成，经#62/#64合入main；实际交付由#54继续。
- 共享资产持久保存：[#50](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/50)已完成；离线兼容与容量：[#54](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/54)。

RTX4090按实际负载共享，任务输出独立。联合音频候选位于
`/srv/shared-data/candidates/audio-universal-20261014/`，文件只读、团队可读取；
该路径仍为当前实例共享副本；最小候选集合已另存[持久归档](data.md#持久归档)，
跨实例恢复尚未演练。日常开发无需重建环境或安装记录。
