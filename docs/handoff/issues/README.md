# 云端 Issue 与领取入口

2026-10-10已按用户授权通过`gh`发布三个独立任务，正文保存在对应Markdown。
三条任务分别承担本方向的数据接入、开发和验收，代码/数据准备可并行，不等待统一环境Issue。
队友已开始推进，领取和进度以各 Issue 为准；本地文件前缀不是GitHub编号。

| 优先级 | GitHub Issue | 正文 | 建议方向 | 交付 |
| --- | --- | --- | --- | --- |
| P1 | [#9](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/9) | [音频](01-audio-comparison.md) | 音频队友 | ASV接入、LogMel43/HuBERT双种子同划分结果 |
| P1 | [#10](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/10) | [视频](02-video-control.md) | 视频队友 | ViF接入、holdout隔离、受控视觉双种子结果 |
| P1 | [#11](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/11) | [数据与泛化](03-generalization-data.md) | 数据队友 | 至少一个可用非语音来源及跨域协议 |

共享环境及全局FFmpeg已可用，剩余资产接入已拆入各任务。
单GPU的profile、特征提取和训练在Issue中约定时段，避免同时满载；只读原数据复用，
派生清单、outputs/cache各任务独立。

迁移修复已通过 [PR #12](https://github.com/lanjianyu060915-web/deepfake-detection-archive/pull/12)
合并到`main`，新任务从最新`main`建立分支；
开发命令为`scripts/devctl run ...`和`scripts/devctl check`。角色、职责、验收与PR要求以各Issue为准。

[离线交付草稿](04-offline-delivery.md)保留待后续发布，启动条件是音频/视频真实候选完成且
由负责人冻结；当前不安排空跑验收。不以训练续跑系统、重复审计或统计证明阻止探索实验。
负责人按[协作说明](../cloud-collaboration.md#4-issue--分支--pr--审查--合并)审查并合并。
