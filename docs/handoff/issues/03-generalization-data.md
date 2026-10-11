# [数据] 接入非语音数据并建立跨来源与生成器泛化协议

## 目标与依据

补齐语音与Hunyuan单生成器实验无法覆盖的通用音频/跨域缺口，至少交付一个许可和真假标签可信、能够接入现有训练接口的非语音来源。本任务以数据与协议为主，可与两个模型任务并行，不依赖它们先跑完训练。

依据：[数据与方法调研](https://github.com/lanjianyu060915-web/deepfake-detection-archive/blob/main/docs/research/literature.md)、[源登记模板](https://github.com/lanjianyu060915-web/deepfake-detection-archive/blob/main/configs/sources.example.jsonl)、[样本清单模板](https://github.com/lanjianyu060915-web/deepfake-detection-archive/blob/main/configs/manifest.example.jsonl)。

## 已有共享资产

- `/srv/shared-data/datasets/fakesound/FakeSound.zip`（约986MB、ZIP目录可读1909项），同目录有 `provenance.json`；来源登记暂无参考哈希。当前仅明确为fake-only诊断数据，不能单独训练二分类、计算真假EER或选择阈值；ZIP项数不等于样本数。
- ASV Parquet和ViF ZIP由音频/视频任务分别接入，本任务不重复物化或重建它们的训练清单。
- `/srv/shared-data/downloads/` 有ASV2019官方原包、ASV2021 LA eval、In-The-Wild等下载任务；开始前检查实际进程、日志和文件完整性。未完成文件不计数据量，不复制下载。ASV eval不进入训练/早停/阈值选择，最终留出未冻结前不开展评分。

## 工作范围

- 先核对FakeSound实际样本数、内容类型、来源版本、许可、派生关系和解码覆盖，登记可追溯清单；若只有fake，先交付诊断清单，并寻找一个合法且条件可匹配的真实来源，或选择一个已有可信真假两类的非语音数据集。
- 优先接入一个可用来源，避免一次收集多个大数据集；对real/fake做对称处理，检查来源、尺寸/采样率/编码等标签捷径。来源与许可无法闭合时换来源，不把EnvSDD/Urban未完成归属问题变成前置门。
- 复用现有source/manifest、audit、split工具和训练入口，按已知原媒体/派生/身份及确认重复组隔离。提交train/dev/holdout划分与来源/生成器留出设计；未知信息保留unknown。
- 给出视频多生成器的共同支持与来源覆盖缺口，优先提供一个实际可用、许可明确、可与现有真实视频匹配的候选入口；本Issue首个可完成交付仍是一个非语音来源，不要求同时建设完整音视频大基准。
- 只新增实际数据格式需要的导入工具、配置和测试。已有完成下载先复用；若取得新公开数据，登记固定版本、许可、大小和必要哈希，避免无关依赖及基础设施。

主要负责新的通用数据来源/导入工具、数据统计与分组协议；不同时修改音频/视频任务的同一导入文件，也不直接修改其冻结清单。向模型负责人提供可读取的输入和建议，由模型任务决定后续训练使用。

## 验收

- [ ] 已有FakeSound用途、实际覆盖和限制明确；单类数据未被当作二分类训练集或有效EER。
- [ ] 至少一个非语音来源具备可用媒体、可信真假标签、来源/许可、实际解码统计及可读清单，能够接入现有训练接口；仅列下载链接不算完成。
- [ ] train/dev/holdout与跨来源/生成器协议明确，已知组不跨分，避免来源成为标签代理。
- [ ] PR包含来源登记、必要导入代码/配置与行为验证、数据统计、团队可读取的产物路径及必要哈希。无需为本Issue跑完整模型训练。

不购买或发布数据、不修改共享原始资产，不把来源研究直接当成比赛使用许可。缺类、许可或实际下载阻塞应具体报告，不能以虚构标签或空集结果完成验收。

## 环境与协作约定

当前服务器为 RTX 4090 24GiB，Python 3.12.15、torch 2.14.1+cu132；共享环境 `/opt/venvs/deepfake-cu132` 已可用，FFmpeg/ffprobe 已全局安装到 `/usr/local/bin`。复用现有依赖，不重新安装环境或修改驱动。

迁移修复已通过 PR #12 合并。在自己的 clone/worktree 中，基于最新 `main` 建立 `issue/<本Issue编号>-<主题>`。在仓库根目录使用统一入口：

```bash
scripts/devctl run python -m deepfake_detection_archive --help
# 代码变更先跑相关测试，PR前完成一次标准检查
scripts/devctl check
```

`/srv/shared-data/datasets` 为共享只读输入，目录/文件权限允许队友读取。派生媒体、清单、outputs/cache 使用自己可写的任务目录；需要把产物发布到 root 管理的共享目录时交由维护者处理，不因权限问题重下原包或放宽共享原媒体权限。大媒体、权重和完整分数不入 Git，PR 给出团队可读取的路径和必要哈希。

三条任务可独立进行数据准备和开发；只有一张 GPU，运行 GPU profile、特征提取或训练前在 Issue 记录预计占用时段，并与另一个模型任务错开。只修改本方向需要的代码；涉及公共训练器/编码器或依赖变更时先协调，避免同时修改。提交 LoRA 风格的小 PR，关联本 Issue，由负责人审查合并；不自行合并、扩容或正式参赛提交。

领取时在本 Issue 留下负责人、分支和第一步；未分派姓名，按方向领取。
