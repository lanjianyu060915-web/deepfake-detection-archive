# AIGC 音视频伪造识别

第七届中国人工智能大赛 AIGC 音视频伪造识别项目，同时参加音频和视频两项。项目输出
逐样本连续伪造分数，以 EER 选择模型，并面向安全屋离线交付。

**当前阶段：全量语音LogMel已完成，开发集EER为0.51%，后续工作转到云端。**
这是已知生成器语音开发集结果，尚无本届比赛成绩。HuBERT和受控视觉模型尚未训练；
视频已准备好尺寸/时长控制数据。负责人发布Issue，队友完成分支/PR，由负责人审查合并。
新云端Codex先读[个人交接](docs/handoff/personal-cloud-codex.md)。

项目只保留影响实验真实性和离线运行的必要检查。配置在入口解析，内部直接传递；每轮实验
保存一份配置、结果和必要日志，不再用重复校验、历史绑定或多份证明代替算法进展。

## 入口

| 文档 | 内容 |
| --- | --- |
| [云端交接](docs/handoff/personal-cloud-codex.md) | 个人交接Prompt、协作流程、资产与首批Issue |
| [项目状态](docs/status.md) | 当前数据、模型结果、验证和缺口 |
| [实施方案](docs/plans/implementation-plan.md) | 音频/视频路线、阶段和下一步 |
| [开发指南](docs/development.md) | data → train → infer → evaluate → package |
| [比赛要求](docs/competition/requirements.md) | 已知官方事实和集中登记的未知项 |
| [文献与参考](docs/research/literature.md) | 候选方法、数据和适用边界 |
| [工具与 skills](docs/research/tooling.md) | 环境、固定参考和项目 skills |
| [历史归档](archive/README.md) | 朋友旧照片项目，默认只读 |

用户已报名，官方没有提供额外资料。项目按公开赛题和本地约定继续开发，不再把未公布的
I/O、硬件或外部资产细则作为研究训练的阻塞项；平台提供样例后只更新适配层。

## 快速开始

Python 3.12，依赖由 `uv.lock` 固定。CUDA后端锁定为cu130；云端先确认实例驱动兼容，见[环境与资产](docs/handoff/cloud-collaboration.md)：

```bash
export UV_CACHE_DIR=/tmp/aigc-uv-cache
uv sync --locked --group media --group research --extra cu130
uv run --no-sync deepfake-detection-archive --help
```

无 GPU 环境改用 `--extra cpu`。CPU 与 cu130 后端不能同时选择。FFmpeg 和 ffprobe 使用
宿主可执行文件。

当前主线命令：

```bash
# 建立媒体清单
uv run --no-sync deepfake-detection-archive index \
  --root data/audio --modality audio --source YOUR_SOURCE \
  --output data/audio-index.jsonl

# 训练、推理与评测：先从 configs/*.example.json 复制并填写实值配置
uv run --no-sync deepfake-detection-archive train \
  --config configs/train-logmel.local.json
uv run --no-sync deepfake-detection-archive infer \
  --config configs/infer-audio-waveform.local.json
uv run --no-sync deepfake-detection-archive evaluate \
  --scores outputs/audio-dev-scores/scores.jsonl \
  --manifest data/audio-dev-only.jsonl \
  --output outputs/audio-dev-scores/evaluation.json
```

这些命令的完整前置条件、数据字段、推理恢复和打包步骤见[开发指南](docs/development.md)。
配置模板不是可直接运行的实验；必须替换占位路径、哈希、清单和输出目录。

## 当前方法

- 音频：已取得全量 LogMel 基线，下一步比较冻结 HuBERT 表征；AASIST-L 保留为对照，再补环境声、
  音乐和音效等通用音频。
- 视频：先控制宽度、来源和编码捷径，再比较主干微调和轻量时间特征；当前冻结帧特征结果
  不能证明已学到可靠生成伪迹。
- 数据：同一原媒体、派生样本和已知身份组不能跨 split；dev 用于模型选择，holdout 在候选
  冻结后验收。
- 评测：内部 `real=0`、`fake=1`，高分更倾向伪造；失败样本保留记录并计入覆盖率。

## 工作区

```text
src/deepfake_detection_archive/  当前比赛代码
tests/                            主动行为与回归测试
configs/                          数据、训练、推理和打包模板
docs/                             当前说明、研究和实验记录
.agents/skills/                   Git 跟踪的 11 项项目 skills
archive/legacy-20261005/          冻结的旧照片项目
data/ models/ outputs/ cache/     Git 忽略的本地数据与产物
external/                         Git 忽略的固定上游参考
```

旧项目的 UI、身份检索、底图溯源、Windows 打包和历史阈值不属于当前主线。归档中的 1,526 个
文件及恢复工具保持原字节，不参与当前包的 lint、类型检查和构建。

## 检查与协作

```bash
export UV_CACHE_DIR=/tmp/aigc-uv-cache
bash tools/check_project.sh
```

11 项项目 skills 位于 `.agents/skills/`，来源由 `skills-lock.json` 固定，协作者克隆后可直接
使用。Git 提交遵循 LoRA：Logical、Observable、Reversible、Atomic。仓库不会自动推送、
发布或正式参赛提交。
