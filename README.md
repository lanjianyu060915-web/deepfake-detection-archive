# AIGC 音视频伪造识别

第七届中国人工智能大赛比赛项目，分别检测音频和视频中的生成内容。
输入本地媒体，输出逐样本连续伪造分数；以 EER 选择模型，最终离线运行。
内部约定 `real=0`、`fake=1`，分数越高越倾向伪造。

**当前阶段：基线开发，尚未完成双分支比赛系统。**

| 方向 | 已完成 | 当前缺口 |
| --- | --- | --- |
| 语音 | 云端 LogMel 双种子 EER **0.7849% / 0.9419%**；冻结 HuBERT 对照已完成 | 未知攻击、跨来源及压缩泛化待做 |
| 非语音 | 231 对真实/生成环境音，已接入训练接口 | 尚未训练；音乐、混合音频和其他生成器覆盖不足 |
| 视频 | 受控 train618/dev146；双种子 EER **20.55% / 21.92%** | 跨生成器泛化、微调和时序模型未完成 |
| 离线系统 | 音视频研究权重、训练/推理/评测及打包入口；24 条语音离线演练 | 候选选择、双分支代表性批量验收和正式平台适配未完成 |

以上是已知攻击语音和单生成器视频开发集结果，不能代表本届比赛或通用音视频效果。
[项目状态](docs/status.md)记录交付事实，[路线图](docs/roadmap.md)列出执行顺序与完成条件。

## 环境

Python 3.12、uv、FFmpeg/ffprobe。当前 RTX 4090 服务器共享环境已就绪：

```bash
scripts/devctl run python -m deepfake_detection_archive --help
```

新机器安装、CPU 环境和共享环境维护见[开发指南](docs/development.md#环境)。
日常运行不重新同步依赖或做全量环境检查。

## 数据 → 训练 → 推理 → 评测

[数据说明](docs/data.md)集中列出来源、实际资产和准备命令。
`configs/` 提供[配置模板](configs/README.md)，相对路径从配置文件目录解析。
以 LogMel 为例，复制模板并填写数据路径：

```bash
cp configs/train-logmel.example.json configs/train-logmel.local.json
scripts/devctl run python -m deepfake_detection_archive train \
  --config configs/train-logmel.local.json

# 填写选定 checkpoint、哈希和 dev 清单，output_dir 设为 ../outputs/audio-dev
cp configs/infer-audio-waveform.example.json configs/infer-audio-waveform.local.json
scripts/devctl run python -m deepfake_detection_archive infer \
  --config configs/infer-audio-waveform.local.json
scripts/devctl run python -m deepfake_detection_archive evaluate \
  --scores outputs/audio-dev/scores.jsonl \
  --manifest data/audio-dev.jsonl --output outputs/audio-dev/evaluation.json
```

视频训练使用 `train-video-head.example.json`，推理使用 `infer-video-head.example.json`。
推理读取本地权重，不联网下载；有标签时再计算 EER。
完整参数、训练输出、推理恢复及离线打包见[开发指南](docs/development.md)。

## 项目结构

```text
src/deepfake_detection_archive/  数据、模型、训练、评测与离线推理
tools/                          数据准备、预训练资源准备、实验辅助工具
configs/                        可复制的配置模板
tests/                          行为回归
docs/                           项目指南、路线图、实验结论与来源许可
data/ models/ outputs/ cache/   本地资产和运行产物，Git 忽略
archive/                        冻结的旧项目，独立于比赛代码
```

| 文档 | 用途 |
| --- | --- |
| [状态](docs/status.md) / [路线图](docs/roadmap.md) | 做了什么、缺什么、下一步交付什么 |
| [数据](docs/data.md) / [开发指南](docs/development.md) | 准备数据、训练、评分、离线运行 |
| [实验结果](docs/experiments/README.md) | 配置、结果和算法结论 |
| [比赛要求](docs/competition/requirements.md) | 已知规则及待适配的官方接口 |
| [方法参考](docs/research/literature.md) | 音频、视频候选与采用依据 |

代码变更先跑对应测试，提交前运行一次 `scripts/devctl check`。
任务在 GitHub Issues 管理，分支与 PR 交付；数据和权重共享只读，输出与缓存按任务独立。
Git 保存代码、配置、必要来源和简洁实验结论；日志、逐样本大分数和临时工作记录留在运行目录。
