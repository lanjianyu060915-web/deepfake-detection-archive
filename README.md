# AIGC 音视频伪造识别

第七届中国人工智能大赛比赛项目，分别检测音频和视频中的生成内容。
输入本地媒体，输出逐样本连续伪造分数；以 EER 选择模型，最终离线运行。
内部约定 `real=0`、`fake=1`，分数越高越倾向伪造。

**v0.1.1 提供带权重与运行时的环境音/视频离线研究基线。**
下载 [Release](https://github.com/lanjianyu060915-web/deepfake-detection-archive/releases)，
按[运行说明](docs/release.md)解压后，对无标签目录直接评分：

```bash
./run.sh predict --config configs/audio-environment.json --input /data/audio --output /results/audio
./run.sh predict --config configs/video.json --input /data/video --output /results/video
```

输出成功样本的 `scores.csv`、全部输入的 `scores.jsonl` 和 `report.json`；支持 `--resume`，
失败逐项记账并返回退出码2，不填充虚构分数。包内包含 CPython 3.12、运行依赖、
环境音 AST 和视频 ResNet18 微调权重，
系统需 FFmpeg/ffprobe；可用 CUDA 或 `--device cpu`。

本版音频仅作为环境音研究候选，尚无覆盖语音、音乐和混合内容的通用音频交付模型。
视频跨生成器效果继续优化。
正式比赛I/O和外部资源适用性尚未确认，详见[发布资产说明](docs/release-licenses.md)。
[项目状态](docs/status.md)记录算法结果与缺口，[路线图](docs/roadmap.md)列出后续任务。

当前源码另提供统一入口，支持单文件、多文件和音视频混合目录；只指定输入和输出，
自动按实际媒体流选择固定模型。源码运行需用 `DEEPFAKE_MODEL_PACKAGE` 指向本地模型包，
命令与输出见[统一预测入口](docs/release.md#统一预测入口当前源码)。
已发布的 v0.1.1 压缩包仍使用上面的显式配置命令。

## 源码开发环境

Python 3.12、uv、FFmpeg/ffprobe。在仓库根目录建立独立CPU环境：

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cpu"
uv sync --locked --group dev --group media --group research --extra cpu
uv run --no-sync python -m deepfake_detection_archive --help
```

源码环境运行统一预测时，另将`DEEPFAKE_MODEL_PACKAGE`设为已有本地模型包目录；
其下须有`configs/models.json`及引用的配置/权重。相对模型包路径从当前工作目录解析，
支持`~`；显式错误目录会报错，不借用源码目录的另一份注册表。
模型不随Git或项目wheel分发。已发布离线包直接使用其`run.sh`，无需源码开发环境。

CPU/cu132后端互斥；CUDA环境、前端一致性和训练参数见[开发指南](docs/development.md#环境)。
现有RTX4090共享环境的`devctl`兼容入口和权限见[团队环境](docs/team-environment.md)。
日常运行不重新同步依赖或做全量环境检查。

## 数据 → 训练 → 推理 → 评测

[数据说明](docs/data.md)集中列出来源、实际资产和准备命令。
`configs/` 提供[配置模板](configs/README.md)，相对路径从配置文件目录解析。
以 LogMel 为例，复制模板并填写数据路径：

```bash
cp configs/train-logmel.example.json configs/train-logmel.local.json
# 填写数据路径；CPU调试同时设置 training.device="cpu"、training.amp=false
uv run --no-sync python -m deepfake_detection_archive train \
  --config configs/train-logmel.local.json

# 填写选定 checkpoint、哈希和 dev 清单，output_dir 设为 ../outputs/audio-dev
cp configs/infer-audio-waveform.example.json configs/infer-audio-waveform.local.json
uv run --no-sync python -m deepfake_detection_archive infer \
  --config configs/infer-audio-waveform.local.json
uv run --no-sync python -m deepfake_detection_archive evaluate \
  --scores outputs/audio-dev/scores.jsonl \
  --manifest data/audio-dev.jsonl --output outputs/audio-dev/evaluation.json
```

视频可训练冻结头或 ResNet18 末端主干，见[开发指南](docs/development.md#训练)。
推理读取本地权重，不联网下载；有标签时再计算 EER。
数据隔离关系与统计独立证据分别记录：AIME prompt 组仍整组隔离，但不能证明
artist/原曲独立。缓存按媒体、预处理配置及准备源码识别；推理恢复检查输入、模型、
配置和代码身份。迁移环境后使用新输出目录重评测。
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
| [发布版运行](docs/release.md) / [资产条款](docs/release-licenses.md) | 下载、无标签评分、恢复与重建 |
| [实验结果](docs/experiments/README.md) | 配置、结果和算法结论 |
| [比赛要求](docs/competition/requirements.md) | 已知规则及待适配的官方接口 |
| [方法参考](docs/research/literature.md) | 音频、视频候选与采用依据 |

代码变更先跑对应测试，提交前在所选环境运行一次 `bash tools/check_project.sh`。
任务在 GitHub Issues 管理，分支与 PR 交付；数据和权重共享只读，输出与缓存按任务独立。
Git 保存代码、配置、必要来源和简洁实验结论；日志、逐样本大分数和临时工作记录留在运行目录。
