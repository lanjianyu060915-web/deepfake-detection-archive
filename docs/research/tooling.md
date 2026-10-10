# 开发与研究工具

更新日期：2026-10-10。实际依赖版本以 `uv.lock` 为准，项目进度以
[项目状态](../status.md)为准。安装成功只说明工具可用，模型结果见对应实验记录。

## 项目 skills

项目使用的 11 项第三方 skills 已通过 `npx skills@1.7.1 --copy` 安装到
`.agents/skills/`，由 `skills-lock.json` 固定并纳入 Git。协作者克隆仓库后可直接发现，
当前工作副本 11 项完整，无需重装；不依赖机器级目录。来源和许可见 [第三方声明](../../.agents/THIRD_PARTY_NOTICES.md)，固定
来源见 [tooling-lock.json](tooling-lock.json)。

其中与本项目主线直接相关的是：

| 类别 | skills | 用途 |
| --- | --- | --- |
| 数据与划分 | `tabular-group-kfold-leak-prevention`、`cv-phash-duplicate-grouping` | 组级划分和近重复候选 |
| 视频 | `cv-video-frame-sampling-pipeline`、`cv-frame-prediction-averaging` | 确定性抽帧和媒体级聚合 |
| 训练 | `cv-mixed-precision-training`、`cv-gradient-accumulation` | AMP 和梯度累积参考 |
| 音频结构 | `cv-separable-temporal-spectral-cnn` | 时频分离卷积设计参考 |
| 研究 | `huggingface-datasets`、`huggingface-papers`、`pdf`、`jupyter-notebook` | 数据、论文和实验辅助 |

skills 提供实现思路，不定义本项目的比赛契约。第三方内容保持上游原样，赛题适配写在本仓库。
全部 535 项 ds-skills 的机器级比对属于历史审计，没有复制进项目，也不参与当前工作流。

需要更新时继续使用固定的 `npx skills@1.7.1`，并审阅 Git 差异和许可。技能中的上传、
发布、遥测或云任务示例不代表本项目授权。

## 本地上游参考

三个参考仓库约定放在被忽略的 `external/`；当前工作副本已恢复固定提交的 ASVspoof 2021，
AASIST 和 Community Forensics 尚未恢复。固定提交见
[reference-projects.json](reference-projects.json)：

- AASIST：参考轻量语音反欺骗结构和输入配方；当前项目已完成 AASIST-L 小样本训练。
- ASVspoof 2021：用于核对 EER 定义；本项目 EER 已实现并测试，它不是本届官方评分程序。
- Community Forensics：参考冻结视觉 backbone 和帧级基线；当前视频 pilot 已使用本地
  ResNet18 权重提取特征并训练媒体级头。

需要恢复时，按清单中的 `repository` 克隆到对应 `local_path`，检出精确的 `commit`，
再核对提交和许可文件。ASVspoof 上游一致性测试缺少该参考时会显式跳过；恢复后再执行。
不为补齐阅读材料重复下载数据或安装上游依赖。

这些仓库只用于阅读和设计核对，没有安装整份上游 requirements，也不进入项目构建。
采用其中代码前仍需按文件核对许可。

## 环境与依赖

项目使用 Python 3.12.15、uv 0.13.0 和以下依赖组：

| 配置 | 内容 |
| --- | --- |
| 默认 `dev` | Ruff、pytest、mypy |
| `media` | NumPy、SciPy、pandas、scikit-learn、SoundFile、librosa、OpenCV headless、Pillow、ImageHash、PyYAML、tqdm |
| `research` | pypdf、Matplotlib、ipykernel |
| `--extra cu132` | torch 2.14.1、torchvision 0.29.1、timm、Transformers，当前主开发环境 |
| `--extra cpu` | 对应 CPU wheels，用于独立离线环境 |

CPU 和 cu132 后端互斥。PyTorch 使用官方显式索引，通用包使用锁定的默认索引。
当前服务器共享环境已具备上述依赖，CUDA 与 AMP 合成探针通过。环境使用与维护统一见
[开发指南](../development.md)，完整检查结果统一见[项目状态](../status.md)。

旧机完成的真实训练、推理，以及独立 CPU 环境 65 包断网安装演练，属于历史实验证据；
这些结果不说明当前服务器已具备全部数据和权重。旧机 AASIST-L reserved 峰值约 7.36 GiB，
当前 RTX 4090 24 GiB 仍按具体任务先 profile。

FFmpeg/ffprobe 已全局安装到 `/opt/ffmpeg/btbn-628033703`，通过 `/usr/local/bin` 调用，
版本为 `N-127271-gba987fe24d-20261010`，支持 `-fps_mode`。下载包SHA-256已与
[BtbN发布清单](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/checksums.sha256)
核对，安装来源记录及GPLv3+许可文本保存在该安装目录。最终离线环境仍须按实际目标系统
准备二进制和许可；项目wheel不包含宿主FFmpeg。

## 已完成的模型准备与迁移缺口

以下为旧机已完成的工作；当前工作副本尚未接入相应媒体和权重，服务器共享下载状态另行核对：

- 已固定并本地加载 ResNet18 视觉权重，完成 ViF 视频特征提取和两种子轻量头训练；
- 已准备本地 HuBERT bundle 和冻结 SSL 训练接口，尚未完成真实 HuBERT 头训练；
- 已完成 LogMel CNN 与 AASIST-L 的真实语音小样本训练；
- 当前没有验证视频主干微调、时序模型或通用音频模型。

权重必须从本地加载并登记来源、许可与哈希，运行时禁止隐式下载。只在具体实验需要时新增
依赖并更新 `pyproject.toml` 和 `uv.lock`；不为候选方法预装整套框架。

GitHub Actions 位于 [checks.yml](../../.github/workflows/checks.yml)。当前服务器验证见状态页；
远端 CI 只按对应提交的实际运行结果报告，不能从本地检查推断成功。
