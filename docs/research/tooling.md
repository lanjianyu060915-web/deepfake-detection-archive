# 开发与研究工具

更新日期：2026-10-09。实际依赖版本以 `uv.lock` 为准，项目进度以
[项目状态](../status.md)为准。安装成功只说明工具可用，模型结果见对应实验记录。

## 项目 skills

项目使用的 11 项第三方 skills 已通过 `npx skills@1.7.1 --copy` 安装到
`.agents/skills/`，由 `skills-lock.json` 固定并纳入 Git。协作者克隆仓库后可直接发现，
不依赖机器级目录。来源和许可见 [第三方声明](../../.agents/THIRD_PARTY_NOTICES.md)，固定
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

三个参考仓库位于被忽略的 `external/`，固定提交见
[reference-projects.json](reference-projects.json)：

- AASIST：参考轻量语音反欺骗结构和输入配方；当前项目已完成 AASIST-L 小样本训练。
- ASVspoof 2021：用于核对 EER 定义；本项目 EER 已实现并测试，它不是本届官方评分程序。
- Community Forensics：参考冻结视觉 backbone 和帧级基线；当前视频 pilot 已使用本地
  ResNet18 权重提取特征并训练媒体级头。

这些仓库只用于阅读和设计核对，没有安装整份上游 requirements，也不进入项目构建。
采用其中代码前仍需按文件核对许可。

## 环境与依赖

项目使用 Python 3.12、uv 0.12.13 和以下依赖组：

| 配置 | 内容 |
| --- | --- |
| 默认 `dev` | Ruff、pytest、mypy |
| `media` | NumPy、SciPy、pandas、scikit-learn、SoundFile、librosa、OpenCV headless、Pillow、ImageHash、PyYAML、tqdm |
| `research` | pypdf、Matplotlib、ipykernel |
| `--extra cu130` | torch 2.14.1、torchvision 0.29.1、timm、Transformers，当前主开发环境 |
| `--extra cpu` | 对应 CPU wheels，用于独立离线环境 |

CPU 和 cu130 后端互斥。PyTorch 使用官方显式索引，通用包使用锁定的默认索引。当前主环境
已完成 CUDA 张量、AMP、真实训练和推理；独立 CPU 环境完成 65 包断网安装及真实 LogMel
推理。AASIST-L 实测 reserved 峰值约 7.36 GiB，因此 8GB 下扩大实验仍需先 profile。

```bash
export UV_CACHE_DIR=/tmp/aigc-uv-cache
uv sync --locked --group media --group research --extra cu130
bash tools/check_project.sh
```

FFmpeg/ffprobe 使用宿主版本。当前 Linuxbrew FFmpeg 是 shared build，最终离线环境需要按实际
目标系统处理其二进制和动态库；本轮没有把宿主 FFmpeg 打入项目 wheel。

## 已准备的模型资源

- 已固定并本地加载 ResNet18 视觉权重，完成 ViF 视频特征提取和两种子轻量头训练；
- 已准备本地 HuBERT bundle 和冻结 SSL 训练接口，尚未完成真实 HuBERT 头训练；
- 已完成 LogMel CNN 与 AASIST-L 的真实语音小样本训练；
- 当前没有验证视频主干微调、时序模型或通用音频模型。

权重必须从本地加载并登记来源、许可与哈希，运行时禁止隐式下载。只在具体实验需要时新增
依赖并更新 `pyproject.toml` 和 `uv.lock`；不为候选方法预装整套框架。

GitHub Actions 位于 [checks.yml](../../.github/workflows/checks.yml)。本地检查已经执行，
远端 CI 尚未触发。
