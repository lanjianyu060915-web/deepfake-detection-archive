# 开发与研究工具

准备日期：2026-10-07。现在安装的是研究/工程辅助能力；训练框架、CUDA、模型权重和数据获取另在方案确认后按候选与平台约束安装。

## 已安装和实际使用的 skills

| Skill | 来源与固定版本 | 用途与本轮验证 |
| --- | --- | --- |
| `jupyter-notebook` | [openai/skills](https://github.com/openai/skills/tree/49f948faa9258a0c61caceaf225e179651397431/skills/.curated/jupyter-notebook)，Apache-2.0 | 用官方脚手架生成临时实验 notebook，验证 nbformat 与 7 个单元；未执行 notebook 单元。 |
| `huggingface-datasets` | [huggingface/skills](https://github.com/huggingface/skills/tree/ca0325bb20b2d0a1b2efa893670c4c72f79e707b/skills/huggingface-datasets)，Apache-2.0 | 实际只读查询 `jungjee/asvspoof5` 的 Dataset Viewer splits，返回 `default/train`；未下载音频或将其认定为已授权训练集。 |
| `huggingface-papers` | [huggingface/skills](https://github.com/huggingface/skills/tree/ca0325bb20b2d0a1b2efa893670c4c72f79e707b/skills/huggingface-papers)，Apache-2.0 | 实际通过公开 Papers API 获取 AASIST 元数据，验证题名和 arXiv ID；只启用公开 GET 查询。 |
| `pdf`，已有 | `/home/maoxiao/.codex/skills/pdf/SKILL.md` | 配合 pypdf 读取 AASIST 的 5 页原始 PDF，并核验提取文本；不重复安装已有 skill。 |
| `plan`、`skill-installer`，已有 | 安装于用户 Codex skills 目录 | 分别用于可审查规划和固定提交的安装；未启动 Team、自动比赛提交或持久训练循环。 |

新 skills 安装在 `/home/maoxiao/.codex/skills/`，下轮会话可通过正常技能发现使用；本轮已按绝对路径阅读和验证。Hugging Face 仓库根许可证补存到两个已安装 skill 目录；来源 SHA 及本机记录见 [tooling-lock.json](tooling-lock.json)。技能若含上传、发布或云任务示例，本项目只读研究并不授权这些额外动作。

未选择的项：已有 `agentic-data-science-competition` 主要面向 Kaggle，其提交命令、榜单时序和自动占位结果不适用于安全屋赛制。没有批量安装同名或宽泛的“全部 ML skills”，也没有新增 MCP 服务。

## 本地上游源码参考

三个仓库已克隆到被忽略的 `external/`，固定 SHA 见 [reference-projects.json](reference-projects.json)。它们用于阅读设计和核对 API/配置；未安装上游 requirements，未运行上游模型。

- [AASIST 固定配置](https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/config/AASIST-L.conf)：审阅输入长度、损失、训练参数与轻量结构，未盲目继承其 100 epoch 配方。
- [Community Forensics 固定模型实现](https://github.com/s-jyeon/CommunityForensicsTrain/blob/ba05607c783a3a70a733d6ffa76668c6f0efc7c7/models.py)：审阅冻结 backbone、输入尺度与增强实现，作为低成本视频帧基线参考。
- [ASVspoof 2021 固定 EER 实现](https://github.com/asvspoof-challenge/2021/blob/9b33f5eca887bd3bf629e3fb79428cd9bb7ac972/eval-package/eval_metrics.py)：核对离散 DET、阈值与最近交点定义；本轮仅阅读，不把它当成本届官方评分程序。仓库根未见统一 LICENSE，采用代码前须逐模块核对许可。

可在其他机器按 manifest 的 URL 克隆并 checkout 对应 commit；`external/` 不是项目依赖，也不应被打入 wheel、sdist 或提交包。临时 notebook、PDF 和合成媒体 smoke fixture 放在 `/tmp/aigc-tool-smoke/`，不进入 Git。

## 开发环境

实际版本以 `uv.lock` 为准：Python 3.12.13；uv 0.12.13；Ruff 0.16.10、pytest 9.1.1、mypy 2.4.0；研究组 pypdf 6.19.0。算法运行依赖仍为空。

```bash
# 当前受限环境需要一个可写的缓存位置；正常机器也可以使用默认 uv 缓存。
export UV_CACHE_DIR=/tmp/aigc-uv-cache
uv sync --locked --group research
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy
uv run --no-sync pytest -q
uv build --no-sources
```

`dev` 是默认开发组；只有论文解析需要 `--group research`，标准 CI 仅同步默认开发组。已验证锁定同步与 wheel/sdist 构建。GitHub Actions 配置位于 [.github/workflows/checks.yml](../../.github/workflows/checks.yml)，官方 actions 使用固定 commit；本轮只做本地等价检查，远端 CI 尚未触发。

## 媒体工具和资源现状

- 已有 FFmpeg/ffprobe 8.1.2；实际生成并探测 1 秒 PCM WAV 与 H.264 合成视频。系统 FFmpeg 编译含 GPL 组件，未来若分发二进制需按实际构建确认许可；本轮未重新打包该二进制。[官方许可说明](https://ffmpeg.org/legal.html)
- 用户确认可用 **RTX 4060 8GB**，后续可能使用云算力；当前 App 进程的 `nvidia-smi` 返回 NVML/GPU 被操作系统阻止。因此记录为“有该硬件、当前会话访问未验证”，不描述为无 GPU，也不认定 PyTorch CUDA 已可用。
- 当前会话可见约 19GiB 内存、716GiB 可用磁盘；这不是安全屋资源配置。

## 按阶段再引入的依赖

| 需求 | 决定与依据 |
| --- | --- |
| 深度学习 / CUDA | 模型与平台确定后选择 PyTorch wheel；先验证 GPU、CUDA、峰值显存和确定性。旧 fairseq/AASIST 环境必要时隔离，不覆盖主环境。 |
| 音频数组 / 解码 | 优先 ffmpeg/ffprobe；按功能评估 SoundFile、PyAV 或 TorchCodec，不同时安装所有媒体栈。[SoundFile](https://python-soundfile.readthedocs.io/en/0.13.1/)、[PyAV](https://github.com/PyAV-Org/PyAV) |
| TorchAudio | 不默认安装。2.11 起使用稳定 ABI 支持 PyTorch ≥2.11；较早版本按官方配对矩阵。旧 API、解码迁移和模型依赖需逐项核对。[官方安装矩阵](https://docs.pytorch.org/audio/stable/installation.html) |
| SSL / 视觉 backbone | 选定模型后引入 Transformers 或 timm，分别核对代码、权重和训练数据许可；准备离线本地加载。[Transformers](https://huggingface.co/docs/transformers/installation)、[timm](https://github.com/huggingface/pytorch-image-models#licenses) |
| EER 交叉核对 | 可使用 scikit-learn 的 ROC/DET 进行独立检查，但必须明确阈值删减、分数方向及插值差异。[ROC 文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) |
| 实验平台 | 先用 Git、配置、JSONL 与哈希管理；出现多机实验协作需求时再评估 MLflow/DVC，当前不引入服务。 |

上游版本、数据或权重变化都需要新记录与测试；安装成功只证明工具可用，不能代替模型泛化、数据许可或安全屋可执行性的验证。
