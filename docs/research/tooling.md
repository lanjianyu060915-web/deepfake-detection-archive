# 开发与研究工具

准备日期：2026-10-07。用户补充授权安装项目相关依赖、后续按需自行安装。已配置研究/音视频工具和可选 PyTorch 后端；模型开发、训练、权重和数据获取仍按待确认的整体方案推进。

## 已安装和实际使用的项目 skills

| Skill | 来源与固定版本 | 用途与本轮验证 |
| --- | --- | --- |
| `jupyter-notebook` | [openai/skills](https://github.com/openai/skills/tree/49f948faa9258a0c61caceaf225e179651397431/skills/.curated/jupyter-notebook)，Apache-2.0 | 用官方脚手架生成临时实验 notebook，验证 nbformat 与 7 个单元；未执行 notebook 单元。 |
| `huggingface-datasets` | [huggingface/skills](https://github.com/huggingface/skills/tree/ca0325bb20b2d0a1b2efa893670c4c72f79e707b/skills/huggingface-datasets)，Apache-2.0 | 实际只读查询 `jungjee/asvspoof5` 的 Dataset Viewer splits，返回 `default/train`；未下载音频或将其认定为已授权训练集。 |
| `huggingface-papers` | [huggingface/skills](https://github.com/huggingface/skills/tree/ca0325bb20b2d0a1b2efa893670c4c72f79e707b/skills/huggingface-papers)，Apache-2.0 | 实际通过公开 Papers API 获取 AASIST 元数据，验证题名和 arXiv ID；只启用公开 GET 查询。 |
| `pdf` | [openai/skills](https://github.com/openai/skills/tree/49f948faa9258a0c61caceaf225e179651397431/skills/.curated/pdf)，Apache-2.0 | 配合 pypdf 读取 AASIST 的 5 页原始 PDF，并核验提取文本。 |
| `plan`、`skill-installer`，已有 | 安装于用户 Codex skills 目录 | 分别用于可审查规划和固定提交的安装；未启动 Team、自动比赛提交或持久训练循环。 |

项目实际使用的 10 项第三方 skills 已由 `npx skills@1.7.1 --copy` 安装在
`.agents/skills/`，并与 `skills-lock.json` 一同纳入 Git。协作者克隆后可直接发现；
不再依赖某台机器的用户级安装。OpenAI skill 自带许可证，Hugging Face 仓库根
Apache-2.0 许可证副本和全部第三方来源说明保存在
[`.agents/THIRD_PARTY_NOTICES.md`](../../.agents/THIRD_PARTY_NOTICES.md)。
来源 SHA、安装路径及本机历史副本见 [tooling-lock.json](tooling-lock.json)。技能若含
上传、发布或云任务示例，本项目只读研究并不授权这些额外动作。

项目仅收录 10 项经过筛选的 skills，没有把全部 535 项 ds-skills 复制进仓库。未选择
`agentic-data-science-competition`：它主要面向 Kaggle，其提交命令、榜单时序和
自动占位结果不适用于安全屋赛制。没有新增 MCP 服务。

## 用户指定的 ds-skills

已读取 [wenmin-wu/ds-skills](https://github.com/wenmin-wu/ds-skills/tree/718b09722905726c428fcc4eab4d470df4842545)，固定 commit `718b09722905726c428fcc4eab4d470df4842545`。逐项比对发现上游 535 项技能都已在机器级 `/home/maoxiao/.agents/skills/` 安装且内容完全相同；该比对只作为历史审计。当前从 GitHub 来源用 `npx skills@1.7.1` 将其中 6 项复制到项目 `.agents/skills/`，由 Git 和 `skills-lock.json` 管理。项目选择、逐文件 SHA-256、用途和适配要求见 [ds-skills.json](ds-skills.json)。README 和 CLI 元数据声明 MIT，仓库根未发现独立 LICENSE 文件。

本轮选用 6 项，实际使用范围由合成环境探针限定：

| Skill | 本项目用途与使用边界 |
| --- | --- |
| `tabular-group-kfold-leak-prevention` | 在 toy 原媒体组上核验训练/验证互斥；实际组仍需汇总人物、来源、派生链和近重复关系，不能只依赖某一列。 |
| `cv-phash-duplicate-grouping` | 合成图片的 pHash 回环；作为近重复候选工具，不保证旋转不变，也不覆盖音频或全部泄漏。 |
| `cv-video-frame-sampling-pipeline` | 合成视频均匀抽 4 帧，重复运行结果一致；真实变帧率、损坏视频等契约后续建立，性能需实测。 |
| `cv-frame-prediction-averaging` | 合成逐帧连续分数聚合；不继承只处理人脸、裁剪概率或静默跳过无脸帧等示例行为。 |
| `cv-mixed-precision-training` | 使用现代 `torch.amp` 做 CUDA autocast、GradScaler 与梯度有限值检查；不把技能中的显存/速度比例当作承诺。 |
| `cv-gradient-accumulation` | toy 梯度与整批参考一致，涵盖不足累积窗口的末尾；无参数更新，实际训练配方仍待实施。 |

这些技能主要是 Kaggle 通用技巧，没有本赛题专用 EER 或通用音频鉴伪实现。EMA/时序网络留作后续候选；不因 TensorFlow 示例额外安装第二套训练框架。第三方 skill 保持上游原样，赛题适配写在本仓库代码或文档中。

## 本地上游源码参考

三个仓库已克隆到被忽略的 `external/`，固定 SHA 见 [reference-projects.json](reference-projects.json)。它们用于阅读设计和核对 API/配置；未安装上游 requirements，未运行上游模型。

- [AASIST 固定配置](https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/config/AASIST-L.conf)：审阅输入长度、损失、训练参数与轻量结构，未盲目继承其 100 epoch 配方。
- [Community Forensics 固定模型实现](https://github.com/s-jyeon/CommunityForensicsTrain/blob/ba05607c783a3a70a733d6ffa76668c6f0efc7c7/models.py)：审阅冻结 backbone、输入尺度与增强实现，作为低成本视频帧基线参考。
- [ASVspoof 2021 固定 EER 实现](https://github.com/asvspoof-challenge/2021/blob/9b33f5eca887bd3bf629e3fb79428cd9bb7ac972/eval-package/eval_metrics.py)：核对离散 DET、阈值与最近交点定义；本轮仅阅读，不把它当成本届官方评分程序。仓库根未见统一 LICENSE，采用代码前须逐模块核对许可。

可在其他机器按 manifest 的 URL 克隆并 checkout 对应 commit；`external/` 不是项目依赖，也不应被打入 wheel、sdist 或提交包。临时 notebook、PDF 和合成媒体 smoke fixture 放在 `/tmp/aigc-tool-smoke/`，不进入 Git。

## 开发环境

实际版本以 `uv.lock` 为准：Python 3.12.13、uv 0.12.13；开发工具沿用 Ruff/pytest/mypy。基础包运行依赖仍为空；按功能安装 dependency groups 和互斥后端：

| 配置 | 内容 |
| --- | --- |
| 默认 `dev` | Ruff、pytest、mypy；轻量 CI 默认只安装此组 |
| `research` | pypdf、Matplotlib、ipykernel；不启动 notebook 服务或注册全局 kernel |
| `media` | NumPy、SciPy、pandas、scikit-learn、SoundFile、librosa、OpenCV headless、Pillow、ImageHash、PyYAML、tqdm |
| `--extra cu130` | torch 2.14.1 + torchvision 0.29.1 官方 CUDA 13.0 wheels、timm、Transformers；当前开发机后端 |
| `--extra cpu` | 同一版本配对的 CPU wheels、timm、Transformers；另一机器可按需选择，本轮未安装 CPU wheel |

两个后端不能同时选择。官方索引设为 `explicit=true`，避免通用包误从 PyTorch 索引解析；配置方式依据 [uv 官方 PyTorch 集成文档](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies)。完整锁含候选平台/后端，不代表它们都已安装；当前只使用一套 `.venv`。

```bash
# 当前受限环境需要一个可写的缓存位置；正常机器也可以使用默认 uv 缓存。
export UV_CACHE_DIR=/tmp/aigc-uv-cache
uv sync --locked --group media --group research --extra cu130
uv run --no-sync ruff check src tests tools/check_environment.py
uv run --no-sync ruff format --check src tests tools/check_environment.py
uv run --no-sync mypy
uv run --no-sync pytest -q
uv build --no-sources
uv run --no-sync python tools/check_environment.py --device cuda --output outputs/environment/cuda.json
```

无 GPU 时改用 `--extra cpu` 和探针 `--device cpu`。日常命令用 `--no-sync`，或提供相同的组/extra，避免普通同步卸载可选环境。安装后用 `uv pip check` 检查依赖一致性。环境探针只用合成输入、随机模型和临时文件，不联网加载权重或更新参数。

本轮安装后：123 个已安装包的依赖一致性通过，130 项锁记录离线校验通过，互斥后端同时请求按预期被拒绝。完整 CUDA 合成探针已通过，快照见 [environment-check.json](environment-check.json)，含实际版本和脚本哈希；同一 CUDA wheel 的 CPU 执行路径也通过，原始报告在被忽略的 `outputs/environment/`。这不证明独立 CPU wheel、实际鉴伪效果或具体模型的 8GB 可运行性。

GitHub Actions 配置位于 [.github/workflows/checks.yml](../../.github/workflows/checks.yml)，官方 actions 使用固定 commit；标准 CI 只同步默认开发组，静态检查包含环境探针，GPU 实验另行执行。远端 CI 尚未触发。

## 媒体工具和资源现状

- 已有 FFmpeg/ffprobe 8.1.2；实际生成并探测 1 秒 PCM WAV 与 H.264 合成视频。系统 FFmpeg 编译含 GPL 组件，未来若分发二进制需按实际构建确认许可；本轮未重新打包该二进制。[官方许可说明](https://ffmpeg.org/legal.html)
- 沙箱外 `nvidia-smi` 已返回 **RTX 4060 Laptop GPU，8188MiB，驱动 616.92**；普通 App 沙箱仍阻止 NVML/GPU 访问。CUDA 13.x 的驱动族条件见 [NVIDIA 兼容表](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)；实际 CUDA 张量计算、NMS 和 AMP 探针通过。模型的 8GB 训练/批量推理预算尚未验证。
- 当前会话可见约 19GiB 内存、716GiB 可用磁盘；这不是安全屋资源配置。

## 依赖采用依据与后续管理

| 需求 | 决定与依据 |
| --- | --- |
| 深度学习 / CUDA | 采用 [torch 2.14.1](https://github.com/pytorch/pytorch/releases/tag/v2.14.1) 与 [torchvision 0.29.1](https://github.com/pytorch/vision/releases/tag/v0.29.1) 官方配对 wheels。系统驱动未改动；wheel 声明的 CUDA redistributables 不等于全局安装另一份开发 toolkit。旧 fairseq/AASIST 配方按实际调用隔离验证，不覆盖主环境。 |
| 音频数组 / 解码 | FFmpeg/ffprobe + [SoundFile](https://pypi.org/project/soundfile/) + [librosa](https://pypi.org/project/librosa/)；合成音频验证读写、重采样和梅尔频谱。PyAV/TorchCodec 待具体模型需求再安装。 |
| 图像 / 视频 | 采用 [OpenCV headless](https://pypi.org/project/opencv-python-headless/) 与 Pillow；只装一种共用 `cv2` 命名空间的 OpenCV wheel，不引入 GUI/contrib。 |
| TorchAudio | 不默认安装。2.11 起使用稳定 ABI 支持 PyTorch ≥2.11；较早版本按官方配对矩阵。旧 API、解码迁移和模型依赖需逐项核对。[官方安装矩阵](https://docs.pytorch.org/audio/stable/installation.html) |
| SSL / 视觉 backbone | 安装 [Transformers](https://huggingface.co/docs/transformers/installation) 本体及 [timm](https://github.com/huggingface/pytorch-image-models#licenses)，未安装 Transformers `[audio]` / `[all]`。随机模型前向仅证明接口可用；具体权重和数据许可、离线加载后续分别验证。 |
| 数据划分 / EER 核对 | scikit-learn 已准备，先做 toy GroupKFold 验证。EER 的阈值删减、分数方向及插值仍须单独实现核对，不因库已安装而视为已完成。[ROC 文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) |
| 实验平台 | 先用 Git、配置、JSONL 与哈希管理；出现多机实验协作需求时再评估 MLflow/DVC，当前不引入服务。 |

后续按用户授权自主安装实际需要的依赖，更新 `pyproject.toml` / `uv.lock`、工具记录并运行针对性验证。上游版本、数据或权重变化都需要新记录；安装成功只证明工具可用，不能代替模型泛化、数据许可或安全屋可执行性的验证。
