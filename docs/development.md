# 开发与实验指南

本文只描述当前实现的主线：准备数据、训练、推理、评测和离线打包。真实进度与结果见
[项目状态](status.md)，算法路线见[实施方案](plans/implementation-plan.md)。

更新日期：2026-10-10。后续在云端执行，接手见[云端协作](handoff/cloud-collaboration.md)。

## 1. 环境

当前服务器使用 Python 3.12.15、uv 0.13.0、PyTorch 2.14.1+cu132 和
TorchVision 0.29.1+cu132，共享环境位于 `/opt/venvs/deepfake-cu132`。
`dev`、`research`、`media` 依赖已齐全；锁文件和当前验证结果见[项目状态](status.md)。

在仓库根目录使用以下入口，无需先激活 direnv：

```bash
scripts/devctl doctor
scripts/devctl run python -m deepfake_detection_archive --help
```

`run`、`test` 和 `check` 优先加载各自工作副本的 `src/`。共享环境不安装任何 checkout 的
项目，因此日常使用模块命令，不依赖 `deepfake-detection-archive` 控制台脚本。
FFmpeg/ffprobe 已全局安装到 `/opt/ffmpeg/btbn-628033703`，入口为
`/usr/local/bin/ffmpeg` 和 `/usr/local/bin/ffprobe`，由 root 管理，所有账号可读取和执行。
当前版本为 `N-127271-gba987fe24d-20261010`；媒体处理要求支持 `-fps_mode`，
仅有旧版二进制不代表媒体测试可运行。共享数据位置和使用方式见[云端协作](handoff/cloud-collaboration.md#3-git-外资产)。

### 共享环境维护和可选激活

进入目录不安装依赖或改锁。仅环境维护者在确需同步时运行：

```bash
scripts/devctl sync
```

该入口使用 `uv sync --locked --no-install-project`，同步 `cu132` 与开发、研究、媒体组，
并用文件锁避免并发写入。精确同步可能移除锁文件以外的包；不要用其他项目管理此共享环境。
依赖已匹配时直接复用，不为接手而升级锁文件。

交互式 Bash 可使用已配置的全局 direnv hook。每位用户首次使用或 `.envrc` 改动后，
先审阅该可执行配置，再运行：

```bash
direnv allow .
```

它设置共享环境、CUDA 路径和当前源码路径，离开目录后恢复。非交互式任务继续使用
`scripts/devctl run ...`。新服务器的管理员可通过 `scripts/install-global-direnv-hook.sh`
配置 Bash hook；当前服务器无需重复安装。容器重建后的保留范围取决于平台持久化配置。

### 独立 CPU 环境

`cpu` 与 `cu132` 互斥，不能使用 `--all-extras`，也不能在共享 GPU 环境同步 CPU 后端。
需要 CPU 验证时，显式建立独立环境并交给同一开发入口：

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venv-cpu" uv sync --locked --no-install-project \
  --group media --group research --extra cpu
UV_PROJECT_ENVIRONMENT="$PWD/.venv-cpu" scripts/devctl run \
  python -m deepfake_detection_archive --help
```

此 CPU 安装示例未在本轮执行；本轮复用并验证共享 GPU 环境。`doctor` 针对当前 cu132/GPU
部署，不用于 CPU 环境验收。

配置模板位于 [configs/](../configs/README.md)。复制需要的模板为 `.local.json`，替换
`REPLACE_*`、零哈希和示例路径。配置内的相对路径以配置文件目录为基准。

## 2. 数据：index → audit → split

先为合法持有的媒体建立索引：

```bash
scripts/devctl run python -m deepfake_detection_archive index \
  --root data/audio --modality audio --source YOUR_SOURCE \
  --output data/audio-index.jsonl
```

索引只登记媒体，不推断标签、身份或许可。训练样本还需补充可信的 `label`、来源登记，
以及已知的 `original_id`、`parent_id`、`identity_id` 或 `derivation_group_id`。内部标签为
`real=0`、`fake=1`；未知信息保留 `unknown`。

准备 `sources.jsonl` 和实值审计配置后，可执行独立解码审计和组级划分：

```bash
scripts/devctl run python -m deepfake_detection_archive audit \
  --config configs/audit.local.json --decode
scripts/devctl run python -m deepfake_detection_archive split \
  --manifest data/audio-decoded.jsonl \
  --output data/audio-split.jsonl --seed 42
```

训练会在准备张量时实际解码并记录失败；已有可信划分时，不需要额外完整预解码才能训练。
独立`audit --decode`仍用于需要完整数据解码报告的场景。

划分会把相同字节、原媒体、派生样本和已登记身份关系组成连通组，整个组只进入一个 split。
近重复检测结果是候选关系，经确认后再写入清单。训练前确认 train/dev 均含真假两类；
holdout 只在候选冻结后使用。

ASVspoof 2019 LA 和 ViF Panda 已有固定导入工具：

```bash
scripts/devctl run python tools/finalize_asvspoof2019.py --help
scripts/devctl run python tools/prepare_asvspoof2019.py --help
scripts/devctl run python tools/prepare_vif_panda.py --help
scripts/devctl run python tools/prepare_asv_main.py --help
scripts/devctl run python tools/prepare_video_control.py --help
```

这些工具的实际参数以 `--help` 和当前研究记录为准。历史导入日志不作为新实验入口。

### dev 压缩与重采样对照

已有 dev 清单和模型后，用同一批样本检查常见媒体处理造成的退化。先准备条件媒体：

```bash
scripts/devctl run python tools/prepare_robustness.py \
  --manifest data/audio-dev-only.jsonl --data-root data/audio \
  --output-dir outputs/robustness/audio-dev \
  --condition audio_aac64k --condition audio_bandlimit8k

scripts/devctl run python tools/prepare_robustness.py \
  --manifest data/video-dev-only.jsonl --data-root data/video \
  --output-dir outputs/robustness/video-dev \
  --condition video_h264_crf23 --condition video_h264_crf35
```

工具接受纯 dev、单模态且包含真假两类的清单；每次使用新输出目录。条件参数不依赖标签，
源文件只读；失败保留在条件清单中并计入覆盖率。条件目录下的 `manifest.jsonl` 和 `media/`
配套使用，`summary.json` 给出各条件的清单、data_root、参数和失败信息。
`baseline.jsonl` 保留原路径，其 data_root 仍为输入媒体根目录。

分别复制现有推理实值配置，只改变 manifest、data_root 和 output_dir，保持本地 checkpoint、
采样/抽帧、聚合与精度一致。音频使用 FFmpeg 后端，使 AAC 和原音频采用相同解码入口；
8 kHz 条件最终保存为 16 kHz FLAC，测试的是带宽损失。每种条件继续运行现有 `infer` 和
`evaluate --manifest`，比较原媒体与扰动后的 EER、覆盖率及分生成器结果。
转码保留源声道，降混统一由现有解码器完成，避免条件额外引入音量变化；损坏输入沿用项目
严格解码参数，转码器不会把部分恢复的媒体计作成功。
如需固定阈值错误率，使用原 dev 选定的同一阈值，不在各条件重新调参。

同一 ID 表示配对样本，条件清单分别评分；不拼成重复的独立样本，也不重新划分。
此工具不读取 holdout 媒体；解码成功不能证明模型鲁棒，真实结果以模型评分为准。

## 3. 训练

选择与模型对应的模板，例如 LogMel：

```bash
cp configs/train-logmel.example.json configs/train-logmel.local.json
# 编辑实值路径、哈希、训练参数和输出目录
scripts/devctl run python -m deepfake_detection_archive train \
  --config configs/train-logmel.local.json
```

当前支持 AASIST-L、LogMel CNN、冻结音频 SSL 编码器加媒体级头，以及冻结视频编码器加
媒体级头。训练只读取 train/dev 媒体；dev 用于早停和选择最佳 checkpoint，holdout 不参与。

CUDA 训练支持 `float16`/`bfloat16` AMP 和梯度累积。先用小批量 profile，再调整 batch、
精度、workers 和累积步数。当前 RTX 4090 有 24 GiB 显存；旧机 8 GiB 的 AASIST-L 峰值不能作为当前 batch 上限。

训练输出包括实验摘要、审计、失败记录、历史、最佳 checkpoint、`dev-scores.jsonl`和
`dev-evaluation.json`。dev分数直接来自准备好的张量，不代表端到端解码/编码吞吐。
`balanced_sampling=true`仅平衡train采样；`feature_batch_size`单独控制冻结特征批量。
`audio.backend="soundfile"`适用于原生目标采样率/单声道，不重采样或混声道；通用输入默认FFmpeg。所有选中样本必须成功准备，
不会静默删除解码失败样本。当前训练**不支持 optimizer/RNG 恢复**；中断后保留旧目录并在
新输出目录重启。`infer --resume` 只用于推理恢复。

冻结 SSL/视觉编码器只从本地加载已登记权重，不通过模型 ID 隐式下载。更换编码器、预处理
或媒体内容会使用新的特征缓存；CLI、评测等无关代码变动可复用缓存，运行报告仍记录全源码版本。
特征缓存采用 v2，旧版本缓存保留，新键首次使用时重新生成；损坏缓存会明确报错。

## 4. 推理和评测

填好与 checkpoint 对应的推理配置后执行：

```bash
scripts/devctl run python -m deepfake_detection_archive infer \
  --config configs/infer-audio-waveform.local.json
scripts/devctl run python -m deepfake_detection_archive evaluate \
  --scores outputs/audio-dev-scores/scores.jsonl \
  --manifest data/audio-dev-only.jsonl \
  --output outputs/audio-dev-scores/evaluation.json
```

每个输入产生一个成功或失败记录。内部连续分数越高越倾向伪造；它通常是 logit，不是概率。
评测输出 EER、覆盖率、失败率及可用的分组结果。存在失败、未知标签或缺类时，应读取原因，
不能把成功子集结果当作完整数据成绩。

中断后使用相同配置和输入恢复：

```bash
scripts/devctl run python -m deepfake_detection_archive infer \
  --config configs/infer-audio-waveform.local.json --resume
```

模型、影响评分的配置或输入发生变化时使用新输出目录。对 holdout 推理需显式加入
`--frozen-audit`；该参数记录流程声明，模型和配置仍应在运行前冻结。

## 5. 导出和离线包

本地分数导出：

```bash
scripts/devctl run python -m deepfake_detection_archive export \
  --config configs/adapter.local.json \
  --scores outputs/audio-dev-scores/scores.jsonl \
  --output outputs/local-scores.csv
```

官方格式尚未公布，`adapter.example.json` 是本地适配模板。平台提供样例后更新列名、方向和
失败处理，不改模型内部评分约定。

冻结候选后，构建项目 wheel，并在打包配置中列出运行所需代码、配置、权重、依赖和许可：

```bash
uv build --no-sources
scripts/devctl run python -m deepfake_detection_archive package \
  --config configs/package.local.json
scripts/devctl run python -m deepfake_detection_archive verify-package \
  --bundle outputs/candidate.zip \
  --sha256 ACTUAL_BUNDLE_SHA256 \
  --extract-to outputs/candidate-verified
```

打包命令不自动下载或发现依赖。最终包不应包含训练数据、缓存、历史归档和研究日志。当前仅
完成 24 条语音的 CPU 断网演练；视频、代表性批量吞吐和官方环境仍待验收，见
[离线记录](research/offline-rehearsal/README.md)。

## 6. 开发检查

代码变更先运行对应测试；准备提交或交付时运行标准检查：

```bash
scripts/devctl check
```

`scripts/devctl check` 委托 `tools/check_project.sh`；本地和 CI 使用同一检查定义，覆盖主动代码与当前开发工具，范围以该脚本为准；历史恢复脚本和归档不做自动格式修复。
工程检查证明实现行为，不代替真实数据上的模型评测。

本轮核对模块及导入工具的 `--help`；上述数据、训练、推理和打包示例需要实值配置与资产，
尚未在当前服务器端到端执行。当前工作副本尚未接入这些训练资产，不能将命令示例当作已完成实验。
