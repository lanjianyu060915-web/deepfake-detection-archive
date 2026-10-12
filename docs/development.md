# 开发与运行

从数据清单到模型训练、连续评分及离线交付。实际进度见[状态](status.md)，
算法和执行优先级见[路线图](roadmap.md)，来源和资产见[数据说明](data.md)。

## 环境

Python 3.12、uv、FFmpeg/ffprobe。当前共享 GPU 环境为 `/opt/venvs/deepfake-cu132`，
日常无需激活或重新同步：

```bash
scripts/devctl run python -m deepfake_detection_archive --help
```

新服务器或确需安装/更新依赖时由维护者执行 `scripts/devctl sync`。
该入口使用锁文件，不向共享环境安装某个 checkout，运行时优先当前 `src/`。
当前全局 FFmpeg/ffprobe 在 `/usr/local/bin`，要求支持 `-fps_mode`。
`doctor` 用于初次安装或故障诊断，不是每轮实验的步骤。

CPU 环境独立建立，不能用 `--all-extras` 或把共享 cu132 环境切成 CPU：

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venv-cpu" uv sync --locked --no-install-project \
  --group media --group research --extra cpu
UV_PROJECT_ENVIRONMENT="$PWD/.venv-cpu" scripts/devctl run \
  python -m deepfake_detection_archive --help
```

可选 direnv：审阅 `.envrc` 后 `direnv allow .`；非交互任务继续用 `devctl run`。

## 数据准备

已有来源优先使用[数据导入工具](data.md#导入与训练清单)。通用媒体可先建立索引：

```bash
scripts/devctl run python -m deepfake_detection_archive index \
  --root data/audio --modality audio --source YOUR_SOURCE \
  --output data/audio-index.jsonl
```

索引不推断标签或许可。补齐来源、标签及已知原媒体/派生/身份关系后，用 `split` 进行组级划分。
已有可信 split 直接训练；需要独立完整解码报告时才运行 `audit --decode`，不重复预解码。
训练会在准备张量时实际解码并记录失败。

## 训练

从 [configs](../configs/README.md) 复制所需模板，填写实际清单、媒体根目录和输出路径。
路径相对配置文件，模板中的权重哈希需使用来源登记值。

```bash
cp configs/train-logmel.example.json configs/train-logmel.local.json
scripts/devctl run python -m deepfake_detection_archive train \
  --config configs/train-logmel.local.json
```

| 方向 | 当前实现与模板 |
| --- | --- |
| 音频波形 | LogMel CNN / AASIST-L：`train-logmel.example.json`、`train-aasist-l.example.json` |
| 语音 SSL | 冻结本地 HuBERT ＋轻量头：`train-audio-ssl-head.example.json` |
| 环境音 | LogMel：`train-logmel-doppelganger.example.json`；[多生成器 AST 微调配置](experiments/audio-envsdd-20261011/README.md) |
| 视频 | 冻结逐帧视觉编码器＋共享头/聚合：`train-video-head.example.json` |
| 视频主干微调 | ResNet18 layer4+头：[实际训练配置](experiments/video-finetune-20261011/seed42.json) |

训练只消费 train/dev，dev 早停并选择最佳 checkpoint；holdout 不进入模型选择。
当前视频支持末端主干微调，尚无时序模型，帧均值不能当作时序学习。
CUDA 支持 BF16/FP16 AMP、梯度累积；新方法先用一次小 profile 确定 batch 和输入规模。
`balanced_sampling` 仅用于 train 的类别平衡；同时开启 `balance_sources` 后，
已存在的来源与类别组合等权抽样，避免小来源被淹没。它不改变 dev 选模指标；
多来源实验需明确选模使用哪个 dev，并分别报告其他来源。
`feature_batch_size` 控制冻结特征提取批量。
`soundfile` 后端只用于原生目标采样率/单声道；通用媒体使用 FFmpeg。

AST 使用 `architecture="audio_ast_v1"`，`model_config.trainable_layers` 指定可训练末端层数，
末层LayerNorm和分类头始终训练。`pretrained.checkpoint`/`sha256` 指向固定本地
`model.safetensors`；只缓存输入filterbank，微调层保留梯度。
训练产出的checkpoint无需预训练目录；推理复用 `model_kind="audio_waveform"`。
4秒输入可用 `backbone_config.max_length=512`；初始化时按 AST 上游方法居中裁剪
预训练时间位置编码，两个特殊 token 和频率位置保持原序，减少无用 padding。
这会改变注意力序列，需重新训练，不能当成旧1024帧模型的无损推理开关。

每次新实验使用新输出目录。输出包含 `experiment.json`、配置和资产标识、失败记录、
`training/history.json`、`training/best.pt`、`dev-scores.jsonl` 与 `dev-evaluation.json`。
缓存准备时检查内容，训练 epoch 内直接读取；运行期间缓存作为只读输入。
源码身份保留在实验摘要，无关文档/CLI 变化不作为特征缓存身份。

训练暂不支持 optimizer/RNG 续跑，中断后保留旧目录并在新目录重启。
算法试验需要长时间运行时再补恢复功能；`infer --resume` 不恢复训练。

## 推理与 EER

无标签媒体可直接 `predict --config MODEL_CONFIG --input DIR --output DIR`，
音频/视频发布配置和完整启动方式见[发布版运行](release.md)。
该入口复用下面的推理与导出逻辑，无需先手写 manifest。

准备与 checkpoint 对应的实值配置；以下示例使用 `data/audio-dev.jsonl`，
推理配置的 `output_dir` 设为 `../outputs/audio-dev`：

```bash
scripts/devctl run python -m deepfake_detection_archive infer \
  --config configs/infer-audio-waveform.local.json
scripts/devctl run python -m deepfake_detection_archive evaluate \
  --scores outputs/audio-dev/scores.jsonl --manifest data/audio-dev.jsonl \
  --output outputs/audio-dev/evaluation.json
```

每样本输出一个连续分数或失败记录。分数越高越伪造，通常为 logit，不保证是概率。
EER 需要两类标签；失败计入覆盖率分母，不能把成功子集当作完整成绩。
模型和输入不变时可加 `infer --resume`，变化时使用新目录。
冻结后评估 holdout 使用 `--frozen-audit`，该参数记录声明，不能代替实际候选冻结。

## 压缩与重采样对照

先取得有价值候选，再对同一批 dev 做配对条件：

```bash
scripts/devctl run python tools/prepare_robustness.py \
  --manifest data/audio-dev.jsonl --data-root data/audio \
  --output-dir outputs/robustness/audio-dev \
  --condition audio_aac64k --condition audio_bandlimit8k
scripts/devctl run python tools/prepare_robustness.py \
  --manifest data/video-dev.jsonl --data-root data/video \
  --output-dir outputs/robustness/video-dev \
  --condition video_h264_crf23 --condition video_h264_crf35
```

每个条件的 manifest/media 配套读取；baseline 清单仍指向原 data_root。
两类处理参数相同，保持 checkpoint、精度、采样和聚合不变，分别 infer/evaluate。
音频比较统一使用 FFmpeg 后端。得到两份分数后：

```bash
scripts/devctl run python tools/compare_scores.py \
  --baseline-scores outputs/audio-original/scores.jsonl \
  --candidate-scores outputs/audio-aac64k/scores.jsonl \
  --output outputs/audio-aac64k/comparison.json
```

报告按 ID 对齐，保留双方失败分母及共同成功子集。ΔEER 为候选减基准，乘100后为百分点。
固定阈值诊断使用基准 dev 选定的同一个 `--threshold`，不在每个条件重新选阈值。

## 导出与离线交付

模型冻结后导出本地连续分数，并打包显式列出的代码、配置、权重、依赖与许可：

```bash
scripts/devctl run python -m deepfake_detection_archive export \
  --config configs/adapter.local.json --scores outputs/audio-dev/scores.jsonl \
  --output outputs/audio-dev/results.csv
uv build --no-sources
scripts/devctl run python -m deepfake_detection_archive package \
  --config configs/package.local.json
scripts/devctl run python -m deepfake_detection_archive verify-package \
  --bundle outputs/candidate.zip --sha256 ACTUAL_BUNDLE_SHA256 \
  --extract-to outputs/candidate-verified
```

官方 I/O 尚未公布，adapter 模板是本地契约，平台样例到来后适配。
包不含训练数据、缓存和历史归档；当前仅有24条语音离线演练，最终双分支验收待完成。

## 开发与协作

RTX4090 可多人并发使用：先用实际小批量确定显存，按剩余显存和利用率启动有价值的任务，
在 Issue 记录占用和预计耗时；负载饱和时按总吞吐调整 batch 或错开重任务，不设个人独占。
每个任务固定独立工作副本和 outputs/cache，数据/权重只读，不修改其他运行任务或共享环境。
PR 给出实际改动、必要测试、样本数和算法结果；无收益实验如实记录。

```bash
# 改动先跑相关测试；提交前运行一次标准检查
scripts/devctl check
```

标准检查范围只在 `tools/check_project.sh` 维护，本地与 CI 共用；覆盖 Ruff、严格 mypy、
行为回归、归档字节和构建。CI 用独立 CPU 环境与固定媒体工具链，来源版本见
[工作流](../.github/workflows/checks.yml)。模型实验不依赖反复运行整套环境探针。
