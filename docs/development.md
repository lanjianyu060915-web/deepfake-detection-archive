# 开发与实验指南

本文只描述当前实现的主线：准备数据、训练、推理、评测和离线打包。真实进度与结果见
[项目状态](status.md)，算法路线见[实施方案](plans/implementation-plan.md)。

更新日期：2026-10-09。

## 1. 环境

项目使用 Python 3.12 和 uv。本机 RTX 4060 开发环境：

```bash
export UV_CACHE_DIR=/tmp/aigc-uv-cache
uv sync --locked --group media --group research --extra cu130
ffmpeg -version
ffprobe -version
uv run --no-sync deepfake-detection-archive --help
```

无 GPU 环境将 `--extra cu130` 改为 `--extra cpu`。两个 PyTorch 后端互斥。日常命令使用
`--no-sync`，避免 uv 按默认组移除已经安装的可选依赖。

配置模板位于 [configs/](../configs/README.md)。复制需要的模板为 `.local.json`，替换
`REPLACE_*`、零哈希和示例路径。配置内的相对路径以配置文件目录为基准。

## 2. 数据：index → audit → split

先为合法持有的媒体建立索引：

```bash
uv run --no-sync deepfake-detection-archive index \
  --root data/audio --modality audio --source YOUR_SOURCE \
  --output data/audio-index.jsonl
```

索引只登记媒体，不推断标签、身份或许可。训练样本还需补充可信的 `label`、来源登记，
以及已知的 `original_id`、`parent_id`、`identity_id` 或 `derivation_group_id`。内部标签为
`real=0`、`fake=1`；未知信息保留 `unknown`。

准备 `sources.jsonl` 和实值审计配置后，执行解码审计和组级划分：

```bash
uv run --no-sync deepfake-detection-archive audit \
  --config configs/audit.local.json --decode
uv run --no-sync deepfake-detection-archive split \
  --manifest data/audio-decoded.jsonl \
  --output data/audio-split.jsonl --seed 42
```

划分会把相同字节、原媒体、派生样本和已登记身份关系组成连通组，整个组只进入一个 split。
近重复检测结果是候选关系，经确认后再写入清单。训练前确认 train/dev 均含真假两类；
holdout 只在候选冻结后使用。

ASVspoof 2019 LA 和 ViF Panda 已有固定导入工具：

```bash
uv run --no-sync python tools/finalize_asvspoof2019.py --help
uv run --no-sync python tools/prepare_asvspoof2019.py --help
uv run --no-sync python tools/prepare_vif_panda.py --help
```

这些工具的实际参数以 `--help` 和当前研究记录为准。历史导入日志不作为新实验入口。

## 3. 训练

选择与模型对应的模板，例如 LogMel：

```bash
cp configs/train-logmel.example.json configs/train-logmel.local.json
# 编辑实值路径、哈希、训练参数和输出目录
uv run --no-sync deepfake-detection-archive train \
  --config configs/train-logmel.local.json
```

当前支持 AASIST-L、LogMel CNN、冻结音频 SSL 编码器加媒体级头，以及冻结视频编码器加
媒体级头。训练只读取 train/dev 媒体；dev 用于早停和选择最佳 checkpoint，holdout 不参与。

CUDA 训练支持 `float16`/`bfloat16` AMP 和梯度累积。先用小批量 profile，再调整 batch、
精度、workers 和累积步数。8GB 显存下不要直接照搬模板参数；实际 AASIST-L 已接近显存上限。

训练输出包括实验摘要、审计、失败记录、历史和最佳 checkpoint。所有选中样本必须成功准备，
不会静默删除解码失败样本。当前训练**不支持 optimizer/RNG 恢复**；中断后保留旧目录并在
新输出目录重启。`infer --resume` 只用于推理恢复。

冻结 SSL/视觉编码器只从本地加载已登记权重，不通过模型 ID 隐式下载。更换编码器、预处理
或媒体内容会使用新的特征缓存；CLI、评测等无关代码变动可复用缓存，运行报告仍记录全源码版本。
本轮切换到缓存v2，旧缓存保留，新键首次使用时重新生成；损坏缓存会明确报错。

## 4. 推理和评测

填好与 checkpoint 对应的推理配置后执行：

```bash
uv run --no-sync deepfake-detection-archive infer \
  --config configs/infer-audio-waveform.local.json
uv run --no-sync deepfake-detection-archive evaluate \
  --scores outputs/audio-dev-scores/scores.jsonl \
  --manifest data/audio-dev-only.jsonl \
  --output outputs/audio-dev-scores/evaluation.json
```

每个输入产生一个成功或失败记录。内部连续分数越高越倾向伪造；它通常是 logit，不是概率。
评测输出 EER、覆盖率、失败率及可用的分组结果。存在失败、未知标签或缺类时，应读取原因，
不能把成功子集结果当作完整数据成绩。

中断后使用相同配置和输入恢复：

```bash
uv run --no-sync deepfake-detection-archive infer \
  --config configs/infer-audio-waveform.local.json --resume
```

模型、影响评分的配置或输入发生变化时使用新输出目录。对 holdout 推理需显式加入
`--frozen-audit`；该参数记录流程声明，模型和配置仍应在运行前冻结。

## 5. 导出和离线包

本地分数导出：

```bash
uv run --no-sync deepfake-detection-archive export \
  --config configs/adapter.local.json \
  --scores outputs/audio-dev-scores/scores.jsonl \
  --output outputs/local-scores.csv
```

官方格式尚未公布，`adapter.example.json` 是本地适配模板。平台提供样例后更新列名、方向和
失败处理，不改模型内部评分约定。

冻结候选后，构建项目 wheel，并在打包配置中列出运行所需代码、配置、权重、依赖和许可：

```bash
uv build --no-sources
uv run --no-sync deepfake-detection-archive package \
  --config configs/package.local.json
uv run --no-sync deepfake-detection-archive verify-package \
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
export UV_CACHE_DIR=/tmp/aigc-uv-cache
bash tools/check_project.sh
```

本地和 CI 使用同一入口，覆盖主动代码与五个开发工具；历史恢复脚本和归档不做自动格式修复。
工程检查证明实现行为，不代替真实数据上的模型评测。
