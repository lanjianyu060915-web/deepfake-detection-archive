# 开发与运行

从数据清单到模型训练、连续评分及离线交付。实际进度见[状态](status.md)，
算法和执行优先级见[路线图](roadmap.md)，来源和资产见[数据说明](data.md)。

## 环境

项目支持Linux x86_64、Python3.12；需要uv和宿主FFmpeg/ffprobe，FFmpeg须支持`-fps_mode`。
在仓库根目录建立独立环境，锁文件固定依赖；media/research组用于开发和数据导入：

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cpu"
uv sync --locked --group dev --group media --group research --extra cpu
uv run --no-sync python -m deepfake_detection_archive --help
```

仅推理使用`uv sync --locked --no-dev --extra cpu`，不安装上述三个组。
CPU调试配置将`device`设为`cpu`、`amp`设为`false`。NVIDIA GPU使用独立目录，
将`--extra cpu`换成`--extra cu126`或`--extra cu132`，先确认宿主驱动可运行该构建。
三种后端互斥，不使用`--all-extras`。当前锁定Torch2.14.1、torchvision0.29.1，
标准wheel元数据同时固定`+cpu`/`+cu126`/`+cu132`构建标签；三个后端共用CPU torchaudio2.11.0。
默认AST前端的NumPy2.5.3、Transformers5.19.0同时固定在安装元数据中，
避免普通pip安装解析到与发布权重契约不符的补丁版本。
其[stable ABI支持Torch2.11及以后](https://docs.pytorch.org/audio/stable/installation.html)，
Kaldi fbank无需另装TorchCodec。普通预编译wheel不要求指定CUDA Toolkit或nvcc位置。
普通依赖默认从官方PyPI解析，PyTorch使用显式官方索引；`uv.lock`保留版本、来源和wheel哈希。
更换镜像时在独立环境重新生成、检查锁文件，不覆盖团队环境或手改资产哈希。
维护公共锁文件使用`uv --no-config lock --default-index https://pypi.org/simple`，
避免个人uv配置中的附加镜像优先于项目默认源；日常安装使用已有锁，不重新解析版本。

### 安装独立wheel

wheel不包含权重，完整推理安装必须选择一个后端extra。以下是CPU示例，
先在Python3.12虚拟环境安装官方CPU三件套，再安装构建出的wheel：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.14.1+cpu' 'torchvision==0.29.1+cpu' 'torchaudio==2.11.0+cpu'
python -m pip install --index-url https://pypi.org/simple \
  './dist/deepfake_detection_archive-0.1.1-py3-none-any.whl[cpu]'
python -m deepfake_detection_archive --help
```

cu126使用官方`https://download.pytorch.org/whl/cu126`安装`torch==2.14.1+cu126`
和`torchvision==0.29.1+cu126`；torchaudio仍单独从CPU索引安装，最后安装`wheel[cu126]`。
cu132同理，替换torch/vision的构建标签、索引和项目extra。pip不读取`tool.uv.sources`，
extra的精确构建标签用于拒绝错误组合，下载索引仍须显式提供；
不靠`--extra-index-url`选择优先级。数据导入、绘图及开发工具不属于wheel运行依赖。

随后设置`DEEPFAKE_MODEL_PACKAGE`并调用`predict`。运行时只读取本地资源，不执行安装或下载。

### 已验证的安装范围

2026-10-15在Linux x86_64（WSL2、glibc2.39）、Python3.12.13上，
以v0.1.1原始AST/ResNet18鉴伪权重和原始`release.json`，完成以下工程小批对照。
输入为4条已有dev语音和2个自生成视频；固定FP32并关闭matmul/cuDNN TF32，
未改变帧数、窗口或聚合，不用于判断模型准确率。

| 安装环境 | 实际依赖包数 | 成功/失败 | 相对源码CPU的最大分数差 |
| --- | ---: | ---: | ---: |
| 锁定最小CPU + 非editable wheel | 42 | 6/0 | 0 |
| 上述pip命令 + CPU wheel | 43（含pip） | 6/0 | 0 |
| 锁定cu126 + 非editable wheel | 61 | 6/0 | 0.00001240 |

源码已暂时移开，安装代码和模型只读，从中文空格CWD运行并使用空HF缓存。
10份实际解码/前端/视频输入张量完全一致；CPU重复运行分数一致。
cu126实机为RTX4060 Laptop 8188MiB、驱动617.42，
PyTorch峰值allocated402.92MiB、reserved486MiB；torchvision CUDA算子和CPU Kaldi fbank均运行通过。
这是相同权重的小批推理兼容结果，训练容量、批量自适应和其他平台仍需分别验证；
已发布cu132整包的既有服务器结果见[发布说明](release.md)。

### 模型与前端

AST的`frontend`默认`numpy_spectrogram_v1`，旧checkpoint同样按此恢复；
新Kaldi实验显式选`torchaudio_kaldi_v1`，使用独立缓存和输出，不把旧权重改成新前端。
安装torchaudio不会改变旧模型有效前端；当前语义已由#67固定，不重复改实现。

配置及资源的相对路径继续以配置文件所在目录解析。统一预测的模型包根通过
`DEEPFAKE_MODEL_PACKAGE`选择，须包含`configs/models.json`及其引用的运行时元数据，
默认AST使用包根目录的原始`release.json`，与配置、权重一并保留。该根相对当前工作目录，
支持`~`并在入口解析为绝对路径；显式目录缺失/不完整时明确失败，不回退到源码注册表。
无显式选择时，源码checkout使用`configs/release/models.json`，完整离线包使用包内
`configs/models.json`。安装后的wheel不携带默认权重，需明确指定已有模型包。

以下命令沿用已选环境，不触发自动同步。共享服务器和可选`devctl`/direnv说明见
[团队环境](team-environment.md)；公共源码运行不依赖服务器的`/opt`目录或全局激活脚本。

## 数据准备

已有来源优先使用[数据导入工具](data.md#导入与训练清单)。通用媒体可先建立索引：

```bash
uv run --no-sync python -m deepfake_detection_archive index \
  --root data/audio --modality audio --source YOUR_SOURCE \
  --output data/audio-index.jsonl
```

索引不推断标签或许可。补齐来源、标签及已知原媒体/派生/身份关系后，用 `split` 进行组级划分。
划分连通组用于防止已知泄漏，不自动构成统计独立证据。AIME 的 `aime-prompt:` 只表示
prompt 标签隔离，不能替代 artist/原曲关系；这些组及关联派生样本仍整组隔离，统计分组
保留 `unknown`，不能据此做独立组置信区间或宣称跨 artist 泛化。补齐真实来源关系后
应重新整理清单和评测，而不是把未知关系当作不同来源。
已有可信 split 直接训练；需要独立完整解码报告时才运行 `audit --decode`，不重复预解码。
训练会在准备张量时实际解码并记录失败。

## 训练

从 [configs](../configs/README.md) 复制所需模板，填写实际清单、媒体根目录和输出路径。
路径相对配置文件，模板中的权重哈希需使用来源登记值。

```bash
cp configs/train-logmel.example.json configs/train-logmel.local.json
uv run --no-sync python -m deepfake_detection_archive train \
  --config configs/train-logmel.local.json
```

需要在 CPU 机器查看 CUDA 训练配置时，可单独使用 `train --check-config` 只读诊断：解析配置与模型参数，
核验本地资源、train/dev 注册字节、来源登记、完整清单隔离关系、两类覆盖及受约束域覆盖。
它不创建输出目录、加载模型、解码媒体或访问 holdout 媒体；CUDA 配置也可检查。
通过只代表报告列出的静态条件成立；模型权重兼容性、实际解码、缓存张量、GPU 容量、
磁盘写入能力仍不在其检查范围。数据检查失败返回2，详细问题在标准输出 JSON；
配置/资源异常写入标准错误。这是可选诊断；正式训练使用同一个数据审计入口。

| 方向 | 当前实现与模板 |
| --- | --- |
| 音频波形 | LogMel CNN / AASIST-L：`train-logmel.example.json`、`train-aasist-l.example.json` |
| 语音 SSL | 冻结本地 HuBERT ＋轻量头：`train-audio-ssl-head.example.json` |
| 环境音 | LogMel：`train-logmel-doppelganger.example.json`；[多生成器 AST 微调配置](experiments/audio-envsdd-20261011/README.md) |
| 视频 | 冻结逐帧视觉编码器＋共享头/聚合：`train-video-head.example.json` |
| 视频主干微调 | ResNet18 layer4+头：[实际训练配置](experiments/video-finetune-20261011/seed42.json) |

训练只消费 train/dev，dev 早停并选择最佳 checkpoint；holdout 不进入模型选择。
当前视频支持末端主干微调及轻量时序头；帧均值本身不能当作时序学习。
CUDA 支持 BF16/FP16 AMP、梯度累积；新方法先用一次小 profile 确定 batch 和输入规模。
`balanced_sampling` 仅用于 train 的类别平衡；同时开启 `balance_sources` 后，
已存在的来源与类别组合等权抽样，避免小来源被淹没。它不改变 dev 选模指标；
多来源实验需明确选模使用哪个 dev，并分别报告其他来源。
三域训练使用 `selection.metric="domain_macro_eer"` 和
`selection.max_bucket_eer`，域名来自清单的 `content_type`。
每个受约束域必须有真假两类；先满足各域 EER 上限，再按各域宏平均选择 checkpoint。
`best_dev_eer` 保留所选 epoch 的 pooled EER，另记 `best_selection_metric/value`。
无合格 epoch 时保留最低 macro 的研究诊断权重，明确 `selection_gate_passed=false`；
候选冻结时须据此判断资格。未配置selection时保持原 pooled EER选模。
具体三域配置与结果边界见[联合训练实验](experiments/universal-audio-20261012/README.md)。
`feature_batch_size` 控制冻结特征提取批量。
`soundfile` 后端只用于原生目标采样率/单声道；通用媒体使用 FFmpeg。

AST 使用 `architecture="audio_ast_v1"`，`model_config.trainable_layers` 指定可训练末端层数，
末层LayerNorm和分类头始终训练。`pretrained.checkpoint`/`sha256` 指向固定本地
`model.safetensors`；只缓存输入filterbank，微调层保留梯度。
训练产出的checkpoint无需预训练目录；推理复用 `model_kind="audio_waveform"`。
4秒输入可用 `backbone_config.max_length=512`；初始化时按 AST 上游方法居中裁剪
预训练时间位置编码，两个特殊 token 和频率位置保持原序，减少无用 padding。
这会改变注意力序列，需重新训练，不能当成旧1024帧模型的无损推理开关。

对使用4秒居中裁剪、短音频对称补零及峰值0.95的数据，设置
`model_config.waveform_preparation="center_pad_64000_peak095_v1"`、
`segment_samples=64000`、`segments=1`。训练缓存和推理共享该策略；静音保持零。
旧 checkpoint 缺少该字段时保留 `legacy` 行为，迁移已有候选后重新评测再选用。
上游 AST NumPy 前端默认配置会出现一列全零 mel 滤波器告警，项目保持预训练前端，
没有通过修改 FFT/mel 参数或压制告警改变旧模型输入，详见[发布说明](release.md)。

每次新实验使用新输出目录。输出包含 `experiment.json`、配置和资产标识、失败记录、
`training/history.json`、`training/best.pt`、`dev-scores.jsonl` 与 `dev-evaluation.json`。
缓存准备时检查内容，训练 epoch 内直接读取；运行期间缓存作为只读输入。
源码身份保留在实验摘要，无关文档/CLI 变化不作为特征缓存身份。

训练暂不支持 optimizer/RNG 续跑，中断后保留旧目录并在新目录重启。
算法试验需要长时间运行时再补恢复功能；`infer --resume` 不恢复训练。

### 无真实数据的 CPU 回归

在已安装锁定依赖的独立 CPU 环境中，可离线跑合成媒体和小随机模型回归：

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venv-cpu" uv run --no-sync python -m pytest -q \
  tests/test_experiments.py tests/test_offline_dual_branch.py
UV_PROJECT_ENVIRONMENT="$PWD/.venv-cpu" bash tools/check_project.sh
```

第一条覆盖 train/dev→缓存→小模型训练→checkpoint→离线评分、音视频包搬迁以及禁止
隐式下载；完整检查还包含数据/分数契约、失败恢复、静态检查和构建。首次环境准备仍需
锁定依赖及 CI 固定的 ASVspoof 参考文件；“离线回归”不代表空环境能离线安装。
这些是执行契约测试，不是 #39/#40/#41 的真实训练、泛化或性能结果。

## 推理与 EER

无标签媒体可直接 `predict INPUT... --output DIR`，支持文件与音视频混合目录；
源码运行先通过 `DEEPFAKE_MODEL_PACKAGE` 指定本地模型包，
见[统一预测入口](release.md#统一预测入口当前源码)。
原 `predict --config MODEL_CONFIG --input DIR --output DIR` 继续用于显式模型配置。
两种入口复用现有推理逻辑，无需先手写 manifest。

准备与 checkpoint 对应的实值配置；以下示例使用 `data/audio-dev.jsonl`，
推理配置的 `output_dir` 设为 `../outputs/audio-dev`：

```bash
uv run --no-sync python -m deepfake_detection_archive infer \
  --config configs/infer-audio-waveform.local.json
uv run --no-sync python -m deepfake_detection_archive evaluate \
  --scores outputs/audio-dev/scores.jsonl --manifest data/audio-dev.jsonl \
  --output outputs/audio-dev/evaluation.json
```

每样本输出一个连续分数或失败记录。分数越高越伪造，通常为 logit，不保证是概率。
`infer` 和 `predict` 只要存在样本失败就返回退出码2，仍保留逐样本记录及汇总；
全部成功返回0。恢复已有失败记录也返回2，不能把成功写出报告当作推理全部成功。
命令返回 JSON 的 `elapsed_seconds` 包含本次配置/媒体预检、模型启动、推理和结果落盘；
`predict` 还包含 CSV 导出。不含 Python 进程启动和最后的标准输出写入。
EER 需要两类标签；失败计入覆盖率分母，不能把成功子集当作完整成绩。
模型和输入不变时可加 `infer --resume`，变化时使用新目录。
冻结后评估 holdout 使用 `--frozen-audit`，该参数记录声明，不能代替实际候选冻结。

`report.json` 的 `timing.current_session_stage_seconds` 区分输入路径/SHA-256 校验、
解码与 CPU 预处理（`decode_and_preprocess`）、编码器与分类模型前向/聚合/传输
（`model_and_feature_scoring`）；模型内部的频谱等前端仍归入模型阶段。
阶段范围串行且不嵌套，包含各自的 I/O；阶段统计是样本调用耗时的分解，不能再次与总调用耗时相加。
未插桩调用和其余样本开销归入 `unattributed_sample_call`，
`current_session_instrumented_rows` 给出实际有阶段计时的新样本数。
runner 初始化、逐行持久写入和报告计算分别记录；runner 总耗时字段明确截止于报告写入前，
命令级完整耗时看返回 JSON 的 `elapsed_seconds`。阶段均为主机墙钟，
成功调用包含现有阻塞 CPU 特征/分数回传，不新增 CUDA 同步，也不是纯 GPU kernel 耗时；
CUDA 调用异常时尚未完成的异步工作可能落在后续调用中。
阶段统计只覆盖本次新增样本（含失败），旧行仅保留原有总耗时；已完成的 resume 不覆盖旧计时。
本轮不迁移运行时指纹，缓存仍为v2，避免让既有AST filterbank整体重提取。
实际媒体、预处理配置或准备源码变化仍按现有身份规则失效；本轮修改features/encoder
源码涉及的视觉与SSL缓存仍会重新生成。推理resume继续检查输入、配置、模型和代码身份；
迁移运行环境后的数值等价性未获保证，应在新目录评分，不能手改身份元数据。

## 压缩与重采样对照

先取得有价值候选，再对同一批 dev 做配对条件：

```bash
uv run --no-sync python tools/prepare_robustness.py \
  --manifest data/audio-dev.jsonl --data-root data/audio \
  --output-dir outputs/robustness/audio-dev \
  --condition audio_aac64k --condition audio_bandlimit8k
uv run --no-sync python tools/prepare_robustness.py \
  --manifest data/video-dev.jsonl --data-root data/video \
  --output-dir outputs/robustness/video-dev \
  --condition video_h264_crf23 --condition video_h264_crf35
```

每个条件的 manifest/media 配套读取；baseline 清单仍指向原 data_root。
两类处理参数相同，保持 checkpoint、精度、采样和聚合不变，分别 infer/evaluate。
音频比较统一使用 FFmpeg 后端。得到两份分数后：

```bash
uv run --no-sync python tools/compare_scores.py \
  --baseline-scores outputs/audio-original/scores.jsonl \
  --candidate-scores outputs/audio-aac64k/scores.jsonl \
  --output outputs/audio-aac64k/comparison.json
```

报告按 ID 对齐，保留双方失败分母及共同成功子集。ΔEER 为候选减基准，乘100后为百分点。
固定阈值诊断使用基准 dev 选定的同一个 `--threshold`，不在每个条件重新选阈值。

## 导出与离线交付

模型冻结后导出本地连续分数，并打包显式列出的代码、配置、权重、依赖与许可：

```bash
uv run --no-sync python -m deepfake_detection_archive export \
  --config configs/adapter.local.json --scores outputs/audio-dev/scores.jsonl \
  --output outputs/audio-dev/results.csv
uv build --no-sources
uv run --no-sync python -m deepfake_detection_archive package \
  --config configs/package.local.json
uv run --no-sync python -m deepfake_detection_archive verify-package \
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
bash tools/check_project.sh
```

标准检查范围只在 `tools/check_project.sh` 维护，本地与 CI 共用；覆盖 Ruff、严格 mypy、
行为回归、归档字节和构建。CI 用独立 CPU 环境与固定媒体工具链，来源版本见
[工作流](../.github/workflows/checks.yml)。模型实验不依赖反复运行整套环境探针。
