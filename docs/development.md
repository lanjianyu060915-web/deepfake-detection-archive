# 开发与实验使用指南

本指南对应当前本地工程接口。官方输入、分数方向、平台硬件、时限和外部资产细则仍以
[赛题要求及待确认项](competition/requirements.md) 为准。模板和随机/合成测试用于证明
程序行为；真实鉴伪性能必须由合规、独立且具有覆盖面的数据实验提供证据。

## 1. 准备环境与配置

从仓库根目录执行：

```bash
export UV_CACHE_DIR=/tmp/aigc-uv-cache
uv sync --locked --group media --group research --extra cu130
ffmpeg -version
ffprobe -version
uv run --no-sync deepfake-detection-archive --help
```

FFmpeg 和 ffprobe 是系统可执行文件，需在 `PATH` 中；Python 依赖同步不替代系统媒体
工具安装。本地 RTX 4060 8 GB 已获使用授权；CUDA 不可用时程序报告错误，CPU 调试需
明确切换 `device`，训练时同时关闭 AMP。CPU wheel 环境改用 `--extra cpu`，两个后端
互斥。安全屋按其实际系统、GPU、驱动与离线安装约束准备对应环境。

所有起点配置在 [configs/](../configs/README.md)。先复制 `.example.json` 到同目录下的
实值配置，再编辑全部 `REPLACE_*`、零哈希和未确认字段。配置中的路径相对于配置文件；
直接传给 CLI 的 `--root`、`--manifest`、`--output` 则相对于当前工作目录。
JSON 不支持注释；额外或拼错的配置字段会被拒绝。

## 2. 登记样本、来源与划分

`manifest.jsonl` 每行一个 `SampleRecord`，至少含 `sample_id`、相对媒体路径、
`modality`、`label` 和 `source`。内部约定 `real=0`、`fake=1`；未获得可信标签的输入
使用 `null`，不能参加监督训练或计算有标签 EER。`sample_id` 全局唯一，`path` 相对于
`data_root`，禁止绝对路径与 `..` 穿越。

可以先为本地合法持有的媒体建立索引：

```bash
uv run --no-sync deepfake-detection-archive index \
  --root data/audio --modality audio --source YOUR_SOURCE \
  --output data/audio-index.jsonl
```

索引不推断真假标签、人物身份或许可。按可信来源补齐标签、原媒体/派生链、
`original_id`、`parent_id`、`identity_id`、`derivation_group_id`、`group_id`、
生成器、内容类型和 `license_id`。这些关联决定哪些媒体必须留在同一划分中；
未知元数据用 `unknown`，不能捏造独立身份。

`index` 默认输出 `split="test"`，供未知标签的推理输入使用。用于训练研究的记录在获得
可信标签后，须明确改成 `split="unassigned"`，再执行下面的审计和划分；比赛未标注
输入保留 `test`。划分入口会拒绝 `test`，不会擅自把测试输入转成训练数据。

`sources.jsonl` 每行一个来源登记，记录来源 URL、版本、获取日期、许可及许可页面、
允许用途和可用的原始归档 SHA-256。核对真实许可后才设置 `license_status="approved"`
和适用的 `allowed_uses`；训练入口要求其中包含 `training`。样本与来源的 `license_id`
须一致。源码、数据、预训练权重分别核对许可；公开可下载不自动等于可用于比赛。

文件哈希来自实际字节，例如：

```bash
sha256sum data/audio/example.wav
```

核对已登记信息后，复制并填好 `configs/audit.example.json`，对尚未划分的清单执行
真实解码审计，再对输出的已解码清单划分：

```bash
uv run --no-sync deepfake-detection-archive audit \
  --config configs/audit.local.json --decode
uv run --no-sync deepfake-detection-archive split \
  --manifest data/audio-decoded.jsonl --output data/audio-split.jsonl --seed 42
```

上述文件名须与实值配置一致。检查 `audit.json` 的 `passed` 和逐项 issues；命令产生了
报告不代表数据合格。`decode_status="ok"` 应由实际解码产生。文件损坏、来源不合格或
哈希改变时应解决原问题，不能直接改状态掩盖失败。

划分按哈希、身份、原媒体、父子派生与组关系的连通分量进行，不能将帧、切片和增强副本
随机分入不同集合。`duplicates` 命令产生近重复候选；核对后将可信关系写入
`derivation_group_id` 或原媒体/父子关系，再生成划分，不能把候选报告当作已经自动合并。
仅有近重复证据时不要填写 `group_id` 或 `identity_id` 来声称统计独立性；这些字段需有
人物、说话人或独立来源组的登记依据。显式运行全清单 `audit --decode` 会读取
其中所有媒体；应在数据治理阶段完成，冻结后的 holdout 不用于探索性建模。

## 3. 训练和冻结候选

四个训练模板分别覆盖 AASIST-L、log-mel CNN、冻结音频 SSL 编码器加媒体级头、
冻结视频编码器加媒体级头：

```bash
uv run --no-sync deepfake-detection-archive train \
  --config configs/train-logmel.local.json
```

训练只使用 `train/dev`。全 manifest 的划分元数据用于排查与 holdout 等集合的连通
泄漏关系；holdout 媒体不会进入训练 runner 的文件审计、解码、特征缓存或早停评测。
每个 train/dev 划分须同时含两类，所有选中样本都须审计和准备成功。
许可、哈希、已有解码状态或划分审计失败时先检查 `audit.json` 和 `experiment.json`；
通过审计后的缓存/实际解码准备失败逐样本写入 `decode_failures.json`。
两阶段任一失败都会终止本轮训练，避免静默删除困难样本。

训练在模型创建前固定种子。CUDA AMP 与按真实样本数归一的梯度累积支持不等 batch 和
末尾不足一个累积窗口的情况；`history.json` 记录实际 optimizer updates 和 AMP 跳步。
`training.amp_dtype` 支持 `float16` 和 `bfloat16`。为兼容旧配置，未填写时使用
`float16`；当前四个模板显式选择 `bfloat16`，与 RTX 4060 的模型 profile 精度一致。
仅 `amp=true` 且 `amp_dtype="float16"` 时启用 GradScaler；bfloat16 直接执行
有限梯度检查并记录真实更新。设备不支持原生 CUDA bfloat16 时明确报错，不自动改用
其他精度。CPU 模式仍要求 `amp=false`。精度性能及稳定性结论需以实际模型、数据和
显存实测为准，随机模型的吞吐验证不能证明真实鉴伪效果。
先 profile batch、显存和吞吐，再调整 workers、batch、精度与累积。多进程 loader 使用
spawn，worker 等待超过 30 秒会明确失败；限制 Unix socket IPC 的沙箱可使用默认
`num_workers=0` 或在已授权的普通本地环境验证。

直接波形训练目前要求 `segments=1`；AASIST-L 使用 16 kHz 和 64600 samples。冻结
音频特征头可用多段，视频特征头使用多帧；模型将 `[B,N,D]` 聚合为每个媒体一个 logit，
训练和评分复用相同聚合。语音 SSL 分支对音乐、环境音等的有效性需要单独验证，帧级
视觉编码器加媒体聚合也不构成已验证的时序伪造检测。

音频 SSL bundle 只接受本地完整、逐文件哈希验证的 safetensors 资源。需要登记
`config.json`、`preprocessor_config.json`、所有权重分片以及存在时的索引文件。
禁止通过 Hub 模型 ID、隐式下载或远程代码加载。视频裸特征权重必须与所选架构严格
匹配，图像尺寸与编码器 `input_size` 一致；当前共享预处理采用配置中的 FFmpeg
resize/pad/crop 加编码器均值/标准差，实验与推理应保持相同配置。

固定长度波形或冻结特征缓存绑定样本 ID、媒体内容、编码器、预处理和代码哈希，读取前
核验缓存 SHA-256。预算同时包含 tensor 和 UTF-8 sidecar，最大 20 GiB；发布时还需
临时文件空间。缓存损坏会被报告，变更编码器或预处理会生成不同身份的缓存。

输出目录含 `experiment.json`、`audit.json`、`decode_failures.json`、初始权重和
`training/`。后者保存逐 epoch 历史、配置/provenance、开发 EER 最佳 checkpoint
及其哈希。严格更低的开发 EER 决定最佳 checkpoint，`min_delta` 决定早停耐心是否
重置；相同 EER 保留先前模型。`max_seconds` 可选且按 epoch 边界检查，属于软限时。
当前训练不恢复 optimizer/RNG；中断后保留记录，用新的输出目录重启实验，并按新实验
记账，不将推理的 `--resume` 当作训练续跑。

候选冻结记录应包含代码、数据、配置、权重、编码器哈希及候选选择依据，写明开发集
使用范围。所有选择、早停、聚合与阈值调整在开发数据完成，随后进行留出验收。

## 4. 推理、评测与恢复

从推理模板建立只包含目标划分、单一模态的输入清单，填入真实 checkpoint 及哈希，
保持模型对应的采样、编码器和聚合配置：

```bash
uv run --no-sync deepfake-detection-archive infer \
  --config configs/infer-audio-waveform.local.json
uv run --no-sync deepfake-detection-archive evaluate \
  --scores outputs/audio-dev-scores/scores.jsonl \
  --manifest data/audio-dev-only.jsonl --output outputs/audio-dev-scores/evaluation.json
```

`--manifest` 让评测核对完整预期 ID 和逐样本标签、模态、来源、内容类型及生成器；
统计组以清单登记依据重新构建，未知仍为未知。每个输入都有持久化分数行，失败样本
保留 `status/error_code/error_message`，不会静默消失。错误计数以全部预期样本为
分母；存在失败、未知标签或缺类时可能无法报告全量 EER，应同时查看原因与成功子集
结果，不能把成功子集 EER 当作全数据或比赛成绩。

内部连续分数越大越像伪造，通常是 logit，不能当作校准概率。EER 是两类错误率相等
处的插值估计；并列分数组合处理，插值阈值不保证存在一个实际阈值恰好达到该 EER。
需要实际二值决策时，先在开发集冻结阈值，再通过 `evaluate --threshold` 报告。
尚未取得官方评测程序时，这些结果是本地协议结果。

中断后的推理恢复使用完全相同的配置、输入顺序和代码：

```bash
uv run --no-sync deepfake-detection-archive infer \
  --config configs/infer-audio-waveform.local.json --resume
```

恢复重新读取全清单媒体哈希和缺失状态，核对有序输入、配置、模型资源及代码身份，
校验已持久化行，仅截断未完整写入
的尾行；并发写入由 run lock 排斥。变更配置、模型哈希、输入顺序或代码后应开新输出
目录，不能拼接不同身份的结果。失败行也属于已记录结果，恢复不会自动重试已完成的
失败样本；修复后另开可追溯运行。

CLI 对包含 `split="holdout"` 的输入要求显式声明已冻结：

```bash
uv run --no-sync deepfake-detection-archive infer \
  --config configs/infer-audio-holdout.local.json --frozen-audit
```

`--frozen-audit` 是流程声明并写入运行身份，**不从技术上阻止使用者反复运行或调参**。
仍须通过冻结记录、独立验收和结果访问管理保证留出集只承担最终验收职责。

## 5. 导出与离线包

`adapter.example.json` 默认 `official_confirmed=false`。本地导出示例：

```bash
uv run --no-sync deepfake-detection-archive export \
  --config configs/adapter.local.json \
  --scores outputs/audio-dev-scores/scores.jsonl --output outputs/local-scores.csv
```

`real_high` 转换采用分数取负，保留反向排序，不假定分数是 `[0,1]` 概率。存在失败样本
时导出会拒绝；比赛失败处理、列名、范围和文件格式须按官方契约适配。
`--official` 要求显式确认后的契约配置；单独切换布尔值不能构成官方兼容性证据。

构建并在 `package.local.json` 中逐个列出代码 wheel、实值配置、完整权重、许可证和
针对目标平台准备的 wheelhouse 文件：

```bash
uv build --no-sources
uv run --no-sync deepfake-detection-archive package --config configs/package.local.json
uv run --no-sync deepfake-detection-archive verify-package \
  --bundle outputs/candidate.zip --sha256 ACTUAL_BUNDLE_SHA256 \
  --extract-to outputs/candidate-verified
```

模板只示范资产映射，未列举完整依赖或外部编码器资源，不能直接代表可运行安全屋包。
每项源文件路径相对于打包配置；ZIP 内路径由映射键指定。包内推理配置的相对路径须按
解压后的布局调整，例如 `configuration/infer.json` 引用 `../weights/best.pt`。
不要将训练媒体、缓存、历史归档混入包；安全屋输入位置按官方执行契约设置。

打包不会下载、构建或发现缺失的依赖，不允许覆盖已有 ZIP 或已有解压目标；显式列举
最终需要的所有 wheel、编码器配置、权重分片、许可证和运行入口。摘要在可信的独立
记录中保存；包内 manifest 验证字节一致性，不能单独证明来源真实性。

在断网、与平台相符的新环境中实际安装并执行全链路后，才能更新
`offline_rehearsal_status` 并记录耗时、吞吐、峰值显存、错误计数和资源哈希。
官方适配与真实离线演练未完成时，元数据应保留 `unconfirmed` / `not_run`，不能用
工程测试或成功生成 ZIP 替代验收证据。
