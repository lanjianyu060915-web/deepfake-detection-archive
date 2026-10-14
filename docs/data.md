# 数据与模型资源

数据和权重放 Git 外。项目使用 `manifest.jsonl`、`sources.jsonl` 和本地权重配置，
格式模板在 [configs](../configs/README.md)。配置相对路径以配置文件目录为准。

## 当前实例的资产

| 资产 | 实际位置 | 当前用途 |
| --- | --- | --- |
| UFD强视频对照 | `/var/tmp/deepfake-video-ufd-20261014/` | 完整检测器、固定配置与三组分数；公开头负结果；最小资产已另存下方C6持久归档 |
| 冻结CLIP视频头 | `/var/tmp/deepfake-video-clip-head-20261014/` | 双seed头与三组完整分数，冻结43为研究候选；复用UFD骨干，最小资产已另存C6，不替换release |
| 统一入口研究包 | `/var/tmp/deepfake-offline54-20261014/relocated/deepfake-v0.1.2-rc1/` | 真实v0.1.1权重、包内解释器和依赖；搬移、440条profile及真实语音万级已完成，未重发Release |
| 三域联合音频冻结候选 | `/srv/shared-data/candidates/audio-universal-20261014/` | seed42主/seed43备权重、sidecar、原始配置/历史、完整dev与冻结评分；文件0444、目录2755，研究用途 |
| 音频补充诊断媒体 | `/srv/shared-data/datasets/audio-review-20261014/` | AIME未训练路径154条、AI-OpenBMAT纯音乐94/混合152条；原清单与媒体哈希保留，只作已见研究诊断 |
| ASVspoof LA Parquet | `/srv/shared-data/datasets/asvspoof2019-la/` | 官方 train/dev 物化输入；#9 已接入协议与 FLAC |
| ASV 训练清单/媒体 | `/var/tmp/lanjianyu-issue9-20261011/data/main/`、`data/LA/` | #9 已完成四次训练和完整dev评分 |
| 音频检测器及分数 | `/var/tmp/lanjianyu-issue9-20261011/runs/` | LogMel/HuBERT各两种子，见云端音频实验卡 |
| 语音泛化共享发布 | `/var/tmp/deepfake-audio-generalization-20261011/` | `data`引用#19数据；`outputs`保存原始分数引用和可读评测摘要；HuBERT引用#9可读bundle |
| HuBERT 本地 bundle | `/var/tmp/lanjianyu-issue9-20261011/hubert/models/hubert-base-ls960/` | #9 已恢复固定版本，内部语音研究 |
| ViF Panda ZIP | `/srv/shared-data/datasets/vif-panda/` | #10 已导入并复现受控划分 |
| 七生成器视频训练及分数 | `/var/tmp/deepfake-video-multigen-20261013/` | #40：2秒统一媒体，train3,614/dev844，双种子及Hunyuan对照；配置与结果见[实验卡](experiments/video-multigen-20261013/README.md) |
| 受控视频资产及检测器 | `/var/tmp/lanjianyu-issue10-20261011/` | `control/`清单、`resnet18-a1/`编码器、`seed42/`与`seed43/`检测器和分数 |
| FakeSound ZIP | `/srv/shared-data/datasets/fakesound/FakeSound.zip` | 固定原包、单类诊断，不能训练二分类 |
| 配对环境音 | `/var/tmp/lanjianyu-issue11-20261011/doppelganger-v2/` | 462 条实际可读媒体，附 manifest/sources/provenance |
| 环境音检测器与分数 | `/var/tmp/deepfake-nonspeech-logmel-20261011/` | LogMel/HuBERT双种子及本地FP32推理；团队可读 |
| AST预训练、环境音微调与分数 | `/var/tmp/deepfake-audio-ast-20261011/` | `pretrained/ast-audioset/`固定预训练资源；四次训练、双种子dev及冻结候选holdout推理；团队可读 |
| EnvSDD 城市环境音扩展 | `/var/tmp/deepfake-audio-expand-20261011/` | `envsdd-urban/` 19,215条；`unseen-dev/`、`unseen-holdout/` 为未训练生成器评测；四组AST、冻结候选和分数团队可读 |
| AIME音乐与混合训练 | `/var/tmp/deepfake-audio-music-20261011/` | `aime/`主清单630条、105个描述组合；`mixed/`保留Urban train并加入音乐；固定原包、模型与分数团队可读 |
| AIME扩展与混合候选 | `/var/tmp/deepfake-audio-music-20261012/` | 119个固定分片；主清单1,914条、319个描述组合；旧混合seed43及扩展seed42的权重与分数 |
| AST时频遮挡与广播外部诊断 | `/var/tmp/deepfake-audio-regularize-20261012/` | 扩展模型第二种子、两档遮挡训练及分数；`ai-openbmat/`含246段外部探索dev；原始子集在`upstream/ai-openbmat/` |
| 对称音乐/语音混合与匹配对照 | `/var/tmp/deepfake-audio-speech-mix-20261012/` | `mixtures/`含3,464个波形变体；`pure-replay/`、`speech-mix/`为同规模联合训练清单；固定源码、权重及九桶评分按任务保存 |
| 语音第二种子与 In-the-Wild | `/var/tmp/lanjianyu-issue39-20261013/` | `runs/`为#39三次LogMel训练与评分；`data/release_in_the_wild/`为31,779条外部诊断媒体，`data/in-the-wild-manifest/`为测试清单 |
| FakeSound 诊断清单 | `/var/tmp/lanjianyu-issue11-20261011/fakesound-v1/` | data_root 为其 `media/`，含缺失 real 记录 |

共享原媒体和模型只读；`outputs/`、`cache/` 使用各自任务目录。
`/var/tmp` 产物当前可定位，但不是已确认的持久卷；候选的最小集合已另存[持久归档](#持久归档)。
音视频任务已交付的目录组为 `dlteam`，团队可读；实际配置与权重标识见实验卡。

## 持久归档

`/var/tmp` 不保证跨实例保留。保留候选的最小集合已复制到矩池云持久网盘
`/mnt/deepfake-archive/`（2026-10-14，#50），原副本保留不动。

| 目录 | 内容 |
| --- | --- |
| `C1-release-v0.1.1/` | 发布包两份权重、sidecar、配置、`release.json`、review各桶分数 |
| `C2-audio-universal/` | 三域AST seed42/43及两个对比组权重、完整dev与冻结评分、7份清单 |
| `C3-speech-logmel-bandwidth/` | LogMel带宽seed42/43与两次白噪声运行、各条件分数与比较、训练/dev/条件清单、In-the-Wild清单 |
| `C4-video-multigen/` | 五次运行权重（冻结为`paired-seed43`）、配置、`canonical-2s`与统一诊断清单、全部评分 |
| `C5-video-temporal/` | 时序头seed42/seed43-retry权重、分数、训练清单 |
| `C6-video-ufd-clip-20261014/` | UFD完整编码器与公开头、冻结CLIP双seed头、配置和三组完整分数；31内容文件约1.219GB |
| `shared/` | `resnet18-a1/encoder.pt`（视频共享基座）与`ast-audioset/`（AST预训练基座） |

原C1–C5与shared共396个内容文件、约3.07GiB，不含原始媒体、特征缓存和可再生的发布分片。
`index.json` 逐文件记录原路径、归档路径、大小和SHA-256；配置中的原路径按该表替换即可指向归档。
C6是后来完成的独立批次，另31内容文件，由其目录内 `index.json` 与 `SHA256SUMS` 登记，
不改写原396文件索引。配置保留原路径，恢复时按索引重映射；C6引用原C4数据清单，不重复备份媒体。
校验：

```bash
cd /mnt/deepfake-archive && sha256sum -c SHA256SUMS
```

恢复演练：三域AST seed42从归档权重经发布包 `infer` 重评两桶各40条，与原分数逐位一致；
七生成器视频 `paired-seed43` 从完整归档权重经仓库源码入口重评20条，最大差4.9e-3。
对照同时改变了缓存/直接解码路径和批量64/8，差异可能与这些条件有关，尚未单独确认原因。发布包v0.1.1不能加载该视频权重
（缺少 `temporal_kernel_size` 字段），需要仓库源码入口。

各候选的归档资产与前端快照在 [`configs/release/candidates.json`](../configs/release/candidates.json)：
完整鉴伪权重或所需基座的归档路径与SHA-256、代码版本、适用范围、研究/发布状态，以及
从当前环境实际读出的有效预处理（AST频谱前端常量、LogMel前端、视频帧采样与RGB mean/std）。
约束：AST前端依赖环境中**没有** torchaudio（装上后 transformers 会静默改用 kaldi fbank）；
视频缩放裁剪全部由 ffmpeg 完成，不能再叠加 timm 的 `crop_pct`；ffmpeg 由主机提供。
该文件仅记录本次归档时的事实，不作为运行配置。统一入口 #57 由
`configs/release/models.json` 及其引用的推理配置选择模型；完整 checkpoint 包含模型参数，
视频 ResNet18 基座仅用于训练初始化，恢复推理无需额外加载基座。
C6的CLIP特征头则需要显式加载同批保存的视觉编码器。以持久路径标准 `infer` 恢复2条原诊断，
与原分数逐值一致；源码/候选用途与限制见[本轮运行包](experiments/offline-entry-20261014/README.md)。

网盘为 `fuse.fx`：`chgrp` 不生效（gid固定为2000），文件统一 0444 全员可读；
`du` 显示为0，大小以 `index.json` 或 `stat` 为准。原始媒体与特征缓存仍只在实例上，需按各工具重建。

## 导入与训练清单

| 工具 | 输入与目的 |
| --- | --- |
| `tools/finalize_asvspoof2019.py` | 已完成的官方 ZIP 发布与提取 |
| `tools/prepare_asvspoof2019.py` | 官方协议和 FLAC → 标准清单 |
| `tools/prepare_asv_parquet.py` | 固定共享 Parquet 与官方协议 → 原始 FLAC；不重编码 |
| `tools/prepare_in_the_wild.py` | 已解压的 In-the-Wild 发布 → 按说话人分组的 test 清单；只作冻结候选的外部诊断 |
| `tools/prepare_asv_eval.py` | 固定官方eval协议与镜像Parquet → 未知攻击研究清单；既有发布直接复用 |
| `tools/prepare_asv_main.py` | 既有 ASV 清单 → 训练子集；历史单条排除应显式配置 |
| `tools/prepare_vif_panda.py` | 固定 ViF ZIP → Panda 媒体、来源与配对关系 |
| `tools/prepare_video_control.py` | 既有 ViF 与 pilot 划分 → 受控 train/dev 与元数据对照 |
| `tools/prepare_video_multigen.py` | 已登记Panda ZIP与既有划分 → 关联排除、配对的2秒统一媒体；保持train/dev，失败逐条记账 |
| `tools/prepare_doppelganger.py` | 固定配对表/原媒体元数据 → 环境音研究清单 |
| `tools/prepare_fakesound.py` | 固定 FakeSound ZIP → 诊断清单和失败记账 |
| `tools/prepare_envsdd.py` | `import`：五成员配对环境音，排除旧研究集关联并重划分；`eval`：同一已划分原录音上的 AudioLCM/TangoFlux 诊断 |
| `tools/prepare_aime.py` | 固定 AIME Parquet → 描述匹配音乐、prompt隔离与未训练生成路径诊断；不推断原曲/artist关系 |
| `tools/prepare_ai_openbmat.py` | 配对广播WAV/JSON → 共同纯音乐及音乐/语音区间的4秒外部dev；按交叉淡化修正时轴，保留来源和失败 |
| `tools/prepare_audio_mixtures.py` | 已划分AIME音乐 + 同split真实ASV语音 → 对称RMS混合与纯音乐重采样对照；保留父音乐、speaker、SNR、输入SHA与失败 |

参数见各工具 `--help`，已有输入直接复用，不重复下载或物化。
通用 `index` 只列媒体；标签、许可和已知组需来源证据，不能从目录猜测。
已有可信清单直接训练，实际解码在准备张量时完成，无需另做一遍全量预解码。

环境音已可直接接入：

```bash
mkdir -p data
# 在自己的新工作副本中建立一次链接；已有同名目录时使用其实际配置
ln -s /var/tmp/lanjianyu-issue11-20261011/doppelganger-v2 data/issue11-doppelganger
cp configs/train-logmel-doppelganger.example.json configs/train-logmel-doppelganger.local.json
scripts/devctl run python -m deepfake_detection_archive train \
  --config configs/train-logmel-doppelganger.local.json
```

GPU 有余量即可执行训练，并在任务 Issue 记录占用。
该清单为 train322/dev70/holdout70、153 个连通组；
`manifest.jsonl` 为可训练样本，`inventory.jsonl` 保留全部572条选中项及排除记录。
最终 manifest SHA-256：`18e30516a73959cca46c55000617b73065f1141d5c7b08d097eba95c7d51ce4b`。
完整获取命令、来源版本和处理规则见[环境音实验卡](experiments/nonspeech-20261011/README.md)。

EnvSDD 扩展使用[固定分片与署名资源](research/source/envsdd/README.md)，
下载完成后在新输出目录导入：

```bash
scripts/devctl run python tools/prepare_envsdd.py import \
  --parquet-dir /var/tmp/deepfake-audio-expand-20261011/upstream/envsdd \
  --credits docs/research/source/envsdd/FREESOUNDCREDITS.txt \
  --urban-metadata /var/tmp/deepfake-audio-expand-20261011/upstream/UrbanSound8K.csv \
  --exclude-manifest data/nonspeech-logmel-20261011/manifest.jsonl \
  --output /var/tmp/deepfake-audio-expand-20261011/envsdd-urban
```

导入只取 UrbanSound8K；同一原切片的 real 与四种生成方法成组使用。
原切片短于3.99秒、旧研究集关联及失败项整组排除并保留库存，
所有使用媒体对称处理为4秒、单声道16kHz、峰值0.95的PCM16。
按原录音与已知上传者重新划分，不沿用存在同录音交叉的上游 train/validation。
本轮可用19,215条，train13,450/dev2,885/holdout2,880；238个连通组。
清单 SHA-256：`eebe2feb8b9fe424ee03eedd7286a434d01b6420b6068667b23306240e0e3a22`。

未参与训练的 AudioLCM/TangoFlux 使用独立的 `upstream/extra/` 分片，
仅按指定 split 的原文件名关联，不重新划分；当前 dev 为577条 real、每生成器577条 fake。
已发布数据可直接使用，不重复执行以下物化命令：

```bash
scripts/devctl run python tools/prepare_envsdd.py eval \
  --parquet-dir /var/tmp/deepfake-audio-expand-20261011/upstream/extra/envsdd \
  --manifest data/audio-expand-20261011/manifest.jsonl \
  --data-root data/audio-expand-20261011 --split dev \
  --output /var/tmp/deepfake-audio-expand-20261011/unseen-dev
```

真实媒体复用既有处理结果，生成媒体执行相同归一化。评测解码失败保留在清单及分母中。

AIME音乐子集获取90个固定分片、共9,087,224,372字节，逐包官方SHA见
[来源登记](research/source/aime/README.md)。已发布数据直接复用；新目录导入命令为：

```bash
scripts/devctl run python tools/prepare_aime.py \
  --parquet-dir /var/tmp/deepfake-audio-music-20261011/upstream/aime \
  --resources docs/research/source/aime/resources.json \
  --output /var/tmp/deepfake-audio-music-20261011/aime --seed 42
```

主清单105条real、525条fake；train438/dev96/holdout96，零解码失败。
MusicGen Small/Medium、AudioLDM 2 Large、Riffusion、Mustango参与训练。
`unseen.jsonl` 另保留dev/holdout各16条real和9条未训练路径fake；训练不读取该清单。
这些真实音乐已包含在主清单，不能相加当作独立媒体数量。
按三标签描述组合和已确认重复隔离；未知原曲、artist与album不当成已确认独立实体。
`aime-prompt:` 只记录划分隔离关系；统计分组返回 `unknown`，仍可报告描述性EER，
不能用描述组合数支持独立artist/原录音的bootstrap结论。音乐/语音派生物即使有已知
speaker，音乐artist仍未知，因此包含该关系的整个连通组保持统计未知；未来补齐
原曲/artist证据后需重新整理清单，不能凭speaker字段自动解除限制。
新导出dev分数使用同一统计分组规则；直接评测旧分数时，原始 `aime-prompt:` 也不计为
已知统计组，bootstrap拒绝此类输入。旧流程若已把prompt名称改写成样本ID，分数本身
无法恢复语义，需用原始清单重新评测；不能直接沿用历史独立组计数或区间结论。

扩展轮复用旧90个分片，新增29个，主清单train1,338/dev288/holdout288，
预留音乐路径dev154条（48real/106fake）；旧697条已分配记录保持split，0冲突。
扩展导入传入 `--previous-manifest`，命令与结果见
[音乐扩展实验](experiments/audio-music-expand-20261012/README.md)。
联合Urban训练14,788条，新候选已见/未训练音乐dev EER0.4167%/18.75%；
FSD迁移退化，本轮未评估holdout。数据和逐曲许可限制没有因扩容而解除。

AI-OpenBMAT固定128对原始成品，按共同区间准备混合152段、纯音乐94段，0准备失败。
均为外部探索dev，不加入训练；标签沿用上游，Suno Extend真人提示前缀是否移除未知，
不能保证每条AI侧4秒窗口均含生成音乐。重复音乐/广播素材须在未来训练前连接分组。
底层BAF素材许可衔接未解决，正式交付未批准。
准备命令及来源见[AI-OpenBMAT登记](research/source/ai-openbmat/README.md)，
实际模型评分见[广播内容实验](experiments/audio-ast-regularization-20261012/README.md)。

音乐/语音混合轮复用AIME及ASV已发布train/dev；真实语音池为606/768条，
原speaker、utterance及已知文件哈希不跨split。按音乐group与SNR共用同一语音，
分别生成整段RMS music/speech -5、+5dB变体，3,464条均成功。
混合train2,676条，已见混合dev576条，未训练路径混合dev308条；两dev桶共享96条real。
纯音乐重采样对照引用原音乐两次，两组联合Urban/AIME均train17,464、共同选模dev864。
未训练生成路径仅诊断，不读取holdout媒体。派生素材适用AIME和ASV许可交集，
正式赛用及AIME原曲/artist关系仍未确认；合成混合不等同真实广播内容。
准备规则、训练配置与结果见[对称混合实验](experiments/audio-speech-mix-20261012/README.md)。

## 划分与许可

内部 `real=0`、`fake=1`；train 训练，dev 早停/选择，holdout 在候选冻结后验收。
同一原媒体、派生、已知身份或确认重复组不跨 split。候选关系保留为候选，不据此捏造身份。
`split` 保留原 `group_id`，不会以统计单元覆盖隔离关系。audit的 `group_count` 及
覆盖桶 `groups` 是隔离连通组数；`known_statistical_groups` 与
`unknown_statistical_samples` 单独报告统计证据。`counts_sufficient` 需各类至少
50条、10个已知统计组且无统计未知样本，只是计数检查，不证明泛化、许可或媒体可用性。
失败保留在库存及覆盖率分母。真实/生成执行对称采样和媒体处理，避免来源成为标签代理。

| 来源 | 当前登记范围 | 材料 |
| --- | --- | --- |
| ASVspoof/VCTK | 来源许可下的研究训练；正式赛用未确认 | [官方许可与归属](research/source/asvspoof2019/) |
| ViF Panda/HD-VILA | 非商业计算研究，正式赛用未确认 | [固定来源与许可](research/source/vif-panda/provenance.json) |
| Doppelganger/FSD50K | CC0 原媒体、CC BY 集合、生成音频附 Stability 条款；内部非商业研究 | [来源/许可和排除项](experiments/nonspeech-20261011/README.md#许可与归属) |
| FakeSound | 媒体权利 unresolved；不批准训练/正式交付 | [诊断结论](experiments/nonspeech-20261011/README.md) |
| EnvSDD/UrbanSound8K | 已取得原录音署名；CC BY 4.0 集合与 CC BY-NC 3.0 原媒体下的内部非商业研究，正式赛用未确认 | [来源与原始 credits](research/source/envsdd/README.md) |
| AIME/MTG-Jamendo | 生成音乐CC BY 4.0、标签CC BY-NC-SA 4.0；真实音乐逐曲CC许可未映射，仅内部非商业研究，正式交付未批准 | [来源与用途](research/source/aime/README.md) |
| In-the-Wild | 作者在 HuggingFace 标注 CC BY-SA 4.0，底层网络媒体权利未核实；登记 unresolved，仅外部诊断 | [实验卡](experiments/speech-robust-20261013/README.md#in-the-wild-外部诊断) |
| AI-OpenBMAT/BAF | 发布CC BY 4.0与BAF底层素材限制衔接未解决；仅外部探索诊断，正式交付未批准 | [来源、标签与许可限制](research/source/ai-openbmat/README.md) |

代码、权重、数据许可分别处理。公开可下载不代表正式比赛使用已获批；
比赛外部资源要求集中在[比赛要求](competition/requirements.md)。

## 预训练模型

固定来源、模型卡和许可在 [pretrained-resources](research/pretrained-resources/README.md)。
HuBERT 为语音表征，ResNet18 为当前全帧空间表征，AST 为 AudioSet 通用音频表征。
这些预训练资源需要另行训练鉴伪头或微调；已训练的检测器位置见上表和实验卡。
必要时运行 `tools/prepare_pretrained.py` 获取/转换资源；权重和生成登记落在
`models/pretrained/`，下载缓存在 `cache/pretrained/`。推理只加载已准备的本地资源。
AST 使用 `scripts/devctl run python tools/prepare_pretrained.py prepare-ast` 获取；
当前工作副本的 `models/pretrained/ast-audioset` 链接到上述共享只读 bundle。

## 旧实验资源恢复

旧全量 LogMel42 checkpoint 未在云端定位；历史路径为
`outputs/experiments/asv-main-20261009/logmel-seed42/training/best.pt`，SHA-256
`bb1a70097a9b0159c9febb56bb8c757ac5ccb9b2ae43e571f01b90cc70b37d8c`。
历史状态包 `outputs/handoff/20261010/research-state.tar.gz`，SHA-256
`5d3880c3e19548e3c3f7f4e3bfa0150bca3bcb05f08320d7df444505c9757e90`，曾包含
固定清单、小模型与分数；本实例尚未确认该包存在，#9 已补跑同条件双种子基线。
旧训练清单哈希为 `8b204f61a679a2789280a731aae1cefb63a29bd6fe009e66b0b34347aa148257`。
完整历史资源索引保留在整理前 Git 提交 `4bbd292`，不作为新训练的额外前置门。
