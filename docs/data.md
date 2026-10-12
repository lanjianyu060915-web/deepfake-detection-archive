# 数据与模型资源

数据和权重放 Git 外。项目使用 `manifest.jsonl`、`sources.jsonl` 和本地权重配置，
格式模板在 [configs](../configs/README.md)。配置相对路径以配置文件目录为准。

## 当前实例的资产

| 资产 | 实际位置 | 当前用途 |
| --- | --- | --- |
| ASVspoof LA Parquet | `/srv/shared-data/datasets/asvspoof2019-la/` | 官方 train/dev 物化输入；#9 已接入协议与 FLAC |
| ASV 训练清单/媒体 | `/var/tmp/lanjianyu-issue9-20261011/data/main/`、`data/LA/` | #9 已完成四次训练和完整dev评分 |
| 音频检测器及分数 | `/var/tmp/lanjianyu-issue9-20261011/runs/` | LogMel/HuBERT各两种子，见云端音频实验卡 |
| 语音泛化共享发布 | `/var/tmp/deepfake-audio-generalization-20261011/` | `data`引用#19数据；`outputs`保存原始分数引用和可读评测摘要；HuBERT引用#9可读bundle |
| HuBERT 本地 bundle | `/var/tmp/lanjianyu-issue9-20261011/hubert/models/hubert-base-ls960/` | #9 已恢复固定版本，内部语音研究 |
| ViF Panda ZIP | `/srv/shared-data/datasets/vif-panda/` | #10 已导入并复现受控划分 |
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
| FakeSound 诊断清单 | `/var/tmp/lanjianyu-issue11-20261011/fakesound-v1/` | data_root 为其 `media/`，含缺失 real 记录 |

共享原媒体和模型只读；`outputs/`、`cache/` 使用各自任务目录。
`/var/tmp` 产物当前可定位，但不是已确认的持久卷；实例回收前由维护者迁入平台持久化存储。
音视频任务已交付的目录组为 `dlteam`，团队可读；实际配置与权重标识见实验卡。

## 导入与训练清单

| 工具 | 输入与目的 |
| --- | --- |
| `tools/finalize_asvspoof2019.py` | 已完成的官方 ZIP 发布与提取 |
| `tools/prepare_asvspoof2019.py` | 官方协议和 FLAC → 标准清单 |
| `tools/prepare_asv_parquet.py` | 固定共享 Parquet 与官方协议 → 原始 FLAC；不重编码 |
| `tools/prepare_asv_eval.py` | 固定官方eval协议与镜像Parquet → 未知攻击研究清单；既有发布直接复用 |
| `tools/prepare_asv_main.py` | 既有 ASV 清单 → 训练子集；历史单条排除应显式配置 |
| `tools/prepare_vif_panda.py` | 固定 ViF ZIP → Panda 媒体、来源与配对关系 |
| `tools/prepare_video_control.py` | 既有 ViF 与 pilot 划分 → 受控 train/dev 与元数据对照 |
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
失败保留在库存及覆盖率分母。真实/生成执行对称采样和媒体处理，避免来源成为标签代理。

| 来源 | 当前登记范围 | 材料 |
| --- | --- | --- |
| ASVspoof/VCTK | 来源许可下的研究训练；正式赛用未确认 | [官方许可与归属](research/source/asvspoof2019/) |
| ViF Panda/HD-VILA | 非商业计算研究，正式赛用未确认 | [固定来源与许可](research/source/vif-panda/provenance.json) |
| Doppelganger/FSD50K | CC0 原媒体、CC BY 集合、生成音频附 Stability 条款；内部非商业研究 | [来源/许可和排除项](experiments/nonspeech-20261011/README.md#许可与归属) |
| FakeSound | 媒体权利 unresolved；不批准训练/正式交付 | [诊断结论](experiments/nonspeech-20261011/README.md) |
| EnvSDD/UrbanSound8K | 已取得原录音署名；CC BY 4.0 集合与 CC BY-NC 3.0 原媒体下的内部非商业研究，正式赛用未确认 | [来源与原始 credits](research/source/envsdd/README.md) |
| AIME/MTG-Jamendo | 生成音乐CC BY 4.0、标签CC BY-NC-SA 4.0；真实音乐逐曲CC许可未映射，仅内部非商业研究，正式交付未批准 | [来源与用途](research/source/aime/README.md) |
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
