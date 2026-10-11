# 数据与模型资源

数据和权重放 Git 外。项目使用 `manifest.jsonl`、`sources.jsonl` 和本地权重配置，
格式模板在 [configs](../configs/README.md)。配置相对路径以配置文件目录为准。

## 当前实例的资产

| 资产 | 实际位置 | 当前用途 |
| --- | --- | --- |
| ASVspoof LA Parquet | `/srv/shared-data/datasets/asvspoof2019-la/` | 官方 train/dev 物化输入；#9 已接入协议与 FLAC |
| ASV 训练清单/媒体 | `/var/tmp/lanjianyu-issue9-20261011/data/main/`、`data/LA/` | #9 已完成四次训练和完整dev评分 |
| 音频检测器及分数 | `/var/tmp/lanjianyu-issue9-20261011/runs/` | LogMel/HuBERT各两种子，见云端音频实验卡 |
| HuBERT 本地 bundle | `/var/tmp/lanjianyu-issue9-20261011/hubert/models/hubert-base-ls960/` | #9 已恢复固定版本，内部语音研究 |
| ViF Panda ZIP | `/srv/shared-data/datasets/vif-panda/` | #10 已导入并复现受控划分 |
| 受控视频资产及检测器 | `/var/tmp/lanjianyu-issue10-20261011/` | `control/`清单、`resnet18-a1/`编码器、`seed42/`与`seed43/`检测器和分数 |
| FakeSound ZIP | `/srv/shared-data/datasets/fakesound/FakeSound.zip` | 固定原包、单类诊断，不能训练二分类 |
| 配对环境音 | `/var/tmp/lanjianyu-issue11-20261011/doppelganger-v2/` | 462 条实际可读媒体，附 manifest/sources/provenance |
| 环境音检测器与分数 | `/var/tmp/deepfake-nonspeech-logmel-20261011/` | LogMel/HuBERT双种子及本地FP32推理；团队可读 |
| AST预训练、环境音微调与分数 | `/var/tmp/deepfake-audio-ast-20261011/` | `pretrained/ast-audioset/`固定预训练资源；四次训练、双种子dev及冻结候选holdout推理；团队可读 |
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
| `tools/prepare_asv_main.py` | 既有 ASV 清单 → 训练子集；历史单条排除应显式配置 |
| `tools/prepare_vif_panda.py` | 固定 ViF ZIP → Panda 媒体、来源与配对关系 |
| `tools/prepare_video_control.py` | 既有 ViF 与 pilot 划分 → 受控 train/dev 与元数据对照 |
| `tools/prepare_doppelganger.py` | 固定配对表/原媒体元数据 → 环境音研究清单 |
| `tools/prepare_fakesound.py` | 固定 FakeSound ZIP → 诊断清单和失败记账 |

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
| EnvSDD/Urban 辅助材料 | 归属与许可问题未闭合，未进入当前训练主线 | 整理前记录见 Git `4bbd292` |

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
