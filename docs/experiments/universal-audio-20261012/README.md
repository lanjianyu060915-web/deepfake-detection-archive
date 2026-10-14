# 三域通用音频联合训练

2026-10-13～14，RTX4090，Issue #41。两档比例、两个种子均完成，冻结等权 seed42 为主候选、
seed43 为备选。三域 dev 改善明显，但未训练音乐与真实混合内容仍弱，**保留研究候选，
不替换现有 v0.1.1 发布权重**。

## 方法与运行

复用 AST 12层、4秒输入（`center_pad_64000_peak095_v1`），按语音/音乐/环境音及真假采样。
注册表68,472条：train40,167 / dev28,017 / holdout288（音乐）；训练只读取 train/dev 媒体。
当前每域恰有一个 source，等权即各域约1/3；另一档为语音/音乐/环境音0.4/0.3/0.3。
音乐只有 prompt 分组，原曲与 artist 独立性未知。

每轮以 dev 分域 macro EER 选 epoch，上限为语音10%、环境音5%、音乐10%。无合格 epoch
才保留最低 macro 的诊断权重，并标记 `selection_gate_passed=false`；本轮四个所选 epoch
均通过门槛。seed42 使用 #45 合并前代码，其原始 `best_dev_eer` 字段实际记录 macro；
下表 pooled EER 从所选 epoch 的 `history.dev_eer` 读取，不改写原始运行记录。

| 配置 | 种子 | 代码 | 完成epoch（所选） | 训练耗时 |
| --- | --- | --- | --- | ---: |
| [joint-domain-equal-seed42](joint-domain-equal-seed42.json) | 42 | `8c81a38` | 18（13） | 12,203 s |
| [joint-speech-40pct-seed42](joint-speech-40pct-seed42.json) | 42 | 同上 | 11（6） | 8,574 s |
| [joint-domain-equal-seed43](joint-domain-equal-seed43.json) | 43 | `49a0efd` | 8（3） | 9,026 s |
| [joint-speech-40pct-seed43](joint-speech-40pct-seed43.json) | 43 | 同上 | 7（2） | 3,106 s |

seed43 配置只改 `training.seed`、`output_dir`，包含 #45 对选模字段与早停计数的修正。
seed42 的40%语音运行目录保留原名 `joint-speech-down-seed42`。四轮均早停，峰值分配
8.45 GiB / 保留9.21 GiB。seed42 两任务并行每档约78样本/秒，单跑约150样本/秒；
seed43 重建特征缓存时与 #39 共用 CPU，等权第3轮耗时5,060 s，其余约550 s。

## dev 与选择

所选 epoch 的 EER（%），pooled 包含完整28,017条，覆盖率均100%、无失败。

| 配置 | 环境音（2,885） | 音乐（288） | 语音（24,844） | macro | pooled | 门槛 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1:1:1 seed42 | 1.603 | 0.000 | 0.510 | 0.704 | 1.135 | 通过 |
| 1:1:1 seed43 | 1.213 | 0.417 | 0.798 | 0.809 | 0.865 | 通过 |
| 0.4/0.3/0.3 seed42 | 1.213 | 0.833 | 0.363 | 0.803 | 0.620 | 通过 |
| 0.4/0.3/0.3 seed43 | 1.733 | 0.417 | 0.549 | 0.900 | 1.501 | 通过 |

双种子 macro 平均：等权0.757%、40%语音0.851%，相差0.095个百分点。音乐 dev 仅48真/240假，
这点差距不足以证明稳定的泛化优势。结合 epoch 稳定性选择等权：40%语音两种子环境音曾达
5.20%/5.89%，超过5%上限；等权两种子全部 epoch 环境音不超过3.64%。
等权内按所选 dev macro 确定主 seed42（`00e6af1c…`）、备 seed43（`18f60bc2…`）。
这是本轮 dev 比较后的选择依据，外部诊断和 holdout 不参与改选。

固定 dev 桶用 v0.1.1 `infer` 运行代码评分，均100%覆盖、无失败；单位 EER %。

| 桶 | 发布版 v0.1.1 | 主 s42 | 备 s43 |
| --- | ---: | ---: | ---: |
| urban-dev（2,885） | 1.386 | 1.603 | 1.213 |
| fsd-dev（70） | 8.571 | 5.714 | 5.714 |
| music-aime96 | 22.50 | 0.000 | 0.000 |
| speech-asv200 | 44.00 | 0.000 | 0.000 |

music-aime96、speech-asv200 来自当前 dev，只说明联合训练改善这些已覆盖条件，不能证明独立泛化。
完整四候选分数摘要见 [summary.json](summary.json)。

## 冻结后诊断

比例与主/备确定后，holdout 各评一次（`infer --frozen-audit`），本次审查保留已有结果，未重跑。

| 集合 | 主 s42 | 备 s43 |
| --- | ---: | ---: |
| fsd-holdout（70） | 8.571 | 5.714 |
| music-holdout（288） | 2.083 | 2.083 |
| In-the-Wild（31,779） | 24.856 | 19.742 |

In-the-Wild 比 #39 带宽 LogMel seed42（30.67%）改善，但仍有明显真实网络语音缺口。
它已是见过的外部诊断集；备选外部表现较好也不改变 dev 确定的主候选。

审查时补齐已有但未评分的音乐/广播视图，使用共享权重、标准 `infer`、64000 samples、
单段、FP32、batch8。每个候选合计400条，100%覆盖、无失败；单位 EER %。

| 已有诊断视图 | 真/假 | 主 s42 | 备 s43 |
| --- | ---: | ---: | ---: |
| AIME未训练音乐路径 | 48/106 | 25.000 | 23.585 |
| AI-OpenBMAT广播纯音乐 | 47/47 | 38.298 | 38.298 |
| AI-OpenBMAT广播混合内容 | 76/76 | 51.316 | 52.632 |

未训练音乐路径包含 MusicGen Large、Suno3.5、Udio，但共享既有 dev 真样本与 prompt 组，
原曲/artist独立性未知。AI-OpenBMAT 是此前用过的探索诊断，存在片段标签纯度与底层权利限制。
这些结果不能当作新盲测；它们说明三域联合尚未解决未训练生成路径和真实混合内容。
不据此继续调整本轮已冻结权重，下一轮优先独立来源数据与更强表征，而非继续围绕此 dev 调参。

## 共享资产与使用

主/备权重、sidecar、原始配置、训练历史、完整 dev 分数和冻结后评分位于：
`/srv/shared-data/candidates/audio-universal-20261014/seed42/`、`seed43/`。
40%语音的配置、历史和完整 dev 评分在同根 `comparisons/`。文件0444、目录2755，团队可读；
`index.json` 登记主/备 SHA-256。新增诊断在各 seed 的 `additional-diagnostics/`。
已按原清单逐文件核对新增诊断媒体哈希，未改标签、分组或划分。

[主候选推理配置](infer-seed42.json)、[备选配置](infer-seed43.json) 默认读取共享 AIME未训练路径：

```bash
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/universal-audio-20261012/infer-seed42.json
```

更换 `manifest`、`data_root`、`output_dir` 即可评分其他输入，模型与预处理参数保持不变。
两份权重已通过标准CLI从共享路径冷加载；这不表示已替换 v0.1.1 发布包。
共享目录仍位于当前实例文件系统，跨实例持久备份尚未确认，由 #50 继续处理。

ASVspoof为ODC-By-1.0，EnvSDD含非商业限制，AIME逐曲条款未决，候选及相关诊断保持研究用途。
原始四轮产物仍在 `/var/tmp/lanjianyu-issue41-20261013/`，完整历史留在 Git 外；
本摘要仅保留选择与复现所需字段。
