# 语音候选的压缩退化与未知攻击实测

2026-10-11，RTX4090，Issue #19。
固定 Issue #9 交付的两个 LogMel 与两个冻结 HuBERT 候选，不重训、不改权重，只回答两个
问题：AAC 64 kbps 与 8 kHz 带宽受限下同攻击 dev 的 EER 退化多少；面对训练时未出现的
A07–A19 攻击，连续分数是否仍成立。

## 候选与端到端路径

四个候选来自 `/var/tmp/lanjianyu-issue9-20261011/runs/`，checkpoint 与哈希记录在
[结果摘要](summary.json)。Issue #9 报告的同攻击基线为 LogMel42 0.7849%、LogMel43
0.9419%、冻结HuBERT42 1.4980%、冻结HuBERT43 1.5306%；本轮用统一端到端路径重测，
LogMel 两个种子逐位复现（0.7849%、0.9419%），两个 HuBERT 得到 1.5306%、1.5339%
（旧值是 BF16 编码缓存分数，新值直接解码 FP32 音频，差异在预期内）。下文所有比较都在
本轮重测的同一条路径内进行。

推理统一为 16 kHz、64,600 采样点、1 segment、`mean` 池化，`infer` 逐样本给出连续分数，
失败样本保留在分母里。FLAC 条件使用进程内 `soundfile` 解码；**AAC 64 kbps 条件必须用
`ffmpeg` 后端**，libsndfile 无法打开 M4A。为确认解码器不构成混杂因素，另做一次交叉检查
（见下），FLAC 输入在两条路径上得到完全相同的波形。

## 数据与来源

| 用途 | 内容 | 数量 | 标识 |
| --- | --- | --- | --- |
| 退化诊断 | ASVspoof2019 LA train/dev 划分出的 dev | 24,844（真2,548/假22,296，A01–A06，20 说话人） | `data/audio-generalization-20261011/data/dev/manifest.jsonl` |
| 未知攻击 | 官方 CM eval 协议标签 + 固定镜像媒体 | 71,237（真7,355/假63,882，A07–A19 各4,914，67 说话人） | `data/audio-generalization-20261011/data/eval/import-summary.json` |
| 未知攻击子集 | 每攻击按说话人轮转取 800 假 + 同说话人 1,600 真 | 12,000（真1,600/假10,400，48 说话人） | `data/audio-generalization-20261011/data/eval-subset/selection.json` |

未知攻击媒体取自固定镜像 `Bisher/ASVspoof_2019_LA@aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c`
的 eval 分片（SHA-256 `8159935a…c86af`，4,381,501,164 字节），标签与攻击 ID 取自官方
`ASVspoof2019.LA.cm.eval.trl.txt`（SHA-256 `84fce576…08036`，由 DataShare 官方归档按
ZIP64 范围读取并校验成员 CRC32）。导入工具要求两个文件的 SHA-256 与清单完全一致、且镜像
元数据逐行等于官方协议，否则拒绝写入任何媒体。

eval 的67个说话人与 train、dev 各20个说话人（合计40个）**没有交集**，三个划分两两不重叠，
因此这一轮同时是未见攻击与未见说话人诊断。ASVspoof2021 与 in-the-wild 仍保留为最终
holdout，本轮未使用。

## 压缩与带宽退化

条件由 `tools/prepare_robustness.py` 生成，原媒体不变，派生媒体只是同一段音频的降质版本：
AAC 64 kbps（16 kHz、保留源声道）、8 kHz 带限后再恢复到 16 kHz（FLAC）。

全 dev 24,844 条，覆盖率 100%，0 失败；ΔEER 为相对同种子 clean 的配对差值
（`compare_scores.py`，`common_coverage` 1.0）：

| 条件 | LogMel42 | LogMel43 |
| --- | ---: | ---: |
| clean | 0.7849% | 0.9419% |
| AAC 64 kbps | 0.9812%（+0.20） | 1.9286%（+0.99） |
| 8 kHz 带限后恢复 16 kHz | 13.0965%（+12.31） | 13.1773%（+12.24） |

两种退化的差别很大：AAC 64 kbps 只让 LogMel42 从 0.78% 升到 0.98%（+0.20 点），
LogMel43 从 0.94% 升到 1.93%（+0.99 点）；而 8 kHz 带限把两个种子都推到 13% 以上，
增加约 12 个点。本轮 AAC 64 kbps 条件的退化小于带限条件，不能推广到其他编码器或码率。
降采样至 8 kHz 后有效带宽约为 4 kHz；结果提示模型对约 4–8 kHz 的信息或相应重采样变化敏感，
下一轮优先测试带宽增强。
AAC 条件的解码器与其它条件不同（见解码器交叉检查），两条路径在 FLAC 上不可区分，
因此这里反映的是编码本身。

## 未知攻击

每攻击子组沿用 Issue #9 的口径：全部真样本 + 该攻击的假样本，因此每行的真样本数相同、
假样本数各 800。子集 12,000 条，覆盖率 100%，0 失败。

| 模型 | 全集 EER | A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LogMel42 | 2.9712% | 0.69 | 0.06 | 0.00 | 1.00 | 0.25 | 0.25 | 0.25 | 1.13 | 2.00 | 0.75 | 11.81 | 1.88 | 2.25 |
| LogMel43 | 4.1442% | 1.31 | 0.50 | 0.00 | 1.31 | 0.63 | 0.63 | 0.56 | 1.31 | 2.38 | 2.25 | 13.94 | 8.38 | 2.06 |
| 冻结HuBERT42 | 6.0096% | 1.75 | 2.75 | 0.38 | 5.00 | 1.31 | 0.56 | 0.06 | 2.88 | 2.50 | 1.50 | 7.19 | 12.38 | 13.75 |
| 冻结HuBERT43 | 6.0096% | 2.06 | 3.00 | 0.38 | 6.06 | 1.88 | 0.50 | 0.00 | 4.13 | 3.25 | 1.75 | 7.19 | 12.50 | 14.56 |

（单位 %，均为 EER。）两个 HuBERT 种子在子集上得到同一个全集 EER，但逐攻击分布并不相同
（例如 A10 为 5.00% 对 6.06%），分数文件哈希也不同，属聚合指标上的巧合而非重复运行。

A17/A18/A19 是四个候选共同的困难族：LogMel42 在 A17 上 EER 11.81%，两个 HuBERT 在 A18/A19
上达到 12%–15%。本子集的 A09 组 EER 为0%–0.38%。已知攻击 A01–A06 的 LogMel dev EER 在 1% 上下，
换到未见攻击后 LogMel42 从 0.78% 升到 2.97%，冻结 HuBERT42 从本轮 clean 的1.5306%升到6.0096%。
这里同时改变攻击族、说话人和样本组成，只能说明在该联合分布上退化，不能将差异单独归因于攻击。

## 解码器交叉检查

200 条 dev FLAC 同时用进程内 `soundfile` 与项目 `ffmpeg` 路径解码：
`bit_identical_waveforms = 200`、`shape_mismatches = 0`、`max_abs_difference = 0.0`。
因此 FLAC 条件用进程内解码（与训练期一致、开销更小），AAC 条件单独使用 `ffmpeg`；
两者在 FLAC 上不可区分，AAC 的差异来自编码本身。

## 复现

共享发布根为 `/var/tmp/deepfake-audio-generalization-20261011/`，组 `dlteam`。
`data` 引用原任务数据，`hubert-base-ls960` 引用 Issue #9 的团队可读 bundle；
`outputs/<run>/scores.jsonl` 引用原始分数，`evaluation.json` 从这些未修改分数重新计算，
12组EER、覆盖率与原摘要一致。原任务的私有metadata/report未冒充共享发布。
在自己的工作副本中建立一次以下链接；已有链接时直接复用对应资产：

```bash
ln -s /var/tmp/deepfake-audio-generalization-20261011 data/audio-generalization-20261011
ln -s /var/tmp/deepfake-audio-generalization-20261011/outputs outputs/audio-generalization-20261011
```

```bash
# 以下准备命令供新目录重建；已发布资产直接使用后面的推理/评测命令。
scripts/devctl run -- python tools/prepare_robustness.py \
  --manifest data/audio-generalization-20261011/data/dev/manifest.jsonl --data-root data/audio-generalization-20261011/data/issue9-asv/LA \
  --output-dir outputs/asv-aac64k-new --condition audio_aac64k
scripts/devctl run -- python tools/prepare_asv_eval.py \
  --parquet data/audio-generalization-20261011/data/eval-parquet/test-00000-of-00001.parquet \
  --protocol data/audio-generalization-20261011/data/eval-official/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
  --attribution-root /var/tmp/lanjianyu-issue9-20261011/protocols --output-root outputs/asv-eval-new
scripts/devctl run -- python -m deepfake_detection_archive evaluate \
  --scores outputs/audio-generalization-20261011/dev-logmel-seed42-clean/scores.jsonl \
  --manifest data/audio-generalization-20261011/data/dev/manifest.jsonl \
  --output outputs/audio-generalization-replay/dev-logmel-seed42-clean-evaluation.json
scripts/devctl run -- python tools/compare_scores.py \
  --baseline-scores outputs/audio-generalization-20261011/dev-logmel-seed42-clean/scores.jsonl \
  --candidate-scores outputs/audio-generalization-20261011/dev-logmel-seed42-aac64k/scores.jsonl \
  --output outputs/audio-generalization-replay/comparison-logmel-seed42-aac64k-vs-clean.json
```

共享发布目录只用于读取已有分数与评测结果，不向它写入新推理产物。
重新推理时，在原配置目录复制一份 `.local.json`，将 `output_dir` 改为新目录
（例如 `../../../outputs/audio-generalization-replay/dev-logmel-seed42-clean`），再运行：

```bash
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/audio-generalization-20261011/dev-logmel-seed42-clean.local.json
```

原任务的私有metadata没有复制，不能把发布目录当作可续跑的原始运行目录。
上面的 evaluate/compare 使用已有只读分数，结果写入自己的 replay 目录。

eval 子集由 `data/audio-generalization-20261011/data/eval/manifest.jsonl` 按“每攻击在 48 个同时含真假样本的说话人间轮转取
800 假，再用同样轮转取 1,600 真”确定，选择结果与清单哈希记录在 `data/audio-generalization-20261011/data/eval-subset/selection.json`；
该规则不参与任何调参，本轮也没有用 eval 结果选择候选。

## 限制

这些数字是项目内部 EER，不是官方 ASVspoof 评分流程（官方指标为 min-tDCF），也没有使用
官方评测脚本；因此不能当作官方成绩。AAC 条件用的是 `ffmpeg` 解码，其余条件用进程内
解码，交叉检查说明两者在 FLAC 上逐比特一致，但 AAC 本身的解码仍然是 libsndfile 不支持的
路径。压缩条件只覆盖 dev 的 A01–A06 已知攻击，未知攻击只报告干净条件下的分数，
两种退化叠加（例如未知攻击再加 AAC）本轮没有做。
