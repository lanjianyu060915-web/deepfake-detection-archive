# ASV 云端音频双种子对照

Issue #9，2026-10-11，RTX 4090 24 GiB。LogMel seed42/43 与冻结 HuBERT seed42/43
使用同一 train/dev、64,600 采样点单片段和逆类别频次有放回采样。
四次训练和完整开发集评分均已完成；GPU profile 仅用于确认配置可执行。

## 实际结果

四次真实运行均完成，分别保存24,844条连续dev分数，覆盖率100%、失败0；
四次训练均无跳过的优化更新。GPU任务时段10:50–11:32（Asia/Shanghai），
完整训练流水线10:56:07–11:32:49，前段另含profile与数据准备。

| 模型/种子 | dev EER | 最佳轮次/完成轮次 | 结束原因 | 完整实验秒数 |
| --- | ---: | ---: | --- | ---: |
| logmel-seed42 | 0.7849% | 23/28 | 早停 | 888.69 |
| logmel-seed43 | 0.9419% | 20/25 | 早停 | 756.32 |
| hubert-seed42 | 1.4980% | 26/30 | 30轮上限 | 350.52 |
| hubert-seed43 | 1.5306% | 20/25 | 早停 | 189.19 |

| 模型/种子 | 训练器秒数 | 训练吞吐中位数（条/秒） | allocated/reserved峰值（GiB） |
| --- | ---: | ---: | ---: |
| logmel-seed42 | 733.50 | 1779.09 | 3.24/5.37 |
| logmel-seed43 | 682.00 | 1695.13 | 3.24/5.37 |
| hubert-seed42 | 155.83 | 11008.35 | 0.08/2.84 |
| hubert-seed43 | 132.58 | 11033.37 | 0.08/0.46 |

以上吞吐针对缓存后的训练输入。HuBERT冻结编码器后仅训练分类头，不能把分类头吞吐
当作端到端速度；seed42完整实验包含首次编码，seed43复用缓存。训练器的allocated峰值
不涵盖编码器提取峰值；reserved还受之前编码阶段的CUDA缓存影响，编码资源另见profile。

| 模型/种子 | A01 | A02 | A03 | A04 | A05 | A06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logmel-seed42 | 0.1884% | 0.1962% | 0.1076% | 0.7849% | 0.8881% | 1.1572% |
| logmel-seed43 | 0.2960% | 0.1177% | 0.4710% | 1.3455% | 0.7804% | 1.5699% |
| hubert-seed42 | 0.4037% | 0.3532% | 0.4306% | 0.8242% | 0.3498% | 3.3359% |
| hubert-seed43 | 0.6279% | 0.9812% | 0.6997% | 0.9812% | 0.4317% | 3.2575% |

本设置下，两个LogMel种子的EER都低于两个冻结HuBERT种子；HuBERT的A06误差尤其明显。
因此本轮不将冻结HuBERT mean-pooling头替换为主候选，保留LogMel42作为本轮较优候选。
下一步应比较压缩/重采样退化以及未知生成器、跨来源表现；这些验证尚未执行。
两个种子仅提供当前设置下的重复观察，不构成最终留出验收。

完整统计、checkpoint与分数SHA256见[summary.json](summary.json)，
逐轮曲线见[training-history.json](training-history.json)。四次运行主动源码SHA256一致，
运行时commit均为`b98280f`；首个run为clean，后续因实验文档/汇总脚本编写而记录dirty。
后续同步main和提交实验卡不改写这些实际运行标识。

## 数据与权重

共享输入为 `Bisher/ASVspoof_2019_LA` 固定 revision
`aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c` 的两个 Parquet。
薄导入工具先校验两份固定 SHA256，逐条核对官方 sample ID、speaker、system 和 label，
再按批写出原 FLAC bytes。没有重编码或修改共享输入。

官方 train/dev 协议和 `README.LA.txt` 通过
[DataShare 原始 LA.zip](https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content)
的 HTTP Range 获取，成员 CRC32 验证通过；不重下完整压缩包。
50,224 个媒体的大小与 CRC32 均匹配官方目录，无缺失或不匹配。
CRC32 不是密码学等价证明，整包 MD5 未在本机验证。

完整原生 FLAC 解码 50,224 条、0 失败。继续排除训练样本 `LA_T_1023667`，
最终 train25,379（real2,580/fake22,799）、dev24,844（real2,548/fake22,296）。
跨划分候选的波形相关与旧实验一致，未把未确认候选扩张为新的分组事实。
训练清单哈希与旧实验完全相同：
`8b204f61a679a2789280a731aae1cefb63a29bd6fe009e66b0b34347aa148257`。
但旧 LogMel42 checkpoint/分数未在当前云端恢复，本轮重新训练两个种子，
旧 0.510204% 只作历史记录，不计入本轮双种子统计。

登记仍沿用 ASV 数据集 ID，`sources.jsonl` 的 URL/version/downloaded_at 已明确镜像渠道、
固定 revision 和共享数据取得日期；单一 archive SHA 不适用于两个分片，保留 unknown，
两个分片的哈希见 [inputs.json](inputs.json)。官方 README、ODC-By-1.0 与 VCTK
归属随数据保存，仅据来源条款开展研究训练，比赛外部数据规则仍未确认。

HuBERT 从 `facebook/hubert-base-ls960@dba3bb02fda4248b6e082697eee756de8fe8aa8a`
恢复，复用既有转换逻辑。转换后 safetensors SHA256 与历史登记一致：
`411e8e7f967ba2a68bc6fba072e6374effc390225c7fdb75b8731edd95717e15`。
config、preprocessor、模型卡及 Apache-2.0 许可齐全，离线严格加载无缺失、多余或不匹配权重。
详见 [恢复证据](hubert-restoration.json)。训练时禁止隐式联网获取资源。

## 配置与评分口径

四份实值配置继承旧实验模型和优化参数，仅调整路径：batch256、workers4、BF16 AMP、
AdamW lr0.001、weight_decay0.0001、accumulation1、最多30轮、patience5。
LogMel 从随机初始化训练；HuBERT 编码器冻结、mean pooling 768维，仅训练64维隐藏层分类头。
HuBERT feature_batch_size32；两个种子复用相同的 BF16 特征缓存。

短 profile 保留上述 batch：LogMel allocated3.19/reserved4.77 GiB；
HuBERT 编码 allocated2.35/reserved2.84 GiB。吞吐与测量范围见 [profile.json](profile.json)，
它们使用合成输入、排除解码和磁盘读写，不能当作端到端性能或模型效果。

最佳 checkpoint 按 dev EER 选择，对已准备好的全部 dev 张量用 FP32 分类头重评分，
保存连续 logits（高分更假）。每个 A01–A06 子组使用全部2,548条 real 和该组3,716条 fake。
HuBERT 的 BF16 编码缓存与离线 FP32 编码路径不同，本实验不声称两条路径分数相同。
ASV eval 标签/媒体及最终 holdout 没有用于训练、选择、阈值或评分。
已知 A01–A06 的语音开发结果不能代表未知生成器、非语音、跨数据集或比赛成绩。

## 复现与产物

团队可读任务根目录：`/var/tmp/lanjianyu-issue9-20261011`。
`data/LA` 存原媒体和协议，`data/main` 存完整与训练清单、来源、解码摘要；
`hubert/models/hubert-base-ls960` 存本地 bundle；`runs/<模型>-seed<种子>` 存
checkpoint、连续 dev 分数、完整训练历史及逐样本准备失败记录；`logs` 存运行及检查日志。
大产物与完整分数不进入 Git。

已有官方协议和共享 Parquet 时，在新任务目录物化：

```bash
task=/var/tmp/your-new-audio-task
scripts/devctl run python tools/prepare_asv_parquet.py \
  --parquet-root /srv/shared-data/datasets/asvspoof2019-la \
  --protocol-root /var/tmp/lanjianyu-issue9-20261011/protocols \
  --output-root "$task/data/LA"
scripts/devctl run python tools/prepare_asvspoof2019.py \
  --root "$task/data/LA" --output-dir "$task/data/main"
```

用本轮已登记的来源文件替换导入器默认的官方渠道登记，然后解码与排除指定样本：

```bash
cp /var/tmp/lanjianyu-issue9-20261011/data/main/sources.jsonl "$task/data/main/sources.jsonl"
scripts/devctl run python tools/prepare_asv_main.py \
  --manifest "$task/data/main/manifest.jsonl" --root "$task/data/LA" \
  --sources "$task/data/main/sources.jsonl" --output-dir "$task/data/main" \
  --workers 8 --exclude-train LA_T_1023667
```

当前 checkout 的四个忽略路径分别链接到任务资产；复现时将它们指向自己的新任务目录：

| 仓库路径 | 本轮目标 |
| --- | --- |
| `data/issue9-asv` | 任务根目录的 `data` |
| `cache/asv-cloud-20261011` | 任务根目录的 `cache` |
| `outputs/experiments/asv-cloud-20261011` | 任务根目录的 `runs` |
| `models/pretrained/hubert-base-ls960` | 本轮 HuBERT bundle |

运行输出必须为空；当前训练不支持 optimizer/RNG 续跑。先按 Issue 协调单 GPU 时段，再依次运行：

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
for run in logmel-seed42 logmel-seed43 hubert-seed42 hubert-seed43; do
  scripts/devctl run python -m deepfake_detection_archive train \
    --config "docs/experiments/asv-cloud-20261011/$run.json"
done
scripts/devctl run python docs/experiments/asv-cloud-20261011/summarize.py \
  --task-root "$task" --output-dir docs/experiments/asv-cloud-20261011
```

## 验证

薄导入14项行为测试和独立评审通过；实验汇总脚本的Ruff、严格mypy与合成行为检查通过，
四组真实分数的完整ID覆盖、来源/标签、EER复算和checkpoint哈希核对通过。
合入`main@4bbd292`后的标准入口 `scripts/devctl check`：**467项pytest通过，无跳过**；
另3项视频工具测试、Ruff、70文件格式检查、严格mypy（33+12文件）与sdist/wheel构建通过。
本地完整日志：`/var/tmp/lanjianyu-issue9-20261011/logs/check-project-integrated.log`。
