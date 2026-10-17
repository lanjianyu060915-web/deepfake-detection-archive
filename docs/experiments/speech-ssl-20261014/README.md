# SSL 语音检测器对照（XLS-R + AASIST 官方 LA 权重）

2026-10-14～17，RTX4090，Issue #53。
检验专门训练的 SSL 语音鉴伪器能否改善跨来源语音表现。选用
[TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)（commit
`4acaa61d`，MIT）的官方检测权重 `LA_model.pth`，与已交付的 LogMel 带宽 seed42（#39）和三域
AST seed42（#41）在同一语音协议下比较。只评公开权重，未做任何微调。

## 权重与实现

- 权重：作者 Google Drive 文件夹中的 `LA_model.pth`，1,271,633,441 字节，SHA-256
  `bd6f3609…91e10c`。从 HuggingFace 镜像 `sempertemper/ssl-antispoofing-weights` 按 tar 成员偏移
  取回，与 Drive 原件做 3 个随机偏移的逐字节比对一致；已含微调后的完整 XLS-R 300M 前端
  （429 个前端键），不需要 3.55GB 的 fairseq 基座。选 LA 版本是因为它键名无 DataParallel 前缀、
  与作者 `main_SSL_LA.py` 加载方式一致，且作者公布了它在 2021 LA 上的成绩。
- 实现：共享加载器 [`models/ssl_antispoofing.py`](../../../src/deepfake_detection_archive/models/ssl_antispoofing.py)，
  标准 `OfflineScorer` 与研究批量工具复用同一实现。不安装
  fairseq：AASIST 后端执行作者固定版本的 `model.py`（校验 SHA-256 `08b2b99b…`），只替换其中
  的 fairseq 前端类；XLS-R 前端用 transformers `Wav2Vec2Model`，把权重的 fairseq 键一对一改名
  后严格加载（667 键，无缺失、无多余，仅丢弃 7 个不参与推理的预训练头参数）。单元测试覆盖
  键映射与 transformers 参数集合完全一致、截断/循环补齐与作者实现一致、源码哈希校验。
- 预处理照作者评测路径：16 kHz 单声道，取前 64,600 点，短于此则整段循环拼接，不归一化，
  fp32 推理。作者输出取真实类 logit（越高越真），本项目记录其相反数，统一为越高越假。
- 一致性：同一台机器上，仓库工具与开发期脚本对 200 条 dev 的 logit 逐位相同；跨两台
  RTX4090 实例（同驱动、同 torch/transformers 版本）最大差 0.0078，属于硬件数值差。

## 同协议对照（EER %，覆盖率均 1.0，0 失败）

| 评测集 | 条数 | LogMel 带宽 s42 | AST 三域 s42 | SSL LA |
| --- | ---: | ---: | ---: | ---: |
| ASV dev clean | 24,844 | 0.90 | 0.51 | **0.08** |
| ASV dev AAC 64k | 24,844 | 0.94 | 0.71 | **0.08** |
| ASV dev 8 kHz 带限 | 24,844 | 1.39 | 1.73 | **0.04** |
| A07–A19 未知攻击子集 | 12,000 | 2.70 | 4.44 | **0.19** |
| In-the-Wild（已见诊断） | 31,779 | 30.67 | 24.86 | **11.21** |

以 manifest 说话人组做配对 bootstrap（1000 次，seed 42），SSL 相对另两者的 10 组 EER 差
95% 区间全部不含 0。dev 三集只有 20 个说话人组，区间较宽，幅度需谨慎解读。

- ASV dev 与 A07–A19 来自 ASVspoof2019 LA：作者权重在同一 LA train 上训练，dev 说话人与 train
  不重叠，但攻击类型 A01–A06 相同，属同分布上界；A07–A19 是训练未见攻击，作者用 2021 LA eval
  选模，对 2019 eval 子集是未调参的留出集，但不是完全盲测。
- 逐攻击：SSL 最差为 A19 0.63%、A18 0.50%、A17 0.38%；LogMel 与 AST 在 A17 上分别为 8.25%、
  18.06%。
- In-the-Wild 已被多轮使用，只作诊断。SSL 最差说话人 kamala-harris 88.0%（该组样本很少），
  其余最差几位是 adam-driver 48.8%、frank-sinatra 39.5%、fdr 36.7%。

## 成本

| 模型 | 权重 | 峰值显存（torch 实测） | 每条耗时（clean，含解码） |
| --- | ---: | ---: | ---: |
| LogMel | 0.14 MB | 41 MiB | 3.3 ms |
| AST 三域 | 327 MB | 382 MiB | 3.8 ms |
| SSL LA | 1,213 MB | 1,998 MiB | 119.94 ms |

SSL AAC64k 含解码平均耗时为 **621.24 ms/条**；以上均来自保留的逐样本 elapsed_seconds，
不是纯 GPU 前向计时，未重新运行完整集合。

## 非语音适用范围

SSL 是语音专用检测器。非语音（环境音、音乐）上的诊断结果见下节；Issue #53 规定这只是适用范围
诊断，不把 SSL 推广为通用音频模型，也不与 AST 分数直接相加。

同一 SSL 权重在非语音 dev/诊断集上的 EER（%），AST 三域 seed42 为现成冻结分数：

| 集合 | 条数 | AST 三域 s42 | SSL LA |
| --- | ---: | ---: | ---: |
| urban-dev（环境音） | 2,885 | 1.60 | 45.28 |
| music-aime96（音乐） | 96 | 0.00 | 45.00 |
| aime-unseen（未训练音乐路径） | 154 | 25.00 | 56.25 |
| openbmat-music（广播纯音乐） | 94 | 38.30 | 59.57 |
| openbmat-mixed（广播混合） | 152 | 51.32 | 51.32 |
| fsd-dev（环境音） | 70 | 5.71 | 未评 |

SSL 在环境音和音乐上接近随机，确认它只适用于语音。fsd-dev 的原始媒体在换实例后不可用，
未用替代音频评分。urban-dev 与 music-aime96 使用与原冻结清单样本 ID、标签完全一致的归档清单。

## 结论

- **保留为语音专家候选**：在语音同协议的全部条件上明显优于 LogMel 与 AST，未知攻击和真实网络
  语音上的收益最大。
- **不替换通用音频默认**：只覆盖语音；比赛音频内容范围未知，三域 AST 仍是通用候选。若将来融合，
  需另行证明分数尺度和非语音退化可控，不按文件路径或真实内容标签路由。
- **资源与许可**：权重 1.2GB、显存约 2GB，离线运行无需下载；代码 MIT，权重在 ASVspoof2019 LA
  上训练，按研究用途登记，正式赛用未确认。
- 下一步：在独立来源的中文语音上评估（#51 人工收集数据），再决定是否值得在三域数据上微调。

运行产物在 `/var/tmp/lanjianyu-issue53-20261014/`（`weights/`、`upstream/`、`out/`、`analysis/`），
持久副本见 [数据说明](../../data.md)。完整数字见 [summary.json](summary.json)。

## 标准离线推理

共享 cu126 环境可直接使用完整 LA 权重，不安装 fairseq 或重建旧依赖环境。
复制[配置模板](../../../configs/infer-audio-ssl-antispoofing.example.json)，替换媒体清单、
媒体根目录、输出目录和三个本地资源路径；相对路径基于配置所在目录：

```bash
cp configs/infer-audio-ssl-antispoofing.example.json configs/infer-ssl.local.json
# 编辑路径，保留对应权重、XLS-R配置哈希和16000Hz/64600点单窗口设置
scripts/devctl run python -m deepfake_detection_archive infer \
  --config configs/infer-ssl.local.json
```

标准 runner 输出连续 `-bonafide logit` 到 `scores.jsonl`，无效输入逐样本写 error，
不从覆盖率分母删除。预处理固定头截断/整段循环补齐，FP32、无音量归一化；
不采用通用音频模型的等距多窗口。不自动替换 AST、按内容路由或融合。
离线 bundle 的显式文件列表须一并提供配置、`LA_model.pth`、固定 `model.py`、
`xlsr_cfg.json`；配置使用包内相对路径。统一预测的资源解析支持这两个 SSL 附件，
搬移后仍从包内加载，不下载资源。现有 release 默认保持 AST。
若建立独立研究 bundle 的 `models.json`，对应模型项使用
`"architecture": "audio_ssl_antispoofing_v1"`、`"checkpoint_schema": 1`，
`config` 指向上述推理配置；加载器返回同一 architecture/schema，由统一预测入口比较。

2026-10-17 在当前 cu126/RTX4090 上，以8条原始 ASV dev 音频（真/假各4条）运行标准 CLI，
8/8成功，与保留的200条工具分数按样本ID对齐，最大绝对差 **2.38×10⁻⁶**（容差10⁻⁴）。
该小组只验证实现接入，不作为新泛化或10k吞吐结论；原五集合分数未重跑。
