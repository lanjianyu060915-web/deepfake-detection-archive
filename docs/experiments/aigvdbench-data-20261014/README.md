# AIGVDBench 独立来源视频数据（首批）

2026-10-14，Issue #51。为视频候选提供现有 Panda/HD-VILA 体系之外的开发集和隔离的最终保留集。
候选对比结果见 [AIGVDBench 新来源视频开发集](../video-aigvdbench-20261014/README.md)，本卡只记录
数据本身、隔离规则、格式捷径诊断和工具。

## 来源与独立性

- HuggingFace `AIGVDBench/AIGVDBench`，revision `bf48acaa4920990af3dd2a511ea88e63354df305`，数据卡
  标 CC-BY-4.0；底层 YouTube 视频权利未核实，登记为 unresolved，只作研究与评估。
- 真实视频为 YouTube 1080p 片段。官方 val/test 共 5,795 个 YouTube ID，与本地 ViF Panda 包的
  2,334 个 ID 重叠为 0。
- 生成器只取既不在当时训练集、也不在 ViF-Bench 中的 17 个：Sora、Gen2、Gen3、Jimeng、Luma、
  vidu、causvid_24fps、Opensora、AccVideo、AnimateDiff、EasyAnimate、IPOC、Pyramid-Flow、
  RepVideo、VideoCrafter、SEINE、SVD；另取与真实视频按 `Video_id` 一一配对的 Open-Sora。
  排除 kling、pika、Wan、CogVideoX、Hunyuan、LTX。Sora 与 ViF-Bench 的 sora-2、Gen2/3 与
  gen4 同属一个家族。
- 注意：此后 #77/#80 已在官方 train 部分加入 SEINE、Open-Sora、AccVideo、EasyAnimate、SVD
  的训练数据，这些生成器对之后的候选不再是未见生成器。

## 选取与划分

所有确定性抽样按“文件名 UTF-8 字节的 SHA-256 十六进制升序”取前 N 个。

| 用途 | 真实 | Open-Sora 配对 | 17 个独立生成器 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| dev（真实取自官方 val） | 600 | 600 | 各 40，共 680 | 1,880 |
| final holdout（真实取自官方 test） | 600 | 600 | 各 40，共 680 | 1,880 |

- 每个 YouTube ID 只取一条；同时出现在官方 val 与 test 的 110 个 ID 两边都排除。官方 train 未用。
- 配对的 Open-Sora 与其真实视频同组；其余生成视频与真实视频无已知关联，单独成组；字节相同的
  文件强制并组，跨划分即排除。结果 2,560 个组，跨划分泄漏 0，字节重复 0。
- 3,760 条按 zip 中央目录逐条 Range 下载，CRC32 全部通过，0 失败，共 8.97 GB。
- 清单 `manifest.jsonl` 3,760 行，SHA-256 `9843c3da…a4b1f9b`；摘要见 [manifest-summary.json](manifest-summary.json)。
  holdout 只能在候选冻结后用 `infer --frozen-audit` 评一次。

## 格式捷径

真实视频全部 1920×1080、时长中位约 5.7 秒；生成视频多为小分辨率（Open-Sora 600 条均为
336×192）、时长中位 1.57 秒（最短 1.375 秒），帧率含 8/10 fps。只用宽、高、面积、时长、帧率五个
元数据特征做逻辑回归（dev 内 5 折），自然媒体 EER **0.55%**（AUC 99.60%）：仅凭格式几乎完全可分。

因此新增受控条件 `video_h264_crf23_448x256_16fps_1p25s`：真假统一 448×256、SAR1、16 fps、
1.25 秒（20 帧）、H.264 CRF23、无音轨。原 2 秒条件放不下 1.375 秒的生成视频。dev 1,880 条
转换 0 失败；受控后同一元数据回归退化为 EER 50.00%。比较候选以受控组为主，自然组只说明格式捷径。

## 三个旧候选的分组 bootstrap

以清单 `group_id` 为独立组（共 1,280 组，其中含真实样本 600 组、含生成样本 1,280 组，
配对的 600 组两类都有；1000 次，seed 42），
差值为候选 EER − 七生成器 ResNet43 EER，正数表示更差。

| 条件 | v0.1.1 视频 | 时序头 43 |
| --- | --- | --- |
| 受控 | +7.25 [+5.00, +9.67] | +5.30 [+3.12, +8.17] |
| 自然 | +11.64 [+9.25, +13.98] | +9.38 [+7.17, +11.85] |

v0.1.1 与时序头之间：受控 −1.95 [−3.67, +0.33]（含 0），自然 −2.27 [−4.14, −0.23]。
完整数值、逐生成器与错误样本见 [summary.json](summary.json)，6 份评分配置在本目录。

## 工具

- [`tools/prepare_aigvdbench.py`](../../../tools/prepare_aigvdbench.py)：读取逐文件下载记录，按上述规则
  生成 dev/holdout 清单、来源记录和隔离报告；失败下载以 missing 保留在分母中。
- `tools/prepare_robustness.py` 新增 `video_h264_crf23_448x256_16fps_1p25s`，与 2 秒条件共用编码参数，
  只改时长，并逐条校验 1.25 秒 / 20 帧。

运行产物在 `/var/tmp/lanjianyu-issue51-20261014/aigvdbench/`：`media/`、`manifests/`、
`controlled-1p25s/`、`scores/`、`analysis/`。
