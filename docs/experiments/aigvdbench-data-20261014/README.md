# AIGVDBench 独立来源视频数据（首批）

2026-10-14，Issue #51。为视频候选提供现有 Panda/HD-VILA 体系之外的开发集和保留集；2026-10-17修正已知母视频分组后才生成当前可用清单。
候选对比结果见 [AIGVDBench 新来源视频开发集](../video-aigvdbench-20261014/README.md)，本卡只记录
数据本身、隔离规则、格式捷径诊断和工具。

## 来源与独立性

- HuggingFace `AIGVDBench/AIGVDBench`，revision `bf48acaa4920990af3dd2a511ea88e63354df305`，数据卡
  标 CC-BY-4.0；底层 YouTube 视频权利未核实，登记为 unresolved，只作研究与评估。
- 真实视频为 YouTube 1080p 片段。官方 val/test 共 5,795 个 YouTube ID，与本地 ViF Panda 包的
  2,334 个文件名候选标识无交集；Panda前11字未取得原始URL映射，已确认母媒体交集仍未知。
- 生成器只取既不在当时训练集、也不在 ViF-Bench 中的 17 个：Sora、Gen2、Gen3、Jimeng、Luma、
  vidu、causvid_24fps、Opensora、AccVideo、AnimateDiff、EasyAnimate、IPOC、Pyramid-Flow、
  RepVideo、VideoCrafter、SEINE、SVD；另取与真实视频按 `Video_id` 一一配对的 Open-Sora。
  排除 kling、pika、Wan、CogVideoX、Hunyuan、LTX。Sora 与 ViF-Bench 的 sora-2、Gen2/3 与
  gen4 同属一个家族。
- 注意：此后 #77/#80 已在官方 train 部分加入 SEINE、Open-Sora、AccVideo、EasyAnimate、SVD
  的训练数据，#90又加入VideoCrafter/AnimateDiff/Pyramid-Flow；这些标签对后续候选不再未见。

## 选取与划分

所有确定性抽样按“文件名 UTF-8 字节的 SHA-256 十六进制升序”取前 N 个。

| 用途 | 真实 | Open-Sora 配对 | 17 个其他生成器 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| dev（真实取自官方 val） | 600 | 600 | 各 40，共 680 | 1,880 |
| 原holdout库存（真实取自官方 test） | 600 | 600 | 各 40，共 680 | 1,880 |

- 真实池每个 YouTube ID 只取一条，同时出现在官方 val/test 的110个ID两边排除。
  其他17个生成器按完整ZIP库存的文件hash选择，再分配用途，并非只来自官方val/test；
  这些首批媒体均只用于开发/保留，不进入梯度训练。
- Open-Sora及AccVideo、AnimateDiff、EasyAnimate、IPOC、Pyramid-Flow、RepVideo、
  VideoCrafter、SEINE、SVD按官方共享`Video_id`的YouTube母视频合组，不同片段也属于同组。
  闭源数字文件名没有确认映射，保持独立登记，不能据此断言统计独立。字节重复继续强制合组。
- 原工具仅合并显式Open-Sora配对，遗漏其余衍生关系；修正后3760条库存共有1919组，
  发现`zU4LIqsBYOk`跨dev/holdout，全部11条整组排除（dev2条、holdout9条），不补选。
  当前可用1918组：dev1878条（599real/1279fake）、holdout1871条（600real/1271fake），各959组，
  已确认分组无交集；原下载失败仍保留在评测分母，本批下载零失败、字节重复零组。
- 原下载与清单仍在原目录，只读保留；新清单在`/var/tmp/deepfake-pr92-corrected-20261017/`。
  `manifest.jsonl`为3760条完整库存（含11 excluded）；实际评测分别使用`dev-manifest.jsonl`和
  `holdout-manifest.jsonl`，完整SHA与排除清单见[manifest-summary.json](manifest-summary.json)。
  holdout本轮没有解码或评分；候选冻结后才用`infer --frozen-audit`，Panda关系仍未知。
- 原3,760条Range下载均通过CRC32，共8.97GB；本次只修正清单元数据，未重新下载或转码。

## 格式捷径

真实视频全部 1920×1080、时长中位约 5.7 秒；生成视频多为小分辨率（Open-Sora 600 条均为
336×192）、时长中位 1.57 秒（最短 1.375 秒），帧率含 8/10 fps。只用宽、高、面积、时长、帧率五个
元数据特征做逻辑回归（dev 内 5 折），自然媒体 EER **0.55%**（AUC 99.60%）：仅凭格式几乎完全可分。

因此新增受控条件 `video_h264_crf23_448x256_16fps_1p25s`：真假统一 448×256、SAR1、16 fps、
1.25 秒（20 帧）、H.264 CRF23、无音轨。原 2 秒条件放不下 1.375 秒的生成视频。dev 1,880 条
转换 0 失败；受控后同一元数据回归退化为 EER 50.00%。比较候选以受控组为主，自然组只说明格式捷径。

## 历史评分与分组修正

[summary.json](summary.json)及本目录6份配置保留原1880条dev的点估计、逐生成器与错误分析。
这些评分已经被多轮使用，是历史诊断，未按新1878条清单重新包装成盲测或新实验。
原bootstrap将同母视频拆组，1280独立组及其置信区间不成立，已撤回；不据此声称显著性。
原数字通过Git追溯，逐样本分数仍保留在原运行目录。无需为此次元数据修正重跑旧模型。

## 工具

- [`tools/prepare_aigvdbench.py`](../../../tools/prepare_aigvdbench.py)：读取逐文件下载记录，按上述规则
  生成 dev/holdout 清单、来源记录和隔离报告；失败下载以 missing 保留在分母中。
- `tools/prepare_robustness.py` 新增 `video_h264_crf23_448x256_16fps_1p25s`，与 2 秒条件共用编码参数，
  只改时长，并逐条校验 1.25 秒 / 20 帧。

运行产物在 `/var/tmp/lanjianyu-issue51-20261014/aigvdbench/`：`media/`、`manifests/`、
`controlled-1p25s/`、`scores/`、`analysis/`。
