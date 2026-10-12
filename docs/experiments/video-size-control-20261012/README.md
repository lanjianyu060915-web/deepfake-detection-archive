# 视频尺寸控制与 H.264 增强

在 #20 的 ViF-Bench 诊断视图上先做**尺寸受控**匹配，再在同一组 sample_id 上比较冻结头、
现有 layer4+头微调（#21）与本轮新增的一轮训练期 H.264 增强候选，判断压缩增强是否值得保留。

## 结论

- **尺寸混杂被显著削弱但仍未消除。** 按“与注册真实片段同宽高”严格匹配后，保留 221 个假
  与 89 个真（去重 310 条），真假同宽高、同帧率、同编码。宽度 oracle 由整体 46.93% 降到
  **配对加权 50.000%**（几何已对齐），但**时长仍被完全混杂**：假全部 5.0625s，真为
  5.0s×64 / 5.0625s×24 / 4.0s×1，时长 oracle **21.24%**。统一 resize 不能证明尺寸影响消失。
- **尺寸受控后，微调的跨生成器优势基本消失。** 配对 ΔEER：受控原始视图 finetune vs freeze
  **−0.691 点**（30.769% vs 31.461%），CRF18/CRF35 **均为 0.000**；而同域 dev146 clean 仍有
  **−4.110 点**优势。跨生成器收益主要由尺寸捷径与同域分布共同驱动，不能外推。
- **H.264 增强候选只在同域压缩上有效，跨生成器反而更差。** 同域 dev146 clean
  16.438%→**15.068%**（Δ−1.370 点），CRF35 24.658%→**17.808%**（Δ−6.849 点）；但尺寸受控
  跨生成器原始 30.769%→**32.579%**（Δ+1.810 点）、CRF18 +1.119 点、CRF35 +0.447 点。
  **结论：保留为同域压缩鲁棒性候选（需再验），停止作为跨生成器泛化候选。**

## 数据与受控视图

输入为 #20 的 ViF-Bench 清单与媒体（`vifbench/manifest.jsonl`，1,949 条，假 1,845 / 真 104），
复用其排除关系与生成器标注。选取规则见 `tools/prepare_video_size_control.py`：
用 ffprobe 取真实解码宽高，**假片段仅在注册真实父片段同宽高时保留**，再按
`(generator, width, height)` 桶以 `sha256(seed:sample_id)` 排名取 `min(假, 真)`。

| 计数 | 值 |
| --- | ---: |
| 输入 | 1,949（假 1,845 / 真 104） |
| 探测成功 / 失败 | 1,949 / 0 |
| 未匹配假片段 | 1,624（全部为 `paired_real_size_differs`） |
| 匹配对数 / 去重样本 | 221 / 310 |
| 去重假 / 真 | 221 / 89 |
| 被复用真片段 | 最多 1 个真配 4 个假；89 个真中 15 个未被使用 |

匹配后真假同宽高（真：456×256 73、460×256 2、464×256 10、468×256 3、480×256 1；假：456×256 205、
其余相同），fps 全 16，编码全 h264。

**有真假成对支持的生成器（6 个，共 221 对）**：pixverse-v4-5 62、hailuo 55、kling-v1 50、
gen4-turbo 38、Wan2.1-VACE-1.3B-T 11、Wan2.2-I2V-14B 5。

**没有任何匹配对的生成器（13 个，仅计入输入数量，不补 EER）**：CogVideoX1.5-5B-T 104、
HunyuanVideo-I2V 104、HunyuanVideo 103、SkyReels-V2 104、Wan2.1-T2V-1.3B 104、Wan2.2-T2V-14B 104、
Wan2.2-TI2V-5B-I 104、Wan2.2-TI2V-5B-T 104、LTX-Video-13B-I 102、SkyReels-V2-I2V-14B-540P 103、
LTX-Video-13B-T 100、sora-2 94、pika-v2 90。这些生成器在本视图下无真样本支撑，样本量不足以给出
有效 EER，故只报数量。

### 规范化重编码视图

在 310 条匹配样本上统一几何与编码：`scale=448:256:flags=bicubic`、`yuv420p`、
`libx264 -preset medium`、`-an`、`fps_mode=passthrough`，CRF18 与 CRF35 各一次。
两条件均 310/310 成功、0 失败，输出统一 448×256 h264。

## 结果

所有对比在同一组 sample_id 上配对进行（`tools/compare_scores.py`），覆盖率 1.0、失败 0。

### 同域 dev146（Hunyuan train618/dev146，146 条）

| 候选 | clean EER | CRF35 EER |
| --- | ---: | ---: |
| 冻结头 seed42 | 20.548% | 24.658% |
| 微调 seed42（#21） | 16.438% | 24.658% |
| **H.264 增强 seed42（本轮）** | **15.068%** | **17.808%** |

### 尺寸受控跨生成器（310 条：假 221 / 真 89）

| 候选 | 原始 | CRF18 | CRF35 |
| --- | ---: | ---: | ---: |
| 冻结头 seed42 | 31.461% | 31.461% | 32.584% |
| 微调 seed42（#21） | 30.769% | 31.461% | 32.584% |
| H.264 增强 seed42（本轮） | 32.579% | 32.579% | 33.032% |

### 配对 ΔEER（候选 − 基线，负值更优）

| 视图 | finetune vs freeze | aug vs finetune | aug vs freeze |
| --- | ---: | ---: | ---: |
| dev146 clean | −4.110 点 | −1.370 点 | −5.479 点 |
| dev146 CRF35 | 0.000 点 | −6.849 点 | −6.849 点 |
| 受控原始 | −0.691 点 | +1.810 点 | +1.119 点 |
| 受控 CRF18 | 0.000 点 | +1.119 点 | +1.119 点 |
| 受控 CRF35 | 0.000 点 | +0.447 点 | +0.447 点 |

### 生成器分组（受控原始视图，样本≥30 对）

| 生成器 | 对数 | 冻结头 | 微调 | H.264 增强 |
| --- | ---: | ---: | ---: | ---: |
| pixverse-v4-5 | 62 | 30.645% | 20.968% | 27.419% |
| hailuo | 55 | 27.273% | 25.455% | 27.273% |
| kling-v1 | 50 | 32.000% | 32.000% | 32.000% |
| gen4-turbo | 38 | 44.737% | 47.368% | 42.105% |

Wan2.1-VACE-1.3B-T（11 对）与 Wan2.2-I2V-14B（5 对）样本不足，1/n 粒度分别约 9 点与 20 点，
EER 不可用于结论，只报对数。CRF18/CRF35 明细见 `outputs/.../per-generator/matched.json`。

## 复现

```bash
# 1) 尺寸受控匹配 + 规范化重编码
scripts/devctl run python tools/prepare_video_size_control.py \
  --manifest /var/tmp/lanjianyu-issue20-20261011/outputs/experiments/\
video-generalization-20261011/vifbench/manifest.jsonl \
  --data-root /var/tmp/lanjianyu-issue20-20261011/outputs/experiments/\
video-generalization-20261011/vifbench/media \
  --output-dir outputs/experiments/video-size-control-20261012/size-control

# 2) 训练期 H.264 增强派生媒体（clean/h264-crf23/h264-crf35 三臂，真假同臂）
scripts/devctl run python tools/prepare_video_augmentation.py \
  --manifest /var/tmp/lanjianyu-issue10-20261011/control/hunyuan_geometry_duration.jsonl \
  --data-root /var/tmp/lanjianyu-issue10-20261011/vif-panda-research/media \
  --output-dir /var/tmp/lanjianyu-issue30-20261012/aug-h264

# 3) 训练（按 dev146 EER 选择 epoch，不碰 ViF-Bench）
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/video-size-control-20261012/aug-h264-seed42.json

# 4) 评分与配对比较（各视图同一组 sample_id）
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/video-size-control-20261012/scores-<view>-<候选>.json
scripts/devctl run python tools/compare_scores.py \
  --baseline-scores outputs/.../scores-<view>-finetune/scores.jsonl \
  --candidate-scores outputs/.../scores-<view>-aug/scores.jsonl --output <cmp>.json
```

## 资源与产物

- H.264 增强训练：batch 8 媒体 / bf16 AMP / grad-scaler 关闭 / `trainable_prefix=layer4`；
  峰值显存 **allocated 385.5 MiB、reserved 588.0 MiB**；11 epoch 早停（best epoch 6，
  dev EER 15.068%），总 296s，稳态约 4.4–5.3s/epoch（前两轮建缓存较慢）。
- 受控重编码：CRF18 294s、CRF35 134s（各 310 条）。
- 分数/评测 JSON、本卡与配置在 `outputs/experiments/video-size-control-20261012/`；
  大媒体与帧缓存在 `/var/tmp/lanjianyu-issue30-20261012/`（团队可读，未提交）。

## 边界

- 受控视图仅 221 对、6 个生成器；**时长未受控**，且真片段在生成器间被复用，桶非独立。
- 诊断数据只用于评测，未进入任何训练；epoch 只按同域 dev146 EER 选择，未按 ViF-Bench 调参。
- 13 个生成器无匹配对，跨生成器结论不能覆盖它们；本轮不下“时序/多生成器训练”的结论。
