# 冻结视频候选的 H.264 退化与跨生成器诊断

2026-10-11，RTX4090，Issue #20。固定已交付的 ResNet18 帧头，先在受控 dev146 上做
H.264 端到端配对评分，再用固定版本的 ViF-Bench 包做外部新生成器诊断。

## 候选与评分路径

候选为 `/var/tmp/lanjianyu-issue10-20261011/seed42/training/best.pt`，SHA-256
`279167fafeeb55ae7c9c8f903834a5489dc92e31918875b08257a7fed62da86d`；
主干为同目录 `resnet18-a1/encoder.pt`，SHA-256
`db6fd0c5cfb2b712d1904bb8b7edcf41a850491bdf42729e4ab485e89f1b6889`。
采样与预处理沿用已交付配置：4 帧、224×224 `center_crop`、单段、帧 logit 均值、
FP32 解码、`batch_size` 8。头结构为 `media_feature_head_v1`（输入 512、隐藏 64、
dropout 0.1、`trim_fraction` 0.1）。

dev146 由已交付受控清单 `control/hunyuan_geometry_duration.jsonl` 的 dev 行导出
（73 真 / 73 假，SHA-256
`4f644f7fbc68187e14a31dd14d7ecd939197c4faefe39adf4104317aaaf19772`），
未读取旧 154 条 holdout。交付时的 dev 分数来自已缓存特征，本页改为从原媒体重新解码、
转码、抽帧、前向和评分，因此是独立的端到端重算。

## H.264 退化结果

配对条件由 `tools/prepare_robustness.py` 的 `video_h264_crf23`、`video_h264_crf35`
生成，两类处理参数一致，只有视频轨参与（`-an`），时间基准保持源帧率。

| 条件 | 派生媒体 | 转码失败 | 评分成功/失败 | dev EER | ΔEER |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原媒体 | 146 | — | 146 / 0 | 20.547945% | — |
| H.264 CRF23 | 146 | 0 | 146 / 0 | 20.547945% | +0.000000 个百分点 |
| H.264 CRF35 | 146 | 0 | 146 / 0 | 24.657534% | +4.109589 个百分点 |

clean 结果与交付记录的训练期 dev EER 完全一致，说明这条端到端路径复现了原选型分数。
CRF23 没有改变 EER，说明中等质量的常见转码不破坏该候选的判别；CRF35 损失 4.11 个
百分点，退化真实存在但幅度有限。

三个条件的覆盖率均为 100%，无失败样本，`tools/compare_scores.py` 报告
`common_coverage` 为 1.0，全量集合与共同成功集合相同，没有用删除失败样本改变比较集合。
转码耗时 70.14 秒（CRF23）与 51.18 秒（CRF35），单线程 FFmpeg
`N-127271-gba987fe24d-20261010`；两次评分各约 43 秒和 38 秒，含媒体解码。

条件媒体清单 SHA-256 分别为
`25bd2022f58d782deb19ada59d9a0d2422c47816043098a848b3917611f64737`（CRF23）与
`180c68f0ac15758eee1b9dc2ab25a8ed99447527f6f39f9ef3e180f7b74137d3`（CRF35）。

## 复现

仓库根目录执行。共享候选与 ViF Panda 媒体保持只读，`data/vif-panda-research`、
`models/pretrained/resnet18-a1` 和 `outputs/experiments/vif-cloud-20261011/seed42`
是本任务目录内指向已交付资产的符号链接，可换成配置副本里的实际路径。

```bash
export PYTHONPATH=src

# dev146：只取已交付受控清单的 dev 行
python - <<'PY'
import json, pathlib
src = pathlib.Path("control/hunyuan_geometry_duration.jsonl")
rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
dev = sorted(
    (row for row in rows if row["split"] == "dev"), key=lambda row: row["sample_id"]
)
out = pathlib.Path("outputs/experiments/video-generalization-20261011/inputs/dev146.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in dev)
)
PY

python tools/prepare_robustness.py \
  --manifest outputs/experiments/video-generalization-20261011/inputs/dev146.jsonl \
  --data-root data/vif-panda-research/media \
  --output-dir outputs/experiments/video-generalization-20261011/robustness-h264 \
  --condition video_h264_crf23 --condition video_h264_crf35

for condition in clean h264-crf23 h264-crf35; do
  python -m deepfake_detection_archive infer \
    --config "docs/experiments/video-generalization-20261011/scores-$condition.json"
done

python tools/compare_scores.py \
  --baseline-scores outputs/experiments/video-generalization-20261011/scores-clean/scores.jsonl \
  --candidate-scores outputs/experiments/video-generalization-20261011/scores-h264-crf35/scores.jsonl \
  --output outputs/experiments/video-generalization-20261011/compare-h264-crf35.json
```

逐样本分数、评分报告、条件媒体与对比报告在
`outputs/experiments/video-generalization-20261011/`，不进入 Git。

## 边界

dev146 在交付时用于早停，不是最终 holdout；结果只覆盖单一 Panda/HunyuanVideo 受控子集，
不能代表其他来源。H.264 是常见平台转码的近似，不是平台自身的编码器配置；CRF35 的
退化说明需要压缩增强，但本轮不重训候选。该候选在未经训练的生成器上的表现、来源隔离
做法与压缩对照见下一节。

## 跨生成器外部诊断（ViF-Bench 固定版本）

固定来源为 `JoeLeelyf/ViF-Bench` 的 revision
`67e11331e26dbf732e0b8c7d4dc52f4442d174dd`，`source_videos.zip` 607,734,233 字节，
整包 SHA-256 `41e79dff9f8ff16f7bcddd99eb18e4019b52f5ed9fe26a6a0fa0739b085907a8`
与上游 LFS 记录一致。包内为 165 条真实媒体与 19 个生成器共 2,995 条生成媒体；
`source_videos/ladm_paths.json` 声明 3,160 条媒体，与包内成员一一对应，但它只带占位
前缀、没有底层来源字段，因此这个包不能整体批准为 Panda 或独立新来源，只作为外部评估
候选。

配对规则 `vifbench_identical_filename_pair_v1`：仅当真实与生成媒体文件名逐字节相同
才视为作者配对，2,979 条生成媒体（99.5%）存在同名真实媒体。文件名前 11 字符沿用
`suspected_filename_component:v1` 作为保守泄漏边，不当作同一原始录制的证据。

保守排除以组件为单位，排除已出现在既有 Panda 完整清单（4,612 条，SHA-256
`3e36cfcacdeeb41f724a3ef2beacc24159396f8462aa8edf6bdb2fafbb5bc273`）和旧 pilot
（1,024 条，含旧 154 条 holdout，SHA-256
`f3b004bb94174d02899b9a8763c3619da121f90f53f919e5884e171c716d6638`）中的组件。
最终保留 104 条真实媒体与 1,845 条生成媒体（103 个组件、19 个生成器）；另有 61 条
真实媒体与 1,150 条生成媒体不进入诊断，其中 1,134 条随组件排除、16 条没有同名真实
媒体。两类都走同一条 4 帧、224×224 `center_crop`、FP32 解码路径，没有任何按类别区别
的处理；因此这是跨生成器诊断，不是跨来源泛化。

### 整体与逐生成器结果

诊断清单 SHA-256
`38918ed862074b8dca2d28bc0760fe8081ce58c90568a2f410a186aa5793221d`，
1,949 条样本全部评分成功，失败 0，覆盖率 100%。

| 分组 | 样本 | real / fake | EER |
| --- | ---: | ---: | ---: |
| 全部生成器 | 1,949 | 104 / 1,845 | **33.279%** |
| 19 个生成器（中位数） | — | — | 30.000% |
| 同域受控 dev146 对照 | 146 | 73 / 73 | 20.548% |

| 生成器 | real / fake | EER | 生成器 | real / fake | EER |
| --- | ---: | ---: | --- | ---: | ---: |
| SkyReels-V2 | 104 / 104 | 21.154% | kling-v1 | 80 / 80 | 30.000% |
| Wan2.1-T2V-1.3B | 104 / 104 | 23.077% | CogVideoX1.5-5B-T | 104 / 104 | 31.731% |
| HunyuanVideo | 103 / 103 | 23.301% | sora-2 | 94 / 94 | 32.979% |
| pika-v2 | 90 / 90 | 24.444% | HunyuanVideo-I2V | 104 / 104 | 39.423% |
| LTX-Video-13B-T | 100 / 100 | 27.000% | Wan2.1-VACE-1.3B-T | 104 / 104 | 39.423% |
| hailuo | 81 / 81 | 27.160% | gen4-turbo | 64 / 64 | 43.750% |
| pixverse-v4-5 | 92 / 92 | 27.174% | LTX-Video-13B-I | 102 / 102 | 44.118% |
| Wan2.2-TI2V-5B-T | 104 / 104 | 28.846% | Wan2.2-I2V-14B | 104 / 104 | 44.231% |
| Wan2.2-T2V-14B | 104 / 104 | 29.808% | SkyReels-V2-I2V-14B-540P | 103 / 103 | 47.573% |
|  |  |  | Wan2.2-TI2V-5B-I | 104 / 104 | 48.077% |

每个生成器桶的真假数量相等，EER 由项目 `evaluate` 在逐样本分数上计算，失败分母为 0。
各行共享同一批真实媒体，行与行之间不是独立实验，也没有计算独立组置信区间。

### 压缩对照（同一诊断集）

同一份清单、同一 checkpoint、同样的采样与聚合，只把媒体换成 H.264 CRF35 派生媒体。
1,949 条转码全部成功（0 失败，855.7 秒），派生清单 SHA-256
`12e9365d629fd6c02860e1209dde9399cd126bcbe8e6f8e885ce68dfe560d311`；
两个条件各有 1,949 条成功分数，共同集合与全量集合相同（`common_coverage` 1.0），
没有通过删除样本改变比较集合。

| 分组 | clean EER | CRF35 EER | ΔEER |
| --- | ---: | ---: | ---: |
| 全部诊断集 | 33.279% | 36.538% | +3.259 个百分点 |
| 19 个生成器（中位数） | 30.000% | — | +1.923 个百分点 |
| 同域受控 dev146 对照 | 20.548% | 24.658% | +4.110 个百分点 |

| 生成器 | clean | CRF35 | Δ | 生成器 | clean | CRF35 | Δ |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| HunyuanVideo-I2V | 39.423% | 33.654% | −5.769 | Wan2.2-TI2V-5B-T | 28.846% | 29.808% | +0.962 |
| LTX-Video-13B-T | 27.000% | 24.000% | −3.000 | HunyuanVideo | 23.301% | 24.272% | +0.971 |
| SkyReels-V2-I2V-14B-540P | 47.573% | 46.602% | −0.971 | Wan2.1-T2V-1.3B | 23.077% | 25.000% | +1.923 |
| CogVideoX1.5-5B-T | 31.731% | 30.769% | −0.962 | Wan2.2-T2V-14B | 29.808% | 31.731% | +1.923 |
| SkyReels-V2 | 21.154% | 21.154% | +0.000 | Wan2.2-TI2V-5B-I | 48.077% | 50.000% | +1.923 |
| Wan2.1-VACE-1.3B-T | 39.423% | 39.423% | +0.000 | pixverse-v4-5 | 27.174% | 29.348% | +2.174 |
| kling-v1 | 30.000% | 30.000% | +0.000 | gen4-turbo | 43.750% | 46.875% | +3.125 |
|  |  |  |  | LTX-Video-13B-I | 44.118% | 48.039% | +3.922 |
|  |  |  |  | Wan2.2-I2V-14B | 44.231% | 49.038% | +4.808 |
|  |  |  |  | hailuo | 27.160% | 32.099% | +4.938 |
|  |  |  |  | sora-2 | 32.979% | 38.298% | +5.319 |
|  |  |  |  | pika-v2 | 24.444% | 32.222% | +7.778 |

压缩使整体 EER 上升 3.259 个百分点，与同域 dev146 的 +4.110 幅度相近，说明压缩敏感度
在多数生成器上保持一致，多数桶变差（中位数 +1.923），最差为 pika-v2、sora-2、
hailuo 与 Wan2.2-I2V-14B。少数桶反而改善，其中 HunyuanVideo-I2V 改善 5.769 个百分点：
该桶的宽度线索在 CRF35 后被削弱，与「分数主要受尺寸线索影响」的判断一致，不能解读为
压缩提升了内容级检测。

### 尺寸捷径与结论

元数据诊断（`metadata-probe.py`，ffprobe 只读媒体头，未解码帧）显示整体上宽度不可分：
真实宽度中位数 456（342–602，8 个不同取值），生成媒体 452（256–1056，21 个取值），
全诊断集上以宽度做最优阈值的 EER 为 82.44%，说明整体没有宽度捷径。

但个别生成器仍有明显尺寸捷径，这些桶的 EER 不能当作内容级检测证据：
HunyuanVideo-I2V 的生成宽度中位数 928 对真实 456（宽度 EER 0%），sora-2 为 4.26%，
Wan2.2-I2V-14B 为 16.35%。其中 HunyuanVideo-I2V（模型 39.423%）和
Wan2.2-I2V-14B（44.231%）都差于同一批样本上的宽度基线，不能作为该生成器可检出的
证据。HunyuanVideo 是训练用生成器，在本诊断的 103 对上仍为 23.301%，明显高于同域
dev146 的 20.548%，说明同域数字不能外推到新条件。

最清晰的信号是条件类型而不是来源：I2V 家族（HunyuanVideo-I2V、LTX-Video-13B-I、
Wan2.2-I2V-14B、SkyReels-V2-I2V-14B-540P、Wan2.2-TI2V-5B-I）EER 落在
39.4%–48.1%，T2V 家族落在 21.2%–29.8%。I2V 以真实首帧为条件，生成伪迹更弱，这与
下一轮应优先补 I2V 增强/训练数据的判断一致。

### 复现

```bash
export PYTHONPATH=src
O=outputs/experiments/video-generalization-20261011

# 固定版本归档：下载后必须与上游 LFS SHA-256 一致
sha256sum "$O/../../vif-bench/source_videos.zip"   # 41e79dff…

python tools/prepare_vif_bench.py \
  --archive outputs/vif-bench/source_videos.zip \
  --output "$O/vif-bench-audit.json" --audit-only

python tools/prepare_vif_bench.py \
  --archive outputs/vif-bench/source_videos.zip \
  --output "$O/vifbench" \
  --exclude-manifest "$O/exclusions/panda-full.jsonl" \
  --exclude-manifest "$O/exclusions/vif-pilot.jsonl"

python -m deepfake_detection_archive infer \
  --config docs/experiments/video-generalization-20261011/scores-vifbench.json

# 压缩对照：同一清单换成 CRF35 派生媒体
python tools/prepare_robustness.py \
  --manifest "$O/vifbench/manifest.jsonl" --data-root "$O/vifbench/media" \
  --output-dir "$O/vifbench-robustness" --condition video_h264_crf35
python -m deepfake_detection_archive infer \
  --config docs/experiments/video-generalization-20261011/scores-vifbench-crf35.json
python tools/compare_scores.py \
  --baseline-scores "$O/scores-vifbench/scores.jsonl" \
  --candidate-scores "$O/scores-vifbench-crf35/scores.jsonl" \
  --output "$O/compare-vifbench-crf35.json"

# 逐生成器 EER：先按诊断子集裁剪同一份分数，再用项目 evaluate 计算
python - <<'PY'
import json, pathlib
O = pathlib.Path("outputs/experiments/video-generalization-20261011")
scores = [json.loads(line) for line in
          (O / "scores-vifbench/scores.jsonl").read_text().splitlines() if line.strip()]
target = O / "per-generator"
target.mkdir(exist_ok=True)
for path in sorted((O / "vifbench/diagnostics").glob("*.jsonl")):
    ids = {json.loads(line)["sample_id"] for line in
           path.read_text().splitlines() if line.strip()}
    (target / f"{path.stem}.scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in scores if row["sample_id"] in ids))
PY
for f in "$O"/vifbench/diagnostics/*.jsonl; do
  name=$(basename "$f" .jsonl)
  python -m deepfake_detection_archive evaluate \
    --scores "$O/per-generator/$name.scores.jsonl" --manifest "$f" \
    --output "$O/per-generator/$name.json"
done

python docs/experiments/video-generalization-20261011/metadata-probe.py \
  --manifest "$O/vifbench/manifest.jsonl" --data-root "$O/vifbench/media" \
  --output "$O/vifbench/metadata-summary.json"
```

许可与边界：ViF-Bench 按 CVPR 2026 官方补充材料以 CC BY 4.0 发布，同时声明遵守底层
来源许可；本包的底层媒体边界未解析，因此这里只登记 `evaluation` 与
`local_noncommercial_computational_research` 两种用途，未取得参赛资产批准，媒体不随
Git 分发。归档下载时曾出现范围内容错位，最终以逐成员 CRC-32 定位并重取受损区间，
整包 SHA-256 与上游一致后才用于评分。

### 运行资源

RTX 4090 24 GiB 与队友任务并发使用，未设独占锁，全部运行均无 OOM、无失败样本。小批量
profile 用同一配置对 64 条诊断媒体评分（PID 520344，`batch_size` 8，4 帧、224×224、
FP32）：20.32 秒、0 失败，该进程设备显存峰值 612 MiB（设备上同时有其他任务）。1,949 条
诊断集评分约 7 分钟，CRF35 转码 855.7 秒、CRF35 评分约 7 分钟；评分报告给出的投影为
每 10,000 条约 3,314 秒，只覆盖模型启动、逐样本预检、样本调用与写入，不代表缓存复用或
端到端离线吞吐。
