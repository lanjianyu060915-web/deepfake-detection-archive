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
做法与压缩对照，由本目录后续补充的实验记录给出。
