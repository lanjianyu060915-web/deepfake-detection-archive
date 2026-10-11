# 视频 ResNet18 部分主干微调（issue #21）

## 目标

把冻结的 ResNet18 帧头推进成可训练的视觉检测器：在同一 train618/dev146、
同一采样与预处理下只解冻 `layer4` 与媒体头，取得真实 dev EER 和可离线推理的
checkpoint。第一轮不引入时序模块、新数据或复杂增强。

## 配置

- 采样与预处理与冻结头对照完全一致：4 帧、224×224 `center_crop`、帧 logit 均值。
- 解冻范围：`backbone.layer4.*` 与 `head.*`；更早层保持 `requires_grad=False`
  且始终留在 `eval()`，因此它们的 BN 统计不被更新（`tests/test_video_finetune.py`
  覆盖该行为）。
- 优化：AdamW，头 `1e-3`，主干 `1e-4`（`backbone_lr_scale=0.1`），权重衰减 `1e-4`，
  max_grad_norm 1.0，bfloat16 AMP，batch 8 个媒体（32 帧/步），早停 patience 5。
- 数据划分、清单与权重沿用 `/var/tmp/lanjianyu-issue10-20261011/`，未重新划分，
  旧 154 条 holdout 未参与。

## 结果

| 候选 | seed42 dev EER | seed43 dev EER | 最佳 epoch |
| --- | ---: | ---: | ---: |
| 冻结 ResNet18 帧头（对照） | 20.547945% | 21.917808% | — |
| layer4 + 头部分微调（本次） | 16.438356% | 16.438356% | 3 |

两个种子都比冻结头低约 4.1 与 5.5 个百分点，覆盖率 1.0、无解码失败样本。
两 seed 的 dev EER 完全相同，与 dev 只有 146 条（EER 粒度约 0.68%）以及
确定性解码、相同早停点一致；这不构成独立的显著性证据，只说明当前划分上
部分微调没有把结果变差。

## 资源

- 618 条训练样本每 epoch 约 3.2 秒（约 194 samples/s），dev 146 条约 0.6 秒，
  两个种子都在第 8 个 epoch 触发早停。
- 帧缓存 764 条媒体，解码后按 224×224 float32 缓存；训练显存占用约 3.3 GiB。

## 离线复算

合并 `main` 后用独立进程、独立输出目录重新走 `config.py` 的 `video_frames`
推理路径，对同一清单的 764 条媒体评分：0 失败、覆盖 1.0，其中 dev 146 条
EER 仍为 16.4384%，与训练期 dev 分数的最大绝对差 0.0275、均值 0.0052
（训练用 bfloat16 AMP，推理为 FP32）。输入只有本地清单、媒体和 hash 校验过的
`best.pt`，没有隐式下载；全过程约 0.22 秒/条，瓶颈在视频解码。
产物：`/var/tmp/lanjianyu-issue21-20261011/e2e-rehearsal/seed42/`
（`scores.jsonl`、`report.json`、`metadata.json`）。

## 产物位置

- 逐样本 dev 分数与评测：`outputs/video-finetune-20261011/seed{42,43}/dev-scores.jsonl`、
  `dev-evaluation.json`
- checkpoint：`outputs/video-finetune-20261011/seed{42,43}/training/best.pt`
  （含微调主干与头，`best.pt.json` 记录架构、配置与 sha256）
- 帧缓存：`cache/video-frames-20261011`、`cache/video-frames-seed43-20261011`

## 未完成与限制

- 训练会话只跑了改动相关的行为测试 `tests/test_video_finetune.py`（2 项通过）；
  合并 `main` 时补齐 `tools/check_project.sh` 的本地与 CI 检查。
- 独立进程只验证了本地加载与无隐式下载，没有做真正切断网络的整机演练。
- dev 只有 146 条，EER 分辨率有限；结论只适用于当前受控划分。
