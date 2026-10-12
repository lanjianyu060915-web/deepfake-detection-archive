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

- 每个 epoch 实测 3.5–3.8 秒，首 epoch 5.3–5.4 秒（含预热），两个种子都在第 8 个
  epoch 触发早停；`training/history.json` 的 `seconds` 没有拆分训练与 dev 的时间。
- 显存补测（同一 `best.pt` 与训练配置，batch 8 媒体=32 帧、bfloat16 AMP、AdamW）：
  一步训练峰值 0.20 GiB（`torch.cuda.max_memory_allocated`）。此前记录的“约 3.3 GiB”
  无法用该配置复现，以本次测量为准。
- 帧缓存 764 条媒体，解码后按 224×224 float32 缓存。

## 离线复算

`video_frames` 直接解码并使用完整 `best.pt`，无需读取训练时的原始 encoder。
恢复回归删除原 encoder、移动 checkpoint 并禁止网络访问；真实双种子权重均成功独立加载。

2026-10-12 将直接解码帧统一为与训练缓存相同的连续内存布局后，seed42 的 146 条 dev
端到端 EER 仍为 **16.4384%**、覆盖率 1.0、零失败。相对原训练期 dev 分数，最大绝对差
0.003553、均值 0.000636；配置、分数和评测在团队可读
`/var/tmp/deepfake-video-review-20261012/normalized-seed42-dev.json` 与
`normalized-seed42-dev/`。

此前独立推理的最大分数差为 0.0275。对最大差异样本检查，缓存与重解码帧逐元素相等，
但原始内存布局不同；cuDNN TF32 和批量形状放大了数值差异。该单样本诊断解释了对应差值，
不代表所有样本误差都已归因。训练 dev 与推理均为 FP32，不能把差异归因于训练反传使用 AMP。

## 产物位置

- 逐样本 dev 分数与评测：`outputs/video-finetune-20261011/seed{42,43}/dev-scores.jsonl`、
  `dev-evaluation.json`
- checkpoint：`outputs/video-finetune-20261011/seed{42,43}/training/best.pt`
  （含微调主干与头，`best.pt.json` 记录架构、配置与 sha256）
- 帧缓存：`cache/video-frames-20261011`、`cache/video-frames-seed43-20261011`

上述输出相对于 `/var/tmp/lanjianyu-issue21-20261011/`，已开放 `dlteam` 只读访问。
完整 checkpoint 恢复只加载 `best.pt` 内的主干和头，不读取原始 encoder 文件；
原始 encoder 只在新训练初始化时使用，其相对路径以训练配置目录为基准。

## 未完成与限制

- 独立进程只验证了本地加载与无隐式下载，没有做真正切断网络的整机演练。
- dev 只有 146 条，EER 分辨率有限；结论只适用于当前受控划分。
