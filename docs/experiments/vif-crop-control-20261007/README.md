# ViF 中心裁剪探索对照：不晋级

2026-10-08 核验完成。中心裁剪相对保比例补边的 dev EER 变化在两个种子间方向相反，
没有稳定改善证据。该实验限 ViF Panda-only 本地非商业计算研究，不代表比赛成绩、
跨域泛化或内容层面的伪造识别能力。

| seed | pad dev EER | center_crop dev EER | crop − pad | crop best epoch / 完成 epochs | dev 成功 / 失败 |
| --- | ---: | ---: | ---: | --- | --- |
| 42 | 32.4675% | 33.7662% | +1.2987 个百分点 | 3 / 8 | 154 / 0 |
| 43 | 32.4675% | 31.1688% | −1.2987 个百分点 | 1 / 6 | 154 / 0 |

dev 固定为 77 real + 77 fake，train 为 358 real + 358 fake。
最佳 epoch 由同一 dev 早停选择，因此这些是开发指标。不能只选 seed43 的较低结果；
两个种子也不足以估计可靠的不确定性。统计组仍为 unknown，未计算独立组 CI。

## 探索性边界与原始记录

局部设计见 [对照方案](../../plans/vif-shortcut-controls.md)。
`configs/comparison.json` 记录配置实际 materialization 时已看过基线 seed42 分数；
即使设计记录先于主代理查看检测器分数，该对照仍不是正式预注册候选。
`preregistration.json` 明确 `baseline_seed42_seen=true`、`formal_candidate=false`。
文件名中的 preregistration 不改变这一事实。

原始 `configs/comparison.json` 的 `executed=false` 是执行前快照，保留原字节；
本次实际完成依据为 `training/seed42/`、`training/seed43/` 的 execution、experiment、
history、scores 和本地权重，以及新增 [核验摘要](training/summary.json)。
不回写原始配置、预登记或训练记录。

## 独立核验证据

[核验脚本](source/verify_results.py) 从四组逐样本 dev 连续分数独立扫描阈值，
按 `score >= threshold` 判 fake，整体处理并列分数并在 FAR/FRR 交点线性插值，
不调用项目指标函数。四组复算均与原报告和最佳 checkpoint 的 dev EER 一致。
逐样本 ID 唯一、无缺样，标签、生成器、来源和模态与锁定 dev 清单一致。

- crop 与对应 pad 的原始配置逐字段比较，仅 `video.resize_mode`、
  `feature_cache`、`output_dir` 不同。帧数 4、224×224、冻结 ResNet18、均值聚合、
  64 维隐藏层、batch 64、bf16 AMP、AdamW 与早停设置一致。
- 原始清单 SHA-256 为
  `f3b004bb94174d02899b9a8763c3619da121f90f53f919e5884e171c716d6638`；
  编码器实际文件 SHA-256 为
  `db6fd0c5cfb2b712d1904bb8b7edcf41a850491bdf42729e4ab485e89f1b6889`。
  清单、train/dev 子集、来源登记、编码器配置、初始权重、配置、uv.lock、runner
  和源码绑定核验通过。详细哈希保存在摘要与原始 preflight。
- 四组运行的包源码内容哈希均为
  `19aaa60d08f1a134540ae81e57fe6aacb877afac9b1b9b9d160c5b77a06de225`，
  核验时与当前包源码相同。原始记录为 `dirty=true`，执行 commit 另保留，
  因此没有声称运行在干净 Git checkout。
- 四组本地 `initial.pt` / `training/best.pt` 先校验哈希，再安全严格加载。
  每个模型的全部 6 组、33,921 个可训练参数元素均改变；同 seed 的 crop/pad
  初始张量完全一致。crop seed42/43 完成 96/72 次 optimizer 更新、0 次跳步，
  最佳 checkpoint 分别来自第 3/1 个 epoch；摘要保留各参数的变化数量和最大差值。
- train/dev 审计通过，训练解码失败清单为空；四组训练及推理退出码均为 0，
  每组 dev 出分 154/154、失败 0。已有元数据与独立性风险不因零运行失败而解除。

本次核验未读取任何媒体或 holdout 分数，未训练或重新推理。只读完整清单的元数据及
train/dev 产物；原训练源码只选择 train/dev，推理配置只含 dev，原报告标记
`holdout_accessed=false`。这支持记录所述范围，但没有历史 OS 文件访问追踪，
不能将这些记录夸大为独立系统审计证明。

## 实测资源

以下为 crop 的原始 GNU time / 每秒 GPU 采样统计；pad 同口径值也写入摘要。

| seed | 全训练进程 / dev 推理秒数 | GPU 采样显存峰值，训练 / 推理 MiB | GPU 采样温度峰值，训练 / 推理 °C | GPU 采样利用率峰值，训练 / 推理 |
| --- | --- | --- | --- | --- |
| 42 | 118.4605 / 22.2701 | 1,228 / 1,310 | 58 / 56 | 58% / 40% |
| 43 | 5.3435 / 23.2855 | 1,086 / 1,142 | 56 / 57 | 9% / 41% |

seed42 首次生成 870 条 train/dev 的 crop 特征缓存，seed43 复用该缓存；
完整运行耗时不能直接视为纯训练速度对比。仅头训练分别 2.5534 / 1.8928 秒；
PyTorch 头训练峰值 allocated 均为 68,962,304 字节，reserved 分别
167,772,160 / 134,217,728 字节。设备采样包括桌面等其他进程，可能漏掉瞬时峰值，
尤其 seed43 训练只有 5 个采样点；不能与进程内 PyTorch 数值混为一谈。
摘要另保留进程峰值 RSS、功耗采样峰值、采样数量与全部原始证据文件哈希。

## 复核命令与后续限制

在仓库根目录运行（依赖本机忽略目录中的原 checkpoint、清单、编码器和输出）：

```bash
PYTHONPATH=src .venv/bin/python docs/experiments/vif-crop-control-20261007/source/verify_results.py > /tmp/vif-crop-review.json
cmp /tmp/vif-crop-review.json docs/experiments/vif-crop-control-20261007/training/summary.json
UV_CACHE_DIR=/tmp/aigc-uv-cache uv run --no-sync ruff check docs/experiments/vif-crop-control-20261007/source/verify_results.py
UV_CACHE_DIR=/tmp/aigc-uv-cache uv run --no-sync ruff format --check docs/experiments/vif-crop-control-20261007/source/verify_results.py
UV_CACHE_DIR=/tmp/aigc-uv-cache uv run --no-sync mypy docs/experiments/vif-crop-control-20261007/source/verify_results.py
```

核验脚本只向 stdout 输出新摘要，并拒绝 `python -O`；原始记录保持只读。
后续若包源码、runner 或冻结资产变化，会明确失败而非静默接受。仅克隆 Git 不含
本机权重和输出，缺少这些文件时不能声称完成 checkpoint 核验。

基线已有宽高比/来源关联，元数据宽度单变量 dev EER 为 24.75%，详见
[基线报告](../vif-pilot-20261007/README.md)。中心裁剪可能去掉补边信息，同时裁掉
画面内容；本对照无法分离这两种效应，也不能证明其他来源捷径消失。
结论为不晋级，后续控制须重新明确设计、冻结配置并使用开发集，最终 holdout 仍待
模型冻结后的验收。赛用许可、官方 I/O/评测对齐与正式 P1/P2 阶段门没有因此通过。
