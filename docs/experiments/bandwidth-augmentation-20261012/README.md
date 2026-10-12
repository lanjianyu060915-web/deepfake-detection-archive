# LogMel 语音候选的训练期带宽增强

2026-10-12，RTX4090，Issue #29。
在 Issue #9 的 LogMel CNN 语音候选上只增加一项训练期增强：一部分训练波形先降采样到
8 kHz 再恢复到 16 kHz，其余保持原样，真假使用同一策略。目标是压低 Issue #19 记录的
8 kHz 带限退化（clean 0.7849% → 带限 13.0965%），同时尽量不损失 clean 性能。本轮不改
模型结构、采样、损失或数据划分，也不动视频与离线打包。

## 方法与增强实现

增强记为 `augmentation.bandwidth = {probability, target_sample_rate}`，在
`src/deepfake_detection_archive/augmentation.py` 中以理想带限实现：对波形做 `rfft`，
按目标 Nyquist 截断（降采样）或补零（升采样）频谱后 `irfft`，并按长度比做幅度补偿，
保证变换线性、无混叠折返。等价于理想低通 + 重采样，不引入编解码器。

增强在**缓存读取之后**逐条施加，缓存张量保持与未增强运行逐字节一致，因此复用
`cache/asv-cloud-20261011/waveforms` 而无需重算派生数据，也没有 epoch 内调用 FFmpeg 的
开销。dev 与推理路径永远不持有增强规范：`ExperimentConfig.bandwidth_spec` 会拒绝非音频、
带编码器或 `segments != 1` 的配置，评分链路完全不受影响。随机门槛来自按 worker 播种的
本地流，与数据加载顺序解耦。

## 数据与来源

| 用途 | 内容 | 数量 | 清单 SHA-256 |
| --- | --- | --- | --- |
| 训练 | 沿用 #9 的 ASV train（A01–A06） | 25,379 | `8b204f61…148257` |
| dev clean | 同攻击 dev | 24,844（真2,548/假22,296） | `31476ba0…3bdbf8` |
| dev AAC 64 kbps | `prepare_robustness.py` 派生 | 24,844 | `a6fb9b53…3756d3f` |
| dev 8 kHz 带限 | 8 kHz 带限后恢复 16 kHz | 24,844 | `b26a9cb0…e095416` |
| 未知攻击子集 | 每攻击按说话人轮转 800 假 + 1,600 真 | 12,000（真1,600/假10,400） | `1d6aa677…ac6e0069` |

数据目录为 `/home/lanjianyu/worktrees/issue-29-bandwidth-augmentation/data/`（指向共享只读
媒体的符号链接）。三个 dev 条件与未知攻击子集与 Issue #19 完全一致，比较使用同一清单。
A07–A19 子集只在最终评分时使用一次，不用于调参；ASVspoof2021 与 in-the-wild 仍为保留
holdout。

## 训练与资源

单组 seed42 增强训练，配置
`docs/experiments/bandwidth-augmentation-20261012/bandwidth-seed42.json`
（`config_sha256 = ffe1595d…c1900991`）。与 Issue #19 的 `logmel-seed42.json` 逐字段相同，
仅新增 `augmentation` 段、改 `output_dir`；`probability=0.5`、`target_sample_rate=8000`。

| 项 | 值 |
| --- | --- |
| batch / workers | 256 / 4 |
| AMP | BF16（`grad_scaler_enabled=false`） |
| lr / wd / patience | 1e-3 / 1e-4 / 5 |
| 训练轮数 | 27（early stopping，best epoch 22） |
| 耗时 | 979.5 s（约 16.3 min，约 36 s/epoch） |
| 峰值显存 | 分配 3.24 GiB / 保留 5.37 GiB |
| checkpoint SHA-256 | `c860394c…08c3fb51` |

## 结果

候选 checkpoint 在同一 `infer`/`evaluate`/`compare_scores.py` 路径上重测四个条件，覆盖率
均为 1.0、0 失败；ΔEER 为相对 Issue #19 同种子基线评分文件的配对差值
（`common_coverage = 1.0`）。

| 条件 | 基线（#19） | 本轮增强 | ΔEER |
| --- | ---: | ---: | ---: |
| dev clean | 0.7849% | 0.9027% | **+0.12** |
| dev AAC 64 kbps | 0.9812% | 0.9419% | −0.04 |
| dev 8 kHz 带限 | 13.0965% | 1.3859% | **−11.71** |
| 未知攻击子集 A07–A19 | 2.9712% | 2.7019% | −0.27 |

带限条件是决定性的：8 kHz 退化几乎被抹平（13.10% → 1.39%），代价是 clean 仅上升约
0.12 点；AAC 64 kbps 与未知攻击子集也略有改善。逐攻击看，改善集中在最难的 A17
（11.81% → 8.25%），A14/A15 亦下降；A16、A18、A19 略有上升，A19 从 2.25% 升到 3.75%。
总体仍净改善，说明增益并非来自对单一攻击的过拟合。

| 攻击 | 基线（#19） | 本轮增强 |
| --- | ---: | ---: |
| A07 | 0.6875% | 1.1250% |
| A08 | 0.0625% | 0.0000% |
| A09 | 0.0000% | 0.0000% |
| A10 | 1.0000% | 1.3750% |
| A11 | 0.2500% | 0.3750% |
| A12 | 0.2500% | 0.3750% |
| A13 | 0.2500% | 0.1250% |
| A14 | 1.1250% | 0.5000% |
| A15 | 2.0000% | 1.6875% |
| A16 | 0.7500% | 1.5625% |
| A17 | 11.8125% | 8.2500% |
| A18 | 1.8750% | 3.3750% |
| A19 | 2.2500% | 3.7500% |

完整机器可读结果见 [summary.json](summary.json)（各条件 EER/覆盖率/失败数、逐攻击桶、
相对基线配对比较与各文件 SHA-256）。

## 结论

**保留该方向。** 训练期 8 kHz 带宽增强以极小的 clean 代价（+0.12 点）换取约 11.7 点的
带限退化消除，并顺带改善 AAC 64 kbps 与未知攻击子集；由于单组 seed42 已给出明确且方向
一致的正收益，未再动增强比例，也不需要 seed43 复现。建议将带宽增强作为 LogMel 语音
候选的默认训练项，并作为带宽/编码类退化的首选缓解手段。

## 局限与下一步

- 仅一组 seed42，尚未做 seed43 复现；clean 的 +0.12 点是否稳定需要第二个种子确认。
- `probability=0.5`、`target_sample_rate=8000` 为一次性设定，未做比例/目标率扫描。
- 逐攻击出现有升有降，A18/A19 小幅退化，尚未定位到具体的通道/生成器特征。
- 未与更宽的目标率（如 4 kHz 或 AAC 类编码增强）叠加，也未验证跨来源真实退化。
- 增强仅作用于训练；若官方 I/O 采用其他采样率或编码，仍以 adapter 边界处理。
