# [音频] 完成全量 LogMel/HuBERT 同划分对照

依据：全量语音实验（`docs/experiments/asv-main-20261009/README.md`）。
LogMel42已完成train25,379/dev24,844、30轮、EER0.510204%，不能与旧dev512直接比较。
HuBERT只有batch32的真实profile，尚未训练。前置：云端环境与资产Issue。

工作：

- 保持同一训练清单、单片段64,600采样点、内部标签方向和全部dev；不触碰ASV eval或最终holdout。
- 运行现有`logmel-seed43.json`、`hubert-seed42.json`、`hubert-seed43.json`。
  云端按一次profile调整batch/feature_batch_size；任何协议变化在实验卡标明。
- 输出目录使用新的云端run ID，不覆盖迁入的LogMel42。先取得基线，再考虑局部解冻或融合。
- 报告总体及A01–A06分组EER、失败率、时间/吞吐/显存、种子和最佳轮次。
  HuBERT缓存BF16与当前离线FP32编码路径有差异，端到端验证用实际分数报告。

交付：一份PR，包含必要源码（有实际问题才改）、实值配置、简洁实验卡；大权重/分数在团队
持久存储给出路径与哈希。代码变更跑相关检查及一次标准检查。训练无提升也如实完成。
完成条件：上述三次运行有明确终态及可比结果；实际资源/数据阻断需记录并上报，不能当作实验完成；
不得以只跑mock或profile代替训练结果。负责人审查合并，不在本任务建恢复/遥测平台。
