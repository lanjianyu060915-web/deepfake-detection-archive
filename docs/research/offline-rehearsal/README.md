# 本机离线验证

更新日期：2026-10-08

## 已完成

- 为 Linux x86_64、CPython 3.12 准备并固定 64 个第三方 CPU wheel；加项目 wheel 后共
  65 个包，在新的 CPU 环境中完成断网安装和 `pip check`。
- 使用真实 LogMel seed42 checkpoint 和 ASVspoof dev 中 12 条 real、12 条 fake 音频，
  完成断网推理、第二次完整运行和受控中断后的恢复。
- 三次运行均得到 24 个有限连续分数，0 失败、0 缺失、0 重复；逐样本分数完全一致。
  中断时已经写入的一条完整记录保持不变，恢复后再次运行没有改变结果文件。

本次 24 条子集 EER 为 25%，只用于确认三次执行一致，不能作为模型效果或比赛成绩。

详细结果、输入、日志和保存的历史脚本见
[2026-10-08 真实模型演练](model-execution-20261008/README.md)。固定 wheel 清单见
[`requirements-cpu-py312.txt`](requirements-cpu-py312.txt)，当时的安装摘要见
[`cpu-installation-summary.json`](cpu-installation-summary.json)。

## 尚未完成

- 视频真实模型的断网安装与推理；
- 音频和视频的代表性大批量吞吐、耗时与资源验收；
- 与官方安全屋系统、输入输出和 EER 程序对齐；
- 最终双分支候选包验证。

这些工作在候选模型冻结后执行。当前 `source/`、`resume/` 和各执行目录中的脚本是历史演练
快照，包含本机路径和当时环境绑定，仅用于追溯，不是通用操作入口。后续直接使用现有
`infer`、`evaluate`、`package` 和 `verify-package` CLI，具体命令见
[开发指南](../../development.md)。不再为每次演练复制一套脚本或绑定无关的历史环境报告。

本机结果证明当前 CPU 环境和该语音子集能够离线运行；不证明安全屋兼容、视频可用、万条
吞吐达标或模型具有足够泛化能力。
