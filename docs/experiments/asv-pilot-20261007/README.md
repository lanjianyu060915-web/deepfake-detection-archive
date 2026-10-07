# ASVspoof 2019 LA 真实语音 pilot

状态：执行准备。归档恢复尚未完成，本目录没有真实媒体验收或模型成绩。

固定选择官方 train/dev 各 256 real + 256 fake，保留 20/10 名 speaker 的
原划分，预先声明 LogMel42、AASIST42、LogMel43、AASIST43 串行顺序。仅使用
语音 train/dev，不读取 ASV eval、比赛测试或最终 holdout。`configs/plan.json`
给出范围、预算、数值故障策略和结果边界；配置中的绝对路径是本机执行记录。

2026-10-07 在任何媒体导入或开发分数产生前，依用户持续授权本地 GPU 的新约定，
将旧 `max_seconds=1200` 改为 `null`。epoch 上限 12、patience 4、采样、学习率、
种子和其他超参保持不变。旧文件原字节保留在 `configs/original-time-cap/`，
新文件及脚本的预评分哈希见 `pre-score-freeze.json`。

`source/prepare_pilot.py` 只处理已通过完整官方大小、MD5 和安全提取检查的
train/dev；真实下载完成日期由执行参数提供，归档 SHA-256 取自实际 acquisition。
保留原 sources 文件，再完成 1,024 条媒体解码、来源/哈希/划分审计、攻击覆盖和
音频时序频谱指纹 + 时长近重复候选检查。候选由人工/代理审查，脚本不会自动合并、
删除或授权训练。未知原 utterance 派生关系仍保持 unknown。

`source/score_pilot.py` 从完整写出的最佳 checkpoint 生成独立 dev 推理配置，
输出 512 条逐样本连续 logit 和错误账，再计算总体、每 speaker 及每攻击对
全部 256 条 real 的开发诊断。攻击诊断复用相同 real，不能当作独立桶比较。
dev 已参与早停且只有 10 个 speaker，不报告正式 CI、晋级或通用音频性能。

当前静态验证：两个脚本 Ruff/格式检查、`MYPYPATH=src uv run --no-sync mypy`
通过，CLI `--help` 正常；launch.sh 的 `bash -n` 通过。四配置经项目严格加载器
加载且没有固定秒数上限。这些检查不等于真实训练运行。
