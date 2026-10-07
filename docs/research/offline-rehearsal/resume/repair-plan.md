# 真实模型离线演练：修复与验收边界

本轮只修复 `resume/model_rehearsal.py` 并验证其守卫，不启动 detector 演练，
不生成或使用伪造训练 checkpoint。旧 `source/` 和 CPU 安装证据保持原字节。

1. 用负例测试锁定训练完成、最佳 epoch、更新次数、训练/dev 计数、来源/清单/权重
   哈希及精确 dev 成员绑定；检查真实参数变化，不能只信布尔或序列化哈希变化。
2. 绑定既有 CPU 安装报告及 65 个 wheel、实际 venv prefix/site-packages、CPU torch、
   安装包与 wheel 和当前源码逐成员哈希；执行前复核 job、脚本和 staged assets。
3. 为 namespace、环境检查和普通命令设有限 timeout。只清理自有 Popen，
   TERM 超时后 KILL/reap，保留异常、退出码和清理结果。
4. 实际观察 `/proc/<pid>/status` 的停止状态，保存 SIGSTOP 原始字节，
   验证退出后的完整行保留停止时前缀；再验证 resume 保留退出前缀。
5. 执行本目录针对性 pytest、Ruff/format、严格 mypy，记录精确脚本及测试 SHA。
   作者提交证据给 root 独立审阅，通过后由 root 冻结 Git 并提供真实音频训练资产。

测试仅使用元数据 fixture、临时文本和自有无模型 Python 子进程。
不把守卫测试、现有安装成功或合成进程探针当作真实模型验收。
当前适用音频 waveform 推理；视频/冻结编码器需单独适配后审阅。
