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

## 2026-10-08：真实来源记录序列化修复

LogMel seed42 的实际 prepare 在来源登记比较处失败：保存为 JSON 后，
`allowed_uses` 变为 list，而读取来源登记得到 tuple。先保留失败日志和修复前脚本，
用真实 JSON round-trip 的正例重现失败，并用许可、版本、日期、允许用途变更负例
锁定来源绑定。比较改用现有规范 JSON 哈希，只统一序列化表示，仍逐字段绑定。
通过针对性检查与独立复核后，使用新的 job/stage 重做 prepare；旧失败目录不覆盖。
本次新增授权步骤是实际训练模型的断网推理、重复与中断恢复演练，
执行期间冻结 Git 与跟踪文件，不把 24 条本地演练当作安全屋或完整集验收。
