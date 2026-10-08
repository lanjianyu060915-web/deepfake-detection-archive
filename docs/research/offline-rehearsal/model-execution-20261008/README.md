# 真实 LogMel42：24 条本机 CPU 断网与恢复演练

2026-10-08 实际执行与独立只读复核通过。选取隔离系列 dev 按 sample_id
排序的前 12 real、12 fake；选择不参考模型分数。主 cu130 环境未替换，
使用之前断网安装的固定 65 包 CPU 环境。

第一次 prepare 因来源 JSON list / SourceRecord tuple 比较失败；
原 job、stdout、stderr 保存在 `failed-*`，修复前脚本与回归见 `../resume/`。
新增 round-trip 正例修复前失败，修复后 64 守卫测试、Ruff/format/严格 mypy
及独立复核通过；使用新 v2 job/stage，未覆盖失败目录。

## 真实执行证据

- LogMel seed42 实际 initial/best 参数变化、最佳 epoch 11、截至最佳 epoch 的
  88 次更新独立核验；checkpoint SHA 为
  `418c0abf06f11d19bda01ee1695078712bf08ceedb6cb8a1a43ece41cb7f9677`。
- CPU torch 2.14.1+cpu；65 个 wheel/分发包、35 个源码/安装/wheel 文件成员一致。
  40 个 staged assets 包含真实媒体、清单、权重、来源、许可及实际训练记录。
- `unshare --user --map-root-user --net` 实际退出 0，用时 13.1584 秒。
  执行网络 namespace 与父进程不同，路由为空，外部 TCP 连接以 errno 101 失败。
  两次完整推理均在该 namespace 内，以空 Hugging Face/XDG 缓存运行。
- reference、offline、resume 各 24 个有限连续分数，缺样、重复、失败均为 0；
  逐样本分数完全相同，最大差 0.0。三次本地子集 EER 均为 25%，只用于核对执行一致性。
- 自有 PID 10387 的 `/proc` 原文观察到 `T (stopped)`，停止及退出均保留
  1 条完整分数，原始前缀字节一致。SIGSTOP → SIGINT → SIGCONT 受控中断
  退出 -2，保留 KeyboardInterrupt stderr 与回收记录；这不是正常退出。
  恢复保留原前缀并补齐 24 条，再次 resume 文件字节不变。
- 7 条正常命令、环境检查和 namespace launcher 均退出 0。执行期间
  Git HEAD 为 62e82f2、dirty=true，跟踪文件冻结；结束后才创建修复/结果提交。

[主记录](results/model-probe.json)、[准备绑定](results/preparation.json)、
[namespace 记录](results/namespace-launch.json) 和原始停止状态/前缀均保留。
`archive-manifest.json` 冻结复制时的文件哈希，不包含其自身及之后新增的说明/复核。
主代理验证原 stage 与归档逐字节一致，并独立比较三套 ID→score；
另有独立代理只读校验真实媒体、权重参数、环境、停止与恢复证据。
结论见 [root-review.json](root-review.json)。

## 范围与复现

`job-v2.json` 与执行脚本保存精确本机路径；原大权重、媒体与 wheelhouse
仍在忽略目录，克隆 Git 本身不能直接复现。已有 stage/results 不允许覆盖；
需要重做时创建新 job/stage、重新绑定实际训练和环境，并在执行期间冻结 Git。
正常运行时无需修改既有 CPU 环境或替换主 CUDA 环境。

本次只证明 24 条真实语音的本机 CPU 断网、重复与恢复机制。
官方安全屋、视频、完整集吞吐、代表性 500 条和官方 EER 对齐尚未验收，
不能从本次子集 EER 推出模型效果或比赛完成。
