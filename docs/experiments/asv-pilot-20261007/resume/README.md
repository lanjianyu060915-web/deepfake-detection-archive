# ASV 实际提取与 speaker 预期修正

2026-10-07 恢复执行。完整归档 7,640,952,520 字节、官方 MD5
`30c98f11d8b2bc21f2c257bfd78bb5c5`、SHA-256
`208a7e4e3913f8c75ae1afd19bf32a5b29ae68435e9e30e23e5e98b6a155e4ec`
通过。修复后的 finalizer 实际运行 83.66 秒，原子发布 50,224 条 train/dev
FLAC、2 份 CM 协议，共 50,226 个文件。142 个协议外 dev 文件仅记录 ZIP
元数据，未读内容；eval 未读内容或提取。媒体尚不能据此声称已解码。

第一次完整 importer 在读取全部 train/dev 协议与媒体哈希后，于 5.77 秒安全拒绝：
项目将 dev 整体 speaker 数误设为 10。没有发布 pilot，没有运行解码或训练。
原始拒绝全文、运行参数、脚本与输出哈希保留在 `import-*`，不覆盖。
`run_import.py` 的 `git_head` 字段在阶段结束采集；各工具的 SHA-256 另记，
父代理并行提交文档可能使结束 HEAD 与开始 HEAD 不同。

## 真实协议证据与预评分修订

`protocol-speaker-audit.json` 直接统计已通过官方归档校验的两份 CM 协议：

| 划分 | 整体 speaker | bonafide speaker | spoof speaker | 真实条数 | 生成条数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 20 | 20 | 20 | 2,580 | 22,800 |
| dev | 20 | 20 | 10 | 2,548 | 22,296 |

train/dev speaker 联集无交集。每个 A01–A06 的 train speaker 为 20，dev 为 10。
因此仅将主动 importer 的 dev 整体 speaker 常量由 10 改为 20；标签计数、
全量先验校验、缺 speaker / 跨划分 / 缺媒体拒绝行为继续保留。
新增回归覆盖真假 speaker 数不对称的正常 dev、标签计数正确但缺 dev speaker
的拒绝边界以及正式整体 speaker 常量。

历史 `source/prepare_pilot.py` 与旧方案保留。在 `resume/source/prepare_pilot.py`
复制原脚本，仅将 dev speaker 断言由 10 改为 20；原脚本不存在 DOC 输出目录变量，
其 `data/asvspoof2019-la-pilot/` 输出仍是已登记且尚不存在的路径。
`pre-score-speaker-revision.json` 在任何真实模型训练/分数出现前记录依据、修订及
未改配置哈希；不改变每类 256、选择算法、两种子、模型或超参。

当前修订等待主代理独立审查，尚未重新执行 importer 或 pilot。
Ruff 与格式通过，两个脚本严格 mypy 通过；ASV acquisition/importer 目标回归
**71 passed（0.45 秒）**。这些测试验证工程行为，不是模型表现。
来源许可范围仍是本地研究；比赛使用未确认。语音训练集不是通用音频覆盖。
即使 dev 整体有 20 speaker，也不能称其为 20 个双类 speaker：只有 10 个
有 spoof，沿用原 pilot 的不作正式 CI / 晋级结论边界。
