# EnvSDD 原 64 条非语音获取恢复

原固定来源与选样不变，UrbanSound8K 32 个 fsID 各取 1 real / 1 AudioLDM2 fake。
旧暂停时 19 条的 SHA-256 全部复验，新签名链接仍指向固定 revision
`b0832726deb55484dbce869c6f09c41f7e9e5fbb`。新获取先完成 63/64，
row15003 发生截断响应，未发布错误媒体；同一来源重新获取后完成 64/64。

现在 32 real / 32 fake、64 个唯一字节哈希、8,194,816 字节。
全部 WAV 实际读出 16kHz 单声道 PCM16、64,000 samples / 4 秒，0 零信号，
0 最终失败；root 独立复核每条字节、完整帧数和记录哈希。
获取失败、恢复、新来源页面及原始执行脚本均保留；成功不能抹去首次失败。
证据见 `original64-execution.json`、`original64-verification.json` 与 `root-review.json`。

这只是原 64 条探索子集：完整 fsID 家族尚未取得，未分训练/开发/留出，
未训练或评分，不能作为非语音覆盖或泛化验收。需按原录音 fsID 统管其全部切片
与生成派生，再进行组级划分；旧上游 split 不可直接当作已无泄漏。
本地非商业研究范围遵守 UrbanSound8K CC-BY-NC-3.0 并保留 Freesound 归属。
正式比赛使用尚未批准，不包含 DCASE/BBC 等其他来源。

`viewer-pages/` 是本轮已取得的扩展元数据预审页，不等于对应媒体已下载、
完整 fsID 已覆盖或许可已通过。后续系列须独立登记选样、缺失与用途。
