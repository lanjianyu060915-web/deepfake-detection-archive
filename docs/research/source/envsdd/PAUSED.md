# EnvSDD UrbanSound8K 下载暂停记录

状态：用户于 2026-10-07 明确暂停；本记录后未再发起网络请求或扩展。

- 固定清单：`data/envsdd-urban/manifest-v1-32fsid.json`，SHA-256 `ade4742cc94a590bf55195ff512e91f6dba00eb7a88c7c17ffe10566757b458b`。
- 已完成：19/64 个 WAV；real=10，fake=9。
- 未完成：45 个（real=22，fake=23）。
- 已完成父 `fsID`：[17615, 17853, 31323, 44736, 50901, 55020, 57320, 57584, 69304, 73277]；其中未完整闭合的父组：[73277]。不得将不完整父组用于任何训练/评估或独立性结论。
- 全部逐文件存在性、RIFF 头、字节数与 SHA-256 见 `data/envsdd-urban/acquisition-status-paused.json`。

证据边界：作者 CSV 固定 commit 为 `d6179d1c773ca78e4939be5933c9914f97815920`，SHA-256 为
`cda95f3cafaed74193a6c988904df8de736a48e42ef7b508df47bd0776d8fe0a`；Viewer 元数据响应为 revision
`b0832726deb55484dbce869c6f09c41f7e9e5fbb`。CSV 和归档 Viewer 页可复核来源关系；Viewer 的
`audio.src` 是会过期的缓存 URL，已保存 WAV 的 SHA-256 才是本地媒体的稳定证据。

恢复命令（恢复前须解除暂停并重新核对许可/赛方规则；下载器会跳过已存在文件）：

```bash
python docs/research/source/envsdd/fetch_urbansound8k_subset.py --manifest data/envsdd-urban/manifest-v1-32fsid.json --destination data/envsdd-urban
```
