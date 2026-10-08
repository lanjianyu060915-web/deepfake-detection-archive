# 完整 CSV 来源组研究获取计划

1. 只读固定作者 CSV 与恢复目录的 33 个 Viewer 页，按数值 fsID、filename、row_idx 排序。
2. 保留且仅保留 Viewer filename 集等于固定 CSV 全部同 fsID filename、每 filename 有 real 与至少 3 种 fake generator 的组；必须恰为 40 fsID / 106 filenames / 530 行，否则停止。
3. 在媒体请求前冻结清单、输入 SHA-256、许可来源和逐条 Freesound fsID 链接。完整仅限作者 CSV 范围，非全部 UrbanSound8K 原录音；不继承 usage 划分。
4. 刷新固定 Hub revision 对应 100 行页；核对 x-revision、全部非 audio 元数据和稳定音频路径；不使用旧签名 URL。
5. 最多 3 并发、45 秒超时、每媒体最多 3 次尝试；逐尝试保留失败。530×3×256089 < 1 GiB；每次请求限制 body 大小。
6. WAV 完整读出 16kHz mono PCM16 / 64000 帧，计算完整 body SHA-256 与 RMS/peak/zero/clipping 后原子发布。断点只恢复同一清单，已发布文件以先前记录哈希复验。
7. 独立离线复算选择和全量媒体验证；不模型、不划分、不训练、不修改旧 64 条或其证据。

用途：本地非商业研究。UrbanSound8K CC-BY-NC-3.0；EnvSDD CC-BY-4.0 不放宽上游许可。比赛使用未批准。逐录音链接保留，原始 FREESOUNDCREDITS.txt 若无法获取则明确作为归属证据缺口，不声称完整许可验收。
