# official → three_module dev 复制基线记录

复制时间：2026-08-25 13:58–13:59（Asia/Shanghai）

源目录：`D:\刑侦查竞赛\face_source_trace_two_search_v0.1_official`  
开发目录：`D:\刑侦查竞赛\face_source_trace_three_module_v0.2_dev`

复制完成时：

- 源文件数：28,361
- 目标文件数：28,361
- 源字节数：3,558,563,829
- 目标字节数：3,558,563,829
- Robocopy：0 failed，0 mismatch

关键基线文件在开始开发前逐一 SHA-256 相同：

| 文件 | SHA-256 |
| --- | --- |
| `orchestrator.py` | `730fd7522ea904e18d1222e39b69cbc6a8317295dd603167a95c01bacd241685` |
| `launcher.py` | `e067a9af200111ad23ebffdec4ad35000f7b1773fd349c0d7258f3aa814fa80a` |
| `config.json` | `fede924962f89ef748c82f506e33200d8429c7a63cbc3691ab11fa97ca2e940b` |
| `frontend/app.py` | `e46c91aa6dcb228acbe99127f15f1985d3b360d8a30c14e4d017dfd2cd81a07b` |
| 身份 `searcher.py` | `3f9120370ba4becd85962fa66486130270c817dc980fc5b7658cd7a5382470a0` |
| 底图 `pipeline.py` | `4a8825a6db17c21eb9817f6177e895d9dbde4d3dad8b4df875e45db1b27990c8` |
| 身份缓存 JSON | `59ca4e8344e519ec3d84b104e8ee4f7343df514ebbf84013d8a3100ffed23538` |
| 身份缓存 NPZ | `75bd239089ab5f65d6ac0b4ced3d71724596562f2031b387e9c42409107693b6` |

后续所有修改只允许发生在开发目录。源 official 和 AIGC 安装目录保持只读；交付前会再次复核源目录关键哈希。
