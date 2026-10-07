# Candidate inventory audit notes

`CANDIDATE_INVENTORY.csv` 有 1,390 条规范源路径、1,329 个唯一 SHA-256。路径计数为 REAL 530、FAKE 540、UNKNOWN 320；唯一内容计数为 REAL 500、FAKE 510、UNKNOWN 319。所有条目的 `is_usable=false`，因此 Core Blind eligible 为 REAL 0、FAKE 0。

这不是矛盾：有标签证据不等于具有独立性。archive 的 1,060 个带标签帧属于已知训练/调参/既往测试/开发数据，另有 10 个 FAKE 是明确禁止用于鉴伪评测的 source-search 合成夹具。

`absolute_path` 对 zip member 使用 `zip://...!/member`，同时保留 `archive_path` 和 `archive_member`，避免把压缩包成员伪装成普通磁盘文件。

`OVERLAP_CHECK.csv` 是证据表：已知 baseline 数据成员 1,060 条、SHA-256 完全重复关系 61 条、同名异内容关系 44 条；直接素材与 archive 未发现 SHA-256 命中，跨集合 dHash64 距离 ≤4 的候选为 0。感知哈希只用于人工复核线索，不作为相等证明。
