# D-v2 Data Leakage Audit

```text
TRAINING_DATA_OVERLAP_STATUS = UNKNOWN
```

本地微调、验证、既往测试快照可精确检查且已全部排除；但 MobileNetV3 上游预训练数据的逐样本清单、公开数据的完整版本/发布者/许可证链路不可得，因此不能写“无泄漏”。

## 审计范围与口径

- 规范源素材路径：1,390；唯一 SHA-256：1,329。
- 路径标签：REAL 530、FAKE 540、UNKNOWN 320。
- 唯一内容标签：REAL 500、FAKE 510、UNKNOWN 319。
- Core Blind eligible：REAL 0、FAKE 0。
- 已暴露：1,075 路径、1,014 个唯一 SHA；其中 archive 的 1,072 路径均属既有开发/训练材料，另有当前 adapter 已实际运行的 query_001—003。

全工作区静态支持格式（JPG/PNG/TIF）有 3,633 条路径、804 个唯一 SHA，670 个重复组，3,499 条路径位于重复组，最大重复度 450。大量内容是 task、报告、review package 和环境资产副本，候选清单只登记规范源位置，避免按副本膨胀样本量；另外 2 个 GIF 均是第三方环境资产，与候选无关。

## 已知训练/开发重叠

`data_all.zip` 的 README 明确声明：formal train 用于频域训练或融合调参，val 用于阈值/权重选择，test 用于既有指标。manifest 中 1,060 个带标签帧因此全部排除：

| 集合 | 路径数 | REAL | FAKE | 处置 |
| --- | ---: | ---: | ---: | --- |
| formal train | 700 | 350 | 350 | 已知训练/调参，排除 |
| formal val | 150 | 75 | 75 | 已知阈值/融合选择，排除 |
| formal test | 150 | 75 | 75 | 已有指标观察，排除 |
| demo | 60 | 30 | 30 | 开发集且全部重复 formal，排除 |

formal 的 200/200 视频路径和 SHA-256、1,000/1,000 帧路径已核对。demo 10 个 pair 全在 formal，60 个帧形成 60 组跨集合完全重复；4 个 pair 还发生 demo/formal 跨 split 重排（272、395、728、775），不能混合宣称独立统计。

## 重复与近重复检查

- `OVERLAP_CHECK.csv`：已知 baseline 数据成员 1,060 条。
- 候选内部 SHA-256 完全重复关系 61 条：60 条 formal/demo 帧重复，1 条 stress 图重复。
- 同名异内容关系 44 条，保留人工复核。
- 本地 318 个规范源素材与 archive 唯一帧之间 SHA-256 命中 0。
- 本地与 archive 跨集合 dHash64 Hamming 距离 ≤4 且 SHA 不同的线索 0。

dHash 只是低分辨率感知线索，不能证明无近重复。源训练全集不可得仍是总体状态 UNKNOWN 的决定性原因。

## 标签与暴露决策

- 主 A/B query 10 张只有身份/底图真值，没有 REAL/FAKE 生成证据，全部 UNKNOWN；query_001—003 另标 `previously_exposed=true`。
- Open Images/source/identity 图片的来源信息不等于鉴伪 REAL 真值，全部 UNKNOWN。
- source-search package 的 10 张 query 有 README 合成证据，可标 FAKE；同一 README 明确其不用于 AIGC 鉴伪评价，故排除。
- 12 条 stress 路径没有父标签继承证据，标 UNKNOWN。

本审计未调用真实鉴伪 runner，模型预测未用于任何标签。
