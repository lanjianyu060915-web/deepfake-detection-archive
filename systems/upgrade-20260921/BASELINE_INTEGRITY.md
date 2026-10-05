# 冻结基线完整性

检查日期：2026-08-20（Asia/Shanghai）

在创建正式集成产品前，按第1.5轮冻结清单重新计算文件大小与 SHA-256；复制运行源码后又对产品内副本逐文件复核。

| 对象 | 冻结清单 | 原冻结目录复核 | 产品运行子集复核 |
|---|---|---:|---:|
| A 人脸身份搜索 | `A_CODE_SHA256.csv` | 15/15 MATCH | 10/10 MATCH |
| B 原始底图搜索 | `B_CODE_SHA256.csv` | 22/22 MATCH | 11/11 MATCH |
| buffalo_l 模型 | `MODEL_ACTUAL_SHA256.csv` | 5/5 MATCH | 配置引用，不复制 |

缺失、大小差异和哈希差异均为 0。正式来源为：

- A：`D:\刑侦查竞赛\first_week\_round1_work\backend\A_identity`
- B：`D:\刑侦查竞赛\first_week\_round1_work\backend\B_source`
- 模型：`C:\Users\lanji\.insightface\models\buffalo_l`
- 冻结清单：`D:\刑侦查竞赛\first_week\_round1_5_blind_acceptance\frozen_manifests`

本产品没有修改 A/B 的模型、特征、相似度、聚合、HSV、SIFT、RANSAC、融合权重、shortlist 或排序公式；仅由独立子进程调用冻结 CLI。

```text
A = MATCH
B = MATCH
MODEL = MATCH
```

