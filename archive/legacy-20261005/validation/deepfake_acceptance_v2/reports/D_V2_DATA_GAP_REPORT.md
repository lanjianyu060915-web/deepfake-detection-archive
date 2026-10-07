# D-v2 Data Gap Report

| 项目 | 目标 | 当前可信数量 | 缺口 |
| --- | ---: | ---: | ---: |
| REAL Core Blind | 100 | 0 | 100 |
| FAKE Core Blind | 100 | 0 | 100 |
| FAKE生成方法数量 | ≥建议3类 | 0 | ≥3类 |
| 已暴露样本 | 0进入Core | 0进入Core（已识别1,014个唯一暴露内容） | 0进入Core |

## 为什么有标签仍不能入 Core

现有唯一内容中有证据标签的 REAL 500、FAKE 510，但 REAL/FAKE archive 数据属于当前 baseline 训练、阈值/融合调参、开发或既往测试；额外 10 个 FAKE 是 README 明确禁止用于鉴伪评测的开发夹具。它们都不是首次独立盲测样本。

## 负责人需要补充

1. 100 张独立 REAL 原图：保留原始采集链路、设备/场景信息和“未做换脸/生成处理”的核验证据，覆盖人物、设备、分辨率、光照和姿态。
2. 100 张独立 FAKE 原图：至少 3 种有明确名称/版本的生成方法，保留生成脚本/日志、时间、source_face、target_face 和原始输出；建议每类约 34 张并避免人物泄漏。
3. 每张图均需确认未参与当前模型训练、validation、阈值/融合选择、既往测试或预测观察，并由独立验收人签署。
4. 提供当前及上游训练数据的完整逐样本清单；在此之前保持 `TRAINING_DATA_OVERLAP_STATUS = UNKNOWN`。
5. 人工填写独立性声明，运行重复/路径/SHA/解码 validator，达到 REAL 100 + FAKE 100 后再冻结 manifest。

不要用 UNKNOWN、训练帧、开发 query 或其扰动补齐配额。当前 Core Blind 不可冻结。
