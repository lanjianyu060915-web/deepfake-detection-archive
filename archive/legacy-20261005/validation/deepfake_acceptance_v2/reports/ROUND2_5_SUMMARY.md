# 第2.5轮 D-v2 鉴伪验收体系总结

| 项目 | 状态 |
| --- | --- |
| 当前鉴伪baseline是否保持不变 | 是；只读核验 50/50 PASS |
| 找到可信REAL候选数 | 500 个唯一SHA（530路径），但均为训练/调参/既往测试重叠，Core可用0 |
| 找到可信FAKE候选数 | 510 个唯一SHA（540路径），但均为重叠或开发夹具，Core可用0 |
| Core Blind是否可冻结 | 否；独立 REAL 0 / FAKE 0 |
| ground truth规范 | PASS；固定15字段，正式集只允许REAL/FAKE |
| 数据validator | PASS（6项数据治理自动测试通过）；当前空Core为DATA_PENDING |
| 扰动生成器 | PASS；9种确定性衍生、标签继承、parent聚合契约 |
| 自动evaluator | PASS；严格映射、固定分母、分组、条件AUC、逐样本归档 |
| stub测试 | PASS；7项测试覆盖8种指定模式+2种附加失败模式 |
| 是否运行真实Core Blind | 否 |

## 结论

```text
DATA_COLLECTION_REQUIRED
ENGINEERING_PASS_DATA_PENDING
```

数据规范、候选审计、真值模板、manifest/SHA 冻结、validator、扰动器、evaluator、stub、盲测协议和基线完整性核验均完成。没有给 UNKNOWN 强贴标签，没有复制虚假 200 张集，没有修改 baseline，也没有运行真实 D-v2 鉴伪。

现有 archive 带标签数据已明确用于当前 baseline 的训练/调参/既往评价，且训练全集仍不完全可核。因此下一步是由负责人组织并隔离收集独立 REAL 100、FAKE 100（至少 3 种有记录方法），完成独立性签署后再冻结 Core Blind。
