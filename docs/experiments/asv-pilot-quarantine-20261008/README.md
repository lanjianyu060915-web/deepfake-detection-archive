# ASV pilot：未确认候选的保守隔离系列

本目录在任何 ASV 模型分数之前建立。预登记见 [PLAN.md](PLAN.md)，
实际准备逻辑见 [source/prepare_quarantine.py](source/prepare_quarantine.py)，
完整准备验证见 [verification.json](verification.json)。两模型、两种子训练及全 dev
评分已完成，实际证据见 [training/summary.json](training/summary.json)。
公开研究许可不等于比赛外部资源许可。

## 实际审计结果

| 集合 | 真 / 假 | 原说话人数 | 保守统计组数 |
| --- | --- | --- | --- |
| train | 243 / 243 | 19 | 19 |
| dev | 256 / 256 | 20（真 20、假 10） | 19（真 19、假 9） |
| 隔离 | 13 / 13 | 1（LA_0087） | 原组件 LA_T_1006969 |

两个指纹匹配仍未确认。dev 内 LA_D_1032020 ↔ LA_D_1073474 只作保守关系连接；
跨集合 LA_D_1062379 ↔ LA_T_1023667 导致其原训练说话人组件的全部 26 条退出 active。
候选相似度及原证据原样保存在 summary.json。active 中已知候选跨划分数为 0。
全部 dev 512 条保留，媒体/身份/group 字段不变；新增 derivation_group_id 保存完整
1,024 条集合加两个候选边的传递关系。候选合并不声明说话人相同或重复已证实。

- active 标准 SampleRecord 清单 998 行：manifest-decoded.jsonl。
- 全集同 schema 账本 1,024 行：all-records-ledger.jsonl；26 条隔离记录标为
  status=excluded、split=unassigned，并显式填写 failure_reason。
- excluded.json 保留全部 26 个 ID、原完整记录（包括原 split）、字节数与排除原因。
  原媒体文件没有移动、删除或改写。
- 原全部 1,024 条媒体重新计算实际哈希、核对许可与来源，审计通过。完整输出见
  byte-source-audit.json；decode_status=ok 绑定原真实解码清单及预检哈希，未重复解码。
- manifest 仅 train/dev。字节审计临时用 unassigned 视图，实际划分用完整关系独立验证；
  没有读取 holdout/eval。审计警告源于未登记的其它媒体关系字段，不据此声称关系完全已知。

## 配置与运行边界

四份新配置只修改 manifest、feature_cache、output_dir 三键；全部其它字段、
超参数、种子、预处理、来源路径均与冻结原配置相同。逐项旧/新哈希在 summary.json。
准备脚本使用项目公共哈希、连通组、来源与审计函数，没有自建哈希或连通算法。

独立目录：

- 数据：data/asvspoof2019-la-pilot-quarantine/
- 配置与遥测：outputs/experiment-planning/asv-pilot-quarantine/
- 后续训练输出：outputs/experiments/asv-pilot-quarantine/<run>/
- 特征缓存：cache/asvspoof2019-la-pilot-quarantine/<architecture>/

新 source/launch.sh 从旧 launcher 派生，保留 GPU 监控、GNU time、日志和退出清理；
旧的 20/10 组和 256/256 训练分母守卫改为调用 source/check_pilot.py。
此检查验证实际 998 条清单、26 条排除账本、身份/媒体字段保留、完整候选关系、
四配置哈希及全部原媒体实际字节。检查本身不会训练，脚本不覆盖现有训练输出或遥测。

可单独重复检查（不启动训练）：

```bash
.venv/bin/python docs/experiments/asv-pilot-quarantine-20261008/source/check_pilot.py \
  --config outputs/experiment-planning/asv-pilot-quarantine/logmel-seed42.json
```

四配置预检、Ruff、format、严格 mypy、bash -n 全通过；重复准备按预期拒绝且数据字节
未变，篡改 seed 的配置也在训练前被拒绝。未把这些工程检查当成模型效果。

该系列删除了一个完整训练组件，不能冒称与原 1,024 条计划在同一数据集上可比。
dev 仅 19 保守组、假类 9 组，且仍参与早停；只报告开发探索点估计，不作正式 CI、
最终留出结论或阶段晋级。通用音频覆盖仍需独立数据验证。

## 精确摘要

- original_manifest_hash: `a93063bb743325c7258fc044c1034813285f6e60200f8edc006f3f5bdf4d04ed`
- active_manifest_hash: `2b69ab30b06e5942c69b729ef79f583a44f8435e07f0d841bc1ee2def8be424e`
- ledger_manifest_hash: `37f8bd494c6be371c3e821934567f5941d760473f9d9a6106cfd24a5b5a4cfa7`
- script_sha256: `197a0b2d5f33a797c1d7d34bf2249e858c7758b7d84b8227aa0c38d4ffaff71c`
- sources_sha256: `6ecc3a94e6b11d60cf4da4b1b5790aa251d95ee75c03df0ce63a4f5f62187039`
- 隔离媒体总字节：1725360；原 1,024 条字节复核全部通过。

## 2026-10-08：真实两模型两种子结果

| 模型 / seed | dev EER | best epoch / 完成 epochs | 全dev评分 / 失败 |
| --- | ---: | --- | --- |
| LogMel / 42 | 21.4844% | 11 / 12 | 512 / 0 |
| LogMel / 43 | 24.2188% | 11 / 12 | 512 / 0 |
| AASIST-L / 42 | 25.7813% | 5 / 9 | 512 / 0 |
| AASIST-L / 43 retry1 | 29.6875% | 12 / 12 | 512 / 0 |

AASIST43 原运行中断在10epoch、没有完成/退出记录，恢复时原进程不存在。
训练器不支持优化器恢复；[retry1](retry1/README.md)从相同种子/方法重做，
配置只改output_dir，原部分结果保留但不纳入完成模型比较。

[source/verify_completed.py](source/verify_completed.py)复用已独立审阅的训练绑定与
独立阈值扫描，严格核对真实initial/best参数变化、最佳epoch/更新、来源/清单/代码/配置、
精确512个dev ID及逐条元数据；EER与训练选择、独立全dev评分一致。
训练、评分退出0，训练解码失败为空。全部原始配置、history、scores、遥测归档。
LogMel42的旧评分仅有GNU time；其余三次补齐设备级GPU采样，不为旧运行补造采样。

新评分wrapper的noqa位置与retry scorer格式在执行后规范化，AST完全相同；
实际已执行字节保留在source/score_with_telemetry.executed.py.txt与
source/score_retry1.executed.py.txt，运行hash按这些快照绑定，不冒称执行了后来的格式版本。

LogMel两种子在该开发集点估计更低，只作为后续语音实验的优先候选，不作正式晋级。
AASIST-L实际torch峰值allocated为3,633,876,992、reserved为7,897,874,432字节
（reserved约7.36GiB），超过最初7GiB资源目标；设备采样峰值42/43为7917/7878MiB，
温度峰值84/88°C，0OOM。显示/其它设备负载与allocator reserved须分开理解；
后续AASIST扩大训练前需重新profile资源，不假定8GB已留足1GB余量。

尚无通用非语音泛化、来源/生成器留出、最终holdout、正式CI、官方EER对齐或比赛许可结论。
