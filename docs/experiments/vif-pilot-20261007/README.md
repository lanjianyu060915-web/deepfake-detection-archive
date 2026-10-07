# ViF Panda pilot 恢复审计结果

2026-10-07 用户明确恢复后执行。范围限本地非商业计算研究；未训练检测头，未产生鉴伪成绩，未通过正式 P1/P2，赛用许可仍待核验。

## 输入与冻结边界

恢复前复验 1,024 条媒体及帧缓存 SHA-256；缓存 616,693,760 字节，0 解码失败，6 条低信息 pHash 仍明确未完成该项审计。预注册、两份训练配置、selection、decode-audit 与原快照逐字节一致，ResNet18 权重 SHA 与预注册一致。详见 `resume-preflight.json`。

原脚本 `docs/research/source/vif-pilot/prepare_pilot.py` 保持 SHA
`c0e0b91dac5e87ae19c07928235295d57e7a5017591c8d24e45244abeda55bdb`。
运行通过 `run_audit.py` 导入原脚本，只将其 `DOC` 输出目录改到本目录；工作目录、所有算法、阈值和配置保持原值，避免重写历史来源快照。
原实际执行 wrapper 字节保存在 `run_audit.executed.py.txt`，SHA
`7351451a2516539a955fc34e0ca2f1667a1030fa2f6ebdf0c5e7963756d7ee65`。
运行后仅对 wrapper 做 import 排版、显式加载失败异常、动态模块赋值的类型检查修复；当前 wrapper 通过 Ruff、格式及严格 mypy，原脚本 7 项行为测试通过。修复未重跑审计，两个 wrapper 哈希均见 `verification.json`。

## 实际运行

```bash
PYTHONPATH=src .venv/bin/python docs/experiments/vif-pilot-20261007/run_audit.py features
PYTHONPATH=src .venv/bin/python docs/experiments/vif-pilot-20261007/run_audit.py lock
```

命令已执行成功，不能再次覆盖本实验产物。精确命令、退出码和全进程墙钟时间分别在 `features-execution.json`、`lock-execution.json`；GNU time 原始 stderr 和 GPU 每秒采样 CSV 保留。普通沙箱 CUDA 不可见，实际 features 在已授权的沙箱外 GPU 执行；无需安装/更换驱动。

- 环境：Python 3.12.13，torch 2.14.1+cu130，RTX 4060 Laptop 8GB，驱动 616.92。
- 冻结特征形状 `[1024, 4, 512]`，float32，无 AMP、无参数更新；前向阶段 5.7569 秒，整个进程 10.6960 秒。
- PyTorch 峰值 allocated 494,337,024 字节、reserved 580,911,104 字节；设备采样最大 2,419MiB、76°C、66% 利用率。设备内存含桌面等其他进程，10 个采样点可能漏过瞬时峰值。
- 523,776 对比较、逐样本 top3 近邻已记录；固定 pHash/余弦阈值产生 0 个候选、0 个新增合并边。这仅表示本审计未发现阈值候选，不证明完全无重复。
- 保守关系组件 249 个，最大 6 条，无巨组件；6 条低信息 pHash 未被静默删除，也未改变阈值。

## 锁定划分与风险

| 划分 | 条数 | real | fake |
| --- | ---: | ---: | ---: |
| train | 716 | 358 | 358 |
| dev | 154 | 77 | 77 |
| holdout | 154 | 77 | 77 |

每份覆盖全部 7 个生成器。清单 SHA-256：
`f3b004bb94174d02899b9a8763c3619da121f90f53f919e5884e171c716d6638`。
实际清单、split-lock、关系保留表及近邻原始记录均存本目录与工作目录。
重新验证未发现已知关系跨划分；`identity_id/group_id/original_id` 全部仍为 unknown，249 个组件不能解释为 249 个独立统计组，不计算独立组置信区间。

**存在明显宽高比捷径。** 真实媒体 483/512 的宽为 456；生成媒体仅 87/512。生成媒体独有的宽 384/436/452/454/466 共 412/512 条；224×224 保比例补边仍会保留宽高比。codec、container、height、pixel_format 在本样本恒定；各划分帧率边际真假对称。时长中位数真实 4.625 秒、生成 5.041667 秒，也存在关联。详见 `shortcut-diagnostics.json`。

固定探索基线可按原配置执行，但任何低 EER 都不能归因为内容层面的伪造识别，更不能直接晋级正式候选。后续需另行预注册对称预处理/编码及宽高比控制消融，避免将此次审计后变更冒充原实验；不得借 holdout 调参。

仅用 train 的媒体参数拟合标准化 LogisticRegression，再报告同一锁定 dev 的诊断：
宽度单变量 EER 为 **24.75%**，宽度/高度/帧率/时长/码率联合为 **31.17%**。
仅 154 条开发样本，未使用留出、未计算独立组 CI，也未据此更改本轮检测器参数。
这表明低成本媒体参数含有标签信息；联合模型较差不能解释为捷径消失。
精确脚本、连续开发分数和输入哈希见 `source/metadata_shortcut.py` 及
`metadata-shortcut.json`。该对照用于风险诊断，不是参赛检测器。

## 后续启动入口（本次未执行）

主代理协调单 GPU 后按两份已冻结配置启动：

```bash
PYTHONPATH=src .venv/bin/python -m deepfake_detection_archive train --config outputs/experiment-planning/vif-pilot/train-seed42.json
PYTHONPATH=src .venv/bin/python -m deepfake_detection_archive train --config outputs/experiment-planning/vif-pilot/train-seed43.json
```

训练只消费 train/dev，holdout 留待冻结验收。研究产物不得自动转为参赛资源批准。
