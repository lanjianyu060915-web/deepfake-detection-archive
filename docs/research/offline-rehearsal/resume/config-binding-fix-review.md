# 实验模型配置绑定修复（2026-10-08，待独立复审）

本轮修复独立复审定位的单项缺口：实际 experiment config 的 `model_config`
原先未与 trainer/checkpoint 模型配置相连。即使配置指定 `channels=32`，
而 run 和 initial/best checkpoint 一致使用 `channels=24`，原守卫仍可放行。

## 最小修改与验证边界

- `model_rehearsal.py` 仅新增 9 行。在实际配置文件及 trainer settings 校验后，
  使用既有 `create_model` 和 `model_spec` 展开默认值，比较完整
  `(architecture, model_config)` 与 run；后续已有 initial/best 与 run 的比较继续生效。
- 新增 3 项测试：实际配置省略全部模型参数、只显式提供 `channels=24` 均接受；
  配置 `channels=32`、真实默认模型规格 `channels=24` 拒绝。
  测试实际写读配置 JSON、构造模型规格并调用完整 `verify_experiment_paths`，
  没有 mock 该函数。既有元数据负例中的 mock 保留，但不用于本轮新增回归。
- 测试只构造 CPU 随机模型以读取规格；没有参数更新、模型推理、checkpoint 文件
  或伪造训练成果，没有执行真实模型离线演练。
- 未改旧 source/install 证据、主动 src、共享状态或 Git；当前脚本仍需 root/verifier
  独立复审，实际演练须使用真实训练资产。

## 新鲜验证

先只增加测试运行
`.venv/bin/python -m pytest -q docs/research/offline-rehearsal/resume/test_model_rehearsal.py -k actual_experiment_model_config`：
原版本 **1 failed, 2 passed, 56 deselected**，失败为 channels=32 负例
`Failed: DID NOT RAISE ValueError`，证明缺口可复现。

再应用 9 行修复：

- `.venv/bin/python -m pytest -q docs/research/offline-rehearsal/resume/test_model_rehearsal.py`
  → **59 passed in 6.31s**。
- `.venv/bin/ruff check` 两个 Python 文件 → **All checks passed**。
- `.venv/bin/ruff format --check` 两个 Python 文件 → **2 files already formatted**。
- `.venv/bin/mypy --strict` 两个 Python 文件 → **Success: no issues found in 2 source files**。

历史 CPU 临时环境 `/tmp/aigc-cpu-verified-20261007/bin/python` 已不存在，初次调用
退出 127；上述验证全部使用当前 `.venv/bin/python`，不冒称为历史 CPU 环境结果。

## 精确字节

| 文件 | SHA-256 |
| --- | --- |
| `before-config-fix.py.txt`（写入后设置只读） | `752ef6a540c0de2f50e01ea13c361b3c2cd8745a197e5b634d7d90f776b3d3f0` |
| `model_rehearsal.py` | `292daa928f18de60a1d5271114308bd2bfab9d1b6050ff54a1162a11c0c7d2f7` |
| `test_model_rehearsal.py` | `2b06cb2b6db1ab7e40e1d1f0608993937b71c505355a8be5738da676550c94ac` |
