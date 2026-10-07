# 暂停与恢复记录

用户明确暂停后停止工作。当前没有本 agent 存活的后台任务/GPU任务；
finalizer session 94650 已 exit 2，未发布 `data/asvspoof2019-la`。

完整归档已由 root 恢复并通过官方大小/MD5：7,640,952,520 bytes，
MD5 `30c98f11d8b2bc21f2c257bfd78bb5c5`，SHA-256
`208a7e4e3913f8c75ae1afd19bf32a5b29ae68435e9e30e23e5e98b6a155e4ec`。
完成时间 `2026-10-07T07:22:43.235315+00:00`，证据在
`outputs/data-acquisition/asvspoof2019-tail-transfer.json`；原始 ZIP 仍在
`cache/downloads/asvspoof2019-LA.zip.part`。不要重新下载。

首次 finalizer 发现真实官方 ZIP 多含 142 个协议外 `LA_D_A<7digits>.flac`。
CM train/dev 协议引用的 25,380/24,844 条媒体均存在，协议外文件只读取目录
元数据，未读取 body；eval 内容没有打开。详情与 CRC/大小见
`archive-membership-audit.json`，首次失败见 `finalizer-first-attempt.json`。

**修复尚未完成，不要直接继续提取或训练。** 暂停收尾已将工具和测试恢复到已提交版本，
未验证改动的原字节及补丁保存在 `docs/checkpoints/2026-10-07-pause/asv-finalizer-draft/`。
草稿针对 `tools/finalize_asvspoof2019.py` 和 `tests/test_asvspoof_acquisition.py` 增加
bounded CM 协议 ID/标签/攻击/计数校验、协议白名单匹配、142 个明确模式的
协议外文件登记排除及回归用例。最后一轮 Ruff 在测试之前停止，存在：

- 两个 PT018：新测试 guarded_open/protocols_only 中组合 assert 拆开。
- `_selected_members` C901/PLR0912：建议将协议白名单核验与排除记录提取为
  小的明确 helper，保持现有路径/预算/原子发布保护。

新修复尚未跑格式、mypy 或 pytest；不能称验证通过。用户恢复后，先按统一恢复入口
核对基线并应用保存的补丁，再完成此最小修复
和针对性回归，再给 root 独立复核，记录新的工具/代码身份，不能改写旧冻结哈希。
root 复核后再次执行 finalizer，接着官方导入器（每 split 每类256）、
`source/prepare_pilot.py --project-root <repo> --downloaded-at 2026-10-07T07:22:43.235315+00:00`，
审查跨 split fingerprint 候选，确认真实审计通过后协调单 GPU 再启动四 run。

四配置与两个独立 orchestration 脚本在归档异常发现之前已通过 Ruff/格式、严格
mypy、help/config/bash 检查，且预评分冻结完成；这些证据不覆盖本次未完成的
finalizer 修复。当前没有媒体导入、真实解码或训练结果，没有 checkpoint 可交付
offline_environment。所有改动尚未由本 agent 提交 Git。
