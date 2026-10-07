# 历史项目归档

`legacy-20261005/` 保存此前朋友项目的源码、图像/视频实验、验收材料、备份元数据与原根 README。它是历史参考，**不是本届比赛的开发入口或执行规范**。

- 迁移前内容来自 `f8cbd4d` 所在的历史仓库；逐文件清单见 [integrity-manifest.json](integrity-manifest.json)。共 1,526 个文件、138,273,682 字节，迁移时逐一核对 SHA-256。
- 内部原目录布局和字节均保留。历史 JSON/CSV 中的 `systems/...` 等路径以归档前的仓库根目录为基准，现在对应 `archive/legacy-20261005/systems/...`。清单中的外部 Windows 绝对路径仅记录来源。
- 根 `.gitattributes` 禁用归档的文本换行转换，防止 `core.autocrlf=input` 在暂存或新检出时改变历史 CRLF 字节；根产物忽略规则限定目录范围，不隐藏旧模型元数据。
- [原归档说明](legacy-20261005/README.md) 保留历史正文；其中命令与目录也是历史上下文，不能把它们当作本届工程要求。
- 当前可用的恢复工具仍在仓库根 [tools/restore_backup.py](../tools/restore_backup.py)，字节未改。使用时从当前仓库根调用，传入实际下载目录和清单路径；Release 附件及 SHA-256 对照见 [发布清单](legacy-20261005/inventory/release-assets.json)。
- 不对历史树批量格式化、自动修复或更新模型阈值。可将有价值的设计迁入新工程，保留出处并重新验证。多数旧脚本包含 Windows 环境假设，迁移没有解决其依赖与可运行性问题。

旧评测器的标准库模拟测试可在当前根目录运行：

```bash
uv run python -B -m unittest discover -s archive/legacy-20261005/validation/deepfake_acceptance_v2/evaluator/tests -p 'test_*.py' -v
```

新项目测试中的归档完整性检查验证历史内容未丢失；模拟测试仅证明评测器行为，不能证明旧模型或本届算法准确率。
