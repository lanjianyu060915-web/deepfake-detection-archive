# 深度伪造检测系统：源码与备份

本私有仓库存档本机已整理的深度伪造检测系统、开发代码、文档和完整备份。整理日期：2026-10-05（Asia/Shanghai）。原始目录保持只读；此仓库是独立副本。

## 源码位置

| 仓库路径 | 原始位置 | 内容 |
| --- | --- | --- |
| `systems/original-deepfake` | `C:\Users\lanji\Desktop\dist\DeepfakeDetectionSystem` | 原版 Streamlit 界面、26 个 Python 模块、配置和模型元数据。含训练函数，但缺少完整训练工程配套。 |
| `systems/v0.4-baseline` | `D:\刑侦查竞赛\face_source_trace_three_module_v0.4.0_visualization_dev` | v0.4.0 集成源码、前端、测试、文档、构建脚本和鉴伪运行代码。 |
| `systems/upgrade-20260921` | `D:\刑科技项目升级2.0\workspace\fsts` | 二代 ViT 检测、旧模型回退、视频入口及集成代码。 |
| `systems/installed-v0.3.1` | `D:\刑侦查竞赛\4` | v0.3.1 安装态的可读代码及配置快照。 |
| `systems/installed-v0.3.0` | `D:\刑侦查竞赛\换脸寻源3.0` | 实际版本 v0.3.0 的安装态代码与配置。 |
| `validation/deepfake_acceptance_v2` | `D:\刑侦查竞赛\deepfake_acceptance_v2` | 独立验收工具、规范、报告及冻结清单。现有报告尚未完成独立盲测。 |
| `documentation/original` | `D:\刑侦查竞赛\AIGC检测` | 原版使用手册与技术报告。 |
| `upgrade-research` | `D:\刑科技项目升级2.0` | 升级设计、实验脚本与文档。 |

自有源码与文档提交到 Git；完整运行环境、模型、数据、安装器和原始归档保存在本仓库的 Release 附件。源对象、附件和 SHA-256 的对应关系见 `inventory/release-assets.json`。

## 完整备份与恢复

Release `backup-2026-10-05` 保存以下内容：

- v0.4.0 完整冻结 ZIP 的 22 个二进制分卷，每卷最多 256 MiB。分卷不能单独解压，需先拼接并校验。
- 升级工作区的增量归档：以 v0.4.0 完整冻结包为基线，保存所有新增或变更文件，并记录删除清单。包含二代模型及其完整 Python 运行时。
- 桌面原版 portable 与两份历史构建的完整归档。
- 原版交付备份、v0.3.0 安装目录归档、v0.3.1 正式安装器、9 月 21 日升级交付 ZIP。

恢复冻结包：下载全部 `baseline-v0.4.0-full.zip.256m.partNNN` 与 `baseline-split-manifest.json`，运行：

```powershell
python tools/restore_backup.py join --manifest baseline-split-manifest.json --parts-dir 下载目录 --output baseline-v0.4.0-full.zip
```

原版运行目录、v0.3.0 运行目录和 v0.3.1 安装器也采用分卷。分别下载对应分卷与 `original-dist-split-manifest.json`、`installed-v0.3.0-split-manifest.json`、`installer-v0.3.1-split-manifest.json`，以相同 `join` 命令拼接，`--output` 使用该清单的 `original_name`。

恢复升级工作区：先将完整冻结 ZIP 解压，得到 `face_source_trace_three_module_v0.4.0_visualization_dev` 目录，再将增量叠加到该目录：

```powershell
python tools/restore_backup.py overlay --root 解压后的目录 --delta upgrade-workspace-delta-20261005.zip
```

恢复工具会校验分卷、完整 ZIP 及增量文件 SHA-256，并检查路径不越出指定目录。叠加增量时会替换明确列出的变更文件；若存在删除清单，需要显式传入 `--apply-deletions`。

## 版本与运行情况

原版核心为 MobileNetV3-Small 空间分支和 87 维频域统计融合。v0.4.0 保留该模型并加入身份检索、底图溯源及真实可视化；二代升级工程增加 CommunityForensics ViT ONNX、按输入尺寸路由及视频处理。

升级配置仍标识 `product_version=0.4.0`，实际内核切换记录在 `upgrade` 字段；“刑科技项目升级2.0”是工作区名称。已有功能验收记录不等于完整的跨域准确率验证，视频跨语料阈值仍有已记录的失效情况。

9 月 21 日的便携 ZIP 与升级输出 ZIP 逐字节相同，但原包存在资源路径与运行时不完整问题，不能认定解压即用。完整基线与增量备份可以恢复开发工作区的文件，运行依赖仍应按原项目说明检查；本次归档未修改算法或启动系统复测。

原项目与第三方库的版权、许可证归属保留。未替上游完整训练仓库、训练数据和可复现训练流程作完整性承诺。
