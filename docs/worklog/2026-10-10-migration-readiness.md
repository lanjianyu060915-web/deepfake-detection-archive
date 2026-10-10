# 迁移后接手与开发准备

日期：2026-10-10。范围和保留行为见[工作区计划](../plans/workspace-preparation.md)。
当前结论：服务器代码开发可进行，真实实验仍需物化/导入共享数据并接入固定清单与权重。
环境准备未启动训练、修改锁文件或升级驱动；后续按用户授权发布Issue，并整理迁移分支提交和推送。

## 环境与工具

复用已安装的Python3.12.15、uv0.13.0及`/opt/venvs/deepfake-cu132`，开发、研究和媒体
依赖齐全；124个环境依赖的锁定同步预检无变化，11项项目skills保持固定上游原字节。
当前RTX4090 24GiB、驱动595.91.07和CUDA13.2可执行CUDA、torchvision NMS及AMP合成检查。

系统FFmpeg4.4.2不支持项目的`-fps_mode`，首轮回归产生2项失败、2项错误，不能将工具存在
视为环境完整。接续用户已经下载的BtbN包，核对完整SHA-256后安装。按用户后续要求改为
root管理的全局目录`/opt/ffmpeg/btbn-628033703`，通过`/usr/local/bin/ffmpeg`及`ffprobe`
供所有账号调用；核对全路径及ACL的读取/执行权限，并清除冗余用户级入口和副本。

- 实际版本：`N-127271-gba987fe24d-20261010`，GPLv3+；许可与安装记录保留在安装目录。
- [发布资产API](https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/latest)：
  release408966939、asset628033703，156,214,712字节。
- [发布SHA-256清单](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/checksums.sha256)：
  `df29c2c5114aa4adeb61dbf1011778b85ecc33710899b09b99c98fb000bb81a7`。

恢复固定提交`9b33f5eca887bd3bf629e3fb79428cd9bb7ac972`的ASVspoof2021只读评分参考，
消除唯一参考测试跳过；不安装上游requirements。当前`.envrc`已审阅并为本用户授权。

## 代码与当前说明

- `scripts/devctl`和`tools/check_project.sh`优先各自绝对源码路径，修复worktree导入另一个
  checkout的问题；同步显式`--no-install-project`，支持独立环境，标准检查只有一份定义。
- 新增4项隔离入口回归，修复前4项失败、修复后通过；使用临时假uv，不修改共享依赖。
- 环境探针删除OpenCV抽帧副本，复用项目真实视频解码，并登记FFmpeg/ffprobe路径与版本。
  新版GPU探针通过；强制使用旧系统FFmpeg会明确失败，未添加静默降级。
- README、开发指南、交接、配置说明及实施方案统一cu132和模块入口，删除重复环境段与旧机
  当前状态说法；原实验、归档、来源快照、资产包哈希和第三方skills保留。

## 共享数据

数据已在`/srv/shared-data/datasets`，无需重新下载；完整位置及边界统一见
[云端协作](../handoff/cloud-collaboration.md#3-git-外资产)和
[资产清单](../handoff/local-assets.json)。两个ASV Parquet及ViF ZIP的完整哈希匹配既有来源
登记，FakeSound仅检查了ZIP目录。目录2755、文件0644、root:dlteam，全路径和ACL允许
队友读取；原媒体由维护者写入，实验outputs/cache各工作副本独立。
本worktree已建立`data/shared-raw`链接，未覆盖原数据、解压大包或启动重复下载。

ASV Parquet train25,380/dev24,844行尚未物化和核对官方protocol，ViF尚未导入，FakeSound
仅fake诊断集。旧实验的7个资产路径仍未接入，不把原始归档存在写成可直接复现实验。

## 验证与剩余事项

独立评审无代码阻塞项，发现的旧机8GB说明已修正。最终`scripts/devctl check`通过：
404项pytest（23.35秒，无跳过）、3项视频工具测试、7项旧评测器stub、Ruff、59文件格式、
严格mypy（32+7文件）、Shell语法及sdist/wheel构建。uv0.13.0与固定uv_build0.12系列的
版本提示不影响本次构建，未因此升级依赖。15份当前文档、76个本地链接、JSON和23个Shell示例块校验通过。
本机检查日志、GPU探针与资产哈希结果在`outputs/environment/`；最终检查日志为
`project-check-final.log`，远端CI按实际运行报告。

已通过`gh`发布音频#9、视频#10和数据#11，分别完成本方向资产接入和实验，入口见
[任务目录](../handoff/issues/README.md)。已完成的环境Issue草稿删除，当前说明统一指向
迁移修复分支；代码独立评审未发现新增静默回退或多余防御分支。
训练恢复和最终双分支离线验收仍未完成。
