# AIGC 音视频伪造识别

第七届中国人工智能大赛 AIGC 音视频伪造识别项目。音频和视频两项均参加；以 EER、离线可执行性、可复现实验和跨域泛化为目标。

**当前阶段：调研与工程准备已完成，整体方案等待用户确认。** 新包仍是占位入口；尚未实现比赛模型、EER 评测或训练流水线，也没有本届成绩。

## 从这里开始

| 文档 | 内容 |
| --- | --- |
| [整体实施方案](docs/plans/implementation-plan.md) | 后续阶段、算法候选、验收、预算与停止规则；确认后按此推进 |
| [比赛要求](docs/competition/requirements.md) | 官网事实、来源、待确认的安全屋契约 |
| [文献与参考项目](docs/research/literature.md) | 方法、局限、数据候选、旧项目取舍 |
| [工具与 skills](docs/research/tooling.md) | 已安装工具、固定版本、验证证据及延后依赖 |
| [项目状态](docs/status.md) | 本轮完成项、当前限制和下一步 |
| [开发约定](AGENTS.md) | 代理与开发者的目录、验证、数据和执行边界 |
| [历史归档](archive/README.md) | 朋友旧项目及备份恢复入口，默认只读 |

## 本地环境

Python 3.12，使用 uv 管理环境和锁文件。当前先安装开发工具；研究组额外提供 PDF 解析。FFmpeg/ffprobe 使用宿主已有版本。

```bash
# 本会话默认缓存不可写；正常环境可省略此行。
export UV_CACHE_DIR=/tmp/aigc-uv-cache
uv sync --locked --group research
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy
uv run --no-sync pytest -q
uv build --no-sources
```

占位 CLI：`uv run --no-sync deepfake-detection-archive`。它只输出问候，不执行检测。

已有 GitHub Actions 配置执行相同的开发检查和旧评测器回归。本轮完成本地验证，尚未触发远端 CI。

## 工作区

```text
src/deepfake_detection_archive/  新比赛包
tests/                        归档完整性回归；功能测试随实施添加
docs/competition/             比赛约束与原始资料快照
docs/research/                文献、工具和参考版本
docs/plans/                   待确认方案与本轮整理计划
docs/worklog/                 工作归档
tools/restore_backup.py       保留的历史恢复工具
archive/legacy-20261005/       旧源码、实验、文档、清单与原 README
```

`external/` 是已克隆的本地源码参考，`data/`、`models/`、`outputs/`、`cache/` 为按需生成的本地产物目录，均被忽略，不作为 Git 中的算法依赖。历史树不参与新包默认检查与构建；归档清单中的旧路径保留历史含义。

## 已知约束

- 当前用户可用 RTX 4060 8GB；当前 App 进程的 GPU 访问尚未验证。云算力预算、安全屋配置和报名/平台规范待定。
- 官方当前以音频、视频分别描述样本与 EER；最终榜单汇总、I/O、时限和资源仍须平台确认。[官方赛题](https://ai.xm.gov.cn/competition/project-detail.html?id=2026c40item0004&competeId=2026c40dbb2347fba8b3c9a6294efa5b)
- 音频不能默认等同语音，生成视频不能默认等同换脸；是否采用候选方法必须通过数据覆盖和跨域实验验证。
- 不将旧模型的历史报告或本轮工具测试解释为本届模型性能。
