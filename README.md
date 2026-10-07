# AIGC 音视频伪造识别

第七届中国人工智能大赛 AIGC 音视频伪造识别项目。音频和视频两项均参加；以 EER、离线可执行性、可复现实验和跨域泛化为目标。

**当前阶段：按已确认方案实施。** 已建立 EER、数据治理、音视频解码、候选模型、训练、可恢复推理和离线打包代码，正在完成集成验证与真实数据准备。合成测试和随机模型 profile 仅证明工程行为；尚无真实鉴伪基线或本届成绩。

## 从这里开始

| 文档 | 内容 |
| --- | --- |
| [整体实施方案](docs/plans/implementation-plan.md) | 阶段、算法候选、验收、预算与停止规则 |
| [开发使用指南](docs/development.md) | 清单登记、配置模板、训练、评测、恢复和打包 |
| [比赛要求](docs/competition/requirements.md) | 官网事实、来源、待确认的安全屋契约 |
| [文献与参考项目](docs/research/literature.md) | 方法、局限、数据候选、旧项目取舍 |
| [工具与 skills](docs/research/tooling.md) | 已安装工具、固定版本、验证证据及延后依赖 |
| [项目状态](docs/status.md) | 本轮完成项、当前限制和下一步 |
| [开发约定](AGENTS.md) | 代理与开发者的目录、验证、数据和执行边界 |
| [历史归档](archive/README.md) | 朋友旧项目及备份恢复入口，默认只读 |

## 本地环境

Python 3.12，使用 uv 管理环境和锁文件。`media` 组提供数据与音视频处理，`research` 组提供 PDF、绘图和 notebook kernel；`cpu` / `cu130` 是互斥的 PyTorch/视觉/SSL 可选依赖。FFmpeg/ffprobe 使用宿主已有版本。

```bash
# 本会话默认缓存不可写；正常环境可省略此行。
export UV_CACHE_DIR=/tmp/aigc-uv-cache
# 当前 RTX 4060 开发环境；仅做开发检查可使用 uv sync --locked。
uv sync --locked --group media --group research --extra cu130
uv run --no-sync ruff check src tests tools/check_environment.py tools/profile_models.py tools/prepare_asvspoof2019.py tools/finalize_asvspoof2019.py
uv run --no-sync ruff format --check src tests tools/check_environment.py tools/profile_models.py tools/prepare_asvspoof2019.py tools/finalize_asvspoof2019.py
uv run --no-sync mypy
uv run --no-sync mypy tools/profile_models.py tools/prepare_asvspoof2019.py tools/finalize_asvspoof2019.py
uv run --no-sync pytest -q
uv build --no-sources

# 使用合成媒体和随机小模型验证安装；不下载权重或数据、不更新模型参数。
uv run --no-sync python tools/check_environment.py --device cuda --output outputs/environment/cuda.json
```

无 GPU 的环境将 `--extra cu130` 改为 `--extra cpu`，探针用 `--device cpu`。不要同时选择两个后端；日常命令保留 `--no-sync`，避免默认同步移除可选依赖。当前 App 沙箱会阻止 GPU 访问，CUDA 探针需在可访问设备的终端执行；这与 wheel 是否正确安装是两项检查。

CLI：`uv run --no-sync deepfake-detection-archive --help`。提供 `index`、`audit`、
`split`、`train`、`infer`、`evaluate`、`duplicates`、`export`、`package` 和
`verify-package`。实际使用步骤与配置模板见 [开发指南](docs/development.md)。

已有 GitHub Actions 配置执行相同的开发检查和旧评测器回归。本轮完成本地验证，尚未触发远端 CI。

## 项目 skills 与协作

项目筛选的 11 项第三方 skills 已复制到 `.agents/skills/` 并纳入 Git，
`skills-lock.json` 记录 `npx skills` 来源。协作者克隆仓库后即可使用，不依赖用户级
skills 目录。

```bash
npx --yes skills@1.7.1 list --json
# 更新会改动项目副本；运行后审阅 Git 差异、来源和许可。
npx --yes skills@1.7.1 update --project --yes
```

清单、更新流程和第三方许可见
[`.agents/README.md`](.agents/README.md)。Git 提交执行
[LoRA 准则](AGENTS.md)：Logical、Observable、Reversible、Atomic。

## 工作区

```text
src/deepfake_detection_archive/  新比赛包
tests/                        主动行为、集成与归档完整性回归
configs/                      配置与来源/清单模板；占位值须替换
docs/competition/             比赛约束与原始资料快照
docs/research/                文献、工具和参考版本
docs/plans/                   实施方案与整理计划
docs/worklog/                 工作归档
.agents/skills/               Git 跟踪的项目级 agent skills
skills-lock.json              npx skills 来源锁
tools/restore_backup.py       保留的历史恢复工具
archive/legacy-20261005/       旧源码、实验、文档、清单与原 README
```

`external/` 是已克隆的本地源码参考，`data/`、`models/`、`outputs/`、`cache/` 为按需生成的本地产物目录，均被忽略，不作为 Git 中的算法依赖。历史树不参与新包默认检查与构建；归档清单中的旧路径保留历史含义。

## 已知约束

- 已读取 RTX 4060 Laptop 8GB、驱动 616.92；App 普通沙箱阻止 GPU 访问。实际环境与模型 profile 见 [项目状态](docs/status.md)。本地 GPU 已获授权自动用于适合的实验且不设固定卡时上限；付费云算力预算、安全屋配置和报名/平台规范待定。
- 官方当前以音频、视频分别描述样本与 EER；最终榜单汇总、I/O、时限和资源仍须平台确认。[官方赛题](https://ai.xm.gov.cn/competition/project-detail.html?id=2026c40item0004&competeId=2026c40dbb2347fba8b3c9a6294efa5b)
- 音频不能默认等同语音，生成视频不能默认等同换脸；是否采用候选方法必须通过数据覆盖和跨域实验验证。
- 不将旧模型的历史报告或本轮工具测试解释为本届模型性能。
