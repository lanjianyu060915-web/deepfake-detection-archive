<!-- AUTONOMY DIRECTIVE — DO NOT REMOVE -->
YOU ARE AN AUTONOMOUS CODING AGENT. EXECUTE TASKS TO COMPLETION WITHOUT ASKING FOR PERMISSION.
DO NOT STOP TO ASK "SHOULD I PROCEED?" — PROCEED. DO NOT WAIT FOR CONFIRMATION ON OBVIOUS NEXT STEPS.
IF BLOCKED, TRY AN ALTERNATIVE APPROACH. ONLY ASK WHEN TRULY AMBIGUOUS OR DESTRUCTIVE.
USE CODEX NATIVE SUBAGENTS FOR INDEPENDENT PARALLEL SUBTASKS WHEN THAT IMPROVES THROUGHPUT. THIS IS COMPLEMENTARY TO OMX TEAM MODE.
<!-- END AUTONOMY DIRECTIVE -->


# 项目约定

本文件适用于整个仓库，补充继承的 OMX 工作流。说明、实验结论和文档默认中文，代码标识、命令与上游名称保留原文。

## 目标与当前授权

- 参加第七届中国人工智能大赛 **AIGC 音视频伪造识别赛**，用户已确认音频和视频都参加。主指标 EER 越低越好，最终交付须在安全屋离线运行并可复现。
- 当前用户授权：充分调研、安装合适 skills/工具、管理依赖与本地 Git、整理归档、编制整体方案。**后续算法开发、训练和完整交付执行，等待用户确认方案。** 不因自治指令越过这一明确阶段边界。
- 方案主文件为 `docs/plans/implementation-plan.md`，当前状态为待确认；每次开始工作先读 `docs/status.md` 和对应阶段。只有用户明确确认才更新实施状态。
- 用户允许复用或推翻旧照片鉴伪方案，不要求继承其模型、UI、身份检索、底图溯源或 Windows 打包接口。
- 用户当前可用 RTX 4060 8GB；云算力可考虑但预算待定。本会话 NVML 访问受阻是运行时可访问性问题，不代表没有该硬件或 CUDA 已验证。

## 事实与依据

- 官方赛题、规则、来源日期及待确认项集中在 `docs/competition/requirements.md`，原始资料在 `docs/competition/source/`；新的官方通知优先于快照。
- 当前公开任务描述音频和视频各约一万条、真实与生成各约一半、两者均用 EER；安全屋离线执行，每项目 6 天（2-1-2-1），至多 3 份结果并关联代码和参数。不要把数据量估计或总赛程当作精确执行契约。
- 官方 I/O、分数方向、时限、平台硬件、榜单汇总、外部数据/权重细则与报名后资源仍待确认；本地接口和选择是项目约定，不能写成赛方要求。
- 算法包目前只有问候入口，未实现 EER、模型或训练。运行依赖为空；开发和 PDF 研究依赖已由 `uv.lock` 固定。工具及模拟测试不证明模型性能。
- 文献与候选边界见 `docs/research/literature.md`。语音反欺骗不等于通用音频检测，图像帧检测不等于已验证的时序视频检测。

## 目录与历史边界

| 路径 | 约定 |
| --- | --- |
| `src/deepfake_detection_archive/` | 新比赛算法代码；保留现包名，除非任务要求更名 |
| `tests/` | 主动工程回归测试，默认 pytest 入口 |
| `docs/competition/` | 赛题要求与不改写的来源快照 |
| `docs/research/` | 文献、许可证边界、工具与参考版本 |
| `docs/plans/`、`docs/status.md`、`docs/worklog/` | 方案、当前状态与可追溯工作记录 |
| `archive/` | 历史归档；遵守子目录 AGENTS，原内容默认只读 |
| `tools/restore_backup.py` | 原字节保留的备份恢复工具，不是比赛入口 |
| `external/` | Git 忽略的固定版本上游参考，不进入包或默认检查 |
| `data/`、`models/`、`outputs/`、`cache/` | 本地数据/产物，先核对忽略规则再生成 |

- `archive/legacy-20261005/` 内的旧 README、计划、Windows 路径和“唯一产出区”只属于历史项目，不能影响本届工程布局与授权。
- 迁移冻结清单 `archive/integrity-manifest.json` 保护 1,526 个旧文件和恢复工具。不要为使测试通过而改写历史哈希；需要改动历史资产时先明确范围并保留原始证据。
- 复用旧实现须最小迁移、保留来源/许可并重新验证；不批量格式化、重排或自动修复归档。旧跨语料阈值失效记录应作为验证设计依据。

## 数据、算法与实验

- 方案获确认后按阶段推进：契约与数据治理、EER、双分支基线、跨域鲁棒性、可选较重方法、离线交付。先有可比较实验，再采用更大模型。
- 内部定义 `fake=1` 和分数方向，保留逐样本连续分数；官方转换独立实现。不把风险分带当概率，不只输出二值标签。
- EER 明确阈值、并列分数、插值、缺类与非有限值处理；无并列样例可与 ASVspoof 参考交叉核对，不能要求不同并列策略结果完全相同。官方程序可用后以其对齐验证。
- 数据登记来源、许可、哈希、原媒体/派生链、人物/说话人、生成器和处理条件。同源切片、视频帧、增强派生样本不跨划分。
- 开发集负责早停/融合选择，最终留出只用于冻结后的验收；不得把最终留出或比赛测试集反复用于调参。真实与生成样本使用对称处理，检查来源、编解码与分辨率捷径。
- 解码失败、无有效帧、音轨缺失、超时和模型异常须逐样本显式记账，不静默丢样本。提交失败策略依官方规范；EER 不可计算时报告原因，不能伪造数字。
- 记录代码版本、数据/权重/配置哈希、种子、环境、命令、耗时、吞吐、峰值显存和错误计数。历史结果、真实运行、模拟测试与假设分开表述。
- 推理从本地加载资源，禁用隐式下载；代码、权重、数据许可分别核对。没有官方依据时不融合视频音轨和独立音频任务。

## 环境与验证

- Python 3.12、uv、`uv.lock`。dev 组为 Ruff/pytest/mypy，research 组为 pypdf；新增依赖按任务实际需要引入并同步锁文件。不要把上游 requirements 整包灌入主环境。
- 当前 CUDA/PyTorch/模型依赖尚未安装，兼容矩阵、授权和 8GB 显存需实测。重型训练、完整数据下载和云资源启用遵守确认后的方案预算。
- 主动代码使用 88 字符行宽、Ruff 和严格 mypy。默认检查范围是 `src` 和 `tests`；历史恢复脚本未修改，不因新规则而重写。

```bash
export UV_CACHE_DIR=/tmp/aigc-uv-cache
uv sync --locked --group research
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy
uv run --no-sync pytest -q
uv run --no-sync python -B -m unittest discover -s archive/legacy-20261005/validation/deepfake_acceptance_v2/evaluator/tests -p 'test_*.py' -v
uv build --no-sources
```

- 2 项主动测试保护归档/恢复工具；7 项旧评测器测试使用 stub。新功能需要自己的行为测试，不能用上述结果替代 EER/模型测试。
- 构建后检查 wheel/sdist 不含归档、外部参考、数据、模型和缓存。CI 配置在 `.github/workflows/checks.yml`；本地通过不意味着远端已运行。
- skills 与源码参考记录在 `docs/research/tooling-lock.json`、`reference-projects.json`。研究阶段只做公开只读查询；技能示例中的上传、发布和云任务不因此获授权。

## 执行与交付

- 清晰授权的可逆本地步骤直接完成；保持小而可审查的提交。用户已授权自行管理 Git/依赖，可创建本地提交；不改写既有历史，不夹带无关用户改动，不自动发布、推送或参赛提交。
- 清理/重构先写计划并锁定现有行为，作者与评审分离。本轮整理计划是 `docs/plans/workspace-preparation.md`；后续方案调整要记录依据及影响。
- 有实质收益的独立任务可用原生子代理，显式选择 `explore`、`researcher`、`dependency-expert`、`executor`、`verifier`、`critic` 等角色；主代理整合并验证，普通任务不使用 `worker` 或暗中启动 Team。
- 显式 `$name` 工作流先读对应 SKILL；在 tmux 外不假定 Team/question 运行时界面可用，不伪造状态或权限。
- 结束时报告完成文件、验证证据、限制和当前等待点；本轮等待点来自用户明确要求的方案确认，不来自自动新增的审批流程。
