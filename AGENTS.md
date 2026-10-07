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
- 用户已于 2026-10-07 明确授权“开始执行，完成整个比赛项目”，整体实施方案已确认；可按阶段自主进行契约、数据治理、算法开发、训练、验证和离线交付准备，按需安装相关依赖并管理本地 Git。
- 用户随后明确要求暂停；“继续收尾”仅授权保存、验证和本地提交，不恢复下载、特征提取或训练。当前检查点见 `docs/checkpoints/2026-10-07-pause/README.md`，待用户明确恢复后继续。
- 方案主文件为 `docs/plans/implementation-plan.md`，当前状态为暂停；每次开始工作先读 `docs/status.md` 和对应阶段。官方适配、真实数据覆盖及提交验收按实际证据判断，不把本地模拟验证等同正式完成。
- 用户允许复用或推翻旧照片鉴伪方案，不要求继承其模型、UI、身份检索、底图溯源或 Windows 打包接口。
- 已在沙箱外读取 RTX 4060 Laptop 8GB、驱动 616.92；普通沙箱 NVML 访问受阻。GPU 探针的实际结果查阅工具记录，不能混淆沙箱限制和 CUDA 安装结果。用户已持续授权自由使用本地 GPU；适合 GPU 的训练、推理、特征提取和 profile 默认自动使用 CUDA，不设人为卡时上限。付费云算力预算仍待定，不包含在该授权中。

## 事实与依据

- 官方赛题、规则、来源日期及待确认项集中在 `docs/competition/requirements.md`，原始资料在 `docs/competition/source/`；新的官方通知优先于快照。
- 当前公开任务描述音频和视频各约一万条、真实与生成各约一半、两者均用 EER；安全屋离线执行，每项目 6 天（2-1-2-1），至多 3 份结果并关联代码和参数。不要把数据量估计或总赛程当作精确执行契约。
- 官方 I/O、分数方向、时限、平台硬件、榜单汇总、外部数据/权重细则与报名后资源仍待确认；本地接口和选择是项目约定，不能写成赛方要求。
- 算法实现进度以 `docs/status.md`、源码和新鲜验证为准；开发、研究、音视频组和 CPU/CUDA 可选后端由 `uv.lock` 固定。工具及模拟测试不证明模型性能。
- 文献与候选边界见 `docs/research/literature.md`。语音反欺骗不等于通用音频检测，图像帧检测不等于已验证的时序视频检测。

## 目录与历史边界

| 路径 | 约定 |
| --- | --- |
| `src/deepfake_detection_archive/` | 新比赛算法代码；保留现包名，除非任务要求更名 |
| `tests/` | 主动工程回归测试，默认 pytest 入口 |
| `docs/competition/` | 赛题要求与不改写的来源快照 |
| `docs/research/` | 文献、许可证边界、工具与参考版本 |
| `docs/plans/`、`docs/status.md`、`docs/worklog/` | 方案、当前状态与可追溯工作记录 |
| `.agents/skills/`、`skills-lock.json` | 由固定版 `npx skills` 管理并纳入 Git 的项目 skills；协作者克隆即用 |
| `archive/` | 历史归档；遵守子目录 AGENTS，原内容默认只读 |
| `tools/restore_backup.py` | 原字节保留的备份恢复工具，不是比赛入口 |
| `external/` | Git 忽略的固定版本上游参考，不进入包或默认检查 |
| `data/`、`models/`、`outputs/`、`cache/` | 本地数据/产物，先核对忽略规则再生成 |

- `archive/legacy-20261005/` 内的旧 README、计划、Windows 路径和“唯一产出区”只属于历史项目，不能影响本届工程布局与授权。
- 迁移冻结清单 `archive/integrity-manifest.json` 保护 1,526 个旧文件和恢复工具。不要为使测试通过而改写历史哈希；需要改动历史资产时先明确范围并保留原始证据。
- 复用旧实现须最小迁移、保留来源/许可并重新验证；不批量格式化、重排或自动修复归档。旧跨语料阈值失效记录应作为验证设计依据。

## 数据、算法与实验

- 按已确认方案分阶段推进：契约与数据治理、EER、双分支基线、跨域鲁棒性、可选较重方法、离线交付。先有可比较实验，再采用更大模型。
- 内部定义 `fake=1` 和分数方向，保留逐样本连续分数；官方转换独立实现。不把风险分带当概率，不只输出二值标签。
- EER 明确阈值、并列分数、插值、缺类与非有限值处理；无并列样例可与 ASVspoof 参考交叉核对，不能要求不同并列策略结果完全相同。官方程序可用后以其对齐验证。
- 数据登记来源、许可、哈希、原媒体/派生链、人物/说话人、生成器和处理条件。同源切片、视频帧、增强派生样本不跨划分。
- 开发集负责早停/融合选择，最终留出只用于冻结后的验收；不得把最终留出或比赛测试集反复用于调参。真实与生成样本使用对称处理，检查来源、编解码与分辨率捷径。
- 解码失败、无有效帧、音轨缺失、超时和模型异常须逐样本显式记账，不静默丢样本。提交失败策略依官方规范；EER 不可计算时报告原因，不能伪造数字。
- 记录代码版本、数据/权重/配置哈希、种子、环境、命令、耗时、吞吐、峰值显存和错误计数。历史结果、真实运行、模拟测试与假设分开表述。
- 充分使用本地 GPU：先用 profile 确定 batch、精度、workers、缓存和累积步数，再以 AMP、梯度累积、异步传输等实测有效手段提高利用率。训练可显式选择 float16/bfloat16 AMP，设备不支持时明确失败；长任务记录温度、显存、吞吐和失败，以实验价值、资源故障和收益递减作为停止条件，不以预设卡时截断有效实验。
- 推理从本地加载资源，禁用隐式下载；代码、权重、数据许可分别核对。没有官方依据时不融合视频音轨和独立音频任务。

## 环境与验证

- Python 3.12、uv、`uv.lock`。dev 组为 Ruff/pytest/mypy，research 组为 pypdf/绘图/kernel，media 组为数据/音视频工具；`cpu` / `cu130` extras 包含配对的 torch/torchvision、timm、Transformers。两个后端互斥，当前开发环境用 cu130。新增依赖按任务实际需要引入并同步锁文件，不灌入上游整份 requirements。
- PyTorch 使用官方显式索引，wheel 的 CUDA 依赖不等于另行修改系统驱动。随机模型和合成媒体探针在 `tools/check_environment.py`；无权重下载或参数更新。具体模型的兼容性、权重许可和 8GB 显存适配仍需后续实测。重型训练和完整数据下载遵守方案确认边界；本地 GPU 已获持续授权，云资源启用仍须确认预算与数据合规。
- 主动代码使用 88 字符行宽、Ruff 和严格 mypy。默认检查范围是 `src` 和 `tests`；历史恢复脚本未修改，不因新规则而重写。

```bash
export UV_CACHE_DIR=/tmp/aigc-uv-cache
uv sync --locked --group media --group research --extra cu130
uv run --no-sync ruff check src tests tools/check_environment.py tools/profile_models.py tools/prepare_asvspoof2019.py tools/finalize_asvspoof2019.py tools/prepare_vif_panda.py
uv run --no-sync ruff format --check src tests tools/check_environment.py tools/profile_models.py tools/prepare_asvspoof2019.py tools/finalize_asvspoof2019.py tools/prepare_vif_panda.py
uv run --no-sync mypy
uv run --no-sync mypy tools/profile_models.py tools/prepare_asvspoof2019.py tools/finalize_asvspoof2019.py tools/prepare_vif_panda.py
uv run --no-sync pytest -q
uv run --no-sync python -B -m unittest discover -s archive/legacy-20261005/validation/deepfake_acceptance_v2/evaluator/tests -p 'test_*.py' -v
uv build --no-sources
```

- 2 项主动测试保护归档/恢复工具；7 项旧评测器测试使用 stub。新功能需要自己的行为测试，不能用上述结果替代 EER/模型测试。
- 模型/GPU profile 工具及官方语音导入器同时纳入 Ruff、格式和严格 mypy；完整命令见 CI。`.gitattributes` 保留归档、恢复工具、项目 skills 和来源/vendor 的原字节，不格式化这些资产。
- 构建后检查 wheel/sdist 不含归档、外部参考、数据、模型和缓存。CI 配置在 `.github/workflows/checks.yml`；本地通过不意味着远端已运行。
- 项目实际使用的 11 项 skills 由 `npx skills@1.7.1 --copy` 安装在 `.agents/skills/`，以 `skills-lock.json` 固定来源并纳入 Git；它们优先于机器级安装，协作者克隆后可直接调用。更新时使用固定 CLI、审阅上游差异和许可，再提交复制结果；不直接手改第三方 skill，项目适配写入本仓库代码或文档。全部 535 项 ds-skills 的机器级比对只保留为历史审计。来源与适配边界见 `docs/research/tooling-lock.json`、`docs/research/ds-skills.json`。技能示例中的上传、发布、遥测和云任务不因此获授权。

## 执行与交付

- 清晰授权的可逆本地步骤直接完成；保持小而可审查的提交。用户已授权自行管理 Git/依赖，可创建本地提交；不改写既有历史，不夹带无关用户改动，不自动发布、推送或参赛提交。
- Git 提交遵循本项目 LoRA 准则：**Logical**（一个明确目标）、**Observable**（提交说明或工作记录给出验证证据）、**Reversible**（能独立回退且不破坏无关工作）、**Atomic**（只含完成该目标所需的最小完整改动）。依赖变更、源码、生成记录和文档按依赖顺序拆分；每个提交执行与范围匹配的检查，提交信息准确描述结果。若用户后续给出不同的 LoRA 定义，以最新明确说明为准。
- 清理/重构先写计划并锁定现有行为，作者与评审分离。本轮整理计划是 `docs/plans/workspace-preparation.md`；后续方案调整要记录依据及影响。
- 有实质收益的独立任务可用原生子代理，显式选择 `explore`、`researcher`、`dependency-expert`、`executor`、`verifier`、`critic` 等角色；主代理整合并验证，普通任务不使用 `worker` 或暗中启动 Team。
- 显式 `$name` 工作流先读对应 SKILL；在 tmux 外不假定 Team/question 运行时界面可用，不伪造状态或权限。
- 结束时报告完成文件、验证证据、限制和剩余工作；用户已授权整体实施，不重复等待方案确认。官方凭据、付费云预算或正式提交仍依据实际授权处理。
