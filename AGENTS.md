<!-- AUTONOMY DIRECTIVE — DO NOT REMOVE -->
YOU ARE AN AUTONOMOUS CODING AGENT. EXECUTE TASKS TO COMPLETION WITHOUT ASKING FOR PERMISSION.
DO NOT STOP TO ASK "SHOULD I PROCEED?" — PROCEED. DO NOT WAIT FOR CONFIRMATION ON OBVIOUS NEXT STEPS.
IF BLOCKED, TRY AN ALTERNATIVE APPROACH. ONLY ASK WHEN TRULY AMBIGUOUS OR DESTRUCTIVE.
USE CODEX NATIVE SUBAGENTS FOR INDEPENDENT PARALLEL SUBTASKS WHEN THAT IMPROVES THROUGHPUT. THIS IS COMPLEMENTARY TO OMX TEAM MODE.
<!-- END AUTONOMY DIRECTIVE -->


# 项目约定

文档与实验结论默认中文，代码、命令和上游名称保留原文。本文件补充继承的 OMX 工作流。

## 目标与授权

- 同时参加第七届中国人工智能大赛 AIGC 音视频伪造识别音频、视频两项，主指标 EER，最终离线交付。
- 已授权自主管理本地 Git、依赖、公开数据、现有云端 GPU、Issue、任务分支、提交与推送；所需公开数据自行下载使用；不代替负责人合并、购买/扩容或正式参赛提交。
- 训练在当前 RTX4090 云端实例执行，原电脑不启动新训练。队友按 Issue 独立分支交付 PR。
- 用户已报名且确认官方无额外资料，不再索取；未知规范集中在 `docs/competition/requirements.md`，平台样例到来后适配。
- 项目入口为 `README.md`；事实看 `docs/status.md`，计划看 `docs/roadmap.md`，资产看 `docs/data.md`。

## 主线优先

1. 优先有效训练、EER与泛化、音视频候选和双分支离线系统。工程仅解决当前实验/交付问题，不用检查或文档数量代替算法进展。
2. 配置和外部输入在入口校验一次，内部传递已解析对象。缓存进入训练前验证，epoch内直接读取；不为假想异常增加分支、重复哈希或重解析。
3. 每轮实验一份配置、必要资产标识、逐样本分数、摘要和必要日志。复用现有工具，不复制脚本、建立证明树或新增平行框架。
4. 项目文档只保留使用、数据/许可、状态、路线图和实验结论；不提交个人交接Prompt、会话流水、暂停检查点、重复Issue正文及安装/验证日志。
5. 历史记录通过Git追溯，不作为当前推进门；无关文档、CLI或环境快照变化不应阻塞实验或导致特征重提取。
6. 小样本结果用于选择下一实验，明确局限后继续；正式统计用于充分证据下的结论，不用机械组数或预注册流程阻止探索。
7. 清理先明确范围和保留行为，覆盖缺失时补最小回归，修改后独立评审。先跑相关测试，提交前一轮标准检查，无新改动/失败不重复验证。

## 数据、模型与评测

- `real=0`、`fake=1`、高分更伪造，保留连续分数；官方格式集中在adapter。EER处理并列、插值、缺类和非有限值。
- 登记数据/权重来源、许可、版本和必要哈希。已知原媒体、派生、人物/说话人、确认重复组不跨划分，候选关系不当作确认事实。
- train训练，dev早停与选择，holdout在候选冻结后验收。真实/生成对称处理，检查当前数据明显的来源、尺寸与编解码捷径。
- 音频包括语音和非语音，视频面向通用生成内容；单一语音/逐帧结果不能证明整个任务有效。无赛题依据不跨项目融合音轨。
- 解码失败、无有效帧、超时和模型异常逐样本记账，不静默丢弃或编造分数；EER不可计算报告原因，失败计入覆盖率分母。
- 推理仅加载本地资源，禁用隐式下载；代码、数据、权重许可分别处理，未解决的资产不进入正式交付。
- 新方法先小profile确定batch/精度，再开展有价值实验；优先AMP、梯度累积与缓存，记录必要耗时、显存和失败。
- 恢复长任务先看实际进程和产物。当前不支持optimizer/RNG训练续跑，中断保留结果并新目录重启，推理resume与训练分开。

## 目录与环境

| 路径 | 用途 |
| --- | --- |
| `src/deepfake_detection_archive/`、`tests/` | 当前实现与行为回归；保持现包名 |
| `configs/`、`tools/` | 配置模板、数据/资源准备和实验辅助 |
| `docs/` | 项目使用、数据、状态、路线图、实验摘要、来源许可 |
| `.agents/skills/`、`skills-lock.json` | 固定第三方skills，不手改上游文件 |
| `archive/`、`tools/restore_backup.py` | 冻结旧项目与恢复工具，默认只读 |
| `data/ models/ outputs/ cache/ external/` | Git忽略的本地资产、运行产物与固定参考 |

- 归档清单保护1526个旧文件及恢复工具，不改写哈希、批量格式化或沿用旧UI/身份检索/Windows接口。
- Python3.12、uv和`uv.lock`；共享环境 `/opt/venvs/deepfake-cu132`。CPU/cu132互斥，CPU用独立环境，不擅改驱动或装上游整份requirements。无明确需要不新增依赖。
- 主动代码88字符、Ruff、严格mypy；维护者需要同步时执行 `scripts/devctl sync`，日常 `scripts/devctl run ...`，提交前 `scripts/devctl check`。
- 本地/CI检查范围只在 `tools/check_project.sh` 维护；CI为独立CPU环境和固定媒体工具链。归档字节检查保留，旧评测器stub不默认运行。

## 协作

- Git提交保持Logical、Observable、Reversible、Atomic，不改写历史或夹带他人工作。任务状态在Issue维护，不在仓库复制任务正文。
- 共享媒体/权重只读，outputs/cache按任务独立。RTX4090按实际显存、利用率和吞吐共享，空闲容量可供多人并发；Issue记录任务占用，不设个人独占。各任务固定工作副本，运行中不切换其源码或配置。
- 有实质收益时使用原生子代理并显式角色；主代理整合、独立评审、统一最终检查。不暗启Team或使用worker角色。
- 显式 `$name` 先读对应SKILL，不伪造OMX状态/权限；没有运行时支持使用普通工作流。
- 最终报告实际改动、必要验证和剩余主线任务，已授权步骤自主完成。

## Claude 与 DeepSeek 协作

负责人要求：Claude 会话工作时由 Claude 担任领导、DeepSeek 作为执行智能体配合完成任务。

- 分工：Claude 拆分任务、写明输入材料/红线/交回格式并下达指令；DeepSeek 完成分析、起草或代码建议；Claude 逐条复现核对、整合，并对最终结论和提交负责。
- 接口：`https://api.deepseek.com`（OpenAI 兼容 `/chat/completions`；Anthropic 兼容 `/anthropic`），模型 `deepseek-flash`（平台名 DeepSeek-V4.1-Flash），重任务可用 `deepseek-v4-pro`。模型先输出推理内容，`max_tokens` 留足。
- 密钥只从环境变量 `DEEPSEEK_API_KEY` 读取，不写入仓库、配置、日志或产物；云端环境需放行 `api.deepseek.com`。
- 指令附带所需材料原文（代码、配置、运行记录），不让 DeepSeek 自行联网取数；其输出视为待核数据：无证据或不能复现的结论不采信，不直接提交其代码。
- 只发送仓库与公开资产内容；凭据、个人信息和未登记许可的私有数据不外发。DeepSeek 不可用时说明原因，由 Claude 独立完成并在报告中注明。
