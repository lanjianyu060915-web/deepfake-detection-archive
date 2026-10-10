<!-- AUTONOMY DIRECTIVE — DO NOT REMOVE -->
YOU ARE AN AUTONOMOUS CODING AGENT. EXECUTE TASKS TO COMPLETION WITHOUT ASKING FOR PERMISSION.
DO NOT STOP TO ASK "SHOULD I PROCEED?" — PROCEED. DO NOT WAIT FOR CONFIRMATION ON OBVIOUS NEXT STEPS.
IF BLOCKED, TRY AN ALTERNATIVE APPROACH. ONLY ASK WHEN TRULY AMBIGUOUS OR DESTRUCTIVE.
USE CODEX NATIVE SUBAGENTS FOR INDEPENDENT PARALLEL SUBTASKS WHEN THAT IMPROVES THROUGHPUT. THIS IS COMPLEMENTARY TO OMX TEAM MODE.
<!-- END AUTONOMY DIRECTIVE -->


# 项目约定

本文件补充继承的 OMX 工作流。文档与实验结论默认中文，代码、命令和上游名称保留原文。

## 目标与授权

- 同时参加第七届中国人工智能大赛 AIGC 音视频伪造识别的音频、视频两项，主指标 EER 越低越好，最终离线交付。
- 用户已确认整体方案并授权完成项目，自主管理本地 Git、依赖、公开数据获取及本地 GPU；暂停后已恢复执行。历史暂停检查点不覆盖后来的恢复指令。
- 用户于2026-10-08确认已报名，官方没有额外信息，不再向其请求比赛资料。未知规范集中记在 `docs/competition/requirements.md`，继续本地开发，实际平台样例到来后适配。
- 用户于2026-10-10将后续开发/训练转到云端，本地仅收尾，不再启动新训练。负责人发布Issue，队友在云端独立分支工作并交付PR，负责人审查合并。云端已分配实例按任务使用；购买/扩容及正式参赛提交仍由负责人决定。
- 云端接手入口是 `docs/handoff/personal-cloud-codex.md`；环境与资产见同目录协作说明和清单。Git不包含数据/权重/大产物，不假定已上传或具备账号权限。当前本地收尾不自动推送或发布Issue。
- 当前事实看 `docs/status.md`，算法路线看 `docs/plans/implementation-plan.md`；历史研究和工程测试不代替本届比赛成绩。

## 主线优先

这是用户明确要求的执行修正，适用于后续所有开发：

1. 优先数据可用、有效训练、EER与泛化、双分支离线交付。工程改动解决眼前实验或交付问题，不用反复证明和扩建基础设施代替算法进展。
2. 配置和外部输入在入口校验，已解析对象直接传递。内部不重复解析、重复计算、无故全量重验，也不为假想异常堆叠分支。
3. 每轮实验保存一份配置、必要资产标识、逐样本分数、简洁摘要和必要日志。复用已有工具，不复制整套实现或为同一事实建立多份证明。
4. 历史快照只用于追溯，不作为当前工具或推进门。无关源码、文档或历史安装报告的变化不应阻塞模型实验或导致特征缓存重建。
5. 小样本实验用于选择下一步，说明局限即可继续。置信区间与严格比较用于充分证据下的结论，不以机械组数或预注册流程阻止探索训练。
6. 先运行与改动对应的行为检查，再在提交前完成一轮标准检查；无新改动或失败时不重复验证。测试验证行为，不堆叠实现细节守卫。
7. 清理先写范围和需保留行为，修改后独立评审。本轮范围在 `docs/plans/workspace-preparation.md`，不新增平行框架和无关依赖。

## 数据、模型和评测

- 内部 `real=0`、`fake=1`，高分更伪造，保留连续分数；官方转换集中在适配层。EER处理并列、插值、缺类和非有限值，官方程序出现后核对差异。
- 登记数据/权重来源、许可、版本和必要哈希。已知原媒体、派生、人物/说话人和确认的近重复组不跨划分；候选关系不直接当作重复事实。
- train训练，dev早停与选择，holdout在候选冻结后验收。真实/生成采用对称处理，检查来源、尺寸、编解码捷径。
- 音频覆盖语音及非语音，视频覆盖通用生成内容；语音或图像帧实验不能证明整个音视频任务有效。没有赛题依据时不跨两项目融合音轨。
- 解码失败、无有效帧、超时和模型异常逐样本记账，不静默丢弃或编造分数。EER无法计算时报告原因，失败计入覆盖率分母。
- 推理只加载本地资源，禁用隐式下载。代码、数据和权重许可分别核对；未解决的资产不进入正式交付。
- 先小规模profile确定batch/精度/workers，再开展有价值的实验；优先AMP、梯度累积与可复用缓存。记录耗时、吞吐、显存和错误，不将单次满载解释为实验充分。
- 长任务恢复先检查实际进程和产物，不依据失效工具handle推断终态。当前训练不支持optimizer/RNG恢复，中断结果保留并在新目录重启；推理resume与训练续跑分开。

## 文件与历史

| 路径 | 用途 |
| --- | --- |
| `src/deepfake_detection_archive/`、`tests/` | 当前代码与行为回归；保持现包名 |
| `configs/`、`tools/` | 当前配置模板、导入/探针和统一检查入口 |
| `docs/status.md`、`docs/plans/`、`docs/development.md` | 当前状态、主线方案、使用指南 |
| `docs/competition/`、`docs/research/`、`docs/experiments/`、`docs/worklog/` | 赛题来源、研究、实验和必要工作记录 |
| `.agents/skills/`、`skills-lock.json` | Git跟踪的项目skills及来源锁 |
| `archive/`、`tools/restore_backup.py` | 旧项目与原字节恢复工具，默认只读 |
| `data/`、`models/`、`outputs/`、`cache/`、`external/` | Git忽略的本地资产和固定上游参考 |

- 旧照片项目不要求复用UI、身份检索、底图溯源、Windows接口或历史阈值。归档内旧路径与约定仅适用于历史项目。
- `archive/integrity-manifest.json`保护1,526个旧文件与恢复工具，不为通过检查改写哈希。复用实现时最小迁移、保留来源并重新验证，不批量格式化归档。
- `.gitattributes`保留归档、恢复工具、来源快照、vendor及第三方skills字节；构建排除历史、外部参考、数据、模型和缓存。

## 环境与验证

- Python3.12、uv和`uv.lock`；dev/research/media按需同步，CPU与cu130 extras互斥，当前主环境用cu130。新增依赖同步锁文件，不安装整份上游requirements或擅改系统驱动。
- 普通沙箱GPU/NVML或IPC受限与安装失败分开判断；实际运行证据看状态与实验记录。
- 主动代码使用88字符行宽、Ruff和严格mypy。本地和CI共享以下入口，具体检查范围只在脚本维护：

```bash
export UV_CACHE_DIR=/tmp/aigc-uv-cache
uv sync --locked --group media --group research --extra cu130
bash tools/check_project.sh
```

- 默认测试保护主动行为及归档；旧评测器7项stub不代替模型效果。远端CI状态按实际运行报告。
- 11项项目skills由固定`npx skills@1.7.1 --copy`管理并纳入Git，协作者克隆即用。更新审阅上游差异及许可，适配写在本仓库，不手改第三方skill。上传、发布、遥测和云示例不代表授权。

## Git与协作

- 本地提交遵循LoRA：Logical（一目标）、Observable（必要验证证据）、Reversible（独立回退）、Atomic（最小完整改动）。不改写历史、不夹带无关用户改动。
- 队友按已分派Issue开发任务分支和PR，不代替负责人合并。共享原媒体/权重只读，任务outputs/cache独立。
- 有实质收益时可用原生子代理并显式指定角色；主代理整合、独立评审、统一最终验证，避免各代理重复运行同一套检查。普通工作不暗启Team或使用worker角色。
- 显式`$name`先读对应SKILL；不伪造OMX状态、运行界面或权限。没有运行时支持时使用普通工作流。
- 结束报告实际改动、验证和剩余主线任务；已授权步骤自主完成，不重复等待方案确认。
