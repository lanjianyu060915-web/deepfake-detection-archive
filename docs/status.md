# 当前项目状态

更新日期：2026-10-07。阶段：**按已确认方案实施，工程集成与真实数据准备中**。

## 当前事实

- 用户确认音频和视频均参加；本地有 RTX 4060 8GB，云算力后续考虑；报名、安全屋权限与执行规范待定。
- 已核对官网赛题/规则、调研相关论文和项目，按 [整体方案](plans/implementation-plan.md) 实施。算法路线是待验证候选，没有真实数据成绩。
- 通过 `npx skills@1.7.1 --copy` 将 11 项相关 skills 安装到项目
  `.agents/skills/`，由 Git 和 `skills-lock.json` 管理，协作者克隆即用。其中
  ds-skills 选择 7 项，其中 6 项完成合成验证、1 项作为音频时频图结构参考；机器级
  535 项一致性比对保留为审计记录。已准备 3 份模型源码参考及 ds-skills 固定源码，记录见
  [工具说明](research/tooling.md)。
- 用户授权后续相关依赖按需自行安装；项目环境已锁定并安装 `dev` / `research` / `media` 和 `cu130` 后端。torch 2.14.1+cu130、torchvision 0.29.1+cu130、timm、Transformers、音视频处理与研究工具已就绪；独立临时环境另安装 CPU wheel 并完成全套测试，未替换项目 CUDA 环境。
- 旧项目已整体迁至 `archive/legacy-20261005/`，1,526 个文件、138,273,682 字节与冻结清单一致；比赛原始资料改放 `docs/competition/source/`。
- 新增严格数据/来源登记、哈希与许可审计、关系连通组划分、近重复候选、实际音视频解码和失败记账。连通组与统计组分开，未知组不冒充独立样本。
- 新增 EER、DET、AUC、固定阈值诊断、分桶报告及配对组 bootstrap；完整分母保留失败样本，完整集 EER 无法计算时为空，成功子集指标另列。
- 新增 AASIST-L、log-mel CNN、冻结本地 SSL/视觉表征及媒体级头、AMP 训练、按实际样本数归一化的梯度累积、dev EER 早停与最佳模型恢复。
- CLI 支持数据治理、训练、推理、评测、导出、打包及校验。恢复绑定清单顺序、配置、模型、代码和实际媒体快照；本地加载资源，无隐式下载。见 [开发指南](development.md) 和 [配置模板](../configs/README.md)。
- 根 README、AGENTS、检查与 CI 已更新。历史、参考仓库和产物不混入默认构建；CI 安装 media、CPU 后端和 FFmpeg，尚未远端运行。

## 验证与限制

- 主动测试覆盖指标、泄漏、解码、模型/缓存完整性、训练、恢复、导出与安全打包；其中 2 项保护归档/恢复工具。旧评测器另有 7 项模拟测试。完整检查记录见 [实施工作记录](worklog/2026-10-07-implementation.md)。
- 新 skills 已做 notebook 脚手架、公开论文 API、Dataset Viewer split 和 PDF 提取验证；FFmpeg/ffprobe 已做合成音视频生成与探测。
- 普通沙箱阻止 NVML/GPU 访问及部分 multiprocessing IPC；沙箱外已验证 RTX 4060、CUDA、AMP、DataLoader worker。随机模型实际前向/反向/AdamW profile 见 [model-profile.json](research/model-profile.json)；AASIST-L batch 16 的保留显存 5.41GiB。合成工程结果不证明鉴伪效果或安全屋可运行性。
- 用户已持续授权充分使用本地 GPU；适合的训练、推理、特征提取和 profile
  默认自动使用 CUDA，不设固定卡时上限，以实验价值、资源故障和收益递减为停止条件。
  付费云资源仍须确认预算与数据合规。
- `uv pip check` 核验 123 个已安装包兼容；离线锁校验通过；CPU/CUDA 同时请求按预期被拒绝。早期 `check_environment.py` 不更新参数，模型/训练器 profile 则验证了实际更新；CUDA wheel 的 CPU 执行路径亦通过。
- 独立 `torch 2.14.1+cpu` 环境完成 **311 passed（16.30s，无跳过）**，74 个依赖兼容，项目环境仍为 CUDA 13.0。命令与原始输出见 [CPU wheel 验证记录](research/cpu-wheel-verification.json)。该检查验证后端兼容性，不是完整模型离线演练。
- 实际训练器的 bf16 合成运行见 [trainer-profile.json](research/trainer-profile.json)：LogMel 和媒体级头各有 2 次有效更新、0 次跳步，checkpoint 重载一致。新 wheel 在无 torch/numpy 的干净环境通过无网安装及基础评测；这只验收基础 CLI，完整模型环境仍待验证。
- 尚无真实训练数据、预训练权重、真实鉴伪基线、比赛成绩或远端 CI 结果；模板占位值不是实验资产。
- 上游代码、数据和权重的许可分别核对；文献表中的候选不能当作已批准使用的数据或提交代码。

## 尚未通过的阶段门

| 阶段 | 当前缺口 |
| --- | --- |
| P0 契约 | 本地契约已有；官方 I/O、分数方向、外部资源规则及平台配置待获取 |
| P1 数据与评测 | 代码已有；真实数据来源许可、媒体登记、覆盖与划分审计待完成 |
| P2 双分支基线 | 模型和训练链已有；真实音频/视频 pilot、两种子可比较结果待完成 |
| P3 鲁棒性 | 跨域、压缩、生成器留出与融合晋级证据待完成 |
| P4 较重方法 | 尚未触发采用门，不因算力空闲扩大模型 |
| P5 离线交付 | 包完整性和恢复代码已有；真实资源、干净安装、无网推理及代表性 500 条验收待完成 |
| P6 正式验收 | 最终冻结、留出和官方适配待完成；未推送、发布或正式提交 |

下一步获取许可明确的数据和本地权重，完成覆盖审计、GPU pilot 与真实实验。
候选资源和边界见 [数据获取研究](research/data-acquisition.md)。Git 提交遵循 LoRA：
Logical、Observable、Reversible、Atomic。工程测试通过不能解释为完整比赛项目已完成。

官方 ASVspoof 2019 LA 下载进行中，尚未完整校验或导入。完成后核验官方 MD5，
只导入 train/dev；归档导入器和许可快照已纳入 Git，不能将部分下载用于训练。
