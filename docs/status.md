# 当前项目状态

更新日期：2026-10-07。阶段：**准备完成，整体实施方案待用户确认**。

## 当前事实

- 用户确认音频和视频均参加；本地有 RTX 4060 8GB，云算力后续考虑；报名、安全屋权限与执行规范待定。
- 已核对官网赛题/规则、调研相关论文和项目，形成 [整体方案](plans/implementation-plan.md)。算法路线是待验证候选，不是已复现结果。
- 首轮安装了 3 个研究 skills 并复用 PDF；本轮核验用户指定 ds-skills 的 535 项均已安装且与固定上游相同，选用 6 项完成合成验证，无覆盖安装。已准备 3 份模型源码参考及 ds-skills 固定源码，记录见 [工具说明](research/tooling.md)。
- 用户授权后续相关依赖按需自行安装；现已锁定并安装 `dev` / `research` / `media` 和 `cu130` 后端。torch 2.14.1+cu130、torchvision 0.29.1+cu130、timm、Transformers、音视频处理与研究工具已就绪；`cpu` 后端可选但本轮未安装该 wheel。
- 旧项目已整体迁至 `archive/legacy-20261005/`，1,526 个文件、138,273,682 字节与冻结清单一致；比赛原始资料改放 `docs/competition/source/`。
- 根 README、AGENTS、忽略规则、限定范围的 Ruff/mypy/pytest 和 CI 配置已更新。归档、参考仓库和产物不会混入默认包构建。

## 验证与限制

- 主动工程 2 项测试检查归档完整性和恢复工具原字节；旧评测器另有 7 项模拟测试。Ruff、格式、mypy、锁定同步与 wheel/sdist 构建已完成本地验证。
- 新 skills 已做 notebook 脚手架、公开论文 API、Dataset Viewer split 和 PDF 提取验证；FFmpeg/ffprobe 已做合成音视频生成与探测。
- 普通沙箱阻止 NVML/GPU 访问；沙箱外已读取 RTX 4060 Laptop 8GB、驱动 616.92，并通过 CUDA 13.0 张量计算、torchvision NMS、AMP、随机音频/视觉模型前向。完整合成环境证据见 [environment-check.json](research/environment-check.json)。实际比赛模型的 8GB 训练/推理预算、安全屋 GPU 仍未验证。
- `uv pip check` 核验 123 个已安装包兼容；离线锁校验通过；CPU/CUDA 同时请求按预期被拒绝。新增探针也纳入 Ruff/格式检查，合成梯度测试没有参数更新；CUDA wheel 的 CPU 执行路径亦通过。
- 尚无新 EER 实现、训练数据、模型基线、真实鉴伪结果、比赛成绩或远端 CI 结果。
- 上游代码、数据和权重的许可分别核对；文献表中的候选不能当作已批准使用的数据或提交代码。

## 确认后的起点

按方案从 P0/P1 开始，先核对平台契约、验证 GPU、登记合法数据并建立评测。不先下载大型语料或直接训练模型。

当前等待用户对 [实施方案](plans/implementation-plan.md) 的确认；不自动转入算法开发。本轮工作记录见 [2026-10-07](worklog/2026-10-07.md)。
