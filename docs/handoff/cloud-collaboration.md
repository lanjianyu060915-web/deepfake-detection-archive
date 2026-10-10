# 云端开发与协作

更新：2026-10-10。后续开发和训练转到云端。负责人发布 Issue、审查 PR 并合并；
队友和各自的 Codex 在独立任务分支完成工作。本地这一轮只做收尾，不再启动新训练。

## 1. 从哪里开始

1. 负责人阅读[个人交接](personal-cloud-codex.md)，通过团队仓库或下述Git bundle取得本轮代码，
   再发布[第一批 Issue](issues/README.md)。当前提交仍只在本地，尚未推送。
2. 先完成 Issue 00：已有实例、环境和资产可用后，音频与视频任务可并行开发。
3. 音频从全量 LogMel/HuBERT 同划分比较开始；视频从 Hunyuan 受控集开始。
4. 云端第一轮取得结果后再决定微调、时序和融合；不因迁移重写模型或建设调度框架。

当前结果看[状态](../status.md)，算法顺序看[方案](../plans/implementation-plan.md)。
云平台、GPU、镜像和预算尚未选定；本文不假定任何厂商或已经存在的远端资源。
已有获配实例可按任务使用，不自动购买、扩容或启用新的付费服务。

## 2. 服务器环境

代码可从团队仓库克隆。为避免接手依赖原电脑，本轮还生成
`outputs/handoff/20261010/project-code.bundle`及同名`.sha256`，保存当前分支的完整Git历史、
代码、11项项目skills和交接文档。与资产包一并传到服务器后可独立恢复：

```bash
# 在已经放好bundle与校验文件的目录
sha256sum -c project-code.bundle.sha256
git clone project-code.bundle deepfake-detection-archive
cd deepfake-detection-archive
```

bundle克隆后的origin指向该本地bundle，**不能直接向它提交PR**。已有团队仓库时，
由云端Codex按负责人提供的仓库地址配置origin，将交接分支纳入团队审查流程；
或在已有clone中fetch bundle的`maocj`分支。由负责人审查合并到团队默认分支后再分派后续任务。
这份bundle没有上传、推送或创建远端仓库。

使用仓库工作副本；数据和产物放持久卷，关机前确认平台不会回收该卷。
不要迁移本地 `.venv`。项目使用 Python 3.12、uv、FFmpeg/ffprobe；CUDA 后端由锁文件固定。

```bash
# 在已克隆的仓库根目录；宿主需先具备 Python/uv 和 FFmpeg/ffprobe
nvidia-smi
uv --version
ffmpeg -version
ffprobe -version
uv sync --locked --group media --group research --extra cu130
uv run --no-sync python tools/check_environment.py --help
uv run --no-sync python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

`cu130` 是当前已锁定、已在本地实测的组合，不代表任意云镜像都支持。先确认实例驱动可运行
该后端；若不支持，由环境 Issue 选择兼容镜像，或以独立依赖 PR 增加合适后端，不能静默改锁。
仅做 CPU 检查时用 `--extra cpu` 替换 `--extra cu130`，两者互斥。后续命令使用 `--no-sync`。

新机器完成一次标准检查：

```bash
bash tools/check_project.sh
```

日常代码变更先跑相关测试，PR 提交前跑一轮标准检查；同一提交不反复重跑。
GPU 型号变化时只做一次小 profile 调 batch/精度，不把 RTX4060 的 batch 当作云平台限制。
长任务用平台任务系统或 tmux，stdout/stderr 留在该 run 的目录，避免终端断开丢失进程状态。
当前没有 optimizer/RNG 恢复；中断需新目录重启，`infer --resume` 只适用于推理。

## 3. Git 外资产

**克隆仓库不会取得数据、权重、缓存或本地训练结果。**
完整清单见[本地资产目录](local-assets.json)。下面是迁移决策：

| 资产 | 云端用途 | 处理 |
| --- | --- | --- |
| ASVspoof train/dev 原媒体 | LogMel/HuBERT 训练 | 复制已有原媒体或按已登记来源重新取得，保持目录布局 |
| ViF Panda 原媒体 | 视频对照 | 同上，使用固定来源版本 |
| HuBERT 本地 bundle、ResNet18 encoder | 冻结特征实验 | 复制已转换权重及许可，核对登记哈希 |
| 全量 LogMel42 best checkpoint | 保留基线、云端离线比较 | 迁移小型状态包中的权重、连续分数和记录 |
| 本轮数据清单、来源与视频受控划分 | 保持可比协议 | 迁移状态包；缺失时按实验卡重建 |
| 特征/波形缓存、`.venv`、wheel 缓存 | 加速或本机环境 | 默认不迁移，在服务器按需重建 |
| EnvSDD、未完 Urban 断点、历史失败 run | 辅助研究与追溯 | 保留本地，首批任务不依赖它们 |

本轮生成 `outputs/handoff/20261010/research-state.tar.gz`，附带成员/哈希清单；
它包含现有清单、选定模型和实验记录，不包含原媒体或预训练编码器。包仍在本地，尚未上传。
在双方具备传输权限的渠道复制后，云端可在仓库根目录执行：

```bash
# 包和校验文件已经放到云端某个传输目录后
sha256sum -c research-state.tar.gz.sha256
# 将路径改成实际上传位置，在仓库根目录恢复
tar -xzf /ACTUAL_UPLOAD_PATH/research-state.tar.gz
```

原媒体和预训练权重另行传输到清单所列路径。也可将工作副本中的 `data/`、`models/` 指向
持久卷的对应目录，保持相对布局；原媒体和权重设为只读，任务的 `outputs/` 和 `cache/` 各自独立。
只对新取得或传输的必要资产做一次完整性核对，不把历史全量审计当作每次训练的前置步骤。

## 4. Issue → 分支 → PR → 审查 → 合并

每个 Issue 写清目标、比较对象、数据划分、输出和完成条件，直接用[任务草稿](issues/README.md)。
队友领取后从仓库默认分支最新提交建立 `issue/<编号>-<主题>`，不同任务用不同 clone/worktree。
当前本地分支名不作为团队主分支约定。不要在同一个 checkout 同时切分支或改锁文件。

- 数据任务维护来源/清单；算法任务各自维护一张实验卡和实值配置。
- 音频、视频可以并行写代码；共用单GPU时在 Issue 中约定占用顺序，避免同时满额训练。
- 输出目录带任务号和 run ID；新实验使用新目录，保留已有 run。相同缓存可以顺序复用，
  并行任务默认独立缓存，不加入缓存锁或统一调度系统。
- PR 关联 Issue，说明实际改动、样本数、EER、失败率、资源和验证；方法无收益也可以如实完成。
- 代码、配置、简洁实验卡进入 Git；原媒体、权重、大分数文件与日志放持久卷/团队授权存储，
  PR 给出可访问路径和必要哈希。不得把服务器的绝对路径写成协作者默认配置。
- 负责人核查协议、结果和验证后合并。需要修改就继续原 PR，不复制成第二套结果。
  `docs/status.md` 在合并时统一更新，避免各分支争抢该文件。

本轮没有发布 Issue、推送分支或创建 PR。后续分派任务时由仓库权限和 Issue 授权执行对应远端操作；
用户保留审查及合并责任，Codex 不擅自合并、购买资源或正式参赛提交。
