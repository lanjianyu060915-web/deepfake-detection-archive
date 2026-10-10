# 云端开发与协作

更新：2026-10-10。后续开发和训练转到云端。负责人发布 Issue、审查 PR 并合并；
队友和各自的 Codex 在独立任务分支完成工作。代码已迁入服务器，本轮核对环境和开发入口，
不启动新训练。

## 1. 从哪里开始

1. 接手者阅读[个人交接](personal-cloud-codex.md)，核对当前工作副本和已分派的
   [Issue](issues/README.md)；不需要重新迁移已在服务器上的代码。
2. 已发布的音频、视频和数据任务各自完成必要资产接入；代码开发可使用现有共享环境，
   不等待统一环境Issue，实验须先确认任务输入可用。
3. 音频从全量 LogMel/HuBERT 同划分比较开始；视频从 Hunyuan 受控集开始。
4. 云端第一轮取得结果后再决定微调、时序和融合；不因迁移重写模型或建设调度框架。

当前结果看[状态](../status.md)，算法顺序看[方案](../plans/implementation-plan.md)。
当前实例已可用：RTX 4090 24 GiB、驱动 595.91.07、CUDA 13.2。
已有获配实例可按任务使用，不自动购买、扩容或启用新的付费服务。

## 2. 服务器环境

当前共享环境为 `/opt/venvs/deepfake-cu132`，Python 3.12.15、uv 0.13.0、
PyTorch 2.14.1+cu132、TorchVision 0.29.1+cu132。开发、研究、媒体依赖与锁文件匹配；
11 项项目 skills 随代码完整迁入。使用仓库根目录的入口：

```bash
scripts/devctl doctor
scripts/devctl run python -m deepfake_detection_archive --help
scripts/devctl check
```

完整验证结果见[状态](../status.md)。共享环境不安装具体工作副本的项目；`devctl` 显式加载
当前源码，避免不同 checkout 串用。依赖维护、direnv 和独立 CPU 环境见
[开发指南](../development.md)。不要将共享 GPU 环境切换为 CPU 后端，也不要迁移旧 `.venv`。
FFmpeg/ffprobe 采用 root 管理的 `/opt/ffmpeg/btbn-628033703`，全局入口为
`/usr/local/bin/ffmpeg`、`/usr/local/bin/ffprobe`，各账号可读取和执行。
版本 `N-127271-gba987fe24d-20261010` 支持项目使用的 `-fps_mode`；系统 `/usr/bin` 下的
4.4.2 仍保留，日常通过 PATH 使用全局新版。

使用独立工作副本；数据和产物放持久卷，关机前确认平台不会回收该卷。
历史交接曾生成 `outputs/handoff/20261010/project-code.bundle` 和同名 `.sha256`。
它是无法访问团队仓库时的可选代码恢复来源，不是当前服务器接手的必做步骤。
需要使用时先核验附带 SHA-256，再从 bundle 克隆；其 origin 指向本地 bundle，须按负责人
提供的团队地址配置后才能提交 PR。该包在当前工作副本不可用，本轮未验证包恢复流程；
包位置、远端权限和分支合并状态应按实际核对。

日常代码变更先跑相关测试，PR 提交前跑一轮标准检查；同一提交不反复重跑。
GPU 型号变化时只做一次小 profile 调 batch/精度，不把 RTX4060 的 batch 当作云平台限制。
长任务用平台任务系统或 tmux，stdout/stderr 留在该 run 的目录，避免终端断开丢失进程状态。
当前没有 optimizer/RNG 恢复；中断需新目录重启，`infer --resume` 只适用于推理。

## 3. Git 外资产

**克隆仓库不会取得数据、权重、缓存或本地训练结果。**
[资产目录](local-assets.json) 包含旧机快照和有日期的服务器核对，用于定位资产与缺口，
不是服务器实时清单。
当前 checkout 尚未接入 `data/`、`models/` 与旧实验 `outputs/`；`external/` 仅恢复了
固定提交的 ASVspoof 2021 参考；
新生成的合成探针输出不代表训练资产已齐全。服务器 `/srv/shared-data` 已有下载产物：

- ASVspoof 2019 LA 两个 parquet 的元数据可读，train 25,380、dev 24,844 行；
  原媒体尚未物化。train 比旧实验多一行，需结合官方 protocol 核对后再导入。
- ViF Panda `source_videos.zip` 约 1.29 GB，ZIP 目录可读、含 7,040 项，尚未解压和导入。
- FakeSound ZIP 约 986 MB，尚未解压；登记为 fake-only 诊断数据，不能单独开展二分类训练。

归档可读不等于训练路径可用。先核对实际进程、落盘产物和来源记录，再按任务接入；
不要据工作副本缺目录判断服务器无资产，也不要启动重复下载。下面是各类资产的处理方式：

当前共享目录归属 `root:dlteam`，团队成员为 `maochuanjun`、`lanjianyu`、`zhangleyan`、
`renruoxi`。目录2755、文件0644，路径和ACL允许各队友读取，原始下载由维护者写入。
两个ASV Parquet和ViF ZIP的完整SHA-256均与各自 `provenance.json` 一致；FakeSound没有
已登记的参考哈希，本轮仅确认ZIP目录可读。下载目录中的未完成文件不能当作已取得的数据。

每个工作副本可使用同一份只读输入入口，无需复制大文件：

```bash
mkdir -p data
ln -s /srv/shared-data/datasets data/shared-raw
ls -lh data/shared-raw/asvspoof2019-la/*.parquet
```

当前worktree已建立此链接。新工作副本只需建立一次；若同名路径已存在，先核对其指向，
不要覆盖。输入按实际格式物化/导入，清单和实验配置使用各自任务路径；`outputs/`、`cache/`
保持各自可写。不要为共享方便而给原媒体开放团队写权限，或对用户家目录执行宽泛权限修改。

| 资产 | 云端用途 | 处理 |
| --- | --- | --- |
| ASVspoof train/dev 原媒体 | LogMel/HuBERT 训练 | 复制已有原媒体或按已登记来源重新取得，保持目录布局 |
| ViF Panda 原媒体 | 视频对照 | 同上，使用固定来源版本 |
| HuBERT 本地 bundle、ResNet18 encoder | 冻结特征实验 | 复制已转换权重及许可，核对登记哈希 |
| 全量 LogMel42 best checkpoint | 保留基线、云端离线比较 | 迁移小型状态包中的权重、连续分数和记录 |
| 本轮数据清单、来源与视频受控划分 | 保持可比协议 | 迁移状态包；缺失时按实验卡重建 |
| 特征/波形缓存、`.venv`、wheel 缓存 | 加速或本机环境 | 默认不迁移，在服务器按需重建 |
| EnvSDD、未完 Urban 断点、历史失败 run | 辅助研究与追溯 | 保留本地，首批任务不依赖它们 |

历史交接生成过 `outputs/handoff/20261010/research-state.tar.gz` 及成员/哈希清单，包含
当时的数据清单、选定模型和实验记录，不含原媒体和预训练编码器。它可作为恢复来源；
当前工作副本没有该包，本轮未验证解包流程，不能沿用旧说明断言它尚未上传或已经迁入。
取得包后先验证附带 SHA-256 和成员路径，在目标工作副本恢复，避免覆盖已有任务输出。

原媒体和预训练权重另行传输到清单所列路径。也可将工作副本中的 `data/`、`models/` 指向
持久卷的对应目录，保持相对布局；原媒体和权重设为只读，任务的 `outputs/` 和 `cache/` 各自独立。
只对新取得或传输的必要资产做一次完整性核对，不把历史全量审计当作每次训练的前置步骤。

## 4. Issue → 分支 → PR → 审查 → 合并

已发布 Issue 的目标、数据划分、输出和完成条件见[任务入口](issues/README.md)。
队友领取后建立 `issue/<编号>-<主题>`；迁移分支合并前的基点见任务入口，合并后使用最新
`main`。不同任务用不同 clone/worktree，不要在同一个 checkout 同时切分支或改锁文件。

- 数据任务维护来源/清单；算法任务各自维护一张实验卡和实值配置。
- 音频、视频可以并行写代码；共用单GPU时在 Issue 中约定占用顺序，避免同时满额训练。
- 输出目录带任务号和 run ID；新实验使用新目录，保留已有 run。相同缓存可以顺序复用，
  并行任务默认独立缓存，不加入缓存锁或统一调度系统。
- PR 关联 Issue，说明实际改动、样本数、EER、失败率、资源和验证；方法无收益也可以如实完成。
- 代码、配置、简洁实验卡进入 Git；原媒体、权重、大分数文件与日志放持久卷/团队授权存储，
  PR 给出可访问路径和必要哈希。不得把服务器的绝对路径写成协作者默认配置。
- 负责人核查协议、结果和验证后合并。需要修改就继续原 PR，不复制成第二套结果。
  `docs/status.md` 在合并时统一更新，避免各分支争抢该文件。

已按用户授权发布三个独立Issue，实际编号见[任务目录](issues/README.md)，暂待领取。
迁移修复由独立分支管理，Issue正文与任务入口同步开发命令和分支基点。
用户保留审查及合并责任，Codex 不擅自合并、购买资源或正式参赛提交。
