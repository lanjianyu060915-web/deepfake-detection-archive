# 本机 CPU 离线验证记录

## 当前入口（2026-10-08）

65 个固定 CPU wheel 已完成干净断网安装；真实 LogMel42 模型对24条开发音频的
断网推理、重复运行及中断恢复已实际通过，三套分数相同，0失败/重复/缺样。
最新结果见 [真实模型演练](model-execution-20261008/README.md)；
整个项目的最新状态见 [项目状态](../../status.md)。

视频、代表性大批量吞吐和官方目标系统尚未验收。这里的 `source/`、`resume/` 和
执行目录下脚本是各次演练的历史快照，当前尚无收敛后的通用演练工具；不要将旧草稿
视为最新操作入口。后续按实施方案第0节收敛工具，不再硬绑定某次历史安装。

## 2026-10-07 准备记录（历史）

以下保留当时的安装与准备记录；文中的“当前未执行”描述当时状态，后续完成情况以上述入口为准。

2026-10-07。**64/64 个 CPU wheel 已获取并完成独立复核，0 个失败，0 次重试。**
本次 HTTP body 接收量与最终 wheel 存储量均为 449,562,055 字节；
实际获取记录及原锁哈希见 `cpu-wheelhouse.json`。

## 范围

目标为本机 Linux x86_64、CPython 3.12、glibc 2.39，选择当前 `uv.lock` 的
`media + cpu` 闭包，排除 dev/research。准备 64 个第三方 wheel，总计
449,562,055 字节（0.419 GiB），保存在 Git 忽略目录
`cache/wheelhouse/cpu-py312/`。该收集步骤未安装环境、未改动 `.venv`、未下载权重或训练数据。
后续已在独立 `outputs/offline-rehearsal-cpu/env` 完成干净安装，详见下节。

第三方 wheelhouse 已补入当时源码构建的项目 wheel，形成独立 stage 的 65 个 wheel。
真实 trained checkpoint、验证媒体和配置仍待就绪；本次借用系统 Python/uv 与 FFmpeg。
**尚未执行完整无网模型推理，不构成安全屋验收。**

## 可审计文件

- `requirements-cpu-py312.txt`：每个包固定版本和原锁 SHA-256，无下载 URL；用于后续
  `--no-index --find-links` 断网安装。专用于上述平台，不适用于其他 Python/架构。
- `cpu-wheelhouse.json`：原锁摘要、每个 wheel 的原锁 URL、实际请求 URL、最终响应 URL、
  HTTP 状态、实测字节数、重新计算的 SHA-256，以及当时项目存储预算快照。
- 原始获取报告和逐文件日志保留在忽略目录
  `outputs/experiment-planning/offline/cpu-acquisition-report.json`、`cpu-acquisition.log`。
  具体无网预演步骤在同目录 `README.md`，等待真实权重和样本到位后执行。

获取流程只访问清单指定的固定版本 wheel。PyTorch 原锁 `download-r2.pytorch.org`
地址 HEAD 返回 403，因此使用同路径官方 `download.pytorch.org` 域名；两个 CPU
wheel 下载内容仍必须匹配原锁 SHA-256。第三方镜像内容同样由原锁哈希约束，不能通过
移除哈希或改版本绕过验证。许可证和发行元数据保留在各 wheel 的 `.dist-info/` 中。

独立复核读取全部本地文件，再次比对 `uv.lock` 的包名、版本、URL 和 SHA-256，核对
实际大小与 inventory，检查 wheel METADATA 包名/版本和当前平台默认运行依赖闭包。
它不执行安装，也不验证 GPU 或模型质量。

## 存储预算

本次下载增加 449,562,055 字节 wheel 缓存，计入项目新增存储预算；不是训练媒体。
`cpu-wheelhouse.json` 的 `budget_snapshot` 统计 cache/data/models/outputs 中普通文件
逻辑字节数，不重复跟随符号链接。快照记录起止时间，可能包含其他任务正在写入的文件，
不能当作原子快照或历史网络流量总账。`.venv`、旧归档和机器级缓存不在这个项目资产口径。
分别对照方案的缓存 20 GiB、项目资产新增 100 GiB 上限；当前余量以记录值为准。
原始下载报告另记录本次接收的 HTTP body 字节，可区分重试流量和最终存储字节。

独立复核时快照（UTC 2026-10-07T06:53:46.102019+00:00）：cache 共
7,494,915,303 字节（6.980 GiB），
四类项目资产共 7,921,529,508 字节（7.377 GiB），
均低于相应 20/100 GiB 上限。该时刻媒体下载仍可能正在写入，后续必须重新计量。

## 已通过：本机 CPU 真断网干净安装

2026-10-07 06:57:08–06:57:15 UTC，以全新 `outputs/offline-rehearsal-cpu`
构建项目 wheel、复制并复核 64 个第三方 wheel，成功安装 65 个发行包。项目构建本身
使用 `uv --offline build --no-sources --wheel` 和已有构建后端缓存，无联网构建。

- 网络命名空间与父进程 ID 不同：`net:[4026532360]` → `net:[4026532291]`。
  namespace 内 `/proc/net/route` 实际读取结果为空字符串，外网 TCP 探针返回
  `PermissionError: [Errno 1] Operation not permitted`，不得改写为其他错误原因。
  该失败可同时受到运行沙箱限制；命名空间切换和本地-only 安装参数是另一组独立证据。
- 全新 venv，从空缓存开始，全部安装命令使用 `--no-config --offline --no-cache`，
  `--no-index --find-links`、`--only-binary=:all:`、`--require-hashes --strict`。
  固定的临时缓存目录在结束后为空，未使用项目 `.venv` 或原有 CPU 环境安装包。
- `pip check` 检查 65 包兼容；torch 为 `2.14.1+cpu`，CUDA build 为 null，
  CUDA available 为 false；项目包路径位于本次新环境的 `site-packages`。
- 基础 CLI `--help` 成功；namespace 内 7 条命令及外部准备命令均返回 0。
  `.venv` 的配置与 246 份发行元数据/RECORD 前后哈希一致。
- **没有运行模型推理、没有构造随机分类器替代真实权重、没有验收模型质量。**

项目 wheel 为 `deepfake_detection_archive-0.1.0-py3-none-any.whl`，83,861 字节，
SHA-256 `465700f71801b819bef9dcfdcfd62fc9f339484bbfa6e7a75b0bd8aba11d8d5d`。
源文件及构建输入的前后哈希一致；具体版本只对这次记录的源码快照成立。

审计文件：

- [安装结果摘要](cpu-installation-summary.json)：通过状态、边界、项目 wheel hash、
  网络证据、原始报告/脚本摘要及最新预算快照。
- [构建及准备原始记录](cpu-installation-preparation.json)：精确 argv、开始/结束时间、
  退出码、stdout/stderr、系统工具路径/哈希/动态库、源码和主环境元数据哈希。
- [namespace 内原始记录](cpu-installation-probe.json)：逐条安装/验证命令、输出、
  明确设置的环境变量、网络 namespace/route/TCP 实测值和实际安装包列表。
- [准备脚本原文](source/prepare_cpu_install.py)、[安装验证脚本原文](source/install_probe.py)：
  字节摘要与上述实际运行记录一致。脚本要求目标目录不存在，重复演练应使用新的
  隔离检出或明确的新目标目录并记录新脚本哈希，不自动删除既有证据。

本次实际启动命令：

```bash
python3 -B docs/research/offline-rehearsal/source/prepare_cpu_install.py \
  /home/maoxiao/WorkSpace/deepfake-detection-archive
```

最新预算快照 UTC 06:58:10：cache 7,810,622,695 字节（7.274 GiB）；
cache/data/models/outputs 合计 10,204,884,867 字节（9.504 GiB），包含 stage 中
复制的 wheel 和新 venv；仍低于 20/100 GiB 门限。并行下载下该快照非原子。

## 剩余：真实权重无网推理与目标系统验收

uv 0.12.13 未提供 download 子命令或原始 wheel 缓存导出接口；现有缓存中的解包目录
不能替代可独立携带、按原锁哈希验证的 wheelhouse。安装分支已通过。真实权重和开发集
小样本复制并验哈希后，在同等 `unshare --user --map-root-user --net` 隔离中，
使用本次新环境执行实际 CLI infer/evaluate/resume。当前仍未执行真实模型推理。

安装阶段采用以下参数形式（实际绝对 argv 与退出码已在原始记录中保存）：

```bash
uv --no-config --offline --no-cache pip install \
  --python /ABSOLUTE/REHEARSAL/env/bin/python \
  --no-index --find-links /ABSOLUTE/REPO/cache/wheelhouse/cpu-py312 \
  --only-binary=:all: --require-hashes --strict \
  -r /ABSOLUTE/REPO/docs/research/offline-rehearsal/requirements-cpu-py312.txt
```

上述简写只指向第三方清单；本次实际 stage/install.txt 已追加项目 wheel 的固定版本和哈希，
并由同一命令一起安装。当前 FFmpeg 是 Linuxbrew
shared build，依赖 Cellar/opt 的多个动态库，不能只复制两个可执行文件就视为系统闭包。
CUDA 演练另需 cu130 wheel、可用 GPU/驱动和对应验证。官方系统信息未知，不能把
本机 CPU 结果外推到安全屋。

依据：[uv 缓存](https://docs.astral.sh/uv/concepts/cache/)、
[uv CLI](https://docs.astral.sh/uv/reference/cli/)、
[pip download](https://pip.pypa.io/en/stable/cli/pip_download/)、
[FFmpeg 许可说明](https://ffmpeg.org/legal.html)。

## 已准备、待真实 checkpoint：24 条开发样本模型演练

[草稿脚本](source/model_rehearsal.py) 已归档，曾通过 Ruff 和格式检查；独立审阅仍有
四项未解决问题，**不得执行，尚未完成模型演练**。见 [待修复记录](model-rehearsal-pending.md)。
训练分支将先提供完成训练且 SHA-256 已核对的 LogMel42 `best.pt`、dev-only 清单、
来源登记、对应推理预处理配置及训练证明。准备模板位于忽略目录
`outputs/experiment-planning/offline/model-job.pending.json`，`training_completed=false`
且路径为显式占位值，不能直接作为实验配置。

脚本固定对 dev-only 清单分别按 `sample_id` 排序，各取 12 真/12 假，再将并集按 ID
排序；不会读取模型分数来选样，也不会读取 holdout 媒体。复制实际 checkpoint、24 条
媒体、来源、许可及训练证明，逐文件验证 SHA-256。每种模型使用独立新 stage，不覆盖
旧安装记录或其他候选结果。

执行在网络命名空间内先完成两次独立完整推理（参考本身也禁网），保留完整结果并逐分数
严格比较，再对新输出目录的实际推理子进程进行 `SIGSTOP`，确认该**精确自有子进程**
状态为 `T/t` 且已有 1–23 条完整记录后，发送 `SIGINT`、`SIGCONT`。等待非零退出后，
使用相同配置 `--resume`，核对已写前缀字节不变、24 条无重复遗漏、分数严格一致；
再次 resume 检查最终分数文件哈希不变。若任一步失败，保留证据并明确失败，不扩大容差
或换随机模型掩盖问题。

已用一个仅写单行后等待的独立合成进程验证 OS 信号控制机制：确认停止状态 `T`，
SIGINT 后退出码 `-2`，文件前缀保持不变。记录位于忽略目录
`outputs/experiment-planning/offline/signal-probe/report.json`。**它只验证进程控制，
不证明实际 detector 的中断恢复通过。**

已安装包的 `code_identity` 从包路径向上读取 Git 信息；当前环境位于仓库内，因此真实
演练前由主代理协调暂缓提交和受跟踪文件写入，完成恢复验证后恢复。脚本和输入先冻结，
运行时仅写忽略目录，保留实际代码/Git 身份，不手工改写身份字段。

模型演练全部通过后才能追加独立真实推理报告；当前完成范围仍仅为依赖获取、干净断网
安装和基础 CLI。24 条样本属于本机工程演练，不满足代表性 500 条验收或官方安全屋验收。
