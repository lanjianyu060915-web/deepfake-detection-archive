# RC3 执行策略与离线容量

2026-10-16；[完整摘要](summary.json)。本轮补齐 #65 的资源路径边界和 #66 的
执行策略，并为 #54 验收新的 cu126 离线包。沿用 v0.1.1 的 AST/ResNet18
研究权重、采样、FP32 与固定 NumPy AST 前端，不训练、不计算 EER、不读取 holdout。

## 固定构建

RC3 由干净 `2e8388ae0b3ce2489d70af33ce821ba23665ddb8` 构建，含包内
CPython3.12.15、Torch2.14.1+cu126 和固定同源码 LGPL FFmpeg/ffprobe。
实际包位于 `/var/tmp/deepfake-media-build-20261016/rootfs/app/离线包 rc3`，
`release.json` SHA256 为
`5b9d3a92092c2235f6edf82408188ca5b5e5dbdf11cebf29b2b34aa1e5b35b4e`。
后续 Docker CI 与 TMPDIR 修正见 [PR #85](https://github.com/lanjianyu060915-web/deepfake-detection-archive/pull/85)；
容量运行显式指定输出目录，源码固定在上述构建，不混称为 PR 后续提交。

默认 `reference` 保持固定执行；显式 `adaptive` 探测 CPU 配额/亲和性、内存及实际
可见 CUDA 设备、加载量和余量，只调整独立帧/窗口的执行 batch。启动 profile 最多
3 次，超过 10 秒不再启动下一次调用，单次模型调用不强制中断；保留至少 256MiB
或空闲显存的 15%。解码 worker 固定为 1，完整时序头和文件聚合保持原语义。
`execution.json` 保存有效计划、加载/profile/正式耗时、峰值及减批事件，resume 复用
已保存计划并绑定设备、软件构建和浮点设置。训练资源要求另见[开发指南](../../development.md)。

## 工程数值与搬移

20 个输入含常见音频格式、多声道/静音、短长/竖屏/VFR/无音轨视频和中文空格路径。
按逻辑相对路径与媒体 SHA 对齐，保留绝对路径产生的原 sample_id，预设分数容差 1e-4。

| 比较 | 最大分数绝对差 | 30 份前端张量最大差 |
| --- | ---: | ---: |
| 原 cu126 CPU → RC3 CPU reference | 0 | 0 |
| 原 cu126 GPU → RC3 GPU adaptive | 0 | 0 |
| RC3 CPU → GPU | 7.6294e-6 | 0 |
| RC3 CPU → Ubuntu24 新用户名 CPU | 4.0531e-6 | 0 |

各运行均 20/20、零失败。Ubuntu24.04.5/glibc2.39 用户态实际使用新用户名
`contest_runner`、UID1002、独立 HOME、任意 CWD、只读包/输入和空缓存，禁止新建
INET/INET6 socket，不挂载宿主 Python、FFmpeg 或共享 venv。该环境共享宿主内核，
不作为 Docker、新物理机器 GPU 或新 8GiB 显卡证据。默认 registry/config/权重/
encoder/runtime contract 均限制在所选包根；显式错误根、缺资源、越界和错误前端
有定向回归，研发显式配置仍可指定本地绝对资源。

## 真实文件容量

RTX4090 24GiB、驱动595.91.07、系统 Toolkit13.2 保留；包内 CUDA12.6 运行库。
两项同时运行，OMP/MKL 线程各 2；只读包、空缓存、禁止新网络连接，实际包内 CLI
完成发现、解码、模型、聚合与写出。每项均按 SHA、完整路径、ID 及 CSV 核对闭合。

| 输入 | 成功/固定分母 | 完整端到端秒 | wait4 RSS KiB | CUDA allocated/reserved B | 有效 batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 音频 | 10000/10000 | 1046.8319 | 1730424 | 400120320 / 434110464 | 1 |
| 视频 | 10000/10000 | 1613.3152 | 1513624 | 110129664 / 134217728 | 4 |

均退出 0、零失败、零遗漏、所有分数有限，无 OOM 减批。wall 为 spawn→wait4，
包含 Python/Torch 冷启动、发现、加载、profile、解码、评分及写出；RSS 为 wait4
所报子进程及已等待后代的最大值，非同时进程树 RSS 总和。显存为进程 CUDA allocator，
非整卡占用。音频仅一个固定窗口，所以请求 batch4 仍有效为 1；视频请求8、四帧有效4。

复用原已见选集，不新挑选媒体；音频为 In-the-Wild 语音诊断集，视频为已见 train/dev
短受控 H264（1.25–3.5 秒）。输入为 20000 个原文件的任务专用硬链接，新增媒体复制
0B，原 inode 权限不变。该结果不证明通用音频、长视频、独立来源泛化或比赛时限；
两项并行，不能与旧串行结果直接比较加速比。

## OOM 与结果保留

RTX4090 上通过真实 CUDA allocator 进程配额触发 OOM：正式 batch8 失败后减为4
成功；随后配额继续收紧，4→2→1 仍失败，任务停止且前两条已提交控制结果保留。
OOM 原因和每次减批已记录。仅捕获 CUDA OOM，最多减半 3 次，不改精度、采样或
补造分数。此实验是实际内存分配控制，不是媒体准确率或真实 8GiB 显卡验收。

## 保存与剩余范围

`C15-runtime-completion-20261016` 持久保存 162 内容文件、109657257B，含完整分数/
选择清单/计划、20 输入回归和 30 张量、源码与依赖锁、27985 个包文件的 SHA 清单，
以及本轮有界 C3 历史源码查找事实；权重复用 C1，工具源码与工程媒体复用 C13。
独立 `index.json`/`SHA256SUMS`、团队只读，不复制完整运行库或训练媒体。

[Docker 实际 CI](https://github.com/lanjianyu060915-web/deepfake-detection-archive/actions/runs/38916789385)
已成功构建 CPU/cu126 两路无权重镜像并完成断网、非 root、只读根/输入/模型、空缓存推理及
save/load。两路均4/4，host→container、host→loaded、repeat 分数差均0，配置digest保存/加载后不变。
镜像不包含模型配置、权重或媒体；实际权重只读挂载，结果按相对路径+媒体SHA对齐。

| 运行库 | 镜像归档字节 | 镜像配置 digest |
| --- | ---: | --- |
| CPU | 469660126 | `sha256:65166a271b273d2e4539654698983e765b9b9c0f23ac099af01223c279e28d54` |
| cu126 | 4050034064 | `sha256:53eeaf4a1acc8d7d9e9b14ae5b73e7f0e8881afb05631127f243cef7a431c826` |

完整小结果及归档SHA保存C16，实际镜像持久复制进行中，完整值见摘要。Actions默认checkout的实际
干净源码为 `250f3b351fb8390352ce20d237d776056bfcb2f5`（PR测试merge），运行的PR head为
`e2ef985cbba2db44a144910c852caafe25f3c40d`；与容量RC3源码分别登记，不冒充同一构建。
两构建均在无 GPU runner 以 CPU 执行，不扩展为 GPU 容器或容器万级容量。
#50 的 C3 历史 dirty 源码仍未找到匹配快照，optimizer/RNG 及跨实例恢复未补齐；
#54 的最终候选、平台 I/O、赛用许可和新机器 GPU 仍有缺口。v0.1.1 发布默认不变。
