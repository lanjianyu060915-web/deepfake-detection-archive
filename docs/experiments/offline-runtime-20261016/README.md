# cu126 与固定媒体工具离线回归

2026-10-16，#54/#65。本轮解决新环境与媒体工具闭包，原 v0.1.1 AST/ResNet
权重、NumPy AST 前端、4帧策略和分数语义保持原配置，不做训练、EER或候选晋升。
源码为干净 `2ecb7a7`，完整指标见 [summary.json](summary.json)。

## 环境与工具

共享开发环境 `/opt/venvs/deepfake-cu126`：Python3.12.15、Torch2.14.1+cu126、
TorchVision0.29.1+cu126、TorchAudio2.11.0+cu126。RTX4090 实测 TorchAudio
原生 CUDA12060、GPU lfilter 和 TorchVision CUDA NMS 成功。
系统 Toolkit13.2 和驱动595.91.07保留；预编译wheel使用自身CUDA12.6运行库。
旧权重仍为 `numpy_spectrogram_v1`，安装TorchAudio不改变前端。
新RC按checkpoint前端收集依赖，NumPy权重包无需包含未使用的TorchAudio。

固定媒体工具使用原服务器 FFmpeg 对应源码 `ba987fe24d6713c4d4b3a15cebaeca18c41fbd31`，
保留 x86 汇编优化和原生解码/滤镜，禁用外部自动探测、GPL/nonfree、网络及硬件解码。
包内含对应源码原包、LGPL许可、构建配方、实际工具及哈希；只依赖系统glibc/libm。
显式bin目录→包内工具→源码开发PATH由同一解析点负责，实际工具字节绑定恢复身份。

曾试用FFmpeg8.1.3最小构建，20条均可评分，但相对旧工具最大分数差0.34718，
未达预先固定1e-4容差，已拒绝。最终采用同源码构建，20条解码数组与原工具完全一致。
该最小构建不支持AV1软件解码，不替换服务器完整GPL工具。

## 真权重回归

固定10条音频、10条视频：FLAC/WAV/MP3/AAC/Opus/Vorbis、仅音轨容器、
多声道/长静音音频及H264/HEVC/VP9/MPEG4、竖屏/无音轨/长视频/VFR。
18条为无标签自生成工程样本，2条复用已见train/dev受控视频；不使用holdout。
以媒体SHA和显式逻辑路径对齐绝对路径派生ID，30份实际波形/AST/视频输入张量随结果保存。

| 比较 | 最大分数绝对差 | 30份输入张量最大差 |
| --- | ---: | ---: |
| 原cu132/原媒体工具CPU → 新cu126/固定工具CPU | 0 | 0 |
| 新cu126 CPU → 同包RTX4090 CUDA | 7.63e-6 | 0 |
| 新cu126 CPU → Ubuntu24.04用户态CPU | 4.05e-6 | 0 |
| Ubuntu24.04 CPU → 同环境重复 | 0 | 0 |

四个新运行各20/20成功、零失败，全部满足预设分数容差1e-4。
GPU allocator allocated/reserved峰值422,489,088/509,607,936B。
CPU/GPU CLI调用计时27.25/16.22秒，**不含测量器预先import Torch和之后的张量API回归**，
不是完整冷启动吞吐承诺，不能和万级端到端耗时直接比较。

UbuntuBase24.04.5/glibc2.39的独立rootfs内没有系统Python、FFmpeg或原共享环境，
使用包内CPython与工具；UID1001、只读模型/输入、空缓存、不同CWD和中文空格路径。
既有seccomp禁止新建INET/INET6连接。该rootfs与宿主共享内核；不是Docker实测或新物理机器，
也未验证新机器GPU。缺少 `/proc/cpuinfo` 的提示保留，评分与张量回归均完成。

## 保存与后续

RC2运行包位于 `/var/tmp/deepfake-media-build-20261016/rootfs/app/离线包 rc2/`，
共27,984文件、7,522,748,437B，实际文件SHA清单绑定本次构建。
C13归档 `/mnt/deepfake-archive/C13-offline-runtime-20261016/` 保存136内容文件、
96,372,367B：工具及对应源码、配置/锁、全量分数、逻辑映射、张量、摘要和必要适配器。
权重复用C1，不重复归档完整Python运行库、训练数据或缓存。

新共享环境标准检查：Ruff、严格mypy、901 pytest成功/1跳过、39工具测试和wheel构建通过。
独立代码审查通过。#50另存C14：9个配置/427条精确根映射，C1四工程样本标准infer
与本RC统一入口分数完全一致，C6新根2条可加载；不证明所有历史评分集或训练复现。

本轮没有发布新Release。Docker build/run/save/load、新构建万级容量、平台I/O与赛用许可
仍未完成，#54保持开放；#50的C3历史dirty源码和完整训练恢复材料仍有缺口。
资源自适应与有界OOM由#66继续，不把既有固定分批称为自动资源规划。
