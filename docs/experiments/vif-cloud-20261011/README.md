# ViF 云端受控视觉双种子对照

2026-10-11，Issue #10。复用固定 ViF Panda ZIP、旧 pilot 分组及
ResNet18-a1，在宽高/FPS/整秒时长桶受控的 train/dev 上训练媒体级头。
本轮范围为本地非商业研究，正式比赛许可仍未确认。

## 数据与可比性

- ZIP revision `c9e9b30e47e216f72c50664c43750674a79c0ded`，SHA-256
  `6f47fc9935c6545c7c237a6d15ef8c0e7e26a6826160bb54a6f2f5e83d2c6fae`；
  从共享只读输入导入 4,612 条 Panda 媒体，用时 7.24 秒；Kinetics 媒体未解包。
- 复用 [旧 pilot 清单](../vif-pilot-20261007/manifest.jsonl) 的
  train716/dev154/holdout154 归属，将 full 与 pilot 的登记关系合并后，
  在 ffprobe 前排除旧 holdout 整组154条。其余4,458条探测0失败，用时33.23秒。
- 导入包含全部 Panda 文件的原字节复制和哈希；旧 holdout 没有执行 ffprobe、
  帧解码、特征提取、评分或调参。不能把原字节恢复写成“完全未读取媒体字节”。
- train/dev 内分别按宽、高、FPS及 `floor(duration_seconds)` 匹配真假；
  未分配组件沿用 seed42 的80/20哈希划分，旧归属不变。实际重现
  **train618（real309/fake309）、dev146（real73/fake73）**。
- fake 全部为 HunyuanVideo，来源只有 ViF Panda；train332/dev80 个保守关系组件，
  跨划分组件0、匹配桶真假失衡0。`suspected_filename_component:v1` 是保守约束，
  不是已证实的原始人物/录像身份，不计算独立组置信区间。
- 受控清单 SHA-256：
  `f055a8ebf966ef3dbc3c90351e0c5bcbd7cd9ea8829660afe8a3432f74d542a2`。

元数据对照仍为 train 拟合 `StandardScaler + LogisticRegression(C=1)`，
对同一146条dev连续评分。宽度 EER **50.00%**；宽/高/FPS/连续时长/码率联合
EER **53.424658%**，与 [原受控准备](../vif-control-20261009/README.md) 一致。
配对比较通过样本ID对齐，复核标签与来源，失败保留在完整分母。

## 模型、配置与资源

固定 `timm/resnet18.a1_in1k` revision
`491b427b45c94c7fb0e78b5474cc919aff584bbf`，复用
[既有转换器](../../research/pretrained-resources/prepare_resources.py) 仅移除分类器两键。
转换后 `encoder.pt` 为44,780,495字节，SHA-256
`db6fd0c5cfb2b712d1904bb8b7edcf41a850491bdf42729e4ab485e89f1b6889`，
与历史字节完全一致；120个张量与固定上游逐值一致，CPU strict load通过。
Apache-2.0许可与模型卡保存在权重目录；Panda/HD-VILA数据及派生权重仍限本地非商业研究。

[seed42](seed42.json) / [seed43](seed43.json) 相比原对应配置只改三处路径：
受控清单、缓存及输出目录。4帧、224×224 `center_crop`、冻结编码器FP32；
每帧512维特征，隐藏层64、dropout0.1，帧logit取均值得媒体logit，`fake=1`。
头训练 batch64、workers0、BF16 AMP、AdamW lr0.001、weight_decay0.0001、
梯度裁剪1、最多20轮、patience5、确定性运行；dev早停选最佳权重后FP32评分。

环境为 RTX4090 24GiB、Python3.12.15、torch2.14.1+cu132及现有共享环境。
训练前用4条train视频做真实解码/冻结特征profile：3.46条/秒（含首条warmup），
allocated/reserved峰值69.90/96MiB；头的合成重复特征batch64三步均为有限输出与梯度。
profile只用于确认运行设置，不作为检测效果或正式离线吞吐。

## 结果

| 模型 | best epoch / 完成轮次 | dev EER | 相对联合元数据ΔEER | 成功 / 失败 |
| --- | --- | ---: | ---: | ---: |
| 宽度元数据 | — | 50.000000% | — | 146 / 0 |
| 联合元数据 | — | 53.424658% | — | 146 / 0 |
| ResNet18媒体头 seed42 | 4 / 9，早停 | **20.547945%** | −32.876712个百分点 | 146 / 0 |
| ResNet18媒体头 seed43 | 4 / 9，早停 | **21.917808%** | −31.506849个百分点 | 146 / 0 |

两种子各618条训练、146条dev，特征准备0失败、训练0跳过更新。
使用 [summarize.py](summarize.py) 从完整dev分数重算EER，核对最佳checkpoint哈希，
复用配对比较器确认同一146条的ID、标签与来源一致，共同覆盖率100%。
实际结果和必要资产哈希见 [summary.json](summary.json)；完整产物位置见下表。

seed42 完整实验179.95秒，包含764条首次特征缓存；seed43复用缓存为6.81秒。
其中头训练分别3.68/3.78秒，训练吞吐中位数2,354.85/2,302.78条/秒；
训练器 allocated 峰值均65.77MiB，reserved分别160/128MiB。
训练流程每秒设备采样192条、显存峰值700MiB；采样可能漏过瞬时峰值。
最终从缓存对146条dev评分分别0.0509/0.0518秒，不含解码，不能解释为离线端到端速度。
GPU已于2026-10-11 11:58:30（Asia/Shanghai）释放。

训练源码为 `main@4bbd292`，源码摘要
`b2bcff71b3df67d624c44e0ecc469a38d62811a69015910167a16d8d560cbc51`。
实验中的 `dirty=true` 来自新增的实验配置/文档，主动训练、编码、解码源码未改动。

## 产物与复现

团队资产根目录：`/var/tmp/lanjianyu-issue10-20261011`，组 `dlteam` 可读。
媒体和权重不入Git；详细哈希见 summary 的资产及各轮输入/输出记录。

| 相对资产根目录 | 内容 |
| --- | --- |
| `vif-panda-research/` | 媒体、完整清单、来源/分组证据、导入摘要 |
| `control/` | 受控清单、元数据和连续分数、`data-isolation-audit.json` |
| `resnet18-a1/` | 固定编码器、LICENSE、上游文件及转换证据 |
| `features/` | 两种子共用的764条FP32特征缓存 |
| `seed42/`、`seed43/` | 实验记录、逐样本分数、失败清单、`training/best.pt`及历史 |
| `logs/` | 导入/准备命令与耗时，训练日志、每秒GPU采样、保留的启动失败日志 |
| `profile_video.py`、`profile.json`、`run.sh` | 本轮实际profile和启动脚本 |

当前工作副本使用下列软链接；在其他克隆中指向团队资产或自行重建的任务目录：

```text
data/vif-panda-research -> <assets>/vif-panda-research
models/pretrained/resnet18-a1 -> <assets>/resnet18-a1
cache/vif-cloud-20261011 -> <assets>/features
outputs/experiments/vif-cloud-20261011 -> <assets>
```

导入/控制的精确命令、环境与退出码见 `logs/data-import.execution.json` 和
`logs/data-control.execution.json`。使用现有 `tools/prepare_vif_panda.py --exploratory`
及 `tools/prepare_video_control.py --seed 42 --workers 4`，pilot指向Git内旧清单。
ResNet只恢复既有固定资源，不在推理或训练时下载。实际训练命令：

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/vif-cloud-20261011/seed42.json
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/vif-cloud-20261011/seed43.json
scripts/devctl run python docs/experiments/vif-cloud-20261011/summarize.py
```

重训应把配置的 `output_dir` 改为新的空目录；已有实验不可覆盖，训练不支持optimizer/RNG续跑。
初次profile脚本命名与Python标准库冲突，随后启动器引用了未安装的`/usr/bin/time`；
均发生在正式训练前，已分别更名和使用实验自身计时，失败日志保留，未产生中断训练结果。

## 验证

实际清单的分组隔离、匹配桶、metadata分数及两种子的完整dev分数均已复核。
新增汇总脚本的Ruff、格式和严格mypy通过；标准入口 `scripts/devctl check`
通过 **453项pytest（27.05秒，无跳过）及3项视频控制工具测试**，
主动代码/工具Ruff、68文件格式、严格mypy（33+11文件）、Shell语法和sdist/wheel构建通过。
完整检查日志位于团队资产 `logs/project-check.log`。

## 解释边界与下一步

这里只能说明同来源、单生成器、小开发集上的结果。元数据匹配减少了指定的捷径，
并不证明所有压缩、码率、内容来源偏差已消除。dev用于早停，未作最终holdout验收。
下一步应在允许的独立来源/生成器及已有压缩鲁棒性条件上测试候选，
再决定是否部分微调或增加时序模型；不据此声称本届比赛成绩或七生成器泛化。
