# ViF 尺寸/时长受控数据与元数据基线

数据于2026-10-09准备。受控数据与元数据基线已完成，视觉训练结果待交付，
后续在云端按下列固定配置推进，见[视频任务 #10](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/10)。本轮把现有视频资产转成可训练的受控数据，
不增加新数据下载，也不访问原 pilot 的 154 条 holdout 媒体或分数。

## 做法与数据

工具合并 full manifest 和 pilot 的已登记关系；保留原 train/dev 归属，排除所有接触旧
holdout 的组件。其余组件按 seed42 的稳定哈希划为 80% train / 20% dev。
实际排除154条，余下4,458条只读取 ffprobe 媒体头，0失败；没有解码帧或使用GPU。
原 full 的3,588条未用样本现在属于扩展探索数据，不构成全新的最终留出。

匹配在 train/dev 内分别执行，按精确宽、高、FPS组成桶，再取每桶真假较少类的数量；
时长版本增加 `floor(duration_seconds)` 的1秒桶。宽高精确匹配同时匹配宽高比。
桶内选择只依赖固定种子与样本ID，未使用检测器分数。选择可以舍弃组件内部分样本，
所有保留样本仍保持原组件划分。各版本实际检查均为0个组件跨train/dev。
这些是防已知关系泄漏的组件，不代表已确认的独立人物或采集单位。

元数据模型只在train拟合 StandardScaler + LogisticRegression(C=1)，dev报告EER。
联合模型输入宽、高、FPS、连续时长和码率，未将元数据分数融合进内容模型。

| 数据版本 | train | dev | 宽度 EER | 联合元数据 EER |
| --- | ---: | ---: | ---: | ---: |
| 扩展、未匹配 | 3,614 | 844 | 22.65% | 32.70% |
| 精确宽高/FPS匹配 | 664 | 164 | 50.00% | 40.24% |
| 加1秒时长桶匹配 | 646 | 158 | 50.00% | 50.63% |
| HunyuanVideo单生成器、加时长匹配 | 618 | 146 | 50.00% | 53.42% |

每一行的train与dev各自真假各半。元数据模型接近50%只说明这两个简单对照没有得到
有效区分，不证明编码、码率非线性关系、来源或内容偏差都已消失。

## 匹配不足与生成器诊断

扩展版本 train/dev 都覆盖7个生成器。但加时长匹配后，train的323条fake中309条是
HunyuanVideo，另有13条Wan2.1和1条CogVideoX-I2V；dev的79条fake中73条是
HunyuanVideo，其余6条分散在4个生成器。这是数据支持不足，不是通用七生成器基准。
因此额外提供明确的HunyuanVideo受控子集，优先验证内容模型能否在该条件下学习。

每个生成器还生成了一个开发诊断：train剔除含该生成器fake的整个组件，dev只保留该
生成器fake及其对应真实母项，再做同样的宽高/FPS/时长匹配。

- CogVideoX-I2V、Wan2.2-T：匹配后dev为空。
- CogVideoX-T、Hunyuan-I2V：dev各2条；Wan2.1、Wan2.2-I：dev各4条。
- HunyuanVideo：dev112条，但train仅18条，仅剩Wan2.1假样本；联合元数据EER51.79%。

这些诊断暴露了尺寸共同支持区太窄，暂不用于报告可靠未知生成器泛化。全体媒体只有
一个登记来源 ViF Panda；没有可执行的跨来源留出。补其他来源/可匹配尺寸数据，或对
全生成器采用对称预处理并另做来源控制，仍是后续工作。

## 复现与下一步训练

从仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 scripts/devctl run python tools/prepare_video_control.py \
  --manifest data/vif-panda-research/manifest.jsonl \
  --pilot outputs/experiment-planning/vif-pilot/manifest.jsonl \
  --data-root data/vif-panda-research/media \
  --output outputs/experiments/vif-control-20261009
```

同一输出目录复用逐样本元数据；结果清单、逐样本连续分数和完整计数都在该忽略目录。
一份`summary.json`记录输入清单哈希和全部基线结果，没有复制历史脚本或另建证明树。
本机已准备两种子训练配置，沿用冻结ResNet18、4帧、center_crop、同一轻量头和训练参数：

```bash
scripts/devctl run python -m deepfake_detection_archive train \
  --config docs/experiments/vif-control-20261009/train-hunyuan_geometry_duration-seed42.json
```

对应seed43配置一并纳入Git，配置相对路径均以配置所在目录为基准。
先运行Hunyuan单生成器受控实验，再决定是否投入微调；配置纳入Git，资产按[数据说明](../../data.md)接入，
准备工具生成划分和元数据基线；训练配置单独保存在本目录。旧pilot holdout继续保留，未经触碰。
