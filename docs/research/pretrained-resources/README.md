# 预训练模型来源

下表原有资源是预训练表征模型，鉴伪分类器需要另行训练；文末UFD另含已训练检测头。当前实例可用位置见
[数据与模型资源](../../data.md#预训练模型)。正式外部权重使用规则待比赛平台确认。

| 资源 | 固定上游版本 | 用途 |
| --- | --- | --- |
| [timm/resnet18.a1_in1k](https://huggingface.co/timm/resnet18.a1_in1k/tree/491b427b45c94c7fb0e78b5474cc919aff584bbf) | `491b427b45c94c7fb0e78b5474cc919aff584bbf` | 全帧空间特征 |
| [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960/tree/dba3bb02fda4248b6e082697eee756de8fe8aa8a) | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` | 语音表征，不能代替非语音覆盖 |
| [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/tree/f826b80d28226b62986cc218e5cec390b1096902) | `f826b80d28226b62986cc218e5cec390b1096902` | AudioSet 通用音频表征，支持末端主干微调 |

ResNet18/HuBERT 原模型卡及 Apache-2.0 文本在本目录保留；原文件URL/哈希见
`upstream-resources.json`，历史转换后哈希见 `converted-resources.json`，
许可材料索引见 `license-materials.json`。这些是固定来源记录，不是当前机器就绪报告。
AST 模型卡声明 BSD-3-Clause，本目录保留上游 [BSD 文本](AST-BSD-3-Clause.txt)；
固定模型卡、原始权重及许可随本地 bundle 保存，来源和哈希也登记在 `upstream-resources.json`。

资源准备入口已移至 `tools/prepare_pretrained.py`，不再在 docs 中维护可执行副本：

```bash
scripts/devctl run python tools/prepare_pretrained.py download
scripts/devctl run python tools/prepare_pretrained.py convert
scripts/devctl run python tools/prepare_pretrained.py prepare-ast
```

已有已登记资源直接复用。需要较快获取HuBERT时可用同工具的 `download-hubert-xet`；
它使用已安装的官方Hub客户端、固定revision及显式本地缓存。
下载位于 `cache/pretrained/`，转换权重位于 `models/pretrained/<资源>/`，
本次生成登记位于 `models/pretrained/registry/`，不写入Git文档目录。

ResNet18仅移除 `fc.weight`/`fc.bias`，保存feature-only state_dict；
HuBERT使用 `torch.load(weights_only=True)` 后转为safetensors，配置保持原字节。
来源哈希在获取/转换边界核对；模型加载继续使用本地严格加载，无隐式下载。
每份转换资源附带本目录Apache许可，部署时随资源保留模型卡和来源。
AST 保留原始 safetensors，通过官方本地加载接口初始化；训练后的 checkpoint
包含完整主干、分类头和前端配置，离线推理无需再读取预训练目录。

旧安装/前向探针和脚本执行流水在整理前提交 `4bbd292`，不作为新训练前置门。

## UniversalFakeDetect 视频强对照

使用[作者项目](https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/030495aea3300a8b54c0ec37ec7fe1dd7e63c619)
固定commit `030495aea3300a8b54c0ec37ec7fe1dd7e63c619` 的 `pretrained_weights/fc_weights.pth`，
SHA-256 `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847`。
完整编码器来自OpenAI官方ViT-L/14 TorchScript资源，URL中的可信SHA-256为
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。
检测器使用768维未归一化视觉投影接线性头；不使用文本提示词或CLIP相似度。

`tools/prepare_ufd.py` 在准备阶段核对两份来源哈希，将视觉Q/K/V及投影映射到已安装的
transformers实现，保存完整 `ufd_video_v1` tensor checkpoint。运行时不读取原JIT、
不需要文本编码器或额外CLIP包、不联网。作者评测预处理为RGB、原尺寸CenterCrop224及
CLIP mean/std，**无Resize**；本项目显式使用 `native_center_crop`，视频分数为四帧rawlogit均值。

UFD及CLIP仓库有MIT代码许可；分类头未见独立model card，训练素材和赛用/再分发范围
仍需按各资源处理。现阶段仅作研究候选，图像论文成绩不代表本赛视频EER。
实际权重、配置和评分位置见[资产说明](../../data.md)，结果见[实验索引](../../experiments/README.md)。

## WaveRep 生成视频检测器

[作者仓库](https://github.com/grip-unina/WaveRep-SyntheticVideoDetection/tree/0fd6010759c14b572b7842a28fa9f85fe1ddd2fd)
固定commit `0fd6010759c14b572b7842a28fa9f85fe1ddd2fd`；G1 权重
`https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet/weights_dinov2_G1.ckpt`
（346,160,324 B，作者 md5 `bc9852a7c1d3bfdb60a20b4497e575bb`，SHA-256
`1c7566b26c0b0afe3f5642dd4e454039328d59f512a5d815d5edaa10b544a897`）。
网络为 timm `vit_base_patch14_reg4_dinov2.lvd142m`、`img_size=504`、单输出 logit，端到端训练于 Pyramid Flow
生成视频及其 3D VAE 重建配对样本；权重为裸 state_dict。

`tools/prepare_waverep.py` 核对来源哈希后以 `pretrained=False` 构建同一网络并严格加载，保存自包含
`waverep_video_v1` checkpoint；运行时不联网、不读取 timm 预训练缓存。预处理为 RGB、原尺寸
`CenterCrop(504)`（不足补零）、ImageNet mean/std，**无 Resize**，即 `native_center_crop`；视频分数为帧 logit 均值。

作者代码及权重下载目录均附GRIP-UNINA informational/nonprofit许可，禁止未授权工业/营利用途，公开结果须引用作者。
权重目录许可证仍以software措辞描述，未单独明确比赛授权，赛用范围待确认。
[代码许可](https://github.com/grip-unina/WaveRep-SyntheticVideoDetection/blob/0fd6010759c14b572b7842a28fa9f85fe1ddd2fd/LICENSE.md)、
[权重目录许可](https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet/LICENSE.txt)。
现阶段仅作研究与可选融合候选，赛用范围须由负责人确认；结果见
[生成机制鉴伪](../../experiments/generator-forensics-20261014/README.md)。
