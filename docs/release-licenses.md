# 发布资产与使用范围

v0.1.0 为公开非商业研究基线。代码、预训练资源、训练数据和运行依赖分别适用各自条款，
不把整包重新许可为 BSD 或 Apache；正式比赛对外部资源的适用性尚未确认。
项目代码保留作者版权，当前仓库没有授予统一开源许可证。

## 音频权重

`audio-ast-urban.pt`：AST 全12层、末层归一化及鉴伪头在 UrbanSound8K / EnvSDD 上微调，
4秒输入，seed42。SHA256：
`8fa61daf42054b29fd61906c8c2fcf679ba4558e80472806197958b013c6e95c`。

- 基座 MIT/ast-finetuned-audioset-10-10-0.4593，revision
  `f826b80d28226b62986cc218e5cec390b1096902`，保留 `AST-model-card.md` 与 BSD-3-Clause。
- [EnvSDD](https://huggingface.co/datasets/EnvSDD/EnvSDD/tree/b0832726deb55484dbce869c6f09c41f7e9e5fbb)
  集合 CC BY 4.0；原 UrbanSound8K 为 CC BY-NC 3.0。随包保留两份许可全文、
  固定来源和逐录音 `FREESOUNDCREDITS.txt`，本版按这些条件用于非商业研究。
- UrbanSound8K 归属：Justin Salamon、Christopher Jacoby、Juan Pablo Bello，
  *A Dataset and Taxonomy for Urban Sound Research*，ACM Multimedia 2014。
  本项目修改是训练鉴伪分类器，未随包发布训练录音或生成音频。

该模型主要验证环境音，对语音、音乐及真实混合内容的泛化未获证明。
AIME 音乐混合 checkpoint 不进入公开附件；内部研究仍保留，不因发布替换训练结论。

## 视频权重

`video-resnet18.pt`：ResNet18 layer4 与鉴伪头微调、4帧均值聚合，seed42。SHA256：
`16abc4e9828f73545d60da874a0da16b2a394c63c465896c9ab7e5c5bf465669`。

- 基座 timm/resnet18.a1_in1k，revision `491b427b45c94c7fb0e78b5474cc919aff584bbf`，
  保留 Apache-2.0 和模型卡。
- 使用 ViF 的 Panda/HD-VILA 真实视频与 Hunyuan 派生视频训练。保留来源登记、
  Panda 与 HD-VILA R-UDA 原文；R-UDA 将不包含大量原数据的训练模型视为 Results，
  限制 Data 和 Results 在商业产品/服务中使用。本版按非商业研究结果发布。
- 未分发原训练视频；该权重不构成通用生成视频有效性或正式赛用许可的证明。

## 运行时与演示

`runtime/site-packages/*dist-info/` 保留每项依赖的许可证与归属；
`runtime/python/lib/python3.12/LICENSE.txt` 保留 CPython 及相关许可。
CUDA 用户态库随安装分发中的许可保留，NVIDIA 驱动及 FFmpeg/ffprobe 不在包内。
演示媒体由 FFmpeg 的正弦波、色块和测试图自行合成，以 CC0 提供，仅用于运行示范，
没有真假标签，不用于计算 EER 或声称模型效果。
