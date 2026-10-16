# 发布资产与使用范围

v0.1.1 为公开非商业研究基线。代码、预训练资源、训练数据和运行依赖分别适用各自条款，
不把整包重新许可为 BSD 或 Apache；正式比赛对外部资源的适用性尚未确认。
项目代码保留作者版权，当前仓库没有授予统一开源许可证。

## 音频权重

`audio-ast-urban.pt`：AST 全12层、末层归一化及鉴伪头在 UrbanSound8K / EnvSDD 上微调，
4秒输入，seed42。SHA256：
`62584d52533fc4204df9c8c37e4f78fd2b5299e5cded13869c23faf4399c3efb`。
由 v0.1.0 的 `8fa61daf42054b29fd61906c8c2fcf679ba4558e80472806197958b013c6e95c`
迁移，所有模型参数不变；配置明确采用居中裁剪/补零至4秒及峰值0.95预处理。

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
`21fdd4e547a5ae7c7a24162eaf2cb526379b7d152e36aa4546891657ae48f115`。
由 v0.1.0 的 `16abc4e9828f73545d60da874a0da16b2a394c63c465896c9ab7e5c5bf465669`
迁移，所有模型参数不变；移除训练机绝对路径，将均值聚合未使用的 `trim_fraction` 置0。

- 基座 timm/resnet18.a1_in1k，revision `491b427b45c94c7fb0e78b5474cc919aff584bbf`，
  保留 Apache-2.0 和模型卡。
- 使用 ViF 的 Panda/HD-VILA 真实视频与 Hunyuan 派生视频训练。保留来源登记、
  Panda 与 HD-VILA R-UDA 原文；R-UDA 将不包含大量原数据的训练模型视为 Results，
  限制 Data 和 Results 在商业产品/服务中使用。本版按非商业研究结果发布。
- 未分发原训练视频；该权重不构成通用生成视频有效性或正式赛用许可的证明。

## 比赛使用边界

包内 `competition_use` 保持 `unconfirmed`。比赛允许开源或自建数据，不等于已经确认
UrbanSound8K 的非商业限制及 Panda/HD-VILA 的 Data/Results 条款适用于本次参赛。
现有资料也不足以断言比赛一定违反这些条款。资源专属许可未落实前，两项权重仅按上述
研究范围提供，不能标为已获正式参赛许可。

## 运行时与演示

`runtime/site-packages/*dist-info/` 保留每项依赖的许可证与归属；
`runtime/python/lib/python3.12/LICENSE.txt` 保留 CPython 及相关许可。
CUDA 用户态库随安装分发中的许可保留，NVIDIA 驱动及 FFmpeg/ffprobe 不在包内。
运行依赖按 wheel 元数据闭包复制，保留其 `dist-info` 和许可证；未手工删除 Torch
声明的 Triton/CUDA 依赖。系统 FFmpeg 的 GPL 构建不随包再分发。
当前构建器另支持独立的 FFmpeg 8.1.3 工具包：禁用 GPL/nonfree/第三方自动探测，
按该构建的 LGPL-2.1-or-later 保存对应完整源码、`LICENSE.md`、`COPYING.LGPLv2.1`、
原版权声明、准确构建配方及二进制哈希，位于 `runtime/media-tools/`。
这不是对服务器现有 GPL 构建重新贴许可，也不改变已发布 v0.1.1 的内容。
演示媒体由 FFmpeg 的正弦波、色块和测试图自行合成，以 CC0 提供，仅用于运行示范，
没有真假标签，不用于计算 EER 或声称模型效果。
