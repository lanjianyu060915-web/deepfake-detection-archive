# 预训练模型来源

这些是预训练表征模型，鉴伪分类器需要另行训练。当前实例可用位置见
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
