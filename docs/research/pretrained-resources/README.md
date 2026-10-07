# 固定预训练资源与离线转换

这两项资源是已授权获取的候选冻结编码器，不是训练完成的鉴伪分类器。真实数据效果、
跨域泛化与官方外部权重规则须另外验证。

完整官方 Apache 2.0 文本归档于 `Apache-2.0.txt`，并原字节复制到两个本地资源目录的
`LICENSE`，供后续本地转移时随附。来源、获取时间与哈希见 `license-materials.json`。
附带许可文本不代表赛方已批准使用外部权重；正式提交前仍须核对官方资源规则。

| 资源 | 固定版本 | 来源许可 | 本地用途 |
| --- | --- | --- | --- |
| [timm/resnet18.a1_in1k](https://huggingface.co/timm/resnet18.a1_in1k/tree/491b427b45c94c7fb0e78b5474cc919aff584bbf) | `491b427b45c94c7fb0e78b5474cc919aff584bbf` | 官方模型卡声明 Apache-2.0 | 全帧通用视觉表征，不要求人脸 |
| [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960/tree/dba3bb02fda4248b6e082697eee756de8fe8aa8a) | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` | 官方模型卡声明 Apache-2.0 | 冻结语音表征；不能据此宣称音乐、环境声覆盖 |

`prepare_resources.py` 保存精确转换过程。所有原文件位于 Git 忽略的 `cache/pretrained/`，
转换权重位于 `models/pretrained/`。执行时先验证脚本内固定的上游 SHA-256 和已知字节数，
再读取权重。`.bin` 仅使用 `torch.load(weights_only=True)` 反序列化。

本次实际转换的脚本原字节另归档于 `source/conversion-script.py`，其 SHA-256 与
`converted-resources.json` 的 `_provenance.script_sha256` 一致。当前入口仅修复
HuBERT 加载信息中 `set` 到 JSON 列表的诊断序列化；权重转换逻辑没有变化。

```bash
.venv/bin/python docs/research/pretrained-resources/prepare_resources.py download
.venv/bin/python docs/research/pretrained-resources/prepare_resources.py convert
.venv/bin/python docs/research/pretrained-resources/prepare_resources.py probe
```

本次大文件 HuBERT 使用已安装官方 Hub/Xet 客户端加速，仅下载声明的三个固定文件；
原 urllib 阶段仅处理 ResNet，完成后才进行全量来源登记。Xet 分支必须显式限制缓存：

```bash
HF_HOME="$PWD/cache/pretrained/hf-home" \
HF_HUB_CACHE="$PWD/cache/pretrained/huggingface-hub" \
HF_XET_CACHE="$PWD/cache/pretrained/xet" \
HF_HUB_DISABLE_TELEMETRY=1 \
.venv/bin/python docs/research/pretrained-resources/prepare_resources.py download-hubert-xet
```

GPU 探针额外使用 `HF_HUB_OFFLINE=1`，现有加载器本身也显式传入 `local_files_only=True`。

ResNet18 仅移除 `fc.weight`、`fc.bias` 后保存 bare state_dict；由现有
`FrozenVisionEncoder('resnet18', ...)` 以 `pretrained=False` 构造再 strict 加载。
HuBERT 不重命名或移除权重键，只将连续 tensor 保存为 metadata `format=pt` 的
safetensors，配置文件保持原字节；由现有 `FrozenAudioEncoder` 离线严格加载。

生成证据：`upstream-resources.json` 保存原 URL、版本、大小和哈希；
`converted-resources.json` 保存转换后的哈希、转换说明、脚本哈希与依赖版本；
`loading-verification.json` 保存真实加载信息及随机输入 CUDA 前向、冻结状态和显存。
模型卡另按固定版本保留原始副本。若严格加载失败，资源不能计为就绪，也不允许通过
忽略未知键来使验证通过。

GPU 探针只验证预训练资源的工程兼容性，不执行优化步骤，也不报告 EER、准确率或比赛成绩。

## 本次结果

两份原资源均已通过固定大小/哈希校验并转换，现有加载器报告 missing、unexpected、
mismatched keys 全部为零。实际命令分支、缓存隔离、诊断修复与工具版本见
`execution-record.json`。

| 资源 | 转换位置 | 输出特征 | 随机输入探针峰值 allocated / reserved |
| --- | --- | --- | --- |
| ResNet18 | `models/pretrained/resnet18-a1/encoder.pt` | `[1, 512]` | 55,282,176 / 71,303,168 bytes |
| HuBERT | `models/pretrained/hubert-base-ls960/model.safetensors` | `[1, 768]` | 419,782,656 / 482,344,960 bytes |

探针输入分别为 224×224 RGB 全帧和 16kHz 一秒音频，全部参数冻结、输出有限；
单次冷启动耗时仅留作诊断，不能解释为稳定吞吐。用于实际运行的权重哈希从
`converted-resources.json` 读取，禁止以运行时临时计算的新哈希代替可信记录。
