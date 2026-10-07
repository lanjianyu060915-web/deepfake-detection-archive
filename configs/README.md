# 实验配置模板

这里的 `.example.json` 是可审查的结构示例，**不是已就绪的真实实验配置**。
所有 `REPLACE_*` 路径、清单、标签、来源以及 64 个 `0` 组成的哈希都必须替换；
零哈希只满足字段形状，不对应任何已验证资产。JSONL 示例故意保留未知标签、许可和
解码状态，不能通过训练审计。模板没有给出模型成绩，也没有批准任何外部权重。

| 模板 | 用途 |
| --- | --- |
| `train-aasist-l.example.json` | AASIST-L 波形训练，16 kHz、64600 samples、单段 |
| `train-logmel.example.json` | 通用音频 log-mel CNN 候选，单段波形训练 |
| `train-audio-ssl-head.example.json` | 冻结本地语音 SSL 编码器，训练媒体级多段特征头 |
| `train-video-head.example.json` | 冻结本地 ResNet18 特征编码器，训练媒体级多帧特征头 |
| `infer-audio-waveform.example.json` | AASIST-L 或 log-mel checkpoint 的开发集推理 |
| `infer-audio-ssl-head.example.json` | 音频 SSL 媒体级头推理 |
| `infer-video-head.example.json` | 视频媒体级头推理 |
| `audit.example.json` | 文件、哈希、来源、划分和解码审计 |
| `adapter.example.json` | 本地连续分数导出；官方接口仍未确认 |
| `package.example.json` | 显式列举本地离线包资产；不自动收集依赖或权重 |
| `manifest.example.jsonl`、`sources.example.jsonl` | 样本与来源登记字段示例 |

从仓库根目录复制模板，在同一 `configs/` 目录填写实值后执行：

```bash
cp configs/train-logmel.example.json configs/train-logmel.local.json
# 编辑 train-logmel.local.json：替换全部占位路径，引用完成审计与划分的真实清单。
uv run --no-sync deepfake-detection-archive train --config configs/train-logmel.local.json
```

配置中的相对路径以**配置文件所在目录**为基准；例如 `../data/` 在本目录中指向
仓库根的 `data/`。复制到其他目录时必须同步调整路径。每次训练使用新的空输出目录。
训练 runner 读取全清单的划分关系，只审计、解码、缓存和训练 `train/dev` 媒体。

模板训练默认 `cuda`、`amp=true`，batch 4 或 8、`num_workers=0` 是 8 GB 环境的
保守起点，是否适合实际数据和编码器需要 profile。CPU 检查须同时设置
`training.device="cpu"`、`training.amp=false`。推理配置只有 `device`，当前没有
`amp` 配置字段。缓存最多 20 GiB，替换为更小上限也可以。

音频 SSL 示例 `input_dim=768`、视频示例 `input_dim=512` 分别对应候选编码器的
特征维数假设；更换编码器必须按实际输出修改。音频本地 bundle 的所有 safetensors
分片及其索引、配置、预处理配置均须登记真实哈希。视觉权重必须是匹配
`timm.create_model(..., pretrained=False, num_classes=0)` 的裸 `state_dict`。
编码器保持冻结；两种特征方案均使用 `media_feature_head_v1` 聚合为每个媒体一个 logit。

完整数据登记、训练、推理、恢复和离线包步骤见 [开发指南](../docs/development.md)。
