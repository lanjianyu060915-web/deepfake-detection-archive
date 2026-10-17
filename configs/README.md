# 配置模板

`.example.json` 是当前 CLI 的起点，不是可直接运行的实验。复制为 `.local.json` 后，替换
`REPLACE_*`、零哈希、示例路径和未确认字段。配置中的相对路径以配置文件所在目录为基准。

## 主线模板

| 阶段 | 模板 |
| --- | --- |
| 数据 | `manifest.example.jsonl`、`sources.example.jsonl`、`audit.example.json` |
| 音频训练 | `train-logmel.example.json`、`train-aasist-l.example.json`、`train-audio-ssl-head.example.json`、`train-logmel-doppelganger.example.json` |
| 视频训练 | `train-video-head.example.json` |
| 推理 | `infer-audio-waveform.example.json`、`infer-audio-ssl-head.example.json`、`infer-audio-ssl-antispoofing.example.json`、`infer-video-head.example.json`、`infer-video-fusion.example.json` |
| 导出与交付 | `adapter.example.json`、`package.example.json` |

最短训练流程：

```bash
cp configs/train-logmel.example.json configs/train-logmel.local.json
# 编辑真实清单、来源、路径、哈希、训练参数和新输出目录
scripts/devctl run python -m deepfake_detection_archive train \
  --config configs/train-logmel.local.json
```

CPU机器需要在启动前查看CUDA配置时，可单独运行
`train --config ... --check-config`。它复用实际训练的配置与train/dev审计，只读本地数据、
资源元数据和权重哈希，不分配模型、解码、下载或创建输出；holdout只检查划分元数据。
这一步是可选诊断，不是每轮训练前的额外门槛。详细范围见[开发指南](../docs/development.md#训练)。

训练使用 train/dev，dev 用于早停；holdout 不参与训练或模型选择。每次训练使用新的输出目录。
模板默认 CUDA 与 AMP，batch 和累积步数仅作起点，按当前 GPU 的实际 profile 调整。
CPU 调试需同时设置 `device="cpu"` 和 `amp=false`。

音频 SSL 模板的 `input_dim=768`、视频模板的 `input_dim=512` 对应当前候选编码器。更换编码器
时按实际特征维度调整，并登记本地权重。编码器和预处理必须与推理配置一致。训练当前不支持
optimizer/RNG 恢复；中断后在新输出目录重启。

`infer-video-fusion.example.json` 对应固定 CLIP43＋WaveRep 研究候选。单个本地
`video_fusion_v1` checkpoint 包含两路权重、CLIP MLP 和固定融合参数，不需要额外 encoder。
配置保留 CLIP 的 4 帧 224 原生中心裁剪；评分器使用相同时间点和视频流，另按 WaveRep 的
504 原生中心裁剪与归一化评分。`batch_size` 只控制独立帧主干批量，不改变帧数或融合。
准备工具为 `tools/prepare_video_fusion.py`；WaveRep 正式赛用许可未确认，仅供研究。

CLIP最后block适配的完整权重使用 `clip_adapt_video_v1`，推理设置
`model_kind="video_frames"`、4帧224 `native_center_crop`，不需要额外encoder。
实际配置及固定预算训练入口见[适配实验](../docs/experiments/video-clip-adapt-20261016/README.md)；
本轮训练通过任务适配器调用现有trainer，公共train CLI暂不接受该架构。

AudioSet AST部分主干微调的实值配置见[环境音AST实验](../docs/experiments/audio-ast-20261011/README.md)。
它复用波形推理模板，checkpoint自包含前端配置和全部权重；预训练来源只在初始化时加载。
新4秒 AST 实验按[开发指南](../docs/development.md#训练)显式设置
`waveform_preparation="center_pad_64000_peak095_v1"`，训练与推理共用居中裁剪、
对称补零和峰值0.95处理；旧 checkpoint 默认 `legacy` 以保持历史结果。
AST可设置 `training.frequency_mask_width` / `time_mask_width`，默认0关闭。
各样本遮挡宽度从 `[0, 配置值)` 抽取，只作用于训练，不改变缓存或推理。
两档实际对照见[AST时频遮挡实验](../docs/experiments/audio-ast-regularization-20261012/README.md)。

`balanced_sampling=true` 按 real/fake 平衡训练抽样。多来源规模差异较大时，同时设置
`balance_sources=true`，使已存在的来源与类别组合等权；只影响 train，不改变 dev/holdout。
音乐与环境音的实际对照配置见[音乐混合训练](../docs/experiments/audio-music-20261011/README.md)。

`adapter.example.json` 只描述本地连续分数导出，官方格式公布后再更新。`package.example.json`
需要显式列出最终代码、配置、权重、依赖和许可，不会自动收集资产。

完整命令和字段边界见[开发指南](../docs/development.md)。历史实验的实值配置保留在对应实验
记录中，不应复制成新实验默认值。

`infer-audio-ssl-antispoofing.example.json` 是官方 XLS-R+AASIST LA 语音专家完整权重，
与仅训练 SSL 特征头的模板不同。需要本地 `LA_model.pth`、固定上游 `model.py` 和
`xlsr_cfg.json` 三个资源；模板已固定本轮权重和配置哈希，不隐式下载。
运行、逐样本失败记账和离线搬移说明见[实验](../docs/experiments/speech-ssl-20261014/README.md#标准离线推理)。
