# 配置模板

`.example.json` 是当前 CLI 的起点，不是可直接运行的实验。复制为 `.local.json` 后，替换
`REPLACE_*`、零哈希、示例路径和未确认字段。配置中的相对路径以配置文件所在目录为基准。

## 主线模板

| 阶段 | 模板 |
| --- | --- |
| 数据 | `manifest.example.jsonl`、`sources.example.jsonl`、`audit.example.json` |
| 音频训练 | `train-logmel.example.json`、`train-aasist-l.example.json`、`train-audio-ssl-head.example.json`、`train-logmel-doppelganger.example.json` |
| 视频训练 | `train-video-head.example.json` |
| 推理 | `infer-audio-waveform.example.json`、`infer-audio-ssl-head.example.json`、`infer-video-head.example.json` |
| 导出与交付 | `adapter.example.json`、`package.example.json` |

最短训练流程：

```bash
cp configs/train-logmel.example.json configs/train-logmel.local.json
# 编辑真实清单、来源、路径、哈希、训练参数和新输出目录
scripts/devctl run python -m deepfake_detection_archive train \
  --config configs/train-logmel.local.json --check-config
scripts/devctl run python -m deepfake_detection_archive train \
  --config configs/train-logmel.local.json
```

`--check-config` 只读预检可在 CPU 环境检查 CUDA 配置，不分配模型、不加载权重、不下载、
不解码，也不创建输出或缓存目录。它复用训练的数据审计，检查配置及模型/媒体字段、
train/dev 双类与受约束 dev 域、全清单的已知分组泄漏、所选 train/dev 的实际字节和来源许可，
以及本地权重/SSL bundle 的路径与 SHA-256；holdout 只读划分元数据，不读媒体。
标准输出为 JSON，预检不通过返回状态码 2，配置/文件异常在标准错误中给出原因。
`passed=true` 仅代表这些检查通过；`unchecked` 列出未检查项，包括实际解码、缓存张量、
模型构建和权重兼容性、编码器维度、CUDA/显存/吞吐、可写权限/剩余磁盘及域基线来源。
AST 预训练资源还需 `model.safetensors` 文件名与同目录的 `config.json`；当前配置只登记
权重哈希，预检会明确提示这个 `config.json` 未有可信哈希，不能据此宣称其完整性已验证。
它不是训练或效果验证，正式启动仍按当时的实际资产重新检查。只想预检时保留该标志；
去掉后才进入训练。最小离线回归可运行 `scripts/devctl run pytest -q tests/test_preflight.py`。

训练使用 train/dev，dev 用于早停；holdout 不参与训练或模型选择。每次训练使用新的输出目录。
模板默认 CUDA 与 AMP，batch 和累积步数仅作起点，按当前 GPU 的实际 profile 调整。
CPU 调试需同时设置 `device="cpu"` 和 `amp=false`。

音频 SSL 模板的 `input_dim=768`、视频模板的 `input_dim=512` 对应当前候选编码器。更换编码器
时按实际特征维度调整，并登记本地权重。编码器和预处理必须与推理配置一致。训练当前不支持
optimizer/RNG 恢复；中断后在新输出目录重启。

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
