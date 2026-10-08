# 配置模板

`.example.json` 是当前 CLI 的起点，不是可直接运行的实验。复制为 `.local.json` 后，替换
`REPLACE_*`、零哈希、示例路径和未确认字段。配置中的相对路径以配置文件所在目录为基准。

## 主线模板

| 阶段 | 模板 |
| --- | --- |
| 数据 | `manifest.example.jsonl`、`sources.example.jsonl`、`audit.example.json` |
| 音频训练 | `train-logmel.example.json`、`train-aasist-l.example.json`、`train-audio-ssl-head.example.json` |
| 视频训练 | `train-video-head.example.json` |
| 推理 | `infer-audio-waveform.example.json`、`infer-audio-ssl-head.example.json`、`infer-video-head.example.json` |
| 导出与交付 | `adapter.example.json`、`package.example.json` |

最短训练流程：

```bash
cp configs/train-logmel.example.json configs/train-logmel.local.json
# 编辑真实清单、来源、路径、哈希、训练参数和新输出目录
uv run --no-sync deepfake-detection-archive train \
  --config configs/train-logmel.local.json
```

训练使用 train/dev，dev 用于早停；holdout 不参与训练或模型选择。每次训练使用新的输出目录。
模板默认 CUDA 与 AMP，只是 8GB 显存的保守起点，实际 batch 和累积步数应由一次 profile 决定。
CPU 调试需同时设置 `device="cpu"` 和 `amp=false`。

音频 SSL 模板的 `input_dim=768`、视频模板的 `input_dim=512` 对应当前候选编码器。更换编码器
时按实际特征维度调整，并登记本地权重。编码器和预处理必须与推理配置一致。训练当前不支持
optimizer/RNG 恢复；中断后在新输出目录重启。

`adapter.example.json` 只描述本地连续分数导出，官方格式公布后再更新。`package.example.json`
需要显式列出最终代码、配置、权重、依赖和许可，不会自动收集资产。

完整命令和字段边界见[开发指南](../docs/development.md)。历史实验的实值配置保留在对应实验
记录中，不应复制成新实验默认值。
