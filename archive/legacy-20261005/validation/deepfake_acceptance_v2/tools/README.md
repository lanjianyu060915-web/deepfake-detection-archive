# Data governance tools

- `audit_candidates.py`：审计规范化源素材与 archive member，不调用鉴伪模型，不依据名称/外观猜标签。
- `build_manifest.py`：从人工审定的固定字段 CSV 重新计算尺寸和 SHA-256，并拒绝 `UNKNOWN`、重复 ID、越界路径和悬空 parent。
- `generate_perturbations.py`：每个可信原图生成 JPEG Q90/Q70/Q50、Scale 0.75/0.5、Gaussian blur 1/2、Brightness -15%/+15% 共 9 个确定性衍生样本。
- `validate_dataset.py`：检查文件、解码、标签、ID、哈希、尺寸、路径、parent、标签继承、统计与完全重复。

所有正式路径都必须相对于 `dataset-root`，禁止绝对路径和 `..` 越界。`build_manifest.py` 默认拒绝覆盖已冻结 manifest。
