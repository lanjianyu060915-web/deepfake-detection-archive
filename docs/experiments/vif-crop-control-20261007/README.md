# ViF 中心裁剪对照

2026-10-08 完成。与[pad基线](../vif-pilot-20261007/README.md)使用同一
train716/dev154、冻结ResNet18、4帧、轻量头，仅改预处理为center_crop。

| seed | pad dev EER | crop dev EER | crop − pad |
| --- | ---: | ---: | ---: |
| 42 | 32.4675% | 33.7662% | +1.2987个百分点 |
| 43 | 32.4675% | 31.1688% | −1.2987个百分点 |

每组154条出分、0失败。两种子改进方向相反，没有稳定收益，未选择为正式候选。
宽度/来源捷径仍存在；裁剪同时改变内容，不能独立归因于去除补边。

配置见 `configs/train-seed42.json`、`configs/train-seed43.json`；核心结果在
`training/summary.json`。旧逐样本评分、验证脚本和运行记录在 Git `4bbd292`。
下一步使用[受控数据](../vif-control-20261009/README.md)训练内容模型，
不继续扩建本次探索的验证材料。
