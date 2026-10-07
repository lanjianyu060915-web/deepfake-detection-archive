# AIGC 模型根因与 checkpoint 审计

## 1. 审计范围与方法

本审计针对以下只读对象完成：

- `D:\刑侦查竞赛\face_source_trace_three_module_v0.3_explainable_dev`
- `D:\刑侦查竞赛\face_source_trace_three_module_v0.2_dev`
- `D:\刑侦查竞赛\AIGC检测`
- `D:\刑侦查竞赛\deepfake_acceptance_v2`
- `C:\Users\lanji\AppData\Local\Programs\DeepfakeDetectionSystem`

checkpoint 使用 `torch.load(..., map_location="cpu", weights_only=True)` 读取，不执行训练、不联网下载、不改写模型。结构兼容性通过 `mobilenet_v3_small(weights=None)`、替换最终分类头为 2 类后，与实际 `state_dict` 逐键、逐 shape 比较，并实际执行 `load_state_dict(..., strict=True)` 验证。

## 2. 报错根因

权威安装态源码快照位于：

`C:\Users\lanji\AppData\Local\Programs\DeepfakeDetectionSystem\src\detectors\spatial_mobilenet.py`

旧加载链的关键位置如下：

- 第 31–32 行：`official_weights_path()` 固定通过 `torch.hub.get_dir()/checkpoints/mobilenet_v3_small-047dcff4.pth` 解析官方权重。
- 第 68–73 行：`build_model()` 在构建模型时，先强制检查官方权重文件是否存在并校验 SHA-256。
- 第 74 行：调用 `mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)`。
- 第 75–76 行：将 ImageNet 的 1000 类最终分类层替换为 2 类分类层。
- 第 87–107 行：完成上述步骤后，才读取项目自己的 checkpoint，并执行 `model.load_state_dict(state_dict, strict=True)`。

当前进程未设置 `TORCH_HOME`。实测 `torch.hub.get_dir()` 返回：

`C:\Users\lanji\.cache\torch\hub`

而以下目录和目标文件在审计时均不存在：

`C:\Users\lanji\.cache\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`

因此程序在项目 checkpoint 真正加载前即失败，图片也尚未进入完整空间域、频域和融合推理流程。

错误提示之所以显示“GPU微调空间模型加载失败”，是因为安装态 `src\detectors\fusion_detector.py` 第 118–121 行把 `load_verified_finetuned_model()` 抛出的所有异常统一包装成该文字。这个包装与实际设备无关，CPU 环境也会显示“GPU失败”，属于错误分类。

此外，原适配器 `modules\deepfake_detection\run_deepfake_detection.py` 会先安装 EXE/PYZ 导入器；`pyinstaller_pyz_runtime.py` 第 238–246 行将该 finder 插入到 Python `PathFinder` 之前。因此旧运行链中的 `src.detectors.*` 以 EXE/PYZ 冻结代码为优先来源，磁盘安装目录中的同名 `.py` 仅是可读源码快照。修复不能只修改该快照，必须作用于实际导入链或适配层。

## 3. 官方 MobileNetV3-Small 权重审计

安装态 torchvision 源码 `torchvision\models\mobilenetv3.py` 第 341–362 行确认：

- 枚举：`MobileNet_V3_Small_Weights.IMAGENET1K_V1`
- `DEFAULT = IMAGENET1K_V1`
- 官方 URL：`https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth`

找到的文件属性统一为：

- 文件名：`mobilenet_v3_small-047dcff4.pth`
- 大小：`10,306,551` bytes
- SHA-256：`047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- 安全读取类型：`OrderedDict`
- tensor/key 数：244
- `features.*`：240 个
- `classifier.*`：4 个
- ImageNet 分类头：`classifier.3.weight` shape 为 `[1000, 1024]`
- 总 state numel：2,555,002

工作区中找到的同哈希副本包括：

1. `D:\刑侦查竞赛\换脸寻源3.0\runtime_assets\deepfake_torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`
2. `D:\刑侦查竞赛\face_source_trace_three_module_v0.2_dev\runtime\cache\deepfake\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`
3. `D:\刑侦查竞赛\face_source_trace_three_module_v0.3_explainable_dev\runtime\cache\deepfake\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`
4. `D:\刑侦查竞赛\face_source_trace_three_module_v0.3_explainable_dev\packaging_v0.3\work\dist\app\runtime_assets\deepfake_torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`
5. `D:\刑侦查竞赛\face_source_trace_v0.2_dev\runtime\cache\deepfake\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`
6. `D:\刑侦查竞赛\packaging_work\face_source_trace_three_module_v0.2_installer\runtime\cache\deepfake\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`
7. `D:\刑侦查竞赛\packaging_work\face_source_trace_three_module_v0.2_installer\packaging\dist_v2\app\runtime_assets\deepfake_torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`
8. `D:\刑侦查竞赛\releases\face_source_trace_three_module_v0.2_official\runtime\cache\deepfake\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`

这是 torchvision 官方 ImageNet 预训练权重，不是本项目训练后的二分类权重。

## 4. 项目空间域 checkpoint 结构

权威 checkpoint：

`C:\Users\lanji\AppData\Local\Programs\DeepfakeDetectionSystem\models\spatial_mobilenet_finetuned_gpu\spatial_mobilenet_finetuned_gpu.pt`

- 大小：`6,202,455` bytes
- SHA-256：`5bb4b33936eb3adc631828ea271a466d62070a2b8de1902e5d96122317534794`
- 与 `model_file_sha256.json` 及融合配置记录一致

checkpoint 顶层 keys：

```text
state_dict
label_mapping
backbone
source_baseline_sha256
config
best_epoch
best_val_metrics_at_0_5
unfrozen_feature_indices
environment
classification_threshold
best_val_metrics_at_selected_threshold
```

`state_dict` 实测结果：

- tensor/key 总数：244
- `features.*`：240 个，包含从 `features.0.0.weight` 到 `features.12.*` 的完整骨干参数及 BatchNorm buffers
- `classifier.*`：4 个，包含 `classifier.0.weight/bias` 和 `classifier.3.weight/bias`
- 二分类头 `classifier.3.weight` shape：`[2, 1024]`
- 总 state numel：1,532,052
- 标签映射：`REAL=0`、`FAKE=1`
- backbone：`mobilenet_v3_small`
- checkpoint 内分类阈值：`0.19513355195522308`
- 解冻训练模块索引：`[10, 11, 12]`

安装态 `src\detectors\spatial_finetuning.py` 第 339–358 行进一步证明保存逻辑使用完整的 `model.state_dict()`，并将其中所有 tensor 转到 CPU 后写入 checkpoint；并非仅保存解冻层或差量参数。

结论：该 checkpoint 属于“情况 A”，包含完整 MobileNetV3-Small 骨干、buffers 和项目二分类头。

## 5. `weights=None` 严格加载与等价性验证

验证结构：

```python
model = mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
result = model.load_state_dict(checkpoint["state_dict"], strict=True)
```

实测：

- 目标架构 keys：244
- checkpoint 匹配 keys：244
- missing keys：0
- unexpected keys：0
- shape mismatch：0
- `strict=True`：加载成功

另构造两份模型进行等价性验证：一份由 `weights=None` 初始化，另一份由 `weights=MobileNet_V3_Small_Weights.DEFAULT` 初始化；两者随后严格加载同一个项目 checkpoint。

结果：

- 加载后的全部 state tensor：逐项完全相等
- 固定随机输入输出 logits，两条路径均为 `[-0.5123671889305115, -0.028965938836336136]`
- 最大绝对差：`0.0`

官方 ImageNet 权重在旧流程中会被完整项目 checkpoint 立即覆盖，不影响最终参数或推理输出。正式推理应使用 `weights=None` 构建网络，再严格加载项目 checkpoint。

`MobileNet_V3_Small_Weights.DEFAULT.transforms()` 可继续用于既有 resize、crop 和 ImageNet normalization 预处理定义；获取 transforms 元数据本身不要求下载官方参数文件。

## 6. 是否仍需要官方权重

正式推理不再需要 `mobilenet_v3_small-047dcff4.pth`，也不应再把它列为启动或检测的必需资源。

官方权重仅可能在以下场景有意义：

- 从 ImageNet 初始化重新训练一个新模型；
- checkpoint 将来改为只保存部分/差量参数。

当前交付 checkpoint 不属于上述情况。若未来更换 checkpoint，必须重新执行本审计中的 strict key/shape 校验，不能沿用本结论而不验证。

## 7. 频域模型与融合配置

频域模型：

- 路径：`C:\Users\lanji\AppData\Local\Programs\DeepfakeDetectionSystem\models\frequency_stat_formal\frequency_stat_detector.joblib`
- 大小：`4,129` bytes
- SHA-256：`2ec6828eb2a3151784667eb0c105f07cdb41de6ec2df587d28b2766ceee063eb`
- 模型：`FrequencyStatDetector 1.0.0`
- 特征维度：87
- 分支分类阈值：`0.3246554908447828`

融合配置：

- 路径：`C:\Users\lanji\AppData\Local\Programs\DeepfakeDetectionSystem\models\fusion_formal\fusion_config.json`
- 大小：`1,077` bytes
- SHA-256：`8f76abeabb55da1119b28689e15ccd31eac7f1066e401e766987efbb4603e0b2`
- 方法：`weighted_probability`
- 空间域权重：`0.62`
- 频域权重：`0.38`
- 单图/逐帧判定阈值：`0.36360106194654707`
- 视频聚合：`median`
- 视频判定阈值：`0.4583425493641632`

checkpoint 内空间分支阈值 `0.19513355195522308` 是空间模型元数据；当前融合最终单图结论使用融合配置的 frame threshold，不得混用或擅自调整。

## 8. 运行版本与 CPU/GPU 限制

安装态正式运行载体记录：

- Python：3.12.10
- PyTorch：`2.11.0+cpu`
- torchvision：`0.26.0+cpu`
- scikit-learn：1.9.0
- OpenCV：4.11.0.86

checkpoint 训练环境记录：

- PyTorch：`2.11.0+cu128`
- torchvision：`0.26.0+cu128`
- CUDA：12.8
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- AMP：启用

训练时使用 GPU 不构成推理时的 GPU 依赖。checkpoint 已转换为 CPU tensor 保存，并已通过 CPU map/load 和严格结构验证，可在 CPU 上真实推理。

现有安装态 PyTorch 是 CPU-only build，因此即使机器存在 NVIDIA GPU，该运行载体也不会启用 CUDA；设备状态应显示为 CPU，而不能把模型资源或 checkpoint 加载错误表述成 GPU 故障。若未来提供 CUDA 版运行时，仍应保留 CPU 降级路径，并将“设备状态”和“模型加载状态”分开报告。

结构与等价性辅助审计使用 Python 3.12.10、PyTorch `2.12.0.dev20260408+cu128` 和 torchvision `0.27.0.dev20260407+cu128` 完成；目标结构与安装态 torchvision 0.26 的 MobileNetV3-Small key/shape 一致。最终安装回归仍须以随安装包提供的正式 CPU 运行时为准。

## 9. 最终审计结论

1. 缺失文件是官方 ImageNet 预训练权重，不是项目微调 checkpoint。
2. 项目 checkpoint 完整，能够在 `weights=None` 构建的 2 类 MobileNetV3-Small 上严格加载。
3. 旧流程对官方权重的依赖属于冗余初始化和错误的资源前置检查。
4. 移除该依赖不会改变最终模型参数或输出；实测等价性差异为 0。
5. 正式版应仅把项目空间 checkpoint、频域模型、融合配置及其必要 sidecar 列为 AIGC 必需模型资源。
6. 错误提示必须区分模型资源缺失、checkpoint 解析失败、输入异常、未检测到人脸以及 CUDA 状态，不得统一显示“GPU微调空间模型加载失败”。
