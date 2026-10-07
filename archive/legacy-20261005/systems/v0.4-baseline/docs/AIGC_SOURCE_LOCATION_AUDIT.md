# AIGC 源码定位与安装版对应关系审计

审计日期：2026-08-25  
审计方式：只读目录盘点、静态代码/配置检查、SHA-256 复验、PyInstaller 归档结构检查；审计阶段未运行真实模型。

## 结论

1. `D:\刑侦查竞赛\AIGC检测` 是发行交付目录，不是完整开发源码仓库。它只有安装器、两份 DOCX 和数据包。
2. `D:\刑侦查竞赛\deepfake_acceptance_v2` 是冻结基线、数据治理和黑盒阅卷工程，无检测权重、无业务 UI，不是安装程序源码。
3. 当前可验证的权威运行载体是 `C:\Users\lanji\AppData\Local\Programs\DeepfakeDetectionSystem` 中的安装态源码快照、PyInstaller EXE/PYZ、模型和配置。最初开发工程元数据曾指向 `D:\信安大作业`，但该目录现已不存在，因此不能声称找到了完整可重建的原始开发仓库。
4. 当前安装版与 `AIGC检测\集成系统\深度伪造识别系统_通用版_Setup(2).exe` 具有高置信发行对应关系：二者产品名、版本 1.0.1、发布者、Inno Setup 安装记录和安装位置一致。由于安装器没有被成功完整解包，不能夸大为逐字节供应链证明。
5. 本项目复用 `face_source_trace_v0.2_dev\modules\deepfake_detection` 的冻结适配器。它不重写模型，而是只读加载安装主 EXE 的 PyInstaller PYZ 权威代码和安装目录资源。

## 关键证据

### 发行物与安装记录

- 安装器：`D:\刑侦查竞赛\AIGC检测\集成系统\深度伪造识别系统_通用版_Setup(2).exe`
  - 大小：214,244,624 字节
  - SHA-256：`347ed97e43afdb1032109928b6731880d4eaa1b98a5d95f0fb7f6ebc9bdd61f2`
  - 产品版本：1.0.1；Inno Setup；未数字签名。
- 主程序：`C:\Users\lanji\AppData\Local\Programs\DeepfakeDetectionSystem\深度伪造识别系统.exe`
  - 大小：54,645,537 字节
  - SHA-256：`bc67e52b77fa0559d4d6fe04f024dcdb4a8edf43128f80f7d70947e3da490203`
  - PyInstaller x64 GUI，Python 3.12，归档 cookie 记录 `pyvers=312` 与 `python312.dll`。
- 当前 HKCU 卸载项记录产品 `深度伪造识别系统 1.0.1`、Inno Setup 7.0.2、同一发布者和上述安装位置。
- 审计时原安装服务仍在 `127.0.0.1:8501` 运行，`/_stcore/health` 返回 HTTP 200 / `ok`。

### 冻结基线

`deepfake_acceptance_v2\frozen_baseline\VERIFY_BASELINE.ps1 -Json` 在 2026-08-25 只读复验结果：

- 代码清单：36/36 匹配；
- 模型清单：10/10 匹配；
- 总计：50/50 PASS；
- 模式：`read_only_sha256_no_model_inference`。

该结果证明当前安装态未偏离 2026-08-20 冻结状态，不等同于安装器签名或完整供应链证明。

### 适配器真实性

- `run_deepfake_detection.py` SHA-256：`fd74aaa13996f735f9bbbe86afd6d2919c3bee5c0722c019c9f3792be6aeab25`
- `pyinstaller_pyz_runtime.py` SHA-256：`c6f3836d4c9252cbb2fc315acfd97e78b5f0b8bf10941e607bea996bcb86d836`
- 适配器强制 Python 3.12、检查 EXE/空间模型/频率模型/融合配置/官方 MobileNet 权重，随后从 `src.pipeline` 导入原 `DeepfakeDetectionPipeline` 并调用 `detect_media`。
- `LOW_CONFIDENCE` 与 `INSUFFICIENT_EVIDENCE` 保持不确定状态，不映射为 REAL。

## 模型、预处理与输出语义

### 空间分支

- 架构：ImageNet 预训练 MobileNetV3-Small，末层替换为二分类层。
- 权重：`models\spatial_mobilenet_finetuned_gpu\spatial_mobilenet_finetuned_gpu.pt`
- SHA-256：`5bb4b33936eb3adc631828ea271a466d62070a2b8de1902e5d96122317534794`
- 分支元数据阈值：`0.19513355195522308`；最终图片判断不使用该阈值，而使用融合阈值。

### 频率分支

- 模型：FrequencyStatDetector 1.0.0，87 维 FFT/DCT/梯度/Laplacian/熵等特征，StandardScaler + LogisticRegression。
- 权重：`models\frequency_stat_formal\frequency_stat_detector.joblib`
- SHA-256：`2ec6828eb2a3151784667eb0c105f07cdb41de6ec2df587d28b2766ceee063eb`
- 分支阈值：`0.3246554908447828`。

### 最终融合

- `fusion_score = 0.62 * spatial_fake_probability + 0.38 * frequency_fake_probability`
- 图片/帧阈值：`0.36360106194654707`
- 视频聚合：有效帧融合分数中位数；视频阈值 `0.4583425493641632`
- `REAL=0`、`FAKE=1`
- score 是当前冻结模型对 FAKE 的风险分数，不是总体准确率、法庭置信度或司法结论。

### 图像预处理

- `numpy.fromfile + cv2.imdecode` 支持 Windows 中文路径；
- OpenCV Haar 正面人脸检测，灰度直方图均衡，最大脸；无脸时使用 BORDER_REPLICATE 扩边重试；
- 人脸框四周扩展 25%，裁剪为 224×224；
- 空间分支使用 MobileNetV3 默认 ImageNet 变换；频率分支使用同一裁剪生成 87 维特征。

## 集成决策

安装目录保持只读。新系统以隔离 Python 3.12 子进程/持久本地工作进程调用冻结适配器，前端、身份环境和底图环境不强行合并。任何模型、预处理、权重、融合比例、阈值或标签映射变化都应视为新模型基线，并重新验收。
