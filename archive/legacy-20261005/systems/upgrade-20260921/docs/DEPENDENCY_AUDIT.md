# 三模块依赖审计与隔离策略

审计日期：2026-08-25

## 实际环境

| 模块 | Python | 核心依赖 | 当前设备 |
| --- | --- | --- | --- |
| AIGC 鉴伪 | 3.12.10 | torch 2.11.0+cpu、torchvision 0.26.0+cpu、OpenCV 4.11.0、NumPy 2.5.1、scikit-learn 1.9.0、joblib 1.5.3 | CPU；安装包无 CUDA |
| 身份搜索 | 3.12.10 | InsightFace 0.7.3、ONNX Runtime 1.20.1、OpenCV 4.10.0、NumPy 1.26.4 | CPUExecutionProvider；无 CUDA provider |
| 底图搜索 | 3.12.10 | OpenCV 4.14.0、NumPy 2.5.2、Pillow 12.3.0 | CPU |
| Streamlit UI | 3.14.3 | Streamlit 1.58.0 | 不执行模型推理 |

主机有 NVIDIA GeForce RTX 5060 Laptop GPU，但当前三个冻结运行环境均没有可用的 CUDA 推理栈。AIGC 安装包是正式 CPU 通用版；身份 ONNX Runtime 也没有 CUDAExecutionProvider。因此 UI 必须显示当前实际设备为 CPU，不能只因主机存在 NVIDIA GPU 就显示 CUDA 可用。

## 冲突判断

- Python 3.14 UI 无法直接加载 Python 3.12 的 PyInstaller PYZ/二进制扩展。
- 身份模块固定 NumPy 1.26.4，底图与 AIGC 使用 NumPy 2.x；强行合并会改变已验证依赖组合。
- AIGC Torch/Torchvision、身份 InsightFace/ONNX Runtime 与底图 OpenCV 锁定版本不同。
- `D:\python\python.exe` 常规 site 初始化当前存在外部 `.pth` 编码问题，但冻结适配器使用 `-S -B`，绕过 site 并只读加载安装 EXE/PYZ，因此该问题不影响既有适配路径。

## 正式策略

采用“Streamlit 主程序 + 本地隔离工作进程”的结构：

1. UI 仅负责任务、状态、进度和证据展示；
2. AIGC 使用 Python 3.12 `-S -B` 与冻结适配器；
3. 身份搜索使用 A 环境；
4. 底图搜索使用 B 环境或既有打包 worker；
5. 每个分支有独立 timeout、错误 JSON 和日志；一个分支失败不阻断其他分支；
6. 可用时启动持久 AIGC/身份 worker 缓存模型；启动失败时降级为原有单次隔离子进程，且在 UI 明示降级，不伪报缓存状态。

该策略保留所有冻结算法和依赖，代价是部署时必须同时具备对应运行时。`start.bat` 和环境检查会在打开 UI 前给出清晰检查结果。
