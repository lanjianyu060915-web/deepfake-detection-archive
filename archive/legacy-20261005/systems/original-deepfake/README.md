# 深度伪造识别系统（Windows CPU 自包含版）

双击 `深度伪造识别系统.exe` 即可启动。程序不要求安装 Python、PyTorch 或创建虚拟环境。

运行要求：Windows 10/11 64 位。启动器会校验正式模型与配置，选择 8501—8599 范围内的可用端口，以同一个冻结 EXE 启动 Streamlit，并在健康检查通过后打开默认浏览器。

需要停止时，请双击同目录的 `停止深度伪造识别系统.exe`。停止程序会核验状态文件中的 EXE 路径、进程创建时间和程序目录，只结束本系统创建的进程树。

检测结果、状态和日志位于：

```text
%LOCALAPPDATA%\DeepfakeDetectionSystem\
├─ outputs\
├─ logs\
└─ runtime\
```

命令行自测：

```text
深度伪造识别系统.exe --self-test
```

自测结果写入 `%LOCALAPPDATA%\DeepfakeDetectionSystem\runtime\self_test_last.json`。系统输出仅用于深度伪造风险筛查和辅助研判，不应单独作为司法鉴定或案件定性依据。
