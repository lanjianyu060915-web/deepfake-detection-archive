# 深度伪造识别系统（Windows CPU 自包含版）

双击 `深度伪造识别系统.exe` 即可启动。程序不要求安装 Python、PyTorch 或创建虚拟环境。

运行要求：Windows 10/11 64 位。首次启动会校验正式模型与配置，随后启动本机 Streamlit 服务并打开默认浏览器。需要停止时，请双击同目录的 `停止深度伪造识别系统.exe`；停止程序只会结束经状态文件验证的本系统进程树。

检测结果与日志位于：

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
