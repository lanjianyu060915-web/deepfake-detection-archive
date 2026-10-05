# v0.4.0 正式安装构建链

本目录与旧 `packaging` 完全隔离。正式链保持前端、身份检索和底图检索
三套 PyInstaller onedir 运行时，AIGC 继续使用受控的便携 Python 与已授权
深伪运行资源。运行任务、报告、日志和缓存统一写入当前用户的
`LOCALAPPDATA/FaceSourceTrace`，安装目录只保存只读程序和资源。

## 构建输入

构建脚本按“显式参数 → 环境变量 → 本仓库相对资源目录”的顺序发现输入，
不会依赖开发者用户名或固定盘符。推荐显式提供：

- `FrontendPython`：含固定版本 Streamlit/PyInstaller 的构建 Python；
- `IdentityPython`：现有身份算法环境 Python；
- `SourcePython`：现有底图算法环境 Python；
- `DataSourceRoot`：含 `identity_library` 和 `source_library`；
- `IdentityModelSource`：经授权的 `buffalo_l` 五文件目录；
- `DeepfakeRuntimeSource` 与其经复核的逐文件清单；
- `PortablePythonSource` 与 MobileNetV3 基础权重；
- `IsccPath`：Inno Setup 编译器。

对应环境变量见 `build_installer_v0.3.ps1`。带中文或空格的路径必须作为
一个 PowerShell 参数传入，脚本内部全部使用 `-LiteralPath` 或参数数组。

## 构建顺序

1. 运行 `scripts/Test-PackagingStatic.ps1`。
2. 如深伪资源发生变化，以显式批准开关生成新的输入清单并人工复核。
3. 运行 `build_installer_v0.3.ps1` 完成三套 onedir、受控装配、哈希更新和
   Inno Setup。
4. 在隔离的“中文 空格”目录运行 `acceptance/phase1_install_start.ps1`。
5. 完成真实单模块、综合分析与报告导出后，保存业务证据 JSON。
6. 运行 `acceptance/phase2_restart_uninstall.ps1` 验证重启、报告和卸载。

正式前端规格不会收入 `mock_results`、`mock_task` 或前端演示数据。正式配置
的 `mock_enabled` 为 `false`；Mock 仅允许开发源码显式启用，不能作为安装版
资源或默认结果入口。

## 可执行命令

在本目录启动 PowerShell 7，先固定已经完成人工授权复核的两个大型运行资源：

```powershell
& .\scripts\New-ApprovedResourceManifest.ps1 `
  -SourceRoot $env:FSTS_DEEPFAKE_RUNTIME_SOURCE `
  -OutputPath .\resources\manifests\deepfake_runtime.sha256.json `
  -Profile deepfake_runtime -ApproveNewManifest

& .\scripts\New-ApprovedResourceManifest.ps1 `
  -SourceRoot $env:FSTS_PORTABLE_PYTHON_SOURCE `
  -OutputPath .\resources\manifests\portable_python.sha256.json `
  -Profile portable_python -ApproveNewManifest
```

人工复核清单的文件范围、许可和 SHA-256 后，锁定主源码并执行正式构建：

```powershell
& .\scripts\Test-PackagingStatic.ps1
& .\build_installer_v0.3.ps1 -CleanWork -ReplaceOutput
```

构建脚本读取以下环境变量，也可用同名参数覆盖：

```text
FSTS_FRONTEND_BUILD_PYTHON
FSTS_IDENTITY_BUILD_PYTHON
FSTS_SOURCE_BUILD_PYTHON
FSTS_IDENTITY_PYINSTALLER_TOOLS
FSTS_SOURCE_PYINSTALLER_TOOLS
FSTS_DATA_SOURCE_ROOT
FSTS_IDENTITY_MODEL_SOURCE
FSTS_DEEPFAKE_RUNTIME_SOURCE
FSTS_DEEPFAKE_RUNTIME_MANIFEST
FSTS_PORTABLE_PYTHON_SOURCE
FSTS_PORTABLE_PYTHON_MANIFEST
FSTS_DEEPFAKE_BASE_WEIGHT
FSTS_ISCC
```

`-CleanWork` 只能清理本目录下批准的 `work` 子目录；已有安装器只有显式传入
`-ReplaceOutput` 才允许替换。正式构建产物固定写入
`output/换脸寻源系统_v0.4.0_Setup.exe`。v0.4.0 使用独立的 AppId、安装目录、
快捷方式、主程序名和启动状态文件，不覆盖已安装的 v0.3。
