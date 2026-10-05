[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProductRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $ProductRoot "config.json"
$FrontendPath = Join-Path $ProductRoot "frontend\app.py"

function Test-FrontendPython {
    param([Parameter(Mandatory = $true)][string]$Executable)

    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        return $false
    }
    & $Executable -B -c "import streamlit" *> $null
    return $LASTEXITCODE -eq 0
}

function Resolve-FrontendPython {
    $candidates = @(
        (Join-Path $ProductRoot "python_runtime\python.exe"),
        (Join-Path $ProductRoot "frontend_runtime\python.exe"),
        (Join-Path $ProductRoot "packaging\.venv_frontend\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-FrontendPython -Executable $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand -and (Test-FrontendPython -Executable $pythonCommand.Source)) {
        return $pythonCommand.Source
    }
    throw "未找到可导入 Streamlit 的前端 Python。请使用正式安装包，或在开发环境安装 Python 与 Streamlit。"
}

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "配置文件不存在：$ConfigPath"
}
if (-not (Test-Path -LiteralPath $FrontendPath -PathType Leaf)) {
    throw "前端入口不存在：$FrontendPath"
}

$FrontendPython = Resolve-FrontendPython
$env:FSTS_CONFIG_PATH = $ConfigPath

Write-Host "[1/4] 运行环境检查" -ForegroundColor Cyan
$EnvironmentChecker = Join-Path $ProductRoot "check_environment.py"
if (Test-Path -LiteralPath $EnvironmentChecker -PathType Leaf) {
    & $FrontendPython -B $EnvironmentChecker --config $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "部分环境检查未通过。UI 仍会启动，并禁用不可用模块。"
    }
}

Write-Host "[2/4] 检查模型与数据库" -ForegroundColor Cyan
$StatusChecker = Join-Path $ProductRoot "system_status.py"
if (Test-Path -LiteralPath $StatusChecker -PathType Leaf) {
    & $FrontendPython -B $StatusChecker --config $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "存在不可用模块；请在 UI 状态卡片和日志中查看原因。"
    }
} else {
    Write-Warning "system_status.py 尚不存在。"
}

$Config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$Address = if ($Config.frontend.address) { [string]$Config.frontend.address } else { "127.0.0.1" }
$Port = if ($Config.frontend.port) { [int]$Config.frontend.port } else { 8767 }

Write-Host "[3/4] 启动本地系统：$Address`:$Port" -ForegroundColor Cyan
Write-Host "[4/4] 浏览器将在服务就绪后自动打开；关闭终端会停止本系统 UI。" -ForegroundColor Cyan

& $FrontendPython -B -m streamlit run $FrontendPath `
    --server.address=$Address `
    --server.port=$Port `
    --server.headless=false `
    --server.fileWatcherType=none `
    --browser.gatherUsageStats=false
exit $LASTEXITCODE
