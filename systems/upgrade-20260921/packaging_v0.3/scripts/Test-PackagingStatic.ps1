#requires -Version 7.0
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Static([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

$PackagingRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd('\')
$requiredFiles = @(
    'config.packaged.json',
    'resource_allowlist.json',
    'build_installer_v0.3.ps1',
    'src/app_launcher.py',
    'src/stop_system.ps1',
    'src/identity_worker_entry.py',
    'pyinstaller/frontend.spec',
    'pyinstaller/identity_worker.spec',
    'pyinstaller/source_search.spec',
    'installer/换脸寻源系统_v0.3.iss',
    'scripts/assemble_app_v0.3.ps1',
    'scripts/New-ApprovedResourceManifest.ps1',
    'scripts/Update-PackagedResourceHashes.ps1',
    'acceptance/phase1_install_start.ps1',
    'acceptance/phase2_restart_uninstall.ps1'
)
foreach ($relative in $requiredFiles) {
    Assert-Static (Test-Path -LiteralPath (Join-Path $PackagingRoot $relative) -PathType Leaf) "缺少打包文件：$relative"
}

$config = Get-Content -LiteralPath (Join-Path $PackagingRoot 'config.packaged.json') -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-Static ([string]$config.schema_version -eq '3.0') 'config schema_version 必须为 3.0。'
Assert-Static ([string]$config.product_version -eq '0.4.0') 'config product_version 必须为 0.4.0。'
Assert-Static ([string]$config.product_short_name -eq '换脸寻源系统 v0.4.0') 'config product_short_name 必须为 v0.4.0。'
Assert-Static ([string]$config.build_flavor -eq 'formal') 'config build_flavor 必须为 formal。'
Assert-Static ([bool]$config.features.real_modules_default) '正式配置必须默认真实模块。'
Assert-Static (-not [bool]$config.features.mock_enabled) '正式配置不得启用 Mock。'
Assert-Static (-not [bool]$config.reporting.external_language_model_enabled) '报告不得调用外部语言模型。'
foreach ($value in @(
    [string]$config.runtime.tasks_root,
    [string]$config.runtime.inbox_root,
    [string]$config.runtime.identity_cache,
    [string]$config.deepfake.python,
    [string]$config.deepfake.runner,
    [string]$config.identity.library,
    [string]$config.source.executable,
    [string]$config.source.library
)) {
    Assert-Static ($value.Contains('{APP_ROOT}') -or $value.Contains('{DATA_ROOT}')) "安装路径未使用 APP_ROOT/DATA_ROOT：$value"
}

# 身份模块既可用打包 exe，也可用 python + runner + 常驻 worker 三段式
# （2.0 升级内核）；两种形态都必须落在 APP_ROOT/DATA_ROOT 下。
$identityRuntimeValues = @()
foreach ($name in @('executable', 'python', 'runner', 'persistent_worker')) {
    $property = $config.identity.PSObject.Properties[$name]
    if ($null -ne $property -and $property.Value) {
        $identityRuntimeValues += [string]$property.Value
    }
}
Assert-Static ($identityRuntimeValues.Count -gt 0) '身份模块既未配置 executable，也未配置 python+runner。'
foreach ($value in $identityRuntimeValues) {
    Assert-Static ($value.Contains('{APP_ROOT}') -or $value.Contains('{DATA_ROOT}')) "身份运行路径未使用 APP_ROOT/DATA_ROOT：$value"
}

$frontendSpec = Get-Content -LiteralPath (Join-Path $PackagingRoot 'pyinstaller\frontend.spec') -Raw -Encoding UTF8
Assert-Static ($frontendSpec.Contains('COMPONENT_ROOT')) '前端 spec 未使用通用组件收集策略。'
Assert-Static ($frontendSpec.Contains('REPORTING_ROOT')) '前端 spec 未收录确定性报告包。'
Assert-Static ($frontendSpec.Contains('progress_events.py')) '前端 spec 未收录进度事件模块。'
Assert-Static ($frontendSpec.Contains('name="换脸寻源系统_v0.4.0"')) '前端 spec 主程序名不是 v0.4.0。'
Assert-Static (-not $frontendSpec.Contains('name="换脸寻源系统_v0.3"')) '前端 spec 仍使用 v0.3 主程序名。'
foreach ($forbidden in @('mock_results', 'mock_task', 'sample_data')) {
    Assert-Static (-not $frontendSpec.Contains($forbidden, [StringComparison]::OrdinalIgnoreCase)) "前端正式 spec 收入了演示资源：$forbidden"
}
$identitySpec = Get-Content -LiteralPath (Join-Path $PackagingRoot 'pyinstaller\identity_worker.spec') -Raw -Encoding UTF8
$sourceSpec = Get-Content -LiteralPath (Join-Path $PackagingRoot 'pyinstaller\source_search.spec') -Raw -Encoding UTF8
Assert-Static ($identitySpec.Contains('"progress_events"')) '身份 worker spec 未收录真实进度事件模块。'
Assert-Static ($sourceSpec.Contains('"progress_events"')) '底图 worker spec 未收录真实进度事件模块。'

$launcherText = Get-Content -LiteralPath (Join-Path $PackagingRoot 'src\app_launcher.py') -Raw -Encoding UTF8
Assert-Static ($launcherText.Contains('PRODUCT_VERSION = "0.4.0"')) '启动器产品版本不是 0.4.0。'
Assert-Static ($launcherText.Contains('MAIN_EXE_NAME = "换脸寻源系统_v0.4.0.exe"')) '启动器主程序名不是 v0.4.0。'
Assert-Static ($launcherText.Contains('STATE_PREFIX = "launcher_state_v0.4.0"')) '启动状态文件未按 v0.4.0 版本化。'
Assert-Static ($launcherText.Contains('FaceSourceTrace_v040_')) '启动互斥锁未按 v0.4.0 隔离。'
Assert-Static (-not $launcherText.Contains('MAIN_EXE_NAME = "换脸寻源系统_v0.3.exe"')) '启动器仍使用 v0.3 主程序名。'
Assert-Static (-not $launcherText.Contains('STATE_PREFIX = "launcher_state_v0.3"')) '启动器仍使用 v0.3 状态前缀。'
Assert-Static ($launcherText.Contains('state_executable')) '启动器未验证既有状态 executable。'
Assert-Static ($launcherText.Contains('state_root')) '启动器未验证既有状态 app_root。'
Assert-Static ($launcherText.Contains('FSTS_MOCK_ENABLED')) '启动器未强制关闭正式 Mock。'
Assert-Static ($launcherText.Contains('FSTS_ENABLE_DEV_MOCK"] = "0"')) '启动器未屏蔽开发 Mock 入口。'
Assert-Static ($launcherText.Contains('resources.get("verify_manifest_on_start") is not True')) '启动器未强制资源哈希校验。'

$stopText = Get-Content -LiteralPath (Join-Path $PackagingRoot 'src\stop_system.ps1') -Raw -Encoding UTF8
Assert-Static ($stopText.Contains("'换脸寻源系统_v0.4.0.exe'")) '停止脚本主程序名不是 v0.4.0。'
Assert-Static ($stopText.Contains("'launcher_state_v0.4.0_{0}.json'")) '停止脚本状态文件不是 v0.4.0。'

$frontendVersionText = Get-Content -LiteralPath (Join-Path $PackagingRoot 'pyinstaller\frontend_version_info.txt') -Raw -Encoding UTF8
$identityVersionText = Get-Content -LiteralPath (Join-Path $PackagingRoot 'pyinstaller\identity_version_info.txt') -Raw -Encoding UTF8
$sourceVersionText = Get-Content -LiteralPath (Join-Path $PackagingRoot 'pyinstaller\source_version_info.txt') -Raw -Encoding UTF8
foreach ($record in @(
    @('前端', $frontendVersionText),
    @('身份', $identityVersionText),
    @('底图', $sourceVersionText)
)) {
    Assert-Static ($record[1].Contains('filevers=(0, 4, 0, 0)')) "$($record[0]) PE FileVersion 不是 0.4.0.0。"
    Assert-Static ($record[1].Contains('prodvers=(0, 4, 0, 0)')) "$($record[0]) PE ProductVersion 不是 0.4.0.0。"
    Assert-Static ($record[1].Contains("StringStruct(u'ProductVersion', u'0.4.0')")) "$($record[0]) 字符串版本不是 0.4.0。"
}
Assert-Static ($frontendVersionText.Contains("StringStruct(u'OriginalFilename', u'换脸寻源系统_v0.4.0.exe')")) '前端 PE 原始文件名不是 v0.4.0。'

$buildText = Get-Content -LiteralPath (Join-Path $PackagingRoot 'build_installer_v0.3.ps1') -Raw -Encoding UTF8
Assert-Static ($buildText.Contains("'换脸寻源系统_v0.4.0_Setup.exe'")) '构建脚本安装器输出名不是 v0.4.0。'
Assert-Static ($buildText.Contains("'换脸寻源系统_v0.4.0.exe'")) '构建脚本主程序名不是 v0.4.0。'

$acceptanceCommonText = Get-Content -LiteralPath (Join-Path $PackagingRoot 'acceptance\acceptance_common.ps1') -Raw -Encoding UTF8
$acceptancePhase1Text = Get-Content -LiteralPath (Join-Path $PackagingRoot 'acceptance\phase1_install_start.ps1') -Raw -Encoding UTF8
Assert-Static ($acceptanceCommonText.Contains("'launcher_state_v0.4.0_*.json'")) '验收脚本状态文件过滤器不是 v0.4.0。'
Assert-Static ($acceptancePhase1Text.Contains("[string]`$ProductVersion = '0.4.0'")) '验收脚本产品版本不是 0.4.0。'
Assert-Static ($acceptancePhase1Text.Contains("[string]`$AppId = '{BA84DD0A-D32A-4A30-88B1-1546A06EE248}'")) '验收脚本 AppId 不是 v0.4.0 独立值。'
Assert-Static ($acceptancePhase1Text.Contains("[string]`$ProductShortName = '换脸寻源系统 v0.4.0'")) '验收脚本快捷方式名称不是 v0.4.0。'
Assert-Static ($acceptancePhase1Text.Contains("[string]`$MainExecutableName = '换脸寻源系统_v0.4.0.exe'")) '验收脚本主程序名不是 v0.4.0。'

$installerText = Get-Content -LiteralPath (Join-Path $PackagingRoot 'installer\换脸寻源系统_v0.3.iss') -Raw -Encoding UTF8
Assert-Static ($installerText.Contains('{BA84DD0A-D32A-4A30-88B1-1546A06EE248}')) 'v0.4.0 AppId 不正确。'
Assert-Static (-not $installerText.Contains('{E39A31B7-0C39-4D73-AE6A-15D3200803A4}')) '安装器仍使用 v0.3 AppId，会覆盖旧版。'
Assert-Static ($installerText.Contains('#define AppVersion "0.4.0"')) '安装器 AppVersion 不是 0.4.0。'
Assert-Static ($installerText.Contains('#define AppShortName "换脸寻源系统 v0.4.0"')) '安装器快捷方式/安装目录名不是 v0.4.0。'
Assert-Static ($installerText.Contains('#define AppExeName "换脸寻源系统_v0.4.0.exe"')) '安装器主程序名不是 v0.4.0。'
Assert-Static ($installerText.Contains('[UninstallRun]')) '安装器缺少卸载前停止入口。'
Assert-Static ($installerText.Contains('desktopicon')) '安装器缺少可选桌面快捷方式。'
Assert-Static ($installerText.Contains('换脸寻源系统_v0.4.0_Setup')) '安装器输出名不正确。'
Assert-Static ($installerText.Contains('StopFaceSourceTraceV040')) '卸载停止钩子未按 v0.4.0 版本化。'

$textExtensions = @('.ps1', '.py', '.json', '.spec', '.iss', '.md', '.txt')
$absoluteDrivePattern = [regex]'(?<![A-Za-z])([A-Za-z]):[\\/]'
$generatedRoots = @(
    ([IO.Path]::GetFullPath((Join-Path $PackagingRoot 'work')).TrimEnd('\') + '\'),
    ([IO.Path]::GetFullPath((Join-Path $PackagingRoot 'output')).TrimEnd('\') + '\'),
    # Large third-party/runtime inputs are governed by approved SHA-256 manifests;
    # they are not packaging source and may legitimately contain empty Python files.
    ([IO.Path]::GetFullPath((Join-Path $PackagingRoot 'resources')).TrimEnd('\') + '\')
)
function Test-IsStaticSourcePath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    foreach ($root in $generatedRoots) {
        if ($full.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { return $false }
    }
    return $true
}
$textFiles = @(
    Get-ChildItem -LiteralPath $PackagingRoot -File -Recurse -Force |
        Where-Object {
            (Test-IsStaticSourcePath $_.FullName) -and
            $textExtensions -contains $_.Extension.ToLowerInvariant() -and
            $_.Name -ne 'static_validation.json'
        }
)
$absoluteFindings = [Collections.Generic.List[string]]::new()
foreach ($file in $textFiles) {
    $text = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
    if ($absoluteDrivePattern.IsMatch($text)) {
        $absoluteFindings.Add([IO.Path]::GetRelativePath($PackagingRoot, $file.FullName))
    }
}
Assert-Static ($absoluteFindings.Count -eq 0) "打包链包含固定盘符路径：$($absoluteFindings -join ', ')"

$parseErrors = [Collections.Generic.List[string]]::new()
$powershellFiles = @(
    Get-ChildItem -LiteralPath $PackagingRoot -File -Recurse -Filter '*.ps1' |
        Where-Object { Test-IsStaticSourcePath $_.FullName }
)
foreach ($file in $powershellFiles) {
    $tokens = $null
    $errors = $null
    $null = [Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$tokens, [ref]$errors)
    foreach ($error in @($errors)) {
        $parseErrors.Add(('{0}:{1}:{2}' -f [IO.Path]::GetRelativePath($PackagingRoot, $file.FullName), $error.Extent.StartLineNumber, $error.Message))
    }
}
Assert-Static ($parseErrors.Count -eq 0) "PowerShell 语法错误：`n$($parseErrors -join "`n")"

$python = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $python) {
    $pythonFiles = @(
        (Join-Path $PackagingRoot 'src\app_launcher.py'),
        (Join-Path $PackagingRoot 'src\identity_worker_entry.py'),
        (Join-Path $PackagingRoot 'pyinstaller\frontend.spec'),
        (Join-Path $PackagingRoot 'pyinstaller\identity_worker.spec'),
        (Join-Path $PackagingRoot 'pyinstaller\source_search.spec')
    )
    foreach ($file in $pythonFiles) {
        & $python.Source -B -c 'import pathlib, sys; compile(pathlib.Path(sys.argv[1]).read_bytes(), sys.argv[1], "exec")' $file
        Assert-Static ($LASTEXITCODE -eq 0) "Python 语法检查失败：$file"
    }
}

$result = [ordered]@{
    schema_version = '1.0'
    status = 'passed'
    verified_at = [DateTimeOffset]::Now.ToString('o')
    required_file_count = $requiredFiles.Count
    text_file_count = $textFiles.Count
    powershell_file_count = $powershellFiles.Count
    fixed_drive_path_findings = 0
    formal_mock_enabled = $false
    installer_uninstall_stop_hook = $true
    product_version = '0.4.0'
    app_id = '{BA84DD0A-D32A-4A30-88B1-1546A06EE248}'
    side_by_side_isolated_from_v0_3 = $true
}
$resultPath = Join-Path $PackagingRoot 'static_validation.json'
[IO.File]::WriteAllText(
    $resultPath,
    (($result | ConvertTo-Json -Depth 5) + "`n"),
    [Text.UTF8Encoding]::new($false)
)
Write-Host "PACKAGING_STATIC_VALIDATION=PASSED"
