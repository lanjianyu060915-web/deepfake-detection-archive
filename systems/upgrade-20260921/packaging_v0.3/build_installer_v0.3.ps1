#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$FrontendPython,
    [string]$IdentityPython,
    [string]$SourcePython,
    [string]$IdentityPyInstallerTools,
    [string]$SourcePyInstallerTools,
    [string]$DataSourceRoot,
    [string]$IdentityModelSource,
    [string]$DeepfakeRuntimeSource,
    [string]$DeepfakeRuntimeManifest,
    [string]$PortablePythonSource,
    [string]$PortablePythonManifest,
    [string]$IsccPath,
    [switch]$SkipBuild,
    [switch]$SkipAssemble,
    [switch]$SkipInstaller,
    [switch]$CleanWork,
    [switch]$CleanAssemble,
    [switch]$ReplaceOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PackagingRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$RepoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PackagingRoot)).TrimEnd('\')
$WorkRoot = Join-Path $PackagingRoot 'work'
$BuildRoot = Join-Path $WorkRoot 'build'
$StageRoot = Join-Path $WorkRoot 'stage'
$AppDir = Join-Path $WorkRoot 'dist\app'
$OutputRoot = Join-Path $PackagingRoot 'output'
$SetupPath = Join-Path $OutputRoot '换脸寻源系统_v0.4.0_Setup.exe'

function Assert-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Label 不存在：$Path" }
}

function Assert-Directory([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { throw "$Label 不存在：$Path" }
}

function Resolve-FileInput {
    param(
        [string]$Explicit,
        [string]$EnvironmentName,
        [string[]]$RelativeCandidates,
        [string]$Label
    )
    $candidates = [Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($Explicit)) { $candidates.Add($Explicit) }
    $environmentValue = [Environment]::GetEnvironmentVariable($EnvironmentName, 'Process')
    if (-not [string]::IsNullOrWhiteSpace($environmentValue)) { $candidates.Add($environmentValue) }
    foreach ($candidate in $RelativeCandidates) { $candidates.Add($candidate) }
    foreach ($candidate in $candidates) {
        $full = [IO.Path]::GetFullPath($candidate)
        if (Test-Path -LiteralPath $full -PathType Leaf) { return $full }
    }
    throw "$Label 未提供。请传入参数或设置环境变量 $EnvironmentName。"
}

function Resolve-DirectoryInput {
    param(
        [string]$Explicit,
        [string]$EnvironmentName,
        [string[]]$RelativeCandidates,
        [string]$Label
    )
    $candidates = [Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($Explicit)) { $candidates.Add($Explicit) }
    $environmentValue = [Environment]::GetEnvironmentVariable($EnvironmentName, 'Process')
    if (-not [string]::IsNullOrWhiteSpace($environmentValue)) { $candidates.Add($environmentValue) }
    foreach ($candidate in $RelativeCandidates) { $candidates.Add($candidate) }
    foreach ($candidate in $candidates) {
        $full = [IO.Path]::GetFullPath($candidate).TrimEnd('\')
        if (Test-Path -LiteralPath $full -PathType Container) { return $full }
    }
    throw "$Label 未提供。请传入参数或设置环境变量 $EnvironmentName。"
}

function Resolve-Iscc([string]$Explicit) {
    $environmentValue = [Environment]::GetEnvironmentVariable('FSTS_ISCC', 'Process')
    foreach ($candidate in @($Explicit, $environmentValue)) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and
            (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return [IO.Path]::GetFullPath($candidate)
        }
    }
    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($null -ne $command) { return [IO.Path]::GetFullPath($command.Source) }
    foreach ($registryPath in @(
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1',
        'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1',
        'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1'
    )) {
        $record = Get-ItemProperty -LiteralPath $registryPath -ErrorAction SilentlyContinue
        if ($null -eq $record -or [string]::IsNullOrWhiteSpace([string]$record.InstallLocation)) { continue }
        $candidate = Join-Path ([string]$record.InstallLocation) 'ISCC.exe'
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return [IO.Path]::GetFullPath($candidate) }
    }
    throw '未找到 Inno Setup 编译器。请传入 -IsccPath 或设置 FSTS_ISCC。'
}

function Reset-ExactWorkDirectory([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $allowed = @($BuildRoot, $StageRoot, (Split-Path -Parent $AppDir)) |
        ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') }
    if ($allowed -notcontains $full) { throw "拒绝清理未批准目录：$full" }
    if (Test-Path -LiteralPath $full) { Remove-Item -LiteralPath $full -Recurse -Force }
    New-Item -ItemType Directory -Path $full -Force | Out-Null
}

function Invoke-PyInstaller {
    param(
        [string]$Python,
        [string]$Spec,
        [string]$WorkPath,
        [string]$DistPath,
        [string]$Label,
        [string]$ToolRoot
    )
    $oldPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
    $oldNoUserSite = [Environment]::GetEnvironmentVariable('PYTHONNOUSERSITE', 'Process')
    $oldNoBytecode = [Environment]::GetEnvironmentVariable('PYTHONDONTWRITEBYTECODE', 'Process')
    $oldAlbumentations = [Environment]::GetEnvironmentVariable('NO_ALBUMENTATIONS_UPDATE', 'Process')
    try {
        if (-not [string]::IsNullOrWhiteSpace($ToolRoot)) { $env:PYTHONPATH = $ToolRoot }
        else { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
        $env:PYTHONNOUSERSITE = '1'
        $env:PYTHONDONTWRITEBYTECODE = '1'
        $env:NO_ALBUMENTATIONS_UPDATE = '1'
        New-Item -ItemType Directory -Path $WorkPath,$DistPath -Force | Out-Null
        Write-Host "[$Label]"
        & $Python -B -m PyInstaller --noconfirm --clean --log-level INFO `
            --workpath $WorkPath --distpath $DistPath $Spec
        if ($LASTEXITCODE -ne 0) { throw "$Label 失败，退出码 $LASTEXITCODE" }
    }
    finally {
        foreach ($entry in @(
            @('PYTHONPATH', $oldPythonPath),
            @('PYTHONNOUSERSITE', $oldNoUserSite),
            @('PYTHONDONTWRITEBYTECODE', $oldNoBytecode),
            @('NO_ALBUMENTATIONS_UPDATE', $oldAlbumentations)
        )) {
            if ($null -eq $entry[1]) { Remove-Item "Env:$($entry[0])" -ErrorAction SilentlyContinue }
            else { [Environment]::SetEnvironmentVariable($entry[0], [string]$entry[1], 'Process') }
        }
    }
}

& (Join-Path $PackagingRoot 'scripts\Test-PackagingStatic.ps1')

if ($CleanWork) {
    Reset-ExactWorkDirectory $BuildRoot
    Reset-ExactWorkDirectory $StageRoot
    Reset-ExactWorkDirectory (Split-Path -Parent $AppDir)
}
else {
    if (-not $SkipBuild) {
        foreach ($path in @($BuildRoot, $StageRoot)) {
            if ((Test-Path -LiteralPath $path) -and @(Get-ChildItem -LiteralPath $path -Force).Count -gt 0) {
                throw "v0.4.0 构建目录非空；确认后传入 -CleanWork：$path"
            }
            New-Item -ItemType Directory -Path $path -Force | Out-Null
        }
    }
    $distRoot = Split-Path -Parent $AppDir
    if ($CleanAssemble) {
        Reset-ExactWorkDirectory $distRoot
    }
    elseif (-not $SkipAssemble) {
        $path = $distRoot
        if ((Test-Path -LiteralPath $path) -and @(Get-ChildItem -LiteralPath $path -Force).Count -gt 0) {
            throw "v0.4.0 装配目录非空；确认后传入 -CleanAssemble 或 -CleanWork：$path"
        }
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

if (-not $SkipBuild) {
    $FrontendPython = Resolve-FileInput $FrontendPython 'FSTS_FRONTEND_BUILD_PYTHON' @(
        (Join-Path $PackagingRoot '.venv_frontend\Scripts\python.exe'),
        (Join-Path $RepoRoot 'packaging\.venv_frontend\Scripts\python.exe')
    ) '前端构建 Python'
    $IdentityPython = Resolve-FileInput $IdentityPython 'FSTS_IDENTITY_BUILD_PYTHON' @(
        (Join-Path $PackagingRoot 'build_envs\identity\Scripts\python.exe')
    ) '身份模块构建 Python'
    $SourcePython = Resolve-FileInput $SourcePython 'FSTS_SOURCE_BUILD_PYTHON' @(
        (Join-Path $PackagingRoot 'build_envs\source\Scripts\python.exe')
    ) '底图模块构建 Python'
    $IdentityPyInstallerTools = Resolve-DirectoryInput $IdentityPyInstallerTools 'FSTS_IDENTITY_PYINSTALLER_TOOLS' @(
        (Join-Path $PackagingRoot '.tools\identity'),
        (Join-Path $RepoRoot 'packaging\.tools\identity')
    ) '身份模块 PyInstaller 工具目录'
    $SourcePyInstallerTools = Resolve-DirectoryInput $SourcePyInstallerTools 'FSTS_SOURCE_PYINSTALLER_TOOLS' @(
        (Join-Path $PackagingRoot '.tools\source'),
        (Join-Path $RepoRoot 'packaging\.tools\source')
    ) '底图模块 PyInstaller 工具目录'

    Invoke-PyInstaller $FrontendPython (Join-Path $PackagingRoot 'pyinstaller\frontend.spec') `
        (Join-Path $BuildRoot 'frontend') (Join-Path $StageRoot 'frontend') '构建正式前端' $null
    Invoke-PyInstaller $IdentityPython (Join-Path $PackagingRoot 'pyinstaller\identity_worker.spec') `
        (Join-Path $BuildRoot 'identity') (Join-Path $StageRoot 'identity') '构建身份工作进程' $IdentityPyInstallerTools
    Invoke-PyInstaller $SourcePython (Join-Path $PackagingRoot 'pyinstaller\source_search.spec') `
        (Join-Path $BuildRoot 'source') (Join-Path $StageRoot 'source') '构建底图工作进程' $SourcePyInstallerTools
}

$frontendDist = Join-Path $StageRoot 'frontend\换脸寻源系统_v0.4.0'
$identityDist = Join-Path $StageRoot 'identity\identity_worker'
$sourceDist = Join-Path $StageRoot 'source\source_search'
foreach ($record in @(
    @($frontendDist, '前端 onedir'),
    @($identityDist, '身份 onedir'),
    @($sourceDist, '底图 onedir')
)) { Assert-Directory $record[0] $record[1] }

if (-not $SkipAssemble) {
    $DataSourceRoot = Resolve-DirectoryInput $DataSourceRoot 'FSTS_DATA_SOURCE_ROOT' @(
        (Join-Path $PackagingRoot 'resources\data'),
        (Join-Path $RepoRoot 'resources\data')
    ) '安装图库源目录'
    $IdentityModelSource = Resolve-DirectoryInput $IdentityModelSource 'FSTS_IDENTITY_MODEL_SOURCE' @(
        (Join-Path $PackagingRoot 'resources\models\buffalo_l')
    ) 'buffalo_l 模型源目录'
    $DeepfakeRuntimeSource = Resolve-DirectoryInput $DeepfakeRuntimeSource 'FSTS_DEEPFAKE_RUNTIME_SOURCE' @(
        (Join-Path $PackagingRoot 'resources\deepfake_runtime')
    ) '深伪运行时源目录'
    $DeepfakeRuntimeManifest = Resolve-FileInput $DeepfakeRuntimeManifest 'FSTS_DEEPFAKE_RUNTIME_MANIFEST' @(
        (Join-Path $PackagingRoot 'resources\manifests\deepfake_runtime.sha256.json')
    ) '深伪运行时受控清单'
    $PortablePythonSource = Resolve-DirectoryInput $PortablePythonSource 'FSTS_PORTABLE_PYTHON_SOURCE' @(
        (Join-Path $PackagingRoot 'resources\portable_python')
    ) '便携 Python 源目录'
    $PortablePythonManifest = Resolve-FileInput $PortablePythonManifest 'FSTS_PORTABLE_PYTHON_MANIFEST' @(
        (Join-Path $PackagingRoot 'resources\manifests\portable_python.sha256.json')
    ) '便携 Python 受控清单'
    & (Join-Path $PackagingRoot 'scripts\assemble_app_v0.3.ps1') `
        -RepoRoot $RepoRoot -FrontendDist $frontendDist -IdentityDist $identityDist `
        -SourceDist $sourceDist -AppDir $AppDir -DataSourceRoot $DataSourceRoot `
        -IdentityModelSource $IdentityModelSource `
        -DeepfakeRuntimeSource $DeepfakeRuntimeSource `
        -DeepfakeRuntimeManifest $DeepfakeRuntimeManifest `
        -PortablePythonSource $PortablePythonSource `
        -PortablePythonManifest $PortablePythonManifest
    & (Join-Path $PackagingRoot 'scripts\Update-PackagedResourceHashes.ps1') -AppDir $AppDir
}

Assert-Directory $AppDir '正式安装 app 目录'
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
if ((Test-Path -LiteralPath $SetupPath -PathType Leaf) -and -not $ReplaceOutput -and -not $SkipInstaller) {
    throw "安装器已存在；确认后传入 -ReplaceOutput：$SetupPath"
}

$appFiles = @(Get-ChildItem -LiteralPath $AppDir -File -Recurse -Force)
$facts = [ordered]@{
    schema_version = '1.0'
    product_version = '0.4.0'
    generated_at = [DateTimeOffset]::Now.ToString('o')
    app_file_count = $appFiles.Count
    app_size_bytes = [int64](($appFiles | Measure-Object Length -Sum).Sum)
    frontend_exe_sha256 = (Get-FileHash -LiteralPath (Join-Path $AppDir '换脸寻源系统_v0.4.0.exe') -Algorithm SHA256).Hash
    identity_exe_sha256 = (Get-FileHash -LiteralPath (Join-Path $AppDir 'workers\identity\identity_worker.exe') -Algorithm SHA256).Hash
    source_exe_sha256 = (Get-FileHash -LiteralPath (Join-Path $AppDir 'workers\source\source_search.exe') -Algorithm SHA256).Hash
    resource_manifest_sha256 = (Get-FileHash -LiteralPath (Join-Path $AppDir 'resources_manifest.sha256.json') -Algorithm SHA256).Hash
}

if (-not $SkipInstaller) {
    $compiler = Resolve-Iscc $IsccPath
    $installerScript = Join-Path $PackagingRoot 'installer\换脸寻源系统_v0.3.iss'
    Assert-File $installerScript 'Inno Setup 脚本'
    & $compiler $installerScript
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup 构建失败，退出码 $LASTEXITCODE" }
    Assert-File $SetupPath '最终安装器'
    $facts['setup_name'] = [IO.Path]::GetFileName($SetupPath)
    $facts['setup_size_bytes'] = [int64](Get-Item -LiteralPath $SetupPath).Length
    $facts['setup_sha256'] = (Get-FileHash -LiteralPath $SetupPath -Algorithm SHA256).Hash
}

[IO.File]::WriteAllText(
    (Join-Path $OutputRoot 'BUILD_FACTS.json'),
    (($facts | ConvertTo-Json -Depth 6) + "`n"),
    [Text.UTF8Encoding]::new($false)
)
Write-Host "v0.4.0 构建链完成。App=$AppDir"
if (-not $SkipInstaller) { Write-Host "Setup=$SetupPath" }
