#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$FrontendDist,
    [Parameter(Mandatory)][string]$IdentityDist,
    [Parameter(Mandatory)][string]$SourceDist,
    [Parameter(Mandatory)][string]$AppDir,
    [Parameter(Mandatory)][string]$DataSourceRoot,
    [Parameter(Mandatory)][string]$IdentityModelSource,
    [Parameter(Mandatory)][string]$DeepfakeRuntimeSource,
    [Parameter(Mandatory)][string]$DeepfakeRuntimeManifest,
    [Parameter(Mandatory)][string]$PortablePythonSource,
    [Parameter(Mandatory)][string]$PortablePythonManifest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-FullPath([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Assert-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Label 不存在：$Path" }
}

function Assert-Directory([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { throw "$Label 不存在：$Path" }
}

function Test-IsSameOrChild([string]$Path, [string]$Root) {
    $candidate = Get-FullPath $Path
    $parent = Get-FullPath $Root
    return $candidate.Equals($parent, [StringComparison]::OrdinalIgnoreCase) -or
        $candidate.StartsWith($parent + '\', [StringComparison]::OrdinalIgnoreCase)
}

function Copy-BuildTree([string]$Source, [string]$Destination) {
    Assert-Directory $Source 'PyInstaller onedir'
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $arguments = @(
        $Source, $Destination, '/E', '/COPY:DAT', '/DCOPY:T', '/R:1', '/W:1',
        '/NP', '/NFL', '/NDL', '/NJH', '/NJS', '/XJ'
    )
    & robocopy @arguments
    if ($LASTEXITCODE -gt 7) { throw "复制 onedir 失败（$LASTEXITCODE）：$Source" }
}

function Copy-ManifestTree(
    [string]$Source,
    [string]$Destination,
    [string]$ManifestPath,
    [string]$ExpectedProfile
) {
    Assert-Directory $Source "$ExpectedProfile 源目录"
    Assert-File $ManifestPath "$ExpectedProfile 受控清单"
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$manifest.schema_version -ne '1.0' -or [string]$manifest.profile -ne $ExpectedProfile) {
        throw "$ExpectedProfile 清单格式或 profile 不正确：$ManifestPath"
    }
    $entries = @($manifest.entries)
    if ($entries.Count -eq 0) { throw "$ExpectedProfile 清单为空。" }
    $sourceRoot = Get-FullPath $Source
    $destinationRoot = Get-FullPath $Destination
    New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null
    foreach ($record in $entries) {
        $relative = [string]$record.path
        if ([string]::IsNullOrWhiteSpace($relative) -or [IO.Path]::IsPathRooted($relative) -or
            @($relative -split '[\\/]' | Where-Object { $_ -eq '..' }).Count -gt 0) {
            throw "清单路径无效：$relative"
        }
        $platformRelative = $relative.Replace('/', '\')
        $sourceFile = Get-FullPath (Join-Path $sourceRoot $platformRelative)
        $destinationFile = Get-FullPath (Join-Path $destinationRoot $platformRelative)
        if (-not (Test-IsSameOrChild $sourceFile $sourceRoot) -or
            -not (Test-IsSameOrChild $destinationFile $destinationRoot)) {
            throw "清单路径越界：$relative"
        }
        Assert-File $sourceFile "$ExpectedProfile 文件"
        $item = Get-Item -LiteralPath $sourceFile
        if ([int64]$item.Length -ne [int64]$record.size) { throw "资源大小变化：$relative" }
        $hash = (Get-FileHash -LiteralPath $sourceFile -Algorithm SHA256).Hash
        if (-not $hash.Equals([string]$record.sha256, [StringComparison]::OrdinalIgnoreCase)) {
            throw "资源 SHA-256 变化：$relative"
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $destinationFile) -Force | Out-Null
        Copy-Item -LiteralPath $sourceFile -Destination $destinationFile
    }
}

function Copy-ImageTree([string]$Source, [string]$Destination, [string[]]$Extensions) {
    Assert-Directory $Source '图库目录'
    $sourceRoot = Get-FullPath $Source
    $destinationRoot = Get-FullPath $Destination
    New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null
    $count = 0
    foreach ($file in Get-ChildItem -LiteralPath $sourceRoot -File -Recurse -Force) {
        if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "图库包含重解析点：$($file.FullName)"
        }
        if ($Extensions -notcontains $file.Extension.ToLowerInvariant()) { continue }
        $relative = [IO.Path]::GetRelativePath($sourceRoot, $file.FullName)
        $destinationFile = Join-Path $destinationRoot $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $destinationFile) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $destinationFile
        $count++
    }
    if ($count -eq 0) { throw "图库没有受支持图片：$sourceRoot" }
    return $count
}

$repo = Get-FullPath $RepoRoot
$packagingRoot = Get-FullPath (Join-Path $repo 'packaging_v0.3')
$app = Get-FullPath $AppDir
$allowedApp = Get-FullPath (Join-Path $packagingRoot 'work\dist\app')
if (-not $app.Equals($allowedApp, [StringComparison]::OrdinalIgnoreCase)) {
    throw "拒绝装配到 v0.4.0 专用目录之外：$app"
}
if ((Test-Path -LiteralPath $app) -and @(Get-ChildItem -LiteralPath $app -Force).Count -gt 0) {
    throw "装配目录必须为空：$app"
}
New-Item -ItemType Directory -Path $app -Force | Out-Null

$allowlistPath = Join-Path $packagingRoot 'resource_allowlist.json'
Assert-File $allowlistPath '资源允许清单'
$allowlist = Get-Content -LiteralPath $allowlistPath -Raw -Encoding UTF8 | ConvertFrom-Json

Copy-BuildTree $FrontendDist $app
Copy-BuildTree $IdentityDist (Join-Path $app 'workers\identity')
Copy-BuildTree $SourceDist (Join-Path $app 'workers\source')
Copy-Item -LiteralPath (Join-Path $packagingRoot 'config.packaged.json') -Destination (Join-Path $app 'config.json')
Copy-Item -LiteralPath (Join-Path $packagingRoot 'src\stop_system.ps1') -Destination (Join-Path $app 'stop_system.ps1')

$adapterSource = Join-Path $repo 'modules\deepfake_detection'
$adapterDestination = Join-Path $app 'modules\deepfake_detection'
New-Item -ItemType Directory -Path $adapterDestination -Force | Out-Null
foreach ($name in @($allowlist.deepfake_adapter_files)) {
    $sourceFile = Join-Path $adapterSource ([string]$name)
    Assert-File $sourceFile "AIGC 适配器 $name"
    Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $adapterDestination ([string]$name))
}

$extensions = @($allowlist.allowed_image_extensions | ForEach-Object { ([string]$_).ToLowerInvariant() })
$identityCount = Copy-ImageTree (Join-Path $DataSourceRoot 'identity_library') (Join-Path $app 'data\identity_library') $extensions
$sourceCount = Copy-ImageTree (Join-Path $DataSourceRoot 'source_library') (Join-Path $app 'data\source_library') $extensions
$sampleSource = Join-Path $DataSourceRoot 'queries'
if (Test-Path -LiteralPath $sampleSource -PathType Container) {
    $null = Copy-ImageTree $sampleSource (Join-Path $app 'samples\queries') $extensions
}

$modelDestination = Join-Path $app 'model_runtime\models\buffalo_l'
New-Item -ItemType Directory -Path $modelDestination -Force | Out-Null
foreach ($name in @($allowlist.identity_model_files)) {
    $sourceFile = Join-Path $IdentityModelSource ([string]$name)
    Assert-File $sourceFile "身份模型 $name"
    Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $modelDestination ([string]$name))
}
$actualModelNames = @(Get-ChildItem -LiteralPath $modelDestination -File | Sort-Object Name | ForEach-Object Name)
$expectedModelNames = @($allowlist.identity_model_files | Sort-Object)
if (Compare-Object $expectedModelNames $actualModelNames) { throw '安装模型清单不是批准的精确五文件集合。' }

Copy-ManifestTree $DeepfakeRuntimeSource (Join-Path $app 'deepfake_runtime') $DeepfakeRuntimeManifest 'deepfake_runtime'
Copy-ManifestTree $PortablePythonSource (Join-Path $app 'python_runtime') $PortablePythonManifest 'portable_python'
foreach ($name in @($allowlist.portable_python_required_files)) {
    Assert-File (Join-Path $app (Join-Path 'python_runtime' ([string]$name))) "便携 Python $name"
}

$licenseDestination = Join-Path $app 'licenses'
New-Item -ItemType Directory -Path $licenseDestination -Force | Out-Null
$notice = Join-Path $packagingRoot 'THIRD_PARTY_AND_DATA_NOTICE.md'
if (Test-Path -LiteralPath $notice -PathType Leaf) {
    Copy-Item -LiteralPath $notice -Destination $licenseDestination
}

$requiredFiles = @(
    (Join-Path $app '换脸寻源系统_v0.4.0.exe'),
    (Join-Path $app 'config.json'),
    (Join-Path $app 'stop_system.ps1'),
    (Join-Path $app 'workers\identity\identity_worker.exe'),
    (Join-Path $app 'workers\source\source_search.exe'),
    (Join-Path $app 'python_runtime\python.exe'),
    (Join-Path $app 'deepfake_runtime\深度伪造识别系统.exe')
)
foreach ($required in $requiredFiles) { Assert-File $required '正式安装资源' }

$files = @(Get-ChildItem -LiteralPath $app -File -Recurse -Force)
Write-Host "v0.4.0 装配完成：$app"
Write-Host "Files=$($files.Count) Bytes=$(($files | Measure-Object Length -Sum).Sum) IdentityImages=$identityCount SourceImages=$sourceCount"
