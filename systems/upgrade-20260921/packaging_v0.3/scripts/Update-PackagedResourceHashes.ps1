#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$AppDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-FullPath([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Write-AtomicUtf8([string]$Path, [string]$Text) {
    $parent = Split-Path -Parent $Path
    $temporary = Join-Path $parent ('.{0}.{1}.tmp' -f ([IO.Path]::GetFileName($Path)), [Guid]::NewGuid().ToString('N'))
    try {
        [IO.File]::WriteAllText($temporary, $Text, [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    }
    finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

$app = Get-FullPath $AppDir
$packagingRoot = Get-FullPath (Split-Path -Parent $PSScriptRoot)
$allowedApp = Get-FullPath (Join-Path $packagingRoot 'work\dist\app')
if (-not $app.Equals($allowedApp, [StringComparison]::OrdinalIgnoreCase)) {
    throw "拒绝更新 v0.4.0 专用目录之外的哈希：$app"
}
$configPath = Join-Path $app 'config.json'
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) { throw "缺少安装配置：$configPath" }

$hashTargets = [ordered]@{
    executable = 'deepfake_runtime/深度伪造识别系统.exe'
    spatial_model = 'deepfake_runtime/models/spatial_mobilenet_finetuned_gpu/spatial_mobilenet_finetuned_gpu.pt'
    frequency_model = 'deepfake_runtime/models/frequency_stat_formal/frequency_stat_detector.joblib'
    fusion_config = 'deepfake_runtime/models/fusion_formal/fusion_config.json'
    adapter = 'modules/deepfake_detection/run_deepfake_detection.py'
    adapter_runtime = 'modules/deepfake_detection/pyinstaller_pyz_runtime.py'
    adapter_inspector = 'modules/deepfake_detection/inspect_installed_archive.py'
}
$knownHashes = [ordered]@{}
$manifestRecords = [Collections.Generic.List[object]]::new()

function Add-ManifestRecord([string]$RelativePath) {
    $platformRelative = $RelativePath.Replace('/', '\')
    $fullPath = Join-Path $app $platformRelative
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) { throw "关键资源不存在：$RelativePath" }
    $item = Get-Item -LiteralPath $fullPath
    $hash = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $script:manifestRecords.Add([ordered]@{
        path = $RelativePath
        size = [int64]$item.Length
        sha256 = $hash
    })
    return $hash
}

foreach ($name in $hashTargets.Keys) {
    $knownHashes[$name] = Add-ManifestRecord ([string]$hashTargets[$name])
}
foreach ($relative in @(
    '换脸寻源系统_v0.4.0.exe',
    'workers/identity/identity_worker.exe',
    'workers/source/source_search.exe',
    'python_runtime/python.exe',
    'model_runtime/models/buffalo_l/1k3d68.onnx',
    'model_runtime/models/buffalo_l/2d106det.onnx',
    'model_runtime/models/buffalo_l/det_10g.onnx',
    'model_runtime/models/buffalo_l/genderage.onnx',
    'model_runtime/models/buffalo_l/w600k_r50.onnx'
)) {
    $null = Add-ManifestRecord $relative
}

$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$config.deepfake.known_sha256 = [pscustomobject]$knownHashes
Write-AtomicUtf8 $configPath (($config | ConvertTo-Json -Depth 20) + "`n")

$manifest = [ordered]@{
    schema_version = '1.0'
    product_version = '0.4.0'
    generated_at = [DateTimeOffset]::Now.ToString('o')
    algorithm = 'SHA-256'
    entries = @($manifestRecords | Sort-Object path)
}
$manifestPath = Join-Path $app 'resources_manifest.sha256.json'
Write-AtomicUtf8 $manifestPath (($manifest | ConvertTo-Json -Depth 6) + "`n")
Write-Host "已更新安装配置哈希与关键资源清单：$manifestPath"
