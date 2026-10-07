#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SourceRoot,
    [Parameter(Mandatory)][string]$OutputPath,
    [Parameter(Mandatory)][ValidateSet('deepfake_runtime', 'portable_python')][string]$Profile,
    [Parameter(Mandatory)][switch]$ApproveNewManifest,
    [switch]$ReplaceExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-FullPath([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Test-IsChild([string]$Path, [string]$Root) {
    $candidate = Get-FullPath $Path
    $parent = Get-FullPath $Root
    return $candidate.StartsWith($parent + '\', [StringComparison]::OrdinalIgnoreCase)
}

function Write-AtomicUtf8([string]$Path, [string]$Text) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $temporary = Join-Path $parent ('.{0}.{1}.tmp' -f ([IO.Path]::GetFileName($Path)), [Guid]::NewGuid().ToString('N'))
    try {
        [IO.File]::WriteAllText($temporary, $Text, [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    }
    finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

if (-not $ApproveNewManifest) {
    throw '必须显式传入 -ApproveNewManifest；生成清单不等于取得资源分发授权。'
}

$packagingRoot = Get-FullPath (Split-Path -Parent $PSScriptRoot)
$manifestRoot = Get-FullPath (Join-Path $packagingRoot 'resources\manifests')
$source = Get-FullPath $SourceRoot
$output = [IO.Path]::GetFullPath($OutputPath)
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "资源源目录不存在：$source"
}
if (-not (Test-IsChild $output $manifestRoot)) {
    throw "清单只能写入 packaging_v0.3/resources/manifests 的子路径：$output"
}
if ((Test-Path -LiteralPath $output) -and -not $ReplaceExisting) {
    throw "清单已存在；复核后显式传入 -ReplaceExisting 才能替换：$output"
}

$excludedSegments = @('__pycache__', '.pytest_cache')
$excludedNames = @('unins000.exe', 'unins000.dat', '*.pyc', '*.bak')
if ($Profile -eq 'deepfake_runtime') {
    $excludedSegments += @('补充核验输出', '基线审查输出')
    $excludedNames += @('停止深度伪造识别系统.exe', '_run_five_samples.py')
}
else {
    $excludedSegments += @(
        'site-packages', 'test', 'tests', 'tkinter', 'turtledemo',
        'idlelib', 'venv', 'ensurepip'
    )
}

$records = [Collections.Generic.List[object]]::new()
foreach ($file in Get-ChildItem -LiteralPath $source -File -Recurse -Force | Sort-Object FullName) {
    if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "资源目录包含重解析点，拒绝生成清单：$($file.FullName)"
    }
    $relative = [IO.Path]::GetRelativePath($source, $file.FullName)
    $segments = $relative -split '[\\/]'
    if (@($segments | Where-Object { $excludedSegments -contains $_ }).Count -gt 0) { continue }
    $skip = $false
    foreach ($pattern in $excludedNames) {
        if ($file.Name -like $pattern) { $skip = $true; break }
    }
    if ($skip) { continue }
    $records.Add([ordered]@{
        path = $relative.Replace('\', '/')
        size = [int64]$file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}
if ($records.Count -eq 0) { throw '没有可批准的资源文件。' }
$sourceSizeBytes = [int64]0
foreach ($record in $records) {
    $sourceSizeBytes += [int64]$record['size']
}

$payload = [ordered]@{
    schema_version = '1.0'
    profile = $Profile
    generated_at = [DateTimeOffset]::Now.ToString('o')
    source_file_count = $records.Count
    source_size_bytes = $sourceSizeBytes
    authorization_note = '清单仅固定已复核字节；资源所有权与再分发授权须另行留档。'
    entries = $records
}
Write-AtomicUtf8 $output (($payload | ConvertTo-Json -Depth 6) + "`n")
Write-Host "已生成受控资源清单：$output"
Write-Host "Profile=$Profile Files=$($records.Count)"
