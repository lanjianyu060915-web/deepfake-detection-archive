[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$BaselineRoot = 'D:\刑侦查竞赛\face_source_trace_three_module_v0.2_dev',

    [Parameter(Mandatory = $false)]
    [string]$ProjectRoot = 'D:\刑侦查竞赛\face_source_trace_three_module_v0.3_explainable_dev'
)

$ErrorActionPreference = 'Stop'
$baselineFullPath = [System.IO.Path]::GetFullPath($BaselineRoot)
$projectFullPath = [System.IO.Path]::GetFullPath($ProjectRoot)
$manifestPath = Join-Path $projectFullPath 'docs\BASELINE_V0.2_SHA256.csv'
$summaryPath = Join-Path $projectFullPath 'docs\baseline_test_results\baseline_manifest_summary.json'
$verificationPath = Join-Path $projectFullPath 'docs\baseline_test_results\baseline_manifest_verification.json'

if (-not (Test-Path -LiteralPath $baselineFullPath -PathType Container)) {
    throw "旧版目录不存在：$baselineFullPath"
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "基线清单不存在：$manifestPath"
}
if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
    throw "清单摘要不存在：$summaryPath"
}

$summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding utf8 | ConvertFrom-Json
$rows = @(Import-Csv -LiteralPath $manifestPath -Encoding utf8)
if ($rows.Count -ne [int64]$summary.file_count) {
    throw "清单行数与摘要不一致：$($rows.Count) != $($summary.file_count)"
}

$seenPaths = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
[int64]$verifiedFiles = 0
[int64]$verifiedBytes = 0

foreach ($row in $rows) {
    if (-not $seenPaths.Add([string]$row.Path)) {
        throw "清单包含重复路径：$($row.Path)"
    }
    $candidatePath = [System.IO.Path]::GetFullPath((Join-Path $baselineFullPath $row.Path))
    $requiredPrefix = $baselineFullPath + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidatePath.StartsWith(
        $requiredPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "清单路径逃逸旧版目录：$($row.Path)"
    }
    $item = Get-Item -LiteralPath $candidatePath -Force
    if ([int64]$item.Length -ne [int64]$row.Size) {
        throw "文件大小不一致：$($row.Path)"
    }
    $actualHash = (Get-FileHash -LiteralPath $candidatePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne ([string]$row.SHA256).ToLowerInvariant()) {
        throw "SHA-256 不一致：$($row.Path)"
    }
    $verifiedFiles++
    $verifiedBytes += [int64]$item.Length
}

if ($verifiedBytes -ne [int64]$summary.total_bytes) {
    throw "逐文件字节总和与摘要不一致：$verifiedBytes != $($summary.total_bytes)"
}
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestHash -ne ([string]$summary.manifest_sha256).ToLowerInvariant()) {
    throw '清单文件自身 SHA-256 与摘要不一致。'
}

$verification = [ordered]@{
    schema_version          = '1.0'
    verified                = $true
    verified_at             = (Get-Date).ToString('o')
    verification_scope      = 'all manifest rows; path containment, uniqueness, size, and SHA-256'
    file_count              = $verifiedFiles
    total_bytes             = $verifiedBytes
    manifest_sha256         = $manifestHash
    manifest_matches_summary = $true
}
$temporaryVerificationPath = Join-Path (Split-Path -Parent $verificationPath) (
    '.baseline_manifest_verification.' + [guid]::NewGuid().ToString('N') + '.tmp'
)
$verification | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath $temporaryVerificationPath -Encoding utf8
[System.IO.File]::Move($temporaryVerificationPath, $verificationPath, $true)
$verification | ConvertTo-Json -Depth 4
