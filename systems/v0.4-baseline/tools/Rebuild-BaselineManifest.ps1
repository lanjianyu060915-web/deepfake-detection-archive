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

if (-not (Test-Path -LiteralPath $baselineFullPath -PathType Container)) {
    throw "旧版目录不存在：$baselineFullPath"
}
if (-not (Test-Path -LiteralPath $projectFullPath -PathType Container)) {
    throw "新版目录不存在：$projectFullPath"
}
if ($baselineFullPath -eq $projectFullPath) {
    throw '旧版目录和新版目录不能相同。'
}

$docsDirectory = Join-Path $projectFullPath 'docs'
$summaryDirectory = Join-Path $docsDirectory 'baseline_test_results'
$manifestPath = Join-Path $docsDirectory 'BASELINE_V0.2_SHA256.csv'
$temporaryManifestPath = Join-Path $docsDirectory (
    '.BASELINE_V0.2_SHA256.' + [guid]::NewGuid().ToString('N') + '.tmp'
)

[int64]$fileCount = 0
[int64]$totalBytes = 0

Get-ChildItem -LiteralPath $baselineFullPath -Recurse -File -Force |
    Sort-Object FullName |
    ForEach-Object {
        $fileCount++
        $totalBytes += [int64]$_.Length
        $relativePath = $_.FullName.Substring($baselineFullPath.Length).TrimStart(
            [char[]]'\/'
        )
        [pscustomobject][ordered]@{
            Path   = $relativePath
            Size   = [int64]$_.Length
            SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    } |
    Export-Csv -LiteralPath $temporaryManifestPath -NoTypeInformation -Encoding utf8

# The temp file lives beside the destination. The overwrite is a single filesystem move,
# so a reader observes either the previous complete CSV or the new complete CSV.
[System.IO.File]::Move($temporaryManifestPath, $manifestPath, $true)

$generatedAt = (Get-Date).ToString('o')
$manifestInfo = Get-Item -LiteralPath $manifestPath
$summary = [ordered]@{
    schema_version        = '1.0'
    source_directory      = $baselineFullPath
    manifest_path         = $manifestPath
    generated_at          = $generatedAt
    file_count            = $fileCount
    total_bytes           = $totalBytes
    manifest_size_bytes   = [int64]$manifestInfo.Length
    manifest_sha256       = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    generation_method     = 'PowerShell Get-ChildItem + Get-FileHash SHA256; atomic same-directory replacement'
}

$summaryPath = Join-Path $summaryDirectory 'baseline_manifest_summary.json'
$temporarySummaryPath = Join-Path $summaryDirectory (
    '.baseline_manifest_summary.' + [guid]::NewGuid().ToString('N') + '.tmp'
)
$summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporarySummaryPath -Encoding utf8
[System.IO.File]::Move($temporarySummaryPath, $summaryPath, $true)

$summary | ConvertTo-Json -Depth 4
