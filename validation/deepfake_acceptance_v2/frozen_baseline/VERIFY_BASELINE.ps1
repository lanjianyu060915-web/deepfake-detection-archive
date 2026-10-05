[CmdletBinding()]
param(
    [string]$BaselineDirectory = $PSScriptRoot,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
$baselineRoot = (Resolve-Path -LiteralPath $BaselineDirectory).Path
$baselineConfigPath = Join-Path $baselineRoot 'DEEPFAKE_BASELINE_CONFIG.json'
$codeManifestPath = Join-Path $baselineRoot 'DEEPFAKE_CODE_SHA256.csv'
$modelManifestPath = Join-Path $baselineRoot 'DEEPFAKE_MODEL_SHA256.csv'

foreach ($required in @($baselineConfigPath, $codeManifestPath, $modelManifestPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "冻结基线文件不存在：$required"
    }
}

$baseline = Get-Content -LiteralPath $baselineConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$checks = New-Object System.Collections.Generic.List[object]

function Add-IntegrityCheck {
    param(
        [string]$Category,
        [string]$Role,
        [string]$Path,
        [long]$ExpectedSize,
        [string]$ExpectedSha256
    )

    $exists = Test-Path -LiteralPath $Path -PathType Leaf
    $actualSize = $null
    $actualSha256 = $null
    $sizeMatches = $false
    $hashMatches = $false
    $errorMessage = $null
    if ($exists) {
        try {
            $item = Get-Item -LiteralPath $Path
            $actualSize = [long]$item.Length
            $actualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
            $sizeMatches = $actualSize -eq $ExpectedSize
            $hashMatches = $actualSha256 -eq $ExpectedSha256.ToLowerInvariant()
        }
        catch {
            $errorMessage = $_.Exception.Message
        }
    }
    else {
        $errorMessage = 'file_missing'
    }

    $checks.Add([pscustomobject]@{
        category = $Category
        role = $Role
        path = $Path
        expected_size_bytes = $ExpectedSize
        actual_size_bytes = $actualSize
        size_matches = $sizeMatches
        expected_sha256 = $ExpectedSha256.ToLowerInvariant()
        actual_sha256 = $actualSha256
        hash_matches = $hashMatches
        passed = ($exists -and $sizeMatches -and $hashMatches -and -not $errorMessage)
        error = $errorMessage
    }) | Out-Null
}

foreach ($manifestSpec in @(
    @{ Category = 'code'; Path = $codeManifestPath },
    @{ Category = 'model'; Path = $modelManifestPath }
)) {
    $rows = @(Import-Csv -LiteralPath $manifestSpec.Path -Encoding UTF8)
    if ($rows.Count -eq 0) {
        throw "冻结清单为空：$($manifestSpec.Path)"
    }
    $duplicatePaths = @($rows | Group-Object -Property absolute_path | Where-Object Count -gt 1)
    if ($duplicatePaths.Count -gt 0) {
        throw "冻结清单存在重复绝对路径：$($manifestSpec.Path)"
    }
    foreach ($row in $rows) {
        if ([string]::IsNullOrWhiteSpace($row.absolute_path) -or
            [string]::IsNullOrWhiteSpace($row.sha256) -or
            $row.sha256 -notmatch '^[0-9a-fA-F]{64}$') {
            throw "冻结清单字段非法：$($manifestSpec.Path) role=$($row.role)"
        }
        $expectedSize = 0L
        if (-not [long]::TryParse($row.size_bytes, [ref]$expectedSize) -or $expectedSize -lt 0) {
            throw "冻结清单size_bytes非法：$($manifestSpec.Path) role=$($row.role)"
        }
        Add-IntegrityCheck -Category $manifestSpec.Category -Role $row.role `
            -Path $row.absolute_path -ExpectedSize $expectedSize -ExpectedSha256 $row.sha256
    }
}

Add-IntegrityCheck -Category 'source_config' -Role 'v0.2_config_json' `
    -Path $baseline.source_config.path `
    -ExpectedSize ([long]$baseline.source_config.size_bytes) `
    -ExpectedSha256 ([string]$baseline.source_config.sha256)

Add-IntegrityCheck -Category 'environment' -Role 'adapter_python' `
    -Path $baseline.runtime_environment.adapter_python.path `
    -ExpectedSize ([long]$baseline.runtime_environment.adapter_python.size_bytes) `
    -ExpectedSha256 ([string]$baseline.runtime_environment.adapter_python.sha256)

Add-IntegrityCheck -Category 'environment' -Role 'installed_python_dll' `
    -Path $baseline.runtime_environment.installed_python_dll.path `
    -ExpectedSize ([long]$baseline.runtime_environment.installed_python_dll.size_bytes) `
    -ExpectedSha256 ([string]$baseline.runtime_environment.installed_python_dll.sha256)

$configValuesMatch = $false
$configValueError = $null
try {
    $currentConfig = Get-Content -LiteralPath $baseline.source_config.path -Raw -Encoding UTF8 | ConvertFrom-Json
    $expectedDeepfake = $baseline.source_config.deepfake | ConvertTo-Json -Compress -Depth 20
    $actualDeepfake = $currentConfig.deepfake | ConvertTo-Json -Compress -Depth 20
    $configValuesMatch = $expectedDeepfake -ceq $actualDeepfake
    if (-not $configValuesMatch) {
        $configValueError = 'deepfake_config_value_mismatch'
    }
}
catch {
    $configValueError = $_.Exception.Message
}
$checks.Add([pscustomobject]@{
    category = 'source_config'
    role = 'deepfake_section_values'
    path = [string]$baseline.source_config.path
    expected_size_bytes = $null
    actual_size_bytes = $null
    size_matches = $null
    expected_sha256 = [string]$baseline.source_config.deepfake_section_canonical_sha256
    actual_sha256 = $null
    hash_matches = $null
    passed = $configValuesMatch
    error = $configValueError
}) | Out-Null

$failed = @($checks | Where-Object { -not $_.passed })
$summary = [pscustomobject]@{
    baseline_id = [string]$baseline.baseline_id
    verified_at = (Get-Date).ToString('o')
    verification_mode = 'read_only_sha256_no_model_inference'
    total_checks = $checks.Count
    passed_checks = $checks.Count - $failed.Count
    failed_checks = $failed.Count
    status = if ($failed.Count -eq 0) { 'PASS' } else { 'FAIL' }
    failed_items = $failed
}

if ($Json) {
    $summary | ConvertTo-Json -Depth 10
}
else {
    Write-Output "Baseline: $($summary.baseline_id)"
    Write-Output "Mode: $($summary.verification_mode)"
    Write-Output "Checks: $($summary.passed_checks)/$($summary.total_checks) passed"
    Write-Output "Status: $($summary.status)"
    if ($failed.Count -gt 0) {
        $failed | Format-Table -AutoSize category, role, path, error
    }
}

if ($failed.Count -gt 0) {
    exit 1
}
exit 0
