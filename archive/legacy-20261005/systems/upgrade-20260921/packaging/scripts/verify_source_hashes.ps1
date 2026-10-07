[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$CompareTo
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedRepo = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd("\")
$moduleRoots = @(
    Join-Path $resolvedRepo "modules\identity_search"
    Join-Path $resolvedRepo "modules\source_search"
)

$records = New-Object System.Collections.Generic.List[object]
foreach ($moduleRoot in $moduleRoots) {
    if (-not (Test-Path -LiteralPath $moduleRoot -PathType Container)) {
        throw "Frozen module root is missing: $moduleRoot"
    }
    Get-ChildItem -LiteralPath $moduleRoot -File -Recurse |
        Where-Object {
            $_.FullName -notmatch "[\\/]__pycache__[\\/]" -and
            $_.Extension -notin @(".pyc", ".pyo")
        } |
        Sort-Object FullName |
        ForEach-Object {
            $relative = $_.FullName.Substring($resolvedRepo.Length).TrimStart("\")
            $records.Add([pscustomobject]@{
                relative_path = $relative.Replace("\", "/")
                size_bytes = [int64]$_.Length
                sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            })
        }
}

$outputParent = Split-Path -Parent ([System.IO.Path]::GetFullPath($OutputPath))
if ($outputParent) {
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
}
$records | Export-Csv -LiteralPath $OutputPath -NoTypeInformation -Encoding UTF8

if ($CompareTo) {
    if (-not (Test-Path -LiteralPath $CompareTo -PathType Leaf)) {
        throw "Source snapshot for comparison is missing: $CompareTo"
    }
    $expected = @(Import-Csv -LiteralPath $CompareTo)
    # Materialize the generic list through the pipeline.  Direct @($records)
    # triggers a PowerShell 7 enumerable binder bug for List[object].
    $actual = @($records | ForEach-Object { $_ })
    $expectedByPath = @{}
    foreach ($item in $expected) {
        $expectedByPath[$item.relative_path] = $item
    }
    $actualByPath = @{}
    foreach ($item in $actual) {
        $actualByPath[$item.relative_path] = $item
    }

    $problems = New-Object System.Collections.Generic.List[string]
    foreach ($path in @($expectedByPath.Keys | Sort-Object)) {
        if (-not $actualByPath.ContainsKey($path)) {
            $problems.Add("removed: $path")
            continue
        }
        $before = $expectedByPath[$path]
        $after = $actualByPath[$path]
        if (([string]$before.sha256).ToLowerInvariant() -ne ([string]$after.sha256).ToLowerInvariant()) {
            $problems.Add("changed: $path")
        }
        elseif ([int64]$before.size_bytes -ne [int64]$after.size_bytes) {
            $problems.Add("size changed: $path")
        }
    }
    foreach ($path in @($actualByPath.Keys | Sort-Object)) {
        if (-not $expectedByPath.ContainsKey($path)) {
            $problems.Add("added: $path")
        }
    }
    if ($problems.Count -gt 0) {
        throw "Frozen A/B source changed during packaging:`n$($problems -join "`n")"
    }
    Write-Host "Frozen A/B source hash verification passed ($($actual.Count) files)."
}
