[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$AppDir,
    [Parameter(Mandatory = $true)] [string]$MainExeName,
    [Parameter(Mandatory = $true)] [string]$ProductDataDirName,
    [Parameter(Mandatory = $true)] [string]$ScratchRoot,
    [Parameter(Mandatory = $true)] [string]$ResultsPath,
    [switch]$SkipFrontendLaunch,
    [switch]$SkipBusinessTest,
    [int]$FrontendTimeoutSeconds = 90
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-CheckedExe {
    param(
        [Parameter(Mandatory = $true)] [string]$FilePath,
        [Parameter(Mandatory = $true)] [string[]]$Arguments,
        [Parameter(Mandatory = $true)] [string]$Label
    )
    $captured = & $FilePath @Arguments 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode.`n$captured"
    }
    return $captured.Trim()
}

$resolvedApp = [System.IO.Path]::GetFullPath($AppDir).TrimEnd("\")
$resolvedScratch = [System.IO.Path]::GetFullPath($ScratchRoot).TrimEnd("\")
if (Test-Path -LiteralPath $resolvedScratch) {
    Remove-Item -LiteralPath $resolvedScratch -Recurse -Force
}
New-Item -ItemType Directory -Path $resolvedScratch -Force | Out-Null

$mainExe = Join-Path $resolvedApp $MainExeName
$identityExe = Join-Path $resolvedApp "workers\identity\identity_search.exe"
$sourceExe = Join-Path $resolvedApp "workers\source\source_search.exe"
foreach ($required in @($mainExe, $identityExe, $sourceExe)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Distribution smoke test prerequisite is missing: $required"
    }
}

$results = [ordered]@{
    schema_version = "1.0"
    started_utc = [DateTime]::UtcNow.ToString("o")
    app_dir = $resolvedApp
    worker_help = [ordered]@{}
    frontend = [ordered]@{ status = "skipped" }
    business = [ordered]@{ status = "skipped" }
}

$identityHelp = Invoke-CheckedExe -FilePath $identityExe -Arguments @("--help") -Label "Identity worker --help"
$sourceHelp = Invoke-CheckedExe -FilePath $sourceExe -Arguments @("--help") -Label "Source worker --help"
$results.worker_help.identity = [ordered]@{ status = "passed"; output_contains_usage = $identityHelp.Contains("usage:") }
$results.worker_help.source = [ordered]@{ status = "passed"; output_contains_usage = $sourceHelp.Contains("usage:") }
if (-not $results.worker_help.identity.output_contains_usage -or -not $results.worker_help.source.output_contains_usage) {
    throw "A worker --help smoke test did not emit an argparse usage line."
}

if (-not $SkipFrontendLaunch) {
    $localAppData = Join-Path $resolvedScratch "local_app_data"
    New-Item -ItemType Directory -Path $localAppData -Force | Out-Null
    $oldLocalAppData = $env:LOCALAPPDATA
    $oldBrowser = $env:BROWSER
    $process = $null
    try {
        $env:LOCALAPPDATA = $localAppData
        $whereExe = Join-Path $env:SystemRoot "System32\where.exe"
        if (Test-Path -LiteralPath $whereExe -PathType Leaf) {
            $env:BROWSER = $whereExe
        }
        $process = Start-Process -FilePath $mainExe -PassThru -WindowStyle Hidden
    }
    finally {
        if ($null -eq $oldLocalAppData) { Remove-Item Env:LOCALAPPDATA -ErrorAction SilentlyContinue }
        else { $env:LOCALAPPDATA = $oldLocalAppData }
        if ($null -eq $oldBrowser) { Remove-Item Env:BROWSER -ErrorAction SilentlyContinue }
        else { $env:BROWSER = $oldBrowser }
    }

    $statePath = Join-Path $localAppData "$ProductDataDirName\launcher_state.json"
    $deadline = [DateTime]::UtcNow.AddSeconds($FrontendTimeoutSeconds)
    $healthPassed = $false
    $port = $null
    try {
        while ([DateTime]::UtcNow -lt $deadline) {
            if ($process.HasExited) {
                break
            }
            if (Test-Path -LiteralPath $statePath -PathType Leaf) {
                try {
                    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
                    $port = [int]$state.port
                    $response = Invoke-WebRequest -Uri "http://127.0.0.1:$port/_stcore/health" -UseBasicParsing -TimeoutSec 2
                    if ([int]$response.StatusCode -eq 200) {
                        $healthPassed = $true
                        break
                    }
                }
                catch {
                    # The state file and server become ready at slightly different times.
                }
            }
            Start-Sleep -Milliseconds 250
        }
        if (-not $healthPassed) {
            $launcherLog = Join-Path $localAppData "$ProductDataDirName\logs\launcher.log"
            $logTail = ""
            if (Test-Path -LiteralPath $launcherLog -PathType Leaf) {
                $logTail = (Get-Content -LiteralPath $launcherLog -Tail 80 | Out-String)
            }
            throw "Frozen frontend health check failed within $FrontendTimeoutSeconds seconds.`n$logTail"
        }
        $results.frontend = [ordered]@{ status = "passed"; port = $port; health_path = "/_stcore/health" }
    }
    finally {
        if ($null -ne $process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            Wait-Process -Id $process.Id -Timeout 10 -ErrorAction SilentlyContinue
        }
    }
}

if (-not $SkipBusinessTest) {
    $query = Join-Path $resolvedApp "samples\queries\query_001.jpg"
    $identityLibrary = Join-Path $resolvedApp "data\identity_library"
    $sourceLibrary = Join-Path $resolvedApp "data\source_library"
    $modelRoot = Join-Path $resolvedApp "model_runtime"
    $businessRoot = Join-Path $resolvedScratch "business"
    $cacheRoot = Join-Path $businessRoot "identity_cache"
    $identityOutput = Join-Path $businessRoot "identity_result.json"
    $sourceOutput = Join-Path $businessRoot "source_result.json"
    New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null

    Invoke-CheckedExe -FilePath $identityExe -Label "Frozen identity business test" -Arguments @(
        "--query", $query,
        "--library", $identityLibrary,
        "--top-k", "5",
        "--output", $identityOutput,
        "--device", "cpu",
        "--ctx", "-1",
        "--cache-dir", $cacheRoot,
        "--model-root", $modelRoot,
        "--offline"
    ) | Out-Null
    Invoke-CheckedExe -FilePath $sourceExe -Label "Frozen source business test" -Arguments @(
        "--query", $query,
        "--library", $sourceLibrary,
        "--top-k", "5",
        "--output", $sourceOutput
    ) | Out-Null

    $identityResult = Get-Content -LiteralPath $identityOutput -Raw | ConvertFrom-Json
    $sourceResult = Get-Content -LiteralPath $sourceOutput -Raw | ConvertFrom-Json
    if ($identityResult.status -ne "success") {
        throw "Frozen identity business result is not successful: $($identityResult.error_message)"
    }
    if ($sourceResult.status -ne "success") {
        throw "Frozen source business result is not successful: $($sourceResult.error_message)"
    }
    if (@($identityResult.candidates).Count -lt 1 -or @($sourceResult.candidates).Count -lt 1) {
        throw "A frozen business test returned no candidates."
    }
    $results.business = [ordered]@{
        status = "passed"
        query = "samples/queries/query_001.jpg"
        identity = [ordered]@{
            status = $identityResult.status
            candidate_count = @($identityResult.candidates).Count
            rank1_identity_id = $identityResult.candidates[0].identity_id
            runtime_seconds = $identityResult.runtime_seconds
        }
        source = [ordered]@{
            status = $sourceResult.status
            candidate_count = @($sourceResult.candidates).Count
            rank1_file = [System.IO.Path]::GetFileName([string]$sourceResult.candidates[0].file_path)
            runtime_seconds = $sourceResult.runtime_seconds
        }
    }
}

$results.finished_utc = [DateTime]::UtcNow.ToString("o")
$resultsParent = Split-Path -Parent ([System.IO.Path]::GetFullPath($ResultsPath))
New-Item -ItemType Directory -Path $resultsParent -Force | Out-Null
$results | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ResultsPath -Encoding UTF8
Write-Host "Distribution smoke tests passed: $ResultsPath"

