[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$RepoRoot,
    [Parameter(Mandatory = $true)] [string]$AppDir,
    [Parameter(Mandatory = $true)] [string]$ReleaseDir,
    [string]$SetupPath,
    [Parameter(Mandatory = $true)] [string]$SourceSnapshotPath,
    [string]$TestResultsPath,
    [Parameter(Mandatory = $true)] [string]$FrontendPython,
    [Parameter(Mandatory = $true)] [string]$IdentityPython,
    [Parameter(Mandatory = $true)] [string]$SourcePython,
    [Parameter(Mandatory = $true)] [string]$PyInstallerVersion,
    [Parameter(Mandatory = $true)] [string]$StreamlitVersion
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-TreeStats([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return [ordered]@{ files = 0; bytes = 0 }
    }
    $files = @(Get-ChildItem -LiteralPath $Path -File -Recurse)
    $bytes = [int64]0
    foreach ($file in $files) { $bytes += [int64]$file.Length }
    return [ordered]@{ files = $files.Count; bytes = $bytes }
}

function Get-PythonVersion([string]$PythonPath) {
    $value = & $PythonPath -c "import platform; print(platform.python_version())"
    if ($LASTEXITCODE -ne 0) { throw "Unable to query Python version: $PythonPath" }
    return ($value | Out-String).Trim()
}

$resolvedApp = [System.IO.Path]::GetFullPath($AppDir).TrimEnd("\")
$resolvedRelease = [System.IO.Path]::GetFullPath($ReleaseDir).TrimEnd("\")
if (-not (Test-Path -LiteralPath $resolvedApp -PathType Container)) {
    throw "Release app directory is missing: $resolvedApp"
}
New-Item -ItemType Directory -Path $resolvedRelease -Force | Out-Null

$manifestPath = Join-Path $resolvedRelease "RELEASE_MANIFEST.txt"
$manifestLines = New-Object System.Collections.Generic.List[string]
$manifestLines.Add("# Face Source Trace release manifest")
$manifestLines.Add("# path`tbytes`tsha256")
foreach ($file in @(Get-ChildItem -LiteralPath $resolvedApp -File -Recurse | Sort-Object FullName)) {
    $relative = $file.FullName.Substring($resolvedApp.Length).TrimStart("\").Replace("\", "/")
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $manifestLines.Add("app/$relative`t$($file.Length)`t$hash")
}
$hasSetup = $false
if ($SetupPath -and (Test-Path -LiteralPath $SetupPath -PathType Leaf)) {
    $setupFile = Get-Item -LiteralPath $SetupPath
    $setupHash = (Get-FileHash -LiteralPath $setupFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $manifestLines.Add("$($setupFile.Name)`t$($setupFile.Length)`t$setupHash")
    $hasSetup = $true
}
[System.IO.File]::WriteAllLines($manifestPath, $manifestLines, (New-Object System.Text.UTF8Encoding($false)))

$shaLines = New-Object System.Collections.Generic.List[string]
if ($hasSetup) {
    $setupFile = Get-Item -LiteralPath $SetupPath
    $setupHash = (Get-FileHash -LiteralPath $setupFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $shaLines.Add("$setupHash  $($setupFile.Name)")
}
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
$shaLines.Add("$manifestHash  RELEASE_MANIFEST.txt")
[System.IO.File]::WriteAllLines((Join-Path $resolvedRelease "SHA256.txt"), $shaLines, (New-Object System.Text.UTF8Encoding($false)))

$sourceSnapshotHash = $null
if (Test-Path -LiteralPath $SourceSnapshotPath -PathType Leaf) {
    $sourceSnapshotHash = (Get-FileHash -LiteralPath $SourceSnapshotPath -Algorithm SHA256).Hash.ToLowerInvariant()
}
$testResults = $null
if ($TestResultsPath -and (Test-Path -LiteralPath $TestResultsPath -PathType Leaf)) {
    $testResults = Get-Content -LiteralPath $TestResultsPath -Raw | ConvertFrom-Json
}

$setupFacts = $null
if ($hasSetup) {
    $setupFile = Get-Item -LiteralPath $SetupPath
    $setupFacts = [ordered]@{
        path = $setupFile.FullName
        bytes = [int64]$setupFile.Length
        sha256 = (Get-FileHash -LiteralPath $setupFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$facts = [ordered]@{
    schema_version = "1.0"
    generated_utc = [DateTime]::UtcNow.ToString("o")
    repo_root = [System.IO.Path]::GetFullPath($RepoRoot)
    platform = [ordered]@{
        os = [System.Environment]::OSVersion.VersionString
        architecture = if ([System.Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }
        powershell = $PSVersionTable.PSVersion.ToString()
    }
    tools = [ordered]@{
        pyinstaller = $PyInstallerVersion
        streamlit = $StreamlitVersion
        frontend_python = [ordered]@{ path = $FrontendPython; version = Get-PythonVersion $FrontendPython }
        identity_python = [ordered]@{ path = $IdentityPython; version = Get-PythonVersion $IdentityPython }
        source_python = [ordered]@{ path = $SourcePython; version = Get-PythonVersion $SourcePython }
    }
    isolation = [ordered]@{
        frontend = "packaging/.venv_frontend"
        identity = "A_env plus packaging/.tools/identity only"
        source = "B_env plus packaging/.tools/source only"
        worker_dependencies_merged = $false
    }
    source_integrity = [ordered]@{
        snapshot = $SourceSnapshotPath
        snapshot_sha256 = $sourceSnapshotHash
        unchanged = $true
    }
    payload = [ordered]@{
        app = Get-TreeStats $resolvedApp
        frontend_runtime = Get-TreeStats (Join-Path $resolvedApp "_internal")
        identity_worker = Get-TreeStats (Join-Path $resolvedApp "workers\identity")
        source_worker = Get-TreeStats (Join-Path $resolvedApp "workers\source")
        identity_library = Get-TreeStats (Join-Path $resolvedApp "data\identity_library")
        source_library = Get-TreeStats (Join-Path $resolvedApp "data\source_library")
        samples = Get-TreeStats (Join-Path $resolvedApp "samples")
        models = Get-TreeStats (Join-Path $resolvedApp "model_runtime\models\buffalo_l")
        licenses = Get-TreeStats (Join-Path $resolvedApp "licenses")
    }
    tests = $testResults
    setup = $setupFacts
    manifest = [ordered]@{ path = $manifestPath; sha256 = $manifestHash }
}
$facts | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $resolvedRelease "BUILD_FACTS.json") -Encoding UTF8
Write-Host "Release manifest and build facts written to: $resolvedRelease"
