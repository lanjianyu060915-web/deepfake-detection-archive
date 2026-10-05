[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$RepoRoot,
    [Parameter(Mandatory = $true)] [string]$FrontendDist,
    [Parameter(Mandatory = $true)] [string]$IdentityDist,
    [Parameter(Mandatory = $true)] [string]$SourceDist,
    [Parameter(Mandatory = $true)] [string]$AppDir,
    [Parameter(Mandatory = $true)] [string]$DataSourceRoot,
    [Parameter(Mandatory = $true)] [string]$ModelSourceDir,
    [Parameter(Mandatory = $true)] [string]$QueryDirectoryName
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Directory([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label directory is missing: $Path"
    }
}

function Copy-DirectoryContents([string]$Source, [string]$Destination) {
    Assert-Directory $Source "Source"
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
    }
}

$resolvedRepo = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd("\")
$resolvedApp = [System.IO.Path]::GetFullPath($AppDir).TrimEnd("\")
$allowedApp = [System.IO.Path]::GetFullPath((Join-Path $resolvedRepo "packaging\dist\app")).TrimEnd("\")
if (-not $resolvedApp.Equals($allowedApp, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to assemble outside the exact packaging/dist/app target: $resolvedApp"
}

foreach ($pair in @(
    @($FrontendDist, "Frontend PyInstaller output"),
    @($IdentityDist, "Identity PyInstaller output"),
    @($SourceDist, "Source PyInstaller output"),
    @($DataSourceRoot, "Release data source"),
    @($ModelSourceDir, "InsightFace model source")
)) {
    Assert-Directory $pair[0] $pair[1]
}

if (Test-Path -LiteralPath $resolvedApp) {
    Remove-Item -LiteralPath $resolvedApp -Recurse -Force
}
New-Item -ItemType Directory -Path $resolvedApp -Force | Out-Null

Copy-DirectoryContents $FrontendDist $resolvedApp
Copy-DirectoryContents $IdentityDist (Join-Path $resolvedApp "workers\identity")
Copy-DirectoryContents $SourceDist (Join-Path $resolvedApp "workers\source")

$packagedConfig = Join-Path $resolvedRepo "packaging\config.packaged.json"
if (-not (Test-Path -LiteralPath $packagedConfig -PathType Leaf)) {
    throw "Packaged config is missing: $packagedConfig"
}
Copy-Item -LiteralPath $packagedConfig -Destination (Join-Path $resolvedApp "config.json") -Force

$identityLibrary = Join-Path $DataSourceRoot "identity_library"
$sourceLibrary = Join-Path $DataSourceRoot "source_library"
$queryImages = Join-Path $DataSourceRoot $QueryDirectoryName
Copy-DirectoryContents $identityLibrary (Join-Path $resolvedApp "data\identity_library")
Copy-DirectoryContents $sourceLibrary (Join-Path $resolvedApp "data\source_library")
Copy-DirectoryContents $queryImages (Join-Path $resolvedApp "samples\queries")

$actualModels = @(Get-ChildItem -LiteralPath $ModelSourceDir -File | Sort-Object Name)
$expectedModelNames = @(
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx"
)
$actualModelNames = @($actualModels | ForEach-Object Name)
$unexpected = @($actualModelNames | Where-Object { $_ -notin $expectedModelNames })
$missing = @($expectedModelNames | Where-Object { $_ -notin $actualModelNames })
if ($unexpected.Count -gt 0 -or $missing.Count -gt 0) {
    throw "buffalo_l inventory mismatch. Missing=[$($missing -join ', ')] Unexpected=[$($unexpected -join ', ')]"
}
$modelDestination = Join-Path $resolvedApp "model_runtime\models\buffalo_l"
New-Item -ItemType Directory -Path $modelDestination -Force | Out-Null
foreach ($model in $actualModels) {
    if ($model.Extension -ne ".onnx") {
        throw "Only audited ONNX model files may be copied: $($model.FullName)"
    }
    Copy-Item -LiteralPath $model.FullName -Destination $modelDestination -Force
}

New-Item -ItemType Directory -Path (Join-Path $resolvedApp "licenses") -Force | Out-Null
$distributionNotice = Join-Path $resolvedRepo "packaging\THIRD_PARTY_AND_DATA_NOTICE.md"
if (Test-Path -LiteralPath $distributionNotice -PathType Leaf) {
    Copy-Item -LiteralPath $distributionNotice -Destination (Join-Path $resolvedApp "licenses\THIRD_PARTY_AND_DATA_NOTICE.md") -Force
}
$requiredFiles = @(
    Join-Path $resolvedApp "config.json"
    Join-Path $resolvedApp "workers\identity\identity_search.exe"
    Join-Path $resolvedApp "workers\source\source_search.exe"
)
foreach ($required in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Assembled release is incomplete: $required"
    }
}
Write-Host "Assembled release app: $resolvedApp"
