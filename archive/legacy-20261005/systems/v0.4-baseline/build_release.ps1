<#
.SYNOPSIS
Build the Windows x64 onedir payload and single-file Inno Setup installer.

.DESCRIPTION
The three components are deliberately isolated:
  * frontend: packaging/.venv_frontend
  * identity: frozen A_env + packaging/.tools/identity
  * source: frozen B_env + packaging/.tools/source

Useful iteration examples:
  .\build_release.ps1 -SkipInstaller -SkipBusinessTest
  .\build_release.ps1 -SkipClean -SkipBootstrap -SkipBuild -SkipInstaller
#>
[CmdletBinding()]
param(
    [switch]$SkipClean,
    [switch]$SkipBootstrap,
    [switch]$SkipBuild,
    [switch]$SkipAssemble,
    [switch]$SkipSmoke,
    [switch]$SkipFrontendSmoke,
    [switch]$SkipBusinessTest,
    [switch]$SkipInstaller,
    [switch]$SkipMetadata,
    [switch]$RefreshBuildTools,
    [string]$IndexUrl = "https://pypi.org/simple",
    [string]$FrontendBasePython = "D:\PythonProject\.venv\Scripts\python.exe",
    [string]$IdentityPython,
    [string]$SourcePython,
    [string]$DataSourceRoot,
    [string]$ModelSourceDir = "C:\Users\lanji\.insightface\models\buffalo_l",
    [string]$InstallerScript,
    [string]$IsccPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PyInstallerVersion = "6.22.2"
$StreamlitVersion = "1.58.0"
$RepoRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd("\")
$WorkspaceRoot = Split-Path -Parent $RepoRoot
$PackagingRoot = Join-Path $RepoRoot "packaging"
$BuildRoot = Join-Path $PackagingRoot "build"
$DistRoot = Join-Path $PackagingRoot "dist"
$StageRoot = Join-Path $PackagingRoot "stage"
$AppDir = Join-Path $DistRoot "app"
$ReleaseDir = Join-Path $RepoRoot "release_v1.0"
$FrontendVenv = Join-Path $PackagingRoot ".venv_frontend"
$FrontendPython = Join-Path $FrontendVenv "Scripts\python.exe"
$IdentityToolRoot = Join-Path $PackagingRoot ".tools\identity"
$SourceToolRoot = Join-Path $PackagingRoot ".tools\source"

function ConvertFrom-CodePoints([int[]]$CodePoints) {
    return -join @($CodePoints | ForEach-Object { [char]$_ })
}

$ProductName = ConvertFrom-CodePoints @(0x6362, 0x8138, 0x5BFB, 0x6E90, 0x53CC, 0x641C, 0x7D22, 0x7CFB, 0x7EDF)
$MainExeName = "$ProductName.exe"
$QueryDirectoryName = ConvertFrom-CodePoints @(0x6362, 0x8138, 0x67E5, 0x8BE2, 0x56FE)
$SetupPrefix = ConvertFrom-CodePoints @(0x6362, 0x8138, 0x5BFB, 0x6E90)
$SetupSuffix = ConvertFrom-CodePoints @(0x53CC, 0x641C, 0x7D22, 0x7CFB, 0x7EDF)
$SetupFileName = "${SetupPrefix}_${SetupSuffix}_Setup_v1.0.exe"
$SetupPath = Join-Path $ReleaseDir $SetupFileName

if (-not $IdentityPython) {
    $IdentityPython = Join-Path $WorkspaceRoot "first_week\_round1_work\backend\envs\A_env\Scripts\python.exe"
}
if (-not $SourcePython) {
    $SourcePython = Join-Path $WorkspaceRoot "first_week\_round1_work\backend\envs\B_env\Scripts\python.exe"
}
if (-not $DataSourceRoot) {
    $DataSourceRoot = Join-Path $WorkspaceRoot "first_week\data"
}

function Assert-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
}

function Assert-Directory([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label is missing: $Path"
    }
}

function Test-ExactPath([string]$Left, [string]$Right) {
    $leftFull = [System.IO.Path]::GetFullPath($Left).TrimEnd("\")
    $rightFull = [System.IO.Path]::GetFullPath($Right).TrimEnd("\")
    return $leftFull.Equals($rightFull, [System.StringComparison]::OrdinalIgnoreCase)
}

function Reset-SafeDirectory([string]$Path, [string[]]$AllowedPaths) {
    $allowed = $false
    foreach ($candidate in $AllowedPaths) {
        if (Test-ExactPath $Path $candidate) {
            $allowed = $true
            break
        }
    }
    if (-not $allowed) {
        throw "Refusing to remove a directory outside the explicit build allow-list: $Path"
    }
    if (Test-Path -LiteralPath $Path) {
        Write-Host "Cleaning $Path"
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Restore-EnvironmentValue([string]$Name, [object]$Value) {
    if ($null -eq $Value) {
        Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
    }
    else {
        [System.Environment]::SetEnvironmentVariable($Name, [string]$Value, "Process")
    }
}

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)] [string]$FilePath,
        [Parameter(Mandatory = $true)] [string[]]$Arguments,
        [Parameter(Mandatory = $true)] [string]$Label,
        [string]$PythonPathOverride
    )
    Assert-File $FilePath "$Label executable"
    $oldPythonPath = [System.Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
    $oldNoUserSite = [System.Environment]::GetEnvironmentVariable("PYTHONNOUSERSITE", "Process")
    $oldNoBytecode = [System.Environment]::GetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "Process")
    $oldHashSeed = [System.Environment]::GetEnvironmentVariable("PYTHONHASHSEED", "Process")
    $exitCode = -1
    try {
        if ($PythonPathOverride) { $env:PYTHONPATH = [System.IO.Path]::GetFullPath($PythonPathOverride) }
        else { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
        $env:PYTHONNOUSERSITE = "1"
        $env:PYTHONDONTWRITEBYTECODE = "1"
        $env:PYTHONHASHSEED = "0"
        Write-Host "[$Label] $FilePath $($Arguments -join ' ')"
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        Restore-EnvironmentValue "PYTHONPATH" $oldPythonPath
        Restore-EnvironmentValue "PYTHONNOUSERSITE" $oldNoUserSite
        Restore-EnvironmentValue "PYTHONDONTWRITEBYTECODE" $oldNoBytecode
        Restore-EnvironmentValue "PYTHONHASHSEED" $oldHashSeed
    }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode."
    }
}

function Test-PythonRequirement {
    param(
        [Parameter(Mandatory = $true)] [string]$PythonPath,
        [Parameter(Mandatory = $true)] [string]$Code,
        [string]$PythonPathOverride
    )
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { return $false }
    $oldPythonPath = [System.Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
    $oldNoUserSite = [System.Environment]::GetEnvironmentVariable("PYTHONNOUSERSITE", "Process")
    try {
        if ($PythonPathOverride) { $env:PYTHONPATH = [System.IO.Path]::GetFullPath($PythonPathOverride) }
        else { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
        $env:PYTHONNOUSERSITE = "1"
        & $PythonPath -c $Code *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
    finally {
        Restore-EnvironmentValue "PYTHONPATH" $oldPythonPath
        Restore-EnvironmentValue "PYTHONNOUSERSITE" $oldNoUserSite
    }
}

function Ensure-WorkerBuildTool {
    param(
        [Parameter(Mandatory = $true)] [string]$PythonPath,
        [Parameter(Mandatory = $true)] [string]$ToolRoot,
        [Parameter(Mandatory = $true)] [string]$Component
    )
    $versionCode = "import PyInstaller; assert PyInstaller.__version__ == '$PyInstallerVersion'"
    if (Test-PythonRequirement -PythonPath $PythonPath -PythonPathOverride $ToolRoot -Code $versionCode) {
        Write-Host "$Component build tool already pinned to PyInstaller $PyInstallerVersion."
        return
    }
    if ($SkipBootstrap) {
        throw "$Component build tool is absent or has the wrong version; rerun without -SkipBootstrap."
    }
    Reset-SafeDirectory $ToolRoot @($IdentityToolRoot, $SourceToolRoot)
    Invoke-CheckedNative -FilePath $PythonPath -Label "Bootstrap $Component PyInstaller" -Arguments @(
        "-m", "pip", "install",
        "--disable-pip-version-check", "--no-input",
        "--index-url", $IndexUrl,
        "--target", $ToolRoot,
        "--requirement", (Join-Path $PackagingRoot "requirements\worker-build.lock.txt")
    )
    if (-not (Test-PythonRequirement -PythonPath $PythonPath -PythonPathOverride $ToolRoot -Code $versionCode)) {
        throw "$Component PyInstaller bootstrap did not produce version $PyInstallerVersion."
    }
}

function Ensure-FrontendEnvironment {
    $versionCode = "import PyInstaller, streamlit; assert PyInstaller.__version__ == '$PyInstallerVersion'; assert streamlit.__version__ == '$StreamlitVersion'"
    if (Test-PythonRequirement -PythonPath $FrontendPython -Code $versionCode) {
        Write-Host "Frontend environment already pinned (PyInstaller $PyInstallerVersion, Streamlit $StreamlitVersion)."
        return
    }
    if ($SkipBootstrap) {
        throw "Frontend build environment is absent or incorrectly pinned; rerun without -SkipBootstrap."
    }
    if (-not (Test-Path -LiteralPath $FrontendPython -PathType Leaf)) {
        Assert-File $FrontendBasePython "Frontend base Python"
        if (Test-Path -LiteralPath $FrontendVenv) {
            Reset-SafeDirectory $FrontendVenv @($FrontendVenv)
            Remove-Item -LiteralPath $FrontendVenv -Recurse -Force
        }
        Invoke-CheckedNative -FilePath $FrontendBasePython -Label "Create frontend venv" -Arguments @("-m", "venv", $FrontendVenv)
    }
    Invoke-CheckedNative -FilePath $FrontendPython -Label "Bootstrap frontend" -Arguments @(
        "-m", "pip", "install",
        "--disable-pip-version-check", "--no-input", "--upgrade",
        "--index-url", $IndexUrl,
        "--requirement", (Join-Path $PackagingRoot "requirements\frontend-build.lock.txt")
    )
    if (-not (Test-PythonRequirement -PythonPath $FrontendPython -Code $versionCode)) {
        throw "Frontend environment version verification failed."
    }
}

function Invoke-PyInstallerBuild {
    param(
        [Parameter(Mandatory = $true)] [string]$PythonPath,
        [Parameter(Mandatory = $true)] [string]$SpecPath,
        [Parameter(Mandatory = $true)] [string]$WorkPath,
        [Parameter(Mandatory = $true)] [string]$DistPath,
        [Parameter(Mandatory = $true)] [string]$Label,
        [string]$PythonPathOverride
    )
    New-Item -ItemType Directory -Path $WorkPath -Force | Out-Null
    New-Item -ItemType Directory -Path $DistPath -Force | Out-Null
    $oldOfflineFlag = [System.Environment]::GetEnvironmentVariable("NO_ALBUMENTATIONS_UPDATE", "Process")
    try {
        $env:NO_ALBUMENTATIONS_UPDATE = "1"
        Invoke-CheckedNative -FilePath $PythonPath -PythonPathOverride $PythonPathOverride -Label $Label -Arguments @(
            "-B", "-m", "PyInstaller",
            "--noconfirm", "--clean", "--log-level", "INFO",
            "--workpath", $WorkPath,
            "--distpath", $DistPath,
            $SpecPath
        )
    }
    finally {
        Restore-EnvironmentValue "NO_ALBUMENTATIONS_UPDATE" $oldOfflineFlag
    }
}

function Find-IsccExecutable {
    if ($IsccPath) {
        Assert-File $IsccPath "ISCC"
        return [System.IO.Path]::GetFullPath($IsccPath)
    }
    $command = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = New-Object System.Collections.Generic.List[string]
    foreach ($baseAndChild in @(
        @([System.Environment]::GetEnvironmentVariable("ProgramFiles(x86)"), "Inno Setup 6\ISCC.exe"),
        @([System.Environment]::GetEnvironmentVariable("ProgramFiles"), "Inno Setup 6\ISCC.exe"),
        @([System.Environment]::GetEnvironmentVariable("LOCALAPPDATA"), "Programs\Inno Setup 6\ISCC.exe"),
        @([System.Environment]::GetEnvironmentVariable("ChocolateyInstall"), "bin\ISCC.exe")
    )) {
        if ($baseAndChild[0]) {
            $candidates.Add((Join-Path $baseAndChild[0] $baseAndChild[1]))
        }
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return $candidate
        }
    }
    throw "Inno Setup 6 ISCC.exe was not found. Install it from the official source or pass -IsccPath."
}

Assert-File $IdentityPython "Frozen A Python"
Assert-File $SourcePython "Frozen B Python"
Assert-Directory $DataSourceRoot "Release data source"
Assert-Directory $ModelSourceDir "buffalo_l model source"
Assert-File (Join-Path $PackagingRoot "src\app_launcher.py") "Frontend launcher entry"

if (-not $SkipClean) {
    $cleanAllowList = @($BuildRoot, $DistRoot, $StageRoot)
    foreach ($target in $cleanAllowList) {
        Reset-SafeDirectory $target $cleanAllowList
    }
}
else {
    foreach ($target in @($BuildRoot, $DistRoot, $StageRoot)) {
        New-Item -ItemType Directory -Path $target -Force | Out-Null
    }
}

$verifyScript = Join-Path $PackagingRoot "scripts\verify_source_hashes.ps1"
$beforeSnapshot = Join-Path $BuildRoot "source_hashes_before.csv"
$afterSnapshot = Join-Path $BuildRoot "source_hashes_after.csv"
& $verifyScript -RepoRoot $RepoRoot -OutputPath $beforeSnapshot
$sourceVerified = $false
$testResultsPath = Join-Path $StageRoot "dist_test_results.json"

try {
    if ($RefreshBuildTools) {
        foreach ($toolPath in @($IdentityToolRoot, $SourceToolRoot)) {
            Reset-SafeDirectory $toolPath @($IdentityToolRoot, $SourceToolRoot)
        }
        if (Test-Path -LiteralPath $FrontendVenv) {
            Reset-SafeDirectory $FrontendVenv @($FrontendVenv)
            Remove-Item -LiteralPath $FrontendVenv -Recurse -Force
        }
    }

    Ensure-FrontendEnvironment
    Ensure-WorkerBuildTool -PythonPath $IdentityPython -ToolRoot $IdentityToolRoot -Component "identity"
    Ensure-WorkerBuildTool -PythonPath $SourcePython -ToolRoot $SourceToolRoot -Component "source"

    if (-not $SkipBuild) {
        Invoke-PyInstallerBuild `
            -PythonPath $FrontendPython `
            -SpecPath (Join-Path $PackagingRoot "pyinstaller\frontend.spec") `
            -WorkPath (Join-Path $BuildRoot "frontend") `
            -DistPath (Join-Path $StageRoot "frontend") `
            -Label "Build frontend"
        Invoke-PyInstallerBuild `
            -PythonPath $IdentityPython `
            -PythonPathOverride $IdentityToolRoot `
            -SpecPath (Join-Path $PackagingRoot "pyinstaller\identity_search.spec") `
            -WorkPath (Join-Path $BuildRoot "identity") `
            -DistPath (Join-Path $StageRoot "identity") `
            -Label "Build identity worker"
        Invoke-PyInstallerBuild `
            -PythonPath $SourcePython `
            -PythonPathOverride $SourceToolRoot `
            -SpecPath (Join-Path $PackagingRoot "pyinstaller\source_search.spec") `
            -WorkPath (Join-Path $BuildRoot "source") `
            -DistPath (Join-Path $StageRoot "source") `
            -Label "Build source worker"
    }

    $frontendDist = Join-Path $StageRoot "frontend\$ProductName"
    $identityDist = Join-Path $StageRoot "identity\identity_search"
    $sourceDist = Join-Path $StageRoot "source\source_search"
    if (-not $SkipAssemble) {
        & (Join-Path $PackagingRoot "scripts\assemble_app.ps1") `
            -RepoRoot $RepoRoot `
            -FrontendDist $frontendDist `
            -IdentityDist $identityDist `
            -SourceDist $sourceDist `
            -AppDir $AppDir `
            -DataSourceRoot $DataSourceRoot `
            -ModelSourceDir $ModelSourceDir `
            -QueryDirectoryName $QueryDirectoryName

        $licenseScript = Join-Path $PackagingRoot "scripts\export_runtime_licenses.py"
        foreach ($licenseJob in @(
            @($FrontendPython, "frontend"),
            @($IdentityPython, "identity"),
            @($SourcePython, "source")
        )) {
            Invoke-CheckedNative -FilePath $licenseJob[0] -Label "Export $($licenseJob[1]) licenses" -Arguments @(
                "-B", $licenseScript,
                "--output-dir", (Join-Path $AppDir "licenses"),
                "--component", $licenseJob[1]
            )
        }
    }

    & $verifyScript -RepoRoot $RepoRoot -OutputPath $afterSnapshot -CompareTo $beforeSnapshot
    $sourceVerified = $true

    if (-not $SkipSmoke) {
        $testArguments = @{
            AppDir = $AppDir
            MainExeName = $MainExeName
            ProductDataDirName = $ProductName
            ScratchRoot = (Join-Path $StageRoot "smoke")
            ResultsPath = $testResultsPath
            SkipFrontendLaunch = $SkipFrontendSmoke
            SkipBusinessTest = $SkipBusinessTest
        }
        & (Join-Path $PackagingRoot "scripts\test_dist.ps1") @testArguments
    }

    New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null
    $metadataSetupPath = ""
    if (-not $SkipInstaller) {
        if (-not $InstallerScript) {
            $installerRoot = Join-Path $PackagingRoot "installer"
            Assert-Directory $installerRoot "Installer script directory"
            $scripts = @(Get-ChildItem -LiteralPath $installerRoot -Filter "*.iss" -File)
            if ($scripts.Count -ne 1) {
                throw "Expected exactly one .iss file in $installerRoot; found $($scripts.Count). Pass -InstallerScript to disambiguate."
            }
            $InstallerScript = $scripts[0].FullName
        }
        Assert-File $InstallerScript "Inno Setup script"
        $resolvedIscc = Find-IsccExecutable
        if (Test-Path -LiteralPath $SetupPath -PathType Leaf) {
            Remove-Item -LiteralPath $SetupPath -Force
        }
        Push-Location $RepoRoot
        try {
            Invoke-CheckedNative -FilePath $resolvedIscc -Label "Build Inno Setup installer" -Arguments @($InstallerScript)
        }
        finally {
            Pop-Location
        }
        Assert-File $SetupPath "Final Setup artifact"
        $metadataSetupPath = $SetupPath
    }

    if (-not $SkipMetadata) {
        & (Join-Path $PackagingRoot "scripts\write_release_metadata.ps1") `
            -RepoRoot $RepoRoot `
            -AppDir $AppDir `
            -ReleaseDir $ReleaseDir `
            -SetupPath $metadataSetupPath `
            -SourceSnapshotPath $afterSnapshot `
            -TestResultsPath $(if (Test-Path -LiteralPath $testResultsPath -PathType Leaf) { $testResultsPath } else { "" }) `
            -FrontendPython $FrontendPython `
            -IdentityPython $IdentityPython `
            -SourcePython $SourcePython `
            -PyInstallerVersion $PyInstallerVersion `
            -StreamlitVersion $StreamlitVersion
    }

    Write-Host "Build pipeline completed."
    Write-Host "App payload: $AppDir"
    if (-not $SkipInstaller) { Write-Host "Setup: $SetupPath" }
    else { Write-Host "Installer skipped by request." }
}
finally {
    if (-not $sourceVerified -and (Test-Path -LiteralPath $beforeSnapshot -PathType Leaf)) {
        & $verifyScript -RepoRoot $RepoRoot -OutputPath $afterSnapshot -CompareTo $beforeSnapshot
    }
}
