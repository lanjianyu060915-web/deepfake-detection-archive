[CmdletBinding()]
param(
    [string]$AppDistPath,
    [string]$ReleaseDirectory,
    [string]$IsccPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$installerDirectory = $PSScriptRoot
$packagingDirectory = Split-Path -Parent $installerDirectory
$projectDirectory = Split-Path -Parent $packagingDirectory
$issPath = Join-Path $installerDirectory '换脸寻源双搜索系统.iss'
$mainExeName = '换脸寻源双搜索系统.exe'
$outputFileName = '换脸寻源_双搜索系统_Setup_v1.0.exe'

if ([string]::IsNullOrWhiteSpace($AppDistPath)) {
    $AppDistPath = Join-Path $packagingDirectory 'dist\app'
}
if ([string]::IsNullOrWhiteSpace($ReleaseDirectory)) {
    $ReleaseDirectory = Join-Path $projectDirectory 'release_v1.0'
}

$AppDistPath = [IO.Path]::GetFullPath($AppDistPath)
$ReleaseDirectory = [IO.Path]::GetFullPath($ReleaseDirectory)

if (-not (Test-Path -LiteralPath $issPath -PathType Leaf)) {
    throw "找不到 Inno Setup 脚本：$issPath"
}
if (-not (Test-Path -LiteralPath $AppDistPath -PathType Container)) {
    throw "找不到应用构建目录：$AppDistPath"
}

$mainExePath = Join-Path $AppDistPath $mainExeName
if (-not (Test-Path -LiteralPath $mainExePath -PathType Leaf)) {
    throw "应用构建目录中缺少主程序：$mainExePath"
}

if ([string]::IsNullOrWhiteSpace($IsccPath)) {
    $command = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
    $candidates = @()
    if ($null -ne $command) {
        $candidates += $command.Source
    }
    if (-not [string]::IsNullOrWhiteSpace(${env:ProgramFiles(x86)})) {
        $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe')
    }
    if (-not [string]::IsNullOrWhiteSpace($env:ProgramFiles)) {
        $candidates += (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe')
    }
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $candidates += (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe')
    }
    $IsccPath = $candidates |
        Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
        Select-Object -First 1
}

if ([string]::IsNullOrWhiteSpace($IsccPath) -or -not (Test-Path -LiteralPath $IsccPath -PathType Leaf)) {
    throw '未找到 ISCC.exe。请安装官方 Inno Setup 6，或通过 -IsccPath 指定编译器路径。'
}

New-Item -ItemType Directory -Path $ReleaseDirectory -Force | Out-Null

$compilerArguments = @(
    "/DAppDistDir=$AppDistPath"
    "/DReleaseDir=$ReleaseDirectory"
    $issPath
)

Write-Host "正在调用 Inno Setup：$IsccPath"
& $IsccPath @compilerArguments
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup 编译失败，退出码：$LASTEXITCODE"
}

$setupPath = Join-Path $ReleaseDirectory $outputFileName
if (-not (Test-Path -LiteralPath $setupPath -PathType Leaf)) {
    throw "Inno Setup 未生成预期文件：$setupPath"
}

$setupItem = Get-Item -LiteralPath $setupPath
$setupHash = Get-FileHash -LiteralPath $setupPath -Algorithm SHA256
$checksumPath = Join-Path $ReleaseDirectory 'SHA256.txt'
$checksumLine = '{0} *{1}' -f $setupHash.Hash.ToLowerInvariant(), $setupItem.Name
Set-Content -LiteralPath $checksumPath -Value $checksumLine -Encoding UTF8

Write-Host "安装包：$($setupItem.FullName)"
Write-Host ("大小：{0:N0} 字节" -f $setupItem.Length)
Write-Host "SHA256：$($setupHash.Hash)"
Write-Host "校验文件：$checksumPath"
