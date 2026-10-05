[CmdletBinding()]
param(
    [string]$SetupPath,
    [string]$TestRoot,
    [ValidateRange(3, 120)]
    [int]$LaunchTimeoutSeconds = 30,
    [ValidateRange(2, 60)]
    [int]$StabilitySeconds = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$productName = '换脸寻源——AI换脸涉案图像智能鉴伪与原始素材溯源系统'
$shortName = '换脸寻源双搜索系统'
$mainExeName = '换脸寻源双搜索系统.exe'
$appId = '{2A96BB7E-3574-4A10-9CF5-E4B3E67494DA}'
$expectedVersion = '1.0.0'
$outputFileName = '换脸寻源_双搜索系统_Setup_v1.0.exe'
$runId = [Guid]::NewGuid().ToString('N')

$packagingDirectory = Split-Path -Parent $PSScriptRoot
$projectDirectory = Split-Path -Parent $packagingDirectory
if ([string]::IsNullOrWhiteSpace($SetupPath)) {
    $SetupPath = Join-Path (Join-Path $projectDirectory 'release_v1.0') $outputFileName
}
if ([string]::IsNullOrWhiteSpace($TestRoot)) {
    if ([string]::IsNullOrWhiteSpace($env:TEMP)) {
        throw 'TEMP 环境变量为空，无法创建隔离安装测试目录。'
    }
    $TestRoot = Join-Path (Join-Path $env:TEMP '换脸寻源安装验证') $runId
}

$SetupPath = [IO.Path]::GetFullPath($SetupPath)
$TestRoot = [IO.Path]::GetFullPath($TestRoot)
$installDirectory = Join-Path $TestRoot 'app'
$installLog = Join-Path $TestRoot 'install.log'
$uninstallLog = Join-Path $TestRoot 'uninstall.log'
$mainExePath = Join-Path $installDirectory $mainExeName

if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    throw 'LOCALAPPDATA 环境变量为空，无法验证用户数据保留行为。'
}
$originalLocalAppData = [string]$env:LOCALAPPDATA
$testLocalAppData = Join-Path (Join-Path $TestRoot 'user-profile') 'LocalAppData'
$userDataRoot = Join-Path $testLocalAppData $shortName
$sentinelDirectory = Join-Path (Join-Path $userDataRoot '.installer-test') $runId
$sentinelPath = Join-Path $sentinelDirectory 'uninstall-must-preserve.txt'
$sentinelValue = "installer-test-$runId"

$uninstallKeyCandidates = @(
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\${appId}_is1"
    "HKCU:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\${appId}_is1"
)

function ConvertTo-NormalizedPath {
    param([Parameter(Mandatory)][string]$Path)
    return [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
}

function Test-SamePath {
    param(
        [Parameter(Mandatory)][string]$Left,
        [Parameter(Mandatory)][string]$Right
    )
    return [string]::Equals(
        (ConvertTo-NormalizedPath -Path $Left),
        (ConvertTo-NormalizedPath -Path $Right),
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Test-PathInsideDirectory {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Directory
    )
    $normalizedPath = ConvertTo-NormalizedPath -Path $Path
    $normalizedDirectory = ConvertTo-NormalizedPath -Path $Directory
    $prefix = $normalizedDirectory + [IO.Path]::DirectorySeparatorChar
    return $normalizedPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Get-ProductUninstallEntry {
    foreach ($candidate in $uninstallKeyCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            return Get-ItemProperty -LiteralPath $candidate
        }
    }
    return $null
}

function Get-InstalledProcesses {
    $processes = Get-CimInstance -ClassName Win32_Process -ErrorAction SilentlyContinue
    $matchingProcesses = foreach ($process in $processes) {
        if ([string]::IsNullOrWhiteSpace($process.ExecutablePath)) {
            continue
        }
        try {
            if ((Test-SamePath -Left $process.ExecutablePath -Right $mainExePath) -or
                (Test-PathInsideDirectory -Path $process.ExecutablePath -Directory $installDirectory)) {
                $process
            }
        }
        catch {
            # 某些系统进程可能报告非常规设备路径；它们不属于本次隔离安装。
            continue
        }
    }
    return @($matchingProcesses)
}

function Stop-InstalledProcesses {
    foreach ($process in @(Get-InstalledProcesses)) {
        Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction SilentlyContinue
    }
}

function Assert-Shortcut {
    param([Parameter(Mandatory)][string]$ShortcutPath)
    if (-not (Test-Path -LiteralPath $ShortcutPath -PathType Leaf)) {
        throw "缺少快捷方式：$ShortcutPath"
    }
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $null
    try {
        $shortcut = $shell.CreateShortcut($ShortcutPath)
        if (-not (Test-SamePath -Left $shortcut.TargetPath -Right $mainExePath)) {
            throw "快捷方式目标错误：$ShortcutPath -> $($shortcut.TargetPath)"
        }
        if (-not (Test-SamePath -Left $shortcut.WorkingDirectory -Right $installDirectory)) {
            throw "快捷方式工作目录错误：$ShortcutPath -> $($shortcut.WorkingDirectory)"
        }
    }
    finally {
        if ($null -ne $shortcut) {
            [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shortcut) | Out-Null
        }
        if ($null -ne $shell) {
            [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shell) | Out-Null
        }
    }
}

function Invoke-InstallerProcess {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$Action
    )
    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "$Action 失败，退出码：$($process.ExitCode)"
    }
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw '本安装包仅支持 Windows x64，当前系统不是 64 位 Windows。'
}
if (-not (Test-Path -LiteralPath $SetupPath -PathType Leaf)) {
    throw "找不到待测试安装包：$SetupPath"
}

$existingEntry = Get-ProductUninstallEntry
if ($null -ne $existingEntry) {
    $existingLocation = [string]$existingEntry.InstallLocation
    throw "检测到已有正式安装，测试已中止以免覆盖或卸载用户现有程序：$existingLocation"
}

New-Item -ItemType Directory -Path $TestRoot -Force | Out-Null
New-Item -ItemType Directory -Path $testLocalAppData -Force | Out-Null
$env:LOCALAPPDATA = $testLocalAppData
$uninstalledCleanly = $false

try {
    Write-Host "[1/7] 静默安装到隔离目录：$installDirectory"
    $installArguments = @(
        '/VERYSILENT'
        '/SUPPRESSMSGBOXES'
        '/NORESTART'
        '/SP-'
        "/DIR=`"$installDirectory`""
        '/MERGETASKS="desktopicon"'
        "/LOG=`"$installLog`""
    )
    Invoke-InstallerProcess -FilePath $SetupPath -Arguments $installArguments -Action '安装'

    Write-Host '[2/7] 验证安装文件与卸载注册表'
    if (-not (Test-Path -LiteralPath $mainExePath -PathType Leaf)) {
        throw "安装后缺少主程序：$mainExePath"
    }

    $entry = Get-ProductUninstallEntry
    if ($null -eq $entry) {
        throw "未找到卸载注册表项：${appId}_is1"
    }
    if ([string]$entry.DisplayName -ne $productName) {
        throw "注册表 DisplayName 不正确：$($entry.DisplayName)"
    }
    if ([string]$entry.DisplayVersion -ne $expectedVersion) {
        throw "注册表 DisplayVersion 不正确：$($entry.DisplayVersion)"
    }
    if (-not (Test-SamePath -Left ([string]$entry.InstallLocation) -Right $installDirectory)) {
        throw "注册表 InstallLocation 不正确：$($entry.InstallLocation)"
    }
    if ([string]::IsNullOrWhiteSpace([string]$entry.UninstallString)) {
        throw '注册表 UninstallString 为空。'
    }

    Write-Host '[3/7] 验证开始菜单和桌面快捷方式'
    $programsDirectory = [Environment]::GetFolderPath([Environment+SpecialFolder]::Programs)
    $desktopDirectory = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
    $startMenuShortcut = Join-Path (Join-Path $programsDirectory $shortName) "$shortName.lnk"
    $desktopShortcut = Join-Path $desktopDirectory "$shortName.lnk"
    Assert-Shortcut -ShortcutPath $startMenuShortcut
    Assert-Shortcut -ShortcutPath $desktopShortcut

    Write-Host '[4/7] 从开始菜单快捷方式启动并验证进程稳定运行'
    Start-Process -FilePath $startMenuShortcut | Out-Null
    $deadline = [DateTime]::UtcNow.AddSeconds($LaunchTimeoutSeconds)
    $launchedProcesses = @()
    do {
        Start-Sleep -Milliseconds 500
        $launchedProcesses = @(Get-InstalledProcesses)
    } while ($launchedProcesses.Count -eq 0 -and [DateTime]::UtcNow -lt $deadline)

    if ($launchedProcesses.Count -eq 0) {
        throw "在 $LaunchTimeoutSeconds 秒内未检测到安装目录中的运行进程。"
    }
    Start-Sleep -Seconds $StabilitySeconds
    if (@(Get-InstalledProcesses).Count -eq 0) {
        throw "安装后的程序在稳定性观察期（$StabilitySeconds 秒）内退出。"
    }

    Write-Host '[5/7] 创建用户数据哨兵并关闭测试启动的进程'
    New-Item -ItemType Directory -Path $sentinelDirectory -Force | Out-Null
    Set-Content -LiteralPath $sentinelPath -Value $sentinelValue -Encoding UTF8
    Stop-InstalledProcesses

    $stopDeadline = [DateTime]::UtcNow.AddSeconds(15)
    while (@(Get-InstalledProcesses).Count -gt 0 -and [DateTime]::UtcNow -lt $stopDeadline) {
        Start-Sleep -Milliseconds 250
    }
    if (@(Get-InstalledProcesses).Count -gt 0) {
        throw '测试启动的程序无法关闭，不能安全执行卸载。'
    }

    Write-Host '[6/7] 静默卸载'
    $uninstallerPath = Join-Path $installDirectory 'unins000.exe'
    if (-not (Test-Path -LiteralPath $uninstallerPath -PathType Leaf)) {
        throw "缺少卸载程序：$uninstallerPath"
    }
    $uninstallArguments = @(
        '/VERYSILENT'
        '/SUPPRESSMSGBOXES'
        '/NORESTART'
        "/LOG=`"$uninstallLog`""
    )
    Invoke-InstallerProcess -FilePath $uninstallerPath -Arguments $uninstallArguments -Action '卸载'

    Write-Host '[7/7] 验证卸载结果和用户数据保留'
    if (Test-Path -LiteralPath $mainExePath) {
        throw "卸载后主程序仍然存在：$mainExePath"
    }
    if ($null -ne (Get-ProductUninstallEntry)) {
        throw '卸载后注册表项仍然存在。'
    }
    if (Test-Path -LiteralPath $startMenuShortcut) {
        throw "卸载后开始菜单快捷方式仍然存在：$startMenuShortcut"
    }
    if (Test-Path -LiteralPath $desktopShortcut) {
        throw "卸载后桌面快捷方式仍然存在：$desktopShortcut"
    }
    if (-not (Test-Path -LiteralPath $sentinelPath -PathType Leaf)) {
        throw '卸载器删除了用户数据哨兵。'
    }
    $preservedValue = (Get-Content -LiteralPath $sentinelPath -Raw).Trim()
    if ($preservedValue -ne $sentinelValue) {
        throw '卸载后用户数据哨兵内容发生变化。'
    }

    $uninstalledCleanly = $true
    Write-Host '安装/启动/卸载验证通过；卸载器未删除用户数据。' -ForegroundColor Green
    Write-Host "安装日志：$installLog"
    Write-Host "卸载日志：$uninstallLog"
    Write-Host "卸载后保留的用户数据哨兵：$sentinelPath"
}
finally {
    Stop-InstalledProcesses

    if (-not $uninstalledCleanly) {
        $entry = Get-ProductUninstallEntry
        if ($null -ne $entry -and
            -not [string]::IsNullOrWhiteSpace([string]$entry.InstallLocation) -and
            (Test-SamePath -Left ([string]$entry.InstallLocation) -Right $installDirectory)) {
            $fallbackUninstaller = Join-Path $installDirectory 'unins000.exe'
            if (Test-Path -LiteralPath $fallbackUninstaller -PathType Leaf) {
                try {
                    Invoke-InstallerProcess -FilePath $fallbackUninstaller -Arguments @(
                        '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'
                    ) -Action '失败清理卸载'
                }
                catch {
                    Write-Warning "测试失败后的卸载清理也失败：$($_.Exception.Message)"
                }
            }
        }
    }
    $env:LOCALAPPDATA = $originalLocalAppData
}
