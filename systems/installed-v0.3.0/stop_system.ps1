[CmdletBinding()]
param(
    [switch]$Silent
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-CanonicalPath([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Test-SamePath([string]$Left, [string]$Right) {
    return [string]::Equals(
        (Get-CanonicalPath $Left),
        (Get-CanonicalPath $Right),
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Test-SameOrChildPath([string]$Path, [string]$Root) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    $candidate = Get-CanonicalPath $Path
    $parent = Get-CanonicalPath $Root
    return $candidate.Equals($parent, [StringComparison]::OrdinalIgnoreCase) -or
        $candidate.StartsWith($parent + '\', [StringComparison]::OrdinalIgnoreCase)
}

function Get-InstanceKey([string]$AppRoot) {
    $normalized = (Get-CanonicalPath $AppRoot).ToLowerInvariant()
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($normalized)
        $hash = $algorithm.ComputeHash($bytes)
        return ([BitConverter]::ToString($hash).Replace('-', '').Substring(0, 12).ToLowerInvariant())
    }
    finally {
        $algorithm.Dispose()
    }
}

function Show-StopMessage([string]$Message, [bool]$IsError) {
    if ($Silent) {
        if ($IsError) { Write-Error $Message } else { Write-Output $Message }
        return
    }
    Add-Type -AssemblyName PresentationFramework
    $icon = if ($IsError) { 'Error' } else { 'Information' }
    [System.Windows.MessageBox]::Show($Message, '换脸寻源系统 v0.3', 'OK', $icon) | Out-Null
}

try {
    $appRoot = Get-CanonicalPath $PSScriptRoot
    $mainExecutable = Join-Path $appRoot '换脸寻源系统_v0.3.exe'
    $baseDataRoot = if (-not [string]::IsNullOrWhiteSpace($env:FSTS_DATA_ROOT)) {
        Get-CanonicalPath $env:FSTS_DATA_ROOT
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        Join-Path $env:LOCALAPPDATA 'FaceSourceTrace'
    }
    else {
        Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'FaceSourceTrace'
    }
    $stateName = 'launcher_state_v0.3_{0}.json' -f (Get-InstanceKey $appRoot)
    $statePath = Join-Path $baseDataRoot $stateName

    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        Show-StopMessage '换脸寻源系统 v0.3 当前未运行。' $false
        exit 0
    }

    $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $pidValue = [int]$state.pid
    if ($pidValue -le 0) { throw '启动状态 PID 无效。' }
    if (-not (Test-SamePath ([string]$state.app_root) $appRoot)) {
        throw '启动状态不属于当前安装目录。'
    }
    if (-not (Test-SamePath ([string]$state.executable) $mainExecutable)) {
        throw '启动状态中的程序路径不属于当前安装目录。'
    }

    $mainProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
    if ($null -ne $mainProcess) {
        if ([string]::IsNullOrWhiteSpace([string]$mainProcess.ExecutablePath) -or
            -not (Test-SamePath ([string]$mainProcess.ExecutablePath) $mainExecutable)) {
            throw '运行进程路径与当前安装目录不一致，拒绝停止。'
        }

        $allProcesses = @(Get-CimInstance Win32_Process)
        $childrenByParent = @{}
        $processById = @{}
        foreach ($process in $allProcesses) {
            $processById[[int]$process.ProcessId] = $process
            $parentId = [int]$process.ParentProcessId
            if (-not $childrenByParent.ContainsKey($parentId)) {
                $childrenByParent[$parentId] = [Collections.Generic.List[int]]::new()
            }
            $childrenByParent[$parentId].Add([int]$process.ProcessId)
        }
        $queue = [Collections.Generic.Queue[int]]::new()
        $queue.Enqueue($pidValue)
        $descendants = [Collections.Generic.List[int]]::new()
        while ($queue.Count -gt 0) {
            $parentId = $queue.Dequeue()
            if (-not $childrenByParent.ContainsKey($parentId)) { continue }
            foreach ($childId in $childrenByParent[$parentId]) {
                $descendants.Add($childId)
                $queue.Enqueue($childId)
            }
        }
        for ($index = $descendants.Count - 1; $index -ge 0; $index--) {
            $childId = $descendants[$index]
            $child = $processById[$childId]
            if ($null -ne $child -and
                (Test-SameOrChildPath ([string]$child.ExecutablePath) $appRoot)) {
                Stop-Process -Id $childId -Force -ErrorAction SilentlyContinue
            }
        }
        Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
        $deadline = [DateTime]::UtcNow.AddSeconds(30)
        while ([DateTime]::UtcNow -lt $deadline -and
            $null -ne (Get-Process -Id $pidValue -ErrorAction SilentlyContinue)) {
            Start-Sleep -Milliseconds 200
        }
        if ($null -ne (Get-Process -Id $pidValue -ErrorAction SilentlyContinue)) {
            throw '主进程未能在 30 秒内停止。'
        }
    }

    Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
    Show-StopMessage '换脸寻源系统 v0.3 已停止。' $false
    exit 0
}
catch {
    Show-StopMessage ("停止系统失败：{0}" -f $_.Exception.Message) $true
    exit 1
}
