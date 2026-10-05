#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RunRoot,
    [Parameter(Mandatory)][string]$BusinessEvidencePath,
    [ValidateRange(60, 3600)][int]$StartupTimeoutSeconds = 600,
    [ValidateRange(60, 3600)][int]$UninstallTimeoutSeconds = 1200
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'acceptance_common.ps1')

$run = Get-AcceptanceFullPath $RunRoot
$phase1Path = Join-Path $run 'phase1_state.json'
Assert-Acceptance (Test-Path -LiteralPath $phase1Path -PathType Leaf) '缺少 phase1_state.json。'
$phase1 = Get-Content -LiteralPath $phase1Path -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-Acceptance ([string]$phase1.status -eq 'ready_for_business_acceptance') 'phase1 状态不允许继续。'
$installRoot = Get-AcceptanceFullPath ([string]$phase1.install_root)
Assert-Acceptance (Test-AcceptanceChildPath $installRoot $run) '安装目录不在本次验收目录内。'

$businessPath = Get-AcceptanceFullPath $BusinessEvidencePath
Assert-Acceptance (Test-Path -LiteralPath $businessPath -PathType Leaf) '缺少真实业务验收证据。'
$business = Get-Content -LiteralPath $businessPath -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-Acceptance ([string]$business.status -eq 'passed') '真实业务验收未通过。'
foreach ($module in @('deepfake', 'identity', 'source', 'comprehensive')) {
    $record = $business.$module
    Assert-Acceptance ($null -ne $record) "业务证据缺少模块：$module"
    Assert-Acceptance ([string]$record.status -in @('success', 'partial_success')) "业务模块未成功：$module"
    Assert-Acceptance (-not [string]::IsNullOrWhiteSpace([string]$record.task_id)) "业务模块缺少 task_id：$module"
}
foreach ($format in @('json', 'txt', 'html')) {
    $reportPath = Get-AcceptanceFullPath ([string]$business.reports.$format)
    Assert-Acceptance (Test-Path -LiteralPath $reportPath -PathType Leaf) "报告不存在：$format"
    Assert-Acceptance ((Get-Item -LiteralPath $reportPath).Length -gt 0) "报告为空：$format"
    Assert-Acceptance (Test-AcceptanceChildPath $reportPath ([string]$phase1.data_root)) "报告不在用户数据目录：$format"
}

$originalLocalAppData = [Environment]::GetEnvironmentVariable('LOCALAPPDATA', 'Process')
try {
    $env:LOCALAPPDATA = [string]$phase1.test_local_app_data
    $oldPid = [int]$phase1.launcher.pid
    $oldUrl = [string]$phase1.launcher.url
    $stopProcess = Invoke-AcceptanceProcess `
        (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f [string]$phase1.stop_script), '-Silent') `
        120 $installRoot
    Assert-Acceptance ($stopProcess.exit_code -eq 0) '首次停止脚本失败。'
    Wait-AcceptanceOffline "$oldUrl/_stcore/health" 60

    Start-Process -FilePath ([string]$phase1.main_executable) -WorkingDirectory $installRoot -WindowStyle Hidden | Out-Null
    $restartState = Wait-AcceptanceState ([string]$phase1.data_root) $StartupTimeoutSeconds
    $restartPayload = $restartState.payload
    Assert-Acceptance ([int]$restartPayload.pid -ne $oldPid) '重新启动没有产生新的 PID。'
    $restartUrl = "http://127.0.0.1:$([int]$restartPayload.port)"
    $restartHealth = Wait-AcceptanceHealth "$restartUrl/_stcore/health" $StartupTimeoutSeconds

    $stopAfterRestart = Invoke-AcceptanceProcess `
        (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f [string]$phase1.stop_script), '-Silent') `
        120 $installRoot
    Assert-Acceptance ($stopAfterRestart.exit_code -eq 0) '重启后停止脚本失败。'
    Wait-AcceptanceOffline "$restartUrl/_stcore/health" 60

    $uninstallLog = Join-Path ([string]$phase1.evidence_root) 'uninstall.log'
    $uninstallProcess = Invoke-AcceptanceProcess ([string]$phase1.uninstaller) @(
        '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', "/LOG=`"$uninstallLog`""
    ) $UninstallTimeoutSeconds $installRoot
    Assert-Acceptance ($uninstallProcess.exit_code -eq 0) "卸载器退出码：$($uninstallProcess.exit_code)"

    $deadline = [DateTime]::UtcNow.AddSeconds(120)
    while ([DateTime]::UtcNow -lt $deadline -and (Test-Path -LiteralPath $installRoot)) {
        Start-Sleep -Milliseconds 500
    }
    Assert-Acceptance (-not (Test-Path -LiteralPath $installRoot)) '卸载后安装目录仍存在。'
    Assert-Acceptance (-not (Test-Path -LiteralPath ([string]$phase1.registry_path))) '卸载后注册表项仍存在。'
    foreach ($shortcutPath in @(
        [string]$phase1.shortcuts.start,
        [string]$phase1.shortcuts.stop,
        [string]$phase1.shortcuts.desktop
    )) { Assert-Acceptance (-not (Test-Path -LiteralPath $shortcutPath)) "卸载后快捷方式仍存在：$shortcutPath" }

    foreach ($format in @('json', 'txt', 'html')) {
        Assert-Acceptance (Test-Path -LiteralPath ([string]$business.reports.$format) -PathType Leaf) "卸载误删报告：$format"
    }

    if (-not [string]::IsNullOrWhiteSpace([string]$phase1.protected_install.root)) {
        $protectedAfterPath = Join-Path ([string]$phase1.evidence_root) 'protected_tree_after.csv'
        $protectedAfter = New-AcceptanceTreeManifest ([string]$phase1.protected_install.root) $protectedAfterPath
        Assert-AcceptanceTreeEqual ([string]$phase1.protected_install.before_manifest) $protectedAfterPath
    }
    else { $protectedAfter = $null }

    $summary = [ordered]@{
        schema_version = '1.0'
        status = 'passed'
        completed_at = [DateTimeOffset]::Now.ToString('o')
        run_root = $run
        business_evidence = $businessPath
        restart = [ordered]@{ old_pid = $oldPid; new_pid = [int]$restartPayload.pid; url = $restartUrl; health = $restartHealth }
        stop = [ordered]@{ first = $stopProcess; after_restart = $stopAfterRestart }
        uninstall = $uninstallProcess
        install_root_removed = $true
        registry_removed = $true
        shortcuts_removed = $true
        reports_preserved = $true
        protected_install_unchanged = if ($null -eq $protectedAfter) { $null } else { $true }
    }
    Write-AcceptanceJson (Join-Path $run 'acceptance_summary.json') $summary
    Write-Host "ACCEPTANCE_PASSED=$run"
}
finally {
    if ($null -eq $originalLocalAppData) { Remove-Item Env:LOCALAPPDATA -ErrorAction SilentlyContinue }
    else { $env:LOCALAPPDATA = $originalLocalAppData }
}
