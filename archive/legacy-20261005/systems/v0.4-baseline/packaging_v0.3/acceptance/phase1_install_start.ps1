#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SetupPath,
    [Parameter(Mandatory)][string]$AcceptanceRoot,
    [string]$ProductVersion = '0.4.0',
    [string]$AppId = '{BA84DD0A-D32A-4A30-88B1-1546A06EE248}',
    [string]$ProductShortName = '换脸寻源系统 v0.4.0',
    [string]$MainExecutableName = '换脸寻源系统_v0.4.0.exe',
    [string]$ProtectedInstallRoot,
    [ValidateRange(60, 3600)][int]$StartupTimeoutSeconds = 600,
    [ValidateRange(60, 7200)][int]$InstallTimeoutSeconds = 2400
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'acceptance_common.ps1')

$originalLocalAppData = [Environment]::GetEnvironmentVariable('LOCALAPPDATA', 'Process')
$runRoot = $null
try {
    $setup = Get-AcceptanceFullPath $SetupPath
    Assert-Acceptance (Test-Path -LiteralPath $setup -PathType Leaf) "安装器不存在：$setup"
    $acceptance = Get-AcceptanceFullPath $AcceptanceRoot
    New-Item -ItemType Directory -Path $acceptance -Force | Out-Null
    $runRoot = Join-Path $acceptance ('run_{0}_{1}' -f (Get-Date -Format 'yyyyMMdd_HHmmss'), [Guid]::NewGuid().ToString('N').Substring(0, 8))
    Assert-Acceptance (Test-AcceptanceChildPath $runRoot $acceptance) '验收运行目录越界。'
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $evidenceRoot = Join-Path $runRoot 'evidence'
    $installRoot = Join-Path $runRoot '中文 空格安装\应用'
    $testLocalAppData = Join-Path $runRoot '用户 配置\LocalAppData'
    New-Item -ItemType Directory -Path $evidenceRoot,$testLocalAppData -Force | Out-Null

    if (-not [string]::IsNullOrWhiteSpace($ProtectedInstallRoot)) {
        $protected = Get-AcceptanceFullPath $ProtectedInstallRoot
        Assert-Acceptance (Test-Path -LiteralPath $protected -PathType Container) "受保护旧安装不存在：$protected"
        Assert-Acceptance (-not (Test-AcceptanceChildPath $installRoot $protected)) '新安装目录与受保护旧安装重叠。'
        $protectedBeforePath = Join-Path $evidenceRoot 'protected_tree_before.csv'
        $protectedBefore = New-AcceptanceTreeManifest $protected $protectedBeforePath
    }
    else {
        $protected = $null
        $protectedBeforePath = $null
        $protectedBefore = $null
    }

    $registryPath = Get-AcceptanceRegistryPath $AppId
    Assert-Acceptance (-not (Test-Path -LiteralPath $registryPath)) "检测到既有 v0.4.0 卸载项：$registryPath"
    $programsRoot = [Environment]::GetFolderPath([Environment+SpecialFolder]::Programs)
    $desktopRoot = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
    $startShortcut = Join-Path (Join-Path $programsRoot $ProductShortName) "$ProductShortName.lnk"
    $stopShortcut = Join-Path (Join-Path $programsRoot $ProductShortName) "停止$ProductShortName.lnk"
    $desktopShortcut = Join-Path $desktopRoot "$ProductShortName.lnk"
    foreach ($shortcut in @($startShortcut, $stopShortcut, $desktopShortcut)) {
        Assert-Acceptance (-not (Test-Path -LiteralPath $shortcut)) "检测到既有 v0.4.0 快捷方式：$shortcut"
    }

    $setupItem = Get-Item -LiteralPath $setup
    $setupRecord = [ordered]@{
        name = $setupItem.Name
        size = [int64]$setupItem.Length
        sha256 = (Get-FileHash -LiteralPath $setup -Algorithm SHA256).Hash
        product_name = ([string]$setupItem.VersionInfo.ProductName).Trim()
        product_version = ([string]$setupItem.VersionInfo.ProductVersion).Trim()
    }
    Assert-Acceptance ($setupRecord.product_version.StartsWith($ProductVersion, [StringComparison]::Ordinal)) '安装器版本不正确。'
    Write-AcceptanceJson (Join-Path $evidenceRoot 'setup.json') $setupRecord

    $env:LOCALAPPDATA = $testLocalAppData
    $installLog = Join-Path $evidenceRoot 'install.log'
    $arguments = @(
        '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-',
        '/NOCLOSEAPPLICATIONS', '/TASKS=desktopicon',
        "/DIR=`"$installRoot`"", "/LOG=`"$installLog`""
    )
    $installProcess = Invoke-AcceptanceProcess $setup $arguments $InstallTimeoutSeconds (Split-Path -Parent $setup)
    Assert-Acceptance ($installProcess.exit_code -eq 0) "安装器退出码：$($installProcess.exit_code)"
    $installRoot = Get-AcceptanceFullPath $installRoot

    $mainExecutable = Join-Path $installRoot $MainExecutableName
    $stopScript = Join-Path $installRoot 'stop_system.ps1'
    $uninstaller = Join-Path $installRoot 'unins000.exe'
    $configPath = Join-Path $installRoot 'config.json'
    foreach ($required in @(
        $mainExecutable, $stopScript, $uninstaller, $configPath,
        (Join-Path $installRoot 'workers\identity\identity_worker.exe'),
        (Join-Path $installRoot 'workers\source\source_search.exe'),
        (Join-Path $installRoot 'python_runtime\python.exe'),
        (Join-Path $installRoot 'resources_manifest.sha256.json')
    )) { Assert-Acceptance (Test-Path -LiteralPath $required -PathType Leaf) "安装文件缺失：$required" }

    $config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-Acceptance ([string]$config.schema_version -eq '3.0') '安装配置 schema_version 不是 3.0。'
    Assert-Acceptance ([bool]$config.features.real_modules_default) '正式模块不是默认入口。'
    Assert-Acceptance (-not [bool]$config.features.mock_enabled) '正式安装配置启用了 Mock。'
    $configText = $config | ConvertTo-Json -Depth 20
    Assert-Acceptance (-not ($configText -match '"[A-Za-z]:[\\/]')) '安装配置包含固定盘符绝对路径。'
    $preferredPort = [int]$config.frontend.port
    $preexistingPortOwners = @(
        Get-NetTCPConnection -State Listen -LocalPort $preferredPort -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )

    Assert-Acceptance (Test-Path -LiteralPath $registryPath) '安装后缺少 HKCU 卸载项。'
    foreach ($record in @(
        @{ Path = $startShortcut; Target = $mainExecutable },
        @{ Path = $desktopShortcut; Target = $mainExecutable }
    )) {
        $shortcut = Get-AcceptanceShortcut $record.Path
        Assert-Acceptance ([bool]$shortcut.exists) "快捷方式不存在：$($record.Path)"
        Assert-Acceptance (Test-AcceptanceSamePath $shortcut.target_path $record.Target) "快捷方式目标错误：$($record.Path)"
        Assert-Acceptance (Test-AcceptanceSamePath $shortcut.working_directory $installRoot) "快捷方式工作目录错误：$($record.Path)"
    }
    Assert-Acceptance (Test-Path -LiteralPath $stopShortcut -PathType Leaf) '停止快捷方式不存在。'

    Start-AcceptanceShortcut $startShortcut
    $dataRoot = Join-Path $testLocalAppData 'FaceSourceTrace'
    $stateRecord = Wait-AcceptanceState $dataRoot $StartupTimeoutSeconds
    $state = $stateRecord.payload
    Assert-Acceptance (Test-AcceptanceSamePath ([string]$state.app_root) $installRoot) '启动状态 app_root 错误。'
    Assert-Acceptance (Test-AcceptanceSamePath ([string]$state.executable) $mainExecutable) '启动状态 executable 错误。'
    if ($preexistingPortOwners.Count -gt 0) {
        Assert-Acceptance ([int]$state.port -ne $preferredPort) '首选端口已被旧进程占用，但新实例仍复用了该端口。'
    }
    $url = "http://127.0.0.1:$([int]$state.port)"
    $health = Wait-AcceptanceHealth "$url/_stcore/health" $StartupTimeoutSeconds
    $selectedPortOwners = @(
        Get-NetTCPConnection -State Listen -LocalPort ([int]$state.port) -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    Assert-Acceptance ($selectedPortOwners.Count -eq 1) '新实例监听端口的进程归属不唯一。'
    Assert-Acceptance ([int]$selectedPortOwners[0] -eq [int]$state.pid) '新实例监听端口不属于启动状态记录的进程。'
    Start-AcceptanceShortcut $desktopShortcut
    Start-Sleep -Seconds 3
    $stateAfterDuplicate = Get-Content -LiteralPath $stateRecord.path -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-Acceptance ([int]$stateAfterDuplicate.pid -eq [int]$state.pid) '重复启动产生了第二个主进程。'

    $phase1 = [ordered]@{
        schema_version = '1.0'
        status = 'ready_for_business_acceptance'
        completed_at = [DateTimeOffset]::Now.ToString('o')
        run_root = $runRoot
        evidence_root = $evidenceRoot
        setup = $setupRecord
        install_root = $installRoot
        test_local_app_data = $testLocalAppData
        data_root = $dataRoot
        main_executable = $mainExecutable
        stop_script = $stopScript
        uninstaller = $uninstaller
        registry_path = $registryPath
        shortcuts = [ordered]@{ start = $startShortcut; stop = $stopShortcut; desktop = $desktopShortcut }
        launcher = [ordered]@{ state_path = $stateRecord.path; pid = [int]$state.pid; port = [int]$state.port; url = $url; health = $health }
        port_selection = [ordered]@{
            preferred_port = $preferredPort
            preexisting_owners = $preexistingPortOwners
            selected_port = [int]$state.port
            selected_owners = $selectedPortOwners
            exclusive_fallback_validated = [bool]($preexistingPortOwners.Count -gt 0)
        }
        protected_install = [ordered]@{ root = $protected; before_manifest = $protectedBeforePath; before = $protectedBefore }
        required_business_evidence = @('deepfake', 'identity', 'source', 'comprehensive', 'reports')
    }
    Write-AcceptanceJson (Join-Path $runRoot 'phase1_state.json') $phase1
    Write-Host "PHASE1_READY=$runRoot"
}
catch {
    if ($null -ne $runRoot -and (Test-Path -LiteralPath $runRoot -PathType Container)) {
        Write-AcceptanceJson (Join-Path $runRoot 'phase1_failure.json') ([ordered]@{
            schema_version = '1.0'; status = 'failed'; error = $_.Exception.Message
        })
    }
    throw
}
finally {
    if ($null -eq $originalLocalAppData) { Remove-Item Env:LOCALAPPDATA -ErrorAction SilentlyContinue }
    else { $env:LOCALAPPDATA = $originalLocalAppData }
}
