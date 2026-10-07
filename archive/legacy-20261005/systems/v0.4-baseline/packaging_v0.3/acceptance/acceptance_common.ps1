#requires -Version 7.0
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-AcceptanceFullPath([string]$Path) {
    return [IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Test-AcceptanceSamePath([string]$Left, [string]$Right) {
    return [string]::Equals(
        (Get-AcceptanceFullPath $Left),
        (Get-AcceptanceFullPath $Right),
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Test-AcceptanceChildPath([string]$Path, [string]$Root) {
    $candidate = Get-AcceptanceFullPath $Path
    $parent = Get-AcceptanceFullPath $Root
    return $candidate.StartsWith($parent + '\', [StringComparison]::OrdinalIgnoreCase)
}

function Assert-Acceptance([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Write-AcceptanceJson([string]$Path, [object]$Payload) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $temporary = Join-Path $parent ('.{0}.{1}.tmp' -f ([IO.Path]::GetFileName($Path)), [Guid]::NewGuid().ToString('N'))
    try {
        [IO.File]::WriteAllText(
            $temporary,
            (($Payload | ConvertTo-Json -Depth 20) + "`n"),
            [Text.UTF8Encoding]::new($false)
        )
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    }
    finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-AcceptanceProcess {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory)][int]$TimeoutSeconds,
        [string]$WorkingDirectory
    )
    $parameters = @{
        FilePath = $FilePath
        ArgumentList = $Arguments
        PassThru = $true
        WindowStyle = 'Hidden'
    }
    if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
        $parameters.WorkingDirectory = $WorkingDirectory
    }
    $process = Start-Process @parameters
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw "进程超时：$FilePath"
    }
    return [ordered]@{ pid = $process.Id; exit_code = $process.ExitCode }
}

function Get-AcceptanceHealth([string]$Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing
        return [ordered]@{ available = $true; status_code = [int]$response.StatusCode; body = [string]$response.Content }
    }
    catch {
        return [ordered]@{ available = $false; status_code = $null; body = $null; error = $_.Exception.Message }
    }
}

function Wait-AcceptanceHealth([string]$Url, [int]$TimeoutSeconds) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $health = Get-AcceptanceHealth $Url
        if ($health.available -and $health.status_code -eq 200) { return $health }
        Start-Sleep -Milliseconds 250
    }
    throw "服务健康检查超时：$Url"
}

function Wait-AcceptanceOffline([string]$Url, [int]$TimeoutSeconds) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (-not (Get-AcceptanceHealth $Url).available) { return }
        Start-Sleep -Milliseconds 250
    }
    throw "服务未按期下线：$Url"
}

function Wait-AcceptanceState([string]$DataRoot, [int]$TimeoutSeconds) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $states = @(Get-ChildItem -LiteralPath $DataRoot -File -Filter 'launcher_state_v0.4.0_*.json' -ErrorAction SilentlyContinue)
        foreach ($file in $states) {
            try {
                $state = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
                if ([int]$state.pid -gt 0 -and [int]$state.port -gt 0) {
                    return [ordered]@{ path = $file.FullName; payload = $state }
                }
            }
            catch { }
        }
        Start-Sleep -Milliseconds 250
    }
    throw "版本化启动状态未就绪：$DataRoot"
}

function Get-AcceptanceShortcut([string]$Path) {
    $result = [ordered]@{ exists = $false; path = $Path }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $result }
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $null
    try {
        $shortcut = $shell.CreateShortcut($Path)
        $result.exists = $true
        $result.target_path = [string]$shortcut.TargetPath
        $result.working_directory = [string]$shortcut.WorkingDirectory
        $result.arguments = [string]$shortcut.Arguments
        return $result
    }
    finally {
        if ($null -ne $shortcut) { [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shortcut) | Out-Null }
        [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shell) | Out-Null
    }
}

function Start-AcceptanceShortcut([string]$Path) {
    Assert-Acceptance (Test-Path -LiteralPath $Path -PathType Leaf) "快捷方式不存在：$Path"
    $shell = New-Object -ComObject WScript.Shell
    try {
        $result = $shell.Run(('"{0}"' -f $Path), 0, $false)
        Assert-Acceptance ($result -eq 0) "快捷方式启动失败：$Path"
    }
    finally {
        [Runtime.InteropServices.Marshal]::FinalReleaseComObject($shell) | Out-Null
    }
}

function New-AcceptanceTreeManifest([string]$Root, [string]$OutputPath) {
    $rootFull = Get-AcceptanceFullPath $Root
    $records = foreach ($file in Get-ChildItem -LiteralPath $rootFull -File -Recurse -Force | Sort-Object FullName) {
        [pscustomobject]@{
            path = [IO.Path]::GetRelativePath($rootFull, $file.FullName).Replace('\', '/')
            size = [int64]$file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        }
    }
    $records | Export-Csv -LiteralPath $OutputPath -NoTypeInformation -Encoding UTF8
    return [ordered]@{
        path = $OutputPath
        file_count = @($records).Count
        byte_count = [int64](($records | Measure-Object size -Sum).Sum)
        manifest_sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
    }
}

function Assert-AcceptanceTreeEqual([string]$BeforeCsv, [string]$AfterCsv) {
    $before = @(Import-Csv -LiteralPath $BeforeCsv)
    $after = @(Import-Csv -LiteralPath $AfterCsv)
    $difference = @(Compare-Object $before $after -Property path,size,sha256)
    Assert-Acceptance ($difference.Count -eq 0) '受保护旧安装目录发生变化。'
}

function Get-AcceptanceRegistryPath([string]$AppId) {
    return "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$($AppId)_is1"
}
