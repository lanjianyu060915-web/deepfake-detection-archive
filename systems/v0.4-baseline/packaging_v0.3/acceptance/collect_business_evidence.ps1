#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RunRoot,
    [string]$DeepfakeTaskId,
    [string]$IdentityTaskId,
    [string]$SourceTaskId,
    [string]$ComprehensiveTaskId,
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'acceptance_common.ps1')

$script:AllowedTaskStatuses = @('success', 'partial_success')
$script:AllowedEventStatuses = @('pending', 'running', 'success', 'warning', 'error', 'skipped')
$script:KnownModules = @('deepfake', 'identity', 'source')

function Get-RequiredProperty {
    param(
        [Parameter(Mandatory)][object]$InputObject,
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Context
    )
    $property = $InputObject.PSObject.Properties[$Name]
    Assert-Acceptance ($null -ne $property) "$Context 缺少字段：$Name"
    return $property.Value
}

function Read-StrictUtf8Text {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Context
    )
    $fullPath = Get-AcceptanceFullPath $Path
    Assert-Acceptance (Test-Path -LiteralPath $fullPath -PathType Leaf) "$Context 不存在：$fullPath"
    $item = Get-Item -LiteralPath $fullPath
    Assert-Acceptance ($item.Length -gt 0) "$Context 为空：$fullPath"
    try {
        $bytes = [IO.File]::ReadAllBytes($fullPath)
        $text = [Text.UTF8Encoding]::new($false, $true).GetString($bytes)
    }
    catch {
        throw "$Context 不是有效 UTF-8：$fullPath；$($_.Exception.Message)"
    }
    if ($text.Length -gt 0 -and $text[0] -eq [char]0xFEFF) {
        $text = $text.Substring(1)
    }
    Assert-Acceptance (-not [string]::IsNullOrWhiteSpace($text)) "$Context 只有空白内容：$fullPath"
    return $text
}

function Read-StrictJson {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Context
    )
    $text = Read-StrictUtf8Text $Path $Context
    try {
        $payload = $text | ConvertFrom-Json -Depth 100
    }
    catch {
        throw "$Context 不是有效 JSON：$Path；$($_.Exception.Message)"
    }
    Assert-Acceptance ($null -ne $payload) "$Context JSON 值为空：$Path"
    return ,$payload
}

function Get-NormalizedModules {
    param(
        [Parameter(Mandatory)][object]$Value,
        [Parameter(Mandatory)][string]$Context
    )
    $modules = @($Value | ForEach-Object { [string]$_ })
    Assert-Acceptance ($modules.Count -gt 0) "$Context selected_modules 为空。"
    Assert-Acceptance (@($modules | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count -eq 0) `
        "$Context selected_modules 包含空值。"
    $unknown = @($modules | Where-Object { $_ -notin $script:KnownModules })
    Assert-Acceptance ($unknown.Count -eq 0) "$Context selected_modules 包含未知模块：$($unknown -join ', ')"
    Assert-Acceptance (@($modules | Sort-Object -Unique).Count -eq $modules.Count) `
        "$Context selected_modules 包含重复模块。"
    return ,$modules
}

function Test-ExactModules {
    param([string[]]$Actual, [string[]]$Expected)
    if ($Actual.Count -ne $Expected.Count) { return $false }
    return @($Actual | Where-Object { $_ -notin $Expected }).Count -eq 0
}

function Assert-ExactModules {
    param(
        [string[]]$Actual,
        [string[]]$Expected,
        [string]$Context
    )
    Assert-Acceptance (Test-ExactModules $Actual $Expected) `
        "$Context selected_modules 不匹配；实际=$($Actual -join ','), 期望=$($Expected -join ',')"
}

function Find-MockMarker {
    param(
        [AllowNull()][object]$Value,
        [string]$Path = '$'
    )
    if ($null -eq $Value) { return $null }
    if ($Value -is [string]) {
        if ($Value -match '(?i)(^|[^a-z])mock([^a-z]|$)|dev[_ -]?mock|simulated[ _-]?data|simulation[ _-]?mode|模拟数据') {
            return "$Path=$Value"
        }
        return $null
    }
    if ($Value -is [System.Collections.IDictionary]) {
        foreach ($key in $Value.Keys) {
            $name = [string]$key
            $child = $Value[$key]
            if ($name -match '(?i)(^|_)(is_)?mock($|_)|(^|_)simulated($|_)|(^|_)simulation($|_)') {
                if (($child -is [bool] -and $child) -or
                    ($child -is [string] -and -not [string]::IsNullOrWhiteSpace($child) -and $child -notmatch '^(?i:false|disabled|none)$') -or
                    ($child -is [ValueType] -and $child -isnot [bool] -and [double]$child -ne 0)) {
                    return "$Path.$name=$child"
                }
            }
            $found = Find-MockMarker $child "$Path.$name"
            if ($null -ne $found) { return $found }
        }
        return $null
    }
    if ($Value -is [pscustomobject]) {
        foreach ($property in $Value.PSObject.Properties) {
            $name = [string]$property.Name
            $child = $property.Value
            if ($name -match '(?i)(^|_)(is_)?mock($|_)|(^|_)simulated($|_)|(^|_)simulation($|_)') {
                if (($child -is [bool] -and $child) -or
                    ($child -is [string] -and -not [string]::IsNullOrWhiteSpace($child) -and $child -notmatch '^(?i:false|disabled|none)$') -or
                    ($child -is [ValueType] -and $child -isnot [bool] -and [double]$child -ne 0)) {
                    return "$Path.$name=$child"
                }
            }
            $found = Find-MockMarker $child "$Path.$name"
            if ($null -ne $found) { return $found }
        }
        return $null
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        $index = 0
        foreach ($item in $Value) {
            $found = Find-MockMarker $item "$Path[$index]"
            if ($null -ne $found) { return $found }
            $index++
        }
    }
    return $null
}

function Assert-NoMockMarkers {
    param(
        [Parameter(Mandatory)][object]$Value,
        [Parameter(Mandatory)][string]$Context
    )
    $marker = Find-MockMarker $Value
    Assert-Acceptance ($null -eq $marker) "$Context 检测到 Mock/模拟数据标记：$marker"
}

function Get-FileEvidence {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$TaskRoot
    )
    $fullPath = Get-AcceptanceFullPath $Path
    Assert-Acceptance (Test-AcceptanceChildPath $fullPath $TaskRoot) "证据文件越出任务目录：$fullPath"
    $item = Get-Item -LiteralPath $fullPath
    return [ordered]@{
        path = $fullPath
        size = [int64]$item.Length
        sha256 = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash
    }
}

function Get-SafeTaskDirectory {
    param(
        [Parameter(Mandatory)][string]$TasksRoot,
        [Parameter(Mandatory)][string]$TaskId
    )
    Assert-Acceptance (-not [string]::IsNullOrWhiteSpace($TaskId)) 'task_id 不能为空。'
    Assert-Acceptance ($TaskId -match '^[A-Za-z0-9][A-Za-z0-9._-]*$') "task_id 包含不安全字符：$TaskId"
    Assert-Acceptance ($TaskId -notin @('.', '..')) "task_id 非法：$TaskId"
    $directory = Get-AcceptanceFullPath (Join-Path $TasksRoot $TaskId)
    Assert-Acceptance (Test-AcceptanceChildPath $directory $TasksRoot) "任务目录越界：$directory"
    Assert-Acceptance (Test-Path -LiteralPath $directory -PathType Container) "任务目录不存在：$directory"
    $item = Get-Item -LiteralPath $directory -Force
    Assert-Acceptance (-not (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) `
        "任务目录不得是重解析点：$directory"
    return $directory
}

function Resolve-LatestTaskId {
    param(
        [Parameter(Mandatory)][string]$TasksRoot,
        [Parameter(Mandatory)][string[]]$ExpectedModules,
        [Parameter(Mandatory)][string]$Label
    )
    $candidates = @()
    foreach ($directory in @(Get-ChildItem -LiteralPath $TasksRoot -Directory -Force | Sort-Object Name)) {
        Assert-Acceptance (-not (($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) `
            "任务根目录包含重解析点，拒绝自动选择：$($directory.FullName)"
        $statusPath = Join-Path $directory.FullName 'task_status.json'
        Assert-Acceptance (Test-Path -LiteralPath $statusPath -PathType Leaf) `
            "任务目录缺少 task_status.json，无法无歧义自动选择：$($directory.FullName)"
        $status = Read-StrictJson $statusPath "任务状态 $($directory.Name)"
        Assert-NoMockMarkers $status "任务状态 $($directory.Name)"
        $taskId = [string](Get-RequiredProperty $status 'task_id' "任务状态 $($directory.Name)")
        Assert-Acceptance ([string]::Equals($taskId, $directory.Name, [StringComparison]::Ordinal)) `
            "task_status.task_id 与目录名不一致：$($directory.FullName)"
        $modules = Get-NormalizedModules (Get-RequiredProperty $status 'selected_modules' "任务状态 $taskId") "任务状态 $taskId"
        if (-not (Test-ExactModules $modules $ExpectedModules)) { continue }
        $finishedText = [string](Get-RequiredProperty $status 'finished_at' "任务状态 $taskId")
        $finished = [DateTimeOffset]::MinValue
        Assert-Acceptance ([DateTimeOffset]::TryParse($finishedText, [ref]$finished)) `
            "任务 finished_at 无效：$taskId"
        $candidates += [pscustomobject]@{ task_id = $taskId; finished_at = $finished }
    }
    Assert-Acceptance ($candidates.Count -gt 0) `
        "未找到 selected_modules=$($ExpectedModules -join ',') 的 $Label 任务。"
    $ordered = @($candidates | Sort-Object finished_at -Descending)
    if ($ordered.Count -gt 1) {
        Assert-Acceptance ($ordered[0].finished_at -ne $ordered[1].finished_at) `
            "$Label 最新任务 finished_at 并列，无法无歧义选择：$($ordered[0].task_id), $($ordered[1].task_id)"
    }
    return [string]$ordered[0].task_id
}

function Assert-StatusAllowed {
    param([string]$Status, [string]$Context)
    Assert-Acceptance ($Status -in $script:AllowedTaskStatuses) `
        "$Context 状态不是 success/partial_success：$Status"
}

function Get-ExpectedProgressStatus {
    param(
        [Parameter(Mandatory)][ValidateSet('success', 'partial_success', 'error')][string]$TaskStatus
    )
    switch ($TaskStatus) {
        'success' { return 'success' }
        'partial_success' { return 'warning' }
        'error' { return 'error' }
    }
}

function Get-ProgressEvidence {
    param(
        [Parameter(Mandatory)][string]$TaskDirectory,
        [Parameter(Mandatory)][string]$TaskId
    )
    $eventsPath = Join-Path $TaskDirectory 'logs\progress_events.jsonl'
    $latestPath = Join-Path $TaskDirectory 'logs\progress_latest.json'
    $text = Read-StrictUtf8Text $eventsPath "进度事件 $TaskId"
    $events = @()
    $eventIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $previousTimestamp = [DateTimeOffset]::MinValue
    $lineNumber = 0
    foreach ($line in @($text -split "`r?`n")) {
        $lineNumber++
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        try { $event = $line | ConvertFrom-Json -Depth 100 }
        catch { throw "进度事件 JSONL 第 $lineNumber 行无效：$eventsPath；$($_.Exception.Message)" }
        Assert-NoMockMarkers $event "进度事件 $TaskId 第 $lineNumber 行"
        foreach ($field in @(
            'event_id', 'timestamp', 'task_id', 'module', 'stage', 'status', 'message',
            'progress_current', 'progress_total', 'progress_percent', 'elapsed_seconds', 'details'
        )) { $null = Get-RequiredProperty $event $field "进度事件 $TaskId 第 $lineNumber 行" }
        $eventId = [string]$event.event_id
        Assert-Acceptance (-not [string]::IsNullOrWhiteSpace($eventId)) "进度事件 event_id 为空：$TaskId 第 $lineNumber 行"
        Assert-Acceptance ($eventIds.Add($eventId)) "进度事件 event_id 重复：$TaskId/$eventId"
        Assert-Acceptance ([string]$event.task_id -eq $TaskId) "进度事件 task_id 不一致：$TaskId 第 $lineNumber 行"
        Assert-Acceptance ([string]$event.status -in $script:AllowedEventStatuses) `
            "进度事件状态非法：$TaskId 第 $lineNumber 行"
        $timestamp = [DateTimeOffset]::MinValue
        Assert-Acceptance ([DateTimeOffset]::TryParse([string]$event.timestamp, [ref]$timestamp)) `
            "进度事件时间戳无效：$TaskId 第 $lineNumber 行"
        Assert-Acceptance ($timestamp -ge $previousTimestamp) "进度事件时间戳倒退：$TaskId 第 $lineNumber 行"
        $previousTimestamp = $timestamp
        if ($null -ne $event.progress_percent) {
            $percent = [double]$event.progress_percent
            Assert-Acceptance ($percent -ge 0 -and $percent -le 100) `
                "进度百分比越界：$TaskId 第 $lineNumber 行"
        }
        $events += $event
    }
    Assert-Acceptance ($events.Count -gt 0) "进度事件为空：$TaskId"
    $last = $events[-1]
    Assert-Acceptance ([string]$last.module -eq 'overall') "最后进度事件不是 overall：$TaskId"
    Assert-Acceptance ([string]$last.stage -eq 'completed') "最后进度事件不是 completed：$TaskId"
    Assert-Acceptance ([string]$last.status -in @('success', 'warning', 'error')) `
        "最终进度状态不是 success/warning/error：$TaskId/$($last.status)"
    Assert-Acceptance ($null -ne $last.progress_percent -and [double]$last.progress_percent -eq 100.0) `
        "最终进度不是 100%：$TaskId"

    $latest = Read-StrictJson $latestPath "最新进度 $TaskId"
    Assert-NoMockMarkers $latest "最新进度 $TaskId"
    Assert-Acceptance ([string]$latest.event_id -eq [string]$last.event_id) `
        "progress_latest 与 JSONL 最后一条事件不一致：$TaskId"
    Assert-Acceptance ([string]$latest.stage -eq 'completed' -and [string]$latest.module -eq 'overall') `
        "progress_latest 不是 overall/completed：$TaskId"

    return [ordered]@{
        event_count = $events.Count
        completed_event_id = [string]$last.event_id
        completed_at = [string]$last.timestamp
        status = [string]$last.status
        elapsed_seconds = [double]$last.elapsed_seconds
        events = $events
        files = [ordered]@{
            jsonl = Get-FileEvidence $eventsPath $TaskDirectory
            latest = Get-FileEvidence $latestPath $TaskDirectory
        }
    }
}

function Assert-RealDeepfakeEvidence {
    param([object]$Result, [string]$TaskId)
    $metrics = Get-RequiredProperty $Result 'metrics' "AIGC 结果 $TaskId"
    foreach ($name in @('spatial_score', 'frequency_score', 'fusion_score', 'decision_threshold', 'valid_frame_count')) {
        $null = Get-RequiredProperty $metrics $name "AIGC metrics $TaskId"
    }
    Assert-Acceptance ([int]$metrics.valid_frame_count -gt 0) "AIGC 没有有效帧/人脸证据：$TaskId"
    $provenance = Get-RequiredProperty $Result 'provenance' "AIGC 结果 $TaskId"
    Assert-Acceptance ([string]$provenance.input_sha256 -match '^[0-9a-fA-F]{64}$') "AIGC 输入 SHA-256 无效：$TaskId"
    $modelHashes = Get-RequiredProperty $provenance 'model_sha256' "AIGC provenance $TaskId"
    foreach ($name in @('spatial', 'frequency', 'fusion_config')) {
        $hash = [string](Get-RequiredProperty $modelHashes $name "AIGC 模型哈希 $TaskId")
        Assert-Acceptance ($hash -match '^[0-9a-fA-F]{64}$') "AIGC 模型哈希无效：$TaskId/$name"
    }
}

function Assert-RealIdentityEvidence {
    param([object]$Result, [string]$TaskId)
    $method = Get-RequiredProperty $Result 'method' "身份结果 $TaskId"
    Assert-Acceptance ([string]$method.model_name -eq 'buffalo_l') "身份模型不是 buffalo_l：$TaskId"
    Assert-Acceptance ([int]$method.embedding_dimension -eq 512) "身份特征维度不是 512：$TaskId"
    $metrics = Get-RequiredProperty $Result 'metrics' "身份结果 $TaskId"
    Assert-Acceptance ([int]$metrics.library_identity_count -gt 0) "身份库人数为 0：$TaskId"
    Assert-Acceptance ([int]$metrics.valid_embedding_count -gt 0) "身份库有效 embedding 为 0：$TaskId"
    Assert-Acceptance (@($Result.candidates).Count -gt 0) "身份搜索没有候选：$TaskId"
    $provenance = Get-RequiredProperty $Result 'provenance' "身份结果 $TaskId"
    Assert-Acceptance (-not [string]::IsNullOrWhiteSpace([string]$provenance.model_fingerprint)) `
        "身份模型指纹为空：$TaskId"
}

function Get-RealSourceProgress {
    param(
        [object]$SourceResult,
        [object[]]$Events,
        [string]$TaskId
    )
    $metrics = Get-RequiredProperty $SourceResult 'metrics' "底图结果 $TaskId"
    foreach ($name in @('library_file_count', 'library_decoded', 'library_skipped_count', 'shortlist_size')) {
        $null = Get-RequiredProperty $metrics $name "底图 metrics $TaskId"
    }
    $fileCount = [int]$metrics.library_file_count
    $decoded = [int]$metrics.library_decoded
    $skipped = [int]$metrics.library_skipped_count
    $shortlist = [int]$metrics.shortlist_size
    Assert-Acceptance ($fileCount -gt 0 -and $decoded -gt 0) "底图库没有实际读取图片：$TaskId"
    Assert-Acceptance ($decoded + $skipped -eq $fileCount) `
        "底图库读取记账不一致：$TaskId；file_count=$fileCount decoded=$decoded skipped=$skipped"
    Assert-Acceptance ($shortlist -gt 0) "底图粗排 shortlist 为空：$TaskId"
    Assert-Acceptance (@($SourceResult.candidates).Count -gt 0) "底图搜索没有候选：$TaskId"

    $scanEvents = @($Events | Where-Object { [string]$_.module -eq 'source' -and [string]$_.stage -eq 'scan_source_library' })
    Assert-Acceptance ($scanEvents.Count -gt 0) "缺少底图库实际扫描进度：$TaskId"
    $scanTotals = @($scanEvents | ForEach-Object { [int]$_.progress_total } | Sort-Object -Unique)
    Assert-Acceptance ($scanTotals.Count -eq 1) "底图库扫描总数发生变化：$TaskId"
    $lastScan = $scanEvents[-1]
    Assert-Acceptance ([int]$lastScan.progress_current -eq [int]$lastScan.progress_total) `
        "底图库未扫描到总数：$TaskId"
    Assert-Acceptance ([int]$lastScan.progress_total -eq $decoded) `
        "底图库扫描总数与 library_decoded 不一致：$TaskId"

    $reviewEvents = @($Events | Where-Object { [string]$_.module -eq 'source' -and [string]$_.stage -eq 'source_geometric_review' })
    Assert-Acceptance ($reviewEvents.Count -gt 0) "缺少 SIFT/RANSAC 几何复核进度：$TaskId"
    $reviewTotals = @($reviewEvents | ForEach-Object { [int]$_.progress_total } | Sort-Object -Unique)
    Assert-Acceptance ($reviewTotals.Count -eq 1) "几何复核总数发生变化：$TaskId"
    $lastReview = $reviewEvents[-1]
    Assert-Acceptance ([int]$lastReview.progress_current -eq [int]$lastReview.progress_total) `
        "底图几何复核未完成 shortlist：$TaskId"
    Assert-Acceptance ([int]$lastReview.progress_total -eq $shortlist) `
        "几何复核总数与 shortlist_size 不一致：$TaskId"

    return [ordered]@{
        library_file_count = $fileCount
        library_decoded = $decoded
        library_skipped_count = $skipped
        scan_event_count = $scanEvents.Count
        scan_completed = [int]$lastScan.progress_current
        scan_total = [int]$lastScan.progress_total
        shortlist_size = $shortlist
        review_event_count = $reviewEvents.Count
        review_completed = [int]$lastReview.progress_current
        review_total = [int]$lastReview.progress_total
    }
}

function Test-TaskEvidence {
    param(
        [Parameter(Mandatory)][string]$TasksRoot,
        [Parameter(Mandatory)][string]$TaskId,
        [Parameter(Mandatory)][string[]]$ExpectedModules,
        [Parameter(Mandatory)][string]$Label
    )
    $taskDirectory = Get-SafeTaskDirectory $TasksRoot $TaskId
    $paths = [ordered]@{
        task_status = Join-Path $taskDirectory 'task_status.json'
        unified = Join-Path $taskDirectory 'unified_result.json'
        deepfake = Join-Path $taskDirectory 'detection_result.json'
        identity = Join-Path $taskDirectory 'identity_result.json'
        source = Join-Path $taskDirectory 'source_result.json'
        report_json = Join-Path $taskDirectory 'final_report.json'
        report_txt = Join-Path $taskDirectory "检测结果_$TaskId.txt"
        report_html = Join-Path $taskDirectory "检测结果_$TaskId.html"
    }
    $status = Read-StrictJson $paths.task_status "$Label task_status"
    $unified = Read-StrictJson $paths.unified "$Label unified_result"
    $results = [ordered]@{
        deepfake = Read-StrictJson $paths.deepfake "$Label detection_result"
        identity = Read-StrictJson $paths.identity "$Label identity_result"
        source = Read-StrictJson $paths.source "$Label source_result"
    }
    $report = Read-StrictJson $paths.report_json "$Label final_report"
    $reportTxt = Read-StrictUtf8Text $paths.report_txt "$Label TXT 报告"
    $reportHtml = Read-StrictUtf8Text $paths.report_html "$Label HTML 报告"
    foreach ($payload in @($status, $unified, $results.deepfake, $results.identity, $results.source, $report)) {
        Assert-NoMockMarkers $payload "$Label JSON 证据"
    }
    Assert-NoMockMarkers $reportTxt "$Label TXT 报告"
    Assert-NoMockMarkers $reportHtml "$Label HTML 报告"

    Assert-Acceptance ([string]$status.schema_version -eq '3.0') "$Label task_status schema_version 不是 3.0。"
    Assert-Acceptance ([string]$status.task_id -eq $TaskId) "$Label task_status.task_id 不一致。"
    Assert-Acceptance (Test-AcceptanceSamePath ([string]$status.task_dir) $taskDirectory) `
        "$Label task_status.task_dir 与实际任务目录不一致。"
    Assert-Acceptance (Test-AcceptanceChildPath ([string]$status.query_image) $taskDirectory) `
        "$Label query_image 不在任务目录。"
    Assert-Acceptance (Test-Path -LiteralPath ([string]$status.query_image) -PathType Leaf) `
        "$Label query_image 不存在。"
    $statusModules = Get-NormalizedModules $status.selected_modules "$Label task_status"
    Assert-ExactModules $statusModules $ExpectedModules "$Label task_status"
    $taskStatus = [string]$status.status
    Assert-StatusAllowed $taskStatus "$Label task_status"

    Assert-Acceptance ([string]$unified.schema_version -eq '3.0') "$Label unified_result schema_version 不是 3.0。"
    Assert-Acceptance ([string]$unified.task_id -eq $TaskId) "$Label unified_result.task_id 不一致。"
    Assert-Acceptance ([string]$unified.status -eq $taskStatus) "$Label unified_result.status 与 task_status 不一致。"
    $unifiedModules = Get-NormalizedModules $unified.selected_modules "$Label unified_result"
    Assert-ExactModules $unifiedModules $ExpectedModules "$Label unified_result"

    $unifiedKeys = [ordered]@{ deepfake = 'deepfake_detection'; identity = 'identity_search'; source = 'source_search' }
    $resultFiles = [ordered]@{ deepfake = 'detection_result.json'; identity = 'identity_result.json'; source = 'source_result.json' }
    $moduleStatuses = [ordered]@{}
    foreach ($module in $script:KnownModules) {
        $selected = $module -in $ExpectedModules
        $moduleState = Get-RequiredProperty $status.modules $module "$Label task_status.modules"
        $topLevelState = Get-RequiredProperty $status $module "$Label task_status"
        $result = $results[$module]
        $unifiedState = Get-RequiredProperty $unified $unifiedKeys[$module] "$Label unified_result"
        Assert-Acceptance ([string]$result.schema_version -eq '3.0') "$Label $module 结果 schema_version 不是 3.0。"
        Assert-Acceptance ([string]$result.module -eq $module) "$Label $module 结果 module 字段不一致。"
        Assert-Acceptance ([bool]$unifiedState.selected -eq $selected) "$Label $module unified selected 不一致。"
        Assert-Acceptance ([string]$unifiedState.result_file -eq $resultFiles[$module]) "$Label $module result_file 不一致。"
        $statuses = @([string]$moduleState.status, [string]$topLevelState.status, [string]$result.status, [string]$unifiedState.status)
        Assert-Acceptance (@($statuses | Sort-Object -Unique).Count -eq 1) `
            "$Label $module 跨文件状态不一致：$($statuses -join ',')"
        if ($selected) { Assert-StatusAllowed $statuses[0] "$Label 所需模块 $module" }
        else { Assert-Acceptance ($statuses[0] -eq 'skipped') "$Label 未选择模块 $module 不是 skipped。" }
        $moduleStatuses[$module] = $statuses[0]
    }

    $reportTask = Get-RequiredProperty $report 'task' "$Label final_report"
    Assert-Acceptance ([string]$reportTask.task_id -eq $TaskId) "$Label final_report.task.task_id 不一致。"
    Assert-Acceptance ([string]$reportTask.status -eq $taskStatus) "$Label final_report.task.status 不一致。"
    $reportModules = Get-NormalizedModules $reportTask.selected_modules "$Label final_report.task"
    Assert-ExactModules $reportModules $ExpectedModules "$Label final_report.task"
    Assert-Acceptance ($reportTxt.Contains($TaskId, [StringComparison]::Ordinal)) "$Label TXT 报告缺少 task_id。"
    Assert-Acceptance ($reportHtml.Contains($TaskId, [StringComparison]::Ordinal)) "$Label HTML 报告缺少 task_id。"
    Assert-Acceptance ($reportHtml -match '(?i)<html|<!doctype\s+html') "$Label HTML 报告结构无效。"

    $progress = Get-ProgressEvidence $taskDirectory $TaskId
    $expectedOverallProgressStatus = Get-ExpectedProgressStatus $taskStatus
    Assert-Acceptance ([string]$progress.status -eq $expectedOverallProgressStatus) `
        "$Label 最终进度状态与 task_status 映射不一致；task=$taskStatus, progress=$($progress.status), expected=$expectedOverallProgressStatus"
    foreach ($module in $script:KnownModules) {
        $moduleEvents = @($progress.events | Where-Object { [string]$_.module -eq $module })
        Assert-Acceptance ($moduleEvents.Count -gt 0) "$Label 缺少 $module 模块进度事件。"
        if ($module -in $ExpectedModules) {
            Assert-Acceptance (@($moduleEvents | Where-Object { [string]$_.status -ne 'skipped' }).Count -gt 0) `
                "$Label 所需模块 $module 只有 skipped 进度。"
            $finishedEvents = @($moduleEvents | Where-Object { [string]$_.stage -eq "${module}_finished" })
            Assert-Acceptance ($finishedEvents.Count -eq 1) `
                "$Label 所需模块 $module 缺少唯一的 ${module}_finished 成功事件。"
            $expectedModuleProgressStatus = Get-ExpectedProgressStatus ([string]$moduleStatuses[$module])
            Assert-Acceptance ([string]$finishedEvents[0].status -eq $expectedModuleProgressStatus) `
                "$Label 所需模块 $module 终态映射不一致；result=$($moduleStatuses[$module]), progress=$($finishedEvents[0].status), expected=$expectedModuleProgressStatus"
        }
        else {
            Assert-Acceptance (@($moduleEvents | Where-Object { [string]$_.status -ne 'skipped' }).Count -eq 0) `
                "$Label 未选择模块 $module 出现非 skipped 进度，任务口径有歧义。"
        }
    }
    if ('deepfake' -in $ExpectedModules) { Assert-RealDeepfakeEvidence $results.deepfake $TaskId }
    if ('identity' -in $ExpectedModules) { Assert-RealIdentityEvidence $results.identity $TaskId }
    if ('source' -in $ExpectedModules) {
        $sourceProgress = Get-RealSourceProgress $results.source $progress.events $TaskId
    }
    else { $sourceProgress = $null }

    $fileEvidence = [ordered]@{}
    foreach ($name in $paths.Keys) { $fileEvidence[$name] = Get-FileEvidence $paths[$name] $taskDirectory }
    $fileEvidence.progress_events = $progress.files.jsonl
    $fileEvidence.progress_latest = $progress.files.latest
    return [ordered]@{
        status = $taskStatus
        task_id = $TaskId
        task_dir = $taskDirectory
        selected_modules = $statusModules
        module_statuses = $moduleStatuses
        real_mode_validated = $true
        mock_marker_found = $false
        progress = [ordered]@{
            event_count = $progress.event_count
            completed_event_id = $progress.completed_event_id
            completed_at = $progress.completed_at
            status = $progress.status
            elapsed_seconds = $progress.elapsed_seconds
        }
        source_progress = $sourceProgress
        files = $fileEvidence
    }
}

$run = Get-AcceptanceFullPath $RunRoot
Assert-Acceptance (Test-Path -LiteralPath $run -PathType Container) "Phase 1 RunRoot 不存在：$run"
$phase1Path = Join-Path $run 'phase1_state.json'
$phase1 = Read-StrictJson $phase1Path 'Phase 1 状态'
Assert-Acceptance ([string]$phase1.status -eq 'ready_for_business_acceptance') 'phase1 状态不允许采集业务证据。'
Assert-Acceptance (Test-AcceptanceSamePath ([string]$phase1.run_root) $run) `
    'phase1 run_root 与输入 RunRoot 不一致。'
$dataRoot = Get-AcceptanceFullPath ([string](Get-RequiredProperty $phase1 'data_root' 'Phase 1 状态'))
Assert-Acceptance (Test-AcceptanceChildPath $dataRoot $run) "Phase 1 DATA_ROOT 不是本次隔离验收目录的子目录：$dataRoot"
Assert-Acceptance (Test-Path -LiteralPath $dataRoot -PathType Container) "Phase 1 DATA_ROOT 不存在：$dataRoot"
$tasksRoot = Get-AcceptanceFullPath (Join-Path $dataRoot 'tasks')
Assert-Acceptance (Test-AcceptanceChildPath $tasksRoot $dataRoot) 'tasks 目录越界。'
Assert-Acceptance (Test-Path -LiteralPath $tasksRoot -PathType Container) "隔离 DATA_ROOT/tasks 不存在：$tasksRoot"

$definitions = @(
    [pscustomobject]@{ Name = 'deepfake'; Label = 'AIGC'; Expected = @('deepfake'); Requested = $DeepfakeTaskId },
    [pscustomobject]@{ Name = 'identity'; Label = '身份搜索'; Expected = @('identity'); Requested = $IdentityTaskId },
    [pscustomobject]@{ Name = 'source'; Label = '底图搜索'; Expected = @('source'); Requested = $SourceTaskId },
    [pscustomobject]@{ Name = 'comprehensive'; Label = '一键综合'; Expected = @('deepfake', 'identity', 'source'); Requested = $ComprehensiveTaskId }
)
$records = [ordered]@{}
foreach ($definition in $definitions) {
    $taskId = if ([string]::IsNullOrWhiteSpace([string]$definition.Requested)) {
        Resolve-LatestTaskId $tasksRoot $definition.Expected $definition.Label
    }
    else { [string]$definition.Requested }
    $records[$definition.Name] = Test-TaskEvidence $tasksRoot $taskId $definition.Expected $definition.Label
}

$comprehensiveFiles = $records.comprehensive.files
$reports = [ordered]@{
    json = [string]$comprehensiveFiles.report_json.path
    txt = [string]$comprehensiveFiles.report_txt.path
    html = [string]$comprehensiveFiles.report_html.path
}
foreach ($format in $reports.Keys) {
    Assert-Acceptance ([IO.Path]::IsPathFullyQualified($reports[$format])) "综合报告路径不是绝对路径：$format"
    Assert-Acceptance (Test-AcceptanceChildPath $reports[$format] $dataRoot) "综合报告不在 DATA_ROOT：$format"
}

$output = if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    Join-Path $run 'business_evidence.json'
}
else { Get-AcceptanceFullPath $OutputPath }
$output = Get-AcceptanceFullPath $output
Assert-Acceptance (Test-AcceptanceChildPath $output $run) "business_evidence 输出必须位于本次 RunRoot 内：$output"
Assert-Acceptance (-not (Test-Path -LiteralPath $output -PathType Container)) "输出路径是目录：$output"

$payload = [ordered]@{
    schema_version = '1.0'
    status = 'passed'
    collected_at = [DateTimeOffset]::Now.ToString('o')
    run_root = $run
    phase1_state = Get-FileEvidence $phase1Path $run
    data_root = $dataRoot
    tasks_root = $tasksRoot
    deepfake = $records.deepfake
    identity = $records.identity
    source = $records.source
    comprehensive = $records.comprehensive
    reports = $reports
    validation = [ordered]@{
        real_mode = 'validated_from_task_artifacts_no_mock_markers'
        task_selection = 'exact_selected_modules_latest_or_explicit_task_id'
        reports = 'strict_utf8_nonempty_json_valid'
        progress = 'jsonl_valid_and_final_overall_completed'
        source = 'scan_completed_to_library_decoded_and_review_completed_to_shortlist'
    }
}
Write-AcceptanceJson $output $payload
$written = Read-StrictJson $output '已写入 business_evidence'
Assert-Acceptance ([string]$written.status -eq 'passed') '写入后的 business_evidence 状态无效。'
Write-Host "BUSINESS_EVIDENCE_PASSED=$output"
