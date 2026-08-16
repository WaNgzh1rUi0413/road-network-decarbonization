param(
    [ValidateSet("network", "revised", "both")]
    [string]$Mode = "both",
    [string]$CityFilter = "",
    [string]$CityIds = "",
    [int]$MaxWorkers = 0,
    [string[]]$ExperimentGroups = @(),
    [string[]]$Scenarios = @(),
    [string]$OriginalFlowGroup = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$networkDir = Join-Path $repoRoot "network"
$revisedDir = Join-Path $repoRoot "revised_network"
$logDir = Join-Path $scriptDir "logs\assignment"
$buildStamp = Join-Path $logDir ".optimized-build.stamp"
$buildSignature = "g++ -std=c++14 -O2 -DNDEBUG -fopenmp open-reproduction-v1"
$referenceExperimentGroup = "neighbor_seed_01"
$reusableAttributeScenarios = @("modified_capacity", "modified_speed_limit")

if ($MaxWorkers -le 0) {
    $MaxWorkers = [Math]::Min([Environment]::ProcessorCount, 8)
}
if ($MaxWorkers -lt 1) {
    throw "MaxWorkers must be at least 1."
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-ScenarioSuffix {
    param([string]$Base)

    $dashIdx = $Base.IndexOf("-")
    if ($dashIdx -lt 0) { return "" }
    return $Base.Substring($dashIdx + 1)
}

function Parse-CityIds {
    param([string]$Raw)

    $set = [System.Collections.Generic.HashSet[int]]::new()
    if (-not $Raw) { return ,$set }
    foreach ($part in $Raw.Split(",")) {
        $text = $part.Trim()
        if (-not $text) { continue }
        if ($text.Contains("-")) {
            $bounds = $text.Split("-", 2)
            $start = 0
            $end = 0
            if ([int]::TryParse($bounds[0].Trim(), [ref]$start) -and [int]::TryParse($bounds[1].Trim(), [ref]$end)) {
                if ($start -le $end) {
                    foreach ($id in $start..$end) { [void]$set.Add($id) }
                } else {
                    foreach ($id in $end..$start) { [void]$set.Add($id) }
                }
            }
        } else {
            $id = 0
            if ([int]::TryParse($text, [ref]$id)) { [void]$set.Add($id) }
        }
    }
    return ,$set
}

function Test-ReusableAttributeScenario {
    param(
        [string]$Variant,
        [string]$Scenario
    )

    if (-not $Variant -or $Variant -eq $referenceExperimentGroup) { return $false }
    return $reusableAttributeScenarios -contains $Scenario
}

function Copy-ReusableAttributeFlows {
    param([object[]]$VariantDirs)

    $flowRoot = Join-Path $repoRoot "flow"
    $refRoot = Join-Path $flowRoot $referenceExperimentGroup
    if (-not (Test-Path $refRoot)) {
        Write-Warning "Reference flow directory not found: $refRoot"
        return
    }

    foreach ($variantDir in $VariantDirs) {
        if ($variantDir.Name -eq $referenceExperimentGroup) { continue }
        $dstRoot = Join-Path $flowRoot $variantDir.Name
        New-Item -ItemType Directory -Force -Path $dstRoot | Out-Null

        foreach ($scenario in $reusableAttributeScenarios) {
            foreach ($od in 1..10) {
                $dirName = "flow${od}_${scenario}"
                $src = Join-Path $refRoot $dirName
                $dst = Join-Path $dstRoot $dirName
                if (-not (Test-Path $src)) {
                    Write-Warning "Reference TA result missing: $src"
                    continue
                }
                New-Item -ItemType Directory -Force -Path $dst | Out-Null
                Copy-Item -Path (Join-Path $src "*") -Destination $dst -Recurse -Force
            }
        }
    }
}

function Invoke-Compiler {
    $targets = @(
        [pscustomobject]@{ Source = "data.cpp"; Exe = "data.exe" },
        [pscustomobject]@{ Source = "generate_od.cpp"; Exe = "generate_od.exe" },
        [pscustomobject]@{ Source = "user_equilibrium.cpp"; Exe = "user_equilibrium.exe" }
    )
    $needsBuild = -not (Test-Path $buildStamp)
    if (-not $needsBuild) {
        $needsBuild = (Get-Content -Path $buildStamp -Raw).Trim() -ne $buildSignature
    }
    foreach ($target in $targets) {
        $sourcePath = Join-Path $scriptDir $target.Source
        $exePath = Join-Path $scriptDir $target.Exe
        if (-not (Test-Path $exePath) -or (Get-Item $sourcePath).LastWriteTimeUtc -gt (Get-Item $exePath).LastWriteTimeUtc) {
            $needsBuild = $true
        }
    }
    if (-not $needsBuild) {
        Write-Host "Using cached optimized executables."
        return
    }

    Push-Location $scriptDir
    try {
        Write-Host "Compiling optimized executables..."
        & g++ -o data.exe -std=c++14 -O2 -DNDEBUG data.cpp -fopenmp
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to compile data.cpp"
        }
        & g++ -o generate_od.exe -std=c++14 -O2 -DNDEBUG generate_od.cpp -fopenmp
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to compile generate_od.cpp"
        }
        & g++ -o user_equilibrium.exe -std=c++14 -O2 -DNDEBUG user_equilibrium.cpp -fopenmp
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to compile user_equilibrium.cpp"
        }
    }
    finally {
        Pop-Location
    }
    $buildSignature | Set-Content -Path $buildStamp -Encoding ascii
}

function Get-NetworkTasks {
    param(
        [string]$SourceDir,
        [string]$Variant = ""
    )

    if (-not (Test-Path $SourceDir)) {
        Write-Warning "Missing network directory: $SourceDir"
        return
    }

    $citySet = Parse-CityIds $CityIds
    $pattern = "*_link.csv"
    if ($CityFilter -and $citySet.Count -eq 0) {
        [void]$citySet.Add([int]$CityFilter)
    }

    foreach ($linkFile in Get-ChildItem -Path $SourceDir -Filter $pattern -File | Sort-Object Name) {
        if ($linkFile.BaseName.EndsWith("new_link", [StringComparison]::OrdinalIgnoreCase)) {
            continue
        }

        $base = $linkFile.BaseName.Substring(0, $linkFile.BaseName.Length - "_link".Length)
        $dashIdx = $base.IndexOf("-")
        if ($dashIdx -lt 0) { continue }
        $cityId = 0
        if (-not [int]::TryParse($base.Substring(0, $dashIdx), [ref]$cityId)) { continue }
        if ($citySet.Count -gt 0 -and -not $citySet.Contains($cityId)) { continue }
        $nodeFile = Join-Path $SourceDir "${base}_node.csv"
        $centerBase = $base
        if ($centerBase.EndsWith("-original", [StringComparison]::OrdinalIgnoreCase)) {
            $centerBase = $centerBase.Substring(0, $centerBase.Length - "-original".Length)
        }
        $centerResultDir = Join-Path $repoRoot "k_center\results"
        if ($Variant) {
            $centerResultDir = Join-Path $centerResultDir $Variant
        }
        $centerRevised = Join-Path $centerResultDir "${centerBase}_kcenter_k10_binary_search_revised.csv"
        $centerPlain = Join-Path $repoRoot "k_center\results\original\${centerBase}_kcenter_k10_binary_search.csv"

        if (-not (Test-Path $nodeFile)) {
            Write-Warning "Skip ${base}: missing node file"
            continue
        }
        if (-not ((Test-Path $centerRevised) -or (Test-Path $centerPlain))) {
            Write-Warning "Skip ${base}: missing center file"
            continue
        }
        if ($Scenarios.Count -gt 0) {
            $matched = $false
            foreach ($scen in $Scenarios) {
                if ($base.EndsWith("-$scen", [StringComparison]::OrdinalIgnoreCase)) { $matched = $true; break }
            }
            if (-not $matched) { continue }
        }

        $scenarioSuffix = Get-ScenarioSuffix $base
        if (Test-ReusableAttributeScenario $Variant $scenarioSuffix) {
            continue
        }

        [pscustomobject]@{
            Base = $base
            SourceDir = $SourceDir
            Variant = $Variant
        }
    }
}

function Complete-OneJob {
    param([System.Collections.ArrayList]$RunningJobs)

    $finishedJob = Wait-Job -Job @($RunningJobs | ForEach-Object { $_.Job }) -Any
    $entry = $RunningJobs | Where-Object { $_.Job.Id -eq $finishedJob.Id } | Select-Object -First 1
    $result = Receive-Job -Job $finishedJob

    if ($finishedJob.State -eq "Completed" -and $result.Success) {
        $script:successCount++
        Write-Host ("[{0}/{1}] Finished {2}" -f ($script:successCount + $script:failedCount), $script:taskCount, $entry.Label)
    }
    else {
        $script:failedCount++
        $message = if ($result.Message) { $result.Message } else { $finishedJob.State }
        Write-Warning ("[{0}/{1}] Failed {2}: {3}" -f ($script:successCount + $script:failedCount), $script:taskCount, $entry.Label, $message)
    }

    Remove-Job -Job $finishedJob -Force
    [void]$RunningJobs.Remove($entry)
}

Invoke-Compiler

$dataExe = Join-Path $scriptDir "data.exe"
$godExe = Join-Path $scriptDir "generate_od.exe"
$igpExe = Join-Path $scriptDir "user_equilibrium.exe"
$tasks = @()
$variantDirs = @()
if ($Mode -in @("network", "both")) {
    $tasks += @(Get-NetworkTasks $networkDir $OriginalFlowGroup)
}
if ($Mode -in @("revised", "both")) {
    $variantDirs = @(Get-ChildItem -Path $revisedDir -Directory -Filter "neighbor_seed_*" | Sort-Object Name)
    if ($ExperimentGroups.Count -gt 0) {
        $allowed = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        foreach ($g in $ExperimentGroups) { [void]$allowed.Add($g) }
        $variantDirs = @($variantDirs | Where-Object { $allowed.Contains($_.Name) })
        if ($variantDirs.Count -eq 0) {
            Write-Warning "No matching experiment groups: $($ExperimentGroups -join ', ')"
        }
    }
    if ($variantDirs.Count -gt 0) {
        foreach ($variantDir in $variantDirs) {
            $tasks += @(Get-NetworkTasks $variantDir.FullName $variantDir.Name)
        }
    }
    else {
        $tasks += @(Get-NetworkTasks $revisedDir)
    }
}

if ($tasks.Count -eq 0) {
    Write-Host "No matching network tasks found."
    exit 0
}

$worker = {
    param($RepoRoot, $ScriptDir, $DataExe, $GodExe, $IgpExe, $LogDir, $Task)

    function Invoke-LoggedProcess {
        param(
            [string]$Exe,
            [object[]]$Arguments,
            [string]$LogFile
        )

        $output = & $Exe @Arguments 2>&1
        $exitCode = $LASTEXITCODE
        $output | Out-File -FilePath $LogFile -Append -Encoding utf8
        if ($exitCode -ne 0) {
            throw "$([System.IO.Path]::GetFileName($Exe)) exited with code $exitCode"
        }
    }

    $taskLogDir = $LogDir
    if ($Task.Variant) {
        $taskLogDir = Join-Path $LogDir $Task.Variant
        New-Item -ItemType Directory -Force -Path $taskLogDir | Out-Null
    }
    $logFile = Join-Path $taskLogDir "$($Task.Base).log"
    Set-Location $RepoRoot
    "Started $($Task.Base) at $(Get-Date -Format s)" | Set-Content -Path $logFile -Encoding utf8

    try {
        Set-Location $ScriptDir
        $dataArgs = @($Task.Base)
        if ($Task.Variant) {
            $dataArgs += $Task.Variant
        }
        Invoke-LoggedProcess $DataExe $dataArgs $logFile
        Set-Location $RepoRoot

        if ($Task.Base -like "*modified_*") {
            $intermediateDir = Join-Path $Task.SourceDir "traffic_assignment_input"
        } else {
            $flowGroup = if ($Task.Variant) { $Task.Variant } else { "original" }
            $intermediateDir = Join-Path $RepoRoot "flow\$flowGroup\traffic_assignment_input"
        }
        $newNode = Join-Path $intermediateDir "$($Task.Base)new_node.csv"
        $newLink = Join-Path $intermediateDir "$($Task.Base)new_link.csv"
        if (-not ((Test-Path $newNode) -and (Test-Path $newLink))) {
            throw "missing generated node or link file"
        }

        Invoke-LoggedProcess $GodExe $dataArgs $logFile

        $newOd = Join-Path $intermediateDir "$($Task.Base)new_od.csv"
        if (-not (Test-Path $newOd)) {
            throw "missing generated OD file"
        }

        foreach ($demandMultiplier in 1..10) {
            "igp: $($Task.Base) OD $demandMultiplier" | Add-Content -Path $logFile -Encoding utf8
            $igpArgs = @($Task.Base, $demandMultiplier)
            if ($Task.Variant) {
                $igpArgs += $Task.Variant
            }
            Invoke-LoggedProcess $IgpExe $igpArgs $logFile
        }

        "Finished $($Task.Base) at $(Get-Date -Format s)" | Add-Content -Path $logFile -Encoding utf8
        [pscustomobject]@{ Success = $true; Base = $Task.Base; Message = "" }
    }
    catch {
        "Failed $($Task.Base): $($_.Exception.Message)" | Add-Content -Path $logFile -Encoding utf8
        [pscustomobject]@{ Success = $false; Base = $Task.Base; Message = $_.Exception.Message }
    }
}

$runningJobs = [System.Collections.ArrayList]::new()
$script:taskCount = $tasks.Count
$script:successCount = 0
$script:failedCount = 0

Write-Host "Running $($tasks.Count) network tasks with $MaxWorkers workers."
if ($CityFilter) {
    Write-Host "City filter: $CityFilter"
}
if ($CityIds) {
    Write-Host "City ids: $CityIds"
}
Write-Host "Worker logs: $logDir"

foreach ($task in $tasks) {
    while ($runningJobs.Count -ge $MaxWorkers) {
        Complete-OneJob $runningJobs
    }

    $job = Start-Job -ScriptBlock $worker -ArgumentList $repoRoot, $scriptDir, $dataExe, $godExe, $igpExe, $logDir, $task
    $label = if ($task.Variant) { "$($task.Variant)/$($task.Base)" } else { $task.Base }
    [void]$runningJobs.Add([pscustomobject]@{ Job = $job; Base = $task.Base; Label = $label })
}

while ($runningJobs.Count -gt 0) {
    Complete-OneJob $runningJobs
}

Write-Host "Completed: $successCount succeeded, $failedCount failed."
if ($failedCount -gt 0) {
    exit 1
}
if ($Mode -in @("revised", "both") -and $variantDirs.Count -gt 0) {
    Copy-ReusableAttributeFlows $variantDirs
}
