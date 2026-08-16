param(
    [int]$MaxWorkers = 8,
    [double]$ChangeRatio = 0.10,
    [string]$CityIds = "",
    [string[]]$ExperimentGroups = @()
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

$revisionArgs = @("-m", "src.interventions.run_interventions", "--strategy", "proportional", "--change-ratio", $ChangeRatio)
if ($CityIds) {
    $revisionArgs += @("--city-ids", $CityIds)
} else {
    $revisionArgs += "--all-cities"
}
if ($ExperimentGroups.Count -eq 1) {
    $revisionArgs += @("--experiment-group", $ExperimentGroups[0])
}
& $python @revisionArgs
& $python -m src.interventions.validate_networks

$groups = @(Get-ChildItem revised_network -Directory -Filter "neighbor_seed_*" | Sort-Object Name)
if ($ExperimentGroups.Count -gt 0) {
    $allowed = [System.Collections.Generic.HashSet[string]]::new([string[]]$ExperimentGroups)
    $groups = @($groups | Where-Object { $allowed.Contains($_.Name) })
}
foreach ($group in $groups) {
    $centerArgs = @(
        "-m", "src.od_selection.run_k_center",
        "--network-dir", $group.FullName,
        "--output-dir", (Join-Path "k_center\results" $group.Name),
        "--revised", "--skip-existing"
    )
    if ($CityIds) { $centerArgs += @("--city-ids", $CityIds) }
    & $python @centerArgs
}

$assignmentArgs = @{
    Mode = "revised"
    MaxWorkers = $MaxWorkers
}
if ($CityIds) { $assignmentArgs.CityIds = $CityIds }
if ($ExperimentGroups.Count -gt 0) { $assignmentArgs.ExperimentGroups = $ExperimentGroups }
& "cpp\traffic_assignment\run_assignment.ps1" @assignmentArgs
& $python -m src.emissions.compute_emissions --flow-root flow --output-root outputs\emissions
& $python -m src.emissions.aggregate_interventions --input-dir outputs\emissions
& $python -m src.interventions.summarize_modifications
Write-Host "Counterfactual intervention pipeline completed."
