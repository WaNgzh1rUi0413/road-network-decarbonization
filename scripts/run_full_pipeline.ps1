param(
    [string]$CityIds = "",
    [int]$MaxWorkers = 8,
    [switch]$RunModeling
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

& $python scripts\verify_inputs.py
& $python -m src.preprocessing.topology_metrics

$centerArgs = @("-m", "src.od_selection.run_k_center", "--skip-existing")
if ($CityIds) { $centerArgs += @("--city-ids", $CityIds) }
& $python @centerArgs

$assignment = @{
    Mode = "network"
    OriginalFlowGroup = "original"
    MaxWorkers = $MaxWorkers
}
if ($CityIds) { $assignment.CityIds = $CityIds }
& "cpp\traffic_assignment\run_assignment.ps1" @assignment

& $python -m src.emissions.compute_emissions --flow-root flow\original --output-root outputs\emissions\original
if (-not $CityIds) {
    & $python -m src.preprocessing.assemble_dataset
}
if ($RunModeling) {
    & "scripts\run_analysis.ps1"
}
Write-Host "Full original-network pipeline completed."

