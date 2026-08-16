param(
    [switch]$SmokeTest,
    [int]$MaxModelsForShap = 0
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

& $python scripts\verify_inputs.py
$trainArgs = @("-m", "src.modeling.train_models")
if ($SmokeTest) { $trainArgs += "--smoke-test" }
& $python @trainArgs
if ($LASTEXITCODE -ne 0) { throw "Model training failed." }

$shapArgs = @("-m", "src.analysis.compute_shap")
if ($MaxModelsForShap -gt 0) { $shapArgs += @("--max-models", $MaxModelsForShap) }
elseif ($SmokeTest) { $shapArgs += @("--od-levels", "OD", "--max-models", 1) }
& $python @shapArgs
if ($LASTEXITCODE -ne 0) { throw "SHAP analysis failed." }

& $python -m src.analysis.cluster_cities
if ($LASTEXITCODE -ne 0) { throw "Clustering failed." }
Write-Host "Core analysis completed. Results are under outputs/."

