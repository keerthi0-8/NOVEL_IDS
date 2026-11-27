$ErrorActionPreference = "Stop"
$Project = "C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Out = "C:\NOVEL_IDS_runs\phaseD_cic_partial_20261126"
$Predictions = Join-Path $Out "cic_predictions_optionA.csv"
$Summary = Join-Path $Out "cic_inference_summary_optionA.json"
$InferenceLog = Join-Path $Out "phaseD_auto_inference.log"
$Metrics = Join-Path $Out "cic_evaluation_metrics.json"
$Log = Join-Path $Out "phaseD_metrics_after_inference.log"

function Write-Log([string]$Message) {
    $Line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $Log -Value $Line
    Write-Output $Line
}
if (Test-Path $Metrics) { throw "Refusing to rerun one-shot CIC metrics." }
if (-not (Test-Path -LiteralPath $Predictions) -or -not (Test-Path -LiteralPath $Summary)) {
    throw "Frozen CIC inference outputs are missing; metrics must only run after successful inference."
}
Write-Log "Running evaluation-only metrics from saved predictions and sample manifest."
& $Python (Join-Path $Project "scripts\evaluate_cic_partial_predictions.py")
$Exit = $LASTEXITCODE
if ($Exit -ne 0) {
    Write-Log "METRICS FAILED exit_code=$Exit; no inference rerun will be attempted."
    exit $Exit
}
Write-Log "METRICS COMPLETE."
