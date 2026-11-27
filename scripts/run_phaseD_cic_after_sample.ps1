$ErrorActionPreference = "Stop"
$Project = "C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Out = "C:\NOVEL_IDS_runs\phaseD_cic_partial_20261126"
$Sample = Join-Path $Out "cic_sample.csv"
$Manifest = Join-Path $Out "cic_sample_manifest.csv"
$Profile = Join-Path $Out "cic_data_profile.json"
$Predictions = Join-Path $Out "cic_predictions_optionA.csv"
$Summary = Join-Path $Out "cic_inference_summary_optionA.json"
$Log = Join-Path $Out "phaseD_auto_inference.log"

function Write-Log([string]$Message) {
    $Line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $Log -Value $Line
    Write-Output $Line
}

if ((Test-Path $Predictions) -or (Test-Path $Summary)) {
    throw "Refusing to rerun CIC inference: output artifact already exists."
}
Write-Log "Waiting for the one-pass CIC sample preparation process to finish."
while ($true) {
    $Active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -match "prepare_cic_partial_sample\.py"
    }
    if (-not $Active) { break }
    Start-Sleep -Seconds 15
}
foreach ($Required in @($Sample, $Manifest, $Profile)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        Write-Log "STOP: required sample artifact missing: $Required"
        exit 2
    }
}
$SampleRows = (Import-Csv -LiteralPath $Manifest | Measure-Object).Count
if ($SampleRows -lt 1 -or $SampleRows -gt 300000) {
    Write-Log "STOP: sample row count outside approved bound: $SampleRows"
    exit 3
}
Write-Log "Starting one frozen Option A inference run on $SampleRows sampled rows; label column excluded; CPU fallback selected."
$env:PYTHONPATH = Join-Path $Project "src"
Set-Location $Project
& $Python -m novel_ids.inference.predict --input $Sample --output $Predictions --summary $Summary --labels-column " Label" --normalization-input (Join-Path $Out "cic_unlabeled_calibration.csv") --checkpoint-dir "C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43" --batch-size 128 --device cpu
$Exit = $LASTEXITCODE
if ($Exit -ne 0) {
    Write-Log "INFERENCE FAILED exit_code=$Exit. No automatic retry will be attempted."
    exit $Exit
}
$PredRows = (& $Python -c "import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))" $Predictions)
if ([int]$PredRows -ne $SampleRows) {
    Write-Log "STOP: prediction row mismatch sample=$SampleRows predictions=$PredRows"
    exit 4
}
Write-Log "INFERENCE COMPLETE rows=$PredRows predictions=$Predictions summary=$Summary"
