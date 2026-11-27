param(
    [ValidateSet("preflight", "execute")]
    [string]$Mode = "preflight"
)

$ErrorActionPreference = "Stop"

Write-Host "`n=== NOVEL-IDS PHASE C: $Mode ===" -ForegroundColor Cyan

python .\scripts\phaseC_recovery_execute.py "--$Mode"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Evaluation did not complete successfully. Report step skipped." -ForegroundColor Red
    exit $LASTEXITCODE
}

if ($Mode -eq "execute") {
    Write-Host "`n=== MULTICLASS TERMINAL METRICS ===" -ForegroundColor Cyan

    python .\scripts\print_saved_multiclass_report.py `
        "C:\NOVEL_IDS_runs\phaseC_recovery_20261126\results.json"

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "`nRunner completed." -ForegroundColor Green
