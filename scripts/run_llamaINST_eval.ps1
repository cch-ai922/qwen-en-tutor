<#
    run_llamaINST_eval.ps1 — generate + mechanical-score the Instruct A1/A3 students.

    Base is dropped (can't emit <|eot_id|>); Instruct generates cleanly. No training.
    Cooldown between generation stages. Resumable (skips done records).
#>
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"

$CoolTempC = 55; $CoolMaxSec = 240; $CoolMinSec = 25
function Get-GpuTemp { try { [int]((nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) 2>$null | Select-Object -First 1) } catch { -1 } }
function Cooldown($label) {
    Write-Host "`n---- cooldown after $label [$(Get-Date -Format HH:mm:ss)] ----" -ForegroundColor DarkCyan
    Start-Sleep -Seconds $CoolMinSec
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $CoolMaxSec) {
        $temp = Get-GpuTemp
        Write-Host ("   temp={0}C (target <= {1})" -f $temp, $CoolTempC)
        if ($temp -ge 0 -and $temp -le $CoolTempC) { break }
        Start-Sleep -Seconds 20
    }
}
function Step($m) { Write-Host "`n==== $m [$(Get-Date -Format HH:mm:ss)] ====" -ForegroundColor Cyan }

foreach ($b in @("paper_llamaINST_a1_sft","paper_llamaINST_a3_sft")) {
    Step "Generate $b on all eval sets"
    python scripts/run_paper_eval.py --baseline $b --test-set all
    if ($LASTEXITCODE -ne 0) { throw "$b gen failed ($LASTEXITCODE)" }
    Cooldown "gen $b"
}

foreach ($b in @("paper_llamaINST_a1_sft","paper_llamaINST_a3_sft")) {
    Step "Mechanical score $b"
    python scripts/run_paper_score.py --baseline $b --metrics mechanical
    if ($LASTEXITCODE -ne 0) { throw "$b score failed ($LASTEXITCODE)" }
}

Step "DONE"
Write-Host "Results: outputs/paper/score/mechanical/paper_llamaINST_a{1,3}_sft/" -ForegroundColor Green
