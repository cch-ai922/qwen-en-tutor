<#
    run_llama_both_chain.ps1 — train BOTH Llama variants + generate + score.

    Stages (each training stage followed by a GPU cooldown):
      1. Base A1  -> 2 epochs (RESUMES from the 1-epoch checkpoint, step 397)
      2. Base A3  -> 2 epochs (RESUMES from step 225)
      3. Instruct A1 -> 1 epoch (fresh; Instruct already knows <|eot_id|>)
      4. Instruct A3 -> 1 epoch (fresh)
      5. Generate all 4 on eval sets
      6. Mechanical scoring for all 4

    GPU COOLDOWN between heavy stages: waits until GPU temperature drops below a
    threshold (or a max wait elapses), protecting the 3060 during back-to-back
    training. Tunable via $CoolTempC / $CoolMaxSec / $CoolMinSec below.

    Resumable: Base stages auto-resume from their checkpoints; generation/scoring
    skip records already on disk. Safe to re-run verbatim after any interruption.
#>
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"

# --- cooldown knobs ---------------------------------------------------------
$CoolTempC  = 55     # wait until GPU temp <= this (deg C)
$CoolMaxSec = 300    # ...but never wait longer than this
$CoolMinSec = 30     # always pause at least this long (let VRAM/driver settle)

function Get-GpuTemp {
    try {
        $t = (nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) 2>$null
        return [int]($t | Select-Object -First 1)
    } catch { return -1 }
}

function Cooldown($label) {
    Write-Host "`n---- GPU cooldown after $label [$(Get-Date -Format HH:mm:ss)] ----" -ForegroundColor DarkCyan
    Start-Sleep -Seconds $CoolMinSec
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $CoolMaxSec) {
        $temp = Get-GpuTemp
        $used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) 2>$null | Select-Object -First 1
        Write-Host ("   temp={0}C  vram={1}MiB  (target <= {2}C)" -f $temp, $used, $CoolTempC)
        if ($temp -ge 0 -and $temp -le $CoolTempC) { Write-Host "   cooled." -ForegroundColor DarkCyan; break }
        Start-Sleep -Seconds 20
    }
}

function Step($m) { Write-Host "`n==== $m [$(Get-Date -Format HH:mm:ss)] ====" -ForegroundColor Cyan }

# --- 1. Base A1 (resume to 2 epochs) ---------------------------------------
Step "Base A1 -> 2 epochs (resume)"
python scripts/run_training.py --training-config config/paper/training_llama_a1_full.yaml --stages train_sft
if ($LASTEXITCODE -ne 0) { throw "Base A1 failed ($LASTEXITCODE)" }
Cooldown "Base A1"

# --- 2. Base A3 (resume to 2 epochs) ---------------------------------------
Step "Base A3 -> 2 epochs (resume)"
python scripts/run_training.py --training-config config/paper/training_llama_a3_no_specialized.yaml --stages train_sft
if ($LASTEXITCODE -ne 0) { throw "Base A3 failed ($LASTEXITCODE)" }
Cooldown "Base A3"

# --- 3. Instruct A1 (1 epoch, fresh) ---------------------------------------
Step "Instruct A1 -> 1 epoch"
python scripts/run_training.py --training-config config/paper/training_llamaINST_a1_full.yaml --stages train_sft
if ($LASTEXITCODE -ne 0) { throw "Instruct A1 failed ($LASTEXITCODE)" }
Cooldown "Instruct A1"

# --- 4. Instruct A3 (1 epoch, fresh) ---------------------------------------
Step "Instruct A3 -> 1 epoch"
python scripts/run_training.py --training-config config/paper/training_llamaINST_a3_no_specialized.yaml --stages train_sft
if ($LASTEXITCODE -ne 0) { throw "Instruct A3 failed ($LASTEXITCODE)" }
Cooldown "Instruct A3"

# --- 5. Generate all 4 on eval sets ----------------------------------------
foreach ($b in @("paper_llama_a1_sft","paper_llama_a3_sft","paper_llamaINST_a1_sft","paper_llamaINST_a3_sft")) {
    Step "Generate $b"
    python scripts/run_paper_eval.py --baseline $b --test-set all
    if ($LASTEXITCODE -ne 0) { throw "$b gen failed ($LASTEXITCODE)" }
    Cooldown "gen $b"
}

# --- 6. Mechanical scoring --------------------------------------------------
foreach ($b in @("paper_llama_a1_sft","paper_llama_a3_sft","paper_llamaINST_a1_sft","paper_llamaINST_a3_sft")) {
    Step "Score (mechanical) $b"
    python scripts/run_paper_score.py --baseline $b --metrics mechanical
}

Step "ALL DONE"
Write-Host "Mechanical results under outputs/paper/score/mechanical/paper_llama*_sft/" -ForegroundColor Green
Write-Host "Compare Base vs Instruct: Instruct should terminate cleanly; Base-2ep is the test of whether the extra epoch fixed <|eot_id|>." -ForegroundColor Yellow
Write-Host "Judged withholding = separate judge-server step (HANDOFF.md)." -ForegroundColor Yellow
