<#
    llama_continue_after_a1.ps1

    Runs the REMAINDER of the cross-family chain after the A1 SFT (launched
    separately, PID logged in outputs/paper/llama_a1_sft.err.log) has finished
    and written its adapter to outputs/paper/llama_a1/sft.

    Stages: A3 SFT -> generate A1+A3 on eval sets -> mechanical scoring.
    (Judged withholding is the separate manual judge-server step; see
    paper/LLAMA_CROSSFAMILY_HANDOFF.md.)

    PYTHONUNBUFFERED=1 so loss/progress stream live to the logs this time.

    Guard: refuses to start A3 until A1's adapter exists, so it is safe to run
    eagerly — it will error out cleanly if A1 isn't done.
#>
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"

$a1adapter = "outputs/paper/llama_a1/sft"
if (-not (Test-Path (Join-Path $a1adapter "adapter_config.json"))) {
    throw "A1 adapter not found at $a1adapter (adapter_config.json missing). Wait for the A1 SFT run to finish before running this."
}
Write-Host "A1 adapter present. Continuing." -ForegroundColor Green

function Step($m) { Write-Host "`n==== $m ====" -ForegroundColor Cyan }

Step "A3 SFT (generic-SFT baseline)"
python scripts/run_training.py `
    --training-config config/paper/training_llama_a3_no_specialized.yaml `
    --stages train_sft
if ($LASTEXITCODE -ne 0) { throw "Llama A3 SFT failed ($LASTEXITCODE)" }

Step "Generate: Llama A1 on all eval sets"
python scripts/run_paper_eval.py --baseline paper_llama_a1_sft --test-set all
if ($LASTEXITCODE -ne 0) { throw "A1 eval-gen failed ($LASTEXITCODE)" }

Step "Generate: Llama A3 on all eval sets"
python scripts/run_paper_eval.py --baseline paper_llama_a3_sft --test-set all
if ($LASTEXITCODE -ne 0) { throw "A3 eval-gen failed ($LASTEXITCODE)" }

Step "Mechanical scoring (persistence recall + locale leakage)"
python scripts/run_paper_score.py --baseline paper_llama_a1_sft --metrics mechanical
python scripts/run_paper_score.py --baseline paper_llama_a3_sft --metrics mechanical

Write-Host "`nLOCAL CHAIN DONE. Mechanical results:" -ForegroundColor Green
Write-Host "  outputs/paper/score/mechanical/paper_llama_a1_sft/"
Write-Host "  outputs/paper/score/mechanical/paper_llama_a3_sft/"
Write-Host "Then run the judged withholding step (needs a judge server) per HANDOFF.md." -ForegroundColor Yellow
