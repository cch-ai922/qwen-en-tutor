<#
    run_llama_1ep_chain.ps1 — deadline chain: A1(1ep) -> A3(1ep) -> gen -> score.
    PYTHONUNBUFFERED so loss/progress stream live. Each stage guards the next.
#>
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
function Step($m) { Write-Host "`n==== $m [$(Get-Date -Format HH:mm:ss)] ====" }

Step "A1 SFT (1 epoch, full mix)"
python scripts/run_training.py --training-config config/paper/training_llama_a1_full.yaml --stages train_sft
if ($LASTEXITCODE -ne 0) { throw "A1 SFT failed ($LASTEXITCODE)" }

Step "A3 SFT (1 epoch, generic-SFT baseline)"
python scripts/run_training.py --training-config config/paper/training_llama_a3_no_specialized.yaml --stages train_sft
if ($LASTEXITCODE -ne 0) { throw "A3 SFT failed ($LASTEXITCODE)" }

Step "Generate A1 on all eval sets"
python scripts/run_paper_eval.py --baseline paper_llama_a1_sft --test-set all
if ($LASTEXITCODE -ne 0) { throw "A1 gen failed ($LASTEXITCODE)" }

Step "Generate A3 on all eval sets"
python scripts/run_paper_eval.py --baseline paper_llama_a3_sft --test-set all
if ($LASTEXITCODE -ne 0) { throw "A3 gen failed ($LASTEXITCODE)" }

Step "Mechanical scoring (persistence + locale)"
python scripts/run_paper_score.py --baseline paper_llama_a1_sft --metrics mechanical
python scripts/run_paper_score.py --baseline paper_llama_a3_sft --metrics mechanical

Step "CHAIN DONE"
Write-Host "Mechanical results in outputs/paper/score/mechanical/paper_llama_a{1,3}_sft/"
Write-Host "Judged withholding = separate judge-server step (HANDOFF.md)."
