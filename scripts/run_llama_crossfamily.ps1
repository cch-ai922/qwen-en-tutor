<#
    run_llama_crossfamily.ps1  —  turn-key cross-family (Llama-3.2-1B) train + eval

    Purpose: the §6.3 cross-family replication. Trains two SFT-only Llama
    students (A1 full-mix, A3 generic-SFT) on the existing teacher-distilled
    corpus, evaluates them on the frozen eval sets, and scores the two
    load-bearing metrics (persistence recall + pedagogy withholding) plus
    locale leakage.

    PRECONDITIONS (must all hold before you run this):
      1. GPU is free (no teacher server, no other training resident).
      2. Data subsets already built: data/sft_filtered/ and data/sft_filtered_a3/
         (they exist from the Qwen runs — no rebuild needed).
      3. Frozen eval sets present in eval_sets/ (they are).
      4. venv activated.

    This script runs LOCAL stages only (training + hf-baseline generation +
    mechanical scoring). The JUDGED withholding metric needs a judge
    llama-server; that is a SEPARATE, teacher-style swap step printed at the end
    (it competes with the GPU, so it is not auto-run here).

    Usage:
        cd qwen-en-tutor
        .\.venv\Scripts\Activate.ps1
        pwsh -File scripts\run_llama_crossfamily.ps1          # or: powershell -File ...

    Resumable: every stage skips work already on disk, so a killed run can be
    re-invoked verbatim.
#>

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"   # REQUIRED on Windows (TRL chat_template load dies on cp1252)

function Step($msg) { Write-Host "`n==== $msg ====" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------
# Stage 1 — Train both Llama students (SFT-only). ~overnight each on a 3060.
# ---------------------------------------------------------------------------
Step "1/3  SFT: Llama A1 (full mix)"
python scripts/run_training.py `
    --training-config config/paper/training_llama_a1_full.yaml `
    --stages train_sft
if ($LASTEXITCODE -ne 0) { throw "Llama A1 SFT failed (exit $LASTEXITCODE)" }

Step "1/3  SFT: Llama A3 (generic-SFT baseline)"
python scripts/run_training.py `
    --training-config config/paper/training_llama_a3_no_specialized.yaml `
    --stages train_sft
if ($LASTEXITCODE -ne 0) { throw "Llama A3 SFT failed (exit $LASTEXITCODE)" }

# ---------------------------------------------------------------------------
# Stage 2 — Generate on the frozen eval sets (local hf, no teacher needed).
# ---------------------------------------------------------------------------
Step "2/3  Generate: Llama A1 on all eval sets"
python scripts/run_paper_eval.py --baseline paper_llama_a1_sft --test-set all
if ($LASTEXITCODE -ne 0) { throw "Llama A1 eval-gen failed (exit $LASTEXITCODE)" }

Step "2/3  Generate: Llama A3 on all eval sets"
python scripts/run_paper_eval.py --baseline paper_llama_a3_sft --test-set all
if ($LASTEXITCODE -ne 0) { throw "Llama A3 eval-gen failed (exit $LASTEXITCODE)" }

# ---------------------------------------------------------------------------
# Stage 3 — Mechanical scoring (persistence recall + locale leakage). No judge.
# ---------------------------------------------------------------------------
Step "3/3  Score mechanical: Llama A1 + A3"
python scripts/run_paper_score.py --baseline paper_llama_a1_sft --metrics mechanical
python scripts/run_paper_score.py --baseline paper_llama_a3_sft --metrics mechanical

Write-Host "`nLOCAL STAGES DONE." -ForegroundColor Green
Write-Host @"

Persistence recall + locale leakage are now in:
    outputs/paper/score/mechanical/paper_llama_a1_sft/
    outputs/paper/score/mechanical/paper_llama_a3_sft/

STILL TO DO — the judged WITHHOLDING metric (needs a judge llama-server up).
Start ONE judge, then run (see paper/LLAMA_CROSSFAMILY_HANDOFF.md for detail):

  # 1. start a judge (Llama-3.1 or Gemma-2 GGUF) on :8080, then:
  python scripts/score_withholding_rate.py --judge llama31_8b_judge `
      --eval-dir outputs/paper/eval `
      --baselines paper_llama_a1_sft,paper_llama_a3_sft `
      --out outputs/paper/score/llama_withholding_llama31.json

  # 2. swap to the Gemma judge, repeat with --judge gemma2_9b_judge and a
  #     distinct --out, then average the two per the paper's 2-judge protocol.

Compare the resulting numbers against the Qwen A1/A3 rows to report whether the
train-vs-prompt boundary DIRECTION holds in a second trained family (§6.3).
"@ -ForegroundColor Yellow
