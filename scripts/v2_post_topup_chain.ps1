# v2_post_topup_chain.ps1 — autonomous chain that runs AFTER the in-flight
# generic-redirect topup finishes.
#
# Sequence:
#   1. Wait for the topup python process to exit
#   2. Revert redirect_fraction in generation.yaml: 0.90 -> 0.30
#   3. Regen A5 persistent at A1-comparable volume (~1h, teacher up)
#   3.3 Re-render system_prompts in all sft_raw dirs (pedagogy bullet added today)
#   3.5 Re-run filter_sft with eval-seed exclusion fix (teacher still up for
#       locale_judge). This produces a clean sft_filtered/ — the topup's own
#       filter_sft ran with the old code and left 20% contamination.
#   4a. Convert A5 -> A6 (string-replace sentinel + re-render system prompt)
#   4b. Convert A1/A2 persistent -> A7 (reads from clean sft_filtered/)
#   5. Delete stage2b checkpoint so it re-runs against fresh data
#   6. Kill teacher llama-server
#   7. Launch the orchestrator (resumes at stage2b, then stage3, then SFT)
#
# Run with:
#   $env:PYTHONUTF8="1"; powershell -File scripts/v2_post_topup_chain.ps1
#
# All output goes to logs/v2_post_topup_chain.log.

$ErrorActionPreference = "Continue"
$root = "c:\project\conversationFactory\qwen-en-tutor"
Set-Location $root

# Activate venv so all `python` calls use the project's packages
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) { & $venvActivate }

$chainLog = Join-Path $root "logs\v2_post_topup_chain.log"
function Log($msg) {
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$stamp  $msg" | Tee-Object -FilePath $chainLog -Append
}

Log "=== v2 post-topup chain starting ==="

# --- Step 1: wait for redirect topup python process to exit ---
# The topup was started by an earlier session; identify the python that
# is running run_generation.py against the teacher server. We poll the
# log file for the completion marker line written by run_generation.py
# at the end.
$topupLog = Join-Path $root "logs\redirect_topup_regen.log"
Log "Step 1: waiting for redirect topup to finish (poll $topupLog)..."
while ($true) {
    # Done when no python process is using the GPU AND the log shows the
    # last [filter_sft] line OR the script's "Wrote" finish line.
    $py = Get-Process python -ErrorAction SilentlyContinue
    $logExists = Test-Path $topupLog
    if (-not $py) {
        Log "  no python process running; checking log..."
        if ($logExists) {
            $tail = Get-Content $topupLog -Tail 3 -ErrorAction SilentlyContinue
            Log ("  log tail: " + ($tail -join " || "))
        }
        break
    }
    Start-Sleep -Seconds 30
}
Log "Step 1 done."

# --- Step 2: revert redirect_fraction 0.90 -> 0.30 ---
# Use Python to avoid PowerShell Set-Content corrupting Unicode chars in the yaml
Log "Step 2: revert generation.yaml redirect_fraction to 0.30"
$genYaml = Join-Path $root "config\generation.yaml"
python -c "
import re, sys
path = sys.argv[1]
txt = open(path, encoding='utf-8').read()
txt2 = re.sub(r'redirect_fraction:\s+0\.90[^\n]*', 'redirect_fraction: 0.30', txt)
if txt2 != txt:
    open(path, 'w', encoding='utf-8').write(txt2)
    print('redirect_fraction updated to 0.30')
else:
    print('redirect_fraction already correct')
import re as re2
m = re2.search(r'^\s*redirect_fraction:.*', txt2, re2.MULTILINE)
print('current:', m.group(0).strip() if m else 'not found')
" $genYaml
Log "Step 2 done."

# --- Step 3: A5 persistent regen at A1-comparable volume ---
# Teacher server must be UP. The redirect topup left it loaded; we reuse it.
Log "Step 3: A5 persistent regen (--fraction-multiplier 2.5 --dialogues-per-seed 1, target ~650 filtered)"
$env:PYTHONUTF8 = "1"
$a5Log = Join-Path $root "logs\v2_a5_persistent_regen.log"
python scripts\regen_persistent_a5.py --fraction-multiplier 2.5 --dialogues-per-seed 1 2>&1 |
    Tee-Object -FilePath $a5Log
$rc = $LASTEXITCODE
Log "  A5 regen rc=$rc"
if ($rc -ne 0) {
    Log "  A5 regen failed. Stopping chain."
    exit 1
}
Log "Step 3 done."

# --- Step 3.3: re-render system_prompts in all sft_raw dirs ---
# The [guidelines] pedagogy bullet was added to the deployment system prompt
# template after existing sft_raw records were generated. Re-render now so
# filter_sft bakes consistent system_prompts into sft_filtered/ for all
# conditions (A3/A5). A6/A7 re-render via their own convert scripts.
Log "Step 3.3: re-render system_prompts in sft_raw/ and sft_raw_a5_persistent/"
$rrLog = Join-Path $root "logs\v2_rerender_system_prompts.log"
python scripts\rerender_system_prompts.py 2>&1 | Tee-Object -FilePath $rrLog
$rc = $LASTEXITCODE
Log "  rerender rc=$rc"
if ($rc -ne 0) {
    Log "  rerender_system_prompts failed. Stopping chain."
    exit 1
}
Log "Step 3.3 done."

# --- Step 3.5: re-run filter_sft with eval-seed exclusion ---
# The topup's filter_sft ran with the old code (no eval-seed exclusion), leaving
# ~20% contamination in sft_filtered/. Run filter_sft again now — teacher is
# still up for the locale_judge filter — to produce a clean sft_filtered/.
# convert_a1_to_a7.py (step 4b) reads from sft_filtered/persistent_*_passed.jsonl,
# so it must run AFTER this step.
Log "Step 3.5: re-run filter_sft (eval-seed exclusion, teacher up for locale_judge)"
$fsLog = Join-Path $root "logs\v2_filter_sft_clean.log"
python scripts\run_generation.py --stages filter_sft 2>&1 | Tee-Object -FilePath $fsLog
$rc = $LASTEXITCODE
Log "  filter_sft rc=$rc"
if ($rc -ne 0) {
    Log "  filter_sft failed. Stopping chain."
    exit 1
}
Log "Step 3.5 done."

# --- Step 4a: convert A5 -> A6 (fixed turn-7 persistent, generic sentinel) ---
Log "Step 4a: convert A5 persistent -> A6 persistent (generic [SESSION_END])"
$conv6Log = Join-Path $root "logs\v2_a6_conversion.log"
python scripts\convert_a5_to_a6.py 2>&1 | Tee-Object -FilePath $conv6Log
$rc = $LASTEXITCODE
Log "  A5->A6 conversion rc=$rc"
if ($rc -ne 0) {
    Log "  A5->A6 conversion failed. Stopping chain."
    exit 1
}
Log "Step 4a done."

# --- Step 4b: convert A1/A2 persistent -> A7 (4-variant persistent, generic sentinel) ---
Log "Step 4b: convert A1/A2 persistent -> A7 persistent (generic [SESSION_END])"
$conv7Log = Join-Path $root "logs\v2_a7_conversion.log"
python scripts\convert_a1_to_a7.py 2>&1 | Tee-Object -FilePath $conv7Log
$rc = $LASTEXITCODE
Log "  A1->A7 conversion rc=$rc"
if ($rc -ne 0) {
    Log "  A1->A7 conversion failed. Stopping chain."
    exit 1
}
Log "Step 4b done."

# --- Step 4c: build per-condition SFT dirs so user can review before orchestrator ---
Log "Step 4c: setup_paper_ablation_data.py --conditions a3,a5,a6,a7"
$setupLog = Join-Path $root "logs\v2_setup_ablation_data.log"
python scripts\setup_paper_ablation_data.py --conditions a3,a5,a6,a7 2>&1 | Tee-Object -FilePath $setupLog
$rc = $LASTEXITCODE
Log "  setup_ablation_data rc=$rc"
if ($rc -ne 0) {
    Log "  setup_ablation_data failed. Stopping chain."
    exit 1
}
Log "Step 4c done."

# --- Step 5: delete stage2b checkpoint so orchestrator re-runs setup (idempotent) ---
$ckpt = Join-Path $root "outputs\paper_v2\.checkpoints\stage2b_setup_ablation_data.done"
if (Test-Path $ckpt) {
    Remove-Item $ckpt -Force
    Log "Step 5: deleted stale stage2b checkpoint"
} else {
    Log "Step 5: no stage2b checkpoint to delete"
}

# --- Step 6: kill teacher llama-server (free VRAM for SFT) ---
Log "Step 6: kill teacher llama-server"
taskkill /F /IM llama-server.exe 2>&1 | Out-Null
Start-Sleep -Seconds 4
Log "Step 6 done."

# --- Step 7: launch the orchestrator ---
Log "Step 7: launch orchestrate_v2_final.py"
$orchOut = Join-Path $root "logs\orchestrate_v2_final_stdout.log"
$orchErr = Join-Path $root "logs\orchestrate_v2_final_stderr.log"
$env:PYTHONUTF8 = "1"
$p = Start-Process -FilePath "python" `
    -ArgumentList "scripts/orchestrate_v2_final.py" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $orchOut `
    -RedirectStandardError $orchErr `
    -PassThru
Log "Step 7: orchestrator launched as PID $($p.Id)"

Log "=== v2 post-topup chain complete (orchestrator now running unattended) ==="
