# Watchdog: fires when the Llama-3.1 judge chain writes its completion marker.
# Then: stop Llama-3.1 server, launch Gemma-2 server, run identical chain,
# then run the aggregation step. Designed to be launched from the assistant
# as a fully-detached background process so the unattended pipeline runs
# end-to-end without a human handoff.

$ErrorActionPreference = "Continue"
$ProjectRoot = "c:\project\conversationFactory\qwen-en-tutor"
$LlamaLog    = "$ProjectRoot\logs\llama31_chain.log"
$GemmaLog    = "$ProjectRoot\logs\gemma2_chain.log"
$ServerLog   = "$ProjectRoot\logs\gemma2_server.log"
$WatchdogLog = "$ProjectRoot\logs\watchdog_gemma2.log"
$LlamaCpp    = "$ProjectRoot\vendor\llama_cpp"
$GgufRel     = "..\models\GGUF\gemma-2-9b-it-Q5_K_M.gguf"

function Log($msg) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp $msg" | Tee-Object -FilePath $WatchdogLog -Append | Out-Host
}

Log "watchdog armed; waiting for 'LLAMA-3.1 CHAIN COMPLETE' marker in $LlamaLog"

# ---- 1. Block until marker appears -----------------------------------------
while ($true) {
    if (Test-Path $LlamaLog) {
        $found = Select-String -Path $LlamaLog -Pattern "LLAMA-3.1 CHAIN COMPLETE" -SimpleMatch -Quiet
        if ($found) { break }
    }
    Start-Sleep -Seconds 30
}
Log "llama-3.1 chain completion detected"

# ---- 2. Stop Llama-3.1 server ----------------------------------------------
Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | ForEach-Object {
    Log "stopping llama-server pid=$($_.Id)"
    Stop-Process -Id $_.Id -Force
}
Start-Sleep -Seconds 5
if (Get-Process -Name "llama-server" -ErrorAction SilentlyContinue) {
    Log "WARNING: llama-server still running after Stop-Process"
} else {
    Log "llama-server stopped cleanly"
}

# ---- 3. Launch Gemma-2 server ----------------------------------------------
Log "launching gemma-2 server"
$serverArgs = @(
    "-m", $GgufRel,
    "-ngl", "80",
    "-c", "16384",
    "--host", "0.0.0.0",
    "--port", "8080",
    "--log-prefix"
)
$serverProc = Start-Process -FilePath "$LlamaCpp\llama-server.exe" `
    -ArgumentList $serverArgs `
    -WorkingDirectory $LlamaCpp `
    -RedirectStandardOutput $ServerLog `
    -RedirectStandardError "$ServerLog.err" `
    -PassThru -NoNewWindow
Log "gemma-2 server pid=$($serverProc.Id)"

# ---- 4. Wait for /v1/models readiness --------------------------------------
$deadline = (Get-Date).AddMinutes(5)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8080/v1/models" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        $body = $r.Content | ConvertFrom-Json
        if ($body.data[0].id -match "gemma-2") {
            Log "gemma-2 server ready: $($body.data[0].id)"
            $ready = $true
            break
        }
    } catch { Start-Sleep -Seconds 3 }
}
if (-not $ready) {
    Log "ERROR: gemma-2 server not ready within 5min; aborting chain"
    exit 1
}

# ---- 5. Run Gemma-2 chain (identical to Llama-3.1) -------------------------
$env:PYTHONUTF8 = "1"
Set-Location $ProjectRoot

$chainCmd = @"
=== [1/19] B4 (teacher) generic ===
python scripts/run_paper_score.py --baseline qwen3_5_9b_teacher --metrics judged --judge gemma2_9b_judge --rubric generic
=== [2/19] B4 (teacher) tutor ===
python scripts/run_paper_score.py --baseline qwen3_5_9b_teacher --metrics judged --judge gemma2_9b_judge --rubric tutor --metric-filter cefr_adherence,naturalness
"@
Log "starting gemma-2 chain (see $GemmaLog)"

$started = Get-Date
"=== gemma-2 chain start $($started.ToString('yyyy-MM-dd HH:mm:ss')) ===" | Out-File -FilePath $GemmaLog -Encoding utf8

& python scripts/run_paper_score.py --baseline qwen3_5_9b_teacher --metrics judged --judge gemma2_9b_judge --rubric generic *>&1 | Tee-Object -FilePath $GemmaLog -Append | Out-Null
& python scripts/run_paper_score.py --baseline qwen3_5_9b_teacher --metrics judged --judge gemma2_9b_judge --rubric tutor --metric-filter cefr_adherence,naturalness *>&1 | Tee-Object -FilePath $GemmaLog -Append | Out-Null
foreach ($b in @("paper_a1","paper_a2","paper_a3_sft","paper_a4_sft","paper_a5_sft","qwen3_5_0_8b_base","qwen3_5_0_8b_instruct","qwen3_5_4b_instruct")) {
    Log "gemma-2: $b tutor"
    & python scripts/run_paper_score.py --baseline $b --metrics judged --judge gemma2_9b_judge --rubric tutor --metric-filter cefr_adherence,naturalness *>&1 | Tee-Object -FilePath $GemmaLog -Append | Out-Null
}
"=== GEMMA-2 CHAIN COMPLETE ===" | Tee-Object -FilePath $GemmaLog -Append | Out-Host

$elapsed = (Get-Date) - $started
Log "gemma-2 chain done in $($elapsed.TotalMinutes.ToString('F1')) min"

# ---- 6. Aggregation step ---------------------------------------------------
Log "running aggregation across all 9 baselines x 3 judges"
$aggLog = "$ProjectRoot\logs\aggregate.log"
$baselines = "paper_a1,paper_a2,paper_a3_sft,paper_a4_sft,paper_a5_sft,qwen3_5_0_8b_base,qwen3_5_0_8b_instruct,qwen3_5_4b_instruct,qwen3_5_9b_teacher"
& python scripts/run_paper_score.py --aggregate --baselines $baselines --judges prometheus_7b_judge,llama31_8b_judge,gemma2_9b_judge *>&1 | Tee-Object -FilePath $aggLog -Append | Out-Null
Log "aggregation complete"

Log "=== WATCHDOG COMPLETE ==="
