# start_llama_cpp.ps1 — start the llama.cpp HTTP server with gpt-oss-20b.
#
# Prereq: a pre-built llama.cpp `llama-server.exe` on this machine. The
# easiest source for Windows is the prebuilt release zip from
#     https://github.com/ggerganov/llama.cpp/releases
# Unzip it and either add the folder to PATH or point $env:LLAMA_CPP_BIN
# at the directory containing llama-server.exe.
#
# Defaults below target an RTX 3060 (12 GB). Adjust --n-gpu-layers and
# --ctx-size for larger cards.

param(
    [string]$ModelDir = "$PSScriptRoot\..\vendor\models\gpt-oss-20b-GGUF",
    [string]$ModelGlob = "*Q4_K_M*.gguf",
    [int]$Port = 8080,
    [int]$ContextSize = 8192,
    [int]$GpuLayers = 99,           # try to offload everything; llama.cpp will fall back to CPU if VRAM is short
    [int]$Threads = 8,
    [string]$ChatTemplate = ""      # empty = let llama-server auto-detect from GGUF metadata
                                    # (gpt-oss → harmony, Qwen3 → chatml, etc.).
                                    # Override only if the auto-detected template is wrong.
)

$ErrorActionPreference = "Stop"

# Locate llama-server.exe
$LlamaBin = $env:LLAMA_CPP_BIN
if ([string]::IsNullOrEmpty($LlamaBin)) {
    $cmd = Get-Command llama-server.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        $LlamaServer = $cmd.Source
    } else {
        Write-Error "llama-server.exe not found on PATH and `$env:LLAMA_CPP_BIN is unset. Set `$env:LLAMA_CPP_BIN to the folder containing llama-server.exe, or add it to PATH."
        exit 2
    }
} else {
    $LlamaServer = Join-Path $LlamaBin "llama-server.exe"
    if (-not (Test-Path $LlamaServer)) {
        Write-Error "Could not find $LlamaServer. Set `$env:LLAMA_CPP_BIN to the folder containing llama-server.exe."
        exit 2
    }
}

# Locate the GGUF file
$ModelFile = Get-ChildItem -Path $ModelDir -Filter $ModelGlob -ErrorAction SilentlyContinue |
             Select-Object -First 1
if ($null -eq $ModelFile) {
    Write-Error "No model matching '$ModelGlob' under '$ModelDir'. Run scripts/setup_offline.py to download the GGUF, or pass a different -ModelDir / -ModelGlob."
    exit 2
}

Write-Host "starting llama.cpp server"
Write-Host "  binary : $LlamaServer"
Write-Host "  model  : $($ModelFile.FullName)"
Write-Host "  port   : $Port"
Write-Host "  layers : $GpuLayers   (offload-all if VRAM allows)"
Write-Host "  ctx    : $ContextSize"
Write-Host ""
Write-Host "OpenAI-compatible endpoint will be at: http://127.0.0.1:$Port/v1"
Write-Host "Test it with:"
Write-Host "    curl http://127.0.0.1:$Port/v1/models"
Write-Host ""

# --chat-template 은 비어 있을 때만 패스합니다. 비워 두면 llama-server 가
# GGUF 메타데이터에서 자동 탐지 (gpt-oss → harmony, Qwen3 → chatml 등).
$LlamaArgs = @(
    "--model", $ModelFile.FullName,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--ctx-size", "$ContextSize",
    "--n-gpu-layers", "$GpuLayers",
    "--threads", "$Threads",
    "--api-key", "sk-no-key-needed"
)
if (-not [string]::IsNullOrEmpty($ChatTemplate)) {
    $LlamaArgs += "--chat-template", $ChatTemplate
}

& $LlamaServer @LlamaArgs
