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
    [string]$ChatTemplate = "",     # empty = let llama-server auto-detect from GGUF metadata
                                    # (gpt-oss → harmony, Qwen3 → chatml, etc.).
                                    # Override only if the auto-detected template is wrong.
    #[string]$BindHost = "0.0.0.0" # network interface llama-server binds to.
    [string]$BindHost = "127.0.0.1" # network interface llama-server binds to.
                                    # default = loopback only (this machine).
                                    # use "0.0.0.0" or a LAN IP to expose to other
                                    # machines on the local network.
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
Write-Host "  bind   : $BindHost"
Write-Host "  port   : $Port"
Write-Host "  layers : $GpuLayers   (offload-all if VRAM allows)"
Write-Host "  ctx    : $ContextSize"
Write-Host ""
# Endpoint advertised in the banner. When bound to 0.0.0.0 we cannot know
# the LAN IP at parse time, so show a "127.0.0.1 (and LAN)" hint instead.
$AdvertiseHost = if ($BindHost -eq "0.0.0.0") { "<this-machine-LAN-ip>" } else { $BindHost }
Write-Host "OpenAI-compatible endpoint will be at: http://${AdvertiseHost}:$Port/v1"
Write-Host "Test it with (from this machine):"
Write-Host "    curl http://127.0.0.1:$Port/v1/models"
if ($BindHost -eq "0.0.0.0" -or $BindHost -ne "127.0.0.1") {
    Write-Host "Test it with (from another LAN machine):"
    Write-Host "    curl http://<this-machine-LAN-ip>:$Port/v1/models"
    Write-Host "WARNING: this server is reachable on the LAN."
    Write-Host "Allow port $Port through the Windows firewall (one-time, elevated PS):"
    Write-Host "    New-NetFirewallRule -DisplayName 'llama-server' -Direction Inbound -LocalPort $Port -Protocol TCP -Action Allow"
}
Write-Host ""

# --chat-template is only passed when explicitly set. Leaving it empty lets
# llama-server auto-detect from the GGUF metadata (gpt-oss -> harmony,
# Qwen3 -> chatml, etc.).
$LlamaArgs = @(
    "--model", $ModelFile.FullName,
    "--host", $BindHost,
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
