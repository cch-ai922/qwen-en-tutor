#!/usr/bin/env bash
# start_llama_cpp.sh — start the llama.cpp HTTP server with gpt-oss-20b.
#
# Prereq: a pre-built `llama-server` binary on this machine. Either set
# LLAMA_CPP_BIN to the directory containing it, or add it to PATH. The
# easiest source on Linux is the prebuilt release tarball from
#     https://github.com/ggerganov/llama.cpp/releases

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/vendor/models/gpt-oss-20b-GGUF}"
MODEL_GLOB="${MODEL_GLOB:-*Q4_K_M*.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"
GPU_LAYERS="${GPU_LAYERS:-99}"
THREADS="${THREADS:-8}"
# empty default = let llama-server auto-detect from GGUF metadata.
# Override only if the auto-detected template is wrong.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
# network interface llama-server binds to. Default = loopback only.
# Use "0.0.0.0" or a specific LAN IP to expose the teacher to other
# machines on the local network.
BIND_HOST="${BIND_HOST:-127.0.0.1}"


if [ -n "${LLAMA_CPP_BIN:-}" ]; then
    LLAMA_SERVER="$LLAMA_CPP_BIN/llama-server"
elif command -v llama-server >/dev/null 2>&1; then
    LLAMA_SERVER="$(command -v llama-server)"
else
    echo "llama-server not on PATH and LLAMA_CPP_BIN is unset." >&2
    echo "Install llama.cpp prebuilt binaries and either set LLAMA_CPP_BIN" >&2
    echo "or add the directory to PATH." >&2
    exit 2
fi

MODEL_FILE="$(ls $MODEL_DIR/$MODEL_GLOB 2>/dev/null | head -n1 || true)"
if [ -z "$MODEL_FILE" ]; then
    echo "no model matching '$MODEL_GLOB' under '$MODEL_DIR'." >&2
    echo "Run scripts/setup_offline.py first." >&2
    exit 2
fi

echo "starting llama.cpp server"
echo "  binary : $LLAMA_SERVER"
echo "  model  : $MODEL_FILE"
echo "  bind   : $BIND_HOST"
echo "  port   : $PORT"
echo "  layers : $GPU_LAYERS"
echo "  ctx    : $CTX"
echo
if [ "$BIND_HOST" = "0.0.0.0" ]; then
    ADVERTISE_HOST="<this-machine-LAN-ip>"
else
    ADVERTISE_HOST="$BIND_HOST"
fi
echo "OpenAI-compatible endpoint: http://$ADVERTISE_HOST:$PORT/v1"
echo "Test it with (from this machine):"
echo "    curl http://127.0.0.1:$PORT/v1/models"
if [ "$BIND_HOST" != "127.0.0.1" ]; then
    echo "Test from another LAN machine:"
    echo "    curl http://<this-machine-LAN-ip>:$PORT/v1/models"
    echo "WARNING: this server is reachable on the LAN. Ensure your firewall is configured accordingly."
fi
echo

ARGS=(
    --model "$MODEL_FILE"
    --host "$BIND_HOST"
    --port "$PORT"
    --ctx-size "$CTX"
    --n-gpu-layers "$GPU_LAYERS"
    --threads "$THREADS"
    --api-key sk-no-key-needed
)
# Only pass --chat-template when explicitly set (otherwise llama-server
# auto-detects from the GGUF: gpt-oss → harmony, Qwen3 → chatml, ...).
if [ -n "$CHAT_TEMPLATE" ]; then
    ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi
exec "$LLAMA_SERVER" "${ARGS[@]}"
