#!/usr/bin/env bash
# install_offline.sh — install the project on a machine with NO internet.
#
# Linux / macOS / WSL counterpart of install_offline.ps1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_WHEELS="$REPO_ROOT/vendor/wheels"
VENV_DIR="$REPO_ROOT/.venv"

if [ ! -d "$VENDOR_WHEELS" ]; then
    echo "vendor/wheels/ not found at $VENDOR_WHEELS." >&2
    echo "Run scripts/setup_offline.py on a connected machine first." >&2
    exit 2
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "[1/5] venv already exists at $VENV_DIR (reusing)"
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "[2/5] upgrading pip from local wheels"
"$PYTHON" -m pip install --no-index --find-links "$VENDOR_WHEELS" \
    --upgrade pip setuptools wheel

echo "[3/5] installing the spaCy English model wheel"
SPACY_WHEEL="$(ls "$VENDOR_WHEELS"/en_core_web_sm-*.whl 2>/dev/null | head -n1 || true)"
if [ -z "$SPACY_WHEEL" ]; then
    echo "  WARNING: no en_core_web_sm wheel under vendor/wheels/. spaCy-dependent filters will fail until you provide one." >&2
else
    "$PIP" install --no-index --find-links "$VENDOR_WHEELS" "$SPACY_WHEEL"
fi

echo "[4/5] installing the project + all deps from vendor/wheels/"
# Prefer the prebuilt project wheel (avoids build-time deps); fall back to
# an editable install with --no-build-isolation if the wheel isn't there.
PROJECT_WHEEL="$(ls "$VENDOR_WHEELS"/qwen_en_tutor-*.whl 2>/dev/null | head -n1 || true)"
if [ -n "$PROJECT_WHEEL" ]; then
    echo "  using prebuilt project wheel: $(basename "$PROJECT_WHEEL")"
    "$PIP" install --no-index --find-links "$VENDOR_WHEELS" "$PROJECT_WHEEL"
    # Dev extras: pytest + pytest-asyncio are needed for the smoke test;
    # ruff is lint-only and installed separately so a missing wheel
    # doesn't kill the whole batch (pip is atomic).
    "$PIP" install --no-index --find-links "$VENDOR_WHEELS" pytest pytest-asyncio
    RUFF_WHEEL="$(ls "$VENDOR_WHEELS"/ruff-*.whl 2>/dev/null | head -n1 || true)"
    if [ -n "$RUFF_WHEEL" ]; then
        "$PIP" install --no-index --find-links "$VENDOR_WHEELS" "$RUFF_WHEEL"
    else
        echo "  (skipping ruff -- no wheel vendored; lint-only, not required to run the project)"
    fi
else
    echo "  no prebuilt project wheel found; doing editable install"
    "$PIP" install --no-index --find-links "$VENDOR_WHEELS" \
        --no-build-isolation -e "$REPO_ROOT[dev]"
fi

echo "[5/5] running the fast test suite to confirm install"
"$PYTHON" -m pytest "$REPO_ROOT/tests" -q -m "not slow"

echo
echo "Install complete."
echo "Activate the venv with:  source $VENV_DIR/bin/activate"
echo "Then start the llama.cpp server (see scripts/start_llama_cpp.sh)"
echo "and run the local pipeline:"
echo "    export QWEN_TUTOR_PROMPTS=compact"
echo "    python scripts/run_full_pipeline.py --config config/pipeline_local.yaml"
