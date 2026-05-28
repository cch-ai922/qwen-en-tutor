# install_offline.ps1 — install the project on a machine with NO internet.
#
# Prereq: vendor/wheels/ already populated by `python scripts/setup_offline.py`
# on a workstation with internet. The whole project (including vendor/) has
# then been copied to this machine.
#
# This script:
#   1. Creates .venv in the project root
#   2. Installs every dep from vendor/wheels/ with --no-index --find-links
#   3. Installs the spaCy en_core_web_sm wheel
#   4. Installs the project itself in editable mode
#   5. Sanity-checks the install by running the fast test suite
#
# Run from the project root:
#     pwsh scripts/install_offline.ps1
# or
#     powershell -ExecutionPolicy Bypass -File scripts/install_offline.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path "$PSScriptRoot/..").Path
$VendorWheels = Join-Path $RepoRoot "vendor\wheels"
$VenvDir = Join-Path $RepoRoot ".venv"

if (-not (Test-Path $VendorWheels)) {
    Write-Error "vendor/wheels/ not found at $VendorWheels. Run scripts/setup_offline.py on a connected machine first."
    exit 2
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "[1/5] creating venv at $VenvDir"
    python -m venv $VenvDir
} else {
    Write-Host "[1/5] venv already exists at $VenvDir (reusing)"
}

$Python = Join-Path $VenvDir "Scripts\python.exe"
$Pip    = Join-Path $VenvDir "Scripts\pip.exe"

Write-Host "[2/5] upgrading pip from local wheels"
& $Python -m pip install --no-index --find-links $VendorWheels --upgrade pip setuptools wheel

Write-Host "[3/5] installing the spaCy English model wheel"
$SpacyWheels = Get-ChildItem $VendorWheels -Filter "en_core_web_sm-*.whl" -ErrorAction SilentlyContinue
if ($SpacyWheels.Count -eq 0) {
    Write-Warning "no en_core_web_sm wheel found in vendor/wheels/. The diversity tracker + locale judge will fail until you provide one."
} else {
    & $Pip install --no-index --find-links $VendorWheels $SpacyWheels[0].FullName
}

Write-Host "[4/5] installing the project + all deps from vendor/wheels/"
# Two paths:
#   A) If setup_offline.py ran with --only project-wheel (or all), there's a
#      prebuilt qwen_en_tutor-*.whl in vendor/wheels/. Install that — no
#      build-time deps required on this machine.
#   B) Otherwise fall back to editable install with --no-build-isolation,
#      which needs hatchling+editables already on PATH (we pulled them as
#      part of the wheel set, so they're available in vendor/wheels/).
$PrebuiltProjectWheel = Get-ChildItem $VendorWheels -Filter "qwen_en_tutor-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -ne $PrebuiltProjectWheel) {
    Write-Host "  using prebuilt project wheel: $($PrebuiltProjectWheel.Name)"
    & $Pip install --no-index --find-links $VendorWheels $PrebuiltProjectWheel.FullName
    # Dev extras (the prebuilt wheel doesn't carry [dev]).
    # pytest + pytest-asyncio are required to run the smoke test below.
    # ruff is optional (lint-only) and installed separately so that a
    # missing ruff wheel doesn't kill the whole batch (pip is atomic).
    & $Pip install --no-index --find-links $VendorWheels pytest pytest-asyncio
    $RuffWheel = Get-ChildItem $VendorWheels -Filter "ruff-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $RuffWheel) {
        & $Pip install --no-index --find-links $VendorWheels $RuffWheel.FullName
    } else {
        Write-Host "  (skipping ruff -- no wheel vendored; lint-only, not required to run the project)"
    }
} else {
    Write-Host "  no prebuilt project wheel found; doing editable install"
    & $Pip install --no-index --find-links $VendorWheels --no-build-isolation -e "$RepoRoot[dev]"
}

Write-Host "[5/5] running the fast test suite to confirm install"
& $Python -m pytest "$RepoRoot\tests" -q -m "not slow"

Write-Host ""
Write-Host "Install complete." -ForegroundColor Green
Write-Host "Activate the venv with:  & $VenvDir\Scripts\Activate.ps1"
Write-Host "Then start the llama.cpp server (see scripts/start_llama_cpp.ps1)"
Write-Host "and run the local pipeline:"
Write-Host "    `$env:QWEN_TUTOR_PROMPTS = 'compact'"
Write-Host "    python scripts/run_full_pipeline.py --config config/pipeline_local.yaml"
