"""Bundle every dependency the project needs into ``vendor/`` so the
target machine can install + run with no internet access.

Run this ONCE on a workstation with internet. It populates:

    vendor/
      wheels/             pip wheels for every Python dep (+ spaCy model)
      models/Qwen3-8B/    HF snapshot of the base model
      models/gpt-oss-20b-GGUF/   GGUF weight (Q4_K_M by default)
      manifest.json       what was downloaded + when

Then ZIP the whole project directory (including vendor/) and copy it
to the offline machine. Run ``scripts/install_offline.ps1`` there to
build a venv and install from the local wheels.

Steps can be run individually with ``--only wheels``, ``--only models``,
etc. — useful when re-downloading just one piece.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
WHEELS = VENDOR / "wheels"
MODELS = VENDOR / "models"
# Match this against whatever spaCy major.minor pip resolves to. The model
# wheel pins `spacy<3.X+1`, so an older model release will refuse to install
# alongside a newer spaCy. 3.8.0 is the current compatible release.
SPACY_WHEEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
QWEN_REPO = "Qwen/Qwen3-8B"
DEFAULT_GPT_OSS_REPO = "bartowski/gpt-oss-20b-GGUF"
DEFAULT_GPT_OSS_PATTERN = "*Q4_K_M*.gguf"


# ---------------------------------------------------------------------------
# Step 1: pip wheels for every dep listed in pyproject.toml
# ---------------------------------------------------------------------------


def _read_pyproject_deps(extras: bool) -> list[str]:
    """Parse pyproject.toml and return the flat list of runtime (+ optional
    dev) dependency specifiers."""
    try:
        import tomllib   # Python 3.11+
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    pyproject = ROOT / "pyproject.toml"
    with pyproject.open("rb") as fh:
        doc = tomllib.load(fh)
    project = doc.get("project", {})
    deps = list(project.get("dependencies", []))
    if extras:
        for extra_name, items in (project.get("optional-dependencies") or {}).items():
            if extra_name == "dev":
                deps.extend(items)
    return deps


def step_wheels(extras: bool = True) -> None:
    WHEELS.mkdir(parents=True, exist_ok=True)
    deps = _read_pyproject_deps(extras=extras)
    # Things we depend on at runtime that aren't in pyproject yet, plus the
    # build-system deps (`hatchling` + its transitives) that pip needs when
    # it runs `pip install -e .` against this project on the offline machine.
    # ``editables`` is what hatchling uses to install in editable mode.
    extra_deps = [
        "wheel",
        "hatchling",
        "editables",
        "pathspec",
        "pluggy",
        "trove-classifiers",
        "tqdm",
        "huggingface_hub",
        "pytest-asyncio",
    ]
    all_deps = sorted({*deps, *extra_deps})
    logger.info("downloading %d top-level deps + transitives -> %s", len(all_deps), WHEELS)
    logger.debug("deps: %s", all_deps)
    # pip resolves transitives, picks wheels matching the current platform.
    subprocess.run(
        [
            sys.executable, "-m", "pip", "download",
            "--dest", str(WHEELS),
            *all_deps,
        ],
        check=True,
    )
    logger.info("wheels OK: %d files", len(list(WHEELS.glob("*"))))


# ---------------------------------------------------------------------------
# Step 1c: pre-build a wheel of THIS project so the offline machine can
# install it without invoking hatchling at all (build isolation needs
# network access by default; this saves us from chasing every hatchling
# transitive in vendor/wheels/).
# ---------------------------------------------------------------------------


def step_project_wheel() -> None:
    WHEELS.mkdir(parents=True, exist_ok=True)
    logger.info("building project wheel -> %s", WHEELS)
    # `pip wheel .` builds the project and writes the wheel to --wheel-dir.
    subprocess.run(
        [
            sys.executable, "-m", "pip", "wheel",
            "--wheel-dir", str(WHEELS),
            "--no-deps",          # we already have the deps
            str(ROOT),
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Step 1b (optional): swap the CPU torch wheel for a CUDA build
#
# `pip download torch` from PyPI ships the CPU-only build, which is fine for
# the RTX-3060 gen+filter phase (no torch is loaded there) but won't actually
# run training on a GPU. Run --only cuda-torch (or include it in --only all)
# to add the CUDA wheel from the PyTorch index BEFORE shipping the bundle to
# a GPU host.
# ---------------------------------------------------------------------------


CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu{cuda}"


def step_cuda_torch(cuda: str) -> None:
    WHEELS.mkdir(parents=True, exist_ok=True)
    index_url = CUDA_TORCH_INDEX.format(cuda=cuda)
    # Drop any CPU-only torch wheels we already have so install picks up the
    # CUDA one. Heuristic: wheels without "+cu" in the filename are CPU.
    removed = 0
    for wheel in WHEELS.glob("torch-*.whl"):
        if "+cu" not in wheel.name:
            wheel.unlink()
            removed += 1
    if removed:
        logger.info("removed %d CPU torch wheel(s) before pulling CUDA build", removed)
    logger.info("downloading CUDA-%s torch -> %s", cuda, WHEELS)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "download",
            "--dest", str(WHEELS),
            "--no-deps",          # transitives already in vendor/wheels/
            "--index-url", index_url,
            "torch",
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Step 2: spaCy en_core_web_sm wheel (separate URL, not on PyPI)
# ---------------------------------------------------------------------------


def step_spacy() -> None:
    WHEELS.mkdir(parents=True, exist_ok=True)
    logger.info("downloading spaCy model wheel -> %s", WHEELS)
    # `pip download <URL>` lands the wheel into WHEELS for offline install.
    subprocess.run(
        [
            sys.executable, "-m", "pip", "download",
            "--dest", str(WHEELS),
            "--no-deps",
            SPACY_WHEEL_URL,
        ],
        check=True,
    )
    logger.info("spaCy wheel OK")


# ---------------------------------------------------------------------------
# Step 3: HuggingFace snapshots — Qwen3-8B base + gpt-oss-20b GGUF
# ---------------------------------------------------------------------------


def step_models(gpt_oss_repo: str, gpt_oss_pattern: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub is not installed in the CURRENT Python — install it "
            "first (`pip install huggingface_hub`) before running --only models, "
            "or run --only wheels first so it's pulled into vendor/wheels/."
        )
        sys.exit(2)

    MODELS.mkdir(parents=True, exist_ok=True)

    # Qwen3-8B base (training target). ~16 GB, mostly safetensors.
    # NOTE: `local_dir_use_symlinks` was removed in huggingface_hub 1.x. With
    # just `local_dir=` the files land directly under that path (the default
    # since 0.23). Resumable + retries are handled by the library.
    # qwen_dir = MODELS / "Qwen3-8B"
    # logger.info("downloading %s -> %s", QWEN_REPO, qwen_dir)
    # snapshot_download(
    #     repo_id=QWEN_REPO,
    #     local_dir=str(qwen_dir),
    #     ignore_patterns=["*.msgpack", "*.h5"],  # skip TF/Flax artifacts
    # )

    # gpt-oss-20b GGUF (the teacher served by llama.cpp).
    gguf_dir = MODELS / "gpt-oss-20b-GGUF"
    logger.info("downloading %s (pattern=%s) -> %s", gpt_oss_repo, gpt_oss_pattern, gguf_dir)
    snapshot_download(
        repo_id=gpt_oss_repo,
        local_dir=str(gguf_dir),
        allow_patterns=[gpt_oss_pattern, "*.json", "*.md"],
    )
    logger.info("models OK")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(gpt_oss_repo: str, gpt_oss_pattern: str) -> None:
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host_python": sys.version,
        "host_platform": platform.platform(),
        "host_pip": subprocess.check_output(
            [sys.executable, "-m", "pip", "--version"], text=True
        ).strip(),
        "wheel_count": len(list(WHEELS.glob("*"))) if WHEELS.exists() else 0,
        "models": {
            "qwen3_base": str((MODELS / "Qwen3-8B").relative_to(ROOT)),
            "gpt_oss_repo": gpt_oss_repo,
            "gpt_oss_pattern": gpt_oss_pattern,
            "gpt_oss_dir": str((MODELS / "gpt-oss-20b-GGUF").relative_to(ROOT)),
        },
        "spacy_wheel_url": SPACY_WHEEL_URL,
    }
    VENDOR.mkdir(parents=True, exist_ok=True)
    (VENDOR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download every dep into vendor/ for offline install.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--only",
        choices=["wheels", "project-wheel", "cuda-torch", "spacy", "models", "all"],
        default="all",
    )
    p.add_argument(
        "--cuda",
        default="121",
        help="CUDA version for the torch wheel (e.g. 121, 124, 128). "
        "Only used when --only is cuda-torch or all.",
    )
    p.add_argument(
        "--gpt-oss-repo",
        default=DEFAULT_GPT_OSS_REPO,
        help="HF repo for gpt-oss-20b GGUF builds",
    )
    p.add_argument(
        "--gpt-oss-pattern",
        default=DEFAULT_GPT_OSS_PATTERN,
        help="glob pattern for the gguf file (default: Q4_K_M)",
    )
    p.add_argument(
        "--no-extras",
        action="store_true",
        help="skip dev extras (pytest, ruff) — saves a few MB",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    VENDOR.mkdir(parents=True, exist_ok=True)
    if args.only in ("wheels", "all"):
        step_wheels(extras=not args.no_extras)
    if args.only in ("project-wheel", "all"):
        step_project_wheel()
    if args.only in ("cuda-torch", "all"):
        step_cuda_torch(args.cuda)
    if args.only in ("spacy", "all"):
        step_spacy()
    if args.only in ("models", "all"):
        step_models(args.gpt_oss_repo, args.gpt_oss_pattern)
    write_manifest(args.gpt_oss_repo, args.gpt_oss_pattern)
    print(f"\nvendor bundle ready under {VENDOR}")
    print("  next: zip the project (including vendor/) and copy to the offline machine.")
    print("  then run: scripts/install_offline.ps1   (Windows)")
    print("       or:  bash scripts/install_offline.sh   (Linux)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
