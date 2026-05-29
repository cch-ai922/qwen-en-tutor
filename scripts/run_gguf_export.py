"""run_gguf_export.py  -  merge LoRA -> convert to GGUF -> (optional) quantize.

End-to-end offline pipeline for taking a trained SFT or DPO adapter and
producing a llama-server-ready GGUF file. Reads paths and options from
``config/training.yaml``'s ``gguf_export`` block; everything is
overridable via CLI flags.

Pipeline:
    1. Merge LoRA into base model      -> outputs/merged/ (safetensors)
    2. Convert merged HF model to GGUF -> outputs/gguf/<name>-<dtype>.gguf
    3. (Optional) Quantize             -> outputs/gguf/<name>-<quant>.gguf
    4. (Optional) Clean up merged dir

After this completes, serve with:

    pwsh scripts/start_llama_cpp.ps1 `
        -ModelDir outputs/gguf `
        -ModelGlob "*Q4_K_M*.gguf" `
        -Port 8080

Prereqs:
    * A llama.cpp checkout with ``convert_hf_to_gguf.py``. Path configured
      via ``gguf_export.llama_cpp_repo`` or ``--llama-cpp-repo``. For
      Qwen3-VL (Qwen3.5 4B) you need a recent llama.cpp (Qwen3-VL support
      landed in late 2025).
    * llama-quantize binary if ``quantize`` is set. Configured via
      ``gguf_export.llama_quantize_bin`` or auto-discovered on PATH /
      $env:LLAMA_CPP_BIN.

Examples:
    # Default flow (DPO adapter -> Q4_K_M GGUF):
    python -m scripts.run_gguf_export

    # SFT adapter, no quantization (keep bf16 GGUF only):
    python -m scripts.run_gguf_export --adapter outputs/sft --quantize none

    # Custom output name and Q5_K_M:
    python -m scripts.run_gguf_export --model-name qwen-tutor-A2 --quantize Q5_K_M
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TRAINING_YAML = Path("config/training.yaml")

logger = logging.getLogger("run_gguf_export")


# ---------------------------------------------------------------------------
# Config + CLI plumbing
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_YAML)
    p.add_argument("--adapter", type=str, default=None,
                   help="LoRA adapter dir. Default: training.yaml gguf_export.adapter_path.")
    p.add_argument("--base-model", type=str, default=None,
                   help="Base model path. Default: training.yaml base_model.model_id.")
    p.add_argument("--merged-dir", type=str, default=None,
                   help="Where to put merged safetensors.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to put the final GGUF files.")
    p.add_argument("--model-name", type=str, default=None,
                   help="Filename prefix for the GGUF.")
    p.add_argument("--outtype", type=str, default=None,
                   choices=["bf16", "f16", "f32", "q8_0"],
                   help="convert_hf_to_gguf.py outtype. Default bf16.")
    p.add_argument("--quantize", type=str, default=None,
                   help="Quantization type (e.g. Q4_K_M, Q5_K_M, Q8_0). 'none' to skip.")
    p.add_argument("--llama-cpp-repo", type=str, default=None,
                   help="Path to llama.cpp checkout (must contain convert_hf_to_gguf.py).")
    p.add_argument("--llama-quantize-bin", type=str, default=None,
                   help="Path to llama-quantize binary.")
    p.add_argument("--keep-merged", action="store_true",
                   help="Keep the intermediate merged-safetensors dir after conversion.")
    p.add_argument("--skip-merge", action="store_true",
                   help="Assume --merged-dir already contains a merged model; skip step 1.")
    p.add_argument("--skip-convert", action="store_true",
                   help="Skip the HF-to-GGUF conversion (useful when only re-quantizing).")
    p.add_argument("--verbose", action="store_true")
    return p


def _resolve(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    """CLI > training.yaml > sensible defaults."""
    ge = cfg.get("gguf_export") or {}
    base_cfg = cfg.get("base_model") or {}
    out: dict[str, Any] = {
        "adapter_path": args.adapter or ge.get("adapter_path", "outputs/dpo"),
        "base_model": args.base_model or base_cfg.get("model_id"),
        "trust_remote_code": base_cfg.get("trust_remote_code", True),
        "merged_dir": args.merged_dir or ge.get("merged_dir", "outputs/merged"),
        "keep_merged": args.keep_merged or ge.get("keep_merged", False),
        "output_dir": args.output_dir or ge.get("output_dir", "outputs/gguf"),
        "model_name": args.model_name or ge.get("model_name", "qwen-en-tutor"),
        "outtype": args.outtype or ge.get("outtype", "bf16"),
        "quantize": (args.quantize or ge.get("quantize", "none")).strip(),
        "llama_cpp_repo": args.llama_cpp_repo or ge.get("llama_cpp_repo", "vendor/llama.cpp"),
        "llama_quantize_bin": args.llama_quantize_bin or ge.get("llama_quantize_bin"),
        "skip_merge": args.skip_merge,
        "skip_convert": args.skip_convert,
    }
    if not out["base_model"]:
        raise SystemExit("ERROR: no base model. Pass --base-model or set base_model.model_id in training.yaml.")
    return out


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


def _find_convert_script(llama_cpp_repo: Path) -> Path:
    """Locate llama.cpp's convert_hf_to_gguf.py inside the checkout."""
    candidates = [
        llama_cpp_repo / "convert_hf_to_gguf.py",
        llama_cpp_repo / "convert-hf-to-gguf.py",  # older name
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        f"ERROR: convert_hf_to_gguf.py not found under {llama_cpp_repo}. "
        f"Clone llama.cpp there or pass --llama-cpp-repo."
    )


def _find_quantize_bin(explicit: str | None) -> Path:
    """Locate llama-quantize binary."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise SystemExit(f"ERROR: llama-quantize not at {p}")
    # PATH
    on_path = shutil.which("llama-quantize") or shutil.which("llama-quantize.exe")
    if on_path:
        return Path(on_path)
    # $env:LLAMA_CPP_BIN
    bin_dir = os.environ.get("LLAMA_CPP_BIN")
    if bin_dir:
        for name in ("llama-quantize.exe", "llama-quantize"):
            p = Path(bin_dir) / name
            if p.exists():
                return p
    raise SystemExit(
        "ERROR: llama-quantize not found. Set --llama-quantize-bin, add it to PATH, "
        "or set $env:LLAMA_CPP_BIN to the directory containing it."
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_merge(opts: dict[str, Any]) -> Path:
    merged_dir = Path(opts["merged_dir"]).resolve()
    if opts["skip_merge"]:
        if not merged_dir.exists():
            raise SystemExit(
                f"--skip-merge passed but {merged_dir} doesn't exist. "
                f"Run without --skip-merge first."
            )
        logger.info("[step 1/3] skipping merge; using existing %s", merged_dir)
        return merged_dir

    from qwen_tutor.training.merge import merge_lora

    logger.info("[step 1/3] merging LoRA")
    logger.info("  base    : %s", opts["base_model"])
    logger.info("  adapter : %s", opts["adapter_path"])
    logger.info("  output  : %s", merged_dir)
    merge_lora(
        base_model_id=opts["base_model"],
        adapter_path=opts["adapter_path"],
        output_path=merged_dir,
        trust_remote_code=opts["trust_remote_code"],
    )
    return merged_dir


def step_convert(opts: dict[str, Any], merged_dir: Path) -> Path:
    output_dir = Path(opts["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{opts['model_name']}-{opts['outtype']}.gguf"

    if opts["skip_convert"]:
        if not outfile.exists():
            raise SystemExit(
                f"--skip-convert passed but {outfile} doesn't exist. "
                f"Run without --skip-convert first."
            )
        logger.info("[step 2/3] skipping convert; using existing %s", outfile)
        return outfile

    convert_script = _find_convert_script(Path(opts["llama_cpp_repo"]).resolve())
    logger.info("[step 2/3] converting HF -> GGUF via %s", convert_script)
    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile", str(outfile),
        "--outtype", opts["outtype"],
    ]
    logger.info("  $ %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(
            f"convert_hf_to_gguf.py exited with code {proc.returncode}. "
            f"For Qwen3-VL (Qwen3.5 4B), ensure your llama.cpp is recent enough "
            f"to recognize the architecture (late-2025 or newer)."
        )
    if not outfile.exists():
        raise SystemExit(
            f"convert_hf_to_gguf.py exited 0 but {outfile} was not created — "
            f"check stdout above for the actual output path."
        )
    return outfile


def step_quantize(opts: dict[str, Any], src_gguf: Path) -> Path | None:
    quant = opts["quantize"]
    if not quant or quant.lower() == "none":
        logger.info("[step 3/3] quantize=none; skipping")
        return None

    bin_path = _find_quantize_bin(opts["llama_quantize_bin"])
    output_dir = Path(opts["output_dir"]).resolve()
    out_gguf = output_dir / f"{opts['model_name']}-{quant}.gguf"
    logger.info("[step 3/3] quantizing -> %s", out_gguf)
    cmd = [str(bin_path), str(src_gguf), str(out_gguf), quant]
    logger.info("  $ %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(
            f"llama-quantize exited with code {proc.returncode}. "
            f"Common causes: invalid quant type, source GGUF unreadable."
        )
    return out_gguf


def step_cleanup(opts: dict[str, Any], merged_dir: Path) -> None:
    if opts["keep_merged"] or opts["skip_merge"]:
        return
    logger.info("removing intermediate merged dir %s (keep_merged=false)", merged_dir)
    shutil.rmtree(merged_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    args = _build_argparser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = _load_yaml(args.training_config) if args.training_config.exists() else {}
    opts = _resolve(args, cfg)

    if not Path(opts["adapter_path"]).exists() and not opts["skip_merge"]:
        raise SystemExit(f"ERROR: adapter not found: {opts['adapter_path']}")

    merged_dir = step_merge(opts)
    f16_gguf = step_convert(opts, merged_dir)
    quant_gguf = step_quantize(opts, f16_gguf)
    step_cleanup(opts, merged_dir)

    print("\n=== GGUF export complete ===")
    print(f"  base GGUF        : {f16_gguf}")
    if quant_gguf is not None:
        print(f"  quantized GGUF   : {quant_gguf}")
    print()
    serve_glob = f"*{opts['quantize']}*.gguf" if quant_gguf else f"*{opts['outtype']}*.gguf"
    print("To serve with llama-server:")
    print(
        f'    pwsh scripts/start_llama_cpp.ps1 '
        f'-ModelDir "{opts["output_dir"]}" '
        f'-ModelGlob "{serve_glob}" '
        f'-Port 8080'
    )
    print()
    print("Then call it from Python:")
    print("    from openai import OpenAI")
    print('    c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-no-key-needed")')
    print('    r = c.chat.completions.create(')
    print('        model="qwen-en-tutor",')
    print('        messages=[')
    print('            {"role": "system", "content": "<deployment system prompt>"},')
    print('            {"role": "user", "content": "/no_think\\nHi!"},')
    print('        ],')
    print('    )')
    return 0


if __name__ == "__main__":
    sys.exit(main())
