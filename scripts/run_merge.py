"""run_merge.py  -  step 1 of GGUF export: merge LoRA into base.

Standalone wrapper around ``qwen_tutor.training.merge.merge_lora``. Splits
the slow LoRA merge (loads full-precision base, attaches adapter, saves
merged safetensors) out of ``run_gguf_export.py`` so you only pay it once
and can iterate on conversion / quantization independently.

Reads defaults from ``config/training.yaml``'s ``gguf_export`` block; all
fields are CLI-overridable. Outputs to ``gguf_export.merged_dir``
(default ``outputs/merged``).

Examples:
    # Merge the DPO adapter (config defaults):
    python scripts/run_merge.py

    # Merge the SFT adapter into a different output dir:
    python scripts/run_merge.py --adapter outputs/sft --merged-dir outputs/merged_sft

After this completes, run ``scripts/run_gguf_export.py`` -- it will detect
the existing merged dir and skip the merge step automatically.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TRAINING_YAML = Path("config/training.yaml")

logger = logging.getLogger("run_merge")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_YAML)
    parser.add_argument("--adapter", type=str, default=None,
                        help="LoRA adapter dir. Default: gguf_export.adapter_path.")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model path. Default: base_model.model_id.")
    parser.add_argument("--merged-dir", type=str, default=None,
                        help="Output dir for merged safetensors. Default: gguf_export.merged_dir.")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing merged dir.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = _load_yaml(args.training_config) if args.training_config.exists() else {}
    base_cfg = cfg.get("base_model") or {}
    ge = cfg.get("gguf_export") or {}

    adapter = args.adapter or ge.get("adapter_path", "outputs/dpo")
    base_model = args.base_model or base_cfg.get("model_id")
    merged_dir = Path(args.merged_dir or ge.get("merged_dir", "outputs/merged"))
    trust_remote_code = base_cfg.get("trust_remote_code", True)

    if not base_model:
        print(
            "ERROR: no base model. Pass --base-model or set base_model.model_id in training.yaml.",
            file=sys.stderr,
        )
        return 2
    if not Path(adapter).exists():
        print(f"ERROR: adapter not found: {adapter}", file=sys.stderr)
        return 2

    # Refuse to clobber unless --force.
    if merged_dir.exists() and any(merged_dir.iterdir()):
        if not args.force:
            print(
                f"ERROR: {merged_dir} already exists and is non-empty.\n"
                f"Pass --force to overwrite, or delete it first.",
                file=sys.stderr,
            )
            return 2
        logger.warning("--force: overwriting existing %s", merged_dir)

    from qwen_tutor.training.merge import merge_lora

    logger.info("merging LoRA")
    logger.info("  base    : %s", base_model)
    logger.info("  adapter : %s", adapter)
    logger.info("  output  : %s", merged_dir)
    logger.info("  dtype   : %s", args.torch_dtype)

    out_abs = merge_lora(
        base_model_id=base_model,
        adapter_path=adapter,
        output_path=merged_dir,
        torch_dtype=args.torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    print(f"\n=== merge complete ===")
    print(f"  merged model: {out_abs}")
    print()
    print("Next step: convert to GGUF (and optionally quantize):")
    print("    python scripts/run_gguf_export.py")
    print("(it will detect the merged dir and skip re-merging.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
