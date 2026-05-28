"""LoRA-adapter merge utility.

Loads the base Qwen3 model, attaches a trained PEFT adapter (from SFT or
DPO), merges the LoRA weights into the base, and writes a standalone
model directory ready for vLLM / Ollama / standard ``from_pretrained``
loading.

Quantization is intentionally turned off during merge — bnb-quantized
weights cannot be merged in place — so this step temporarily needs the
full-precision base in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def merge_lora(
    base_model_id: str,
    adapter_path: str | Path,
    output_path: str | Path,
    *,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    save_tokenizer: bool = True,
) -> str:
    """Merge ``adapter_path`` into ``base_model_id`` and save under ``output_path``.

    Returns the absolute path to the merged model directory.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from qwen_tutor.training.model_loader import load_base_model_for_training

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("loading base model %s in %s for merge", base_model_id, torch_dtype)
    base = load_base_model_for_training(
        base_model_id,
        torch_dtype=getattr(torch, torch_dtype),
        attn_implementation="sdpa",
        trust_remote_code=trust_remote_code,
    )
    logger.info("attaching adapter %s", adapter_path)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    logger.info("merging LoRA weights into base")
    merged = model.merge_and_unload()
    logger.info("saving merged model to %s", out)
    merged.save_pretrained(str(out), safe_serialization=True)
    if save_tokenizer:
        tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
        tok.save_pretrained(str(out))
    return str(out.resolve())
