"""diagnose_quantization.py — Verify QLoRA 4-bit is actually applied.

Loads the model with the same BitsAndBytesConfig sft.py uses, inspects
the dtype + class of every linear layer in the language tower, and
reports total VRAM. If quantization worked, most linear layers should be
`Linear4bit`/`Params4bit` and total weight memory should be ~500MB for
the 0.8B model.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_DIR = ROOT / "vendor" / "models" / "Qwen_3.5_0.8B-Base"


def main() -> int:
    import torch
    from transformers import BitsAndBytesConfig
    from qwen_tutor.training.model_loader import load_base_model_for_training

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {MODEL_DIR} with bnb 4-bit...")
    model = load_base_model_for_training(
        str(MODEL_DIR), quantization_config=bnb,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True,
    )
    print(f"Model class: {type(model).__name__}")
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"VRAM reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # Inspect layer dtypes
    from collections import Counter
    dtype_counter = Counter()
    class_counter = Counter()
    for name, module in model.named_modules():
        cls = type(module).__name__
        if any(k in cls for k in ("Linear", "Embedding")):
            class_counter[cls] += 1
            # Get param dtype
            for p in module.parameters(recurse=False):
                dtype_counter[(cls, str(p.dtype))] += 1
                break

    print(f"\nLinear+Embedding layer classes: {dict(class_counter)}")
    print(f"\nLayer (class, param dtype) counts:")
    for (cls, dtype), cnt in dtype_counter.most_common():
        print(f"  {cnt:>4}  {cls:<20s} {dtype}")

    # Sample a known LM-head/embedding for inspection
    print("\nSample modules:")
    for path in ("model.language_model.layers.0.self_attn.q_proj",
                 "model.language_model.layers.0.mlp.gate_proj",
                 "language_model.layers.0.self_attn.q_proj",
                 "lm_head"):
        for n, m in model.named_modules():
            if n.endswith(path) or n == path:
                p = next(m.parameters(), None)
                print(f"  {n}: cls={type(m).__name__} dtype={p.dtype if p is not None else 'no params'}")
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
