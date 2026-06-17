"""diagnose_collator_pass.py — Verify our collator + model forward gives loss.

Bypasses Trainer entirely. Builds ONE batch using DataCollatorForSeq2Seq
(same one sft.py uses), passes through model.forward, prints loss.

If loss>0 → Trainer is the issue (logging suppression, etc.).
If loss=None → collator drops labels.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_DIR = ROOT / "vendor" / "models" / "Qwen_3.5_0.8B-Base"
SFT_FILE = ROOT / "data" / "sft_filtered" / "normal_A1_passed.jsonl"


def main() -> int:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq
    from qwen_tutor.training.model_loader import load_base_model_for_training
    from qwen_tutor.training.formatter import ChatFormatter, LOSS_IGNORE_INDEX
    from qwen_tutor.schemas import SFTExample

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    formatter = ChatFormatter(tokenizer=tok, max_seq_length=2048)

    # Build 4 examples
    rows = []
    with SFT_FILE.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 4: break
            inner = json.loads(line).get("example")
            ex = SFTExample.model_validate(inner)
            tokd = formatter.format_for_training(ex)
            if len(tokd.input_ids) > 2048: continue
            rows.append({
                "input_ids": tokd.input_ids,
                "attention_mask": tokd.attention_mask,
                "labels": tokd.labels,
            })
    print(f"prepared {len(rows)} examples")

    # Use the same collator sft.py uses
    collator = DataCollatorForSeq2Seq(
        tokenizer=tok, padding=True, pad_to_multiple_of=8,
        label_pad_token_id=LOSS_IGNORE_INDEX,
    )
    batch = collator(rows)
    print(f"batch keys: {list(batch.keys())}")
    print(f"batch input_ids shape: {batch['input_ids'].shape}")
    print(f"batch labels shape: {batch['labels'].shape}")
    n_labeled = (batch['labels'] != LOSS_IGNORE_INDEX).sum().item()
    print(f"labeled tokens in batch: {n_labeled}")

    # Load model with QLoRA + LoRA exactly like sft.py
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = load_base_model_for_training(
        str(MODEL_DIR), quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.train()  # training mode

    # Move batch to device
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**batch)
    print(f"\noutput loss: {out.loss}")
    print(f"loss is None: {out.loss is None}")
    if out.loss is not None:
        print(f"loss.item(): {out.loss.item()}")
        print(f"logits shape: {tuple(out.logits.shape)}")

    # Test backward
    if out.loss is not None and out.loss.item() > 0:
        out.loss.backward()
        # Find a LoRA param and check its grad
        for name, p in model.named_parameters():
            if "lora_A" in name and p.requires_grad and p.grad is not None:
                grad_norm = p.grad.norm().item()
                print(f"sample LoRA grad ({name}): norm={grad_norm:.4e}")
                break

    return 0 if out.loss is not None and out.loss.item() > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
