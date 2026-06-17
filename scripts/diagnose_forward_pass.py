"""diagnose_forward_pass.py — Verify loss>0 on a single forward pass.

Loads the Qwen3.5-0.8B-Base model (multimodal arch wrapper), runs ONE
real SFT example through the formatter, then through model.forward(),
and reports the loss. Run twice with different settings to isolate:

  Test A: bf16 on GPU, no quantization, no LoRA — pure forward.
  Test B: 4-bit QLoRA + LoRA-prepared, same inputs — what training sees.

If A gives loss>0 but B gives loss=0, the bug is in QLoRA/PEFT plumbing.
If A also gives loss=0, the bug is in the formatter/labels or model
forward path for multimodal arch.

Non-destructive: no training, no checkpoint writes.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_DIR = ROOT / "vendor" / "models" / "Qwen_3.5_0.8B-Base"
SFT_FILE = ROOT / "data" / "sft_filtered" / "normal_A1_passed.jsonl"


def _load_one_example():
    import torch
    from transformers import AutoTokenizer
    from qwen_tutor.training.formatter import ChatFormatter
    from qwen_tutor.schemas import SFTExample

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    formatter = ChatFormatter(tokenizer=tok, max_seq_length=2048)
    with SFT_FILE.open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    inner = first.get("example", first)
    ex = SFTExample.model_validate(inner)
    tokd = formatter.format_for_training(ex)
    input_ids = torch.tensor([tokd.input_ids], dtype=torch.long)
    attention_mask = torch.tensor([tokd.attention_mask], dtype=torch.long)
    labels = torch.tensor([tokd.labels], dtype=torch.long)
    return tok, input_ids, attention_mask, labels


def test_a_bf16_no_lora():
    """Pure forward, bf16, no quantization, no PEFT."""
    import torch
    from qwen_tutor.training.model_loader import load_base_model_for_training

    print("\n=== Test A: bf16 + no quant + no LoRA ===")
    tok, input_ids, attention_mask, labels = _load_one_example()

    model = load_base_model_for_training(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="cuda:0",
    )
    model.eval()

    input_ids = input_ids.to("cuda:0")
    attention_mask = attention_mask.to("cuda:0")
    labels = labels.to("cuda:0")
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f"  loss = {out.loss}")
    print(f"  loss.item() = {out.loss.item() if out.loss is not None else 'None'}")
    print(f"  logits.shape = {tuple(out.logits.shape) if out.logits is not None else 'None'}")
    if out.logits is not None:
        print(f"  logits[0, 0, :5] = {out.logits[0, 0, :5].float().cpu().tolist()}")
    # Free GPU memory before next test
    del model, out
    torch.cuda.empty_cache()


def test_b_qlora_lora():
    """4-bit QLoRA + LoRA, training-mode model."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    from qwen_tutor.training.model_loader import load_base_model_for_training

    print("\n=== Test B: 4-bit QLoRA + LoRA + prepare_model_for_kbit_training ===")
    tok, input_ids, attention_mask, labels = _load_one_example()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = load_base_model_for_training(
        str(MODEL_DIR),
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    labels = labels.to(model.device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f"  loss = {out.loss}")
    print(f"  loss.item() = {out.loss.item() if out.loss is not None else 'None'}")
    if out.logits is not None:
        print(f"  logits.shape = {tuple(out.logits.shape)}")
        print(f"  logits[0, 0, :5] = {out.logits[0, 0, :5].float().cpu().tolist()}")


def main() -> int:
    test_a_bf16_no_lora()
    test_b_qlora_lora()
    return 0


if __name__ == "__main__":
    sys.exit(main())
