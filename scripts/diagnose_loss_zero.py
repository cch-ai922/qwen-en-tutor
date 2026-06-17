"""diagnose_loss_zero.py — Pin down why A1 SFT trained at loss=0.

Loads ONE real SFT record, runs it through ChatFormatter.format_for_training,
and reports:
  - total token count
  - count of labels != -100
  - the encoded marker bytes the formatter is looking for
  - the actual bytes around each <|im_start|> in the full tokenization
  - whether the marker scan finds any assistant spans

Non-destructive: read-only on the tokenizer + one data file.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_tutor.training.formatter import ChatFormatter, LOSS_IGNORE_INDEX  # noqa: E402
from qwen_tutor.schemas import SFTExample  # noqa: E402

MODEL_DIR = ROOT / "vendor" / "models" / "Qwen_3.5_0.8B-Base"
SFT_FILE = ROOT / "data" / "sft_filtered" / "normal_A1_passed.jsonl"


def main() -> int:
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {MODEL_DIR}...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    print(f"  vocab_size={tok.vocab_size}, pad={tok.pad_token_id}, eos={tok.eos_token_id}")

    formatter = ChatFormatter(tokenizer=tok, max_seq_length=2048)

    # 1. Show how the marker tokenizes in isolation
    marker_str = "<|im_start|>assistant\n"
    end_str = "<|im_end|>"
    start_ids = tok.encode(marker_str, add_special_tokens=False)
    end_ids = tok.encode(end_str, add_special_tokens=False)
    print(f"\nMarker '{marker_str!r}' encodes to: {start_ids}")
    print(f"  decoded back: {tok.decode(start_ids)!r}")
    print(f"End marker '{end_str!r}' encodes to: {end_ids}")

    # 2. Load ONE real SFTExample and format it
    if not SFT_FILE.exists():
        print(f"\nMISSING: {SFT_FILE}")
        return 1
    with SFT_FILE.open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    # Filter wrapper: actual record is under 'example'.
    inner = first.get("example", first)
    ex = SFTExample.model_validate(inner)
    print(f"\nLoaded SFTExample id={ex.id}  messages={len(ex.messages)}")

    # 3. Run formatter
    try:
        tokd = formatter.format_for_training(ex)
    except Exception as exc:
        print(f"FORMATTER RAISED: {type(exc).__name__}: {exc}")
        return 2

    n_total = len(tokd.input_ids)
    n_labeled = sum(1 for L in tokd.labels if L != LOSS_IGNORE_INDEX)
    print(f"\n total tokens : {n_total}")
    print(f" labeled tokens: {n_labeled} ({100*n_labeled/n_total:.1f}%)")

    if n_labeled == 0:
        print("\n*** ROOT CAUSE: zero labeled tokens. ***")
        # Find every <|im_start|> in the actual full tokenization
        print("\nLocating every <|im_start|> in the tokenized chat...")
        im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        print(f"  <|im_start|> token id = {im_start_id}")
        positions = [i for i, t in enumerate(tokd.input_ids) if t == im_start_id]
        print(f"  found {len(positions)} <|im_start|> tokens at positions {positions[:20]}...")
        for p in positions[:6]:
            window = tokd.input_ids[p : p + 8]
            decoded = tok.decode(window)
            print(f"    pos {p}: tokens={window} decoded={decoded!r}")

        # Compare what the formatter expects vs what's actually there
        print(f"\nFormatter expects marker={start_ids} (len {len(start_ids)})")
        print(f"What's actually at the first <|im_start|>?")
        if positions:
            p = positions[0]
            actual = tokd.input_ids[p : p + len(start_ids)]
            print(f"  actual={actual}")
            print(f"  match={actual == start_ids}")

    # 4. Show a small slice of input_ids + labels alignment
    print("\nFirst 40 tokens preview:")
    for i in range(min(40, n_total)):
        tid = tokd.input_ids[i]
        lbl = tokd.labels[i]
        tok_str = tok.decode([tid])
        marker = "  ←labeled" if lbl != LOSS_IGNORE_INDEX else ""
        print(f"  [{i:3d}] id={tid:>7d} tok={tok_str!r:<24s} lbl={lbl:>7d}{marker}")

    return 0 if n_labeled > 0 else 3


if __name__ == "__main__":
    sys.exit(main())
