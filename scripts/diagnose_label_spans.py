"""diagnose_label_spans.py — Show which tokens get loss labels.

For one real SFT record, print:
  - the tokenized chat with role boundaries shown,
  - which token ranges are labeled (assistant turns),
  - which token ranges are masked (system + user turns),
  - decoded text of labeled vs masked regions so you can visually verify.

Used to confirm that system + user prompts are correctly masked (-100)
and only assistant turns contribute to the SFT loss.
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


def find_runs(labels: list[int]) -> list[tuple[str, int, int]]:
    """Return [(kind, start, end_exclusive)] where kind is 'mask' or 'label'."""
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(labels):
        kind = "mask" if labels[i] == LOSS_IGNORE_INDEX else "label"
        j = i
        while j < len(labels) and ((labels[j] == LOSS_IGNORE_INDEX) == (kind == "mask")):
            j += 1
        out.append((kind, i, j))
        i = j
    return out


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    formatter = ChatFormatter(tokenizer=tok, max_seq_length=2048)

    with SFT_FILE.open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    ex = SFTExample.model_validate(first.get("example", first))
    tokd = formatter.format_for_training(ex)

    total = len(tokd.input_ids)
    n_labeled = sum(1 for L in tokd.labels if L != LOSS_IGNORE_INDEX)
    n_masked = total - n_labeled

    print(f"id={ex.id}")
    print(f"messages={len(ex.messages)} (system={sum(1 for m in ex.messages if m.role=='system')}, "
          f"user={sum(1 for m in ex.messages if m.role=='user')}, "
          f"assistant={sum(1 for m in ex.messages if m.role=='assistant')})")
    print(f"total tokens: {total}")
    print(f"  labeled (assistant): {n_labeled} ({100*n_labeled/total:.1f}%)")
    print(f"  masked (system + user): {n_masked} ({100*n_masked/total:.1f}%)")
    print()

    runs = find_runs(tokd.labels)
    print(f"{'kind':<6} {'tokens':<10}  preview")
    print("-" * 100)
    for kind, s, e in runs:
        snippet = tok.decode(tokd.input_ids[s:e], skip_special_tokens=False)
        snippet = snippet.replace("\n", "\\n")
        if len(snippet) > 90:
            snippet = snippet[:87] + "..."
        print(f"{kind:<6} [{s:>4}:{e:<4}] {e-s:>3}  {snippet}")
    print()

    # Sanity check: every labeled span should start with "<|im_start|>assistant\n"
    print("=== sanity check: every LABELED span starts with <|im_start|>assistant\\n ===")
    labeled_starts = [s for kind, s, _ in runs if kind == "label"]
    expected_marker = "<|im_start|>assistant\n"
    all_good = True
    for s in labeled_starts:
        end = min(s + 5, total)
        prefix = tok.decode(tokd.input_ids[s:end], skip_special_tokens=False)
        ok = prefix.startswith("<|im_start|>assistant")
        if not ok:
            all_good = False
        print(f"  span at [{s:>4}]: starts with {prefix[:30]!r} — {'OK' if ok else 'WRONG'}")
    if all_good:
        print("  ✓ all labeled spans correctly start with assistant marker")

    print()
    print("=== sanity check: no LABELED span overlaps a system or user role ===")
    # Decode the FIRST labeled chunk fully to verify it contains the
    # expected assistant content.
    if labeled_starts:
        s, e = labeled_starts[0], next(end for kind, st, end in runs if kind == "label" and st == labeled_starts[0])
        full = tok.decode(tokd.input_ids[s:e], skip_special_tokens=False)
        for forbidden in ("<|im_start|>system", "<|im_start|>user"):
            if forbidden in full:
                print(f"  PROBLEM: first labeled span contains {forbidden!r}")
                all_good = False
        if all_good:
            print("  ✓ first labeled span contains no system/user markers")
            print("    (first 200 chars of first labeled chunk:)")
            print(f"    {full[:200]}")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
