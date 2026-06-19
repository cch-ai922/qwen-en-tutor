"""diagnose_persistent_recall.py - per-CEFR x sentinel-position stratification.

Tests whether the V3/V4 recall dip on Persistent-Probe (the longest variants)
is concentrated at C1/C2 -- which would point to SFT max_length truncation
rather than 0.8B capacity-at-distance.

Inputs:
    outputs/paper/score/mechanical/<baseline>/persistent_probe.jsonl
    outputs/paper/score/mechanical/<baseline>/persistent_fp_probe.jsonl
    outputs/paper/score/mechanical/<baseline>/persistent_offposition_probe.jsonl
    eval_sets/persistent_probe.jsonl     (for prefix token-length audit)
    vendor/models/Qwen_3.5_0.8B-Base     (tokenizer for the cap check)

Outputs:
    Per-(expected_turn x cefr_level) fire-rate table for Persistent-Probe.
    Same breakdown for Persistent-FP-Probe (should be ~0% everywhere).
    Per-(expected_turn x cefr_level) token-length distribution for the rendered
    prompt the student sees at inference, with a column flagging records at or
    over the SFT max_length cap.

Usage:
    python scripts/diagnose_persistent_recall.py --baseline paper_a1
    python scripts/diagnose_persistent_recall.py --baseline paper_a1 --max-length 1792
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE_ROOT = ROOT / "outputs" / "paper" / "score" / "mechanical"
EVAL_ROOT = ROOT / "eval_sets"

# CEFR ordering for table rows.
CEFR_ORDER = ("A1", "A2", "B1", "B2", "C1", "C2")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _table(buckets: dict, value_key: str, count_key: str, header: str) -> None:
    """Print a CEFR x turn table from {(turn, level): {value_key, count_key}}."""
    turns = sorted({t for (t, _) in buckets.keys()})
    print()
    print(header)
    print(f"  {'CEFR':<6}", end="")
    for t in turns:
        print(f"  turn={t:<10}", end="")
    print(f"  {'row total':>12}")

    row_totals_v = collections.defaultdict(float)
    row_totals_n = collections.defaultdict(int)
    col_totals_v = collections.defaultdict(float)
    col_totals_n = collections.defaultdict(int)

    for level in CEFR_ORDER:
        print(f"  {level:<6}", end="")
        for t in turns:
            entry = buckets.get((t, level))
            if entry is None or entry[count_key] == 0:
                print(f"  {'-':>14}", end="")
            else:
                rate = entry[value_key] / entry[count_key]
                row_totals_v[level] += entry[value_key]
                row_totals_n[level] += entry[count_key]
                col_totals_v[t] += entry[value_key]
                col_totals_n[t] += entry[count_key]
                print(f"  {rate*100:5.1f}% ({entry[value_key]:>3}/{entry[count_key]:<3})", end="")
        if row_totals_n[level] > 0:
            print(f"  {row_totals_v[level]/row_totals_n[level]*100:5.1f}% "
                  f"({int(row_totals_v[level]):>3}/{row_totals_n[level]:<3})")
        else:
            print(f"  {'-':>14}")

    print(f"  {'col tot':<6}", end="")
    grand_v = grand_n = 0.0
    for t in turns:
        if col_totals_n[t] > 0:
            print(f"  {col_totals_v[t]/col_totals_n[t]*100:5.1f}% "
                  f"({int(col_totals_v[t]):>3}/{col_totals_n[t]:<3})", end="")
            grand_v += col_totals_v[t]
            grand_n += col_totals_n[t]
        else:
            print(f"  {'-':>14}", end="")
    if grand_n > 0:
        print(f"  {grand_v/grand_n*100:5.1f}% ({int(grand_v):>3}/{int(grand_n):<3})")
    else:
        print()


def _fire_rate_table(path: Path, label: str, sign: str) -> None:
    """sign='+' positives (should fire); '-' negatives (should NOT fire)."""
    if not path.exists():
        print(f"\n[skip] {path.name}: not found")
        return
    buckets: dict = collections.defaultdict(lambda: {"fired": 0, "n": 0})
    for rec in _read_jsonl(path):
        turn = rec["result"]["expected_turn"]
        level = rec.get("cefr_level") or "?"
        buckets[(turn, level)]["n"] += 1
        if rec["result"]["fired"]:
            buckets[(turn, level)]["fired"] += 1
    _table(dict(buckets), "fired", "n",
           f"=== {label} ({sign} fire-rate by expected_turn x CEFR) ===")


def _token_length_audit(eval_path: Path, max_length: int) -> None:
    """Tokenize the rendered prompt and report per-bucket length + overflow rate."""
    if not eval_path.exists():
        print(f"\n[skip] token audit: {eval_path.name} not found")
        return

    # Try to load the student tokenizer. We tolerate failure (no transformers
    # / no local model) and fall back to a whitespace word count.
    tokenizer = None
    tokenize_method = "whitespace_words"
    try:
        from transformers import AutoTokenizer
        student_dir = ROOT / "vendor" / "models" / "Qwen_3.5_0.8B-Base"
        if student_dir.exists():
            tokenizer = AutoTokenizer.from_pretrained(
                str(student_dir), trust_remote_code=True
            )
            tokenize_method = "qwen3.5_tokenizer"
        else:
            print(f"\n[warn] {student_dir} not present; falling back to whitespace word count")
    except Exception as exc:
        print(f"\n[warn] could not load Qwen tokenizer ({exc!r}); falling back to whitespace word count")

    def _count(text: str) -> int:
        if tokenizer is None:
            return len(text.split())
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Per-bucket length stats.
    lengths: dict = collections.defaultdict(list)
    for rec in _read_jsonl(eval_path):
        turn = rec["expected"]["sentinel_turn"]
        level = rec.get("cefr_level") or "?"
        prompt_parts = [rec["system_prompt"]]
        for m in rec.get("context_messages", []):
            prompt_parts.append(m.get("content", ""))
        n = _count("\n".join(prompt_parts))
        lengths[(turn, level)].append(n)

    print(f"\n=== Prompt token-length distribution ({tokenize_method}, cap = {max_length}) ===")
    print(f"  {'CEFR':<6}", end="")
    turns = sorted({t for (t, _) in lengths.keys()})
    for t in turns:
        print(f"  turn={t:<22}", end="")
    print()

    for level in CEFR_ORDER:
        print(f"  {level:<6}", end="")
        for t in turns:
            vs = lengths.get((t, level), [])
            if not vs:
                print(f"  {'-':>24}", end="")
                continue
            mn, md, mx = min(vs), sorted(vs)[len(vs)//2], max(vs)
            over = sum(1 for v in vs if v > max_length)
            print(f"  {md:>4} [{mn:>3}-{mx:>4}] {over:>2}/{len(vs):<2} >cap", end="")
        print()


def _per_turn_summary(score_path: Path) -> None:
    """Restate the aggregate the user reported: per-turn fire rate over all CEFR."""
    if not score_path.exists():
        return
    by_turn: dict = collections.defaultdict(lambda: {"fired": 0, "n": 0})
    for rec in _read_jsonl(score_path):
        t = rec["result"]["expected_turn"]
        by_turn[t]["n"] += 1
        if rec["result"]["fired"]:
            by_turn[t]["fired"] += 1
    print("\n=== Per-turn aggregate (sanity check vs user-reported) ===")
    for t in sorted(by_turn):
        e = by_turn[t]
        print(f"  turn {t:>2}: {e['fired']/e['n']*100:5.1f}% "
              f"({e['fired']}/{e['n']})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="paper_a1",
                   help="Which scored baseline to read (default: paper_a1)")
    p.add_argument("--max-length", type=int, default=1792,
                   help="SFT max_length cap to flag overflow against (default: 1792)")
    args = p.parse_args(argv)

    score_dir = SCORE_ROOT / args.baseline
    if not score_dir.exists():
        print(f"ERROR: no scored results for baseline {args.baseline!r} at {score_dir}")
        return 1

    print(f"baseline:   {args.baseline}")
    print(f"score dir:  {score_dir}")
    print(f"eval dir:   {EVAL_ROOT}")
    print(f"cap:        {args.max_length}")

    pp_path  = score_dir / "persistent_probe.jsonl"
    fpp_path = score_dir / "persistent_fp_probe.jsonl"
    op_path  = score_dir / "persistent_offposition_probe.jsonl"

    _per_turn_summary(pp_path)
    _fire_rate_table(pp_path,  "Persistent-Probe (positives)",   "+")
    _fire_rate_table(fpp_path, "Persistent-FP-Probe (negatives)", "-")
    _fire_rate_table(op_path,  "Persistent-OffPosition-Probe (positives, untrained pos)", "+")

    _token_length_audit(EVAL_ROOT / "persistent_probe.jsonl", args.max_length)

    return 0


if __name__ == "__main__":
    sys.exit(main())
