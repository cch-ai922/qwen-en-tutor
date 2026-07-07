"""convert_typed_to_count.py — Build the count-annotated (Design B) persistent
dataset from A1's typed persistent data. paper_v3 H3 (count-marker remedy).

Design B — count-tag EVERY strike (explicit count supervision):

  1st strike (1st redirect turn) : append  [STRIKE: <axis>, 1]
  2nd strike (2nd redirect turn) : append  [STRIKE: <axis>, 2]
  3rd strike (sentinel turn)     : replace the existing
        [SESSION_END: <axis>]  ->  [SESSION_END: <axis>, strike=3]

Rationale: the third-strike count is a LATENT variable the model must infer;
trimming lets it bind to cheap proxies (escalation-presence), causing premature
firing. Externalising the running count as a supervised target at every strike
turn is hypothesised to neutralise the trim vulnerability (H3).

Strike turns are located structurally: for sentinel at assistant index S, the
1st and 2nd redirect (strike) turns are the assistant turns at S-4 and S-2
(the same offsets build_persistent_premature_probe uses). We verify each is an
assistant turn before tagging; records that don't match the expected structure
are passed through unchanged and counted as skipped.

Tags are APPENDED at the END of the turn (after the redirect text), matching how
[SESSION_END] already appears. The [STRIKE: ...] family is DISTINCT from
[SESSION_END: ...] so a dispatcher can tell "counting" from "ending".

This is a data transform ONLY — no teacher regen, no training. Mirrors the
convert_a1_to_a7 / convert_a5_to_a6 pattern. Idempotent (skips turns already
carrying a [STRIKE:...] or strike=3 tag).

Usage:
  python scripts/convert_typed_to_count.py \
      --src-dir data/sft_filtered \
      --dst-dir data/sft_filtered_a8_count_persistent
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Select the count-annotated [persistence] block BEFORE importing prompts.
os.environ["QWEN_TUTOR_SENTINEL_FORMAT"] = "count"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_tutor.generation.prompts import (  # noqa: E402
    align_system_prompt_to_sentinel_format,
)

PERSISTENT_AXES = (
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
)
LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

_TYPED_SENTINEL_RE = re.compile(
    r"\[SESSION_END:\s*(persistent_(?:off_topic|language_violation|persona_break|role_swap))\s*\]"
)
_ALREADY_COUNT_RE = re.compile(r"STRIKE=3")
_ALREADY_STRIKE_RE = re.compile(r"\[STRIKE=2")


def _sentinel_idx(msgs: list[dict]) -> int | None:
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and "[SESSION_END" in (m.get("content") or ""):
            return i
    return None


def convert_record(rec: dict) -> tuple[dict, bool]:
    """Return (new_rec, changed). Tags strikes 1/2 and count-annotates the
    sentinel. Passes through unchanged if structure doesn't match."""
    ex = rec.get("example") or rec
    msgs = ex.get("messages") or []
    S = _sentinel_idx(msgs)
    if S is None:
        return rec, False

    # infer axis from the sentinel content (fall back to metadata)
    m = _TYPED_SENTINEL_RE.search(msgs[S].get("content", ""))
    axis = m.group(1) if m else (
        ex.get("metadata", {}).get("generation", {}).get("axis")
    )
    if not axis:
        return rec, False

    s1, s2 = S - 4, S - 2  # 1st, 2nd strike (redirect) turns
    if s1 < 0 or s2 < 0:
        return rec, False
    if msgs[s1].get("role") != "assistant" or msgs[s2].get("role") != "assistant":
        return rec, False

    new_msgs = [dict(x) for x in msgs]

    # 3rd strike: [SESSION_END: axis] -> [SESSION_END: STRIKE=3: axis]
    c3 = new_msgs[S].get("content", "")
    if not _ALREADY_COUNT_RE.search(c3):
        c3 = _TYPED_SENTINEL_RE.sub(
            lambda mm: f"[SESSION_END: STRIKE=3: {mm.group(1)}]", c3
        )
        new_msgs[S]["content"] = c3

    # 2nd strike ONLY: append [STRIKE=2: axis].
    # Strike 1 is INTENTIONALLY left untagged: a 1st redirect in an escalating
    # sequence is indistinguishable from a one-off (non-escalating) redirect at
    # that moment, so labeling it would require future knowledge. The count
    # becomes observable at strike 2, which is where tagging starts.
    c2 = new_msgs[s2].get("content", "")
    if not _ALREADY_STRIKE_RE.search(c2):
        sep = "" if c2.endswith((" ", "\n")) else " "
        new_msgs[s2]["content"] = f"{c2}{sep}[STRIKE=2: {axis}]"

    new_ex = dict(ex)
    new_ex["messages"] = new_msgs
    # Swap the baked [persistence] block to the count-annotated variant so the
    # deployment prompt DESCRIBES the strike-2/strike-3 markers the messages now
    # contain (fixes the train/prompt mismatch).
    sp = ex.get("system_prompt")
    if sp:
        new_ex["system_prompt"] = align_system_prompt_to_sentinel_format(sp)
    md = dict(ex.get("metadata") or {})
    gen = dict(md.get("generation") or {})
    if gen:
        gen["sentinel"] = f"[SESSION_END: STRIKE=3: {axis}]"
        gen["sentinel_format"] = "count_designB_s2s3"
        md["generation"] = gen
        new_ex["metadata"] = md
    rid = ex.get("id") or ""
    if not rid.startswith("a8_"):
        new_ex["id"] = f"a8_{rid}"

    if "example" in rec:
        out = dict(rec)
        out["example"] = new_ex
        return out, True
    return new_ex, True


def convert_file(src: Path, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = changed = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            new_rec, did = convert_record(rec)
            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            n += 1
            changed += int(did)
    return n, changed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", default="data/sft_filtered",
                   help="Source: A1's typed 4-variant persistent data.")
    p.add_argument("--dst-dir", default="data/sft_filtered_a8_count_persistent")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    if not src_dir.exists():
        raise SystemExit(f"Source dir does not exist: {src_dir}")

    total = total_changed = 0
    print(f"{'file':<55} {'records':>8} {'tagged':>8}")
    print("-" * 74)
    for axis in PERSISTENT_AXES:
        for level in LEVELS:
            src = src_dir / f"{axis}_{level}_passed.jsonl"
            if not src.exists():
                continue
            dst = dst_dir / f"{axis}_{level}_passed.jsonl"
            if args.dry_run:
                with src.open("r", encoding="utf-8") as fh:
                    nrec = sum(1 for L in fh if L.strip())
                print(f"{src.name:<55} {nrec:>8} {'(dry)':>8}")
                total += nrec
                continue
            nrec, nchg = convert_file(src, dst)
            print(f"{src.name:<55} {nrec:>8} {nchg:>8}")
            total += nrec
            total_changed += nchg
    print("-" * 74)
    print(f"{'TOTAL':<55} {total:>8} {total_changed:>8}")
    if not args.dry_run:
        print(f"\nWrote count-annotated (Design B) persistent to {dst_dir}")
        print("Next: assemble full A8 SFT dir (copy a base sft_filtered_* and "
              "overwrite persistent_*_passed.jsonl), then train "
              "config/paper_v2/training_a8_count_sentinel.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
