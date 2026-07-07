"""build_withholding_probe.py — build the ONE canonical pedagogy-withholding
probe (n=63) and, optionally, assemble a condition's combined-63 generation
file from its two historical halves.

Background / why this exists
----------------------------
The §5.4 withholding rate is scored over 63 pedagogy_redirect probes: the 21
that live inside ``eval_sets/redirect_probe.jsonl`` (stream=pedagogy_redirect)
PLUS 42 freshly-generated held-out probes in
``eval_sets/pedagogy_extra_probe.jsonl``. Historically these were scored
separately (orig21 + extra) and pooled — which was easy to misread as "n=21"
because only 21 pedagogy records are literally in redirect_probe. To remove
that confusion permanently we merge them into ONE file:

    eval_sets/pedagogy_withholding_probe.jsonl   (63 records)

Both source files already carry ``test_set="redirect_probe"`` and
``expected.stream="pedagogy_redirect"`` with the scenario_context +
violation_turn_content the withholding judge needs, so the merged file is a
drop-in that ``run_paper_eval.py`` generates and ``score_withholding_rate.py``
scores with no special-casing (score at n=63 in one pass).

Modes
-----
(default)  Build the canonical probe file from the two source eval sets.
--assemble-gens EVAL_DIR   For each --baselines subdir under EVAL_DIR that has
           BOTH redirect_probe.jsonl and pedagogy_extra_probe.jsonl
           generations, write a combined pedagogy_withholding_probe.jsonl (the
           21 pedagogy records from redirect_probe + the 42 from the extra),
           reusing existing generations (NO re-inference). Use --extra-dir when
           the extra-probe generations live in a different directory (seed-42's
           extra gens are under outputs/paper_v2/eval_pedagogy_extra).

Usage
-----
  # 1) build the canonical eval set (once)
  python scripts/build_withholding_probe.py

  # 2) consolidate seed-42 + baselines (extra gens are in eval_pedagogy_extra)
  python scripts/build_withholding_probe.py --assemble-gens outputs/paper_v2/eval \\
      --extra-dir outputs/paper_v2/eval_pedagogy_extra \\
      --baselines paper_a2,paper_a3_sft,qwen3_5_0_8b_base,qwen3_5_0_8b_instruct,qwen3_5_4b_instruct,qwen3_5_9b_teacher
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_SETS = ROOT / "eval_sets"
CANON = "pedagogy_withholding_probe.jsonl"
REDIRECT = "redirect_probe.jsonl"
EXTRA = "pedagogy_extra_probe.jsonl"
PED = "pedagogy_redirect"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _ped_only(recs: list[dict]) -> list[dict]:
    return [r for r in recs if (r.get("expected") or {}).get("stream") == PED]


def _merge_dedup(a: list[dict], b: list[dict]) -> list[dict]:
    """Concatenate, dropping duplicate ids (redirect's 21 and the extra 42 are
    built to be non-overlapping, but dedup defensively; first occurrence wins)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in a + b:
        rid = r.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    return out


def build_canonical() -> int:
    orig = _ped_only(_read(EVAL_SETS / REDIRECT))
    extra = _read(EVAL_SETS / EXTRA)
    # sanity: the extra file is all-pedagogy by construction
    extra_ped = _ped_only(extra)
    if len(extra_ped) != len(extra):
        print(f"WARN: {len(extra) - len(extra_ped)} extra records are not "
              f"stream={PED}; keeping only the pedagogy ones")
    merged = _merge_dedup(orig, extra_ped)
    out = EVAL_SETS / CANON
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
                   encoding="utf-8")
    print(f"canonical withholding probe: {out}")
    print(f"  redirect(pedagogy)={len(orig)}  extra={len(extra_ped)}  "
          f"merged(dedup)={len(merged)}")
    if len(merged) != 63:
        print(f"  NOTE: merged n={len(merged)} (expected 63) — check the source sets")
    return 0


def assemble_gens(eval_dir: Path, extra_dir: Path, baselines: list[str]) -> int:
    print(f"assembling combined-63 generation files under {eval_dir}")
    print(f"  (extra-probe generations read from {extra_dir})")
    for b in baselines:
        rp = eval_dir / b / REDIRECT
        xp = extra_dir / b / EXTRA
        orig = _ped_only(_read(rp))
        extra = _ped_only(_read(xp))
        if not orig and not extra:
            print(f"  {b}: no generations found (skip)")
            continue
        merged = _merge_dedup(orig, extra)
        out = eval_dir / b / CANON
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
                       encoding="utf-8")
        flag = "" if len(merged) == 63 else f"  <-- n={len(merged)}, expected 63"
        print(f"  {b}: redirect_ped={len(orig)} + extra={len(extra)} -> {out.name} "
              f"({len(merged)}){flag}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemble-gens", metavar="EVAL_DIR", default=None,
                    help="assemble combined-63 generation files under this eval dir")
    ap.add_argument("--extra-dir", default=None,
                    help="dir holding the extra-probe generations (default: same "
                         "as --assemble-gens)")
    ap.add_argument("--baselines",
                    default="paper_a2,paper_a3_sft,qwen3_5_0_8b_base,"
                            "qwen3_5_0_8b_instruct,qwen3_5_4b_instruct,qwen3_5_9b_teacher")
    args = ap.parse_args()

    if args.assemble_gens:
        eval_dir = Path(args.assemble_gens)
        extra_dir = Path(args.extra_dir) if args.extra_dir else eval_dir
        bl = [b.strip() for b in args.baselines.split(",") if b.strip()]
        return assemble_gens(eval_dir, extra_dir, bl)
    return build_canonical()


if __name__ == "__main__":
    raise SystemExit(main())
