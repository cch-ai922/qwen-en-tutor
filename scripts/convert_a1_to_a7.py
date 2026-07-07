"""convert_a1_to_a7.py — Build the A7 persistent dataset from A1/A2's
data by string-replacing the sentinel and re-rendering the deployment
system prompt with the generic [persistence] block.

Why
----
A7 closes out the 2×2 ablation for paper §5.4.x:

                  4-variant persistent     fixed turn-7
  axis-specific:        A2                       A5
  generic [SESSION_END]: A7  (this script)        A6

  - A2 vs A7  isolates the axis-label effect with natural (4-variant) position spread.
  - A5 vs A6  isolates the axis-label effect with fixed turn-7 anchoring.
  - A2 vs A5  isolates the position-anchoring effect with the axis label held.
  - A6 vs A7  isolates the position-anchoring effect with the label held generic.

What this does
--------------
For every persistent record in ``data/sft_filtered/<axis>_<level>_passed.jsonl``
(the A1/A2 4-variant persistent dataset):
  1. Re-render ``system_prompt`` with QWEN_TUTOR_SENTINEL_FORMAT=generic.
  2. Replace every ``[SESSION_END: persistent_<axis>]`` in assistant
     turns with ``[SESSION_END]``.
  3. Fix ``metadata.generation.sentinel`` for audit-trail accuracy.
  4. Prefix the record id with ``a7_`` so A1/A7 ids stay distinct.
  5. Write to ``data/sft_filtered_a7_persistent/``.

Re-running on the same source dir is idempotent (overwrites destination).

Pre-req: ``data/sft_filtered/persistent_*_passed.jsonl`` exists (it does
after Stage 2's regen of the main pipeline).

Usage
-----
::

  python scripts/convert_a1_to_a7.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# MUST be set BEFORE prompts is imported.
os.environ["QWEN_TUTOR_SENTINEL_FORMAT"] = "generic"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_tutor.generation.prompts import render_scenario_deployment_system_prompt  # noqa: E402

SRC_DIR = ROOT / "data" / "sft_filtered"                  # A1/A2's persistent files live here
DST_DIR = ROOT / "data" / "sft_filtered_a7_persistent"    # destination

PERSISTENT_AXES = (
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
)

_SENTINEL_RE = re.compile(
    r"\[SESSION_END:\s*persistent_(?:off_topic|language_violation|persona_break|role_swap)\s*\]"
)


def _rerender_system_prompt(metadata: dict) -> str:
    user_role = metadata.get("user_role") or {}
    model_role = metadata.get("model_role") or {}
    subtopics = metadata.get("subtopics") or []
    return render_scenario_deployment_system_prompt(
        cefr_level=metadata.get("cefr_level", "B1"),
        locale_name=metadata.get("locale"),
        topic=metadata.get("topic", ""),
        subtopics=subtopics,
        user_role_name=user_role.get("name", ""),
        user_role_description=user_role.get("description", ""),
        model_role_name=model_role.get("name", "Tutor"),
        model_role_description=model_role.get("description", ""),
    )


def _replace_sentinel_in_messages(messages: list[dict]) -> tuple[list[dict], int]:
    out = []
    replaced = 0
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            new_content, n = _SENTINEL_RE.subn("[SESSION_END]", content)
            if n:
                replaced += n
            new_m = dict(m)
            new_m["content"] = new_content
            out.append(new_m)
        else:
            out.append(m)
    return out, replaced


def convert_record(rec: dict) -> tuple[dict, int]:
    ex = rec.get("example") or rec
    new_ex = dict(ex)
    new_ex["system_prompt"] = _rerender_system_prompt(ex.get("metadata") or {})
    new_ex["messages"], n_replaced = _replace_sentinel_in_messages(
        ex.get("messages") or []
    )
    md = ex.get("metadata") or {}
    if isinstance(md, dict):
        new_md = dict(md)
        gen_md = new_md.get("generation")
        if isinstance(gen_md, dict) and "sentinel" in gen_md:
            new_gen_md = dict(gen_md)
            new_gen_md["sentinel"] = "[SESSION_END]"
            new_gen_md["sentinel_format"] = "generic"
            new_md["generation"] = new_gen_md
            new_ex["metadata"] = new_md
    rid = ex.get("id") or ""
    if not rid.startswith("a7_"):
        new_ex["id"] = f"a7_{rid}"
    if "example" in rec:
        new_rec = dict(rec)
        new_rec["example"] = new_ex
        return new_rec, n_replaced
    return new_ex, n_replaced


def convert_file(src: Path, dst: Path) -> tuple[int, int]:
    n_rec = 0
    n_sent = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            new_rec, replaced = convert_record(rec)
            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            n_rec += 1
            n_sent += replaced
    return n_rec, n_sent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", default=str(SRC_DIR),
                   help="Source dir (A1/A2 main filtered).")
    p.add_argument("--dst-dir", default=str(DST_DIR),
                   help="Destination dir (A7 persistent filtered).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    if not src_dir.exists():
        raise SystemExit(f"Source dir does not exist: {src_dir}")

    # Sanity: confirm env var was read by prompts.py.
    test_prompt = render_scenario_deployment_system_prompt(
        cefr_level="A1", topic="taking a taxi", subtopics=["going to a park"],
        user_role_description="a woman traveling", model_role_name="Driver",
        model_role_description="a taxi driver",
    )
    if "[SESSION_END]" not in test_prompt or "[SESSION_END: persistent_" in test_prompt:
        raise SystemExit(
            "prompts.py is rendering the axis-specific [persistence] block — "
            "QWEN_TUTOR_SENTINEL_FORMAT=generic didn't take effect. Check the "
            "env var set order in this script (must be set BEFORE the import)."
        )
    print("Sanity check: rendered system prompt contains generic [SESSION_END]. OK.")
    print()

    total_records = 0
    total_replaced = 0
    print(f"{'file':<60} {'records':>8} {'replaced':>9}")
    print("-" * 80)
    for axis in PERSISTENT_AXES:
        for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
            src = src_dir / f"{axis}_{level}_passed.jsonl"
            dst = dst_dir / f"{axis}_{level}_passed.jsonl"
            if not src.exists():
                continue
            if args.dry_run:
                with src.open("r", encoding="utf-8") as fh:
                    n_rec = sum(1 for L in fh if L.strip())
                print(f"{src.name:<60} {n_rec:>8} {'(dry)':>9}")
                total_records += n_rec
                continue
            n_rec, n_sent = convert_file(src, dst)
            print(f"{src.name:<60} {n_rec:>8} {n_sent:>9}")
            total_records += n_rec
            total_replaced += n_sent

    print("-" * 80)
    print(f"{'TOTAL':<60} {total_records:>8} {total_replaced:>9}")
    print()
    if not args.dry_run:
        print(f"Wrote A7 persistent to {dst_dir}")
        print()
        print("Next: python scripts/setup_paper_ablation_data.py --conditions a7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
