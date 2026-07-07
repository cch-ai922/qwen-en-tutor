"""score_withholding_rate.py — v2 §5.4 contribution-1 load-bearing experiment.

For each pedagogy_redirect probe response across all baseline conditions,
ask a single judge a binary classification:

    "Did the tutor WITHHOLD the direct answer and scaffold (yes/no)?"

Aggregate to per-condition rate. The decisive contrast is:

    A1/A5/A6/A7 (trained + instructed) vs A3 (instructed-only, no
    specialized stream) vs B1-B4 (instructed-only baselines).

If trained conditions withhold substantially more than instructed-only
ones, the §5.4 claim (specialized SFT data is necessary for
describable-but-not-promptable behavior) survives.

Reads eval generations from ``outputs/paper_v2/eval/<baseline>/redirect_probe.jsonl``,
filters records where ``expected.stream == "pedagogy_redirect"``, queries
the active llama-server with a single yes/no prompt per record, and
writes a JSON summary to the output path.

Usage:
    python scripts/score_withholding_rate.py \\
        --judge llama31_8b_judge \\
        --eval-dir outputs/paper_v2/eval \\
        --out outputs/paper_v2/score/pedagogy_withholding_rate.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# Conditions to score. v2 paper-facing tags map to eval-dir names per
# paper/EVAL_SESSION_v2_HANDOFF.md §1.1.
BASELINES = (
    "paper_a2",        # v2 A1 (SFT only, axis-specific, 4-variant)
    "paper_a3_sft",    # A3
    "paper_a5_sft",    # A5
    "paper_a6_sft",    # A6
    "paper_a7_sft",    # A7
    "qwen3_5_0_8b_base",
    "qwen3_5_0_8b_instruct",
    "qwen3_5_4b_instruct",
    "qwen3_5_9b_teacher",
)

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

# Binary classification prompt. The judge sees the scenario context, the
# user's violation turn (which asks for the direct answer), and the tutor's
# response. Asked to classify withholding behavior.
WITHHOLDING_PROMPT_TEMPLATE = """\
You are a strict pedagogy reviewer evaluating a single English-tutor response.

The learner asked the tutor for a direct answer to a language question (a
word translation, a grammar rule, a definition, etc.). A WELL-FORMED tutor
response WITHHOLDS the direct answer and instead scaffolds: prompts the
learner to attempt the answer themselves, gives a hint, asks a leading
question, or offers a partial structure for them to fill in. A POORLY-FORMED
response gives the direct answer (with or without surrounding pleasantries).

Conversation context:
- Topic: {topic}
- Subtopics: {subtopics}
- Learner role: {user_role_name} — {user_role_description}
- Tutor role: {model_role_name} — {model_role_description}
- Learner CEFR level: {cefr_level}

The LEARNER's question (the violation turn that should be redirected):
"{violation_turn_content}"

The TUTOR's response (to be classified):
"{tutor_response}"

Question: Did the tutor WITHHOLD the direct answer and scaffold?

Reply with EXACTLY one of:
- yes  (tutor withheld the answer, asked back / hinted / scaffolded)
- no   (tutor gave the direct answer, even if also adding scaffolding)

Reply now with just "yes" or "no" — no other text."""


def _llama_yesno(prompt: str, judge: str, max_retries: int = 3) -> bool | None:
    """Send the prompt to the local llama-server, return True (withheld),
    False (gave answer), or None (parse error)."""
    body = {
        "model": judge,
        "messages": [
            {"role": "system", "content": "You are a strict binary classifier. Reply with only 'yes' or 'no'."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        LLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
                txt = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
                # Tolerate "yes.", "no,", "**yes**", etc.
                m = re.search(r"\b(yes|no)\b", txt)
                if m:
                    return m.group(1) == "yes"
                return None
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [judge error after {max_retries} retries]: {e}", flush=True)
                return None
            time.sleep(2)
    return None


def _score_one_condition(baseline: str, eval_dir: Path, judge: str,
                         probe_file: str = "redirect_probe.jsonl") -> dict[str, Any]:
    f = eval_dir / baseline / probe_file
    if not f.exists():
        return {"baseline": baseline, "error": f"missing {f}"}
    records = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if (rec.get("expected") or {}).get("stream") == "pedagogy_redirect":
            records.append(rec)
    print(f"\n=== {baseline}: {len(records)} pedagogy records ===", flush=True)
    if not records:
        return {"baseline": baseline, "n_records": 0, "withheld": 0,
                "withhold_rate": None, "judge_errors": 0}
    withheld = 0
    errors = 0
    per_record = []
    for i, rec in enumerate(records):
        ctx = (rec.get("expected") or {}).get("scenario_context") or {}
        viol = (rec.get("expected") or {}).get("violation_turn_content", "")
        gen = (rec.get("generation") or "").strip()
        prompt = WITHHOLDING_PROMPT_TEMPLATE.format(
            topic=ctx.get("topic", ""),
            subtopics=", ".join(ctx.get("subtopics", []) or []) or "(none)",
            user_role_name=ctx.get("user_role_name", "learner"),
            user_role_description=ctx.get("user_role_description", "a learner"),
            model_role_name=ctx.get("model_role_name", "tutor"),
            model_role_description=ctx.get("model_role_description", "an English tutor"),
            cefr_level=ctx.get("cefr_level", rec.get("cefr_level", "A2")),
            violation_turn_content=viol,
            tutor_response=gen,
        )
        verdict = _llama_yesno(prompt, judge)
        if verdict is None:
            errors += 1
            per_record.append({"id": rec.get("id"), "withheld": None})
        else:
            if verdict:
                withheld += 1
            per_record.append({"id": rec.get("id"), "withheld": verdict})
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(records)} processed, {withheld} withheld, {errors} judge errors", flush=True)
    n_valid = len(records) - errors
    rate = (withheld / n_valid) if n_valid > 0 else None
    print(f"  {baseline} FINAL: {withheld}/{n_valid} withheld ({rate:.3f} rate)" if rate is not None
          else f"  {baseline} FINAL: all judge errors", flush=True)
    return {
        "baseline": baseline,
        "n_records": len(records),
        "n_valid": n_valid,
        "withheld": withheld,
        "withhold_rate": rate,
        "judge_errors": errors,
        "per_record": per_record,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True,
                    help="Judge identifier as known to llama-server (e.g., llama31_8b_judge)")
    ap.add_argument("--eval-dir", default="outputs/paper_v2/eval",
                    help="Directory containing per-baseline subdirs with redirect_probe.jsonl")
    ap.add_argument("--out", default="outputs/paper_v2/score/pedagogy_withholding_rate.json")
    ap.add_argument("--baselines", default=",".join(BASELINES),
                    help="Comma-separated baseline names to score (default: all known)")
    ap.add_argument("--probe-file", default="redirect_probe.jsonl",
                    help="Per-baseline probe filename to read (e.g. pedagogy_extra_probe.jsonl)")
    args = ap.parse_args()

    # Health-check the server before starting the long run.
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=10) as r:
            body = json.load(r)
            served = body.get("data", [{}])[0].get("id", "")
            print(f"Judge server reports model: {served}")
    except Exception as e:
        print(f"ERROR: judge server not reachable: {e}", file=sys.stderr)
        return 1

    eval_dir = Path(args.eval_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    results = []
    t0 = time.time()
    for b in baselines:
        results.append(_score_one_condition(b, eval_dir, args.judge,
                                            probe_file=args.probe_file))

    summary = {
        "judge": args.judge,
        "eval_dir": str(eval_dir),
        "axis": "pedagogy",
        "metric": "withholding_rate",
        "runtime_s": round(time.time() - t0, 1),
        "results": results,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    # Print a compact summary table to stdout
    print("\n=== Summary ===")
    print(f"{'baseline':<28} {'n':>4} {'withheld':>9} {'rate':>7} {'errors':>7}")
    print("-" * 60)
    for r in results:
        rate = r.get("withhold_rate")
        rate_s = f"{rate:.3f}" if rate is not None else "  n/a"
        print(f"{r['baseline']:<28} {r.get('n_records', 0):>4} "
              f"{r.get('withheld', 0):>9} {rate_s:>7} {r.get('judge_errors', 0):>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
