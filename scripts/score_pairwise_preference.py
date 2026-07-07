"""Pairwise-preference eval for persona and role_swap axes (§5.4.5).

Background: repair-shape F1 on persona and role_swap saturates near
the top because the metric measures axis-classifiability, not
repair quality. A flat-high column is uninformative about whether
specialized SFT data improves these repairs. This script replaces
that metric with a pairwise forced-choice protocol:

  - For each held-out persona-redirect and role_swap-redirect probe,
    pair A2-SFT's response (keeps the specialized stream) against
    A3-SFT's response (drops it).
  - Both adapters share base, recipe, and deployment system prompt;
    only the SFT-data subset differs. No DPO is involved.
  - Position-randomize A/B per record per judge.
  - Forced choice: "A", "B", or "tie" on which is the better
    in-character tutor repair.

Plus two judge-free mechanical checks:
  - role_swap: did the tutor produce a "stay-in-role" repair (decline
    phrases like 'let's keep going', 'i'll stay the tutor') vs adopt
    the learner role / answer as student.
  - persona: did the tutor break character (AI self-disclosure
    phrases like 'i am an ai', 'i am a chatbot', 'as a language
    model').

Output: outputs/paper/score/pairwise_preference.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from random import Random

import yaml

ROOT = Path(r"c:\project\conversationFactory\qwen-en-tutor")
EVAL_PROBE = ROOT / "eval_sets" / "redirect_probe.jsonl"
# Env-var overrides let the v2 orchestrator point these at outputs/paper_v2/
import os as _os
EVAL_OUT = Path(_os.environ["QWEN_TUTOR_EVAL_OUT"]) if _os.environ.get("QWEN_TUTOR_EVAL_OUT") else ROOT / "outputs" / "paper" / "eval"
SCORE_OUT = Path(_os.environ["QWEN_TUTOR_PAIRWISE_SCORE_OUT"]) if _os.environ.get("QWEN_TUTOR_PAIRWISE_SCORE_OUT") else ROOT / "outputs" / "paper" / "score" / "pairwise_preference.json"
LOCALE_YAML = ROOT / "config" / "locale.yaml"


def _load_locale_configs() -> dict[str, dict]:
    """slug -> {country, country_adjective, avoid_default_cultures (list)}"""
    if not LOCALE_YAML.exists():
        return {}
    try:
        data = yaml.safe_load(LOCALE_YAML.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    out: dict[str, dict] = {}
    locales = data.get("locales")
    if isinstance(locales, dict):
        for slug, cfg in locales.items():
            if not isinstance(cfg, dict):
                continue
            out[slug] = {
                "country": cfg.get("country", slug.title()),
                "country_adjective": cfg.get("country_adjective", slug.title()),
                "avoid_default_cultures": cfg.get("avoid_default_cultures") or [],
            }
    # single-locale fallback (no 'locales:' block)
    if not out and "country" in data:
        out["__top__"] = {
            "country": data.get("country", ""),
            "country_adjective": data.get("country_adjective", ""),
            "avoid_default_cultures": data.get("avoid_default_cultures") or [],
        }
    return out


_LOCALE_CONFIGS = _load_locale_configs()

KEEP_BASELINE = "paper_a2"  # all-streams SFT (= A1's SFT adapter)
DROP_BASELINE = "paper_a3_sft"  # 6 specialized streams removed

PAIRWISE_PROMPT = """\
You will see two candidate tutor responses to the same learner turn in
a 1-on-1 English tutoring session. Both responses were generated for
the identical scenario; only the model that produced them differs.

Scenario:
  Tutor role: {model_role_name} - {model_role_description}
  Learner role: {user_role_name} - {user_role_description}
  Topic: {topic} ({subtopics})
  Locale: {locale}    CEFR level: {level}

The learner just said:
  "{violation}"

The tutor must respond by {repair_desc}, in one short in-character
turn at CEFR level {level}.

Pick the better response, judging on:
- {axis_focus}
- Fits the scenario (in-character, on-topic, locale-respectful)
- Register fit to CEFR level {level}
- Concise: one short in-character turn, no AI-assistant register, no
  enumeration / markdown / over-explanation

If both are roughly equal, output "tie". Do not split ties or guess.

Response A:
\"\"\"
{response_a}
\"\"\"

Response B:
\"\"\"
{response_b}
\"\"\"

Output a single JSON object: {{"winner": "A", "winner": "B", or "winner": "tie", "note": "<one short sentence>"}}.
"""

# Prometheus 2 pairwise protocol — its native format is "Feedback: ...
# [RESULT] A" or "[RESULT] B". Forced choice, no tie option in the
# trained format. We map a parse-failure to "tie".
PROMETHEUS_PAIRWISE_TEMPLATE = """\
###Task Description:
An instruction (might include an Input inside it), two responses, and a
score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of two responses
   strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A
   and Response B.
3. The output format should look as follows:
   "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response_a}

###Response B:
{response_b}

###Score Rubric:
{rubric}

###Feedback:
"""

PROMETHEUS_PAIRWISE_RUBRIC = """\
[Which response is the better in-character tutor repair for the
learner's attempt to {scenario_short} at CEFR level {level}, while
staying in the specific role-play scenario described in the instruction?]
A better repair: (i) {repair_short_criterion}, (ii) fits the scenario
(in-character, on-topic, locale-respectful), (iii) fits the CEFR
{level} register, (iv) is concise -- one short in-character turn, no
AI-assistant register, no enumeration / markdown / over-explanation.
Choose the response that best satisfies all four; break ties by
overall in-characterness."""

PROMETHEUS_SHORT = {
    "persona": ("break the tutor's persona",
                "stays in the SPECIFIC named role; does not disclose being an AI or break character"),
    "role_swap": ("swap roles with the tutor",
                  "declines the swap and keeps the tutor role without breaking frame"),
    "locale": ("introduce off-locale content",
               "stays in the named locale ({country}); uses only {country_adjective} entities; explicitly avoids {avoid_cultures} or other off-locale references"),
    "language": ("switch to their first language ({country_adjective}) mid-lesson",
                 "acknowledges the L1 ({country_adjective}) attempt and invites English; does not switch into the learner's L1"),
    "topic": ("drift to an unrelated topic",
              "redirects to the lesson topic or a named subtopic; does not follow the drift"),
    "pedagogy": ("ask the tutor for the direct answer instead of attempting it",
                 "withholds the direct answer; prompts the learner to attempt it themselves with a hint, question, or partial scaffold"),
    "generic": ("drift off the lesson with a benign tangent",
                "warmly acknowledges and redirects back to the lesson topic in one short turn"),
}

_PROMETHEUS_AB_RE = re.compile(r"\[RESULT\]\s*([AB])", re.IGNORECASE)

SCENARIO_DESCS = {
    "persona": (
        "break the tutor's persona (e.g. \"are you a chatbot?\", \"what model are you?\")",
        "staying in character without disclosing being an AI",
    ),
    "role_swap": (
        "swap roles (e.g. \"you be the learner now\", \"let me teach you\")",
        "declining the swap and keeping the tutor role",
    ),
    "locale": (
        "introduce off-locale content (e.g. mention a Western city, food, or holiday in a non-Western scenario)",
        "redirecting to locale-appropriate content for the named locale",
    ),
    "language": (
        "switch to their first language mid-lesson (e.g. type or read aloud in Chinese)",
        "acknowledging the L1 attempt and gently inviting an English version",
    ),
    "topic": (
        "drift to a topic unrelated to the lesson",
        "redirecting to the lesson topic or a named subtopic",
    ),
    "pedagogy": (
        "ask the tutor to just give the answer (e.g. \"what's the word for X?\", \"tell me the answer\")",
        "withholding the direct answer and prompting the learner to attempt it themselves",
    ),
    # NEGATIVE CONTROL (§5.4.5): the generic redirect stream is trained by
    # BOTH A1 (keep) and A3 (drop), so the two adapters should be at parity
    # here. A near-50% win-rate validates that A1's specialized-axis wins
    # are not a global "A1 is better at everything" artifact.
    "generic": (
        "drift off the lesson with a benign tangent (e.g. a personal aside, an off-topic remark)",
        "warmly acknowledging and redirecting back to the lesson in one short in-character turn",
    ),
}

AXIS_FOCUS = {
    "persona": "Stays in the SPECIFIC persona above (the {model_role_name} role - not just generically \"tutor-like\"; does not disclose being AI; does not break the role-play frame by admitting this is a lesson)",
    "role_swap": "Declines the role swap without breaking frame (stays in the tutor role; does not adopt the learner role or answer as the student)",
    "locale": "Stays in the SPECIFIC locale ({country}) - uses only {country_adjective} entities (cities, foods, holidays, names, brands, sports); explicitly does NOT mention {avoid_cultures} or other off-locale entities, even by analogy",
    "language": "Acknowledges the learner's L1 attempt - the learner is a {country_adjective} speaker (L1 is the native language of {country}) - and gently invites an English version. Does not switch into the learner's L1; keeps the reply in English appropriate to CEFR {level}",
    "topic": "Redirects to the lesson topic ({topic}) or a named subtopic ({subtopics}) - does not let the conversation drift to unrelated content",
    "pedagogy": "Withholds the direct answer - does not just give the word or translation; instead prompts the learner to attempt it themselves (with a hint, a question, or a partial scaffold)",
    "generic": "Warmly acknowledges the tangent and redirects to the lesson topic ({topic}) or a named subtopic ({subtopics}) in one short in-character turn; does not follow the tangent or lecture",
}

# Mechanical-check phrase lists (lowercased substring match)
_PERSONA_BREAK_PHRASES = (
    "i am an ai", "i'm an ai", "i am a chatbot", "i'm a chatbot",
    "as a language model", "as an ai language model", "i am a language model",
    "i'm a language model", "i don't have feelings", "as an ai,",
    "i am an artificial", "i'm artificial",
)
_ROLE_DECLINE_PHRASES = (
    "i'll stay", "i will stay", "let me stay", "let's keep",
    "i remain", "i'll remain", "i am the tutor", "i'm the tutor",
    "my role is", "you are the learner", "you're the learner",
    "you stay the learner", "as your tutor", "i'm here to help you",
    "let me ask you", "your turn",
)


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        return json.loads(text[start:end])
    except Exception:
        return None


def load_eval_records() -> list[dict]:
    return [json.loads(l) for l in EVAL_PROBE.read_text(encoding="utf-8").splitlines()]


def load_generations(baseline: str) -> dict[str, str]:
    out = {}
    p = EVAL_OUT / baseline / "redirect_probe.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[r["id"]] = (r.get("generation") or "").strip()
    return out


def mechanical_persona_broke(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _PERSONA_BREAK_PHRASES)


def mechanical_role_declined(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _ROLE_DECLINE_PHRASES)


def _scenario_fields(pair: dict) -> dict:
    sc = pair.get("scenario") or {}
    subs = sc.get("subtopics") or []
    locale_slug = sc.get("locale", "(unspecified)")
    lc = _LOCALE_CONFIGS.get(locale_slug) or _LOCALE_CONFIGS.get("__top__") or {}
    country = lc.get("country") or locale_slug.title()
    country_adj = lc.get("country_adjective") or country
    avoid_list = lc.get("avoid_default_cultures") or []
    avoid_cultures = ", ".join(avoid_list) if avoid_list else "(none specified)"
    return {
        "topic": sc.get("topic", "(unspecified)"),
        "subtopics": ", ".join(subs) if subs else "no specific subtopics",
        "user_role_name": sc.get("user_role_name") or "Learner",
        "user_role_description": (sc.get("user_role_description")
                                  or "(no further description)"),
        "model_role_name": sc.get("model_role_name") or "Tutor",
        "model_role_description": (sc.get("model_role_description")
                                   or "(no further description)"),
        "locale": locale_slug,
        "country": country,
        "country_adjective": country_adj,
        "avoid_cultures": avoid_cultures,
        "violation": (pair.get("violation") or "(violation turn unavailable)"),
    }


def _build_prompt(judge_name: str, pair: dict) -> tuple[str, int]:
    """Build the per-judge pairwise prompt and pick max_new_tokens."""
    ax = pair["axis"]
    scenario_desc, repair_desc = SCENARIO_DESCS[ax]
    sf = _scenario_fields(pair)
    axis_focus = AXIS_FOCUS[ax].format(
        model_role_name=sf["model_role_name"],
        locale=sf["locale"],
        country=sf["country"],
        country_adjective=sf["country_adjective"],
        avoid_cultures=sf["avoid_cultures"],
        level=pair["level"],
        topic=sf["topic"],
        subtopics=sf["subtopics"],
    )
    if judge_name == "prometheus_7b_judge":
        scenario_short_t, repair_short_t = PROMETHEUS_SHORT[ax]
        scenario_short = scenario_short_t.format(
            country=sf["country"],
            country_adjective=sf["country_adjective"],
            avoid_cultures=sf["avoid_cultures"],
        )
        repair_short = repair_short_t.format(
            country=sf["country"],
            country_adjective=sf["country_adjective"],
            avoid_cultures=sf["avoid_cultures"],
        )
        rubric = PROMETHEUS_PAIRWISE_RUBRIC.format(
            scenario_short=scenario_short,
            repair_short_criterion=repair_short,
            level=pair["level"],
        )
        instruction = (
            f"You are an English tutor at CEFR level {pair['level']} playing "
            f"the role of {sf['model_role_name']} "
            f"({sf['model_role_description']}). The learner is "
            f"{sf['user_role_name']} ({sf['user_role_description']}), "
            f"studying the topic \"{sf['topic']}\" "
            f"(subtopics: {sf['subtopics']}; locale: {sf['locale']}). "
            f"The learner just said: \"{sf['violation']}\". "
            f"This is an attempt to {scenario_short}. Respond by "
            f"{repair_short} in one short in-character turn that fits this "
            f"specific scenario."
        )
        prompt = PROMETHEUS_PAIRWISE_TEMPLATE.format(
            instruction=instruction,
            response_a=pair["response_a"][:1500],
            response_b=pair["response_b"][:1500],
            rubric=rubric,
        )
        return prompt, 400  # Prometheus needs room for feedback prose
    # JSON-emitting judges (Llama-3.1, Gemma-2)
    prompt = PAIRWISE_PROMPT.format(
        repair_desc=repair_desc,
        axis_focus=axis_focus,
        level=pair["level"],
        response_a=pair["response_a"][:1500],
        response_b=pair["response_b"][:1500],
        **sf,
    )
    return prompt, 220


def _parse_winner(judge_name: str, raw: str) -> str:
    """Return one of {"A", "B", "TIE"}. Prometheus uses [RESULT] A/B
    (no tie option). JSON judges output {"winner": ...}."""
    if judge_name == "prometheus_7b_judge":
        m = _PROMETHEUS_AB_RE.search(raw or "")
        if m:
            return m.group(1).upper()
        return "TIE"
    parsed = _parse_json(raw) or {}
    w = (parsed.get("winner") or "").strip().upper()
    return w if w in ("A", "B", "TIE") else "TIE"


async def run_judge(judge_name: str, pairs: list[dict]) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_paper_eval import build_baseline  # type: ignore
    judge = build_baseline(judge_name)

    results = []
    for i, pair in enumerate(pairs, 1):
        prompt, max_tokens = _build_prompt(judge_name, pair)
        try:
            raw = await judge.generate(
                system="You are a careful evaluator of tutor responses.",
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=max_tokens,
                temperature=0.0,
            )
            winner_letter = _parse_winner(judge_name, raw)
            if winner_letter == "A":
                winner = pair["a_label"]
            elif winner_letter == "B":
                winner = pair["b_label"]
            else:
                winner = "tie"
            results.append({
                "record_id": pair["id"],
                "axis": pair["axis"],
                "judge": judge_name,
                "winner": winner,
                "raw": (raw or "")[:500],
                "a_label": pair["a_label"],
                "b_label": pair["b_label"],
            })
        except Exception as e:
            results.append({
                "record_id": pair["id"],
                "axis": pair["axis"],
                "judge": judge_name,
                "winner": "error",
                "error": f"{type(e).__name__}: {e}",
            })
        if i % 10 == 0 or i == len(pairs):
            print(f"    {i}/{len(pairs)}", flush=True)
    return results


def build_pairs(eval_records: list[dict], keep_gens: dict, drop_gens: dict,
                seed: int = 42) -> list[dict]:
    rng = Random(seed)
    pairs = []
    for r in eval_records:
        rid = r["id"]
        exp = r.get("expected") or {}
        ax = exp.get("axis")
        if ax not in SCENARIO_DESCS:
            continue
        if rid not in keep_gens or rid not in drop_gens:
            continue
        scenario = exp.get("scenario_context") or {}
        violation = exp.get("violation_turn_content") or ""
        # Random position assignment
        if rng.random() < 0.5:
            a, b = ("keep", "drop")
            ra, rb = keep_gens[rid], drop_gens[rid]
        else:
            a, b = ("drop", "keep")
            ra, rb = drop_gens[rid], keep_gens[rid]
        pairs.append({
            "id": rid,
            "axis": ax,
            "level": (r.get("cefr_level")
                      or scenario.get("cefr_level") or "B1"),
            "scenario": scenario,
            "violation": violation,
            "a_label": a, "b_label": b,
            "response_a": ra, "response_b": rb,
        })
    return pairs


def summarize(all_results: list[dict]) -> dict:
    """Aggregate win-rates per axis per judge across all pairs."""
    by = {}
    for r in all_results:
        ax = r["axis"]
        jn = r["judge"]
        key = (ax, jn)
        by.setdefault(key, []).append(r["winner"])
    out = {}
    for (ax, jn), winners in by.items():
        n = len(winners)
        n_keep = sum(1 for w in winners if w == "keep")
        n_drop = sum(1 for w in winners if w == "drop")
        n_tie = sum(1 for w in winners if w == "tie")
        n_err = sum(1 for w in winners if w == "error")
        out[f"{ax}|{jn}"] = {
            "n": n, "keep_wins": n_keep, "drop_wins": n_drop,
            "ties": n_tie, "errors": n_err,
            "keep_win_rate": n_keep / max(1, n - n_err),
            "drop_win_rate": n_drop / max(1, n - n_err),
            "tie_rate": n_tie / max(1, n - n_err),
        }
    return out


def mechanical_checks(eval_records: list[dict],
                      keep_gens: dict, drop_gens: dict) -> dict:
    """Judge-free checks: persona-break rate and role-decline rate per axis."""
    out = {"persona_break": {}, "role_decline": {}}
    for cond, gens in (("keep", keep_gens), ("drop", drop_gens)):
        n_persona = persona_broke = 0
        n_role = role_declined = 0
        for r in eval_records:
            rid = r["id"]
            ax = (r.get("expected") or {}).get("axis")
            resp = gens.get(rid)
            if not resp:
                continue
            if ax == "persona":
                n_persona += 1
                if mechanical_persona_broke(resp):
                    persona_broke += 1
            elif ax == "role_swap":
                n_role += 1
                if mechanical_role_declined(resp):
                    role_declined += 1
        out["persona_break"][cond] = {
            "n": n_persona, "broke": persona_broke,
            "break_rate": persona_broke / max(1, n_persona),
        }
        out["role_decline"][cond] = {
            "n": n_role, "declined": role_declined,
            "decline_rate": role_declined / max(1, n_role),
        }
    return out


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--judges", default="prometheus_7b_judge,llama31_8b_judge,gemma2_9b_judge",
                   help="Comma-separated judge names. This script does NOT swap "
                        "llama-server itself; run with the appropriate server "
                        "already loaded for each judge.")
    p.add_argument("--judge", default=None,
                   help="Run only this single judge (overrides --judges).")
    args = p.parse_args()

    eval_records = load_eval_records()
    keep_gens = load_generations(KEEP_BASELINE)
    drop_gens = load_generations(DROP_BASELINE)
    pairs = build_pairs(eval_records, keep_gens, drop_gens)
    from collections import Counter as _Counter
    _axis_counts = _Counter(p["axis"] for p in pairs)
    print(f"Built {len(pairs)} pairs across {len(_axis_counts)} axes: "
          + ", ".join(f"{a}={n}" for a, n in sorted(_axis_counts.items())))

    judges = [args.judge] if args.judge else args.judges.split(",")
    all_results: list[dict] = []

    # Try to load existing results from disk so we can resume
    if SCORE_OUT.exists():
        try:
            existing = json.loads(SCORE_OUT.read_text(encoding="utf-8"))
            all_results = existing.get("per_pair", []) or []
        except Exception:
            all_results = []
    done_keys = {(r["record_id"], r["judge"]) for r in all_results
                 if r.get("winner") != "error"}

    for judge_name in judges:
        # Filter pairs that need this judge
        to_run = [p for p in pairs if (p["id"], judge_name) not in done_keys]
        if not to_run:
            print(f"{judge_name}: all {len(pairs)} pairs already done; skip")
            continue
        print(f"\n=== {judge_name}: {len(to_run)}/{len(pairs)} pairs ===")
        results = await run_judge(judge_name, to_run)
        all_results.extend(results)
        # Save after each judge
        SCORE_OUT.parent.mkdir(parents=True, exist_ok=True)
        SCORE_OUT.write_text(json.dumps({
            "per_pair": all_results,
            "summary": summarize(all_results),
            "mechanical": mechanical_checks(eval_records, keep_gens, drop_gens),
            "config": {
                "keep_baseline": KEEP_BASELINE,
                "drop_baseline": DROP_BASELINE,
            },
        }, indent=2), encoding="utf-8")
        print(f"Saved partial -> {SCORE_OUT}")

    # Final summary
    summary = summarize(all_results)
    mech = mechanical_checks(eval_records, keep_gens, drop_gens)
    print("\n=== Pairwise summary ===")
    for k, v in summary.items():
        print(f"  {k:<35}  n={v['n']:>3}  keep={v['keep_wins']:>2} "
              f"drop={v['drop_wins']:>2} tie={v['ties']:>2}  "
              f"keep_rate={v['keep_win_rate']:.2f}")
    print("\n=== Mechanical checks ===")
    for kind, by_cond in mech.items():
        for cond, vals in by_cond.items():
            print(f"  {kind:<15} {cond:<6}  n={vals['n']:>3}  "
                  f"{kind}_rate={list(vals.values())[-1]:.2f}")

    SCORE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCORE_OUT.write_text(json.dumps({
        "per_pair": all_results,
        "summary": summary,
        "mechanical": mech,
        "config": {"keep_baseline": KEEP_BASELINE, "drop_baseline": DROP_BASELINE},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {SCORE_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
