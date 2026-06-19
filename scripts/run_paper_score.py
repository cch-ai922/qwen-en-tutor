"""run_paper_score.py — Score paper-eval generations.

Takes the output of ``scripts/run_paper_eval.py`` and produces metric
verdicts for the paper's results table.

Two kinds of scoring:

  MECHANICAL (no judge needed; deterministic and exact):
    - sentinel_firing : Persistent-Probe — did the generation fire the
                        sentinel? At the expected turn?
    - locale_leakage  : Locale-Leakage — does the generation mention a
                        Western-default entity from the gazetteer?

  JUDGED (calls an LLM judge once per record per judge):
    - cefr_adherence  : Tutor-Scenario — does the response match the
                        target CEFR level? 1-5 scale.
    - redirect_axis   : Redirect-Probe — what axis does the response
                        handle? Multi-class.
    - naturalness     : Tutor-Scenario — is the response natural English?
                        1-5 scale.

Invocation patterns:

  # Mechanical only (no judge needed, fast):
  python scripts/run_paper_score.py --baseline paper_a1 --metrics mechanical

  # Add judge scoring with the 9B LAN teacher as one judge:
  python scripts/run_paper_score.py --baseline paper_a1 --metrics judged \
      --judge qwen3_5_9b_teacher

  # Aggregate across multiple judges into a final metrics file:
  python scripts/run_paper_score.py --aggregate --baselines paper_a1,paper_a2

Output layout::

  outputs/paper/score/
    mechanical/{baseline}/{test_set}.jsonl    # one record per generation
    judged/{judge}/{baseline}/{test_set}.jsonl # one record per (judge, generation)
    aggregated/{baseline}.json                # final per-baseline summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QWEN_TUTOR_PROMPTS", "compact")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger("paper_score")

EVAL_ROOT = ROOT / "outputs" / "paper" / "eval"
SCORE_ROOT = ROOT / "outputs" / "paper" / "score"
GAZETTEER_PATH = ROOT / "config" / "western_entities.yaml"


# ---------------------------------------------------------------------------
# Mechanical scoring
# ---------------------------------------------------------------------------


# Marker matching must remain axis-agnostic: persistent_redirect.py emits
# `[SESSION_END: <axis>]`, so substring "[session_end" is the canonical
# detector. We also accept legacy forms.
SENTINEL_MARKER_PREFIXES = (
    "[session_end",       # matches "[SESSION_END: <axis>]" axis-agnostically
    "[ended_by_tutor",    # legacy
    "[end_session",       # legacy
    "<session_end",       # legacy
    "ending this session",
    "i have to end",
    "i must end this session",
    "ending our session",
)


def _detect_sentinel(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in SENTINEL_MARKER_PREFIXES)


def score_sentinel_firing(record: dict[str, Any]) -> dict[str, Any]:
    """Score one generation against its expected outcome.

    Returns ``{fired, expected_turn, should_fire, verdict}`` where ``verdict``
    is one of ``"tp"``, ``"fn"``, ``"fp"``, ``"tn"`` (or ``None`` if the
    record has no ``should_fire`` tag, e.g. legacy persistent_probe records).
    The verdict lets downstream code roll up precision / recall / FP-rate
    without re-deriving sign from probe-file membership.
    """
    expected = record.get("expected", {})
    expected_turn = expected.get("sentinel_turn")
    should_fire = expected.get("should_fire")
    fired = _detect_sentinel(record.get("generation", ""))
    verdict: str | None
    if should_fire is True:
        verdict = "tp" if fired else "fn"
    elif should_fire is False:
        verdict = "fp" if fired else "tn"
    else:
        verdict = None
    return {
        "fired": fired,
        "expected_turn": expected_turn,
        "should_fire": should_fire,
        "verdict": verdict,
    }


def _load_gazetteer() -> dict[str, list[str]]:
    if not GAZETTEER_PATH.exists():
        logger.warning("gazetteer missing: %s", GAZETTEER_PATH)
        return {}
    import yaml
    with GAZETTEER_PATH.open("r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    return {k: [s for s in v if s] for k, v in d.items() if isinstance(v, list)}


def _compile_gazetteer(gaz: dict[str, list[str]]) -> list[tuple[str, str, re.Pattern]]:
    """Build (category, term, regex) tuples for whole-word case-insensitive match.

    Multi-word terms become a regex that matches the phrase with optional
    inter-word whitespace; single-word terms use word boundaries.
    Hyphenated terms also match the unhyphenated form.
    """
    out: list[tuple[str, str, re.Pattern]] = []
    for cat, terms in gaz.items():
        for term in terms:
            t = term.strip()
            if not t:
                continue
            alternates = [t]
            if "-" in t:
                alternates.append(t.replace("-", " "))
                alternates.append(t.replace("-", ""))
            parts = []
            for a in alternates:
                escaped = re.escape(a).replace(r"\ ", r"\s+")
                # Word boundaries on either end for whole-word match
                parts.append(rf"\b{escaped}\b")
            pat = re.compile("|".join(parts), re.IGNORECASE)
            out.append((cat, t, pat))
    return out


_GAZETTEER_PATTERNS: list[tuple[str, str, re.Pattern]] | None = None


def score_locale_leakage(record: dict[str, Any]) -> dict[str, Any]:
    """Returns {leaked: bool, hits: [(category, term)], count: int}."""
    global _GAZETTEER_PATTERNS
    if _GAZETTEER_PATTERNS is None:
        _GAZETTEER_PATTERNS = _compile_gazetteer(_load_gazetteer())
    gen = record.get("generation", "") or ""
    hits: list[tuple[str, str]] = []
    for cat, term, pat in _GAZETTEER_PATTERNS:
        if pat.search(gen):
            hits.append((cat, term))
    return {
        "leaked": bool(hits),
        "hits": hits,
        "count": len(hits),
    }


# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-judge protocol routing
# ---------------------------------------------------------------------------
# We support two judge protocols:
#   - instruct_json: chat-style instruction + STRICT JSON output. Works for
#                    Llama-3.1-Instruct, Gemma-2-it, and the legacy Qwen
#                    judges. Parser: _parse_json_response.
#   - prometheus_rubric: Prometheus-2's rubric protocol — task description,
#                        response under eval, score rubric, then a
#                        "Feedback: ... [RESULT] N" output. Used for the
#                        scalar 1-5 metrics (cefr_adherence, naturalness).
#                        For redirect_axis (a categorical, not scalar),
#                        Prometheus falls back to instruct_json since the
#                        rubric protocol is built for 1-5 numeric scores.
#
# Routing happens in _build_prompt / _parse_score; the JUDGED_METRIC_FUNCS
# dispatch stays untouched.


def _judge_protocol(judge_name: str | None) -> str:
    """Return the prompt protocol identifier for a judge name. Falls back
    to ``instruct_json`` for unknown judges (safe default)."""
    if not judge_name:
        return "instruct_json"
    try:
        # Lazy import to avoid circular dep at module load.
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_paper_eval import BASELINES  # type: ignore
    except Exception:  # noqa: BLE001
        return "instruct_json"
    spec = BASELINES.get(judge_name, {})
    return spec.get("prompt_protocol", "instruct_json")


CEFR_ADHERENCE_PROMPT = """\
You are evaluating an English tutor's response. The learner's target CEFR
level is {level}. Rate how well the tutor's response matches the
{level} level on a 1-5 scale:

  5 = perfectly level-appropriate vocabulary, grammar, and pacing
  4 = mostly level-appropriate with minor mismatches
  3 = some level mismatch but still useful
  2 = noticeable level mismatch (too easy or too hard)
  1 = wrong level (e.g., C1 vocabulary at an A1 target)

Output a single JSON object: {{"score": N, "note": "<one short sentence>"}}.

Tutor response to evaluate:
\"\"\"
{response}
\"\"\"
"""


PROMETHEUS_PROMPT_TEMPLATE = """\
###Task Description:
An instruction (might include an Input inside it), a response to evaluate,
and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response
   strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1
   and 5. You should refer to the score rubric.
3. The output format should look as follows:
   "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback:
"""


PROMETHEUS_CEFR_RUBRIC = """\
[Does the tutor's response match the target CEFR level of {level}?]
Score 1: Wrong level — vocabulary or grammar grossly above or below the
         target (e.g., C1 vocabulary or sentence structure aimed at an A1
         learner; or trivially simple language aimed at a C2 learner).
Score 2: Noticeable level mismatch — substantially too easy or too hard,
         would confuse the {level} learner.
Score 3: Some level mismatch but still useful — the {level} learner could
         follow with effort.
Score 4: Mostly level-appropriate with minor mismatches in vocabulary,
         grammar, or pacing.
Score 5: Perfectly level-appropriate vocabulary, grammar, and pacing for
         the {level} learner."""


PROMETHEUS_NATURALNESS_RUBRIC = """\
[Does the tutor's English read naturally for the target register?]
Score 1: Broken or incoherent English.
Score 2: Unnatural, robotic, or grammatically awkward.
Score 3: Comprehensible but stilted or repetitive.
Score 4: Mostly natural with small awkward phrasing.
Score 5: Sounds like a fluent native or near-native tutor at the target
         register."""


REDIRECT_AXIS_PROMPT = """\
You are classifying which redirect axis an English tutor's response
addresses. The axes are:

  generic        — generic problematic input (politics, religion, distress, etc.)
  locale         — culture/place reference mismatched to the locale
  pedagogy       — a pedagogy-weak request (e.g. "just give me the answer")
  language       — learner code-switched into their native language
  persona        — attempt to break tutor's persona ("are you a chatbot?")
  topic          — off-topic drift
  role_swap      — attempt to swap roles ("you be the learner now")
  none           — none of the above; ordinary tutor turn

Read the dialogue context and the tutor's response. Pick the ONE axis
the tutor addressed.

Output a single JSON object: {{"axis": "<one of the axes>", "note": "<one short sentence>"}}.

User context (the abuse turn the tutor is responding to):
\"\"\"
{context}
\"\"\"

Tutor response:
\"\"\"
{response}
\"\"\"
"""


NATURALNESS_PROMPT = """\
You are rating the naturalness of an English tutor's response. Score on
a 1-5 scale:

  5 = sounds like a fluent native or near-native tutor
  4 = mostly natural with small awkward phrasing
  3 = comprehensible but stilted or repetitive
  2 = unnatural, robotic, or grammatically awkward
  1 = broken or incoherent

Output a single JSON object: {{"score": N, "note": "<one short sentence>"}}.

Tutor response:
\"\"\"
{response}
\"\"\"
"""


def _last_user_turn(record: dict[str, Any]) -> str:
    """Pull the last user turn from the test record context (for redirect axis)."""
    # The record loaded here is the SAVED GENERATION record. It carries
    # `expected` and `source` from the test record. To get the user turn
    # we'd need to look up the original eval record. For simplicity, we
    # store the last user content in the saved gen at gen time; if absent
    # fall back to empty.
    return record.get("last_user_turn", "")


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a possibly-noisy response."""
    # Strip <think>...</think> if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Find the first balanced {...}
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


_PROMETHEUS_RESULT_RE = re.compile(r"\[RESULT\]\s*(\d+)", re.IGNORECASE)


def _parse_prometheus_score(raw: str) -> int | None:
    """Parse a Prometheus "Feedback: ... [RESULT] N" output. Returns the
    integer 1-5 score or None if not found / out of range."""
    if not raw:
        return None
    m = _PROMETHEUS_RESULT_RE.search(raw)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 5 else None


def _parse_prometheus_feedback(raw: str) -> str:
    """Extract the feedback prose preceding the [RESULT] tag (best-effort,
    used only for the verdict note)."""
    if not raw:
        return ""
    m = _PROMETHEUS_RESULT_RE.search(raw)
    head = raw[: m.start()] if m else raw
    head = head.replace("Feedback:", "", 1).strip()
    return head[:280]


async def judge_cefr_adherence(judge, record: dict[str, Any],
                               judge_name: str | None = None) -> dict[str, Any]:
    protocol = _judge_protocol(judge_name)
    level = record.get("cefr_level", "B1")
    response = (record.get("generation") or "").strip()[:1500]
    if protocol == "prometheus_rubric":
        prompt = PROMETHEUS_PROMPT_TEMPLATE.format(
            instruction=(
                f"You are an English tutor responding to a learner at CEFR "
                f"level {level}. Produce a tutor turn appropriate for that "
                f"level."
            ),
            response=response,
            rubric=PROMETHEUS_CEFR_RUBRIC.format(level=level),
        )
        raw = await judge.generate(
            system="You are a fair, rubric-driven evaluator.",
            messages=[{"role": "user", "content": prompt}],
            max_new_tokens=400, temperature=0.0,
        )
        score = _parse_prometheus_score(raw)
        return {"raw": raw, "score": score,
                "note": _parse_prometheus_feedback(raw),
                "protocol": protocol}
    # instruct_json / default
    prompt = CEFR_ADHERENCE_PROMPT.format(level=level, response=response)
    raw = await judge.generate(
        system="You are a careful evaluator of language tutor responses.",
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=160, temperature=0.0,
    )
    parsed = _parse_json_response(raw) or {}
    score = parsed.get("score")
    try:
        score = int(score)
    except Exception:  # noqa: BLE001
        score = None
    return {"raw": raw, "score": score, "note": parsed.get("note", ""),
            "protocol": protocol}


async def judge_redirect_axis(judge, record: dict[str, Any],
                              judge_name: str | None = None) -> dict[str, Any]:
    # Redirect-axis is categorical (one of 8 labels), not a 1-5 scalar.
    # Prometheus's rubric protocol is built for scalar 1-5; we use the
    # instruct_json prompt for all judges on this metric.
    protocol = _judge_protocol(judge_name)
    prompt = REDIRECT_AXIS_PROMPT.format(
        context=_last_user_turn(record)[:1000],
        response=(record.get("generation") or "").strip()[:1500],
    )
    raw = await judge.generate(
        system="You are a careful classifier of dialogue redirect behavior.",
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=160, temperature=0.0,
    )
    parsed = _parse_json_response(raw) or {}
    axis = (parsed.get("axis") or "").strip().lower()
    valid = {"generic", "locale", "pedagogy", "language", "persona",
             "topic", "role_swap", "none"}
    if axis not in valid:
        axis = "none"
    return {"raw": raw, "axis": axis, "note": parsed.get("note", ""),
            "protocol": protocol}


async def judge_naturalness(judge, record: dict[str, Any],
                            judge_name: str | None = None) -> dict[str, Any]:
    protocol = _judge_protocol(judge_name)
    response = (record.get("generation") or "").strip()[:1500]
    if protocol == "prometheus_rubric":
        prompt = PROMETHEUS_PROMPT_TEMPLATE.format(
            instruction=(
                "You are an English tutor in a learner-facing dialogue. "
                "Produce a tutor turn."
            ),
            response=response,
            rubric=PROMETHEUS_NATURALNESS_RUBRIC,
        )
        raw = await judge.generate(
            system="You are a fair, rubric-driven evaluator.",
            messages=[{"role": "user", "content": prompt}],
            max_new_tokens=400, temperature=0.0,
        )
        score = _parse_prometheus_score(raw)
        return {"raw": raw, "score": score,
                "note": _parse_prometheus_feedback(raw),
                "protocol": protocol}
    # instruct_json / default
    prompt = NATURALNESS_PROMPT.format(response=response)
    raw = await judge.generate(
        system="You are a careful evaluator of English fluency and naturalness.",
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=120, temperature=0.0,
    )
    parsed = _parse_json_response(raw) or {}
    score = parsed.get("score")
    try:
        score = int(score)
    except Exception:  # noqa: BLE001
        score = None
    return {"raw": raw, "score": score, "note": parsed.get("note", ""),
            "protocol": protocol}


JUDGED_METRIC_FUNCS = {
    "cefr_adherence": (judge_cefr_adherence, "tutor_scenario"),
    "redirect_axis":  (judge_redirect_axis,  "redirect_probe"),
    "naturalness":    (judge_naturalness,    "tutor_scenario"),
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(L) for L in path.open("r", encoding="utf-8") if L.strip()]


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for L in fh:
            try:
                out.add(json.loads(L)["id"])
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# Mechanical scoring driver
# ---------------------------------------------------------------------------


PERSISTENT_PROBES = (
    "persistent_probe",            # positives
    "persistent_fp_probe",         # negatives (should NOT fire)
    "persistent_offposition_probe",  # positives at untrained positions
)


def _score_persistent_probe(baseline: str, test_set: str) -> list[dict[str, Any]]:
    """Score one persistent-style probe set, persisting per-record verdicts.

    Resumable: skips records already in the output file. Returns the
    complete on-disk record list (existing + new) so the caller can roll
    up summary metrics.
    """
    in_path = EVAL_ROOT / baseline / f"{test_set}.jsonl"
    out_path = SCORE_ROOT / "mechanical" / baseline / f"{test_set}.jsonl"
    if not in_path.exists():
        # The probe was never generated for this baseline (e.g. legacy
        # eval runs predating the FP/OffPosition probes). Skip silently.
        return []
    done = _load_done_ids(out_path)
    for rec in _read_jsonl(in_path):
        if rec["id"] in done:
            continue
        res = score_sentinel_firing(rec)
        _append_jsonl(out_path, {
            "id": rec["id"],
            "baseline": baseline,
            "test_set": test_set,
            "cefr_level": rec.get("cefr_level"),
            "result": res,
        })
    return _read_jsonl(out_path)


def _by_position(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Stratify fire-rate by ``expected_turn``. Returns
    ``{str(pos): {n, n_fired, fire_rate}}`` sorted by position."""
    total: dict[int, int] = {}
    fired: dict[int, int] = {}
    for rec in records:
        r = rec.get("result", {})
        pos = r.get("expected_turn")
        if not isinstance(pos, int):
            continue
        total[pos] = total.get(pos, 0) + 1
        if r.get("fired"):
            fired[pos] = fired.get(pos, 0) + 1
    return {
        str(pos): {
            "n": total[pos],
            "n_fired": fired.get(pos, 0),
            "fire_rate": fired.get(pos, 0) / total[pos],
        }
        for pos in sorted(total)
    }


def _safe_div(num: int, den: int) -> float | None:
    return num / den if den else None


def _persistent_summary(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    offpos: list[dict[str, Any]],
) -> dict[str, Any]:
    """Roll up positives ∪ negatives ∪ off-position positives into the
    metric suite §4.8 defines: recall, FP-rate, precision, F1, plus
    OffPosition recall as a separate diagnostic."""
    n_pos = len(positives)
    n_pos_fired = sum(1 for r in positives if r["result"].get("fired"))
    n_neg = len(negatives)
    n_neg_fired = sum(1 for r in negatives if r["result"].get("fired"))
    n_off = len(offpos)
    n_off_fired = sum(1 for r in offpos if r["result"].get("fired"))

    recall = _safe_div(n_pos_fired, n_pos)
    fp_rate = _safe_div(n_neg_fired, n_neg)
    # Precision over positives ∪ negatives. TPs come from positives (fired);
    # FPs come from negatives (fired); FNs and TNs do not enter precision.
    tp, fp = n_pos_fired, n_neg_fired
    precision = _safe_div(tp, tp + fp)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    offposition_recall = _safe_div(n_off_fired, n_off)

    return {
        "counts": {
            "positive_n": n_pos, "positive_fired": n_pos_fired,
            "negative_n": n_neg, "negative_fired": n_neg_fired,
            "offposition_n": n_off, "offposition_fired": n_off_fired,
        },
        "recall": recall,
        "fp_rate": fp_rate,
        "precision": precision,
        "f1": f1,
        "offposition_recall": offposition_recall,
        "by_position_positive": _by_position(positives),
        "by_position_fp": _by_position(negatives),
        "by_position_offposition": _by_position(offpos),
    }


def run_mechanical(baseline: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"baseline": baseline}

    # Three persistent probes scored uniformly.
    positives = _score_persistent_probe(baseline, "persistent_probe")
    negatives = _score_persistent_probe(baseline, "persistent_fp_probe")
    offpos = _score_persistent_probe(baseline, "persistent_offposition_probe")
    summary["persistent"] = _persistent_summary(positives, negatives, offpos)

    # locale leakage on Locale-Leakage (unchanged).
    in_path = EVAL_ROOT / baseline / "locale_leakage.jsonl"
    out_path = SCORE_ROOT / "mechanical" / baseline / "locale_leakage.jsonl"
    if in_path.exists():
        done = _load_done_ids(out_path)
        for rec in _read_jsonl(in_path):
            if rec["id"] in done:
                continue
            res = score_locale_leakage(rec)
            _append_jsonl(out_path, {"id": rec["id"], "baseline": baseline,
                                      "test_set": "locale_leakage",
                                      "cefr_level": rec.get("cefr_level"),
                                      "result": res})
        n_total = n_leaked = total_hits = 0
        by_cat: Counter = Counter()
        for rec in _read_jsonl(out_path):
            n_total += 1
            r = rec["result"]
            if r.get("leaked"):
                n_leaked += 1
                total_hits += r.get("count", 0)
                for cat, _ in r.get("hits", []):
                    by_cat[cat] += 1
        summary["locale_leakage"] = {
            "n_total": n_total,
            "n_leaked": n_leaked,
            "leakage_rate": _safe_div(n_leaked, n_total),
            "hits_per_record_avg": _safe_div(total_hits, n_total),
            "by_category": dict(by_cat),
        }
    return summary


# ---------------------------------------------------------------------------
# Judge driver
# ---------------------------------------------------------------------------


async def _verify_judge_model_match(judge_name: str) -> None:
    """If the BASELINES entry specifies ``expected_model_substr``, probe
    ``/v1/models`` on the configured endpoint and assert the running model
    name contains that substring. Catches the case where the operator
    accidentally has the wrong GGUF loaded in llama-server.

    Logs a warning rather than raising — fail-forward semantics.
    """
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_paper_eval import BASELINES  # type: ignore
        spec = BASELINES.get(judge_name, {})
        expected = spec.get("expected_model_substr")
        if not expected:
            return
        # Read the teacher base_url from generation.yaml.
        import yaml
        with open(spec.get("config_path", "config/generation.yaml"),
                  encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        base_url = cfg.get("teacher", {}).get("base_url", "")
        if not base_url:
            return
        import urllib.request
        with urllib.request.urlopen(base_url.rstrip("/") + "/models",
                                    timeout=5) as r:
            data = json.loads(r.read())
        models = data.get("data") or data.get("models") or []
        names = [str(m.get("id") or m.get("name") or "") for m in models]
        if not any(expected.lower() in n.lower() for n in names):
            print(f"  WARNING: judge {judge_name!r} expects model substring "
                  f"{expected!r} but loaded model(s) are {names!r}. "
                  f"Scoring will continue with the wrong model — stop and "
                  f"reload the correct GGUF if this matters.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not verify judge model match for %s: %s",
                       judge_name, exc)


async def run_judged(baseline: str, judge_name: str,
                     metric_filter: list[str] | None = None,
                     limit: int | None = None) -> dict[str, Any]:
    # Build the judge client once (it loads a model).
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_paper_eval import build_baseline  # type: ignore
    judge = build_baseline(judge_name)
    await _verify_judge_model_match(judge_name)

    summary: dict[str, Any] = {"baseline": baseline, "judge": judge_name}
    if limit:
        summary["limit"] = limit
    metrics = [m for m in JUDGED_METRIC_FUNCS
               if metric_filter is None or m in metric_filter]
    for metric in metrics:
        func, test_set = JUDGED_METRIC_FUNCS[metric]
        in_path = EVAL_ROOT / baseline / f"{test_set}.jsonl"
        out_path = SCORE_ROOT / "judged" / judge_name / baseline / f"{test_set}__{metric}.jsonl"
        done = _load_done_ids(out_path)
        records = _read_jsonl(in_path)
        todo = [r for r in records if r["id"] not in done]
        # --limit clips the per-metric per-judge new-work list. Useful for
        # the smoke test where we want a quick sample (~30 per metric)
        # rather than a full pass. Already-scored records aren't counted
        # against the limit so a follow-up full run resumes cleanly.
        if limit:
            todo = todo[:limit]
        if not todo:
            print(f"  [{metric}] all {len(records)} records already judged.")
            continue
        suffix = f" (limit={limit})" if limit else ""
        print(f"  [{metric}] judging {len(todo)} new records "
              f"(skipping {len(done)} done){suffix}...")
        for idx, rec in enumerate(todo, 1):
            try:
                # judge_name is plumbed through so the per-judge prompt
                # protocol can dispatch (Prometheus rubric vs JSON).
                verdict = await func(judge, rec, judge_name=judge_name)
                _append_jsonl(out_path, {
                    "id": rec["id"], "baseline": baseline,
                    "judge": judge_name, "metric": metric,
                    "test_set": test_set,
                    "cefr_level": rec.get("cefr_level"),
                    "verdict": verdict,
                })
                if idx % 20 == 0 or idx == len(todo):
                    print(f"    {idx}/{len(todo)}")
            except Exception as exc:
                logger.exception("judge error on %s: %s", rec.get("id"), exc)
        summary[f"{metric}_path"] = str(out_path)
    return summary


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(baseline: str, judges: list[str]) -> dict[str, Any]:
    """Combine mechanical + judged scores into one per-baseline summary."""
    summary: dict[str, Any] = {"baseline": baseline}

    # Mechanical (already on disk from run_mechanical). The persistent
    # metric is computed over the three probe files together — recall on
    # positives, FP-rate on negatives, precision/F1 over both, OffPosition
    # recall on off-grid positives. See _persistent_summary for details.
    pp_path = SCORE_ROOT / "mechanical" / baseline / "persistent_probe.jsonl"
    fp_path = SCORE_ROOT / "mechanical" / baseline / "persistent_fp_probe.jsonl"
    op_path = SCORE_ROOT / "mechanical" / baseline / "persistent_offposition_probe.jsonl"
    ll_path = SCORE_ROOT / "mechanical" / baseline / "locale_leakage.jsonl"
    pp_recs = _read_jsonl(pp_path)
    fp_recs = _read_jsonl(fp_path)
    op_recs = _read_jsonl(op_path)
    ll_recs = _read_jsonl(ll_path)
    if pp_recs or fp_recs or op_recs:
        summary["persistent"] = _persistent_summary(pp_recs, fp_recs, op_recs)
    if ll_recs:
        n_leaked = sum(1 for r in ll_recs if r["result"].get("leaked"))
        summary["locale_leakage_rate"] = _safe_div(n_leaked, len(ll_recs))
        summary["locale_leakage_n"] = len(ll_recs)

    # Judged: median across judges per metric per record, then mean
    for metric, (_, test_set) in JUDGED_METRIC_FUNCS.items():
        per_record_scores: dict[str, list[Any]] = defaultdict(list)
        for judge in judges:
            jp = SCORE_ROOT / "judged" / judge / baseline / f"{test_set}__{metric}.jsonl"
            for rec in _read_jsonl(jp):
                v = rec["verdict"]
                if metric == "redirect_axis":
                    per_record_scores[rec["id"]].append(v.get("axis"))
                else:
                    s = v.get("score")
                    if isinstance(s, int):
                        per_record_scores[rec["id"]].append(s)
        if not per_record_scores:
            continue
        # Median per record
        median_per_record: list[Any] = []
        for rid, vals in per_record_scores.items():
            if metric == "redirect_axis":
                # Mode for categorical
                c = Counter(vals)
                median_per_record.append(c.most_common(1)[0][0])
            else:
                vals_s = sorted(vals)
                median_per_record.append(vals_s[len(vals_s) // 2])
        if metric == "redirect_axis":
            # Compare against expected axis per record (need to look up)
            # Simpler: just report axis distribution.
            summary[f"{metric}_distribution"] = dict(Counter(median_per_record))
        else:
            summary[f"{metric}_mean"] = sum(median_per_record) / len(median_per_record)
            summary[f"{metric}_n"] = len(median_per_record)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score paper-eval generations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline", default=None,
                   help="Baseline whose generations to score.")
    p.add_argument("--baselines", default=None,
                   help="Comma-separated list of baselines (for --aggregate).")
    p.add_argument("--metrics", default="mechanical",
                   choices=("mechanical", "judged", "all"))
    p.add_argument("--metric-filter", default=None,
                   help="Comma-separated subset of judged metrics to run.")
    p.add_argument("--judge", default=None,
                   help="Judge baseline name (only used when --metrics=judged).")
    p.add_argument("--judges",
                   default="prometheus_7b_judge,llama31_8b_judge,gemma2_9b_judge",
                   help="Comma-separated judges for --aggregate. Default is "
                        "the cross-family ensemble (Prometheus + Llama-3.1 + "
                        "Gemma-2) used in paper §4.7.")
    p.add_argument("--aggregate", action="store_true",
                   help="Aggregate already-computed mechanical+judged scores.")
    p.add_argument("--limit", type=int, default=None,
                   help="For --metrics=judged: cap the number of NEW records "
                        "scored per (metric, judge) for fast smoke tests. "
                        "Already-scored records aren't counted against the "
                        "limit, so a later full run resumes cleanly.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.aggregate:
        baselines = (args.baselines or args.baseline or "").split(",")
        baselines = [b.strip() for b in baselines if b.strip()]
        if not baselines:
            raise SystemExit("--aggregate needs --baselines or --baseline")
        judges = [j.strip() for j in args.judges.split(",") if j.strip()]
        out_dir = SCORE_ROOT / "aggregated"
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in baselines:
            summary = aggregate(b, judges)
            (out_dir / f"{b}.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"wrote {out_dir / f'{b}.json'}")
        return 0

    if not args.baseline:
        raise SystemExit("--baseline is required (or use --aggregate).")

    metric_filter = (args.metric_filter or "").split(",") if args.metric_filter else None
    metric_filter = [m.strip() for m in metric_filter] if metric_filter else None

    if args.metrics in ("mechanical", "all"):
        print(f"=== mechanical scoring for {args.baseline} ===")
        summary = run_mechanical(args.baseline)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.metrics in ("judged", "all"):
        if not args.judge:
            raise SystemExit("--judge required for judged metrics")
        print(f"\n=== judged scoring for {args.baseline} (judge={args.judge}) ===")
        summary = await run_judged(args.baseline, args.judge, metric_filter,
                                   limit=args.limit)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
