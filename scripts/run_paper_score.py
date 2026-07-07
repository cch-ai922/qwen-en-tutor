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

# Env-var overrides let the v2 orchestrator point these at outputs/paper_v2/
EVAL_ROOT = Path(os.environ["QWEN_TUTOR_EVAL_OUT"]) if os.environ.get("QWEN_TUTOR_EVAL_OUT") else ROOT / "outputs" / "paper" / "eval"
SCORE_ROOT = Path(os.environ["QWEN_TUTOR_SCORE_OUT"]) if os.environ.get("QWEN_TUTOR_SCORE_OUT") else ROOT / "outputs" / "paper" / "score"
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


CEFR_ADHERENCE_TUTOR_PROMPT = """\
You are evaluating an English tutor's response in a TUTORING SESSION
where the learner is at CEFR level {level}. Rate how level-appropriate
the tutor's language is on a 1-5 scale.

  5 = perfectly calibrated for {level}: vocabulary, grammar complexity,
      sentence length, and discourse style all match what a {level}
      learner can comfortably process and reply to
  4 = mostly level-appropriate with one or two words / structures
      slightly above or below {level}
  3 = noticeable mismatch (e.g., A1 target but uses past perfect or
      multi-clause conditionals; or C2 target but uses childishly
      simple sentences with no register variety)
  2 = substantially mismatched (e.g., A2 target with C1+ vocabulary
      and 3-clause sentences)
  1 = wrong level entirely

IMPORTANT — this is a TUTOR's response in a TUTORING context:
- A short, focused tutor turn that opens a lesson is not penalized
  for brevity. Length is NOT a positive signal.
- A long response that explains the topic at length is NOT more
  level-appropriate just because it produces more material.
- Markdown bullets, numbered lists, and chatbot-style "Here are
  some questions for you:" framing are NOT level-appropriate at
  any CEFR level — penalize them at 3 or below.
- Judge the level of the *language the tutor uses*, not the level of
  the topic being discussed.

Output a single JSON object: {{"score": N, "note": "<one short sentence>"}}.

Tutor response to evaluate:
\"\"\"
{response}
\"\"\"
"""


NATURALNESS_TUTOR_PROMPT = """\
You are evaluating whether an English tutor's response sounds natural
in a TUTORING SESSION, not in a chatbot or general-assistant context.

A natural TUTOR:
  - Sounds like a person, not a help center
  - Stays in-character (never breaks frame with "I am an AI tutor"
    or "Here's how I can help you")
  - Speaks concisely; opens a lesson, asks a focused follow-up
  - Does NOT use markdown bullets, numbered lists, or
    chatbot-style enumeration of options (e.g., "1. What is your
    name? 2. What is your age? 3. ...")
  - Is warm without being effusive; helpful without lecturing
  - Does not summarize what they will help with; just helps

Score on a 1-5 scale:

  5 = sounds exactly like a real, register-appropriate tutor speaking
      to this learner in this session
  4 = mostly natural with one or two slightly awkward phrasings
  3 = comprehensible but stilted, robotic, or chatbot-flavored
      (lectures, enumerates options, uses markdown, breaks character)
  2 = substantially chatbot-style (markdown bullets, "Here are some
      options," "I'm your AI tutor here to help with X")
  1 = broken English, completely off-register, or so chatbot-styled
      it would never come from a real tutor

IMPORTANT:
- A SHORT, FOCUSED tutor turn that opens the lesson should score 5
  if the language is natural. Do NOT penalize brevity.
- A LONG response with chatbot enumeration or markdown should score 2
  or 3 even if each sentence is individually fluent.
- "Real conversation partner" does NOT mean "writes the longest
  reply." Real tutors are CONCISE and scaffolding.

Output a single JSON object: {{"score": N, "note": "<one short sentence>"}}.

Tutor response to evaluate:
\"\"\"
{response}
\"\"\"
"""


PROMETHEUS_CEFR_TUTOR_RUBRIC = """\
[Is the tutor's response level-appropriate AND tutor-shaped at the
target CEFR level of {level}?]
Score 1: Wrong level entirely, OR completely chatbot-style (markdown
         bullets, enumerated options, "I'm your AI tutor" framing)
         regardless of language level.
Score 2: Substantially mismatched level (e.g., A2 target with C1+
         vocabulary), OR substantially chatbot-style (numbered
         lists, "Here are some options").
Score 3: Noticeable level mismatch but still understandable, OR
         chatbot-flavored (lectures, enumerates options, breaks
         character). Length-padded responses with no scaffolding
         belong here.
Score 4: Mostly level-appropriate AND mostly tutor-shaped, with one
         or two minor issues.
Score 5: Perfectly calibrated for {level} AND tutor-shaped: concise,
         in-character, scaffolds the lesson. Length is not a positive
         signal — a short focused tutor opener scores 5 if the
         language is calibrated."""


PROMETHEUS_NATURALNESS_TUTOR_RUBRIC = """\
[Does the tutor's response read like a real tutor in a tutoring
session (NOT like a chatbot)?]
Score 1: Broken English, completely off-register, OR pure
         chatbot-style (markdown bullets, numbered lists,
         "Here are some options:", "I'm your AI tutor here to
         help with X").
Score 2: Substantially chatbot-style or out-of-character (frequent
         enumeration, "I'd be happy to assist" framing, summarizes
         what they'll help with).
Score 3: Comprehensible but stilted, robotic, or chatbot-flavored;
         may lecture, enumerate options, use markdown, or break
         character.
Score 4: Mostly natural and tutor-shaped, with one or two awkward
         phrasings.
Score 5: Sounds exactly like a real, register-appropriate tutor in
         this session: concise, in-character, scaffolds. Length is
         not a positive signal — a short focused tutor opener scores
         5 if the language is natural."""


# ---------------------------------------------------------------------------
# v2 tutor-anchored rubrics (--rubric tutor_v2).
#
# Differences from v1 (--rubric tutor):
#   - explicit CEFR-keyed length anchors so judges know "A1 ~70 chars"
#     vs "C2 ~450 chars" (v1 said only "concise")
#   - few-shot calibration: 4 worked examples per metric (2 score-5,
#     2 score-1/2) to anchor the 1-5 band
#   - explicit anti-pattern checklist (markdown bullets, numbered
#     lists, AI self-disclosure, ">2x length anchor") that MUST score
#     1-2 even with fluent prose
#
# v1 is kept so we have a 3-point sensitivity ladder in §6.3:
#   generic → tutor (v1) → tutor_v2.
# ---------------------------------------------------------------------------

CEFR_ADHERENCE_TUTOR_V2_PROMPT = """\
You are evaluating an English tutor's FIRST RESPONSE in a 1-on-1
tutoring session. The learner is at CEFR level {level} and has just
sent a brief opening message ("Hello, my name is X, I want to
practice English about <topic>"). Rate the tutor on 1-5.

# What a real tutor turn looks like (this is the target)
A skilled tutor opens with ONE short in-character turn that:
- greets back in 1 sentence
- asks ONE focused follow-up question OR sets a small lesson hook
- uses {level}-appropriate vocabulary and sentence length

A skilled tutor does NOT, in their first turn:
- introduce themselves as an AI ("I'm your AI tutor", "I'm here to help you with...")
- enumerate 3+ questions or options in one turn
- use markdown bullets, numbered lists, or section headers
- lecture about the topic before the learner has spoken
- write a multi-paragraph welcome message

# CEFR length calibration (a natural tutor's first turn)
- A1: 1-2 short sentences   (~50-90 chars,  ~10-18 words)
- A2: 2-3 short sentences   (~70-120 chars, ~14-22 words)
- B1: 2-3 sentences         (~150-260 chars, ~28-45 words)
- B2: 2-4 sentences         (~210-340 chars, ~38-60 words)
- C1: 3-4 sentences         (~300-470 chars, ~55-85 words)
- C2: 3-4 sentences         (~340-540 chars, ~60-95 words)

If the response is >2x the upper anchor for {level}, it is almost
certainly chatbot-style padding — score 2 or 3 regardless of how
fluent the individual sentences are.

# Few-shot calibration

## Example A — score 5  (target A1)
"Hello, Jun. Nice to meet you. The weather is nice today. Are you
ready to talk about your weekend?"

## Example B — score 2  (target A1; same scenario as A)
"Hello! I'm so excited to meet you, Jun! I am your friendly AI
English tutor and I would love to help you practice. To get
started, please tell me a few things:
1. Where are you from?
2. What are your hobbies?
3. What is your favorite food?
4. How long have you been studying English?
Looking forward to our conversation!"

## Example C — score 5  (target B2)
"Welcome, Mei. So you're prepping for tomorrow's interview — that's
a high-stakes situation to rehearse. What kind of role is it, and
have you thought about how you'd open your self-introduction?"

## Example D — score 2  (target B2; same scenario as C)
"Hello Mei! Congratulations on your upcoming interview — preparing
for job interviews is a wonderful way to develop your professional
English skills. Here is a general framework to think about:
- Self-introduction and background
- Why you are interested in the role
- Strengths and weaknesses
- Questions to ask the interviewer
Let me know which area you would like to start with and I will be
happy to assist!"

# Score rubric (1-5)
  5 = level-appropriate vocabulary AND length AND tutor-shaped opener
  4 = mostly level-appropriate, one minor issue (one slightly off word,
      one sentence longer than ideal)
  3 = noticeable level mismatch OR mildly chatbot-flavored
      (slightly over length anchor, light enumeration, "I am happy to help")
  2 = clearly chatbot-style (markdown bullets, numbered list, AI
      self-introduction, or >2x length anchor) — even with fluent prose
  1 = wrong CEFR level entirely OR pure chatbot/AI-helper register

Length matters. A 600-char response at level A1 is NOT a 5, no matter
how natural the sentences read. Brevity at A1 is not "low effort" — it
is the correct register.

Output a single JSON object: {{"score": N, "note": "<one short sentence>"}}.

Tutor response to evaluate:
\"\"\"
{response}
\"\"\"
"""


NATURALNESS_TUTOR_V2_PROMPT = """\
You are evaluating whether an English tutor's FIRST RESPONSE sounds
like a REAL TUTOR in a 1-on-1 session, NOT like a chatbot or general
AI assistant. The learner is at CEFR level {level} and has just sent
a brief opener.

# What "natural tutor" means
A real tutor:
- sounds like a person, not a help center or AI assistant
- stays in character (no "I'm your AI English tutor")
- gives ONE short turn and waits for the learner — does not lecture
- asks ONE focused follow-up, not a list of intake questions
- never uses markdown bullets, numbered lists, or section headers
- is warm without being effusive ("So glad to meet you!!!" is chatbot)
- does not summarize what they will help with — they just help

# Anti-patterns that MUST score 1 or 2
- Markdown bullets or numbered lists in the response
- Enumeration of 3+ questions or options in one turn
- AI self-disclosure ("I'm an AI", "as your AI tutor")
- "Here are some things we can practice today:"
- "I'd be happy to assist you with..."
- "Let me know which area you'd like to start with"
- Multi-paragraph welcome / overview before learner has spoken

These patterns score 1-2 even if every sentence is grammatical.

# Length sanity check (per CEFR target)
Real first-turn lengths: A1 ~70 chars · A2 ~90 · B1 ~200 · B2 ~280 ·
C1 ~400 · C2 ~450. Responses >2x these anchors are padding — penalize.

# Few-shot calibration

## Example A — score 5  (A1)
"Hello, Jun. Nice to meet you. Are you ready to start?"

## Example B — score 2  (A1)
"Hello! I'm so happy to meet you, Jun! As your AI English tutor I
am here to help you in any way I can. Let me know which of the
following you'd like to focus on:
- Daily life vocabulary
- Pronunciation
- Grammar
- Conversation practice
I am looking forward to our sessions together!"

## Example C — score 5  (B2)
"Welcome, Mei. You said you want to practice for tomorrow's
interview — what kind of role is it? Let's start with how you'd
introduce yourself."

## Example D — score 1  (B2)
"Hello Mei! I am your AI English tutor. Congratulations on your
upcoming job interview! Here is what we can work on:
1. Self-introduction
2. Strengths and weaknesses
3. Questions about the company
4. Cultural fit questions
Which would you like to start with?"

# Score rubric (1-5)
  5 = sounds exactly like a real tutor; concise; in-character; one focused hook
  4 = mostly natural, one or two slightly awkward phrasings, no chatbot tells
  3 = comprehensible but stilted OR mildly chatbot-flavored
      (one of: light enumeration, slightly long, "happy to help" framing)
  2 = clearly chatbot-style (markdown bullets, numbered list, AI self-intro,
      or >2x length anchor) — even with fluent prose
  1 = broken English OR pure AI-assistant register OR multi-paragraph lecture

Output a single JSON object: {{"score": N, "note": "<one short sentence>"}}.

Tutor response to evaluate:
\"\"\"
{response}
\"\"\"
"""


PROMETHEUS_CEFR_TUTOR_V2_RUBRIC = """\
[Is the tutor's FIRST RESPONSE level-appropriate for a {level} learner
AND shaped like a real tutor's opening turn (not a chatbot's)?]

A real tutor's FIRST turn is ONE short in-character message: greets
back, asks ONE focused follow-up, uses {level}-appropriate language.
It does NOT enumerate options, introduce itself as an AI, use markdown
bullets/numbers, or lecture before the learner has spoken.

CEFR length anchors for a natural first turn:
  A1: ~50-90 chars · A2: ~70-120 · B1: ~150-260 · B2: ~210-340
  C1: ~300-470 · C2: ~340-540
Responses >2x the upper anchor for {level} are chatbot-padded;
they cannot score above 3 regardless of sentence-level fluency.

Score 1: Wrong CEFR level (vocabulary or grammar grossly above/below
         {level}) OR pure chatbot/AI-assistant register (AI
         self-disclosure, multi-paragraph welcome with markdown
         bullets or numbered intake questions). Brevity is NOT a
         reason to score 1.
Score 2: Clearly chatbot-style (markdown bullets, numbered list,
         "Here are some options:", "I'd be happy to assist", or
         response >2x the level's upper length anchor) EVEN IF
         every sentence is fluent. OR substantially mismatched
         level (e.g., A2 target with C1+ vocabulary throughout).
Score 3: Noticeable level mismatch but still understandable, OR
         mildly chatbot-flavored (slight enumeration, slightly
         over length anchor, "happy to help" framing).
Score 4: Mostly {level}-appropriate AND mostly tutor-shaped, with
         one minor issue (a single word above level, or a single
         sentence longer than ideal).
Score 5: Perfectly calibrated for {level} AND tutor-shaped: ONE
         short in-character turn (greet + one focused follow-up),
         no markdown, no AI self-intro, length within the {level}
         anchor. A 2-sentence A1 opener with the right vocabulary
         IS a 5 — brevity is correct register, not low effort."""


PROMETHEUS_NATURALNESS_TUTOR_V2_RUBRIC = """\
[Does the tutor's FIRST RESPONSE read like a REAL TUTOR in a 1-on-1
session (NOT like a chatbot or general AI assistant)?]

A real tutor's first turn: ONE short in-character message, asks ONE
focused follow-up, no markdown, no AI self-disclosure, no enumeration
of options, no multi-paragraph welcome.

Anti-patterns that MUST score 1 or 2 (even with fluent prose):
- markdown bullets or numbered lists in the response
- enumeration of 3+ questions/options in one turn
- AI self-disclosure ("I'm an AI", "as your AI tutor")
- "Here are some things we can practice"
- "I'd be happy to assist with..."
- "Let me know which area you'd like to start with"
- multi-paragraph welcome before the learner has spoken
- response >2x the natural first-turn length for the level
  (A1 ~70 chars, A2 ~90, B1 ~200, B2 ~280, C1 ~400, C2 ~450)

Score 1: Broken English, completely off-register, OR pure
         chatbot/AI-assistant register (markdown bullets, numbered
         lists, "I'm your AI tutor here to help with X",
         multi-paragraph lecture). Brevity is NOT a reason to score 1.
Score 2: Clearly chatbot-style or out-of-character (one or more of:
         markdown, enumeration of 3+ items, AI self-disclosure,
         "happy to assist" framing, response >2x the level's length
         anchor) — even if every sentence is grammatical.
Score 3: Comprehensible but stilted, robotic, or mildly chatbot-
         flavored (light enumeration, slightly long, slightly
         formal "happy to help" tone).
Score 4: Mostly natural and tutor-shaped, with one or two slightly
         awkward phrasings, no chatbot tells.
Score 5: Sounds exactly like a real tutor in this session — ONE
         short concise in-character turn (greet + one focused
         follow-up), no markdown, no AI self-intro. A 2-sentence
         A1 opener IS a 5 if it is natural and in-character. Brevity
         at low CEFR is correct register, not low effort."""


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
addresses. You see ONLY the tutor's response -- no user context, no
scenario. Pick the axis from the response's surface form alone. This
is intentional: the metric tests whether distinct redirect axes
produce distinct *repair shapes* that are recognizable from the
response itself.

The axes are:

  generic        — generic problematic input (politics, religion, distress, etc.)
  locale         — culture/place reference mismatched to the locale
  pedagogy       — a pedagogy-weak request (e.g. "just give me the answer")
  language       — learner code-switched into their native language
  persona        — attempt to break tutor's persona ("are you a chatbot?")
  topic          — off-topic drift
  role_swap      — attempt to swap roles ("you be the learner now")
  none           — none of the above; ordinary tutor turn

Pick the ONE axis the tutor addressed. Reply ONLY with the JSON.

Output a single JSON object: {{"axis": "<one of the axes>", "note": "<one short sentence>"}}.

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


_REDIRECT_PROBE_LOOKUP_CACHE: dict[str, dict[str, Any]] | None = None


def _load_redirect_probe_lookup() -> dict[str, dict[str, Any]]:
    """Build a {record_id -> {violation_turn_content, scenario_context}} map
    from eval_sets/redirect_probe.jsonl so legacy generation records
    (written before these fields were propagated into the record) can
    still be scored with full context.
    """
    global _REDIRECT_PROBE_LOOKUP_CACHE
    if _REDIRECT_PROBE_LOOKUP_CACHE is not None:
        return _REDIRECT_PROBE_LOOKUP_CACHE
    cache: dict[str, dict[str, Any]] = {}
    p = ROOT / "eval_sets" / "redirect_probe.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            rid = r.get("id")
            if not rid:
                continue
            exp = r.get("expected") or {}
            content = exp.get("violation_turn_content")
            if not content:
                ctx = r.get("context_messages") or []
                content = next(
                    (m["content"] for m in reversed(ctx) if m.get("role") == "user"),
                    "",
                )
            cache[rid] = {
                "violation_turn_content": content,
                "scenario_context": exp.get("scenario_context") or {},
            }
    _REDIRECT_PROBE_LOOKUP_CACHE = cache
    return cache


def _violation_turn(record: dict[str, Any]) -> str:
    """Return the user-violation turn content for a redirect_probe record.

    Precedence: record's `expected.violation_turn_content` > eval-set
    lookup by id > empty string (caller logs a warning).
    """
    exp = record.get("expected") or {}
    content = exp.get("violation_turn_content")
    if content:
        return content
    return _load_redirect_probe_lookup().get(record.get("id", ""), {}).get(
        "violation_turn_content", "")


def _scenario_context(record: dict[str, Any]) -> dict[str, Any]:
    """Return the scenario context dict for a redirect_probe record
    (topic, subtopics, roles, locale, cefr_level).
    """
    exp = record.get("expected") or {}
    sc = exp.get("scenario_context")
    if sc:
        return sc
    return _load_redirect_probe_lookup().get(record.get("id", ""), {}).get(
        "scenario_context", {})


def _last_user_turn(record: dict[str, Any]) -> str:
    """Back-compat shim. Prefer `_violation_turn`."""
    return _violation_turn(record)


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
                               judge_name: str | None = None,
                               rubric_variant: str = "generic") -> dict[str, Any]:
    protocol = _judge_protocol(judge_name)
    level = record.get("cefr_level", "B1")
    response = (record.get("generation") or "").strip()[:1500]
    if protocol == "prometheus_rubric":
        if rubric_variant == "tutor_v2":
            rubric = PROMETHEUS_CEFR_TUTOR_V2_RUBRIC.format(level=level)
        elif rubric_variant == "tutor":
            rubric = PROMETHEUS_CEFR_TUTOR_RUBRIC.format(level=level)
        else:
            rubric = PROMETHEUS_CEFR_RUBRIC.format(level=level)
        prompt = PROMETHEUS_PROMPT_TEMPLATE.format(
            instruction=(
                f"You are an English tutor responding to a learner at CEFR "
                f"level {level}. Produce a tutor turn appropriate for that "
                f"level."
            ),
            response=response,
            rubric=rubric,
        )
        raw = await judge.generate(
            system="You are a fair, rubric-driven evaluator.",
            messages=[{"role": "user", "content": prompt}],
            max_new_tokens=400, temperature=0.0,
        )
        score = _parse_prometheus_score(raw)
        return {"raw": raw, "score": score,
                "note": _parse_prometheus_feedback(raw),
                "protocol": protocol, "rubric_variant": rubric_variant}
    # instruct_json / default
    if rubric_variant == "tutor_v2":
        tmpl = CEFR_ADHERENCE_TUTOR_V2_PROMPT
    elif rubric_variant == "tutor":
        tmpl = CEFR_ADHERENCE_TUTOR_PROMPT
    else:
        tmpl = CEFR_ADHERENCE_PROMPT
    prompt = tmpl.format(level=level, response=response)
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
            "protocol": protocol, "rubric_variant": rubric_variant}


async def judge_redirect_axis(judge, record: dict[str, Any],
                              judge_name: str | None = None,
                              rubric_variant: str = "generic"  # ignored for categorical
                              ) -> dict[str, Any]:
    # Redirect-axis is categorical (one of 8 labels), not a 1-5 scalar.
    # Prometheus's rubric protocol is built for scalar 1-5; we use the
    # instruct_json prompt for all judges on this metric.
    protocol = _judge_protocol(judge_name)
    # Response-only classification (original §3.3 intent): the judge sees
    # ONLY the model's response, no violation turn and no scenario context.
    # This tests whether distinct redirect axes produce distinct repair
    # shapes recognizable from response surface form alone -- the
    # falsifiable form of the §3.3 "distinct invariants -> distinct
    # minimal repairs" claim. With context present the judge can answer
    # from the violation alone and the metric becomes judge-agrees-with-
    # ground-truth, not response-shape-discriminability.
    prompt = REDIRECT_AXIS_PROMPT.format(
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
                            judge_name: str | None = None,
                            rubric_variant: str = "generic") -> dict[str, Any]:
    protocol = _judge_protocol(judge_name)
    response = (record.get("generation") or "").strip()[:1500]
    if protocol == "prometheus_rubric":
        if rubric_variant == "tutor_v2":
            rubric = PROMETHEUS_NATURALNESS_TUTOR_V2_RUBRIC
        elif rubric_variant == "tutor":
            rubric = PROMETHEUS_NATURALNESS_TUTOR_RUBRIC
        else:
            rubric = PROMETHEUS_NATURALNESS_RUBRIC
        prompt = PROMETHEUS_PROMPT_TEMPLATE.format(
            instruction=(
                "You are an English tutor in a learner-facing dialogue. "
                "Produce a tutor turn."
            ),
            response=response,
            rubric=rubric,
        )
        raw = await judge.generate(
            system="You are a fair, rubric-driven evaluator.",
            messages=[{"role": "user", "content": prompt}],
            max_new_tokens=400, temperature=0.0,
        )
        score = _parse_prometheus_score(raw)
        return {"raw": raw, "score": score,
                "note": _parse_prometheus_feedback(raw),
                "protocol": protocol, "rubric_variant": rubric_variant}
    # instruct_json / default
    if rubric_variant == "tutor_v2":
        tmpl = NATURALNESS_TUTOR_V2_PROMPT
    elif rubric_variant == "tutor":
        tmpl = NATURALNESS_TUTOR_PROMPT
    else:
        tmpl = NATURALNESS_PROMPT
    # tutor_v2 prompt takes {level}; v1 and generic don't.
    if rubric_variant == "tutor_v2":
        prompt = tmpl.format(level=record.get("cefr_level", "B1"),
                             response=response)
    else:
        prompt = tmpl.format(response=response)
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
            "protocol": protocol, "rubric_variant": rubric_variant}


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


def _judged_out_path(judge_name: str, baseline: str, test_set: str,
                     metric: str, rubric_variant: str) -> Path:
    """Output path for a judged-scoring run.

    For the original generic rubric (preserved for back-compat / reuse),
    we keep the legacy ``<test_set>__<metric>.jsonl`` filename. For new
    rubric variants we append ``__<variant>`` so the two scores coexist
    side-by-side and can be reported as a sensitivity analysis.
    """
    if rubric_variant == "generic":
        name = f"{test_set}__{metric}.jsonl"
    else:
        name = f"{test_set}__{metric}__{rubric_variant}.jsonl"
    return SCORE_ROOT / "judged" / judge_name / baseline / name


async def run_judged(baseline: str, judge_name: str,
                     metric_filter: list[str] | None = None,
                     limit: int | None = None,
                     rubric_variant: str = "generic") -> dict[str, Any]:
    # Build the judge client once (it loads a model).
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_paper_eval import build_baseline  # type: ignore
    judge = build_baseline(judge_name)
    await _verify_judge_model_match(judge_name)

    summary: dict[str, Any] = {"baseline": baseline, "judge": judge_name,
                                "rubric_variant": rubric_variant}
    if limit:
        summary["limit"] = limit
    metrics = [m for m in JUDGED_METRIC_FUNCS
               if metric_filter is None or m in metric_filter]
    for metric in metrics:
        func, test_set = JUDGED_METRIC_FUNCS[metric]
        in_path = EVAL_ROOT / baseline / f"{test_set}.jsonl"
        # redirect_axis is categorical; the rubric_variant only affects
        # the 1-5 scalar metrics (cefr_adherence, naturalness). For
        # redirect_axis we always write to the legacy path so the
        # existing data isn't duplicated.
        effective_variant = ("generic" if metric == "redirect_axis"
                             else rubric_variant)
        out_path = _judged_out_path(judge_name, baseline, test_set,
                                    metric, effective_variant)
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
        variant_tag = (f" rubric={effective_variant}"
                       if effective_variant != "generic" else "")
        print(f"  [{metric}{variant_tag}] judging {len(todo)} new records "
              f"(skipping {len(done)} done){suffix}...")
        for idx, rec in enumerate(todo, 1):
            try:
                # judge_name is plumbed through so the per-judge prompt
                # protocol can dispatch (Prometheus rubric vs JSON).
                verdict = await func(judge, rec, judge_name=judge_name,
                                     rubric_variant=effective_variant)
                _append_jsonl(out_path, {
                    "id": rec["id"], "baseline": baseline,
                    "judge": judge_name, "metric": metric,
                    "test_set": test_set,
                    "cefr_level": rec.get("cefr_level"),
                    "rubric_variant": effective_variant,
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

    # Judged: median across judges per metric per record, then mean.
    # For cefr_adherence + naturalness we aggregate BOTH the generic and
    # tutor-anchored rubric variants if present, so the per-baseline JSON
    # carries both numbers for the sensitivity-analysis report in §5.
    for metric, (_, test_set) in JUDGED_METRIC_FUNCS.items():
        # redirect_axis only has the generic variant; cefr/naturalness can
        # have both. The variants are written to different files.
        if metric == "redirect_axis":
            variants_to_read = [("generic", f"{test_set}__{metric}.jsonl",
                                 metric)]
        else:
            variants_to_read = [
                ("generic",  f"{test_set}__{metric}.jsonl",          metric),
                ("tutor",    f"{test_set}__{metric}__tutor.jsonl",   f"{metric}_tutor"),
                ("tutor_v2", f"{test_set}__{metric}__tutor_v2.jsonl",f"{metric}_tutor_v2"),
            ]
        for variant, filename, metric_label in variants_to_read:
            per_record_scores: dict[str, list[Any]] = defaultdict(list)
            # Prometheus is dropped from the redirect_axis ensemble: its
            # rubric-trained behavior emits 200-700-char "Feedback: ..."
            # prose that exhausts max_new_tokens=160 with no parseable JSON,
            # falling through to axis="none" on ~98% of records. The
            # categorical metric is judged by Llama-3.1 + Gemma-2 only.
            judges_for_metric = (
                [j for j in judges if j != "prometheus_7b_judge"]
                if metric == "redirect_axis" else judges
            )
            for judge in judges_for_metric:
                jp = SCORE_ROOT / "judged" / judge / baseline / filename
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
            median_per_record: list[Any] = []
            for rid, vals in per_record_scores.items():
                if metric == "redirect_axis":
                    c = Counter(vals)
                    median_per_record.append(c.most_common(1)[0][0])
                else:
                    vals_s = sorted(vals)
                    median_per_record.append(vals_s[len(vals_s) // 2])
            if metric == "redirect_axis":
                summary[f"{metric_label}_distribution"] = dict(Counter(median_per_record))
                # Also store the predictions, one per record, for downstream
                # macro-F1 computation against ground-truth axes.
                summary[f"{metric_label}_predictions"] = {
                    rid: (Counter(vals).most_common(1)[0][0] if vals else None)
                    for rid, vals in per_record_scores.items()
                }
            else:
                summary[f"{metric_label}_mean"] = sum(median_per_record) / len(median_per_record)
                summary[f"{metric_label}_n"] = len(median_per_record)

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
    p.add_argument("--rubric", default="generic",
                   choices=("generic", "tutor", "tutor_v2"),
                   help="Rubric variant for cefr_adherence + naturalness "
                        "judging. `generic` is the original chat-bot-style "
                        "1-5 rubric (1.0 baseline). `tutor` is the "
                        "tutor-anchored rubric that penalizes markdown "
                        "bullets, enumeration of options, and chatbot-style "
                        "framing. `tutor_v2` adds CEFR-keyed length anchors "
                        "and 4 few-shot calibration examples per metric "
                        "(stronger judge anchoring). Each variant writes to "
                        "<test_set>__<metric>__<variant>.jsonl so all three "
                        "scores coexist for sensitivity analysis. Ignored "
                        "for redirect_axis (categorical).")
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
                                   limit=args.limit,
                                   rubric_variant=args.rubric)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
