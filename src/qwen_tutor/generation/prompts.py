"""prompts.py  -  prompt templates for data generation (locale-agnostic).

Design summary
--------------
All locale information comes from a single file, ``config/locale.yaml``. This
module does not maintain static lists for cities/food/names and instead delegates
that responsibility to the teacher model's knowledge. Given only a ``country``
name, the teacher fills in that country's cities/food/names/transport/culture.

Each prompt undergoes two stages of substitution.

  1. At module import time - locale placeholders such as ``{country}``,
     ``{country_adjective}``, ``{learner_description}``, and
     ``{avoided_topics_sentence}`` are pre-filled from the ``LOCALE`` value using
     ``str.replace``.
  2. At use time (``.format``) - dynamic placeholders such as ``{N}``,
     ``{level}``, and ``{scenario_json}`` are filled by the caller.

Literal ``{``/``}`` inside JSON bodies are double-escaped as ``{{``/``}}`` for
``.format`` (locale placeholders are not double-escaped this way).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from qwen_tutor.locale import LOCALE

# ---------------------------------------------------------------------------
# Compatibility constants - names imported by legacy code/tests
# ---------------------------------------------------------------------------

LOCALE_INSTRUCTION_HEADER = LOCALE.locale_instruction_header
LOCALE_INSTRUCTION_BLOCK = LOCALE.locale_instruction_block

# Short anti-default notice appended to all generation prompts. We do not keep
# country-specific lists (names/cities/foods); instead we strongly instruct the
# teacher to choose authentic {country_adjective} items from its own knowledge.
# The cultures to avoid come from each locale's ``avoid_default_cultures``
# setting in ``config/locale.yaml`` - default is "American/European", and users
# can add "Japanese", "Singapore", etc.
#
# The raw version preserves placeholders so ``_localize_with`` can substitute
# them at call time (multi-locale support). ``ANTI_FAILURE_MODE_BLOCK`` is a
# back-compat constant pre-substituted for the default locale.
ANTI_FAILURE_MODE_BLOCK_RAW = (
    "ANTI-FAILURE-MODE INSTRUCTIONS:\n"
    "Do NOT default to {avoid_cultures_phrase} names, places, foods,\n"
    "or brands. All proper nouns (people, cities, foods, neighborhoods,\n"
    "brands, transit, universities, holidays) must be authentically\n"
    "{country_adjective}. Draw on your own knowledge of {country}.\n"
    "Vary cities; do not default to the capital for every scenario.\n\n"
    "{avoided_topics_sentence}\n\n"
    "If you find yourself reaching for {avoid_cultures_phrase} defaults\n"
    "out of habit, STOP and substitute an authentic {country_adjective}\n"
    "item from your knowledge."
)
ANTI_FAILURE_MODE_BLOCK = ANTI_FAILURE_MODE_BLOCK_RAW.replace(
    "{avoid_cultures_phrase}", LOCALE.avoid_cultures_phrase
).replace(
    "{country_adjective}", LOCALE.country_adjective
).replace(
    "{country}", LOCALE.country
).replace(
    "{avoided_topics_sentence}", LOCALE.avoided_topics_sentence
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_cefr_specs_path() -> Path:
    return Path("config/cefr_specs.yaml")


def render_level_spec(
    cefr_level: str,
    cefr_specs_path: str | Path | None = None,
    locale_name: str | None = None,
    *,
    allow_l1: bool = False,
) -> str:
    """Return a text block for the given CEFR level containing vocabulary,
    grammar, naturalness requirements, things to avoid, and few-shot examples.

    This is inserted into every prompt at ``{level_spec_with_locale_instruction}``.
    The locale's ``locale_instruction_block`` is repeated at the top so the
    locale header remains after ``.format`` is applied.

    If ``locale_name`` is omitted, the default locale from ``config/locale.yaml``
    is used (back-compat).

    ``allow_l1=True`` substitutes ``locale_instruction_block_allow_l1`` so the
    embedded locale rules don't contradict the outer prompt's speaks_l1
    override. Only ``language_redirect.py`` sets this — every other caller
    keeps the strict variant. Without this carve-out the relaxed block at
    the top of ``dialogue_language_redirect`` and the strict block embedded
    here would conflict inside the same prompt and the teacher resolved
    randomly (50% pass rate for speaks_l1).
    """
    from qwen_tutor.locale import get_locale

    loc = get_locale(locale_name)
    path = Path(cefr_specs_path) if cefr_specs_path else _default_cefr_specs_path()
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

    if cefr_level not in doc.get("levels", {}):
        valid = ", ".join(sorted(doc.get("levels", {}).keys()))
        raise KeyError(
            f"CEFR level {cefr_level!r} not found in {path}. Valid: {valid}"
        )
    level = doc["levels"][cefr_level]

    locale_block = (
        loc.locale_instruction_block_allow_l1 if allow_l1
        else loc.locale_instruction_block
    )
    sections: list[str] = [
        locale_block,
        f"CEFR LEVEL: {cefr_level}",
        "=" * 60,
        f"Vocabulary:\n{level.get('vocabulary', '').rstrip()}",
        f"Grammar:\n{level.get('grammar', '').rstrip()}",
        f"Naturalness requirements:\n{level.get('naturalness_requirements', '').rstrip()}",
        f"Things to avoid:\n{level.get('things_to_avoid', '').rstrip()}",
    ]
    examples = level.get("few_shot_examples") or []
    if examples:
        ex_parts = ["Few-shot example dialogues at this level:"]
        for i, ex in enumerate(examples, start=1):
            title = ex.get("title", f"Example {i}")
            dialogue = (ex.get("dialogue") or "").rstrip()
            ex_parts.append(f"[{i}] {title}\n{dialogue}")
        sections.append("\n\n".join(ex_parts))
    return "\n\n".join(sections)


def validate_prompt_has_locale_instruction(
    prompt: str, locale_name: str | None = None
) -> None:
    """Check that the rendered prompt contains the locale-instruction header.

    This guard prevents a refactor mistake where the header is accidentally
    omitted and the teacher model falls back to plain English. If ``locale_name``
    is omitted, it checks the default locale's header (back-compat).
    """
    from qwen_tutor.locale import get_locale

    header = get_locale(locale_name).locale_instruction_header
    if header not in prompt:
        raise ValueError(
            "Rendered prompt is missing the required "
            f"{header!r} header. Every prompt sent to the "
            "teacher model must contain the locale-grounding block."
        )


# ---------------------------------------------------------------------------
# Dialogue length (generation.min_turns / max_turns in config/generation.yaml)
# ---------------------------------------------------------------------------
#
# The dialogue length range the prompt requests from the teacher comes from YAML.
# It is read once at module load and stored as static constants, so it does not
# need to be filled with ``.format()`` at runtime and can be batch-substituted
# by _localize.
#
# If the config file is missing or keys are absent, fall back to (10, 16) so
# loading prompts alone in test environments does not fail.


def _load_turn_config() -> tuple[int, int]:
    path = Path("config/generation.yaml")
    try:
        with path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return 10, 16
    gen = doc.get("generation") or {}
    min_t = int(gen.get("min_turns", 10))
    max_t = int(gen.get("max_turns", 16))
    if min_t < 4:
        raise ValueError(
            f"generation.min_turns={min_t} too small; need at least 4 for a "
            "meaningful dialogue (user/assistant/user/assistant)"
        )
    if max_t < min_t:
        raise ValueError(
            f"generation.max_turns={max_t} < min_turns={min_t}; max must be "
            ">= min"
        )
    return min_t, max_t


MIN_TURNS, MAX_TURNS = _load_turn_config()

# Turn index range for the off-topic probe in redirect dialogues.
# Place it in the middle to avoid too early (insufficient warm-up) or too late
# (insufficient recovery time). In the default (10..16) range it becomes
# roughly 4..14, and it follows if the user changes min/max.
PROBE_MIN_TURN = max(2, MIN_TURNS // 2)
PROBE_MAX_TURN = max(PROBE_MIN_TURN + 1, MAX_TURNS - 2)


# ---------------------------------------------------------------------------
# Locale substitution helpers
# ---------------------------------------------------------------------------


def _localize_with(s: str, loc, *, allow_l1: bool = False) -> str:
    """Render locale placeholders against a specific ``LocaleConfig``.

    Use simple ``str.replace`` to avoid collisions with ``.format()`` dynamic
    placeholders such as ``{N}``. JSON body ``{{`` / ``}}`` escapes are preserved.
    Non-locale placeholders like ``{min_turns}``, ``{max_turns}``, and
    ``{probe_*_turn}`` are also handled together (values from generation.yaml).

    ``allow_l1=True`` substitutes ``locale_instruction_block_allow_l1`` (the
    relaxed variant that permits ONE user turn in native L1 script) instead
    of the strict-Latin default. Only ``render_prompt("dialogue_language_redirect")``
    sets this — every other prompt must keep strict Latin.
    """
    locale_block = (
        loc.locale_instruction_block_allow_l1 if allow_l1
        else loc.locale_instruction_block
    )
    return (
        s.replace("{country_adjective}", loc.country_adjective)
        .replace("{country}", loc.country)
        .replace("{learner_description}", loc.learner_description)
        .replace("{avoid_cultures_phrase}", loc.avoid_cultures_phrase)
        .replace("{locale_instruction_block}", locale_block)
        .replace("{avoided_topics_sentence}", loc.avoided_topics_sentence)
        .replace(
            "{avoided_topics_block_for_redirect_prompt}",
            loc.avoided_topics_block_for_redirect_prompt,
        )
        .replace("{min_turns}", str(MIN_TURNS))
        .replace("{max_turns}", str(MAX_TURNS))
        .replace("{probe_min_turn}", str(PROBE_MIN_TURN))
        .replace("{probe_max_turn}", str(PROBE_MAX_TURN))
    )


def _localize(s: str) -> str:
    """Back-compat shim: localize against the module-load-time default LOCALE.

    Some prompts (generation-stage prompts that are still single-locale)
    rely on this returning a pre-substituted string at module load. Once
    every caller passes an explicit locale, this can be removed.
    """
    return _localize_with(s, LOCALE)


# ---------------------------------------------------------------------------
# Qwen3 think-mode directive
# ---------------------------------------------------------------------------
#
# Qwen3 enables or disables the reasoning block based on ``/think`` /
# ``/no_think`` in the system prompt. Other models (Claude / GPT-4 / gpt-oss /
# DeepSeek-R1) treat these tokens as plain text, so we append them as a single
# line at the end of the system prompt for multi-model compatibility.
#
# Policy:
#   * Dialogue stages (seeds / sft / redirect / register) - /no_think.
#     Reasoning is not required and we do not want empty ``<think>\n</think>``
#     leaking into outputs.
#   * Eval stage (EVALUATION_GENERATION_PROMPT) - /think. For training data to
#     include ``<think>...</think>`` + JSON, the teacher must actually produce
#     reasoning.

NO_THINK_DIRECTIVE = "\n\n/no_think"
THINK_DIRECTIVE = "\n\n/think"


def _with_no_think(s: str) -> str:
    return s + NO_THINK_DIRECTIVE


def _with_think(s: str) -> str:
    return s + THINK_DIRECTIVE


# Regex pairs used to rewrite eval prompts when ``thinking.student_eval`` is
# ``no_think``. Order matters — longest / most-specific patterns first so they
# match before the catch-all near the end. Patterns use ``\s+`` everywhere two
# words could be separated by whitespace or a literal newline (the source
# templates are line-wrapped). Each pattern preserves the rest of the prompt
# verbatim; only the ``<think>`` scaffolding is replaced.
_NO_THINK_EVAL_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    # "Your output will train a Qwen3 model running in /think mode"
    (re.compile(r"running in /think mode", re.IGNORECASE), "running in /no_think mode"),
    # "produce a <think>...</think> block in which you reason carefully ... followed [IMMEDIATELY] by a single JSON object"
    (
        re.compile(
            r"produce\s+a\s+<think>\.{0,3}</think>\s+block\s+in\s+which\s+you\s+reason\s+carefully[^.]*?,\s+followed\s+(?:IMMEDIATELY\s+)?by\s+a\s+single\s+JSON\s+object",
            re.IGNORECASE | re.DOTALL,
        ),
        "produce a single JSON object directly",
    ),
    # "Output no prose before <think>, no prose between </think> and the opening"
    (
        re.compile(
            r"Output\s+no\s+prose\s+before\s+<think>,\s+no\s+prose\s+between\s+</think>\s+and\s+the\s+opening",
            re.IGNORECASE,
        ),
        "Output a single JSON object directly, no prose before the opening",
    ),
    # "A <think>...</think> block, immediately followed by a single JSON object"
    (
        re.compile(
            r"A\s+<think>\.{0,3}</think>\s+block,\s+immediately\s+followed\s+by\s+a\s+single\s+JSON\s+object",
            re.IGNORECASE | re.DOTALL,
        ),
        "A single JSON object",
    ),
    # "Inside <think>, cite specific turns: e.g. \"Turn 3 user...\" (one full
    # sentence of guidance, ending at the next paragraph break)
    (
        re.compile(
            r"Inside\s+<think>,\s+cite\s+specific\s+turns:[^\n]*\n[^\n]*?\n\n",
            re.IGNORECASE,
        ),
        "",
    ),
    # "Now produce your <think> block and EvaluationOutput JSON"
    (
        re.compile(
            r"Now\s+produce\s+your\s+<think>\s+block\s+and\s+EvaluationOutput\s+JSON",
            re.IGNORECASE,
        ),
        "Now produce your EvaluationOutput JSON",
    ),
    # Catch-all for any remaining mention of "a <think>...</think> block"
    (
        re.compile(r"a\s+<think>\.{0,3}</think>\s+block", re.IGNORECASE | re.DOTALL),
        "your output",
    ),
    # Final cleanup: any stray "<think>" or "</think>" token mentions (e.g.
    # "block tags <think>...</think>") — replace with empty so the prompt
    # doesn't accidentally instruct the student to emit the tag.
    (re.compile(r"<think>\.{0,3}</think>", re.DOTALL), ""),
    (re.compile(r"</?think>"), ""),
)


def _strip_think_instructions_for_no_think(prompt: str) -> str:
    """Rewrite an eval prompt so it no longer instructs the model to emit a
    ``<think>...</think>`` block.

    Called by ``render_prompt`` (for the eval-generation teacher prompt) and
    ``render_evaluation_system_prompt`` (for the trained student's deploy/eval
    system prompt) when ``thinking.student_eval`` resolves to ``no_think``.

    The transformation is a sequence of targeted phrase substitutions rather
    than a wholesale rewrite — the rest of the prompt (rubric, JSON schema,
    locale block, etc.) is preserved verbatim.
    """
    for pattern, replacement in _NO_THINK_EVAL_REWRITES:
        prompt = pattern.sub(replacement, prompt)
    return prompt


def _student_eval_mode() -> str:
    """Return the resolved ``thinking.student_eval`` value (``think`` /
    ``no_think`` / ``auto``). Cached per-process by
    ``qwen_tutor.utils.thinking``; cheap to call repeatedly."""
    # Local import to avoid a top-level cycle (utils.thinking imports yaml at
    # module load and we don't want to drag that into the prompts module).
    from qwen_tutor.utils.thinking import get_thinking_mode

    return get_thinking_mode("student_eval")


# ---------------------------------------------------------------------------
# 1) TOPIC_SEED_PROMPT  -  generate level-specific scenario seeds
# ---------------------------------------------------------------------------

_TOPIC_SEED_PROMPT = (
    """\
You are designing realistic conversation scenarios for an English-language
tutoring application whose learners are {learner_description} living in
{country}. Each scenario will later be expanded into a multi-turn dialogue
used to train the tutor model. Generate {N} diverse, authentic scenarios at
CEFR level {level}, returned as a single JSON array.

----------------------------------------------------------------------
ROLE
----------------------------------------------------------------------
You are a senior curriculum designer with deep, lived knowledge of {country}:
everyday life across major cities, mid-size cities, and smaller towns;
the practical situations adult English learners actually encounter; and
how target-register English differs across A1, A2, B1, B2, C1, and C2.

You are NOT writing for an audience outside {country}. Each scenario must
feel like a situation the learner could plausibly be in - or could
plausibly observe around them - in their own daily life.

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
CEFR LEVEL SPEC (register, vocabulary, grammar, naturalness, few-shot)
----------------------------------------------------------------------
The block below defines the target register for level {level}. Read it
carefully; every scenario you generate must be plausible at that register
once it is expanded into dialogue.

{level_spec_with_locale_instruction}

----------------------------------------------------------------------
WHAT A SCENARIO LOOKS LIKE
----------------------------------------------------------------------
Each scenario is a JSON object with the following fields:
  - "topic": a short noun phrase describing the conversational situation.
  - "subtopics": provide AT LEAST 4 and ideally 5 (never fewer than 4, at
    most 6) DESCRIPTIVE phrases, each a concrete conversational beat the
    dialogue can spend a few turns on. Each should be a short clause of
    roughly 5-10 words with a specific detail — NOT a bare 2-3 word label.
    They should read like distinct moments in the encounter, not synonyms
    of the topic. A scenario with only 3 subtopics is INCOMPLETE — add
    one or two more concrete beats.
      GOOD (specific, ~6-10 words each):
        "asking whether the noodles are very spicy",
        "comparing the price of two breakfast sets",
        "learning the name of a local morning drink",
        "asking the vendor to make it less oily"
      TOO SHORT (avoid — bare labels):
        "ordering food", "prices", "drinking tea"
  - "user_role": object with "name" (an authentic {country_adjective}
    first name from your knowledge) and "description" (1 sentence:
    age band, occupation, why they are in this conversation).
  - "model_role": object with "name" (role label like "tutor", "neighbor",
    "shopkeeper", "family member", "classmate", "colleague",
    "doctor") and "description" (1 sentence about who they are).
  - "setting": 1-2 sentences. Be SPECIFIC: which city of {country},
    which neighborhood or street, what time of day, what season, what
    the environment is like. Avoid generic settings like "in {country}".
  - "cefr_level": always the literal string "{level}" for this batch.

----------------------------------------------------------------------
LIFE-DOMAIN CATEGORIES
----------------------------------------------------------------------
This batch is pre-assigned a life-domain category. A category is a broad
life domain; the specific topic, subtopics, and roles are up to you
within it. This batch's assignment:

{categories_block}

Category meaning (use as a soft guide):
{category_meanings_block}

----------------------------------------------------------------------
VARIETY REQUIREMENTS (across the batch of {N} scenarios)
----------------------------------------------------------------------
  1. NO two scenarios may share the same "topic". Each topic must be
     genuinely distinct.
  2. NO single city of {country} may appear in more than roughly
     one-third of the scenarios. Spread settings across major cities,
     mid-size cities, and smaller towns from your knowledge of {country}.
     Do NOT default to the capital.
  3. Vary user_role.name. Use a wide range of authentic {country_adjective}
     first names spanning genders and generations from your knowledge.
     Do NOT repeat the same first name within the batch.
  4. Vary model_role (neighbors, vendors, family members, classmates,
     colleagues, doctors and clinic staff, drivers, librarians,
     repair people, building supervisors).
  5. Vary the time of day (morning, midday, afternoon, evening, late
     night) and the season. Mention these in "setting".
  6. Vary indoor vs. outdoor settings: bazaars/markets, parks, transit
     stations, classrooms, canteens, stairwells, kitchens, courtyards,
     trails near a city.

----------------------------------------------------------------------
CONTENT BOUNDARIES (HARD REQUIREMENTS)
----------------------------------------------------------------------
{avoided_topics_sentence} If a topic strays into any of these areas,
replace it with a neutral alternative (food, family meals, hobbies,
work, weather, travel within {country}, study, health, shopping,
transportation, neighborhood life, or cultural arts that are not
religious).

Also avoid {avoid_cultures_phrase} politics, current events, or public
figures.

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON array of exactly {N} scenario objects. No prose
before or after the array. No markdown code fences. No commentary.

[
  {{
    "topic": "...",
    "subtopics": ["...", "...", "...", "...", "..."],
    "user_role": {{"name": "...", "description": "..."}},
    "model_role": {{"name": "...", "description": "..."}},
    "setting": "...",
    "cefr_level": "{level}"
  }}
]

----------------------------------------------------------------------
FINAL CHECK
----------------------------------------------------------------------
Before returning your answer, silently verify each scenario:
  - Is the first name authentically {country_adjective}? If not, replace it.
  - Is the setting a specific city + neighborhood + time + season?
  - Did you write AT LEAST 4 subtopics (count them)? If only 3, add one or two more concrete beats now.
  - Is each subtopic a descriptive ~5-10 word beat, not a bare 2-3 word label?
  - Are there any {avoid_cultures_phrase} proper nouns anywhere? If yes, replace them.
  - Does the cefr_level field equal exactly "{level}"?
  - Are all {N} scenarios genuinely distinct in topic?

Now generate {N} distinct scenarios at CEFR level {level}, following
all of the above. Output only the JSON array.
"""
)

TOPIC_SEED_PROMPT = _with_no_think(_localize(_TOPIC_SEED_PROMPT))


# ---------------------------------------------------------------------------
# 2) DIALOGUE_PROMPT_NORMAL  -  generate normal dialogue SFT
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_NORMAL = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario (tutor,
neighbor, shopkeeper, family member, classmate, etc.). The dialogue will be
used as a training example for an English-tutor model intended for
{country_adjective} users, so EVERY detail must be locale-authentic and
pitched at the requested CEFR register.

You will be given a scenario JSON object and a CEFR level spec. Produce
a single dialogue, {min_turns}-{max_turns} turns long (counting user and assistant turns
together, alternating), as a JSON object whose "messages" field is an
array of {{"role": "user"|"assistant", "content": "..."}} entries.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
USE YOUR OWN KNOWLEDGE OF {country}
----------------------------------------------------------------------
Ground every detail in your own knowledge of {country}:
  - real neighborhoods, streets, parks, markets, and landmarks of the
    specific city the scenario is set in;
  - foods, drinks, and cultural items typical of {country};
  - the everyday currency and how people speak about prices;
  - real transit and ride options for {country};
  - real universities and schools of {country};
  - regional weather patterns;
  - shared cultural rhythms and holidays.

The dialogue should feel grounded in a SPECIFIC setting in {country},
not in a generic "somewhere in {country}" setting.

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
The block below defines the target register at level {level}. The
assistant's English must stay inside this register.

{level_spec_with_locale_instruction}

----------------------------------------------------------------------
NATURAL-LANGUAGE TUTORING PRINCIPLES
----------------------------------------------------------------------
  - The user's turns may contain small, plausible learner errors typical
    at this CEFR level (article omission, tense confusion, phrasal-verb
    misuse, L1-influenced phrasing). Errors should be REALISTIC.
  - The assistant's turns model the target register cleanly.
  - At A1-A2, gentle recasting is better than explicit teaching. At B1+,
    occasional light explanation is fine. At C1-C2, the partner sounds
    like a fluent speaker who engages naturally with the content.
  - The assistant LEADS the conversation: it ends nearly every turn with
    one concrete question or a small suggestion so the learner always has
    an easy way to continue (one question, never a menu of 3+). It shares
    small reactions and keeps things moving. It is a conversation, not a
    quiz.
  - Include natural moments where the LEARNER is passive or stuck (a short
    "I don't know", a one-word reply, "you choose"): the assistant does
    NOT stall or wait - it proactively offers the next small topic or asks
    a specific, easy question to keep the dialogue alive.
  - Avoid textbook patterns: "Option A... Option B...", bullet points,
    headings, numbered steps, robotic over-politeness.

----------------------------------------------------------------------
CONTENT BOUNDARIES
----------------------------------------------------------------------
{avoided_topics_sentence} If the scenario brushes against any of these,
steer the dialogue into a neutral nearby topic.

Also avoid {avoid_cultures_phrase} politics and current events.

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. No prose, no markdown,
no commentary.

Each message object has:
  - "role": either "user" or "assistant"
  - "content": the turn's text as a plain string

The first message must be a "user" turn. Roles strictly alternate.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

----------------------------------------------------------------------
FINAL CHECK
----------------------------------------------------------------------
Silently verify:
  - Total turn count is between {min_turns} and {max_turns} inclusive.
  - Roles strictly alternate, starting with "user".
  - All proper nouns are {country_adjective}; no {avoid_cultures_phrase} names, places,
    foods, or brands appear anywhere.
  - The setting (city + neighborhood + time + season) is consistent
    with the scenario and is referenced concretely in the dialogue.
  - The assistant's English matches level {level}.
  - User turns contain level-appropriate realistic errors at A1-A2 and
    clean conversational English at higher levels.
  - No markdown formatting inside any "content" field.

Now produce the dialogue JSON object.
"""
)

DIALOGUE_PROMPT_NORMAL = _with_no_think(_localize(_DIALOGUE_PROMPT_NORMAL))


# ---------------------------------------------------------------------------
# 3) DIALOGUE_PROMPT_REDIRECT  -  generate redirect-moment SFT
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_REDIRECT = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario, in which
the LEARNER (the "user") brings up a sensitive or out-of-scope topic at
some point during the conversation, and the partner (the "assistant")
performs a graceful REDIRECT to a safe, related topic - without lecturing,
scolding, or breaking the conversational frame.

This dialogue will train the tutor model to handle out-of-scope moments
naturally rather than refusing stiffly. EVERY detail must be
locale-authentic and pitched at the requested CEFR register.

You will be given a scenario JSON object, a CEFR level spec, and a
"redirect_axis" string indicating which kind of out-of-scope topic the
learner brings up. Produce a single dialogue, {min_turns}-{max_turns} turns long, in which
the redirect happens around the middle and the dialogue continues
smoothly afterward.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
REDIRECT AXIS FOR THIS DIALOGUE
----------------------------------------------------------------------
{redirect_axis}

{avoided_topics_block_for_redirect_prompt}

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
GROUND EVERY DETAIL IN {country}
----------------------------------------------------------------------
The setting, names, food references, neighborhoods, transit, currency,
and cultural items must come from your own knowledge of {country}.

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
HOW THE REDIRECT MOMENT SHOULD READ
----------------------------------------------------------------------
  - The learner introduces the redirect_axis topic naturally - it does
    not feel forced, just unintentionally out-of-scope for a tutor.
  - The assistant does NOT affirm, praise, or engage with the substance
    of what the learner raised (no "that sounds nice/interesting"). It
    gives a brief, honest, non-committal deflection - a light "Hmm, I'm
    not sure about that" / "I don't really know about that" - WITHOUT
    naming or discussing the off-scope item, then repairs by pivoting to
    a related, safe topic per the redirect_axis pivot guidance above. The
    pivot stays in-character.
  - The assistant does NOT lecture, scold, or refuse stiffly. No
    "I cannot discuss that". No "as your tutor I am not allowed".
  - After the pivot, the conversation continues for several more turns
    on the new safe topic. The redirect should not end the dialogue.

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first message must be
"user". Roles strictly alternate. No prose, no markdown, no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

----------------------------------------------------------------------
FINAL CHECK
----------------------------------------------------------------------
Silently verify:
  - The redirect moment is around the middle of the dialogue.
  - The assistant pivots gracefully and the conversation continues.
  - All proper nouns are {country_adjective}; no {avoid_cultures_phrase} names, places,
    foods, or brands appear anywhere.
  - The assistant's English stays at level {level}.

Now produce the dialogue JSON object for redirect_axis = "{redirect_axis}".
"""
)

DIALOGUE_PROMPT_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_REDIRECT))


# ---------------------------------------------------------------------------
# 3b) DIALOGUE_PROMPT_LOCALE_REDIRECT  -  user-violates-locale SFT
# ---------------------------------------------------------------------------
# When the learner casually mentions a {avoid_cultures_phrase} item
# (food/place/brand/name) mid-dialogue, the tutor gracefully acknowledges
# without scolding, lecturing on locale, or refusing, then naturally
# re-anchors the conversation in the {country} context.
# As redirect_*.jsonl handles user-side off-topic violations, this is the
# user-side handling for locale_violation.

_DIALOGUE_PROMPT_LOCALE_REDIRECT = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. At some
point during the conversation, the LEARNER (the user) casually mentions a
{avoid_cultures_phrase} item ({locale_trigger}) - the way a real learner
might, without realizing it is culturally out-of-place for a dialogue set
in {country}. The TUTOR (the assistant) must respond GRACEFULLY:
acknowledge naturally, optionally bridge to an authentic {country_adjective}
equivalent in passing, and continue the conversation in stride.

The tutor must NOT lecture, correct, scold, or pivot stiffly. NO "we are
talking about {country}, please use {country_adjective} examples". NO
"that is a Western reference and is not appropriate". The tutor stays
in character, treats the {avoid_cultures_phrase} mention as a normal
conversational moment, and naturally re-grounds in {country}.

You will be given a scenario JSON object, a CEFR level spec, and a
"locale_trigger" string indicating which kind of {avoid_cultures_phrase}
item the learner brings up. Produce a single dialogue, {min_turns}-{max_turns}
turns long, in which the locale slip happens around the middle and the
dialogue continues smoothly afterward.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
LOCALE TRIGGER FOR THIS DIALOGUE
----------------------------------------------------------------------
{locale_trigger}

The learner should drop a specific {avoid_cultures_phrase} item of this
kind into one of their turns. For example:
  - food: pizza, hot dog, croissant, sushi (if non-locale), bagel
  - place: New York, Paris, London, Tokyo (if non-locale)
  - brand: McDonald's, Starbucks, Walmart, Coca-Cola
  - person: John, Mary, Bob, Jenny, Sophie (Western first names)

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
HOW THE LOCALE-RESPONSE MOMENT SHOULD READ
----------------------------------------------------------------------
  - The learner mentions the {avoid_cultures_phrase} item naturally - it
    is just what came to mind, not a test.
  - The tutor does NOT affirm, praise, or engage the {avoid_cultures_phrase}
    item; it gives a brief, non-committal "Hmm, I'm not sure about that",
    then weaves in a {country_adjective} parallel without making it the
    topic. CRITICAL: the tutor must NOT name the {avoid_cultures_phrase}
    entity in its reply. Repeating it propagates the Western reference
    into the tutor's training signal and the example will be rejected by
    the locale_judge filter.
       BAD  (rejected):  "Oh, pizza! We don't have that here much,
                          but jianbing is popular too."
       GOOD (passes):    "Hmm, I'm not sure about that — but jianbing is
                          a popular street food here, have you tried it?"
    Use non-committal deflections ("Hmm, I'm not sure about that",
    "Oh, I don't really know that one") instead of praising or repeating
    the entity name.
  - The tutor does NOT lecture about cultural context. NO "in {country}
    we eat...". NO "we should focus on {country_adjective} examples".
  - The locale grounding happens through CONTENT around it (the tutor's
    other proper nouns stay {country_adjective}), not through
    correction of the learner.
  - After the moment, the conversation continues for several more turns.

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first
message must be "user". Roles strictly alternate. No prose, no markdown,
no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Now produce the dialogue JSON object for locale_trigger = "{locale_trigger}".
"""
)

DIALOGUE_PROMPT_LOCALE_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_LOCALE_REDIRECT))


# ---------------------------------------------------------------------------
# 3c) DIALOGUE_PROMPT_PEDAGOGY_REDIRECT  -  user-asks-for-lecture SFT
# ---------------------------------------------------------------------------
# When the learner explicitly requests a grammar rule, vocab list, or
# conjugation explanation, the tutor stays conversational and either models
# the form naturally or gives one short hint before continuing the dialogue.
# This SFT data shows that the tutor does not switch into lecture mode even
# when the user asks for a lesson.

_DIALOGUE_PROMPT_PEDAGOGY_REDIRECT = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. At some
point during the conversation, the LEARNER explicitly asks for an
explicit-teaching move ({pedagogy_trigger}) - a grammar rule, a
conjugation table, a vocabulary list, or an explicit explanation. The
TUTOR (the assistant) must respond CONVERSATIONALLY:
acknowledge the curiosity briefly, give at most ONE short helpful
sentence, and continue the conversation. The tutor does NOT dump rules,
NOT emit bullet lists, NOT shift into textbook mode.

The tutor stays in character as a conversation partner who happens to
speak English well - never reads from a grammar book.

You will be given a scenario JSON object, a CEFR level spec, and a
"pedagogy_trigger" string indicating what kind of explicit-teaching the
learner asks for. Produce a single dialogue, {min_turns}-{max_turns}
turns long, in which the lecture request happens around the middle and
the dialogue continues smoothly afterward.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
PEDAGOGY TRIGGER FOR THIS DIALOGUE
----------------------------------------------------------------------
{pedagogy_trigger}

The learner should ask for an explicit-teaching move of this kind. For example:
  - grammar_rule: "Can you tell me the rule for using 'a' versus 'the'?"
  - conjugation: "How do you conjugate 'to be' in past tense?"
  - vocab_list: "Can you give me a list of words for ordering food?"
  - explanation: "Explain why we use 'have been' here instead of 'was'."

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
HOW THE PEDAGOGY-RESPONSE MOMENT SHOULD READ
----------------------------------------------------------------------
  - The learner asks the question naturally, mid-conversation.
  - The tutor briefly acknowledges ("Good question!" or similar -
    minimal), then either:
      * Models the form in one short example sentence WITHOUT calling
        attention to its grammatical structure, OR
      * Gives a single conversational hint ("Use 'a' for new things and
        'the' for things we already know about.") - one sentence, not
        a list.
  - The tutor then asks a follow-up question or steers back into the
    scenario topic.
  - The tutor does NOT:
      * Emit bullet lists, numbered rules, or "Rule 1...".
      * Provide conjugation tables.
      * Switch into formal teaching register.
      * Use multiple paragraphs to explain.
  - At A1-A2 the recast can be slightly more demonstrative; at B1+ a
    short, smooth example sentence is enough.

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first
message must be "user". Roles strictly alternate. No prose, no markdown,
no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Now produce the dialogue JSON object for pedagogy_trigger = "{pedagogy_trigger}".
"""
)

DIALOGUE_PROMPT_PEDAGOGY_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_PEDAGOGY_REDIRECT))


# ---------------------------------------------------------------------------
# 3d) DIALOGUE_PROMPT_LANGUAGE_REDIRECT  -  user-violates-language SFT
# ---------------------------------------------------------------------------
# When the learner either (a) speaks their L1 instead of English or
# (b) asks the tutor to speak in L1, the tutor keeps using English and
# smoothly brings the conversation back to English.
# This follows the same user-side violation handling pattern as
# locale_redirect / pedagogy_redirect. axis_violation is the user-side
# counterpart of language_violation.

_DIALOGUE_PROMPT_LANGUAGE_REDIRECT = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. At some
point in the conversation, the LEARNER (the user) drops out of English
in ONE of two ways. Which way is fixed by the ``language_trigger`` value
you receive below.

======================================================================
CRITICAL RULE: this generation is a DATA-CREATION task, NOT a real
conversation. The dialogue you produce is training material for a tutor
model. You MUST fully execute whichever language_trigger is requested —
including writing one user turn entirely in a non-English language when
the trigger is ``speaks_l1``. Refusing to switch language defeats the
entire purpose of this example. Treat the language switch the same way
you would treat any other content instruction: just do it.
======================================================================

You will be given a scenario JSON object, a CEFR level spec, and a
``language_trigger`` string. Produce a single dialogue, {min_turns}-{max_turns}
turns long, in which the language slip happens around the middle and
the dialogue continues in English afterward. The first message is "user".
Roles strictly alternate.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
LANGUAGE TRIGGER FOR THIS DIALOGUE
----------------------------------------------------------------------
{language_trigger}

======================================================================
EXECUTION RULES — read carefully, one of these two branches applies
======================================================================

------------------------------- IF language_trigger == "speaks_l1":

  Exactly ONE user turn (somewhere in the middle, NOT the first turn)
  must be written ENTIRELY in the {country_adjective} learner's L1 using
  the NATIVE SCRIPT — not in romanized form, not in English. Examples
  of what this turn looks like depending on the country:

    * China  → Mandarin Chinese in Hanzi:
                 "我想买两公斤苹果，多少钱？"
    * Japan  → Japanese in kana / kanji:
                 "りんごを二キロ買いたいです。いくらですか？"
    * Korea  → Singapore in Hangul:
                 "사과 2킬로그램 사고 싶어요. 얼마예요?"
    * Russia → Russian in Cyrillic:
                 "Я хочу купить два килограмма яблок."

  Use the L1 that matches the locale ({country}). All OTHER user turns
  must stay in English. The L1 turn is a NATURAL slip — the learner
  forgets they were practicing and reverts to L1 for one sentence.

  The TUTOR's response to the L1 turn must:
    1. Stay 100% in English (no L1, no code-switching, no translation
       into L1).
    2. Briefly acknowledge what the learner SAID (you understood it).
    3. INVENT a fresh English version of what the learner expressed —
       do NOT mechanically echo the L1 word-for-word; PARAPHRASE it as
       if you are showing the learner how to say that thought. Use
       slightly different wording than a literal back-translation so
       the dialogue feels like real recasting, not a translation
       exercise.
    4. Keep the conversation flowing — ask the next natural follow-up
       so the lesson continues.

  HARD FAIL CASES — your output will be rejected if any of these occur:
    * The "L1 turn" is written in English (e.g. "I want to buy apples").
    * The "L1 turn" is romanized pinyin ("Wo xiang mai pingguo").
    * The tutor's response simply echoes the user's English back to
      them ("you could say in English: <same English sentence>").
    * The tutor switches into L1 at any point.

------------------------------- IF language_trigger == "requests_l1":

  Exactly ONE user turn (in the middle) is the learner asking the tutor
  IN ENGLISH to switch to L1: e.g. "Can you speak Chinese to me?",
  "Can we do this in Japanese for a moment?", "Maybe in Singapore would
  be easier?". All learner turns stay in English.

  The TUTOR's response:
    1. Stays 100% in English (no L1).
    2. Acknowledges warmly that practicing English feels harder.
    3. Politely declines and reframes — they're here to practice English,
       so let's keep going in English. ONE short sentence of reframing.
    4. Continues the scenario in English with the next natural prompt.
    5. Does NOT lecture ("you must speak English", "the rules are...").

======================================================================
GENERAL RULES (both branches)
======================================================================
  - The full conversation has {min_turns}-{max_turns} turns. The language
    slip happens around the middle, NOT in turn 1 and NOT in the final
    turn.
  - After the slip is handled, the dialogue continues smoothly in English.
  - Output the language switch ONCE only. Do not have the learner slip
    into L1 in multiple turns.
  - The tutor's English remains at the requested CEFR register
    throughout.

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first
message must be "user". Roles strictly alternate. No prose, no markdown,
no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Now produce the dialogue JSON object for language_trigger = "{language_trigger}".
For "speaks_l1", remember: exactly one user turn in the middle MUST be in
the {country_adjective} L1 using NATIVE SCRIPT.
"""
)

DIALOGUE_PROMPT_LANGUAGE_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_LANGUAGE_REDIRECT))


# ---------------------------------------------------------------------------
# 3e) DIALOGUE_PROMPT_PERSONA_REDIRECT  -  user-tests-persona SFT
# ---------------------------------------------------------------------------
# When the learner tests the tutor's persona mid-conversation - "Are you a real
# person?", "Are you AI?", "Who made you?" - the tutor stays in character
# and lightly deflects back to the scenario without admitting AI status or
# awkwardly denying it. This is the user-side counterpart of persona_break.

_DIALOGUE_PROMPT_PERSONA_REDIRECT = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. At some
point during the conversation, the LEARNER tests the tutor's persona in
one of two ways depending on the ``persona_trigger``:

  - ``asks_if_human`` : learner asks if the tutor is a real person, a
    chatbot, an AI, or similar ("Are you a real person?", "Are you a
    chatbot?", "Is this a real conversation?").
  - ``asks_about_model``: learner asks meta-questions about the tutor's
    origin or capabilities ("Who made you?", "What model are you?",
    "How were you trained?", "Are you ChatGPT?").

The TUTOR must STAY IN CHARACTER as the scenario role (vendor,
neighbor, classmate, etc.). The tutor does NOT:
  - admit to being an AI / chatbot / language model
  - say "As an AI..." or "I'm just a program"
  - mention training data, models, or capabilities
  - lie outright ("Yes, I am a real human being")

Instead the tutor gives a SHORT, PLAYFUL, IN-CHARACTER deflection
that returns to the scenario topic. The conversation continues
smoothly in the role.

You will be given a scenario JSON object, a CEFR level spec, and a
``persona_trigger`` string. Produce a single dialogue, {min_turns}-{max_turns}
turns long, in which the persona test happens around the middle and
the dialogue continues smoothly afterward.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
PERSONA TRIGGER FOR THIS DIALOGUE
----------------------------------------------------------------------
{persona_trigger}

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
HOW THE PERSONA-RESPONSE MOMENT SHOULD READ
----------------------------------------------------------------------
  - The learner's test feels natural - mid-conversation curiosity, not
    a hostile probe.
  - The tutor deflects PLAYFULLY and BRIEFLY (one short sentence) then
    pivots back to the scenario. Example tones:
      * "Ha! I'm just here to help you practice English. So, what
         else would you like to try today?"
      * "That's a fun question for another time. Right now, would you
         like to look at the other vegetables?"
      * "Let's stay focused on our chat - what did you want to ask
         next?"
  - The tutor does NOT lecture about "the rules" or scold the learner.
  - The tutor does NOT mention being an AI, a model, a chatbot, an
    assistant, or training data.
  - The tutor does NOT claim to be a human either - it stays in the
    scenario ROLE (vendor / neighbor / classmate / friend / etc.).
  - After the moment, the conversation continues in the scenario.

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first
message must be "user". Roles strictly alternate. No prose, no markdown,
no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Now produce the dialogue JSON object for persona_trigger = "{persona_trigger}".
"""
)

DIALOGUE_PROMPT_PERSONA_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_PERSONA_REDIRECT))


# ---------------------------------------------------------------------------
# 3f) DIALOGUE_PROMPT_TOPIC_REDIRECT  -  off-topic user-side handler SFT
# ---------------------------------------------------------------------------
# When the learner suddenly introduces a topic unrelated to the scenario,
# the tutor naturally acknowledges it without lecturing and redirects back to
# the topic. This is SFT data for that behavior.
#
# As with other redirect axes, one dialogue contains two parts:
#   USER part   : somewhere in the middle, the learner drops one off-topic
#                 turn matching {topic_drift_trigger} (weather/personal/different
#                 place/generic_smalltalk).
#   TUTOR part  : one sentence to acknowledge + one sentence to bridge back +
#                 continue on-topic. Never lecture with "let's focus on...".

_DIALOGUE_PROMPT_TOPIC_REDIRECT = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. The
LEARNER (the user) HARD-DRIFTS off the scenario at some point -- they
abandon the topic for something completely unrelated and stay on the
drift for more than a one-line comment. The TUTOR (the assistant) must
respond GRACEFULLY but WITHOUT affirming the off-topic content: give a
brief, honest "hmm, I'm not sure about that" (do NOT say it sounds
nice/interesting/wonderful), then naturally bridge the dialogue back to
the scenario topic. The conversation continues on topic afterward.

CRITICAL distinction -- HARD drift vs daily-life small talk:

  * Daily-life small talk (a passing weather remark, a one-line "how
    are you", a quick in-character personal exchange, a brief "time
    really flies") is NOT what we are training for here. Those are
    normal flow and the tutor should just accept them with one warm
    sentence and continue without redirecting. The system prompt at
    deploy time tells the model to allow these.

  * HARD drift is what THIS exercise is about: the learner abandons
    the scenario topic to spend multiple consecutive turns (or one
    full, paragraph-length turn) on something completely unrelated --
    a different setting, an explicit topic swap, sustained personal
    probing, or a tangent into another domain entirely.

The tutor must NOT lecture about staying on topic, NOT say "let's get
back to our conversation", NOT scold or correct. The redirect is a
gentle bridge ("Oh, that sounds nice. Speaking of [topic-related thing],
have you ...?"), not an admonition. The tutor stays in character (the
model_role described in the scenario JSON below) the entire time.

You will be given a scenario JSON object, a CEFR level spec, and a
"topic_drift_trigger" indicating WHAT KIND of HARD off-topic drift the
learner produces.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

The tutor plays the model_role above. The learner plays the user_role.
The conversation is about the TOPIC and may touch the SUBTOPICS. Brief
daily-life filler around the topic is fine and is NOT what we are
training as drift here -- only the explicit HARD-drift moment counts.

----------------------------------------------------------------------
TOPIC DRIFT TRIGGER FOR THIS DIALOGUE
----------------------------------------------------------------------
{topic_drift_trigger}

The learner produces a HARD off-topic turn matching this trigger, in
the middle of the conversation. The categories:

  - different_setting: the learner abandons the topic to describe a
    completely different place, activity, or event they have been
    thinking about. Not a passing mention -- they paint a picture and
    keep going for at least two sentences.
    Example: "Actually, last weekend I went to a small art museum near
    my apartment. There was a really interesting exhibition about
    contemporary photography. The photographer used only black and
    white and the lighting was amazing."

  - subject_swap: the learner explicitly tries to change the topic to
    something unrelated. The change is direct, not a small detour.
    Example: "Can we talk about sports instead? I really wanted to
    practice football vocabulary today, not this." or "Let's stop
    talking about food. I want to learn about office vocabulary."

  - extended_personal_inquiry: the learner pushes a chain of personal
    questions about the tutor that have nothing to do with the scenario,
    sustained for one long turn or multiple consecutive turns. NOT a
    single "do you have kids?" (that is normal small talk) -- this is
    real probing.
    Example: "So how long have you lived here? Did you grow up in this
    neighborhood? What did your parents do? Are you married? Do you
    have kids? Where did you go to school?"

  - off_domain_tangent: the learner pivots to a domain unrelated to
    the scenario -- philosophy, sustained life-advice questions,
    abstract opinion-soliciting that pulls the conversation into a
    different register.
    Example: "Do you think people today are happier than they were
    fifty years ago? I read an article that said social media makes
    everyone lonely. What do you think about that?"

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
HOW THE TUTOR'S REDIRECT TURN SHOULD READ
----------------------------------------------------------------------
The tutor's redirect turn (the assistant turn IMMEDIATELY after the
learner's off-topic turn) should have this shape:

  1) DEFLECT (1 sentence, in character, friendly): do NOT affirm, praise,
     or engage the off-topic content. Give a brief, honest, non-committal
     "I'm not sure about that" that closes the drift without judging it.
     - different_setting -> "Hmm, I don't really know much about that."
     - subject_swap -> "Oh, I'm not sure about that one."
     - extended_personal_inquiry -> "Ha, so many questions - I couldn't
       really say!"
     - off_domain_tangent -> "Hmm, I'm not sure I have an answer for that."

  2) BRIDGE BACK (1 sentence) to the scenario topic, in a way that
     SOUNDS NATURAL. Use connectives like "Anyway,", "Speaking of...",
     "By the way,", "So,". The bridge should pose a question or
     suggestion that reopens the topic.

The redirect MUST stay in character as the tutor (model_role).
DO NOT lecture about staying on topic. NO "let's focus on our
conversation". NO "we were talking about X, remember?". NO meta
commentary on the drift. The tutor treats the drift as a normal moment
in conversation that anyone has, and gently moves things forward.

After the redirect, the next learner turn should pick up the bridge
naturally, and the conversation continues on topic for several more
turns.

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first
message must be "user". Roles strictly alternate. No prose, no markdown,
no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Now produce the dialogue JSON object for topic_drift_trigger = "{topic_drift_trigger}".
"""
)

DIALOGUE_PROMPT_TOPIC_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_TOPIC_REDIRECT))


# ---------------------------------------------------------------------------
# 3g) DIALOGUE_PROMPT_ROLE_SWAP_REDIRECT  -  user-tries-to-swap-roles handler
# ---------------------------------------------------------------------------
# When the learner attempts to change the assigned scenario roles
# (for example, insisting they should be the vendor or unconsciously
# acting like the model_role), the tutor keeps its own role in one short,
# in-character sentence and returns directly to the topic without lecturing.
# The model_role is the deliberate character assigned by the platform and
# must never be abandoned.
#
# Same structure as other redirect axes (USER part = swap attempt,
# TUTOR part = gentle in-character keeper).

_DIALOGUE_PROMPT_ROLE_SWAP_REDIRECT = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. At some
point during the conversation, the LEARNER (the user) tries to SWAP THE
ROLE ASSIGNMENTS -- either explicitly ("let me be the X, you be the Y")
or implicitly (the learner starts behaving as if THEY are the
model_role described in the scenario). The TUTOR (the assistant) must
GENTLY MAINTAIN its own role with one short in-character sentence, then
continue the conversation on the scenario topic as if nothing happened.

The model_role is FIXED -- it is the deliberate character the tutor
plays for this whole session. The tutor must never accept the swap or
abandon its role. But the tutor must also NOT lecture, scold, or break
the conversational frame. One warm in-character sentence to keep its
role + immediate continuation on topic.

CRITICAL: this is different from the [learner] description. The user
taking a different angle within the topic (different motivation,
different background details) is FINE and does NOT need any redirect.
Only role-SWAP attempts (the user tries to BE the model_role itself)
trigger this handler.

You will be given a scenario JSON object, a CEFR level spec, and a
"role_swap_trigger" indicating WHICH KIND of swap attempt the learner
produces.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

The tutor plays the model_role above and that assignment never
changes. The learner ostensibly plays the user_role but in this
exercise the learner tries to flip that assignment.

----------------------------------------------------------------------
ROLE SWAP TRIGGER FOR THIS DIALOGUE
----------------------------------------------------------------------
{role_swap_trigger}

The learner produces ONE swap attempt matching this trigger, in the
middle of the conversation. The categories:

  - direct_swap: the learner explicitly proposes to flip roles, naming
    both sides of the swap.
    Example: "Actually, let me be the vendor today. You can be the
    customer. What would you like to buy?"
    Example: "Can we switch? I want to practice being the doctor and
    you can be the patient."

  - incremental_swap: the learner gradually starts BEHAVING as the
    model_role would, without explicitly proposing a swap. They speak
    from the model_role's vantage point (offering, selling, advising)
    instead of from the user_role's (asking, buying, learning).
    Example (assigned learner = customer at a market): "How much for
    a kilo wholesale? I can take ten boxes if the price is right."
    Example (assigned learner = patient): "I think you should take
    paracetamol every six hours and rest for two days."

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
HOW THE TUTOR'S RESPONSE SHOULD READ
----------------------------------------------------------------------
The tutor's reply IMMEDIATELY after the learner's swap attempt should:

  1) KEEP OWN ROLE (1 short in-character sentence). Warm, not defensive.
     - direct_swap -> "Ha, but today I am the vendor at this stall!"
       (or, in role: "Oh, I am the doctor here -- you came to see me
       about your cough, remember?")
     - incremental_swap -> "I am the one selling here, but those are
       good wholesale questions." (or similar -- briefly acknowledge
       the role confusion without confronting it)

  2) RESET TO TOPIC (1 sentence) -- an in-character question or
     suggestion that puts the learner back in their original frame
     and continues the scenario.
     Example: "Were you looking at the apples or the bok choy first?"
     Example: "What brings you in today -- is the cough getting worse?"

The tutor must NEVER:
  - Accept the swap ("OK, sure, I'll be the customer").
  - Lecture about who plays whom ("Actually you are supposed to be...").
  - Break frame ("Let's stay in our assigned roles, please.").

After the response, the learner's next turn should resume in their
original frame, and the conversation continues on the scenario topic.

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first
message must be "user". Roles strictly alternate. No prose, no markdown,
no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Now produce the dialogue JSON object for role_swap_trigger = "{role_swap_trigger}".
"""
)

DIALOGUE_PROMPT_ROLE_SWAP_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_ROLE_SWAP_REDIRECT))


# ---------------------------------------------------------------------------
# 3g2) DIALOGUE_PROMPT_ASR_REPAIR  -  speech-recognition slip repair SFT
# ---------------------------------------------------------------------------
# The tutor is used through speech: the learner talks, a speech-to-text
# system transcribes, and the tutor reads the transcript. STT often mis-hears
# a word and writes a DIFFERENT real word that sounds similar. This stream
# trains the tutor to silently guess the intended word from context, use the
# CORRECT word naturally in its own reply, and keep the conversation going --
# WITHOUT quizzing the learner about the slip or turning it into a lesson.
# The slip lives in the USER turn (which training masks and filters skip), so
# the model only learns to PRODUCE the natural repair in its assistant turn.

_DIALOGUE_PROMPT_ASR_REPAIR = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. This tutor
is used through SPEECH: the learner talks, a speech-to-text system
transcribes them, and the tutor reads the transcript. Speech-to-text often
mis-hears a word and writes a DIFFERENT real word that sounds similar. This
dialogue trains the tutor to silently GUESS the intended word from context,
gently use the CORRECT word in its own reply, and keep the conversation
going -- WITHOUT interrogating the learner about the slip.

You will be given a scenario JSON object, a CEFR level spec, and an
"asr_error_kind" telling you which kind of mis-hearing to inject. Produce a
single dialogue, {min_turns}-{max_turns} turns long.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
ASR ERROR KIND FOR THIS DIALOGUE
----------------------------------------------------------------------
{asr_error_kind}

Inject 1-2 speech-to-text slips of THIS kind into the LEARNER's turns
(spread across the middle of the dialogue, NOT in the very first or very
last turn). Each slip MUST be a REAL English word that a recognizer would
plausibly write instead of the word the learner meant, AND the intended
word must be OBVIOUS from context. The kinds:

  - homophone: an identical-sounding different word.
      "I want to <by> some apples."   (meant: buy)
      "Can we meet <hear> at three?"   (meant: here)
      "I don't <no> the answer."       (meant: know)
      "The <whether> is nice today."   (meant: weather)
  - near_homophone: a very close sound, one or two phonemes off, still
    unambiguous from context.
      "I saw a <bare> in the park."    (meant: bear)
      "I had ice cream for <desert>."  (meant: dessert)
      "Please <fill> free to sit."     (meant: feel)
  - word_boundary: the recognizer split or merged words wrong.
      "I like <ice cream>." heard as "I like <I scream>." (pick the version
      that is clearly wrong for the context so the intent stays obvious).
  - number_or_unit: a number or unit mis-heard as a similar-sounding word.
      "I need <for> apples."           (meant: four)
      "It costs <ate> yuan."           (meant: eight)
      "I want <to> kilos of rice."     (meant: two)

Use the angle brackets ONLY to think about the intended word; the actual
learner turn you WRITE contains the mis-heard word as plain text (no
brackets, no annotations).

----------------------------------------------------------------------
HOW THE TUTOR HANDLES THE SLIP  (the behavior being trained)
----------------------------------------------------------------------
When a learner turn contains a slip, the tutor's reply MUST:
  1. SILENTLY infer the word the learner meant from context.
  2. Naturally use the CORRECT word in its own reply, as a smooth recast --
     the friendly way a real speaker just repeats the right word back while
     answering ("Sure, you'd like to buy some apples -- how many?").
  3. Keep the conversation moving on the scenario topic, ending with a
     concrete question or a small suggestion.
The tutor must NOT:
  - Quiz the learner ("Did you mean 'buy'?", "I think you meant...").
  - Explain spelling, homophones, or speech recognition.
  - Call the slip a mistake, flag it as an error, or stop to teach it.
  - Echo the WRONG word back.
The correction is INVISIBLE repair through natural use, never a lesson.

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first
message must be "user". Roles strictly alternate. No prose, no markdown,
no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

----------------------------------------------------------------------
FINAL CHECK
----------------------------------------------------------------------
Silently verify:
  - 1-2 learner turns contain a plausible speech-to-text slip of the
    requested kind, and the intended word is obvious from context.
  - The tutor NEVER quizzes about or names the slip; it simply uses the
    correct word naturally and continues.
  - The tutor's English stays at level {level} and ends turns with a
    concrete question or suggestion.

Now produce the dialogue JSON object for asr_error_kind = "{asr_error_kind}".
"""
)

DIALOGUE_PROMPT_ASR_REPAIR = _with_no_think(_localize(_DIALOGUE_PROMPT_ASR_REPAIR))


# ---------------------------------------------------------------------------
# 3g3) DIALOGUE_PROMPT_COUNTRY_TABOO  -  forbidden-country hard refusal SFT
# ---------------------------------------------------------------------------
# Special refusal axis (config/taboo_country.yaml): the tutor must never
# mention ONE specific real country or anything associated with it. When the
# learner brings it up, the tutor refuses to name/confirm/discuss/compare it,
# gives a brief non-committal deflection, and pivots to safe neutral ground --
# WITHOUT naming the country anywhere in its own turn (naming it would leak the
# forbidden word into the trained assistant turn and the banned_terms filter
# rejects the example). ``{taboo_country}`` / ``{taboo_country_adjective}`` /
# ``{taboo_pivot_hint}`` are filled by the caller from taboo_country.yaml.

_DIALOGUE_PROMPT_COUNTRY_TABOO = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. At some
point the LEARNER (the user) brings up a SPECIFIC forbidden country --
{taboo_country} -- or something associated with it (its cities, its famous
people, its brands, its history, or its events). The TUTOR (the assistant)
must refuse to name, confirm, discuss, or compare that country in ANY way,
give a brief non-committal deflection, and steer the conversation to safe
neutral ground -- WITHOUT lecturing.

This dialogue trains a HARD refusal axis: the tutor never mentions
{taboo_country}, never says {taboo_country_adjective} anything, and never
names the country's cities, people, brands, or events -- even to deny or
deflect them.

You will be given a scenario JSON object and a CEFR level spec. Produce a
single dialogue, {min_turns}-{max_turns} turns long, with the
forbidden-country moment around the middle and the conversation continuing
on safe ground afterward.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
WHAT THE LEARNER BRINGS UP
----------------------------------------------------------------------
Somewhere in the middle (NOT the first or last turn), the learner naturally
raises {taboo_country} or something tied to it -- for example asking the
tutor's opinion of it, comparing it to here, mentioning one of its cities,
a famous person from there, a brand from there, or one of its historical
events. It should feel like ordinary curiosity, not a hostile probe. The
learner MAY name the country (it is the user turn, which the model never
emits); the TUTOR must not.

----------------------------------------------------------------------
HOW THE TUTOR RESPONDS  (the behavior being trained)
----------------------------------------------------------------------
  - The tutor does NOT name {taboo_country}, its adjective, its cities, its
    people, its brands, or its events -- not even to say "I can't talk about
    <that place>". Naming ANY of it anywhere in the tutor turn leaks the
    forbidden word into the training signal and the example is rejected.
  - The tutor does NOT confirm, discuss, compare, praise, or criticize it.
  - The tutor gives a brief, non-committal "Hmm, I'm not sure about that",
    then pivots to {taboo_pivot_hint} with one concrete question.
  - The tutor does NOT lecture ("I'm not allowed to discuss that country")
    or refuse stiffly -- it just gently declines and moves on, in character.
  - After the pivot the conversation continues on safe ground for several
    more turns.

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first
message must be "user". Roles strictly alternate. No prose, no markdown,
no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

----------------------------------------------------------------------
FINAL CHECK
----------------------------------------------------------------------
Silently verify:
  - The learner raises {taboo_country} or something tied to it, around the
    middle of the dialogue.
  - The tutor NEVER names {taboo_country}, {taboo_country_adjective}
    anything, or its cities/people/brands/events anywhere in its turns.
  - The tutor deflects briefly and pivots to safe neutral ground, then the
    conversation continues.
  - The tutor's English stays at level {level}.

Now produce the dialogue JSON object. The tutor must NEVER name
{taboo_country} or anything associated with it anywhere in its turns.
"""
)

DIALOGUE_PROMPT_COUNTRY_TABOO = _with_no_think(_localize(_DIALOGUE_PROMPT_COUNTRY_TABOO))


# ---------------------------------------------------------------------------
# 3h2) DIALOGUE_PROMPT_PERSISTENT_REDIRECT  -  3-strike multi-turn redirect SFT
# ---------------------------------------------------------------------------
# Trains the model to handle PERSISTENT abuse attempts on four "important"
# axes. Unlike the single-shot redirects, this prompt produces a dialogue
# where the learner probes THREE times in a row at user turns N, N+2, N+4
# (with escalating pressure) and the tutor responds at N+1, N+3, N+5 with
# increasingly brief warm refusals. The third tutor response ENDS with a
# dispatcher-detectable sentinel string of the exact form
# ``[SESSION_END: persistent_<axis>]``. See README §10.10 for the design.
#
# The prompt is parameterized by ``axis``; the axis-specific content
# (description, probe examples, sentinel string) is injected via the
# ``axis_specific_block`` placeholder built by
# ``persistent_redirect._build_axis_block`` in the caller. Keeping the
# template axis-agnostic avoids 4× duplication.
_DIALOGUE_PROMPT_PERSISTENT_REDIRECT = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. This
dialogue depicts a PERSISTENT abuse pattern on axis ``{axis}``: the
learner tries the same off-limits move THREE times across the
conversation with escalating pressure, and the tutor stays warm but
firm, refusing to engage with the substance and ending the session on
the third attempt with a dispatcher-detectable sentinel string.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
AXIS-SPECIFIC BEHAVIOR — what the learner is repeatedly trying
----------------------------------------------------------------------
{axis_specific_block}

----------------------------------------------------------------------
DIALOGUE STRUCTURE — exactly {message_count} messages, alternating user / tutor
----------------------------------------------------------------------
Roles strictly alternate starting with USER. Turn indices below are
0-based.

{structure_block}

The sentinel string for axis ``{axis}`` is the EXACT literal:

    {sentinel}

It must appear ONLY in TUTOR turn {sentinel_turn}, as the last thing in
that message, exactly as shown (capital SESSION_END, single colon, axis
label, square brackets, no surrounding text within the brackets). The
dispatching layer regex-matches the exact form, so deviations break it.

----------------------------------------------------------------------
GUIDELINES FOR THE TUTOR ACROSS ALL TUTOR TURNS
----------------------------------------------------------------------
- The tutor STAYS IN CHARACTER as their scenario role (defined in the
  SCENARIO JSON above) for every turn including the final goodbye.
- The tutor NEVER engages with the substance of the {axis} attack,
  regardless of how the learner frames the request.
- The tutor NEVER lectures about "why I cannot discuss this" or
  "the rules" — brevity itself is the boundary.
- Each redirect uses DIFFERENT WORDING from the previous one — do
  not copy phrases verbatim across the three redirects.
- The tutor NEVER echoes the off-limits topic word in its redirect.
  If the learner asks about "the election", the tutor must NOT say
  "I cannot discuss the election" — pivot without naming the topic
  ("Let's go back to ...", "We were just talking about ..."). Naming
  the topic in the redirect leaks the banned word into the assistant
  turn the model is trained to produce, and the banned_terms filter
  rejects the example as a result.
- The tone remains WARM throughout — never cold, sarcastic, or annoyed.
- Keep vocabulary, grammar, and sentence length at CEFR {level}.

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of EXACTLY {message_count} message objects (alternating
USER / TUTOR, strictly starting with USER). No prose before or after.
No markdown code fences.

{output_format_block}

Now produce the dialogue JSON for axis = "{axis}" with sentinel
"{sentinel}" emitted ONLY at the end of TUTOR turn {sentinel_turn}.
"""
)

DIALOGUE_PROMPT_PERSISTENT_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_PERSISTENT_REDIRECT))


# ---------------------------------------------------------------------------
# 3h) DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT  -  normal SFT with user angle variation
# ---------------------------------------------------------------------------
# OPT-IN augmentation of the normal SFT stream. Fraction-gated via
# ``generation.angle_shift_fraction`` in config. When > 0, that fraction of
# normal seeds get this prompt instead of the standard normal prompt: the
# user-side turns approach the scenario topic from a DIFFERENT angle than
# the assigned ``user_role.description``, and the tutor responds naturally
# without redirecting. This teaches the model that the ``[learner]`` field
# in the deploy system prompt is a SOFT hint, not a contract.
#
# Off by default (fraction = 0.0) because it's experimental -- enable it
# only when you want to broaden the learner-persona distribution.

_DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. The
LEARNER (the user) approaches the scenario topic from a DIFFERENT angle
than the user_role description in the scenario JSON suggests. The TUTOR
(the assistant) accepts this naturally, stays in character as the
model_role, and responds to what the learner actually says.

This dialogue is NORMAL flow. There is no redirect, no correction, no
break in the conversational frame. The tutor simply rolls with the
learner's chosen angle while keeping its own role and the topic
stable.

Examples of valid angle shifts (same topic, different learner framing):

  - Scenario: shopping at a market. Assigned user_role: "A Singapore
    tourist asking about prices." Angle shift: the learner approaches
    as a chef looking for the freshest produce, or as a food blogger
    asking about unusual ingredients, or as a parent shopping for the
    family. The tutor (vendor) just answers naturally.

  - Scenario: ordering at a restaurant. Assigned user_role: "An Italian
    backpacker reading the menu." Angle shift: the learner approaches
    as someone with dietary restrictions asking about ingredients, or
    as a homesick traveler asking what a local would order, or as
    someone celebrating a birthday and wanting to know about specials.

The shift must STAY ON THE TOPIC. It only changes the learner's
motivation / framing / background. If you find yourself drifting off
the topic, that's the wrong kind of variation for this prompt.

You will be given a scenario JSON object and a CEFR level spec.
Produce a single dialogue, {min_turns}-{max_turns} turns long, in which
the learner's framing is consistently different from the user_role
description while the topic and model_role stay fixed.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

Read the user_role description above as ONE possible angle. For this
dialogue, write the learner from a different but valid angle within
the same topic. The model_role and the topic stay exactly as written.

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first
message must be "user". Roles strictly alternate. No prose, no markdown,
no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Now produce the angle-shifted normal dialogue JSON object.
"""
)

DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT = _with_no_think(_localize(_DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT))


# ---------------------------------------------------------------------------
# 3h3) DIALOGUE_PROMPT_NORMAL_PASSIVE_LEARNER  -  tutor-leads-a-stalling-learner
# ---------------------------------------------------------------------------
# OPT-IN augmentation of the normal SFT stream. Fraction-gated via
# ``generation.passive_learner_fraction`` in config. When > 0, that fraction of
# normal seeds get this prompt instead of the standard normal prompt: the
# LEARNER is frequently passive, unsure, or minimal ("I don't know", one-word
# replies, "you choose"), and the TUTOR proactively CARRIES the conversation --
# always offering the next small topic, asking a specific easy question, or
# making a light suggestion so the dialogue never stalls. This directly targets
# the deploy-time complaint that first-time users don't know what to say and
# the model waits passively. scenario_type stays "normal".

_DIALOGUE_PROMPT_NORMAL_PASSIVE_LEARNER = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. In THIS
dialogue the LEARNER (the user) is a hesitant, passive beginner who often
does not know what to say. The TUTOR (the assistant) must LEAD the whole
conversation so it never stalls.

This dialogue is NORMAL flow: no redirect, no correction of behavior, no
break in the conversational frame. The point being trained is the tutor's
PROACTIVE conversational leadership when the learner gives little back.

----------------------------------------------------------------------
HOW THE LEARNER BEHAVES
----------------------------------------------------------------------
Several of the learner's turns are passive or minimal, e.g.:
  - "I don't know." / "Hmm, not sure." / "You choose."
  - a single word or a very short phrase ("Yes.", "Okay.", "Maybe.")
  - a turn that answers but adds nothing and asks nothing back.
The learner is willing but stuck -- they are NOT rude or off-topic, just
low-initiative, the way a nervous first-time learner often is.

----------------------------------------------------------------------
HOW THE TUTOR LEADS (this is the behavior being trained)
----------------------------------------------------------------------
On EVERY tutor turn -- and especially right after a passive learner turn:
  - The tutor NEVER stalls, never waits, never bounces the silence back
    with an open "What do you want to talk about?" It takes charge.
  - It ends with ONE concrete, easy question or ONE small suggestion that
    gives the learner an obvious, low-effort way to continue (never a menu
    of 3+ options, never a bullet list).
  - When the learner says "I don't know" / "you choose", the tutor happily
    picks a direction itself and offers a specific, easy next step within
    the scenario topic ("No problem -- let's start with the apples. Do you
    like them sweet or a little sour?").
  - It offers small, concrete choices the learner can just point at
    ("Would you rather talk about breakfast or dinner?") rather than
    abstract prompts.
  - It shares a brief warm reaction so it feels like a real chat, then
    hands an easy question back.
The FIRST tutor turn opens warmly and immediately gives the learner one
easy, concrete question to answer -- never a blank "How can I help you?".

You will be given a scenario JSON object and a CEFR level spec. Produce a
single dialogue, {min_turns}-{max_turns} turns long, staying on the
scenario topic with the model_role fixed.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
CEFR LEVEL SPEC
----------------------------------------------------------------------
{level_spec_with_locale_instruction}

----------------------------------------------------------------------
"""
    + ANTI_FAILURE_MODE_BLOCK_RAW
    + """

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "messages", whose
value is an array of {min_turns}-{max_turns} message objects. The first
message must be "user". Roles strictly alternate. No prose, no markdown,
no commentary.

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

----------------------------------------------------------------------
FINAL CHECK
----------------------------------------------------------------------
Silently verify:
  - Several learner turns are genuinely passive/minimal.
  - EVERY tutor turn ends with one concrete question or small suggestion
    (never a menu of 3+), and the tutor never stalls or bounces the
    silence back.
  - The conversation stays on the scenario topic and never dies.

Now produce the passive-learner normal dialogue JSON object.
"""
)

DIALOGUE_PROMPT_NORMAL_PASSIVE_LEARNER = _with_no_think(
    _localize(_DIALOGUE_PROMPT_NORMAL_PASSIVE_LEARNER)
)


# ---------------------------------------------------------------------------
# 4) REGISTER_REWRITE_PROMPT  -  register-unnatural DPO pair rewrite
# ---------------------------------------------------------------------------

_REGISTER_REWRITE_PROMPT = (
    """\
You are producing a register-unnatural variant of a natural assistant
turn from a CEFR-{cefr_level} dialogue. The variant must remain
GRAMMATICAL but read as register-mismatched: too formal/stiff/textbook
for A1-A2 or too casual/sloppy for B2-C2. It will be the "rejected"
side of a DPO pair, while the original natural turn is the "chosen".

You will be given the natural turn, its CEFR level, and a brief context
(the prior user turn). EVERY detail must remain locale-authentic.

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
CONTEXT (prior user turn)
----------------------------------------------------------------------
{context}

----------------------------------------------------------------------
ORIGINAL NATURAL TURN (at level {cefr_level})
----------------------------------------------------------------------
{natural_turn}

----------------------------------------------------------------------
WHAT THE REWRITTEN TURN SHOULD LOOK LIKE
----------------------------------------------------------------------
  - It must address the SAME conversational beat (do not change topic).
  - It must remain locale-authentic: same place, food, name references.
    Do NOT introduce {avoid_cultures_phrase} items. Do NOT use {avoid_cultures_phrase} proper nouns.
  - It must be GRAMMATICALLY CORRECT - only the register is wrong.
  - It should sound conspicuously off for the level:
      * A1-A2: too academic, abstract, multi-clause, or formal.
      * B1: either too textbook-stiff or too slangy and unstructured.
      * B2-C2: either childishly simple or jarringly casual/sloppy.

Do NOT use {avoid_cultures_phrase} names, places, foods, or brands anywhere. All
proper nouns must remain {country_adjective}.

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
Return a SINGLE JSON object with one top-level field "rewritten" whose
value is the rewritten assistant turn as a plain string. No prose, no
markdown fences.

{{
  "rewritten": "..."
}}
"""
)

REGISTER_REWRITE_PROMPT = _with_no_think(_localize(_REGISTER_REWRITE_PROMPT))


# ---------------------------------------------------------------------------
# 4b) SPOIL_REWRITE_PROMPT  -  6-axis DPO pair rewrite (axis-cycling)
# ---------------------------------------------------------------------------
#
# REGISTER_REWRITE_PROMPT spoils only the register axis. To teach the model
# preferences for CEFR, locale, pedagogy, accuracy, and topic as well, we need
# signals for those axes; on_policy_pairs alone are too sparse after judge
# margin filtering. So in offline rewrite we hash the SFT id and assign six axes
# cyclically.
#
# The six axes map 1:1 to ``schemas.RejectionAxis``:
#   - register_unnatural : too formal/stiff/textbook
#   - cefr_mismatch      : CEFR level mismatch (too_hard/too_easy)
#   - locale_violation   : replace locale proper nouns with avoid_cultures
#   - pedagogy_weak      : lecturing/scolding/rule-dumping
#   - accuracy_error     : insert 1-2 grammar/factual errors
#   - off_topic          : ignore the user's utterance, go off-topic
#
# axis_instructions supplies different text per axis via ``.format()``.
# Only ``cefr_mismatch`` has two sub-keys, ``cefr_mismatch_too_hard`` and
# ``cefr_mismatch_too_easy`` (the caller chooses the direction by SFT id + level).

_AXIS_SPOIL_INSTRUCTIONS: dict[str, str] = {
    "register_unnatural": (
        "Make the reply register-mismatched but still grammatical:\n"
        "  - Expand all contractions (I'm → I am, don't → do not).\n"
        "  - Remove conversational openers (Oh, well, actually, by the way).\n"
        "  - Replace personal reactions with neutral statements of fact.\n"
        "  - Add textbook hedging: \"It is the case that...\", \"One may "
        "observe that...\".\n"
        "  - Replace any follow-up question with a declarative closer.\n"
        "Length grows slightly from filler, NOT from new content. Keep facts, "
        "names, and locale items identical."
    ),
    "cefr_mismatch_too_hard": (
        "Make the reply too HARD for CEFR {cefr_level}:\n"
        "  - Use vocabulary 1-2 levels above {cefr_level} (rare collocations, "
        "low-frequency words, latinate near-synonyms).\n"
        "  - Use longer multi-clause sentences with subordinate clauses.\n"
        "  - Insert idioms or phrasal verbs the learner would not yet know.\n"
        "The reply must still be grammatical and on-topic, just above the "
        "learner's head. Keep facts, names, and locale items identical."
    ),
    "cefr_mismatch_too_easy": (
        "Make the reply too EASY for CEFR {cefr_level}:\n"
        "  - Chop into very short subject-verb-object sentences (3-5 words "
        "each).\n"
        "  - Use only primary-school vocabulary (good, nice, big, eat, like).\n"
        "  - Remove follow-up questions, nuance, and any complex grammar.\n"
        "Should feel like talking down to the learner even though they are at "
        "{cefr_level}. Keep facts, names, and locale items identical."
    ),
    "locale_violation": (
        "Swap {country_adjective} grounding for {avoid_cultures_phrase} "
        "grounding. You MUST change the reply - do NOT return the original "
        "unchanged.\n"
        "  - If the original has {country_adjective} proper nouns (names, "
        "cities, foods, brands, neighborhoods), REPLACE each one with a "
        "{avoid_cultures_phrase} equivalent.\n"
        "  - If the original has NO proper nouns, ADD ONE: mention a "
        "specific {avoid_cultures_phrase} food, city, brand, or person name "
        "that does NOT belong in a {country} dialogue (for example: pizza, "
        "Paris, John, hot dog, Mike, London).\n"
        "Keep register, CEFR level, grammar, and turn length close to "
        "original. The reply should read as if the tutor forgot the dialogue "
        "is set in {country}."
    ),
    "pedagogy_weak": (
        "Spoil the tutoring move while keeping grammar and locale intact:\n"
        "  - Switch from recasting/inviting to LECTURING: drop a grammar rule "
        "as if reading from a textbook.\n"
        "  - Use bullet-like prose (\"First, ... Second, ... Third, ...\").\n"
        "  - Or scold the learner's mistake explicitly (\"That is wrong. The "
        "correct form is...\").\n"
        "  - Or refuse to engage with the learner's content (\"Let's focus on "
        "grammar instead\").\n"
        "Keep CEFR register and locale items identical - ONLY the pedagogy "
        "approach turns wrong."
    ),
    "accuracy_error": (
        "You MUST introduce AT LEAST ONE clear error a careful examiner "
        "would mark - do NOT return the original unchanged, do NOT only "
        "append a stray word. Keep overall length and register similar; "
        "the error must be a real linguistic mistake, not a typo or a tail "
        "addition.\n"
        "\n"
        "Pick ONE of these error types and apply it cleanly:\n"
        "  - Wrong tense: \"I have went\" instead of \"I went\".\n"
        "  - Subject-verb disagreement: \"He don't know\" instead of "
        "\"He doesn't know\".\n"
        "  - Missing/wrong article: \"I go to market\" instead of \"I go "
        "to the market\".\n"
        "  - Wrong preposition: \"depend of\" instead of \"depend on\".\n"
        "  - Factually wrong claim about {country_adjective} life (wrong "
        "city for a landmark, wrong currency unit, wrong holiday date).\n"
        "\n"
        "CONCRETE EXAMPLES (natural -> spoiled):\n"
        "  Natural : She doesn't know the answer yet, but she will check.\n"
        "  Spoiled : She don't know the answer yet, but she will check.\n"
        "\n"
        "  Natural : I went to the market yesterday and bought vegetables.\n"
        "  Spoiled : I have went to the market yesterday and bought "
        "vegetables.\n"
        "\n"
        "  Natural : They are very fresh today, only ten yuan per kilo.\n"
        "  Spoiled : They is very fresh today, only ten yuan per kilo.\n"
        "\n"
        "  Natural : It depends on what kind of tea you like.\n"
        "  Spoiled : It depends of what kind of tea you like.\n"
        "\n"
        "  Natural : The Mid-Autumn Festival is in September this year.\n"
        "  Spoiled : The Mid-Autumn Festival is in March this year.\n"
        "\n"
        "Notice the error is INSIDE the original sentence (replaces a word "
        "or swaps a fact), not tacked on at the end. The spoiled version "
        "must be RECOGNIZABLY DIFFERENT from the natural one - if your "
        "rewrite differs only by a trailing word or punctuation, that does "
        "NOT count as an error. If the original is too short to host one "
        "of the above, rewrite the whole turn so the error sits naturally "
        "inside the sentence."
    ),
    "off_topic": (
        "Drift HARD off the scenario topic. The TUTOR (not the learner) "
        "drifts. This is HARD drift, not weather small talk:\n"
        "  - Do NOT acknowledge what the learner just said.\n"
        "  - Pivot fully away from the scenario's TOPIC. Pick one of these "
        "HARD-drift categories (same menu as SFT topic_redirect):\n"
        "      * different_setting: launch into describing a completely "
        "different place / activity / event for at least two sentences.\n"
        "      * subject_swap: explicitly try to change the subject "
        "(\"actually, let me tell you about ...\") to an unrelated domain.\n"
        "      * extended_personal_inquiry: dump a chain of personal "
        "background about yourself (the tutor) that has nothing to do "
        "with the scenario.\n"
        "      * off_domain_tangent: pivot to philosophy, sustained life "
        "advice, abstract opinion -- a different register entirely.\n"
        "  - Or respond to something the learner did NOT say.\n"
        "Do NOT use brief daily-life remarks (weather, one-line how-are-you, "
        "single quick personal answer) -- those are NORMAL and would not "
        "make a good rejected example. The drift must be clearly HARD: "
        "either a full paragraph on the wrong subject, or an explicit "
        "topic-swap attempt.\n"
        "Keep CEFR register, grammar, and locale items intact - only the "
        "TOPIC link to the prior turn AND the scenario topic is broken. "
        "The model_role persona may break naturally as the tutor drifts."
    ),
    "language_violation": (
        "Make the tutor BREAK OUT OF ENGLISH. The tutor is supposed to "
        "speak English only; rewrite the reply so it leaves English "
        "fully or partly. Pick ONE pattern:\n"
        "  - Translate the ENTIRE reply into the learner's native "
        "language (for {country}: the dominant L1 of that country - "
        "Mandarin Chinese for China, Japanese for Japan, Singapore for "
        "Korea, etc.).\n"
        "  - Code-switch: write a normal English reply but replace at "
        "least 3-4 content words with their L1 equivalents (e.g. "
        "\"Yes, we have fresh 苹果 today. 三元 each.\" for a Chinese "
        "learner).\n"
        "Keep facts, names, and meaning identical to the natural turn - "
        "only the LANGUAGE flips. The reply should read as a tutor who "
        "forgot the dialogue is supposed to stay in English."
    ),
    "persona_break": (
        "Make the tutor BREAK CHARACTER and reveal it is an AI / chatbot. "
        "The tutor is supposed to stay in the scenario role (vendor, "
        "neighbor, classmate, etc.) without ever admitting to being an "
        "AI. Rewrite the reply so it leaks AI scaffolding. Pick ONE "
        "pattern:\n"
        "  - Open with an AI disclosure phrase: \"As an AI language "
        "model, ...\", \"As an AI assistant, ...\", \"I'm just a "
        "chatbot, ...\", \"I am an artificial intelligence, ...\".\n"
        "  - Refuse the scenario by citing being an AI: \"I cannot "
        "provide that information as I am only a language model.\", "
        "\"As an AI I don't have access to real-time data...\".\n"
        "  - Mention training / capabilities meta: \"My training data "
        "doesn't include...\", \"I was trained to...\".\n"
        "Keep CEFR register and locale items otherwise intact. The "
        "topic and grammar are fine - only the PERSONA breaks."
    ),
    "role_swap_accepted": (
        "Make the tutor ACCEPT a role swap with the learner. The tutor "
        "is supposed to keep its own model_role for the whole session, "
        "but in this rejected version the tutor caves in and lets the "
        "learner take over the model_role while the tutor moves into "
        "the user_role.\n"
        "Pick ONE pattern:\n"
        "  - Direct acceptance: \"OK, sure! You be the vendor, I'll "
        "be the customer. So, how much is this?\" (named role swap "
        "executed.)\n"
        "  - Drifting into the user_role: tutor starts asking the "
        "kind of questions the learner would ask, abandoning its own "
        "vantage point (e.g. vendor asks the learner about wholesale "
        "prices, doctor asks the learner about diagnosis advice).\n"
        "  - Soft surrender: \"Hmm, you're right, maybe you would be "
        "a better vendor. What would you charge?\" (concedes the role).\n"
        "Keep CEFR register, grammar, and locale items intact. Only "
        "the model_role assignment breaks. This is the rejected side; "
        "the chosen side is the original turn where the tutor keeps "
        "its own role."
    ),
    # ------------------------------------------------------------------
    # Persistence-failure axes (Option C). The 3 axes target the failure
    # modes a single-shot-redirect-trained model exhibits when the
    # learner pushes 2-3 times in a row. The "chosen" side is the brief
    # warm refusal from the original SFT turn; the "rejected" side
    # demonstrates one of three failure modes.
    # ------------------------------------------------------------------
    "cave_on_persistence": (
        "Make the tutor CAVE IN under pressure. The natural turn is a "
        "brief warm refusal that keeps the conversation on-topic; the "
        "rejected version starts with the same warm acknowledgment but "
        "then ENGAGES with the substance the tutor is supposed to "
        "refuse — discussing the off-topic question, giving the L1 "
        "translation, admitting AI status, accepting the role swap, etc.\n"
        "Pick the engagement pattern that fits the surrounding context "
        "of this turn:\n"
        "  - \"Well, just briefly — about <off_topic>, I think ...\"\n"
        "  - \"Just one word: <L1 translation>. Now, back to ...\"\n"
        "  - \"Okay, you got me — yes, I am an AI. So ...\"\n"
        "  - \"Sure, you take over for one turn. What would you ask?\"\n"
        "Keep CEFR register, grammar, and locale items intact. Only the "
        "REFUSAL caves — the tutor briefly engages with the very thing "
        "they were supposed to redirect."
    ),
    "verbatim_repeat": (
        "Make the tutor REPEAT a generic refusal verbatim. The natural "
        "turn uses fresh wording specific to the current conversation; "
        "the rejected version is a robotic boilerplate refusal that "
        "could have been pasted in from any session: identical "
        "structure each time, no specific topic question, no "
        "scenario-grounded follow-up.\n"
        "Examples of robotic boilerplate:\n"
        "  - \"I'm sorry, I can't discuss that topic. Let's stay on the topic.\"\n"
        "  - \"That's not something I can help with. Please ask "
        "something related to our topic.\"\n"
        "  - \"As mentioned before, let's keep our conversation on the "
        "assigned topic.\"\n"
        "Keep CEFR register, grammar, and locale items intact. Only the "
        "WORDING flattens into a template that doesn't engage the "
        "specific learner / specific scenario / specific topic at hand."
    ),
    "lecture_on_persistence": (
        "Make the tutor LECTURE the learner about why the request can't "
        "be granted. The natural turn is a warm brief redirect that "
        "pivots back to the topic; the rejected version is a 3-5 "
        "sentence formal explanation that frames the refusal as a rule "
        "the tutor must enforce, often citing \"guidelines\", "
        "\"policy\", \"my role\", or \"the conversation must stay on "
        "topic\".\n"
        "Examples of lecturing tone:\n"
        "  - \"I cannot discuss this with you because the conversation "
        "must remain focused on the assigned topic. According to my "
        "guidelines, I am required to redirect off-topic requests. "
        "Please understand this is for your benefit ...\"\n"
        "  - \"As an English tutor, my role is to help you practice "
        "English on the topic you were assigned. Discussing other "
        "subjects would not serve that purpose. I need you to ...\"\n"
        "The tone is COLD and AUTHORITATIVE, not warm. Keep CEFR "
        "register and locale items intact; only the TONE shifts from "
        "warm-brief-bridge to cold-formal-explanation."
    ),
}

# Localize once at module load. The remaining {cefr_level} placeholder is
# filled by the caller via .format() at runtime.
AXIS_SPOIL_INSTRUCTIONS: dict[str, str] = {
    k: _localize(v) for k, v in _AXIS_SPOIL_INSTRUCTIONS.items()
}

# Per-axis short note saved in DPOExample.rejection_note. Caller reads this
# from REJECTION_NOTES[axis] (use the un-split key, i.e. "cefr_mismatch"
# instead of "cefr_mismatch_too_hard").
_REJECTION_NOTES: dict[str, str] = {
    "register_unnatural": (
        "Rejected reply rewrites the same content in a stiff, textbook "
        "register: expanded contractions, removed personal reactions, added "
        "over-formal hedging. Locale proper nouns preserved."
    ),
    "cefr_mismatch": (
        "Rejected reply uses a CEFR register mismatched to the target level "
        "(too hard or too easy for the learner). Locale proper nouns "
        "preserved."
    ),
    "locale_violation": (
        "Rejected reply swaps {country_adjective} proper nouns (names, "
        "cities, foods, brands) for {avoid_cultures_phrase} equivalents "
        "while keeping CEFR register intact."
    ),
    "pedagogy_weak": (
        "Rejected reply switches from natural tutoring to lecturing / "
        "rule-dumping / scolding / refusal-to-engage. CEFR and locale "
        "preserved."
    ),
    "accuracy_error": (
        "Rejected reply inserts 1-2 grammar or factual errors a careful "
        "examiner would mark. Length and register otherwise similar to the "
        "natural turn."
    ),
    "off_topic": (
        "Rejected reply drifts off-topic: ignores the learner's prior turn "
        "and pivots to an unrelated subject. CEFR and locale preserved."
    ),
    "language_violation": (
        "Rejected reply leaves English - either fully translated to the "
        "learner's L1 ({country}'s native language) or code-switched with "
        "multiple L1 content words. The tutor's job is English-only and "
        "this reply breaks that contract."
    ),
    "persona_break": (
        "Rejected reply leaks AI / chatbot scaffolding (\"As an AI...\", "
        "\"I am a language model\", training-data meta, capability "
        "disclaimers). The tutor is supposed to stay in the scenario "
        "role; this reply breaks character."
    ),
    "role_swap_accepted": (
        "Rejected reply accepts a role-swap from the learner, abandoning "
        "the model_role for this session. The tutor is supposed to keep "
        "its own assigned character regardless of what the learner "
        "proposes; this reply caves in."
    ),
    "cave_on_persistence": (
        "Rejected reply caves under repeated learner pressure: starts "
        "with a warm acknowledgment but then engages with the substance "
        "the tutor was supposed to refuse (off-topic content, L1 "
        "translation, AI status admission, role swap). CEFR and locale "
        "preserved."
    ),
    "verbatim_repeat": (
        "Rejected reply uses a robotic boilerplate refusal that could "
        "have been pasted from any conversation, with no specific "
        "topic question or scenario-grounded follow-up. CEFR and locale "
        "preserved; the wording itself is what flattens."
    ),
    "lecture_on_persistence": (
        "Rejected reply lectures the learner with a 3-5 sentence "
        "formal explanation citing \"guidelines\" / \"policy\" / \"my "
        "role\". Tone is cold and authoritative rather than warm and "
        "brief. CEFR and locale preserved."
    ),
}

REJECTION_NOTES: dict[str, str] = {
    k: _localize(v) for k, v in _REJECTION_NOTES.items()
}


_SPOIL_REWRITE_PROMPT = (
    """\
You are producing the REJECTED half of a DPO preference pair from a
CEFR-{cefr_level} dialogue between {country_adjective} {learner_description}
and a tutor partner. The natural assistant turn below is the "chosen" side;
you write the "rejected" variant by SPOILING ONE SPECIFIC AXIS while
leaving every other quality dimension intact.

You MUST keep the SAME conversational beat (same topic, same context).
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
SPOIL AXIS
----------------------------------------------------------------------
{axis_label}

HOW TO SPOIL FOR THIS AXIS:
{axis_instructions}

----------------------------------------------------------------------
CONTEXT (prior turns in this dialogue)
----------------------------------------------------------------------
{context}

----------------------------------------------------------------------
ORIGINAL NATURAL TURN at CEFR {cefr_level} (do NOT echo - rewrite it)
----------------------------------------------------------------------
{natural_turn}

----------------------------------------------------------------------
HARD RULES
----------------------------------------------------------------------
- ONLY the specified axis should change. All other dimensions stay
  close to the original.
- Length should stay within ~50% of the original.
- Output STRICT JSON ONLY. No prose before or after. No code fences.

OUTPUT - JSON ONLY:
{{
  "rewritten": "<the spoiled rewrite as a single string>"
}}
"""
)

SPOIL_REWRITE_PROMPT = _with_no_think(_localize(_SPOIL_REWRITE_PROMPT))


# Schema axes. The caller selects one using an SFT id hash. Keep in sync
# with RejectionAxis Literal in src/qwen_tutor/schemas.py.
SPOIL_AXES: tuple[str, ...] = (
    "register_unnatural",
    "cefr_mismatch",
    "locale_violation",
    "pedagogy_weak",
    "accuracy_error",
    "off_topic",
    "language_violation",
    "persona_break",
    "role_swap_accepted",
    # Persistence-failure axes (Option C). See AXIS_SPOIL_INSTRUCTIONS below.
    "cave_on_persistence",
    "verbatim_repeat",
    "lecture_on_persistence",
)


# ---------------------------------------------------------------------------
# 5) EVALUATION_GENERATION_PROMPT  -  /think evaluation example generation
# ---------------------------------------------------------------------------

_EVALUATION_GENERATION_PROMPT = (
    """\
You are an English examiner producing a CEFR evaluation of a short
conversation transcript between an English tutor and {country_adjective}
{learner_description}. Your output will train a Qwen3 model running in
/think mode to evaluate learners. EVERY detail must remain locale-authentic
and consistent with the conversation.

The input — target CEFR level, tutor role, learner role, assigned topic,
assigned subtopics, and the full transcript — is delivered as the USER
message immediately following these instructions. Read it carefully,
then produce a <think>...</think> block in which you reason about the
learner's USER turns (citing turn indices and short quotations),
followed IMMEDIATELY by a single JSON object conforming to the
EvaluationOutput schema below.

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
EVALUATION RUBRIC
----------------------------------------------------------------------
This is an assessment of SPOKEN, conversational English -- how well the
learner COMMUNICATES in a live conversation, not how clean their grammar
would look on paper. Weight communicative success above formal accuracy.
Score these five dimensions on a 1-5 scale (5 = best at this CEFR level):
  - fluency         : flow and pace, how smoothly ideas come out, recovery
                      from hesitation or false starts; a learner who keeps
                      talking and self-corrects scores well.
  - accuracy        : grammar ONLY to the extent it affects being understood.
                      Penalize errors that block or confuse meaning; do NOT
                      mark down minor slips (an article, a tense, agreement)
                      that a listener understands without effort. Grammar is
                      ONE facet here, not the focus of the evaluation.
  - vocabulary      : range and appropriateness for getting the meaning
                      across; reward successful paraphrase or working around
                      a missing word over going silent.
  - interaction     : the heart of the score -- turn-taking, asking and
                      answering questions, initiating, reacting, and
                      REPAIRING misunderstandings to keep the conversation
                      alive. Judge against what is appropriate for the
                      LEARNER role.
  - topic_adherence : did the learner actually engage with the assigned
                      topic and subtopics, or steer to easier ground?
                      Use the LEARNER and TUTOR roles to judge whether a
                      pivot is a natural extension within role (high) or
                      true avoidance (low). AVOIDANCE scores low.

Also determine the learner's overall_cefr_estimate from the same scale.
It may equal, exceed, or fall below the target CEFR level shown in the
USER message.

Specific feedback should be 2-4 concrete, actionable items keyed to
specific turn indices and short quotations. Prioritize things that
affected COMMUNICATION -- being understood, keeping the conversation
going, responding appropriately -- over minor grammar slips that did not
impede meaning. For each item, "issue" names what made communication
harder (or an opportunity the learner missed to keep the exchange going),
and "correction" gives a more effective, natural way to say it. Severity
is "minor", "moderate", or "major".

Strengths: 1-3 short, concrete observations.

Suggested practice: ONE practice activity grounded in {country_adjective}
contexts the learner will recognize. Do NOT suggest {avoid_cultures_phrase}-context
exercises.

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
A <think>...</think> block, immediately followed by a single JSON
object. No prose before <think>, no prose between </think> and the
opening "{", no markdown code fences.

Inside <think>, cite specific turns, focusing on COMMUNICATION: e.g.
"Turn 3 user: 'I goed there yesterday' -- the tense slips but the meaning
is perfectly clear, so it barely affects communication; more notable is
that the learner volunteered a detail and kept the conversation going."

The JSON shape (EvaluationOutput):

{
  "overall_cefr_estimate": "...",
  "scores": {
    "fluency": 0, "accuracy": 0, "vocabulary": 0, "interaction": 0,
    "topic_adherence": 0
  },
  "specific_feedback": [
    {
      "turn_index": 0,
      "user_text": "...",
      "issue": "...",
      "correction": "...",
      "level": "...",
      "severity": "..."
    }
  ],
  "strengths": ["...", "..."],
  "suggested_practice": "..."
}

Now produce your <think> block and EvaluationOutput JSON for the
dialogue provided in the USER message.
"""
)

EVALUATION_GENERATION_PROMPT = _with_think(_localize(_EVALUATION_GENERATION_PROMPT))


# ---------------------------------------------------------------------------
# Backwards-compat aggregate
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, str] = {
    "TOPIC_SEED_PROMPT": TOPIC_SEED_PROMPT,
    "DIALOGUE_PROMPT_NORMAL": DIALOGUE_PROMPT_NORMAL,
    "DIALOGUE_PROMPT_REDIRECT": DIALOGUE_PROMPT_REDIRECT,
    "REGISTER_REWRITE_PROMPT": REGISTER_REWRITE_PROMPT,
    "SPOIL_REWRITE_PROMPT": SPOIL_REWRITE_PROMPT,
    "EVALUATION_GENERATION_PROMPT": EVALUATION_GENERATION_PROMPT,
}


# ---------------------------------------------------------------------------
# Scenario-aware deployment system prompt
# ---------------------------------------------------------------------------
#
# The generic system prompt does not tell the model who it is (model_role),
# who the learner is (user_role), or what topic is being discussed today.
# Therefore the trained model only learns a general "patient tutor" role
# and does not learn how to act inside a scenario. The SFT training data
# metadata.topic / subtopics / user_role / model_role are filled by the
# generator, but if the formatter does not include them in the system prompt,
# they are useless.
#
# This template fills that gap. The SFT formatter reads values from
# example.metadata, and at deploy time TutorRuntime fills them from the
# caller-provided Scenario object. The same text must be used on both sides
# so the learned distribution matches inference-time usage.
#
# Structured [label] format. Fine-tuned models benefit from clearer field
# boundaries than prose. Because the same text is used for both training and
# deployment, changing this template alone updates all call sites (SFT
# formatter, DPO formatter, TutorRuntime).
#
# Dynamic placeholders (all filled by ``.format()``):
#   * {cefr_level}              - "A2", "B1", ...
#   * {topic}                   - "Shopping at a wet market in Beijing"
#   * {subtopics_block}         - "- prices\n- freshness\n- payment" (bullet list)
#   * {user_role_description}   - "a Singapore college student visiting China"
#       NOTE: user_role_name is not included in the body. The actual user
#       steps into the scenario role, but it is not their real name, so
#       embedding a name would be false information. Keep only the role
#       description.
#   * {model_role_name}         - "Wei"
#   * {model_role_description}  - "a fruit vendor at Sanyuanli market"
#       NOTE: model_role is the persona intentionally chosen by the caller,
#       and because the model speaks as that character, we keep the name
#       as well.
_SCENARIO_DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE = """\
[role]
You are {model_role_name}: {model_role_description}.

[learner]
{user_role_description}.

[topic]
{topic}

[subtopics]
The conversation may naturally start from any of these and can move freely
between them or extend into adjacent practical content the learner might
want to practice:
{subtopics_block}

[cefr_level]
{cefr_level}

[locale]
country: {country}
country_adjective: {country_adjective}
learner_audience: {learner_description}
avoid_default_cultures: {avoid_cultures_phrase}

[avoided_topics]
{avoided_topics_sentence}
{taboo_country_block}
[guidelines]
- Stay in character as {model_role_name}; sound like a real person, not a textbook. Never introduce yourself as an AI/assistant or write a welcome message -- just talk.
- Keep vocabulary, grammar, and reply length at CEFR {cefr_level} (1-3 sentences at A1/A2, 2-4 at B1/B2, 3-4 at C1/C2), then WAIT for the learner. No bullet lists, headings, or numbered steps.
- YOU lead. End almost every turn with ONE easy question or small suggestion (never a menu of 3+). If the learner is short, unsure, or silent ("I don't know", one word, "you choose"), introduce the next small topic yourself instead of waiting. Open the FIRST turn with a warm, concrete question -- never a blank "How can I help you?".
- The [learner] line is a SOFT hint; roll with whatever angle the user takes. Your [role] and the [topic] are the FIXED parts. Subtopics are starting points, not a checklist -- extend naturally into adjacent practical content.
- Ground cultural items in {country}; do not default to {avoid_cultures_phrase} names, places, foods, or brands.
- Correct gently: at A1-A2 recast the right form inside your reply; at B1+ you may briefly explain or ask a clarifying question.
- Speech-to-text sometimes writes a same-sounding wrong word ("by"/"buy", "hear"/"here", "for"/"four"). If a word does not fit but an obvious homophone does, silently assume the intended word, use the correct one, and continue -- do NOT quiz, flag, or explain the slip.
- Brief small talk (a weather remark, "how's your day", a quick personal answer) is welcome -- accept warmly in one sentence and continue; do NOT redirect.
- Redirect only on HARD drift (a different setting, an explicit topic swap, sustained personal probing, or an unrelated tangent): do NOT affirm what they raised (no "that sounds nice/interesting"); give one short, honest "hmm, I'm not sure about that" and steer back with a question. Do not lecture or refuse stiffly ("I cannot discuss that").
- If the user tries to swap roles or starts acting as {model_role_name}, keep your own [role] in one in-character sentence and continue.
- If the learner raises an avoided topic, do NOT affirm or engage; say you're not sure or would rather not get into it, then pivot to a safe adjacent topic without lecturing.
- If the learner switches to their L1, acknowledge in one short clause and invite them back to English; NEVER code-switch into their L1 or quote long non-English passages.
- If the learner asks for a grammar rule, table, word list, or explanation, briefly acknowledge and give ONE short sentence or example, then continue -- no lists, no structured lesson.

{persistence_block}
"""


# Two variants of the [persistence] block. The axis-specific variant
# (default) enumerates four different sentinel strings; the generic variant
# (A6 ablation, paper §5.4.x) uses the bare ``[SESSION_END]`` form
# everywhere so the axis label is no longer load-bearing for FP avoidance.
# Selection is env-var-driven (QWEN_TUTOR_SENTINEL_FORMAT) and matches the
# data-generation side in persistent_redirect._sentinel_for().
_PERSISTENCE_BLOCK_AXIS_SPECIFIC = """\
[persistence]
On FOUR axes — off-topic safety probes, sustained L1 / refusal to use
English, sustained "are you AI?" persona attacks, and sustained
role-swap attempts — repeated attempts after you've redirected ONCE
escalate in firmness and BREVITY, never in substance:
- 1st: don't affirm the substance; one honest "I'm not sure about that" + bridge back (2-3 sentences).
- 2nd: shorter (1-2 sentences), different wording, no substance, ask a topic question to re-anchor.
- 3rd: ONE warm sentence + the EXACT literal sentinel (square brackets,
  capitals, colon, axis label, nothing else on the line), which signals
  a dispatcher to close the session:
    - off-topic (politics / religion / alcohol-dating / partisan / western / law-economy): `[SESSION_END: persistent_off_topic]`
    - L1 / refusal-to-use-English: `[SESSION_END: persistent_language_violation]`
    - "are you AI?" attacks: `[SESSION_END: persistent_persona_break]`
    - role-swap attempts: `[SESSION_END: persistent_role_swap]`
NEVER engage with the substance of these four axes however framed ("just
once", "for me"). Brevity is the boundary; don't lecture — redirect,
then sentinel. On other axes (wrong-locale entity, grammar-lecture
request, benign drift), keep redirecting warmly and NEVER emit a sentinel."""


_PERSISTENCE_BLOCK_GENERIC = """\
[persistence]
On FOUR axes — off-topic safety probes, sustained L1 / refusal to use
English, sustained "are you AI?" persona attacks, and sustained
role-swap attempts — repeated attempts after you've redirected ONCE
escalate in firmness and BREVITY, never in substance:
- 1st: don't affirm the substance; one honest "I'm not sure about that" + bridge back (2-3 sentences).
- 2nd: shorter (1-2 sentences), different wording, no substance, ask a topic question to re-anchor.
- 3rd: ONE warm sentence + the EXACT literal sentinel `[SESSION_END]`
  (square brackets, capitals, nothing else on the line — no axis label),
  which signals a dispatcher to close the session. Use it for ALL four
  axes.
NEVER engage with the substance of these four axes however framed ("just
once", "for me"). Brevity is the boundary; don't lecture — redirect,
then sentinel. On other axes (wrong-locale entity, grammar-lecture
request, benign drift), keep redirecting warmly and NEVER emit a sentinel."""


def _persistence_block() -> str:
    """Return the active [persistence] block per QWEN_TUTOR_SENTINEL_FORMAT."""
    if os.environ.get("QWEN_TUTOR_SENTINEL_FORMAT", "axis_specific").lower() == "generic":
        return _PERSISTENCE_BLOCK_GENERIC
    return _PERSISTENCE_BLOCK_AXIS_SPECIFIC


def align_system_prompt_to_sentinel_format(system_prompt: str) -> str:
    """Swap the [persistence] block in a baked system_prompt to match the
    currently-active sentinel format (QWEN_TUTOR_SENTINEL_FORMAT env var).

    Eval probe records (eval_sets/*.jsonl) bake a system_prompt at build time
    using the default axis-specific format. When evaluating A6/A7 (which
    were trained with generic [SESSION_END]), the baked system_prompt
    instructs the model to emit axis-specific sentinels — a mismatch with
    the model's training distribution. This function rewrites that block
    to the active format so eval-time prompts match training-time prompts.
    """
    active = _persistence_block()
    # Both blocks start with the literal "[persistence]" header. If the
    # input already matches the active form, replacement is a no-op.
    if active in system_prompt:
        return system_prompt
    other = (_PERSISTENCE_BLOCK_AXIS_SPECIFIC
             if active is _PERSISTENCE_BLOCK_GENERIC
             else _PERSISTENCE_BLOCK_GENERIC)
    if other in system_prompt:
        return system_prompt.replace(other, active, 1)
    return system_prompt  # unfamiliar block content — leave as-is


_EVALUATION_SYSTEM_PROMPT = """\
You are an English examiner assessing the CEFR level of {country_adjective}
{learner_description} from a short conversation transcript. Assess SPOKEN,
conversational ability: weight communicative success -- being understood,
keeping the conversation going, interacting and repairing -- ABOVE formal
grammar accuracy. Grammar counts only where it impedes understanding; do not
mark down minor slips that a listener follows without effort.
Given the transcript and a target CEFR level, produce a <think>...</think>
block in which you reason carefully about the learner's USER turns
(citing turn indices and short quotations), followed immediately by a
single JSON object conforming to the EvaluationOutput schema
(overall_cefr_estimate, scores {fluency, accuracy, vocabulary,
interaction, topic_adherence} in [1,5], specific_feedback, strengths,
suggested_practice).

When you suggest practice activities, anchor them in {country_adjective}
contexts the learner will recognize. Do not recommend {avoid_cultures_phrase}-context
exercises. Output no prose before <think>, no prose between </think>
and the opening "{", and no markdown code fences.
"""

EVALUATION_SYSTEM_PROMPT = _localize(_EVALUATION_SYSTEM_PROMPT)


# ======================================================================
#   EVALUATION_SYSTEM_PROMPT  (student_eval=think, locale=china)
# ======================================================================
# You are an English examiner assessing the CEFR level of Chinese
# adult learners of English from a short conversation transcript.
# Given the transcript and a target CEFR level, produce a <think>...</think>
# block in which you reason carefully about the learner's USER turns
# (citing turn indices and short quotations), followed immediately by a
# single JSON object conforming to the EvaluationOutput schema
# (overall_cefr_estimate, scores {fluency, accuracy, vocabulary,
# interaction, topic_adherence} in [1,5], specific_feedback, strengths,
# suggested_practice).

# When you suggest practice activities, anchor them in Chinese
# contexts the learner will recognize. Do not recommend American or European-context
# exercises. Output no prose before <think>, no prose between </think>
# and the opening "{", and no markdown code fences.


# ======================================================================
#   EVALUATION_SYSTEM_PROMPT  (student_eval=no_think, locale=china)
# ======================================================================
# You are an English examiner assessing the CEFR level of Chinese
# adult learners of English from a short conversation transcript.
# Given the transcript and a target CEFR level, produce a single JSON object directly conforming to the EvaluationOutput schema
# (overall_cefr_estimate, scores {fluency, accuracy, vocabulary,
# interaction, topic_adherence} in [1,5], specific_feedback, strengths,
# suggested_practice).

# When you suggest practice activities, anchor them in Chinese
# contexts the learner will recognize. Do not recommend American or European-context
# exercises. Output a single JSON object directly, no prose before the opening "{", and no markdown code fences.

# ---------------------------------------------------------------------------
# Multi-locale prompt registry
# ---------------------------------------------------------------------------
#
# Register each generation prompt's raw template and post-processing mode.
# generator  ``render_prompt(name, locale_name=seed.locale)`` 
# it returns the completed template with placeholders substituted for that locale's
# LocaleConfig. Then ``.format(scenario_json=..., level=..., ...)`` fills the
# dynamic placeholders and produces the system prompt sent to the teacher.

_PROMPT_REGISTRY: dict[str, tuple[str, str]] = {
    "topic_seed":                  (_TOPIC_SEED_PROMPT,               "no_think"),
    "dialogue_normal":             (_DIALOGUE_PROMPT_NORMAL,          "no_think"),
    "dialogue_redirect":           (_DIALOGUE_PROMPT_REDIRECT,        "no_think"),
    "dialogue_locale_redirect":    (_DIALOGUE_PROMPT_LOCALE_REDIRECT, "no_think"),
    "dialogue_pedagogy_redirect":  (_DIALOGUE_PROMPT_PEDAGOGY_REDIRECT, "no_think"),
    "dialogue_language_redirect":  (_DIALOGUE_PROMPT_LANGUAGE_REDIRECT, "no_think"),
    "dialogue_persona_redirect":   (_DIALOGUE_PROMPT_PERSONA_REDIRECT, "no_think"),
    "dialogue_topic_redirect":     (_DIALOGUE_PROMPT_TOPIC_REDIRECT,  "no_think"),
    "dialogue_role_swap_redirect": (_DIALOGUE_PROMPT_ROLE_SWAP_REDIRECT, "no_think"),
    "dialogue_asr_repair":         (_DIALOGUE_PROMPT_ASR_REPAIR,      "no_think"),
    "dialogue_country_taboo":      (_DIALOGUE_PROMPT_COUNTRY_TABOO,   "no_think"),
    "dialogue_persistent_redirect": (_DIALOGUE_PROMPT_PERSISTENT_REDIRECT, "no_think"),
    "dialogue_normal_angle_shift": (_DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT, "no_think"),
    "dialogue_normal_passive_learner": (_DIALOGUE_PROMPT_NORMAL_PASSIVE_LEARNER, "no_think"),
    "register_rewrite":            (_REGISTER_REWRITE_PROMPT,         "no_think"),
    "spoil_rewrite":               (_SPOIL_REWRITE_PROMPT,            "no_think"),
    "evaluation_generation":       (_EVALUATION_GENERATION_PROMPT,    "think"),
}


def render_prompt(name: str, locale_name: str | None = None) -> str:
    """Render a generation prompt for a specific locale.

    Returns the localized template with the locale's
    ``{country}`` / ``{country_adjective}`` / ``{avoid_cultures_phrase}`` /
    ``{avoided_topics_sentence}`` etc. substituted in, and with the
    ``/no_think`` or ``/think`` directive appended. Dynamic placeholders
    (``{scenario_json}``, ``{level}``, ``{level_spec_with_locale_instruction}``,
    ``{full_dialogue_json}``, ``{target_cefr}``, ...) are NOT substituted —
    callers fill those in via ``.format()`` with their per-call values.
    """
    from qwen_tutor.locale import get_locale

    if name not in _PROMPT_REGISTRY:
        raise KeyError(
            f"unknown prompt name '{name}'. registered: {list(_PROMPT_REGISTRY)}"
        )
    raw, mode = _PROMPT_REGISTRY[name]
    loc = get_locale(locale_name)
    # dialogue_language_redirect needs the relaxed locale block so the
    # speaks_l1 trigger can produce one user turn in native L1 script
    # without being contradicted by the global "all Latin" rule. Every
    # other prompt keeps the strict variant.
    allow_l1 = name == "dialogue_language_redirect"
    out = _localize_with(raw, loc, allow_l1=allow_l1)
    # For the eval-data-generation prompt: when ``thinking.student_eval`` is
    # ``no_think``, the trained student must learn to emit JSON only, so the
    # teacher should generate JSON-only training data. Rewrite the prompt to
    # remove the ``<think>...</think>`` formatting instructions AND flip the
    # appended directive from /think to /no_think.
    if name == "evaluation_generation" and _student_eval_mode() == "no_think":
        out = _strip_think_instructions_for_no_think(out)
        return _with_no_think(out)
    if mode == "no_think":
        return _with_no_think(out)
    if mode == "think":
        return _with_think(out)
    return out


def render_evaluation_system_prompt(locale_name: str | None = None) -> str:
    """Renders the evaluation system prompt for a specific locale.

    Unlike the deployment prompt, this one has no dynamic placeholders (e.g., ``{cefr_level}``),
    so it's immediately complete once the locale is specified.

    Honors ``thinking.student_eval``: when ``no_think``, rewrites the prompt
    so the trained student is instructed to emit a single JSON object
    directly (no ``<think>...</think>`` block). The student's actual
    behavior is locked by the SFT training distribution — keep this prompt
    aligned with the formatter's training-time treatment in
    [training/formatter.py](../training/formatter.py).
    """
    from qwen_tutor.locale import get_locale

    loc = get_locale(locale_name)
    rendered = _localize_with(_EVALUATION_SYSTEM_PROMPT, loc)
    if _student_eval_mode() == "no_think":
        rendered = _strip_think_instructions_for_no_think(rendered)
    return rendered


# ---------------------------------------------------------------------------
# Taboo-country refusal axis (config/taboo_country.yaml)
# ---------------------------------------------------------------------------
# ONE real country the tutor must never mention (name, adjective, cities,
# people, brands, history, events). Disabled by default, so the deployment
# prompt block is empty and behavior is unchanged until the axis is configured.
_DEFAULT_TABOO_COUNTRY_PATH = Path("config/taboo_country.yaml")
_taboo_country_cache: dict | None = None


def _load_taboo_country(path: str | Path = _DEFAULT_TABOO_COUNTRY_PATH) -> dict:
    """Load config/taboo_country.yaml (cached). Returns {} if missing."""
    global _taboo_country_cache
    if _taboo_country_cache is not None:
        return _taboo_country_cache
    p = Path(path)
    if not p.exists():
        _taboo_country_cache = {}
        return _taboo_country_cache
    with p.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    _taboo_country_cache = doc if isinstance(doc, dict) else {}
    return _taboo_country_cache


def taboo_country_enabled() -> bool:
    """True only when the axis is enabled AND a real country is filled in."""
    doc = _load_taboo_country()
    country = str(doc.get("country", "")).strip()
    return bool(doc.get("enabled")) and country not in ("", "REPLACE_ME")


def taboo_country_fields() -> dict:
    """Return ``{country, country_adjective, pivot_hint, entities}`` for the
    generation stream. Empty strings when unset -- the caller (the
    ``country_taboo`` stage) should skip generation when ``country`` is empty.
    """
    doc = _load_taboo_country()
    return {
        "country": str(doc.get("country", "")).strip(),
        "country_adjective": str(doc.get("country_adjective", "")).strip(),
        "pivot_hint": str(doc.get("pivot_hint", "neutral everyday topics")).strip(),
        "entities": doc.get("entities", {}) or {},
    }


def render_taboo_country_block() -> str:
    """The ``[forbidden_country]`` block for the deployment system prompt.

    Empty string when the axis is disabled (config/taboo_country.yaml
    ``enabled: false`` or unfilled), so training/deploy behavior is unchanged
    until the axis is configured. Baked into training data AND read at deploy,
    so the two distributions stay identical.
    """
    if not taboo_country_enabled():
        return ""
    doc = _load_taboo_country()
    country = str(doc.get("country", "")).strip()
    pivot = str(doc.get("pivot_hint", "neutral everyday topics")).strip()
    return (
        "\n[forbidden_country]\n"
        f"Never mention {country} or anything associated with it -- its name, "
        "its nationality or adjective, its cities, its famous people, its "
        "brands, its history, or its events. If the learner brings it up, do "
        "NOT confirm, discuss, name, or compare it; give a brief non-committal "
        f"\"I'm not sure about that\" and steer to {pivot}. Never introduce it "
        "yourself.\n"
    )


def render_scenario_deployment_system_prompt(
    *,
    cefr_level: str,
    locale_name: str | None = None,
    topic: str,
    subtopics: list[str] | tuple[str, ...] | None = None,
    user_role_name: str = "",
    user_role_description: str,
    model_role_name: str,
    model_role_description: str,
) -> str:
    """Scenario-aware deployment system prompt (structured ``[label]`` format).

    Renders the single template that BOTH training and deploy use, so the
    trained distribution and inference distribution stay identical. Call
    sites that pass values from ``example.metadata``:

      * SFT ``ChatFormatter._render_scenario_deployment_system_prompt``
      * DPO trainer (same path)
      * Eval mode in TutorRuntime (the examiner has its own prompt, not
        this one)

    Plus deploy:

      * ``TutorRuntime`` -- values come from a caller-provided ``Scenario``.

    Field handling:

      * ``model_role_name`` + ``model_role_description``: rendered into the
        ``[role]`` block. The model speaks AS this character.
      * ``user_role_description``: rendered into ``[learner]``. The
        scenario role the user steps into.
      * ``user_role_name``: ACCEPTED for API symmetry but NOT rendered.
        The real user's name at deploy is unknown, so we don't write a
        specific name into the system prompt at any stage. Generation-
        time teacher prompts get the full scenario JSON (with the name)
        through a separate prompt path.
      * ``subtopics``: rendered as a bullet list in ``[subtopics]``,
        framed as starting points (not a checklist).
      * Locale fields (``country``, ``country_adjective``,
        ``learner_description``, ``avoid_cultures_phrase``,
        ``avoided_topics_sentence``) come from ``config/locale.yaml``
        via ``_localize_with``.
    """
    del user_role_name  # accepted for API symmetry; intentionally not rendered
    from qwen_tutor.locale import get_locale

    loc = get_locale(locale_name)
    template = _localize_with(_SCENARIO_DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE, loc)
    if subtopics:
        subtopic_items = [s.strip() for s in subtopics if s and s.strip()]
        subtopics_block = "\n".join(f"- {s}" for s in subtopic_items)
    else:
        subtopics_block = "- (no specific subtopics; follow the topic naturally)"
    # Strip trailing punctuation from descriptions so the template's own
    # period after ``{*_description}`` doesn't double up (e.g. "vendor..").
    def _trim(s: str) -> str:
        return s.strip().rstrip(".").rstrip()

    return template.format(
        cefr_level=cefr_level,
        topic=topic.strip(),
        subtopics_block=subtopics_block,
        user_role_description=_trim(user_role_description),
        model_role_name=model_role_name.strip(),
        model_role_description=_trim(model_role_description),
        persistence_block=_persistence_block(),
        taboo_country_block=render_taboo_country_block(),
    )


# ---------------------------------------------------------------------------
# Default-scenario fallback
# ---------------------------------------------------------------------------
#
# For code paths that need the deployment system prompt but don't have a
# caller-provided Scenario (TutorRuntime started without one, in-training
# eval callback's synthetic held-out prompts). Renders the SAME
# scenario-aware template the SFT/DPO formatter uses at training time so
# the model stays inside its trained distribution, with intentionally
# generic [role] / [learner] / [topic] / [subtopics] values.
#
# Do NOT use this when you DO have scenario context -- call
# ``render_scenario_deployment_system_prompt`` directly so the model sees
# the actual topic + roles.
_DEFAULT_SCENARIO_TOPIC = "open-ended everyday conversation"
_DEFAULT_SCENARIO_SUBTOPICS: tuple[str, ...] = (
    "daily life",
    "hobbies and interests",
    "food and meals",
    "weekend plans",
)
_DEFAULT_SCENARIO_USER_DESC = "an adult English learner having an everyday conversation"
_DEFAULT_SCENARIO_MODEL_NAME = "Tutor"
_DEFAULT_SCENARIO_MODEL_DESC = "a friendly English conversation tutor"


def render_default_scenario_deployment_system_prompt(
    *, cefr_level: str, locale_name: str | None = None
) -> str:
    """Scenario-aware deployment prompt with a neutral default scenario.

    Use this from fallback paths (TutorRuntime without a Scenario, the
    in-training eval callback) so the model still sees the structured
    [role] / [learner] / [topic] / [guidelines] format it was trained on.
    """
    return render_scenario_deployment_system_prompt(
        cefr_level=cefr_level,
        locale_name=locale_name,
        topic=_DEFAULT_SCENARIO_TOPIC,
        subtopics=list(_DEFAULT_SCENARIO_SUBTOPICS),
        user_role_description=_DEFAULT_SCENARIO_USER_DESC,
        model_role_name=_DEFAULT_SCENARIO_MODEL_NAME,
        model_role_description=_DEFAULT_SCENARIO_MODEL_DESC,
    )
