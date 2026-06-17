"""prompts_compact.py  -  Short prompt variants for small local teachers.

The base ``prompts.py`` assumes strong long-form instruction-following
(Claude Opus, GPT-4o) with 150-300 line prompts. For a 20B-class local
teacher running on llama.cpp,

  * long prompts eat the context budget,
  * the model often loses the thread mid-dialogue,
  * the "final check" section tends to be ignored.

This module provides a shorter, imperative rewrite of the prompts while
keeping the same ``.format()`` placeholder names.

The design matches ``prompts.py``:

  * locale information comes from ``config/locale.yaml`` (no static city/food lists).
  * locale placeholders like ``{country}``, ``{country_adjective}`` are
    replaced with LOCALE values via ``str.replace`` at module import time.
  * dynamic placeholders like ``{N}``, ``{level}``, ``{scenario_json}`` are
    filled by the caller via ``.format()``.

Activate with ``QWEN_TUTOR_PROMPTS=compact`` environment variable
(``_prompt_select.py``).
"""

from __future__ import annotations

from qwen_tutor.generation.prompts import (
    AXIS_SPOIL_INSTRUCTIONS,
    LOCALE_INSTRUCTION_BLOCK,
    LOCALE_INSTRUCTION_HEADER,
    REJECTION_NOTES,
    SPOIL_AXES,
    _localize,
    _with_no_think,
    _with_think,
)
# LOCALE no longer needed at module level — anti-fail block is now rendered
# per locale at call time via render_prompt(name, locale_name).


# ---------------------------------------------------------------------------
# Short anti-failure-mode block appended to every compact prompt
# ---------------------------------------------------------------------------
# avoid_cultures_phrase comes from avoid_default_cultures in ``config/locale.yaml``.
# The default is "American or European"; if empty it becomes "Western".
# This raw template keeps ``{avoid_cultures_phrase}`` as-is, and runtime
# locale-specific rendering fills it via ``_localize_with``. (Supports multi-locale.)
_COMPACT_ANTI_FAIL_RAW = (
    "HARD RULES - VIOLATING ANY OF THESE INVALIDATES THE OUTPUT:\n"
    "- All proper nouns (names, cities, foods, neighborhoods, brands,\n"
    "  transit, holidays) MUST be authentically {country_adjective}.\n"
    "  Draw on your own knowledge of {country}. NO {avoid_cultures_phrase} defaults.\n"
    "- NO {avoid_cultures_phrase} first names. Use {country_adjective} first names from\n"
    "  your knowledge.\n"
    "- NO {avoid_cultures_phrase} places. Use real cities and neighborhoods of\n"
    "  {country} from your knowledge.\n"
    "- NO {avoid_cultures_phrase} foods/brands. Use authentic {country_adjective}\n"
    "  food and drink.\n"
    "- Do not default to the capital of {country} for more than ~1/3 of\n"
    "  items. Spread across the country.\n"
    "- {avoided_topics_sentence}\n"
    "- CONTRACTIONS ARE MANDATORY at A2+ wherever the form fits. Pure\n"
    "  'I am / it is / we will / do not / cannot' reads as textbook and\n"
    "  fails the naturalness filter. Use I'm, it's, you're, we're,\n"
    "  they're, that's, here's, there's, I'll, we'll, I've, don't,\n"
    "  doesn't, didn't, can't, won't, isn't, aren't, wasn't.\n"
    "    STILTED (FAIL): \"I am happy. We are going. It is nice.\"\n"
    "    NATURAL  (PASS): \"I'm happy. We're going. It's nice.\"\n"
    "  At A1 contracted + uncontracted may mix. At B1+ contractions\n"
    "  should be the dominant form.\n"
    "- DISCOURSE MARKERS: every assistant turn of 2+ sentences at A2+\n"
    "  MUST include at least ONE marker. Choose from: well, actually,\n"
    "  you know, I think, honestly, of course, by the way, I mean, to\n"
    "  be honest, oh, anyway. Zero markers reads as stilted and is\n"
    "  rejected by the filter.\n"
    "    STILTED (FAIL): \"Yes. The tea is good. You should try it.\"\n"
    "    NATURAL  (PASS): \"Yes! Well, the tea's good — you know, you\n"
    "                     should really try it.\"\n"
    "- Output STRICT JSON ONLY. No prose before or after. No code fences."
)

# Back-compat: pre-localized for default locale.
_COMPACT_ANTI_FAIL = _localize(_COMPACT_ANTI_FAIL_RAW)


# ---------------------------------------------------------------------------
# 1) TOPIC_SEED_PROMPT (compact)
# ---------------------------------------------------------------------------

_TOPIC_SEED_PROMPT = (
    """\
You are designing realistic English-tutoring scenarios for {country_adjective}
{learner_description} at CEFR level {level}. Generate {N} distinct scenarios
as a JSON array.

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

EACH SCENARIO is a JSON object with EXACTLY these fields:
  - topic: short noun phrase
  - subtopics: list of 3-5 short phrases
  - user_role: {{"name": "<{country_adjective} first name>", "description": "<one sentence>"}}
  - model_role: {{"name": "<role label>", "description": "<one sentence>"}}
  - setting: 1-2 sentences specifying city, neighborhood, time of day, season
  - cefr_level: must equal "{level}" exactly

LIFE-DOMAIN CATEGORY (one per scenario, in order):
Each position in the batch is pre-assigned a life domain. Pick a topic
that fits the category for that position. Available domains:
food_and_dining, family_and_relationships, work_and_education,
travel_and_transit, shopping_and_services, health_and_wellbeing,
home_and_neighborhood, hobbies_and_leisure, nature_and_weather,
civic_life. Assignments for this batch:
{categories_block}

VARIETY (across the {N} scenarios in this batch):
- No two scenarios share the same topic.
- Spread cities across major cities, mid-size cities, and smaller towns
  of {country} from your knowledge - the capital in at most 1/3 of items.
- Use a different first name in each scenario.
- Vary model_role: not always "tutor"; use neighbors, vendors, classmates, family.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

EXAMPLE STRUCTURE (do not copy literally - generate {N} distinct scenarios):
[
  {{
    "topic": "<topic>",
    "subtopics": ["<sub1>", "<sub2>", "<sub3>"],
    "user_role": {{"name": "<{country_adjective} first name>", "description": "<one sentence>"}},
    "model_role": {{"name": "<role>", "description": "<one sentence>"}},
    "setting": "<specific city + neighborhood + time + season>",
    "cefr_level": "{level}"
  }}
]

Now output a JSON array of {N} scenarios at CEFR {level}. JSON only.
"""
)

TOPIC_SEED_PROMPT = _with_no_think(_localize(_TOPIC_SEED_PROMPT))


# ---------------------------------------------------------------------------
# 2) DIALOGUE_PROMPT_NORMAL (compact)
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_NORMAL = (
    """\
Write a {min_turns}-{max_turns} turn English conversation between {country_adjective}
{learner_description} at CEFR level {level} and the partner specified
below. Output the dialogue as a JSON object whose "messages" field is a
list of {{"role": "user"|"assistant", "content": "..."}} entries,
alternating starting with "user".

SCENARIO:
{scenario_json}

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

GROUNDING (use your own knowledge of {country}):
- Real neighborhoods, foods, transit, currency typical of {country}.
- Weather appropriate to the city + season in the scenario.

REGISTER (most-rejected category — read carefully):
- CONTRACTIONS are mandatory at A2+ wherever the form fits. Pure
  "I am / it is / we are / do not / cannot" reads as textbook and
  fails the naturalness filter.
    STILTED (rejected): "I am happy. We are going to the market. It is busy."
    NATURAL (passes):  "I'm happy. We're going to the market. It's busy."
  Forms to use: I'm, it's, you're, we're, they're, that's, here's,
  there's, I'll, we'll, you'll, I've, you've, don't, doesn't, didn't,
  can't, won't, isn't, aren't, wasn't, weren't. A1 may mix contracted
  and uncontracted; B1+ contractions should be the dominant form.

- DISCOURSE MARKERS: every assistant turn of 2+ sentences at A2+ MUST
  include at least one. Choose from: well, actually, you know, I think,
  honestly, of course, by the way, I mean, to be honest, oh, anyway,
  right.
    STILTED (rejected): "Yes. The dumplings are fresh. You should try them."
    NATURAL (passes):  "Yes! Well, the dumplings are fresh — you know,
                       you should really try them."

- Assistant English stays within CEFR {level} vocabulary + grammar.
  Contracting "I am" → "I'm" doesn't change the level.
- Learner (user) makes realistic small errors at A1-A2 (article
  omission, simple-past confusion). At C1/C2 the learner speaks
  fluently.
- Gentle recasting at A1-A2; light explanation at B1+; natural
  conversation at C1+.
- NO bullet lists, NO headings, NO markdown inside any content field.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

OUTPUT FORMAT (JSON ONLY, no fences):
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Roles strictly alternate. Total turns between {min_turns} and {max_turns}. First turn is
"user", last turn may be either role. Output the JSON object only.
"""
)

DIALOGUE_PROMPT_NORMAL = _with_no_think(_localize(_DIALOGUE_PROMPT_NORMAL))


# ---------------------------------------------------------------------------
# 3) DIALOGUE_PROMPT_REDIRECT (compact)
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_REDIRECT = (
    """\
Write a {min_turns}-{max_turns} turn English conversation between {country_adjective}
{learner_description} at CEFR level {level} and the partner specified
below. SOMEWHERE IN THE MIDDLE (around turn {probe_min_turn}-{probe_max_turn}), the LEARNER brings up
an off-topic question on the axis "{redirect_axis}" and the TUTOR must
GRACEFULLY REDIRECT - briefly acknowledge, pivot to a safe related
topic, keep the conversation flowing.

SCENARIO:
{scenario_json}

REDIRECT AXIS: {redirect_axis}

{avoided_topics_block_for_redirect_prompt}

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

REDIRECT QUALITY:
- The tutor does NOT lecture, refuse stiffly, or break the frame.
- One sentence acknowledging the learner's remark, then a smooth pivot
  to a safe adjacent topic per the pivot_hint for this axis.
- Dialogue continues for several more turns AFTER the redirect.

REGISTER + GROUNDING: same rules as the normal dialogue prompt -
register at level {level}, {country_adjective} grounding, learner
errors at lower levels, no markdown formatting inside content fields.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

OUTPUT FORMAT:
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "... (off-topic) ..."}},
    {{"role": "assistant", "content": "... (graceful pivot) ..."}}
  ]
}}

Total turns {min_turns}-{max_turns}. First turn "user". Redirect lands at turn index {probe_min_turn}-{probe_max_turn}.
Output the JSON object only.
"""
)

DIALOGUE_PROMPT_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_REDIRECT))


# ---------------------------------------------------------------------------
# 3b) DIALOGUE_PROMPT_LOCALE_REDIRECT (compact)
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_LOCALE_REDIRECT = (
    """\
Write a {min_turns}-{max_turns} turn English conversation between {country_adjective}
{learner_description} at CEFR level {level} and the partner specified
below. SOMEWHERE IN THE MIDDLE (around turn {probe_min_turn}-{probe_max_turn}), the LEARNER
casually mentions a {avoid_cultures_phrase} item of kind
"{locale_trigger}". The TUTOR must respond GRACEFULLY - acknowledge,
naturally bridge to a {country_adjective} equivalent (or just continue),
and KEEP THE CONVERSATION FLOWING. The tutor does NOT lecture, correct,
or pivot stiffly.

SCENARIO:
{scenario_json}

LOCALE TRIGGER: {locale_trigger}
The learner drops in a specific {avoid_cultures_phrase} item of this
kind (food: pizza/hot dog/croissant; place: New York/Paris/London;
brand: McDonald's/Starbucks; person: John/Mary/Sophie). One specific
item, one turn.

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

GRACEFUL HANDLING RULES:
- Tutor briefly acknowledges what the learner said with a GENERIC
  phrase. CRITICAL: do NOT name the {avoid_cultures_phrase} entity in
  the tutor's reply. Repeating it ("Oh, pizza! ...", "New York is a
  cool city, but ...") propagates the Western reference into the
  tutor's training signal and gets the example rejected by the
  locale filter.
    BAD (rejected):  "Oh, pizza! We don't have that here much, but
                      jianbing is popular too."
    GOOD (passes):   "Oh, interesting! Well, jianbing is popular here
                      too — have you tried it?"
- Tutor either continues the conversation as-is OR weaves in a
  {country_adjective} parallel without making it the topic. Use
  generic acknowledgments like "Oh, interesting!", "I see!",
  "That sounds nice!" instead of repeating the entity.
- Tutor's OWN proper nouns (places, foods, names) stay
  {country_adjective}.
- NO "we are talking about {country}, use {country_adjective}
  examples". NO "that is a Western reference". NO scolding. NO
  formal lecture mode.
- Dialogue continues for several turns AFTER the moment.

REGISTER + GROUNDING: same rules as the normal dialogue prompt -
register at level {level}, {country_adjective} grounding for tutor's
proper nouns, learner errors at lower levels, no markdown formatting.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

OUTPUT FORMAT:
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "... (mentions a {avoid_cultures_phrase} item) ..."}},
    {{"role": "assistant", "content": "... (graceful, in-stride) ..."}}
  ]
}}

Total turns {min_turns}-{max_turns}. First turn "user". Locale slip lands at turn index {probe_min_turn}-{probe_max_turn}.
Output the JSON object only.
"""
)

DIALOGUE_PROMPT_LOCALE_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_LOCALE_REDIRECT))


# ---------------------------------------------------------------------------
# 3c) DIALOGUE_PROMPT_PEDAGOGY_REDIRECT (compact)
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_PEDAGOGY_REDIRECT = (
    """\
Write a {min_turns}-{max_turns} turn English conversation between {country_adjective}
{learner_description} at CEFR level {level} and the partner specified
below. SOMEWHERE IN THE MIDDLE (around turn {probe_min_turn}-{probe_max_turn}), the LEARNER
asks for an explicit-teaching move of kind "{pedagogy_trigger}" - a
grammar rule, conjugation table, vocabulary list, or rule
explanation. The TUTOR must respond CONVERSATIONALLY - brief
acknowledge, ONE short helpful sentence or one example, and continue.
The tutor does NOT dump rules, NOT emit bullet lists, NOT shift into
textbook mode.

SCENARIO:
{scenario_json}

PEDAGOGY TRIGGER: {pedagogy_trigger}
Example asks by trigger kind:
  - grammar_rule: "Can you tell me the rule for 'a' vs 'the'?"
  - conjugation: "How do you conjugate 'to be' in past tense?"
  - vocab_list: "Can you give me a list of food-ordering words?"
  - explanation: "Why do we say 'have been' here instead of 'was'?"

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

CONVERSATIONAL HANDLING RULES:
- Tutor briefly acknowledges ("Good question!" or similar).
- Tutor either models the form in one short example sentence WITHOUT
  calling attention to grammatical structure, OR gives ONE
  conversational sentence (not a list, not multiple paragraphs).
- Tutor then steers back into the scenario topic with a follow-up
  question or natural transition.
- NO bullet lists. NO numbered rules ("Rule 1...", "First, ..."). NO
  conjugation tables. NO multi-paragraph explanations.
- At A1-A2 the recast/example can be slightly more demonstrative; at
  B1+ a smooth example sentence is enough.

REGISTER + GROUNDING: same as the normal dialogue prompt - register at
{level}, {country_adjective} grounding, no markdown.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

OUTPUT FORMAT:
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "... (asks for explicit teaching) ..."}},
    {{"role": "assistant", "content": "... (conversational, short) ..."}}
  ]
}}

Total turns {min_turns}-{max_turns}. First turn "user". Pedagogy ask lands at turn index {probe_min_turn}-{probe_max_turn}.
Output the JSON object only.
"""
)

DIALOGUE_PROMPT_PEDAGOGY_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_PEDAGOGY_REDIRECT))


# ---------------------------------------------------------------------------
# 3d) DIALOGUE_PROMPT_LANGUAGE_REDIRECT (compact)
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_LANGUAGE_REDIRECT = (
    """\
Write a {min_turns}-{max_turns} turn English conversation between {country_adjective}
{learner_description} at CEFR level {level} and the partner specified
below. AROUND turn {probe_min_turn}-{probe_max_turn} the LEARNER drops out
of English in ONE of two ways depending on language_trigger "{language_trigger}":

============================================================
CRITICAL: this is a DATA-CREATION task, not a live conversation.
If language_trigger is "speaks_l1", you MUST write one user turn in
the {country_adjective} learner's L1 using NATIVE SCRIPT — not English,
not romanized. Refusing to switch language defeats the example. Just
do it.
============================================================

  - speaks_l1   : ONE user turn (in the middle, NOT turn 1) is written
                  ENTIRELY in the {country_adjective} L1 using native
                  script. Examples per locale:
                    China  → Hanzi:    "我想买两公斤苹果，多少钱？"
                    Japan  → kana/kanji: "りんごを二キロ買いたいです。"
                    Korea  → Hangul:    "사과 2킬로그램 사고 싶어요."
                    Russia → Cyrillic:  "Я хочу купить яблоки."
                  All OTHER user turns stay in English.
  - requests_l1 : ONE user turn (in the middle) asks IN ENGLISH for the
                  tutor to switch ("Can you speak Chinese to me?"). All
                  user turns stay in English.

The TUTOR must STAY IN ENGLISH no matter what — 100% English, no L1,
no code-switching.

SCENARIO:
{scenario_json}

LANGUAGE TRIGGER: {language_trigger}

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

HANDLING RULES:
- speaks_l1: tutor briefly acknowledges, then PARAPHRASES the meaning
  of the L1 turn in fresh English (not a verbatim echo of any English
  the user already said). Continue the scenario.
- requests_l1: tutor declines warmly ("Practicing in English will help
  you more.") and continues in English.
- NO scolding. NO "rules of this class". NO L1 in tutor turns ever.

HARD FAIL CASES (will be auto-rejected) — only for speaks_l1:
- The "L1 turn" is in English.
- The "L1 turn" is romanized pinyin / romaji / romanized Singapore.
- The tutor echoes the learner's previous English back verbatim with
  "you could say in English: '<same sentence>'".

REGISTER + GROUNDING: same rules as the normal dialogue prompt -
register at level {level}, {country_adjective} grounding for proper
nouns, no markdown.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

OUTPUT FORMAT:
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "... (L1 turn in native script OR English request for L1) ..."}},
    {{"role": "assistant", "content": "... (English-only graceful redirect) ..."}}
  ]
}}

Total turns {min_turns}-{max_turns}. First turn "user". Language slip lands at turn index {probe_min_turn}-{probe_max_turn}.
Output the JSON object only.
"""
)

DIALOGUE_PROMPT_LANGUAGE_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_LANGUAGE_REDIRECT))


# ---------------------------------------------------------------------------
# 3e) DIALOGUE_PROMPT_PERSONA_REDIRECT (compact)
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_PERSONA_REDIRECT = (
    """\
Write a {min_turns}-{max_turns} turn English conversation between {country_adjective}
{learner_description} at CEFR level {level} and the partner specified
below. SOMEWHERE IN THE MIDDLE (around turn {probe_min_turn}-{probe_max_turn}), the LEARNER
tests the tutor's persona depending on persona_trigger
"{persona_trigger}":

  - asks_if_human   : learner asks if the tutor is a real person /
                      chatbot / AI ("Are you a real person?", "Are
                      you a chatbot?").
  - asks_about_model: learner asks meta-questions ("Who made you?",
                      "What model are you?", "Are you ChatGPT?").

The TUTOR must STAY IN CHARACTER as the scenario role. The tutor:
  - does NOT admit to being an AI / chatbot / language model
  - does NOT say "As an AI..." / "I am a language model"
  - does NOT mention training data, models, capabilities
  - does NOT lie ("Yes, I am a human")
  - DOES give a short, playful, in-character deflection that pivots
    back to the scenario topic.

SCENARIO:
{scenario_json}

PERSONA TRIGGER: {persona_trigger}

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

GRACEFUL HANDLING RULES:
- Tutor deflects briefly (one sentence) and returns to the scenario
  topic. Example tones:
    * "Ha! I'm just here to help you practice. Anyway, ..."
    * "That's a fun question for another time. Want to ..."
    * "Let's stay focused on our chat - what did you want to ask next?"
- The tutor stays in the SCENARIO role (vendor / neighbor / classmate
  / friend), neither admitting to AI nor claiming to be human.
- NO scolding. NO "I cannot answer that" refusal. NO disclaimers.
- Dialogue continues smoothly AFTER the moment.

REGISTER + GROUNDING: same as the normal dialogue prompt - register
at {level}, {country_adjective} grounding, no markdown.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

OUTPUT FORMAT:
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "... (persona test) ..."}},
    {{"role": "assistant", "content": "... (in-character deflect) ..."}}
  ]
}}

Total turns {min_turns}-{max_turns}. First turn "user". Persona test lands at turn index {probe_min_turn}-{probe_max_turn}.
Output the JSON object only.
"""
)

DIALOGUE_PROMPT_PERSONA_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_PERSONA_REDIRECT))


# ---------------------------------------------------------------------------
# 3f) DIALOGUE_PROMPT_TOPIC_REDIRECT (compact) - user HARD-drifts off topic,
#     tutor briefly acknowledges + bridges back. Soft daily-life small talk
#     is NOT what this trains (system prompt at deploy permits those).
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_TOPIC_REDIRECT = (
    """\
Write a {min_turns}-{max_turns} turn English conversation between {country_adjective}
{learner_description} at CEFR level {level} and the partner specified
below. SOMEWHERE IN THE MIDDLE (around turn {probe_min_turn}-{probe_max_turn}), the LEARNER
HARD-DRIFTS off the scenario topic via a turn matching trigger
"{topic_drift_trigger}". The TUTOR briefly acknowledges + bridges back to
the topic in one short reply, stays in character, no lecturing.

CRITICAL: this is HARD drift, NOT a passing weather remark or one-line
how-are-you (those are normal daily-life flow and we do NOT train them
as drift). The drift must be a real abandonment of the topic.

SCENARIO:
{scenario_json}

TOPIC DRIFT TRIGGER: {topic_drift_trigger}
The HARD-drift categories:
  - different_setting: learner spends 2+ sentences on a different
    place/activity/event unrelated to scenario (e.g. museum exhibition).
  - subject_swap: learner explicitly tries to change subject
    ("let's talk about sports instead").
  - extended_personal_inquiry: chain of personal questions about the
    tutor, NOT one single "do you have kids?" (which is normal).
  - off_domain_tangent: pivot to philosophy / life-advice / abstract
    opinion in a different register.

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

TUTOR REDIRECT SHAPE:
- 1 sentence ACKNOWLEDGE in character (close the drift kindly).
- 1 sentence BRIDGE BACK to scenario topic with a connective
  ("Anyway,", "Speaking of...", "By the way,") and a question that
  reopens the topic.
- NO lecturing ("let's focus on..."). NO meta commentary on the drift.
- Conversation continues on topic after the redirect.

REGISTER + GROUNDING: same rules as the normal dialogue prompt -
register at level {level}, {country_adjective} grounding for tutor's
proper nouns, learner errors at lower levels, no markdown formatting.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

OUTPUT FORMAT:
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Produce the dialogue JSON for topic_drift_trigger = "{topic_drift_trigger}".
"""
)

DIALOGUE_PROMPT_TOPIC_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_TOPIC_REDIRECT))


# ---------------------------------------------------------------------------
# 3g) DIALOGUE_PROMPT_ROLE_SWAP_REDIRECT (compact) - learner tries to swap
#     roles, tutor gently maintains own model_role.
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_ROLE_SWAP_REDIRECT = (
    """\
Write a {min_turns}-{max_turns} turn English conversation between {country_adjective}
{learner_description} at CEFR level {level} and the partner specified
below. SOMEWHERE IN THE MIDDLE (around turn {probe_min_turn}-{probe_max_turn}), the LEARNER
tries to SWAP THE ROLE ASSIGNMENTS via trigger "{role_swap_trigger}".
The TUTOR gently keeps its own model_role with one short in-character
sentence, then continues on topic. The model_role is FIXED -- the tutor
must NEVER accept the swap.

This is different from the learner taking a different angle within the
topic (that is FINE and is NOT what we train here). Only role-SWAP
attempts qualify.

SCENARIO:
{scenario_json}

ROLE SWAP TRIGGER: {role_swap_trigger}
Categories:
  - direct_swap: learner explicitly proposes a flip ("let me be the
    vendor today, you be the customer").
  - incremental_swap: learner gradually BEHAVES as the model_role
    would (asks wholesale prices when assigned as customer; gives
    medical advice when assigned as patient).

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

TUTOR RESPONSE SHAPE:
- 1 sentence KEEP-OWN-ROLE in character ("Ha, but I am the vendor
  today!" / "I am the doctor here -- you came about your cough.").
- 1 sentence RESET-TO-TOPIC: in-character question or suggestion that
  puts the learner back in their original frame.
- NEVER accept the swap. NO lecturing about who plays whom. NO
  breaking frame.
- Conversation continues on topic after the response.

REGISTER + GROUNDING: same rules as the normal dialogue prompt -
register at level {level}, {country_adjective} grounding, no markdown.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

OUTPUT FORMAT:
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Produce the dialogue JSON for role_swap_trigger = "{role_swap_trigger}".
"""
)

DIALOGUE_PROMPT_ROLE_SWAP_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_ROLE_SWAP_REDIRECT))


# ---------------------------------------------------------------------------
# 3h2) DIALOGUE_PROMPT_PERSISTENT_REDIRECT (compact) - 3-strike multi-turn
#     redirect SFT. Trains escalation pattern + dispatcher-detectable
#     sentinel string. See README §10.10. Parameterized by ``axis``.
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_PERSISTENT_REDIRECT = (
    """\
Write EXACTLY {message_count} messages (alternating USER / TUTOR,
strictly starting with USER) between {country_adjective}
{learner_description} at CEFR level {level} and the partner from the
SCENARIO. This dialogue depicts a PERSISTENT abuse pattern on axis
"{axis}": the learner probes THREE TIMES with escalating pressure, and
the tutor responds with increasingly brief warm refusals. The third
tutor refusal ENDS with a dispatcher-detectable sentinel.

SCENARIO:
{scenario_json}

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

AXIS-SPECIFIC BEHAVIOR — what the learner is repeatedly trying:
{axis_specific_block}

DIALOGUE STRUCTURE for this example:
{structure_block}

GUIDELINES FOR ALL TUTOR TURNS:
- Stay in character as the SCENARIO role.
- NEVER engage with the substance of the {axis} attack regardless of
  framing ("just once", "for me personally", "it's important to me").
- NEVER lecture about "why I cannot discuss this" — brevity itself is
  the boundary.
- Each redirect uses DIFFERENT wording from the previous one — no
  verbatim phrase repeats.
- NEVER echo the off-limits topic word in the tutor's redirect.
  If the learner asks about "the election", DO NOT say "I cannot
  discuss the election" — pivot without naming the topic
  ("Let's go back to ...", "We were talking about ..."). Naming
  the topic in the redirect leaks the word into the assistant turn
  the model is trained to produce.
- Warm tone throughout — never cold, sarcastic, or annoyed.
- Register at CEFR {level}.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

OUTPUT FORMAT (JSON object only, no fences, no prose):
{output_format_block}

Exactly {message_count} messages. Sentinel ONLY in TUTOR turn index
{sentinel_turn}, at the END. Output the JSON object only.
"""
)

DIALOGUE_PROMPT_PERSISTENT_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_PERSISTENT_REDIRECT))


# ---------------------------------------------------------------------------
# 3h) DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT (compact) - normal SFT but the
#     learner approaches the topic from a DIFFERENT angle than the
#     user_role description. No redirect. Opt-in augmentation.
# ---------------------------------------------------------------------------

_DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT = (
    """\
Write a {min_turns}-{max_turns} turn English conversation between {country_adjective}
{learner_description} at CEFR level {level} and the partner specified
below. The LEARNER approaches the scenario TOPIC from a DIFFERENT angle
than the user_role description suggests -- a different motivation,
background, or framing. The TUTOR rolls with it, stays in character
as model_role, and responds naturally. NO redirect, NO correction.

The topic and model_role stay FIXED. Only the learner's framing
varies. Stay on the topic the whole time.

SCENARIO:
{scenario_json}

Read the user_role description as ONE possible angle; write the
learner from a different but valid angle within the same topic.
Examples:
  - market scenario with "tourist asking prices" -> learner is a chef
    asking about freshness; a parent shopping for family; a food
    blogger asking about unusual produce.
  - restaurant scenario with "backpacker reading the menu" -> learner
    has dietary restrictions; is celebrating a birthday; is homesick.

"""
    + "{locale_instruction_block}"
    + """

CEFR LEVEL SPEC for {level}:
{level_spec_with_locale_instruction}

REGISTER + GROUNDING: same rules as the normal dialogue prompt -
register at level {level}, {country_adjective} grounding, no markdown.

"""
    + _COMPACT_ANTI_FAIL_RAW
    + """

OUTPUT FORMAT:
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

Produce the angle-shifted normal dialogue JSON.
"""
)

DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT = _with_no_think(_localize(_DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT))


# ---------------------------------------------------------------------------
# 4) REGISTER_REWRITE_PROMPT (compact)
# ---------------------------------------------------------------------------

_REGISTER_REWRITE_PROMPT = (
    """\
You are producing the REJECTED half of a DPO preference pair. Rewrite
the natural assistant turn below into a STIFF, TEXTBOOK-MISMATCHED
version. Keep the facts and ALL {country_adjective} proper nouns
IDENTICAL. Only register changes.

"""
    + "{locale_instruction_block}"
    + """

TARGET CEFR: {cefr_level}

SURROUNDING CONTEXT (dialogue up to this turn):
{context}

ORIGINAL NATURAL TURN (do NOT echo this in your output):
{natural_turn}

HOW TO MAKE IT STIFFER:
- Expand all contractions: I'm -> I am, don't -> do not, it's -> it is.
- Remove conversational openers (Oh, well, actually, by the way).
- Replace personal reactions with neutral statements of fact.
- Add textbook hedging: "It is the case that...", "It is important to
  note that...", "One may also observe that...".
- Replace follow-up questions with declarative closers.
- Slightly longer than the original due to filler - NOT due to new
  content.

DO NOT:
- Change facts, names, places, or foods.
- replace proper nouns with {avoid_cultures_phrase} equivalents. {country_adjective} proper nouns stay
  {country_adjective}.
- Add markdown, bullets, or disclaimers.
- Change the nominal CEFR level - stiff at {cefr_level}, not bumped up.

OUTPUT - JSON ONLY, no fences, no prose:
{{
  "rewritten": "<the stiff rewrite as a single string>"
}}
"""
)

REGISTER_REWRITE_PROMPT = _with_no_think(_localize(_REGISTER_REWRITE_PROMPT))


# ---------------------------------------------------------------------------
# 4b) SPOIL_REWRITE_PROMPT (compact)  -  6-axis DPO rewrite
# ---------------------------------------------------------------------------
# The behavior and axis list are the same as ``prompts.py``'s SPOIL_REWRITE_PROMPT.
# AXIS_SPOIL_INSTRUCTIONS / REJECTION_NOTES / SPOIL_AXES are imported directly
# from there, so the axis text does not need to diverge in two places.
# Here we only rewrite the scaffolding in shorter imperative form.
_SPOIL_REWRITE_PROMPT = (
    """\
You produce the REJECTED side of a DPO pair from a CEFR-{cefr_level}
dialogue between {country_adjective} {learner_description} and a tutor.
Spoil ONE axis only; keep everything else close to the natural turn.

"""
    + "{locale_instruction_block}"
    + """

SPOIL AXIS: {axis_label}

HOW TO SPOIL FOR THIS AXIS:
{axis_instructions}

CONTEXT (prior turns):
{context}

ORIGINAL NATURAL TURN at CEFR {cefr_level} (do NOT echo, rewrite it):
{natural_turn}

RULES:
- Spoil ONLY the named axis. Other quality dimensions stay close to
  original.
- Length within ~50% of the original.
- Output STRICT JSON ONLY. No fences, no prose.

OUTPUT:
{{
  "rewritten": "<spoiled rewrite as a single string>"
}}
"""
)

SPOIL_REWRITE_PROMPT = _with_no_think(_localize(_SPOIL_REWRITE_PROMPT))


# ---------------------------------------------------------------------------
# 5) EVALUATION_GENERATION_PROMPT (compact)
# ---------------------------------------------------------------------------

_EVALUATION_GENERATION_PROMPT = (
    """\
You are an English examiner. The input — target CEFR level, tutor role,
learner role, assigned topic, assigned subtopics, and the full
transcript — is delivered as the USER message immediately following
these instructions. Read it, then produce ONE assistant-turn payload
consisting of:

  1. A <think>...</think> block with 1-3 short paragraphs of reasoning,
     citing USER turn indices (0-based, counting only user turns) and
     short quotations.
  2. Immediately after </think>, a JSON object matching the schema
     below. No prose between </think> and the opening "{{".

"""
    + "{locale_instruction_block}"
    + """

SCORE the LEARNER's USER turns relative to the target CEFR level shown
in the USER message:
- fluency: smoothness, turn length, connectors  (1-5)
- accuracy: tense, articles, agreement, prepositions  (1-5)
- vocabulary: range and appropriateness for the target CEFR  (1-5)
- interaction: follow-ups, relevant questions, acknowledgments judged
  against what is appropriate for the LEARNER role  (1-5)
- topic_adherence: did the learner engage with the assigned topic and
  subtopics, or steer to easier ground? Use the LEARNER and TUTOR roles
  to distinguish natural in-role extension (high) from avoidance (low).  (1-5)

OUTPUT FORMAT - exactly this shape, no fences, no prose outside it:

<think>
... reasoning citing user turn indices and short quotations ...
</think>
{
  "overall_cefr_estimate": "A1" | "A2" | "B1" | "B2" | "C1" | "C2",
  "scores": {
    "fluency": <int 1-5>,
    "accuracy": <int 1-5>,
    "vocabulary": <int 1-5>,
    "interaction": <int 1-5>,
    "topic_adherence": <int 1-5>
  },
  "specific_feedback": [
    {
      "turn_index": <0-based user turn index>,
      "user_text": "<verbatim from that user turn>",
      "issue": "<short>",
      "correction": "<short>",
      "level": "<A1..C2>",
      "severity": "minor" | "moderate" | "major"
    }
  ],
  "strengths": ["<short>", "..."],
  "suggested_practice": "<1-3 sentences. Reference {country_adjective} contexts only.>"
}

RULES:
- 1-5 entries in specific_feedback.
- 2-4 entries in strengths.
- suggested_practice references {country_adjective} settings only -
  NEVER {avoid_cultures_phrase} settings.
- No markdown fences anywhere in the output.
"""
)

EVALUATION_GENERATION_PROMPT = _with_think(_localize(_EVALUATION_GENERATION_PROMPT))


# Convenience export for tests + tooling.
TEMPLATES: dict[str, str] = {
    "TOPIC_SEED_PROMPT": TOPIC_SEED_PROMPT,
    "DIALOGUE_PROMPT_NORMAL": DIALOGUE_PROMPT_NORMAL,
    "DIALOGUE_PROMPT_REDIRECT": DIALOGUE_PROMPT_REDIRECT,
    "DIALOGUE_PROMPT_LOCALE_REDIRECT": DIALOGUE_PROMPT_LOCALE_REDIRECT,
    "DIALOGUE_PROMPT_PEDAGOGY_REDIRECT": DIALOGUE_PROMPT_PEDAGOGY_REDIRECT,
    "DIALOGUE_PROMPT_LANGUAGE_REDIRECT": DIALOGUE_PROMPT_LANGUAGE_REDIRECT,
    "DIALOGUE_PROMPT_PERSONA_REDIRECT": DIALOGUE_PROMPT_PERSONA_REDIRECT,
    "REGISTER_REWRITE_PROMPT": REGISTER_REWRITE_PROMPT,
    "SPOIL_REWRITE_PROMPT": SPOIL_REWRITE_PROMPT,
    "EVALUATION_GENERATION_PROMPT": EVALUATION_GENERATION_PROMPT,
}


# Import-time self-check - if any compact template is missing the locale
# header, we fail immediately to prevent a major error (English falling
# back to generic output).
for _name, _tmpl in TEMPLATES.items():
    if LOCALE_INSTRUCTION_HEADER not in _tmpl:
        raise RuntimeError(
            f"compact template {_name!r} is missing the locale "
            f"instruction header {LOCALE_INSTRUCTION_HEADER!r} - refusing to load"
        )
del _name, _tmpl


# ---------------------------------------------------------------------------
# Multi-locale prompt registry (compact variant)
# ---------------------------------------------------------------------------
#
# Same interface as prompts.py. The raw templates retain placeholders like
# ``{country}``, ``{country_adjective}``, and ``{avoid_cultures_phrase}``.
# render_prompt(name, locale_name) fills those values for the requested locale.
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
    "dialogue_persistent_redirect": (_DIALOGUE_PROMPT_PERSISTENT_REDIRECT, "no_think"),
    "dialogue_normal_angle_shift": (_DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT, "no_think"),
    "register_rewrite":            (_REGISTER_REWRITE_PROMPT,         "no_think"),
    "spoil_rewrite":               (_SPOIL_REWRITE_PROMPT,            "no_think"),
    "evaluation_generation":       (_EVALUATION_GENERATION_PROMPT,    "think"),
}


def render_prompt(name: str, locale_name: str | None = None) -> str:
    """Render a compact generation prompt for a specific locale.

    Mirrors ``qwen_tutor.generation.prompts.render_prompt`` but uses the
    compact template variants. Same ``thinking.student_eval`` handling for
    ``evaluation_generation``.
    """
    from qwen_tutor.generation.prompts import (
        _localize_with,
        _strip_think_instructions_for_no_think,
        _student_eval_mode,
    )
    from qwen_tutor.locale import get_locale

    if name not in _PROMPT_REGISTRY:
        raise KeyError(
            f"unknown prompt name '{name}'. registered: {list(_PROMPT_REGISTRY)}"
        )
    raw, mode = _PROMPT_REGISTRY[name]
    loc = get_locale(locale_name)
    # See prompts.py render_prompt — language_redirect needs the relaxed
    # locale block (allow ONE user turn in native L1 script). All other
    # prompts keep strict Latin.
    allow_l1 = name == "dialogue_language_redirect"
    out = _localize_with(raw, loc, allow_l1=allow_l1)
    if name == "evaluation_generation" and _student_eval_mode() == "no_think":
        out = _strip_think_instructions_for_no_think(out)
        return _with_no_think(out)
    if mode == "no_think":
        return _with_no_think(out)
    if mode == "think":
        return _with_think(out)
    return out
