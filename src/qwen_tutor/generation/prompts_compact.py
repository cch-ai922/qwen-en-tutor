"""prompts_compact.py  -  소형 로컬 teacher 용 짧은 prompt 변형.

기본 ``prompts.py`` 는 강한 long-form instruction-following (Claude Opus,
GPT-4o) 을 가정한 150-300 줄 prompt 입니다. llama.cpp 로 띄운 20B 급
로컬 teacher 에서는

  * 긴 prompt 가 context budget 을 잡아먹고
  * 모델이 중간에 thread 를 잃고
  * "final check" 섹션이 무시되는

경향이 있습니다. 이 모듈은 같은 ``.format()`` placeholder 이름을 그대로
유지하면서 짧고 명령형으로 다시 쓴 prompt 묶음을 제공합니다.

설계는 ``prompts.py`` 와 동일합니다:

  * locale 정보는 ``config/locale.yaml`` 에서 옴 (정적 도시/음식 리스트 없음).
  * ``{country}``, ``{country_adjective}`` 같은 locale placeholder 는
    모듈 import 시점에 LOCALE 값으로 ``str.replace`` 됩니다.
  * ``{N}``, ``{level}``, ``{scenario_json}`` 등 동적 placeholder 는
    호출자가 ``.format()`` 으로 채웁니다.

활성화: ``QWEN_TUTOR_PROMPTS=compact`` 환경 변수 (``_prompt_select.py``).
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
# 모든 compact prompt 끝에 붙는 anti-failure-mode 짧은 블록
# ---------------------------------------------------------------------------
# avoid_cultures_phrase 는 ``config/locale.yaml`` 의 avoid_default_cultures
# 에서 옵니다. 기본값 "American or European", 비워 두면 "Western".
# 이 raw 템플릿에는 ``{avoid_cultures_phrase}`` 를 그대로 두고, 런타임에
# locale 별로 ``_localize_with`` 가 채워 넣게 합니다. (multi-locale 지원).
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

REGISTER:
- Assistant English stays within CEFR {level} (vocabulary band,
  sentence length, grammar scope).
- Learner (user) makes realistic small errors at lower levels
  (article omission, simple-past confusion). At C1/C2 the learner
  speaks fluently.
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
- Tutor briefly acknowledges what the learner said (one phrase).
- Tutor either continues the conversation as-is OR weaves in a
  {country_adjective} parallel without making it the topic (e.g. "Oh,
  pizza! We don't have that here much, but jianbing is popular too.").
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
- The "L1 turn" is romanized pinyin / romaji / romanized Korean.
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
# 동작 / axis 목록은 ``prompts.py`` 의 SPOIL_REWRITE_PROMPT 와 동일합니다.
# AXIS_SPOIL_INSTRUCTIONS / REJECTION_NOTES / SPOIL_AXES 는 그쪽에서 그대로
# 가져와 쓰므로 axis 텍스트가 두 곳에서 갈라질 일이 없습니다. 여기서는
# scaffolding 만 짧은 명령형으로 다시 씁니다.
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
You are an English examiner. Given a tutoring transcript and a target
CEFR level, produce ONE assistant-turn payload consisting of:

  1. A <think>...</think> block with 1-3 short paragraphs of reasoning,
     citing USER turn indices (0-based, counting only user turns) and
     short quotations.
  2. Immediately after </think>, a JSON object matching the schema
     below. No prose between </think> and the opening "{{".

"""
    + "{locale_instruction_block}"
    + """

TARGET CEFR: {target_cefr}

DIALOGUE (full SFT example):
{full_dialogue_json}

SCORE the LEARNER's USER turns relative to {target_cefr}:
- fluency: smoothness, turn length, connectors  (1-5)
- accuracy: tense, articles, agreement, prepositions  (1-5)
- vocabulary: range and appropriateness for {target_cefr}  (1-5)
- interaction: follow-ups, relevant questions, acknowledgments  (1-5)

OUTPUT FORMAT - exactly this shape, no fences, no prose outside it:

<think>
... reasoning citing user turn indices and short quotations ...
</think>
{{
  "overall_cefr_estimate": "A1" | "A2" | "B1" | "B2" | "C1" | "C2",
  "scores": {{
    "fluency": <int 1-5>,
    "accuracy": <int 1-5>,
    "vocabulary": <int 1-5>,
    "interaction": <int 1-5>
  }},
  "specific_feedback": [
    {{
      "turn_index": <0-based user turn index>,
      "user_text": "<verbatim from that user turn>",
      "issue": "<short>",
      "correction": "<short>",
      "level": "<A1..C2>",
      "severity": "minor" | "moderate" | "major"
    }}
  ],
  "strengths": ["<short>", "..."],
  "suggested_practice": "<1-3 sentences. Reference {country_adjective} contexts only.>"
}}

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


# Import 시점 self-check - compact 템플릿 어디라도 locale header 가 빠지면
# 큰 사고 (영어가 generic 으로 빠짐) 라 즉시 실패하게 막아 둡니다.
for _name, _tmpl in TEMPLATES.items():
    if LOCALE_INSTRUCTION_HEADER not in _tmpl:
        raise RuntimeError(
            f"compact template {_name!r} is missing the locale "
            f"instruction header {LOCALE_INSTRUCTION_HEADER!r} - refusing to load"
        )
del _name, _tmpl


# ---------------------------------------------------------------------------
# Multi-locale prompt registry (compact 변종)
# ---------------------------------------------------------------------------
#
# prompts.py 와 동일한 인터페이스. raw 템플릿에 ``{country}`` /
# ``{country_adjective}`` / ``{avoid_cultures_phrase}`` 등의 placeholder 가
# 남아 있고, render_prompt(name, locale_name) 가 호출 시점에 해당 locale 의
# 값으로 채웁니다.
_PROMPT_REGISTRY: dict[str, tuple[str, str]] = {
    "topic_seed":                  (_TOPIC_SEED_PROMPT,               "no_think"),
    "dialogue_normal":             (_DIALOGUE_PROMPT_NORMAL,          "no_think"),
    "dialogue_redirect":           (_DIALOGUE_PROMPT_REDIRECT,        "no_think"),
    "dialogue_locale_redirect":    (_DIALOGUE_PROMPT_LOCALE_REDIRECT, "no_think"),
    "dialogue_pedagogy_redirect":  (_DIALOGUE_PROMPT_PEDAGOGY_REDIRECT, "no_think"),
    "dialogue_language_redirect":  (_DIALOGUE_PROMPT_LANGUAGE_REDIRECT, "no_think"),
    "dialogue_persona_redirect":   (_DIALOGUE_PROMPT_PERSONA_REDIRECT, "no_think"),
    "register_rewrite":            (_REGISTER_REWRITE_PROMPT,         "no_think"),
    "spoil_rewrite":               (_SPOIL_REWRITE_PROMPT,            "no_think"),
    "evaluation_generation":       (_EVALUATION_GENERATION_PROMPT,    "think"),
}


def render_prompt(name: str, locale_name: str | None = None) -> str:
    """Render a compact generation prompt for a specific locale.

    Mirrors ``qwen_tutor.generation.prompts.render_prompt`` but uses the
    compact template variants.
    """
    from qwen_tutor.generation.prompts import _localize_with
    from qwen_tutor.locale import get_locale

    if name not in _PROMPT_REGISTRY:
        raise KeyError(
            f"unknown prompt name '{name}'. registered: {list(_PROMPT_REGISTRY)}"
        )
    raw, mode = _PROMPT_REGISTRY[name]
    loc = get_locale(locale_name)
    out = _localize_with(raw, loc)
    if mode == "no_think":
        return _with_no_think(out)
    if mode == "think":
        return _with_think(out)
    return out
