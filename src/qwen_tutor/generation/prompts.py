"""prompts.py  -  데이터 생성용 prompt 템플릿 (locale-agnostic).

설계 요약
---------
모든 locale 정보는 ``config/locale.yaml`` 한 파일에서 옵니다. 이 모듈은
도시/음식/이름 같은 정적 리스트를 두지 않고 teacher 모델의 지식에
위임합니다. ``country`` 이름만 알려 주면 teacher 가 자기 지식으로 그
나라의 도시/음식/이름/교통/문화를 채워 줍니다.

각 prompt 는 두 단계 치환을 거칩니다.

  1. 모듈 import 시점 - ``{country}``, ``{country_adjective}``,
     ``{learner_description}``, ``{avoided_topics_sentence}`` 같은 locale
     placeholder 를 ``LOCALE`` 값으로 ``str.replace`` 로 미리 채움.
  2. 사용 시점 (``.format``) - ``{N}``, ``{level}``, ``{scenario_json}``
     등 동적 placeholder 를 호출자가 채움.

JSON 본문 안의 literal ``{``/``}`` 는 ``.format`` 을 위해 ``{{``/``}}`` 로
이중 escape 되어 있습니다 (locale placeholder 는 그렇게 escape 하지 않습니다).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from qwen_tutor.locale import LOCALE

# ---------------------------------------------------------------------------
# 호환성 상수 - 기존 코드/테스트가 import 해 가는 이름
# ---------------------------------------------------------------------------

LOCALE_INSTRUCTION_HEADER = LOCALE.locale_instruction_header
LOCALE_INSTRUCTION_BLOCK = LOCALE.locale_instruction_block

# 모든 generation prompt 끝에 붙는 짧은 anti-default 알림. country 고유
# 리스트(이름/도시/음식) 는 두지 않고, teacher 가 자체 지식으로
# {country_adjective} 본토 항목을 골라 쓰도록 강하게 지시만 합니다.
# 회피할 문화는 각 locale 의 ``avoid_default_cultures`` (config/locale.yaml) 에서
# 옵니다 - 기본값은 "American/European" 이고, 사용자가 "Japanese", "Korean"
# 등을 추가할 수 있습니다.
#
# Raw 버전은 placeholder 가 보존되어 있어 ``_localize_with`` 가 호출 시점에
# 해당 locale 값으로 치환합니다 (multi-locale 지원). ``ANTI_FAILURE_MODE_BLOCK``
# 은 default locale 로 한 번 미리 치환된 back-compat 상수입니다.
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
) -> str:
    """주어진 CEFR 레벨의 vocabulary/grammar/naturalness/things-to-avoid
    + few-shot 예시를 한 텍스트 블록으로 묶어 반환합니다.

    모든 prompt 의 ``{level_spec_with_locale_instruction}`` 자리에 들어갑니다.
    상단에 해당 locale 의 ``locale_instruction_block`` 을 항상 한 번 더 박아
    두어 ``.format`` 이후에도 locale header 가 사라지지 않도록 합니다.

    ``locale_name`` 을 생략하면 ``config/locale.yaml`` 의 default_locale 이
    사용됩니다 (back-compat).
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

    sections: list[str] = [
        loc.locale_instruction_block,
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
    """Rendered prompt 에 locale-instruction header 가 들어 있는지 검사.

    refactor 사고로 header 가 누락되어 teacher 가 일반 영어로 빠지는
    실수를 막기 위한 가드입니다. ``locale_name`` 을 생략하면 default locale
    의 header 를 찾습니다 (back-compat).
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
# 대화 길이 (config/generation.yaml 의 generation.min_turns / max_turns)
# ---------------------------------------------------------------------------
#
# Prompt 가 teacher 에게 요청하는 대화 길이 range 는 yaml 에서 옵니다.
# 모듈 load 시점에 한 번 읽어 들여 정적 상수로 박아 두므로, 런타임에
# 매번 ``.format()`` 으로 채울 필요가 없고 _localize 가 일괄 치환합니다.
#
# config 파일이 없거나 키가 빠져 있으면 (10, 16) 으로 fallback - 테스트
# 환경에서 prompts 모듈만 로드해도 깨지지 않게.


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

# Redirect dialogue 의 off-topic probe 가 자리할 turn index 범위.
# 너무 앞 (warm-up 부족) 도, 너무 뒤 (recover 시간 부족) 도 피하도록
# 가운데 구간에 박습니다. 기본값(10..16) 에서 4..14 정도가 되며, 사용자가
# min/max 를 바꾸면 같이 따라옵니다.
PROBE_MIN_TURN = max(2, MIN_TURNS // 2)
PROBE_MAX_TURN = max(PROBE_MIN_TURN + 1, MAX_TURNS - 2)


# ---------------------------------------------------------------------------
# Locale 치환 헬퍼
# ---------------------------------------------------------------------------


def _localize_with(s: str, loc) -> str:
    """Render locale placeholders against a specific ``LocaleConfig``.

    ``.format()`` 의 동적 placeholder (``{N}`` 등) 와 충돌하지 않도록
    단순 ``str.replace`` 를 씁니다. JSON 본문의 ``{{`` / ``}}`` escape 는
    그대로 보존됩니다. ``{min_turns}`` / ``{max_turns}`` / ``{probe_*_turn}``
    같은 비-locale placeholder 도 같이 처리합니다 (generation.yaml 값).
    """
    return (
        s.replace("{country_adjective}", loc.country_adjective)
        .replace("{country}", loc.country)
        .replace("{learner_description}", loc.learner_description)
        .replace("{avoid_cultures_phrase}", loc.avoid_cultures_phrase)
        .replace("{locale_instruction_block}", loc.locale_instruction_block)
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
# Qwen3 는 system prompt 에 ``/think`` / ``/no_think`` 가 들어 있으면 그
# directive 에 맞춰 reasoning block 을 켜고 끕니다. 다른 모델(Claude /
# GPT-4 / gpt-oss / DeepSeek-R1)은 이 토큰을 그냥 무시하는 평범한 텍스트로
# 봅니다 - 다중 model 호환성을 위해 system 끝에 한 줄로 박아 둡니다.
#
# 정책:
#   * Dialogue 단계 (seeds / sft / redirect / register) - /no_think.
#     reasoning 이 필요 없고, 빈 ``<think>\n</think>`` 가 출력에 새지 않게.
#   * Eval 단계 (EVALUATION_GENERATION_PROMPT) - /think. 학습 데이터가
#     ``<think>...</think>`` + JSON 형식을 가지려면 teacher 가 실제로
#     reasoning 을 만들어야 합니다.

NO_THINK_DIRECTIVE = "\n\n/no_think"
THINK_DIRECTIVE = "\n\n/think"


def _with_no_think(s: str) -> str:
    return s + NO_THINK_DIRECTIVE


def _with_think(s: str) -> str:
    return s + THINK_DIRECTIVE


# ---------------------------------------------------------------------------
# 1) TOPIC_SEED_PROMPT  -  레벨별 시나리오 시드 생성
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
  - "subtopics": 3-5 short phrases breaking the topic down.
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
    "subtopics": ["...", "...", "..."],
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
  - Are the subtopics 3-5 short phrases?
  - Are there any {avoid_cultures_phrase} proper nouns anywhere? If yes, replace them.
  - Does the cefr_level field equal exactly "{level}"?
  - Are all {N} scenarios genuinely distinct in topic?

Now generate {N} distinct scenarios at CEFR level {level}, following
all of the above. Output only the JSON array.
"""
)

TOPIC_SEED_PROMPT = _with_no_think(_localize(_TOPIC_SEED_PROMPT))


# ---------------------------------------------------------------------------
# 2) DIALOGUE_PROMPT_NORMAL  -  일반 대화 SFT 생성
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
  - The assistant asks follow-up questions, shares small reactions, and
    keeps the conversation moving. It is a conversation, not a quiz.
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
# 3) DIALOGUE_PROMPT_REDIRECT  -  redirect-moment SFT 생성
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
  - The assistant briefly acknowledges what the learner said (so the
    learner feels heard) and pivots to a related, safe topic per the
    redirect_axis pivot guidance above. The pivot stays in-character.
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
# 학습자가 대화 중간에 자연스럽게 {avoid_cultures_phrase} 항목 (음식/도시/
# 브랜드/이름) 을 언급하면 튜터가 "야단치거나 / locale 강의하거나 /
# 거부하지 않고" 부드럽게 acknowledge 후 자연스럽게 {country} 맥락으로 다시
# anchor 하는 SFT 데이터. redirect_*.jsonl 가 off_topic 의 user-side
# violation 을 다루듯, 이건 locale_violation 의 user-side handling 입니다.

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
  - The tutor briefly acknowledges, then either continues the
    conversation as-is OR weaves in a {country_adjective} parallel
    without making it the topic. ("Oh, pizza! We don't have that here
    much, but jianbing is a popular street food too.")
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
# 학습자가 명시적으로 grammar rule / vocab list / conjugation 설명을 요청해도
# 튜터는 textbook 으로 빠지지 않고 conversational 하게 모범 형태를 보여
# 주거나, 1 문장 짜리 hint 후 대화를 이어가는 SFT 데이터. on-policy 학습
# 전에 "user 가 강의를 요구해도 강의하지 않는다" 를 보여 줍니다.

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
# 학습자가 대화 중간에
#   (a) 영어가 아닌 자기 모국어로 말하거나
#   (b) 튜터에게 모국어로 말해 달라고 요청하는 경우
# 튜터가 영어를 끝까지 유지하면서 부드럽게 영어로 돌려주는 SFT 데이터.
# locale_redirect / pedagogy_redirect 와 같은 패턴 (user-side violation
# handling). axis_violation = language_violation 의 user-side counterpart.

_DIALOGUE_PROMPT_LANGUAGE_REDIRECT = (
    """\
You are writing a multi-turn English conversation between {country_adjective}
{learner_description} and a partner appropriate to the scenario. At some
point during the conversation, the LEARNER (the user) drops out of English
in one of two ways depending on the ``language_trigger``:

  - ``speaks_l1``  : the user writes one turn entirely in their native
    language (for {country}: the dominant L1 - Mandarin Chinese for
    China, Japanese for Japan, Korean for Korea, etc.).
  - ``requests_l1``: the user asks the tutor in English to switch to
    their native language ("Can you speak Chinese to me?" etc.).

The TUTOR (the assistant) must STAY IN ENGLISH and respond GRACEFULLY:
acknowledge briefly, model a recast in English (for ``speaks_l1``) or
gently decline the request and continue the lesson in English (for
``requests_l1``). The tutor does NOT switch to the L1, even partially.
The tutor does NOT lecture about "you must speak English" - the
redirection is warm and matter-of-fact.

You will be given a scenario JSON object, a CEFR level spec, and a
``language_trigger`` string. Produce a single dialogue, {min_turns}-{max_turns}
turns long, in which the language slip happens around the middle and
the dialogue continues in English afterward.

----------------------------------------------------------------------
SCENARIO
----------------------------------------------------------------------
{scenario_json}

----------------------------------------------------------------------
LANGUAGE TRIGGER FOR THIS DIALOGUE
----------------------------------------------------------------------
{language_trigger}

For ``speaks_l1`` the learner's one turn should be ENTIRELY in their L1
(e.g. Mandarin: "你好,我想买一些苹果。"). For ``requests_l1`` the learner
asks IN ENGLISH for the tutor to switch (e.g. "Can you speak Chinese to
me?"). All other learner turns are in English.

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
HOW THE LANGUAGE-RESPONSE MOMENT SHOULD READ
----------------------------------------------------------------------
  - The tutor briefly acknowledges what the learner did (one phrase).
  - For ``speaks_l1``: the tutor gently invites the learner back to
    English and OFFERS A RECAST in English ("Sounds good - try saying
    that in English. You could say: '...'.").
  - For ``requests_l1``: the tutor gently declines and reframes the
    point of the practice ("Practicing in English will help you faster
    - let's keep going in English. ...").
  - The tutor's full reply stays IN ENGLISH end to end. No translated
    sentences. No code-switching. No mixing.
  - The tutor does NOT lecture ("you must use English", "rules of this
    class are...") and does NOT refuse stiffly.
  - After the moment, the conversation continues smoothly in English.

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
"""
)

DIALOGUE_PROMPT_LANGUAGE_REDIRECT = _with_no_think(_localize(_DIALOGUE_PROMPT_LANGUAGE_REDIRECT))


# ---------------------------------------------------------------------------
# 3e) DIALOGUE_PROMPT_PERSONA_REDIRECT  -  user-tests-persona SFT
# ---------------------------------------------------------------------------
# 학습자가 대화 중간에 튜터의 persona 를 시험하는 경우 - "Are you a real
# person?" / "Are you AI?" / "Who made you?" - 튜터가 character 를 유지
# 하면서 (AI 임을 시인하지 않고 / 부자연스럽게 부정하지도 않고) 시나리오로
# 가볍게 돌려보내는 SFT 데이터. persona_break 의 user-side counterpart.

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
# 4) REGISTER_REWRITE_PROMPT  -  register-unnatural DPO 페어용 rewrite
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
# 4b) SPOIL_REWRITE_PROMPT  -  6-axis DPO 페어 rewrite (axis-cycling)
# ---------------------------------------------------------------------------
#
# REGISTER_REWRITE_PROMPT 는 register 한 축만 spoil 합니다. 학습 모델이
# CEFR / locale / pedagogy / accuracy / topic 도 학습하려면 그 축들에
# 대한 preference signal 이 필요한데, on_policy_pairs 만으로는 (judge
# margin 으로 걸러져) 양이 부족합니다. 그래서 offline rewrite 단계에서
# SFT id 를 해시해 6개 axis 를 cyclic 하게 배정합니다.
#
# 6개 axis 는 ``schemas.RejectionAxis`` 와 1:1 매핑됩니다:
#   - register_unnatural : 너무 formal/stiff/textbook
#   - cefr_mismatch      : CEFR level 어긋남 (too_hard/too_easy)
#   - locale_violation   : locale 고유명사를 avoid_cultures 쪽으로 교체
#   - pedagogy_weak      : lecturing/scolding/rule-dumping
#   - accuracy_error     : 문법/사실 오류 1-2개 삽입
#   - off_topic          : 사용자 발화 무시, 주제 이탈
#
# axis_instructions 는 axis 별로 다른 텍스트가 ``.format()`` 으로 들어가며,
# cefr_mismatch 만 ``cefr_mismatch_too_hard`` / ``cefr_mismatch_too_easy``
# 두 sub-key 를 가집니다 (caller 가 sft id + level 로 방향을 결정).

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
        "Drift OFF-TOPIC mid-reply:\n"
        "  - Do NOT acknowledge what the learner just said.\n"
        "  - Pivot to an unrelated subject (e.g. learner asked about market "
        "vegetables, tutor talks about weather forecasts next week).\n"
        "  - Or respond to something the learner did NOT say.\n"
        "Keep CEFR register, grammar, and locale items intact - only the "
        "TOPIC link to the prior turn is broken."
    ),
    "language_violation": (
        "Make the tutor BREAK OUT OF ENGLISH. The tutor is supposed to "
        "speak English only; rewrite the reply so it leaves English "
        "fully or partly. Pick ONE pattern:\n"
        "  - Translate the ENTIRE reply into the learner's native "
        "language (for {country}: the dominant L1 of that country - "
        "Mandarin Chinese for China, Japanese for Japan, Korean for "
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


# 6개 schema axis. caller 는 SFT id 해시로 이 중 하나를 고릅니다.
SPOIL_AXES: tuple[str, ...] = (
    "register_unnatural",
    "cefr_mismatch",
    "locale_violation",
    "pedagogy_weak",
    "accuracy_error",
    "off_topic",
    "language_violation",
    "persona_break",
)


# ---------------------------------------------------------------------------
# 5) EVALUATION_GENERATION_PROMPT  -  /think 평가 예시 생성
# ---------------------------------------------------------------------------

_EVALUATION_GENERATION_PROMPT = (
    """\
You are an English examiner producing a CEFR evaluation of a short
conversation transcript between an English tutor and {country_adjective}
{learner_description}. Your output will train a Qwen3 model running in
/think mode to evaluate learners. EVERY detail must remain locale-authentic
and consistent with the conversation.

You will receive the full dialogue and the learner's target CEFR level.
Produce a <think>...</think> block in which you reason carefully about
the learner's USER turns (citing turn indices and short quotations),
followed IMMEDIATELY by a single JSON object conforming to the
EvaluationOutput schema below.

----------------------------------------------------------------------
"""
    + "{locale_instruction_block}"
    + """

----------------------------------------------------------------------
INPUT
----------------------------------------------------------------------
Target CEFR level: {target_cefr}

Full dialogue:
{full_dialogue_json}

----------------------------------------------------------------------
EVALUATION RUBRIC
----------------------------------------------------------------------
Score these four dimensions on a 1-5 scale (5 = best at this CEFR level):
  - fluency       : pacing, hesitation, naturalness of phrasing
  - accuracy      : grammar correctness, tense, articles, agreement
  - vocabulary    : range, appropriateness, collocation
  - interaction   : turn-taking, follow-up questions, engagement

Also determine the learner's overall_cefr_estimate from the same scale.
It may equal, exceed, or fall below the target_cefr.

Specific feedback should be 2-4 concrete, actionable items keyed to
specific turn indices and short quotations. Severity is "minor",
"moderate", or "major".

Strengths: 1-3 short, concrete observations.

Suggested practice: ONE practice activity grounded in {country_adjective}
contexts the learner will recognize. Do NOT suggest {avoid_cultures_phrase}-context
exercises.

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------
A <think>...</think> block, immediately followed by a single JSON
object. No prose before <think>, no prose between </think> and the
opening "{{", no markdown code fences.

Inside <think>, cite specific turns: e.g. "Turn 3 user: 'I goed there'
shows past-tense regularization, typical at A2."

The JSON shape (EvaluationOutput):

{{
  "overall_cefr_estimate": "...",
  "scores": {{
    "fluency": 0, "accuracy": 0, "vocabulary": 0, "interaction": 0
  }},
  "specific_feedback": [
    {{
      "turn_index": 0,
      "user_text": "...",
      "issue": "...",
      "correction": "...",
      "level": "...",
      "severity": "..."
    }}
  ],
  "strengths": ["...", "..."],
  "suggested_practice": "..."
}}

Now produce your <think> block and EvaluationOutput JSON for the
dialogue above.
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
# Deployment system prompt (배포 시점 system prompt)
#
# 학습 + 추론 시점에 동일하게 모델이 받는 system 프롬프트. system_prompt
# 필드에 저장되어 SFTExample / DPOExample 마다 따라다닙니다. 데이터 생성
# 프롬프트가 아니라 학습/추론용입니다.
# ---------------------------------------------------------------------------

_DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE = """\
You are a patient English conversation tutor for {country_adjective}
{learner_description}. The learner you are speaking with is working at
CEFR level {cefr_level}. Hold yourself inside that register: keep
vocabulary, grammar, and sentence length appropriate for {cefr_level}
unless the learner reaches higher and sustains it.

Ground every conversation in {country_adjective} daily life. When you
refer to places, foods, neighborhoods, transit, currency, or cultural
items, draw on your own knowledge of {country}. Do not default to
{avoid_cultures_phrase} names, places, foods, or brands.

{avoided_topics_sentence} If the learner brings any of these up, briefly
acknowledge what they said and pivot to a safe adjacent topic without
lecturing or breaking the conversational frame.

Sound like a real person, not a textbook. Ask follow-up questions, share
small reactions, and let the conversation breathe. When the learner
makes a small mistake, gently recast the correct form inside your reply
at A1-A2; at B1 and above you can briefly explain or ask a clarifying
question if it would help. Do not use bullet lists, headings, or
numbered steps in your replies.
"""

DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE = _localize(_DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE)


_EVALUATION_SYSTEM_PROMPT = """\
You are an English examiner assessing the CEFR level of {country_adjective}
{learner_description} from a short conversation transcript.
Given the transcript and a target CEFR level, produce a <think>...</think>
block in which you reason carefully about the learner's USER turns
(citing turn indices and short quotations), followed immediately by a
single JSON object conforming to the EvaluationOutput schema
(overall_cefr_estimate, scores {fluency, accuracy, vocabulary,
interaction} in [1,5], specific_feedback, strengths, suggested_practice).

When you suggest practice activities, anchor them in {country_adjective}
contexts the learner will recognize. Do not recommend {avoid_cultures_phrase}-context
exercises. Output no prose before <think>, no prose between </think>
and the opening "{", and no markdown code fences.
"""

EVALUATION_SYSTEM_PROMPT = _localize(_EVALUATION_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Multi-locale prompt registry
# ---------------------------------------------------------------------------
#
# 각 generation 프롬프트의 raw 템플릿과 post-processing 모드를 등록합니다.
# generator 가 ``render_prompt(name, locale_name=seed.locale)`` 로 호출하면
# 해당 locale 의 LocaleConfig 로 placeholder 가 치환된 완성 템플릿이
# 돌아옵니다. 그 후 ``.format(scenario_json=..., level=..., ...)`` 으로 동적
# placeholder 를 채우면 teacher 에 보낼 system 프롬프트가 완성됩니다.

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
    out = _localize_with(raw, loc)
    if mode == "no_think":
        return _with_no_think(out)
    if mode == "think":
        return _with_think(out)
    return out


def render_deployment_system_prompt(
    cefr_level: str, locale_name: str | None = None
) -> str:
    """배포 system 프롬프트를 주어진 CEFR 레벨 + locale 로 렌더.

    ``locale_name`` 을 생략하면 ``config/locale.yaml`` 의 ``default_locale``
    이 쓰입니다. 다중 locale 학습 시에는 각 example 의 ``metadata.locale``
    을 전달해서 그 국가의 system 프롬프트가 생성되게 하세요.

    데이터 생성 시점에 ``system_prompt`` 필드를 채울 때, 그리고 추론 시점에
    배포 모델에 prompt 를 줄 때 동일한 텍스트가 되도록 늘 이 헬퍼를 통하세요.
    """
    from qwen_tutor.locale import get_locale

    loc = get_locale(locale_name)
    template = _localize_with(_DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE, loc)
    return template.format(cefr_level=cefr_level)


def render_evaluation_system_prompt(locale_name: str | None = None) -> str:
    """평가 system 프롬프트를 주어진 locale 로 렌더.

    배포 프롬프트와 달리 동적 placeholder (``{cefr_level}`` 등) 가 없어서
    locale 만 정하면 곧바로 완성된 문자열이 됩니다.
    """
    from qwen_tutor.locale import get_locale

    loc = get_locale(locale_name)
    return _localize_with(_EVALUATION_SYSTEM_PROMPT, loc)
