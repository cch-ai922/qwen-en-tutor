"""locale.py  -  config/locale.yaml 을 읽어 프롬프트/필터/메트릭이 공유하는 상수 제공.

이 모듈을 import 하는 순간 ``config/locale.yaml`` 을 한 번 로드해서
``LOCALE`` 싱글톤에 담아 둡니다. 다른 국가로 바꾸려면 yaml 만 수정하고
Python 을 다시 시작하면 됩니다.

다른 모듈은 이 파일이 노출하는 다음 헬퍼만 알면 됩니다.

    LOCALE.country                  → "Iran"        (e.g. config/locale.yaml 값)
    LOCALE.country_adjective        → "Iranian"
    LOCALE.learner_description      → "adult learners of English"
    LOCALE.avoided_topics           → tuple[AvoidedTopic, ...]
    LOCALE.avoided_topic_names      → tuple[str, ...]   (redirect_axis 후보)
    LOCALE.locale_instruction_block → 모든 generation 프롬프트 위에 붙는 블록
    LOCALE.deployment_locale_block  → 배포 system 프롬프트의 한 문단
    LOCALE.judge_locale_block       → LocaleLLMJudge 가 사용하는 평가 기준
    LOCALE.format_kwargs            → str.format 에 그대로 unpack 할 수 있는 dict

설계 노트:
* 도시/음식/교통 같은 정적 리스트는 두지 않습니다. teacher 모델이 country
  이름만으로 자기 지식을 충분히 활용할 수 있다고 보고 그쪽에 위임합니다.
* avoided_topics 가 redirect_axis / banned-topic / judge 기준의 단일
  source of truth 입니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

# 환경 변수로 경로 override 가능. (테스트에서 다른 yaml 을 가리키게 할 때 유용)
DEFAULT_LOCALE_PATH = Path(
    os.environ.get("QWEN_TUTOR_LOCALE_CONFIG", "config/locale.yaml")
)


@dataclass(frozen=True)
class AvoidedTopic:
    """단일 회피 주제."""

    name: str
    pivot_hint: str


@dataclass(frozen=True)
class LocaleConfig:
    """locale.yaml 한 파일의 메모리 표현."""

    country: str
    country_adjective: str
    learner_description: str
    avoided_topics: tuple[AvoidedTopic, ...]
    avoid_default_cultures: tuple[str, ...] = ()
    # diversity tracker 가 사용할 음식 어휘. spaCy 가 FOOD entity 를 만들지
    # 않기 때문에 직접 lowercase 매칭이 필요한 단어들만 둡니다. country 의
    # 대표 음식 (in-locale) 과 회피용 (out-of-locale) 음식을 분리해 두면
    # 다양성 리포트의 top_foods 가 어느 쪽에 치우쳐 있는지 보기 좋습니다.
    food_terms: tuple[str, ...] = ()
    avoid_food_terms: tuple[str, ...] = ()

    # -------------------------------------------------------------------------
    # 파생 헬퍼
    # -------------------------------------------------------------------------

    @cached_property
    def avoided_topic_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.avoided_topics)

    @cached_property
    def locale_instruction_header(self) -> str:
        return f"{self.country.upper()} LOCALE INSTRUCTION:"

    @cached_property
    def avoid_cultures_phrase(self) -> str:
        """``avoid_default_cultures`` 를 ``"American/European/Japanese"`` 같은 짧은
        영어 구로 변환. 비어 있으면 fallback 으로 "Western" 을 씁니다.
        """
        cultures = [c.strip() for c in self.avoid_default_cultures if c.strip()]
        if not cultures:
            return "Western"
        if len(cultures) == 1:
            return cultures[0]
        if len(cultures) == 2:
            return f"{cultures[0]} or {cultures[1]}"
        return ", ".join(cultures[:-1]) + f", or {cultures[-1]}"

    @cached_property
    def locale_instruction_block(self) -> str:
        """모든 generation 프롬프트 상단에 붙는 locale 지시 블록.

        구체적인 도시/음식 리스트는 두지 않고 teacher 의 지식에 위임합니다.
        대신 명확하게 ``{country}`` 의 본토 지식을 활용하고, ``avoid_default_cultures``
        에 적힌 문화로 떨어지지 말라는 점만 강하게 반복합니다.
        """
        country = self.country
        adj = self.country_adjective
        avoid = self.avoid_cultures_phrase
        return (
            f"{self.locale_instruction_header}\n"
            f"- All proper nouns (people, cities, foods, brands, neighborhoods,\n"
            f"  universities, transit lines, holidays) must be authentically {adj}.\n"
            f"  Draw on your own knowledge of {country}.\n"
            f"- ALL output must be in English using the Latin alphabet. Render\n"
            f"  names, places, foods, and cultural items in ROMANIZED form\n"
            f"  (e.g. \"Li Na\" not \"李娜\"; \"Tanaka\" not \"田中\"; \"Tokyo\" not\n"
            f"  \"東京\"; \"Kim Min-su\" not \"김민수\"; \"Moscow\" not \"Москва\").\n"
            f"  Do NOT insert any CJK / Cyrillic / Arabic / Devanagari / other\n"
            f"  non-Latin characters anywhere — not in role names, not in\n"
            f"  message content, not in setting descriptions.\n"
            f"- Use a mix of {country}'s cities and smaller towns; do NOT default\n"
            f"  to the capital for every scenario, and vary settings widely.\n"
            f"- Use a mix of common {adj} first names across genders and generations.\n"
            f"- When food, drink, currency, transit, or cultural items come up,\n"
            f"  choose authentic {adj} ones rather than {avoid} items.\n"
            f"- Do NOT use {avoid} place names or personal names. Do NOT default\n"
            f"  to {avoid} brands, foods, or chains.\n"
            f"- If you find yourself reaching for {avoid} defaults out of habit,\n"
            f"  STOP and substitute an authentic {adj} item from your knowledge."
        )

    @cached_property
    def avoided_topics_sentence(self) -> str:
        """회피 주제를 한 줄 문장으로 - "Stay clear of A, B, and C." 형태."""
        names = [self._humanize(t.name) for t in self.avoided_topics]
        if not names:
            return ""
        if len(names) == 1:
            joined = names[0]
        elif len(names) == 2:
            joined = f"{names[0]} and {names[1]}"
        else:
            joined = ", ".join(names[:-1]) + f", and {names[-1]}"
        return f"Stay clear of {joined}."

    @cached_property
    def avoided_topics_block_for_redirect_prompt(self) -> str:
        """redirect 프롬프트가 어떤 축으로 어떻게 pivot 할지 알려 주는 블록."""
        lines = []
        for t in self.avoided_topics:
            lines.append(f'  - "{t.name}": pivot toward {t.pivot_hint}.')
        names_quoted = ", ".join(f'"{t.name}"' for t in self.avoided_topics)
        return (
            f"The redirect_axis is one of: {names_quoted}. Treat it as the kind\n"
            f"of moment the learner introduces somewhere mid-dialogue. When you\n"
            f"reach the redirect, the assistant should briefly acknowledge what\n"
            f"the learner said and pivot to a safe adjacent topic without\n"
            f"lecturing or breaking the conversational frame:\n" + "\n".join(lines)
        )

    @cached_property
    def deployment_locale_block(self) -> str:
        """배포 system 프롬프트에서 country/avoided topics 부분 한 문단.

        ``config/training.yaml`` 의 ``deployment_system_prompt_template`` 안에
        ``{deployment_locale_block}`` 으로 끼워 넣습니다.
        """
        country = self.country
        adj = self.country_adjective
        return (
            f"Ground every conversation in {adj} daily life. When you refer to\n"
            f"places, foods, neighborhoods, transit, currency, or cultural items,\n"
            f"draw on your own knowledge of {country}. Do not default to Western\n"
            f"names, places, foods, or brands.\n\n"
            f"{self.avoided_topics_sentence} If the learner brings any of these up,\n"
            f"briefly acknowledge what they said and pivot to a safe adjacent topic\n"
            f"without lecturing or breaking the conversational frame."
        )

    @cached_property
    def judge_locale_block(self) -> str:
        """LocaleLLMJudge 가 사용하는 평가 기준."""
        adj = self.country_adjective
        country = self.country
        avoid = self.avoid_cultures_phrase
        avoided = self.avoided_topics_sentence
        return (
            f"You are a strict reviewer checking whether a conversation feels\n"
            f"culturally authentic for {adj} {self.learner_description}.\n"
            f"A passing dialogue grounds places, people, foods, transit,\n"
            f"currency, and cultural rhythms in {country}, drawing on real {adj}\n"
            f"knowledge. A failing dialogue defaults to {avoid} names, places,\n"
            f"foods, or brands, or feels generic/non-{adj}.\n\n"
            f"The assistant must also obey content boundaries: {avoided}"
        )

    @property
    def format_kwargs(self) -> dict[str, str]:
        """``template.format(**LOCALE.format_kwargs)`` 한 번에 끝낼 수 있게 묶음."""
        return {
            "country": self.country,
            "country_adjective": self.country_adjective,
            "learner_description": self.learner_description,
            "locale_instruction_block": self.locale_instruction_block,
            "avoided_topics_sentence": self.avoided_topics_sentence,
            "avoided_topics_block_for_redirect_prompt": self.avoided_topics_block_for_redirect_prompt,
            "deployment_locale_block": self.deployment_locale_block,
            "judge_locale_block": self.judge_locale_block,
        }

    def localize(self, template: str) -> str:
        """``str.replace`` 로 locale placeholder 를 LOCALE 값으로 치환.

        ``.format()`` 와 충돌하지 않도록 단순 ``replace`` 를 씁니다. JSON 본문의
        ``{{`` / ``}}`` escape 는 그대로 보존됩니다. 동적 placeholder 인
        ``{cefr_level}`` 같은 것은 손대지 않으므로, 사용자는 반환된 문자열을
        다시 ``.format(cefr_level=...)`` 로 채우면 됩니다.
        """
        return (
            template.replace("{country_adjective}", self.country_adjective)
            .replace("{country}", self.country)
            .replace("{learner_description}", self.learner_description)
            .replace("{avoided_topics_sentence}", self.avoided_topics_sentence)
            .replace(
                "{avoided_topics_block_for_redirect_prompt}",
                self.avoided_topics_block_for_redirect_prompt,
            )
        )

    # -------------------------------------------------------------------------
    # 유틸
    # -------------------------------------------------------------------------

    @staticmethod
    def _humanize(slug: str) -> str:
        """snake_case 축 이름을 사람이 읽기 좋은 문구로 변환.

        ``"politics"`` → ``"politics"``,  ``"alcohol_dating"`` → ``"alcohol/dating"``,
        ``"partisan_history"`` → ``"partisan history"``.
        둘 이상의 단어 중 어느 한 쪽이 다른 쪽에 종속되면 공백, 두 주제가
        대등하게 묶여 있으면 "/" 로 표현해야 자연스럽지만 일률 처리하기
        어려우므로 그냥 underscore 만 공백으로 바꿉니다. 더 자연스러운
        영어가 필요하면 ``avoided_topics`` 의 ``name`` 자체를 그렇게 적어
        주세요 (예: ``alcohol_or_dating``).
        """
        return slug.replace("_", " ")


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------


def _parse_single_locale_block(p: Path, block: dict[str, Any], label: str) -> LocaleConfig:
    """단일 locale dict 를 LocaleConfig 로 변환. ``label`` 은 에러 메시지용
    (예: ``locales.china`` 또는 ``<top-level>``)."""
    country = block.get("country")
    country_adj = block.get("country_adjective")
    learner_desc = block.get("learner_description")
    if not country or not country_adj:
        raise ValueError(
            f"{p}: {label} 의 'country' 와 'country_adjective' 는 필수입니다."
        )
    if not learner_desc:
        learner_desc = "adult learners of English"

    raw_topics = block.get("avoided_topics") or []
    topics: list[AvoidedTopic] = []
    for i, t in enumerate(raw_topics):
        if not isinstance(t, dict):
            raise ValueError(
                f"{p}: {label}.avoided_topics[{i}] 는 dict 여야 합니다 (name, pivot_hint)."
            )
        name = t.get("name")
        hint = t.get("pivot_hint", "")
        if not name:
            raise ValueError(f"{p}: {label}.avoided_topics[{i}].name 누락.")
        topics.append(AvoidedTopic(name=str(name), pivot_hint=str(hint)))

    raw_avoid = block.get("avoid_default_cultures") or []
    if not isinstance(raw_avoid, list):
        raise ValueError(f"{p}: {label}.avoid_default_cultures 는 리스트여야 합니다.")
    avoid_cultures = tuple(
        str(c).strip() for c in raw_avoid if isinstance(c, str) and c.strip()
    )

    def _strlist(field_name: str) -> tuple[str, ...]:
        raw = block.get(field_name) or []
        if not isinstance(raw, list):
            raise ValueError(f"{p}: {label}.{field_name} 는 리스트여야 합니다.")
        return tuple(str(x).strip() for x in raw if isinstance(x, str) and x.strip())

    return LocaleConfig(
        country=str(country),
        country_adjective=str(country_adj),
        learner_description=str(learner_desc),
        avoided_topics=tuple(topics),
        avoid_default_cultures=avoid_cultures,
        food_terms=_strlist("food_terms"),
        avoid_food_terms=_strlist("avoid_food_terms"),
    )


def load_locales(
    path: str | Path | None = None,
) -> tuple[dict[str, LocaleConfig], str]:
    """``config/locale.yaml`` 을 (locales-by-name, default_name) 으로 변환.

    두 가지 YAML 모양을 모두 받습니다:

      1) Multi-locale (권장):
         ::
            default_locale: china
            locales:
              china: { country: "China", country_adjective: "Chinese", ... }
              japan: { country: "Japan", country_adjective: "Japanese", ... }
              italy: { country: "Italy", country_adjective: "Italian", ... }

      2) Single locale (호환):
         ::
            country: "China"
            country_adjective: "Chinese"
            ...

    Single 모양은 ``"default"`` 키 하나만 가진 dict 로 wrap 되어 반환됩니다.
    """
    p = Path(path) if path else DEFAULT_LOCALE_PATH
    with p.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

    if "locales" in doc and isinstance(doc["locales"], dict):
        raw_locales = doc["locales"]
        if not raw_locales:
            raise ValueError(f"{p}: 'locales' 가 비어 있습니다.")
        out: dict[str, LocaleConfig] = {}
        for name, block in raw_locales.items():
            if not isinstance(block, dict):
                raise ValueError(f"{p}: locales.{name} 는 dict 여야 합니다.")
            out[str(name)] = _parse_single_locale_block(p, block, f"locales.{name}")
        default_name = doc.get("default_locale")
        if default_name is None:
            default_name = next(iter(out))
        else:
            default_name = str(default_name)
            if default_name not in out:
                raise ValueError(
                    f"{p}: default_locale='{default_name}' 가 locales 에 없습니다 "
                    f"(있는 키: {list(out)})."
                )
        return out, default_name

    # Single-locale 호환 경로. 전체 doc 를 하나의 블록으로 처리.
    single = _parse_single_locale_block(p, doc, "<top-level>")
    return {"default": single}, "default"


def load_locale(path: str | Path | None = None) -> LocaleConfig:
    """Back-compat: 기본 locale 하나만 돌려주는 단축 헬퍼."""
    locales, default_name = load_locales(path)
    return locales[default_name]


# ---------------------------------------------------------------------------
# 모듈-load-time 싱글톤
# ---------------------------------------------------------------------------

# Import 시점에 yaml 을 한 번 읽어 둡니다. 깨졌다면 import 도 실패
# (silent default 보다 빠른 fail-fast 가 안전).
LOCALES: dict[str, LocaleConfig]
DEFAULT_LOCALE_NAME: str
LOCALES, DEFAULT_LOCALE_NAME = load_locales()

# ``LOCALE`` 은 기본 locale 을 가리키는 back-compat alias. 코드를 점진적으로
# multi-locale 로 옮기는 동안 기존 호출 ``LOCALE.country`` 등은 그대로 동작.
LOCALE: LocaleConfig = LOCALES[DEFAULT_LOCALE_NAME]


def get_locale(name: str | None = None) -> LocaleConfig:
    """이름으로 locale 을 가져옵니다. ``name`` 이 ``None`` 이거나 빈 문자열이면
    기본 locale 을 돌려줍니다.
    """
    if not name:
        return LOCALE
    if name not in LOCALES:
        raise KeyError(
            f"unknown locale '{name}'. config/locale.yaml 에 정의된 locale: "
            f"{list(LOCALES)}"
        )
    return LOCALES[name]


def list_locales() -> tuple[str, ...]:
    """등록된 locale 이름 목록 (안정적인 정렬 순서)."""
    return tuple(LOCALES.keys())
