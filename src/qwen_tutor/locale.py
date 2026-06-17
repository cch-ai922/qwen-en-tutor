"""locale.py  -  load config/locale.yaml and provide shared constants for prompts, filters, and metrics.

This module reads ``config/locale.yaml`` once at import time and stores it in the
``LOCALE`` singleton. To switch countries, update the YAML and restart Python.

Other modules only need to know the following helpers exposed by this file:

    LOCALE.country                  → "Iran"        (e.g. value from config/locale.yaml)
    LOCALE.country_adjective        → "Iranian"
    LOCALE.learner_description      → "adult learners of English"
    LOCALE.avoided_topics           → tuple[AvoidedTopic, ...]
    LOCALE.avoided_topic_names      → tuple[str, ...]   (redirect_axis candidates)
    LOCALE.locale_instruction_block → block prepended to all generation prompts
    LOCALE.deployment_locale_block  → one paragraph for the deployment system prompt
    LOCALE.judge_locale_block       → evaluation criteria used by LocaleLLMJudge
    LOCALE.format_kwargs            → dict that can be unpacked directly into str.format()

Design notes:
* Do not maintain static city/food/transportation lists. We rely on the teacher
  model to use its knowledge from the country name alone.
* ``avoided_topics`` is the single source of truth for redirect_axis, banned-topic,
  and judge criteria.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

# Allow overriding the path via environment variable. Useful in tests when
# pointing at a different YAML file.
DEFAULT_LOCALE_PATH = Path(
    os.environ.get("QWEN_TUTOR_LOCALE_CONFIG", "config/locale.yaml")
)


@dataclass(frozen=True)
class AvoidedTopic:
    """A single avoided topic."""

    name: str
    pivot_hint: str


@dataclass(frozen=True)
class LocaleConfig:
    """In-memory representation of one locale.yaml file."""

    country: str
    country_adjective: str
    learner_description: str
    avoided_topics: tuple[AvoidedTopic, ...]
    avoid_default_cultures: tuple[str, ...] = ()
    # Food vocabulary used by the diversity tracker. spaCy does not reliably
    # create FOOD entities, so only include words that need direct lowercase
    # matching. Separating in-locale staple foods from out-of-locale avoidance
    # foods makes it easier to see which side the top_foods report favors.
    food_terms: tuple[str, ...] = ()
    avoid_food_terms: tuple[str, ...] = ()

    # -------------------------------------------------------------------------
    # Derived helpers
    # -------------------------------------------------------------------------

    @cached_property
    def avoided_topic_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.avoided_topics)

    @cached_property
    def locale_instruction_header(self) -> str:
        # Code-marker style (lowercase, bracketed, hyphenated) instead of the
        # older all-caps "CHINA LOCALE INSTRUCTION:" form. Small teachers
        # (4B Q4) habitually echo all-caps prose headers verbatim into their
        # <think> reasoning, which then trips the scaffolding_leakage banned-
        # terms filter and kills the eval example. Code markers don't trigger
        # the same echo reflex. Filter coverage for the new format lives in
        # config/banned_terms.yaml -> scaffolding_leakage.
        return f"[locale-rules:{self.country.lower()}]"

    @cached_property
    def avoid_cultures_phrase(self) -> str:
        """Convert ``avoid_default_cultures`` into a short English phrase like
        ``"American/European/Japanese"``. Use "Western" if the list is empty.
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
        """Locale instruction block prepended to all generation prompts.

        Strict-Latin variant: forbids any non-Latin script anywhere. This is
        the default used by every prompt EXCEPT ``dialogue_language_redirect``
        with ``speaks_l1`` trigger, which needs one user turn in native L1
        script. See ``locale_instruction_block_allow_l1`` for that exception.
        """
        return self._build_locale_instruction_block(strict_latin=True)

    @property
    def locale_instruction_block_allow_l1(self) -> str:
        """Variant locale block that ALLOWS one user turn in native L1 script.

        Used only by ``dialogue_language_redirect`` to support the
        ``speaks_l1`` trigger. Without this carve-out, the strict-Latin
        paragraph in the default block contradicts the speaks_l1 override
        inside the language_redirect prompt — the teacher reads the global
        rule first and ignores the per-prompt override, leaving every
        speaks_l1 example with no L1 turn (caught by speaks_l1_sanity at
        100% rejection). With this variant the global rule is relaxed but
        non-L1 leakage is still tightly bounded to a single user turn by
        the prompt itself.

        Everything else (authentic proper nouns, avoid Western defaults,
        city variety, etc.) is unchanged from the strict variant.
        """
        return self._build_locale_instruction_block(strict_latin=False)

    def _build_locale_instruction_block(self, strict_latin: bool) -> str:
        """Shared body of the two locale-block variants.

        Do not enumerate concrete city/food lists; defer to the teacher model's
        knowledge. Instead, clearly instruct using native knowledge of
        ``{country}`` and repeatedly warn against falling back to cultures
        listed in ``avoid_default_cultures``.
        """
        country = self.country
        adj = self.country_adjective
        avoid = self.avoid_cultures_phrase
        # Strict variant forbids non-Latin everywhere. Relaxed variant allows
        # ONE user turn in native L1 script (scoped tight by the calling
        # prompt's own override; this only removes the global ban).
        if strict_latin:
            latin_rule = (
                f"- ALL output must be in English using the Latin alphabet. Render\n"
                f"  names, places, foods, and cultural items in ROMANIZED form\n"
                f"  (e.g. \"Li Na\" not \"李娜\"; \"Tanaka\" not \"田中\"; \"Tokyo\" not\n"
                f"  \"東京\"; \"Kim Min-su\" not \"김민수\"; \"Moscow\" not \"Москва\").\n"
                f"  Do NOT insert any CJK / Cyrillic / Arabic / Devanagari / other\n"
                f"  non-Latin characters anywhere — not in role names, not in\n"
                f"  message content, not in setting descriptions.\n"
            )
        else:
            latin_rule = (
                f"- The TUTOR's turns must be in English using the Latin alphabet.\n"
                f"  The LEARNER's English turns must also use Latin script. ONE\n"
                f"  user turn may be in the {adj} learner's L1 using native\n"
                f"  script (this is the speaks_l1 case, explicitly required by\n"
                f"  the prompt body); all OTHER user turns and ALL tutor turns\n"
                f"  remain Latin-only. Render proper nouns (names, places,\n"
                f"  foods, brands) in ROMANIZED form in every Latin-script turn\n"
                f"  (e.g. \"Li Na\" not \"李娜\"; \"Tanaka\" not \"田中\").\n"
            )
        return (
            f"{self.locale_instruction_header}\n"
            f"- All proper nouns (people, cities, foods, brands, neighborhoods,\n"
            f"  universities, transit lines, holidays) must be authentically {adj}.\n"
            f"  Draw on your own knowledge of {country}.\n"
            f"{latin_rule}"
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
        """Convert avoided topics into a one-line sentence, e.g. "Stay clear of A, B, and C."."""
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
        """Block explaining how the redirect prompt should pivot along each axis."""
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
        """One paragraph for the country/avoided-topics section of the
        deployment system prompt.
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
        """Evaluation criteria used by LocaleLLMJudge."""
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
        """Bundle locale values so ``template.format(**LOCALE.format_kwargs)``
        works in one pass."""
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
        """Replace locale placeholders with LOCALE values using ``str.replace``.

        Use simple ``replace`` to avoid conflicts with ``.format()``. JSON
        literal ``{{`` / ``}}`` escapes are preserved. Dynamic placeholders like
        ``{cefr_level}`` are left alone, so callers can later call
        ``.format(cefr_level=...)`` on the returned string.
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
    # Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _humanize(slug: str) -> str:
        """Convert a snake_case slug into a more readable label.

        ``"politics"`` → ``"politics"``,  ``"alcohol_dating"`` → ``"alcohol/dating"``,
        ``"partisan_history"`` → ``"partisan history"``.
        If one word modifies another, spaces may be more natural; if both words
        are equal, "/" may be more natural. For simplicity, this helper just
        replaces underscores with spaces. To get more natural English, write
        the exact ``name`` in ``avoided_topics`` (for example,
        ``alcohol_or_dating``).
        """
        return slug.replace("_", " ")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _parse_single_locale_block(p: Path, block: dict[str, Any], label: str) -> LocaleConfig:
    """Convert a single locale dict into a LocaleConfig. ``label`` is used for
    error messages (for example, ``locales.china`` or ``<top-level>``)."""
    country = block.get("country")
    country_adj = block.get("country_adjective")
    learner_desc = block.get("learner_description")
    if not country or not country_adj:
        raise ValueError(
            f"{p}: {label}: 'country' and 'country_adjective' are required."
        )
    if not learner_desc:
        learner_desc = "adult learners of English"

    raw_topics = block.get("avoided_topics") or []
    topics: list[AvoidedTopic] = []
    for i, t in enumerate(raw_topics):
        if not isinstance(t, dict):
            raise ValueError(
                f"{p}: {label}.avoided_topics[{i}] must be a dict (name, pivot_hint)."
            )
        name = t.get("name")
        hint = t.get("pivot_hint", "")
        if not name:
            raise ValueError(f"{p}: {label}.avoided_topics[{i}].name is missing.")
        topics.append(AvoidedTopic(name=str(name), pivot_hint=str(hint)))

    raw_avoid = block.get("avoid_default_cultures") or []
    if not isinstance(raw_avoid, list):
        raise ValueError(f"{p}: {label}.avoid_default_cultures must be a list.")
    avoid_cultures = tuple(
        str(c).strip() for c in raw_avoid if isinstance(c, str) and c.strip()
    )

    def _strlist(field_name: str) -> tuple[str, ...]:
        raw = block.get(field_name) or []
        if not isinstance(raw, list):
            raise ValueError(f"{p}: {label}.{field_name} must be a list.")
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
    """Load ``config/locale.yaml`` and return (locales-by-name, default_name).

    Two YAML shapes are accepted:

      1) Multi-locale (recommended):
         ::
            default_locale: china
            locales:
              china: { country: "China", country_adjective: "Chinese", ... }
              japan: { country: "Japan", country_adjective: "Japanese", ... }
              italy: { country: "Italy", country_adjective: "Italian", ... }

      2) Single locale (compatible):
         ::
            country: "China"
            country_adjective: "Chinese"
            ...

    The single shape is wrapped into a dict containing only the ``"default"`` key.
    """
    p = Path(path) if path else DEFAULT_LOCALE_PATH
    with p.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

    if "locales" in doc and isinstance(doc["locales"], dict):
        raw_locales = doc["locales"]
        if not raw_locales:
            raise ValueError(f"{p}: 'locales' is empty.")
        out: dict[str, LocaleConfig] = {}
        for name, block in raw_locales.items():
            if not isinstance(block, dict):
                raise ValueError(f"{p}: locales.{name} must be a dict.")
            out[str(name)] = _parse_single_locale_block(p, block, f"locales.{name}")
        default_name = doc.get("default_locale")
        if default_name is None:
            default_name = next(iter(out))
        else:
            default_name = str(default_name)
            if default_name not in out:
                raise ValueError(
                    f"{p}: default_locale='{default_name}' is not defined in locales "
                    f"(available keys: {list(out)})."
                )
        return out, default_name

    # Single-locale compatibility path. Treat the entire document as one block.
    single = _parse_single_locale_block(p, doc, "<top-level>")
    return {"default": single}, "default"


def load_locale(path: str | Path | None = None) -> LocaleConfig:
    """Back-compat: helper that returns a single default locale."""
    locales, default_name = load_locales(path)
    return locales[default_name]


# ---------------------------------------------------------------------------
# Module load-time singleton
# ---------------------------------------------------------------------------

# Read the YAML once at import time. If the file is broken, import fails too.
# This fail-fast behavior is safer than silently falling back to defaults.
LOCALES: dict[str, LocaleConfig]
DEFAULT_LOCALE_NAME: str
LOCALES, DEFAULT_LOCALE_NAME = load_locales()

# ``LOCALE`` is a back-compat alias for the default locale. Existing calls
# like ``LOCALE.country`` continue to work while migrating code to multi-locale.
LOCALE: LocaleConfig = LOCALES[DEFAULT_LOCALE_NAME]


def get_locale(name: str | None = None) -> LocaleConfig:
    """Get a locale by name. If ``name`` is ``None`` or empty, return the default locale."""
    if not name:
        return LOCALE
    if name not in LOCALES:
        raise KeyError(
            f"unknown locale '{name}'. locales defined in config/locale.yaml: "
            f"{list(LOCALES)}"
        )
    return LOCALES[name]


def list_locales() -> tuple[str, ...]:
    """Return the registered locale names in a stable order."""
    return tuple(LOCALES.keys())
