"""Locale LLM-judge filter.

The target country is read from ``config/locale.yaml``'s country /
country_adjective fields. A regex pulls capitalized proper-noun candidates
out of each assistant turn and a cheap judge model labels each entity as
one of ``in_locale`` / ``ambiguous`` / ``out_of_locale``. Verdicts are
cached in SQLite so the same entity is never re-judged.

score = ``(n_in_locale + AMBIGUOUS_WEIGHT * n_ambiguous) / n_total``

Failure conditions:
  * score < min_score (default 0.5), OR
  * any entity is judged ``out_of_locale``.

Default values (min_score=0.5, AMBIGUOUS_WEIGHT=0.7) reflect the reality
that low-level conversations like A2 contain few specific cities or
neighborhoods and the judge labels most candidates as ambiguous. An
``out_of_locale`` hit fails the example immediately regardless of weight,
so lowering the threshold never lets a genuinely off-locale dialogue
through.

Placed late in the pipeline so it only runs on examples that already
passed every mechanical filter.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from qwen_tutor.generation.filters.base import Filter, FilterableExample, FilterResult
from qwen_tutor.generation.teacher import TeacherClient
from qwen_tutor.locale import LOCALE
from qwen_tutor.schemas import Message
from qwen_tutor.utils.runner import extract_first_json
from qwen_tutor.utils.thinking import with_thinking_directive

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/_caches/locale_judge.sqlite")
DEFAULT_MIN_SCORE = 0.5
DEFAULT_AMBIGUOUS_WEIGHT = 0.7
VALID_VERDICTS = {"in_locale", "ambiguous", "out_of_locale"}


# Raw judge-prompt template with locale placeholders. ``_render_judge_header``
# substitutes them for a specific locale at call time. The default-locale-
# pre-rendered ``JUDGE_PROMPT_HEADER`` constant is kept for back-compat.
_JUDGE_PROMPT_HEADER_RAW = (
    "You classify proper nouns by cultural origin for a dataset of English\n"
    "dialogues whose intended audience is {country_adjective}\n"
    "{learner_description}.\n"
    "\n"
    "The input — a list of entities to classify — is delivered as the\n"
    "USER message immediately following these instructions. For each\n"
    "entity in that list, output one verdict from this set:\n\n"
    "  - \"in_locale\": clearly a {country_adjective} person name,\n"
    "    {country_adjective} place, {country_adjective}\n"
    "    neighborhood, {country_adjective} food or dish,\n"
    "    {country_adjective} institution, or culturally\n"
    "    {country_adjective} item.\n"
    "  - \"ambiguous\": a generic role (e.g. \"doctor\", \"teacher\",\n"
    "    \"neighbor\"), a common noun the NER tagger misclassified, or a\n"
    "    personal name that could plausibly be from many cultures.\n"
    "  - \"out_of_locale\": clearly NOT from {country} - in particular\n"
    "    {avoid_cultures_phrase} items, or anything else that is not\n"
    "    {country_adjective}.\n\n"
    "Output STRICT JSON ONLY, no prose, no markdown fences. Shape:\n\n"
    "{\"results\": [{\"entity\": \"<verbatim>\", \"verdict\": \"<one of the three>\"}, ...]}\n"
)


def _render_judge_header(locale_name: str | None = None) -> str:
    """Render the judge prompt header for the given locale (or default)."""
    from qwen_tutor.locale import get_locale

    loc = get_locale(locale_name)
    return (
        _JUDGE_PROMPT_HEADER_RAW
        .replace("{country_adjective}", loc.country_adjective)
        .replace("{country}", loc.country)
        .replace("{learner_description}", loc.learner_description)
        .replace("{avoid_cultures_phrase}", loc.avoid_cultures_phrase)
    )


JUDGE_PROMPT_HEADER = _render_judge_header()


# Regex-based proper-noun extraction so we don't depend on an external NER
# (spaCy). The candidates are sent directly to the LLM judge, which classifies
# each as in_locale / ambiguous / out_of_locale; false positives get labeled
# ambiguous by the judge (and the SQLite cache means each entity is only
# asked once).
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]{1,}(?:[\s\-][A-Z][a-zA-Z'\-]+){0,3})\b"
)

# For eval examples the assistant turn is ``<think>...</think>{json}``. The
# <think> block is private examiner reasoning that the trained student will
# never emit at deploy time — only the post-</think> JSON is the user-facing
# output that needs scanning. ``banned_terms`` and ``non_latin_script`` already
# slice to post-</think> for eval; ``locale_judge`` was missed, which caused
# the model's literal mentions of "JSON" / "American" / "European" inside its
# <think> reasoning to be flagged as out_of_locale entities.
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)

# Used to drop sentence-initial common-word false positives.
_PROPER_STOPWORDS: frozenset[str] = frozenset(
    s.lower()
    for s in (
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
        "Sunday", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Today", "Tomorrow", "Yesterday", "Tonight", "Morning", "Afternoon",
        "Evening", "Night", "Spring", "Summer", "Autumn", "Fall", "Winter",
        "The", "A", "An", "I", "You", "He", "She", "We", "They", "It",
        "This", "That", "These", "Those", "My", "Your", "His", "Her", "Our",
        "Their", "Some", "Any", "No", "Yes", "Oh", "Well", "OK", "Okay",
        "Sure", "Hi", "Hello", "Hey", "Bye", "Thanks", "Thank",
        "When", "Where", "Why", "How", "What", "Who", "Which", "If", "Or",
        "And", "But", "So", "Then", "There", "Here",
    )
)


# Language names. References like "let's keep this in English" or "say it
# in Mandarin" are meta-discussion about which language to use, NOT
# Western-default cultural leaks. The proper-noun regex picks language
# names up because they're capitalized; the judge then wrongly tags
# "English" as out_of_locale for a China-locale scenario. Skip exact
# matches at extraction. Multi-word phrases like "English breakfast"
# still flow through to the judge — only the bare language name is
# exempt.
_LANGUAGE_NAMES: frozenset[str] = frozenset(
    s.lower()
    for s in (
        "English", "Mandarin", "Cantonese", "Chinese", "Japanese",
        "Korean", "Italian", "Spanish", "French", "German", "Russian",
        "Arabic", "Portuguese", "Hindi", "Bengali", "Vietnamese",
        "Thai", "Indonesian", "Malay", "Tagalog", "Turkish", "Hebrew",
        "Persian", "Farsi", "Polish", "Dutch", "Greek", "Swedish",
        "Norwegian", "Danish", "Finnish", "Czech", "Hungarian",
        "Romanian", "Ukrainian", "Latin",
    )
)

# Compound language-policy phrases: "English-only", "Mandarin-only",
# "English-first", etc. The proper-noun regex captures these as one
# entity; treat the entire phrase as meta-discussion (same class as
# bare language names above) rather than an out-of-locale entity.
_LANGUAGE_POLICY_PHRASE_RE = re.compile(
    r"^(?:" + "|".join(re.escape(n) for n in (
        "English", "Mandarin", "Cantonese", "Chinese", "Japanese",
        "Korean", "Italian", "Spanish", "French", "German", "Russian",
    )) + r")[\-\s](?:only|first|speaking|native|fluent)$",
    re.IGNORECASE,
)


# Common English words that frequently appear capitalized at the start
# of a sentence and get caught by the proper-noun regex. The judge model
# then wrongly tags them as out_of_locale (proper-noun heuristic doesn't
# distinguish "Precision is the opposite..." from "Precision Inc."). These
# are locale-agnostic — they're not proper nouns in any locale.
_COMMON_SENTENCE_INITIAL_WORDS: frozenset[str] = frozenset(
    s.lower() for s in (
        # adverbs / interjections that frequently open sentences
        "Absolutely", "Actually", "Always", "Alright", "Anyway", "Anyhow",
        "Basically", "Certainly", "Clearly", "Currently", "Earlier",
        "Eventually", "Everywhere", "Exactly", "Finally", "Frankly",
        "Generally", "Hands-on", "Honestly", "Indeed", "Lastly",
        "Later", "Likely", "Maybe", "Naturally", "Normally",
        "Obviously", "Perhaps", "Personally", "Possibly", "Probably",
        "Quickly", "Rarely", "Really", "Recently", "Seriously",
        "Slowly", "Somewhere", "Suddenly", "Surely", "Typically",
        "Ultimately", "Unfortunately", "Usually", "Yeah",
        # common nouns that get sentence-initial caps
        "Precision", "Quality", "Service", "Experience", "Comfort",
        "Tradition", "Culture", "Style", "Flavor", "Texture",
        "Balance", "Harmony", "Rhythm", "Practice",
        # Universal tech / acronyms — not locale-specific
        "Wi-Fi", "Wifi", "WiFi", "SMS", "GPS", "URL", "API", "PDF",
        "CBD", "ATM", "PIN", "QR", "VR", "AR",
        # Meta-cultural label: "Western" appears in our own avoid-defaults
        # discussion ("avoid Western references") in tutor turns. Real
        # Western-cultural leaks like "Western breakfast" or "Western movie"
        # are caught as multi-word entities by the judge, since the
        # proper-noun regex captures multi-word phrases.
        "Western",
        # English modal / auxiliary verbs frequently open sentences
        # ("Will your grandfather come?", "May I ask..."). They overlap
        # with Western person names (Will/William, May, Mark) and the
        # judge mis-tags them as out-of-locale name leakage.
        "Will", "May", "Might", "Could", "Would", "Should", "Shall",
        # English connectors / common nouns frequently capitalized at
        # sentence start ("Plus, you can also...", "Line 7 is faster.",
        # "Coffee is popular here."). Dominant FP source in the 9B run.
        "Plus", "Line", "Coffee",
        # Universal software / tools — present in every locale's modern
        # daily life. Judge sometimes flags as Western brand.
        "Python", "Photoshop", "Google", "Google Maps", "Zoom", "CapCut",
    )
)


# Per-locale allowlist of KNOWN-IN-LOCALE entities. The judge model
# (especially 4B-class) sometimes wrongly tags genuine locale entities
# as out_of_locale — e.g. "WeChat", "Alipay", "Yunnan",
# "Mid-Autumn Festival" for China. These rejections are 100%
# false-positive: they ARE in locale. Skipping at extraction prevents
# both the cache poison and the wasted judge call.
#
# Add an entity here when you've confirmed (manually) it's genuinely in
# locale and the judge keeps mis-labeling it. NEVER add entities you
# aren't sure about — false-allowlisting a Western-default entity
# defeats the whole locale_judge.
_KNOWN_IN_LOCALE: dict[str, frozenset[str]] = {
    "china": frozenset(
        s.lower() for s in (
            # Chinese tech / apps / brands
            "WeChat", "WeChat Channels", "WeChat ID", "Alipay", "Weibo",
            "Tmall", "Taobao", "Tencent", "Huawei", "Baidu", "Douyin",
            "Xiaomi", "SF Express", "Sina", "DiDi", "Meituan", "Pinduoduo",
            # Major cities and provinces
            "Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu",
            "Hangzhou", "Xiamen", "Suzhou", "Nanjing", "Tianjin",
            "Chongqing", "Wuhan", "Changsha", "Qingdao", "Dalian",
            "Kunming", "Harbin", "Shenyang", "Lhasa", "Sanya",
            "Xi'an", "Zhengzhou", "Jinan", "Hefei", "Lanzhou",
            "Yunnan", "Sichuan", "Guangdong", "Fujian", "Zhejiang",
            "Jiangsu", "Anhui", "Hunan", "Hubei", "Shandong",
            "Henan", "Hebei", "Shanxi", "Shaanxi", "Gansu",
            "Tibet", "Xinjiang", "Inner Mongolia", "Liaoning", "Jilin",
            # Cultural / historical
            "Mid-Autumn Festival", "Spring Festival", "Lunar New Year",
            "Dragon Boat Festival", "Chongyang Festival", "Qingming",
            "Qingming Festival",
            "Tang Dynasty", "Song Dynasty", "Ming Dynasty",
            "Qing Dynasty", "Han Dynasty", "Sui Dynasty", "Yuan Dynasty",
            "Tang", "Song", "Ming", "Qing", "Han", "Sui", "Yuan",
            "Great Wall", "Forbidden City", "Summer Palace",
            "Terracotta Warriors", "Stone Buddha Cave",
            # Universities
            "Tsinghua University", "Peking University", "Fudan University",
            "Tongji University", "Southeast University", "West Lake University",
            # Geography / landmarks
            "Yangtze", "Yangtze River", "Yellow River", "Pearl River",
            "Suzhou Creek", "Canton Tower", "Taikoo Li",
            "West District", "West Gate", "West Hill", "Wanda Square",
            "Wulin Park", "Hongshan Park", "Jinli Ancient Street",
            # Added 2026-06-08 after dominant FPs in 9B run:
            "West Lake", "West Street", "Muslim Quarter", "Drum Tower",
            "Green Lake Park", "Wanda Plaza", "Old Street", "People's Park",
            "Blue Moon Valley",
            # Universities (dominant FPs in 9B run):
            "Northwest University", "Southwest University",
            # Cultural practices (commonly spelled both ways)
            "Tai Chi", "Taichi", "Qigong", "Wushu", "Mahjong",
            "Hanfu", "Cheongsam", "Qipao", "Jianzhi",
        )
    ),
    "japan": frozenset(),  # add as needed
    "italy": frozenset(),  # add as needed
}


# ---------------------------------------------------------------------------
# SQLite verdict cache
# ---------------------------------------------------------------------------


class _LocaleCache:
    """Tiny SQLite-backed (entity, locale) → verdict cache.

    Keyed on the lowercased entity text + the locale name. Verdicts are
    immutable within a locale: once we decide an entity is ``in_locale``
    for ``china``, we never re-ask the judge for that pair. The same entity
    can legitimately have a different verdict in another locale
    (e.g. "Tokyo" is in_locale for japan, out_of_locale for china).

    Migration: pre-multi-locale rows have no ``locale`` column. On open we
    add the column (defaulting to ``""`` = "legacy / unspecified") and
    treat empty-string locales as default-locale data so the existing
    cache stays useful for the default locale.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS locale_verdicts (
            entity TEXT NOT NULL,
            verdict TEXT NOT NULL,
            judged_at REAL NOT NULL,
            source_model TEXT,
            locale TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (entity, locale)
        )
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(self._SCHEMA)
        # Migration for pre-multi-locale DBs: add ``locale`` column if it
        # is missing. PRIMARY KEY stays on entity in that case, but new
        # writes with explicit locale will still INSERT OR REPLACE on the
        # entity column alone — which means a multi-locale cache MUST be
        # opened fresh OR you need to re-create the table (we just live
        # with single-locale-collisions on the old DB until users rebuild).
        cols = {row[1] for row in self._conn.execute(
            "PRAGMA table_info(locale_verdicts)"
        ).fetchall()}
        if "locale" not in cols:
            self._conn.execute(
                "ALTER TABLE locale_verdicts ADD COLUMN locale TEXT NOT NULL DEFAULT ''"
            )
        self._conn.commit()

    def get_many(
        self, entities: Iterable[str], locale: str = ""
    ) -> dict[str, str]:
        """Return verdicts for the given entities WITHIN the given locale.

        Empty ``locale`` (back-compat default) returns rows where the cached
        locale is empty too (legacy entries). With a non-empty locale we
        ALSO accept legacy empty-locale rows when ``locale`` matches the
        default — those entries were generated when only the default locale
        existed and are still valid for it.
        """
        keys = [e.lower() for e in entities]
        if not keys:
            return {}
        out: dict[str, str] = {}
        with self._lock:
            placeholders = ",".join("?" for _ in keys)
            if locale:
                from qwen_tutor.locale import DEFAULT_LOCALE_NAME

                if locale == DEFAULT_LOCALE_NAME:
                    rows = self._conn.execute(
                        f"SELECT entity, verdict FROM locale_verdicts "
                        f"WHERE entity IN ({placeholders}) "
                        f"AND (locale = ? OR locale = '')",
                        [*keys, locale],
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT entity, verdict FROM locale_verdicts "
                        f"WHERE entity IN ({placeholders}) AND locale = ?",
                        [*keys, locale],
                    ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT entity, verdict FROM locale_verdicts "
                    f"WHERE entity IN ({placeholders}) AND locale = ''",
                    keys,
                ).fetchall()
        for ent, verdict in rows:
            out[ent] = verdict
        return out

    def put_many(
        self,
        verdicts: dict[str, str],
        source_model: str | None = None,
        locale: str = "",
    ) -> None:
        if not verdicts:
            return
        ts = time.time()
        rows = [
            (k.lower(), v, ts, source_model, locale)
            for k, v in verdicts.items()
            if v in VALID_VERDICTS
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO locale_verdicts "
                "(entity, verdict, judged_at, source_model, locale) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class LocaleLLMJudge(Filter):
    name = "locale_judge"

    def __init__(
        self,
        judge: TeacherClient,
        cache_path: str | Path = DEFAULT_CACHE_PATH,
        min_score: float = DEFAULT_MIN_SCORE,
        ambiguous_weight: float = DEFAULT_AMBIGUOUS_WEIGHT,
        max_entities_per_call: int = 40,
        max_tokens: int = 800,
        temperature: float = 0.0,
        **_legacy: object,  # compatibility: ignore the legacy spacy_model kwarg
    ) -> None:
        if not 0.0 <= ambiguous_weight <= 1.0:
            raise ValueError(
                f"ambiguous_weight must be in [0, 1], got {ambiguous_weight}"
            )
        self.judge = judge
        self.cache = _LocaleCache(cache_path)
        self.min_score = min_score
        self.ambiguous_weight = ambiguous_weight
        self.max_entities_per_call = max_entities_per_call
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ----- entity extraction ---------------------------------------------

    def _extract_entities(
        self, example: FilterableExample, locale_name: str = "",
    ) -> list[str]:
        """Extract capitalized phrases (people / place / brand candidates) from
        assistant turns with a regex. Classification is left to the LLM judge -
        this step is a cheap filter.

        ``locale_name`` is consulted to skip entities that are known to be
        in-locale (``_KNOWN_IN_LOCALE``). This avoids judge false-positives
        on entities like ``WeChat`` / ``Yunnan`` / ``Mid-Autumn Festival``
        for China that the 4B judge mis-classifies as out_of_locale.

        For eval examples the assistant turn is ``<think>...</think>{json}``.
        The <think> block is private reasoning the trained student would never
        emit at deploy time, so we slice to post-</think> before scanning.
        Matches the same scoping that ``banned_terms`` and ``non_latin_script``
        already apply to eval records.
        """
        md = getattr(example, "metadata", None)
        is_eval = md is not None and hasattr(md, "source_dialogue_id")
        msgs = getattr(example, "messages", None) or []
        texts: list[str] = []
        for m in msgs:
            if m.role != "assistant":
                continue
            content = m.content
            if is_eval:
                close_match = _THINK_CLOSE_RE.search(content)
                if close_match is not None:
                    content = content[close_match.end():]
            texts.append(content)
        if not texts:
            return []
        known_in_locale = _KNOWN_IN_LOCALE.get(locale_name.lower(), frozenset())
        seen: dict[str, str] = {}
        for text in texts:
            if not text:
                continue
            for m in _PROPER_NOUN_RE.finditer(text):
                cleaned = m.group(1).strip()
                if len(cleaned) < 3:
                    continue
                first_word = cleaned.split()[0].lower()
                if first_word in _PROPER_STOPWORDS:
                    continue
                # Strip a leading article like "the Tajrish".
                cleaned = re.sub(
                    r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE
                ).strip()
                if not cleaned:
                    continue
                cleaned_lc = cleaned.lower()
                # Bare language name: meta-discussion, not a Western-default
                # leak. Skip exact matches only — "English breakfast" still
                # gets judged.
                if cleaned_lc in _LANGUAGE_NAMES:
                    continue
                # Compound language-policy phrase ("English-only",
                # "Mandarin-first", "Korean-speaking"). Same class as bare
                # language names — meta-discussion, not a cultural entity.
                if _LANGUAGE_POLICY_PHRASE_RE.match(cleaned):
                    continue
                # Common English word at sentence start ("Precision is...",
                # "Absolutely!"): not a proper noun.
                if cleaned_lc in _COMMON_SENTENCE_INITIAL_WORDS:
                    continue
                # Known in-locale entity for this scenario's locale.
                if cleaned_lc in known_in_locale:
                    continue
                seen.setdefault(cleaned_lc, cleaned)
        return list(seen.values())

    # ----- judge call ----------------------------------------------------

    async def _judge_entities(
        self, entities: list[str], locale_name: str | None = None
    ) -> dict[str, str]:
        """Send uncached entities to the judge model in chunks.

        ``locale_name`` controls which locale's judge prompt is sent so the
        model classifies entities against that country's expectations.
        """
        verdicts: dict[str, str] = {}
        # System carries instructions only; the per-chunk entity list goes in
        # the USER message. cacheable_prefix=system_prompt lets llama.cpp
        # cache the (large) instruction block once and reuse it across chunks.
        system_prompt = with_thinking_directive(
            _render_judge_header(locale_name), role="judge"
        )
        for start in range(0, len(entities), self.max_entities_per_call):
            chunk = entities[start : start + self.max_entities_per_call]
            user_message = "Entities:\n" + "\n".join(f"  - {e}" for e in chunk)
            try:
                raw = await self.judge.generate(
                    system=system_prompt,
                    messages=[Message(role="user", content=user_message)],
                    cacheable_prefix=system_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                parsed = extract_first_json(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("locale_judge: judge call failed: %s", exc)
                # On judge failure, mark this chunk's entities as "ambiguous"
                # rather than failing the example outright — keeps the
                # filter usable when the judge model is briefly unavailable.
                for ent in chunk:
                    verdicts[ent] = "ambiguous"
                continue

            if not isinstance(parsed, dict) or "results" not in parsed:
                logger.warning("locale_judge: judge returned malformed JSON")
                for ent in chunk:
                    verdicts[ent] = "ambiguous"
                continue

            for row in parsed["results"]:
                if not isinstance(row, dict):
                    continue
                ent = row.get("entity")
                verdict = row.get("verdict")
                if not isinstance(ent, str) or not isinstance(verdict, str):
                    continue
                verdict = verdict.strip().lower()
                if verdict not in VALID_VERDICTS:
                    continue
                verdicts[ent.strip()] = verdict
        return verdicts

    # ----- main check ----------------------------------------------------

    async def check(self, example: FilterableExample) -> FilterResult:
        # Read the example's locale so cache lookup + judge prompt + verdict
        # storage are all scoped to that locale. Existing single-locale data
        # has metadata.locale defaulted to "china" via the schema, so the
        # behavior on legacy data is unchanged.
        locale_name = getattr(getattr(example, "metadata", None), "locale", "") or ""
        entities = self._extract_entities(example, locale_name=locale_name)
        if not entities:
            return FilterResult(
                passed=True,
                score=1.0,
                reason="no proper-noun entities in assistant turns",
                metadata={"n_entities": 0},
            )

        cached = self.cache.get_many(entities, locale=locale_name)
        missing = [e for e in entities if e.lower() not in cached]
        new_verdicts: dict[str, str] = {}
        if missing:
            new_verdicts = await self._judge_entities(missing, locale_name=locale_name)
            self.cache.put_many(
                new_verdicts,
                source_model=self.judge.config.model,
                locale=locale_name,
            )

        verdicts: dict[str, str] = {}
        for ent in entities:
            lk = ent.lower()
            if lk in cached:
                verdicts[ent] = cached[lk]
            elif ent in new_verdicts:
                verdicts[ent] = new_verdicts[ent]
            else:
                # judge didn't return this one — treat as ambiguous
                verdicts[ent] = "ambiguous"

        counts = Counter(verdicts.values())
        n_total = len(verdicts)
        n_in_locale = counts.get("in_locale", 0)
        n_ambiguous = counts.get("ambiguous", 0)
        n_out_of_locale = counts.get("out_of_locale", 0)
        score = (
            (n_in_locale + self.ambiguous_weight * n_ambiguous) / n_total
            if n_total
            else 1.0
        )

        out_of_locale_entities = [e for e, v in verdicts.items() if v == "out_of_locale"]

        passed = (score >= self.min_score) and (n_out_of_locale == 0)
        reason = None
        if not passed:
            if n_out_of_locale > 0:
                reason = (
                    f"out-of-locale entity detected ({n_out_of_locale}): "
                    + ", ".join(repr(e) for e in out_of_locale_entities[:5])
                )
            else:
                reason = (
                    f"locale_score {score:.2f} below threshold {self.min_score:.2f} "
                    f"(in_locale={n_in_locale}, ambiguous={n_ambiguous}, total={n_total})"
                )

        return FilterResult(
            passed=passed,
            score=score,
            reason=reason,
            metadata={
                "n_entities": n_total,
                "n_in_locale": n_in_locale,
                "n_ambiguous": n_ambiguous,
                "n_out_of_locale": n_out_of_locale,
                "out_of_locale_entities": out_of_locale_entities,
                "verdicts": verdicts,
                "newly_judged": list(new_verdicts.keys()),
            },
        )
