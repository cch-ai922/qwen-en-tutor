"""Locale LLM-judge filter.

대상 국가는 ``config/locale.yaml`` 의 country / country_adjective 에서 옵니다.
정규식으로 assistant turn 에서 대문자 고유명사 후보를 뽑고 cheap judge 모델
에게 각 entity 가 ``in_locale`` / ``ambiguous`` / ``out_of_locale`` 중 어느
쪽인지 묻습니다. 결과는 SQLite 캐시에 저장되어 같은 entity 는 다시 묻지
않습니다.

score = ``(n_in_locale + AMBIGUOUS_WEIGHT * n_ambiguous) / n_total``

실패 조건:
  * score < min_score (기본 0.5), OR
  * 단 하나라도 ``out_of_locale`` 로 판정된 entity 가 있는 경우.

기본값(min_score=0.5, AMBIGUOUS_WEIGHT=0.7)은 A2 같은 저레벨 대화에서
specific city/neighborhood 가 적고 대부분 ambiguous 로 잡히는 현실을
반영합니다. ``out_of_locale`` 검출은 weight 와 무관하게 항상 즉시 fail 처리
되므로, threshold 를 낮춰도 진짜 off-locale 대화는 통과하지 않습니다.

기계적 필터를 모두 통과한 예시에 대해서만 호출되도록 파이프라인 뒤쪽에
배치합니다.
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
    "{learner_description}. For each entity in the list below,\n"
    "output one verdict from this set:\n\n"
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


# 외부 NER(spaCy) 의존을 없애고 정규식 기반으로 고유명사를 추출합니다.
# 추출된 후보는 그대로 LLM judge 에게 보내져 in_locale / ambiguous /
# out_of_locale 로 분류되므로, false positive 도 judge 가 ambiguous 로
# 처리해 줍니다 (그리고 SQLite cache 로 한 번만 묻습니다).
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]{1,}(?:[\s\-][A-Z][a-zA-Z'\-]+){0,3})\b"
)

# sentence-initial common-word false positive 제거용.
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
        **_legacy: object,  # 호환성: 옛 spacy_model kwarg 무시
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

    def _extract_entities(self, example: FilterableExample) -> list[str]:
        """assistant turn 에서 대문자 phrase (사람/지명/브랜드 후보) 를 정규식
        으로 뽑습니다. 분류는 LLM judge 가 합니다 - 이 단계는 cheap filter.
        """
        msgs = getattr(example, "messages", None) or []
        texts = [m.content for m in msgs if m.role == "assistant"]
        if not texts:
            return []
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
                # "the Tajrish" 같은 leading 관사 제거.
                cleaned = re.sub(
                    r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE
                ).strip()
                if cleaned:
                    seen.setdefault(cleaned.lower(), cleaned)
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
        header = _render_judge_header(locale_name)
        for start in range(0, len(entities), self.max_entities_per_call):
            chunk = entities[start : start + self.max_entities_per_call]
            payload = "\n".join(f"  - {e}" for e in chunk)
            prompt = header + "\nEntities:\n" + payload + "\n"
            try:
                raw = await self.judge.generate(
                    system=prompt,
                    messages=[],
                    cacheable_prefix=header,
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
        entities = self._extract_entities(example)
        if not entities:
            return FilterResult(
                passed=True,
                score=1.0,
                reason="no proper-noun entities in assistant turns",
                metadata={"n_entities": 0},
            )

        # Read the example's locale so cache lookup + judge prompt + verdict
        # storage are all scoped to that locale. Existing single-locale data
        # has metadata.locale defaulted to "china" via the schema, so the
        # behavior on legacy data is unchanged.
        locale_name = getattr(getattr(example, "metadata", None), "locale", "") or ""

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
