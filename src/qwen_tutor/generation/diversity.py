"""Cross-cutting locale-diversity tracker for generated dialogues.

After any generation batch finishes, the pipeline scans the assistant
turns and tracks the running distribution of:

  * proper-noun PERSON entities (target-locale first names we hope to see),
  * proper-noun GPE entities (target-locale cities and neighborhoods),
  * food terms (matched against a curated food vocabulary).

Proper nouns (single-word → PERSON, multi-word → GPE) come from a
regex over capitalized phrases. Foods come from matching a curated
lowercase vocabulary against the assistant text. No external NER
dependency.

The tracker emits a per-stage diversity report with the top items in each
category and warns (does NOT fail) when the top item in any category
exceeds 25% of mentions — a soft signal that the teacher model has
overfitted to a single name, city, or dish.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Food vocabulary
# ---------------------------------------------------------------------------
#
# 음식 이름은 spaCy 가 entity 로 잡지 못해 (FOOD 라벨 없음) 직접 lowercase
# 매칭으로 카운트합니다. 이 vocab 은 country-specific 이므로 정적 리스트를
# 모듈에 두지 않고 ``config/locale.yaml`` 의 ``food_terms`` 필드에서
# 가져옵니다. 비워 두면 음식 카운트는 단순히 0 으로 잡히며, 다양성 리포트
# 의 PERSON / GPE 부분 (spaCy NER 기반) 은 영향 없이 정상 동작합니다.
#
# locale.yaml 에 ``food_terms`` 를 채워 두면 그 country 의 대표 음식들이
# diversity 리포트의 top_foods 섹션에 집계됩니다. 추가로 ``avoid_food_terms``
# 를 채우면 teacher 가 Western 등 잘못된 default 로 새는 음식 단어도 함께
# 카운트되어 "안 좋은 신호" 를 일찍 잡을 수 있습니다.

from qwen_tutor.locale import LOCALES as _LOCALES_DV

# 호환성: 기존 import 가 IRANIAN_FOOD_TERMS / WESTERN_FOOD_TERMS / DEFAULT_FOOD_TERMS
# 를 참조해도 깨지지 않도록 비어 있는 tuple 로 노출 (locale.yaml 가 source).
IRANIAN_FOOD_TERMS: tuple[str, ...] = ()
WESTERN_FOOD_TERMS: tuple[str, ...] = ()

# 다중 locale: 모든 locale 의 food_terms + avoid_food_terms 의 합집합을
# diversity 카운트 대상으로 둡니다. 한 locale 의 대표 음식이 다른 locale
# 의 conversation 에 흘러들면 그것도 다양성 리포트에 잡힙니다.
_seen_terms: set[str] = set()
_combined: list[str] = []
for _loc in _LOCALES_DV.values():
    for _t in (*_loc.food_terms, *_loc.avoid_food_terms):
        if _t and _t.lower() not in _seen_terms:
            _seen_terms.add(_t.lower())
            _combined.append(_t)
DEFAULT_FOOD_TERMS: tuple[str, ...] = tuple(_combined)
del _seen_terms, _combined

OVERREP_TOP_SHARE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FreqEntry:
    item: str
    count: int
    share: float

    def to_dict(self) -> dict:
        return {"item": self.item, "count": self.count, "share": round(self.share, 4)}


@dataclass(slots=True)
class DiversityReport:
    total_documents: int = 0
    total_person_mentions: int = 0
    total_gpe_mentions: int = 0
    total_food_mentions: int = 0
    top_names: list[FreqEntry] = field(default_factory=list)
    top_cities: list[FreqEntry] = field(default_factory=list)
    top_foods: list[FreqEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_documents": self.total_documents,
            "total_person_mentions": self.total_person_mentions,
            "total_gpe_mentions": self.total_gpe_mentions,
            "total_food_mentions": self.total_food_mentions,
            "top_names": [e.to_dict() for e in self.top_names],
            "top_cities": [e.to_dict() for e in self.top_cities],
            "top_foods": [e.to_dict() for e in self.top_foods],
            "warnings": list(self.warnings),
        }

    def format_text(self) -> str:
        lines = [
            f"Documents scanned: {self.total_documents}",
            f"  PERSON entities: {self.total_person_mentions}",
            f"  GPE entities:    {self.total_gpe_mentions}",
            f"  food mentions:   {self.total_food_mentions}",
        ]
        for label, entries in (
            ("Top names (PERSON)", self.top_names),
            ("Top cities (GPE)", self.top_cities),
            ("Top foods", self.top_foods),
        ):
            lines.append("")
            lines.append(label + ":")
            if not entries:
                lines.append("  (none)")
                continue
            for e in entries:
                lines.append(f"  {e.share * 100:5.1f}%  {e.count:>4d}  {e.item}")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
#
# 외부 NER(spaCy) 의존을 없애고 정규식 기반으로 고유명사를 추출합니다.
# romanized 한국어/중국어/일본어 이름도 NER 학습 도메인 밖이라 spaCy 가
# 잘 못 잡았던 점을 해결합니다.
#
# 단점: PERSON 과 GPE 를 정확히 구분할 수 없습니다. 휴리스틱으로 single-word
# 후보 (단어 1개) 는 PERSON, multi-word 후보는 GPE 로 분류합니다. 완벽하진
# 않지만 다양성 리포트의 "어떤 이름/지명이 너무 자주 등장하는가" 라는
# 단일 목적에 충분합니다.

# 대문자로 시작하는 단어 1-4개의 연속 (people/place 이름 패턴).
# Multi-word capitalized phrase: "Naqsh-e Jahan Square", "Times Square",
# "Lin Mei", "Tehran", ...
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]{1,}(?:[\s\-][A-Z][a-zA-Z'\-]+){0,3})\b"
)

# Sentence-initial 흔히 대문자로 시작하는 일반어, false positive 제거용.
_PROPER_STOPWORDS: frozenset[str] = frozenset(
    s.lower()
    for s in (
        # 요일 / 월
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
        "Sunday", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        # 시간 / 계절
        "Today", "Tomorrow", "Yesterday", "Tonight", "Morning", "Afternoon",
        "Evening", "Night", "Spring", "Summer", "Autumn", "Fall", "Winter",
        # 관사 / 대명사 / 자주 쓰는 sentence-initial
        "The", "A", "An", "I", "You", "He", "She", "We", "They", "It",
        "This", "That", "These", "Those", "My", "Your", "His", "Her", "Our",
        "Their", "Some", "Any", "No", "Yes", "Oh", "Well", "OK", "Okay",
        "Sure", "Hi", "Hello", "Hey", "Bye", "Thanks", "Thank",
        # 자주 등장하는 함수어
        "When", "Where", "Why", "How", "What", "Who", "Which", "If", "Or",
        "And", "But", "So", "Then", "There", "Here",
    )
)


def _is_likely_proper_noun(candidate: str) -> bool:
    """sentence-initial common word 같은 false positive 를 거릅니다."""
    if len(candidate) < 3:
        return False
    first_word = candidate.split()[0].lower()
    return first_word not in _PROPER_STOPWORDS


def _classify_proper_noun(candidate: str) -> str:
    """간단한 휴리스틱: 단어 수로 PERSON/GPE 를 추정.

    1 단어 - PERSON (이름)
    2+ 단어 - GPE (도시/구역명, "Lin Mei" 같은 multi-word 한국/중국 이름이
              GPE 로 잘못 잡힐 수 있지만 diversity 카운트 목적엔 충분).
    """
    return "PERSON" if len(candidate.split()) == 1 else "GPE"


class DiversityTracker:
    """Accumulates name/city/food mention counts across a scan stream."""

    def __init__(
        self,
        food_terms: Iterable[str] | None = None,
        **_legacy: object,  # 호환성: 옛 spacy_model kwarg 무시
    ) -> None:
        self.food_terms = tuple(food_terms) if food_terms else DEFAULT_FOOD_TERMS
        # Sort food terms by length descending so multi-word matches win
        # over their substrings (e.g., "ash reshteh" before "ash").
        self._food_terms_sorted = tuple(
            sorted({t.lower() for t in self.food_terms}, key=len, reverse=True)
        )
        self.documents_scanned = 0
        self.persons: Counter[str] = Counter()
        self.gpes: Counter[str] = Counter()
        self.foods: Counter[str] = Counter()

    # ----- scanning -------------------------------------------------------

    def scan_text(self, text: str) -> None:
        if not text:
            return
        self.documents_scanned += 1
        for m in _PROPER_NOUN_RE.finditer(text):
            candidate = m.group(1).strip()
            if not _is_likely_proper_noun(candidate):
                continue
            kind = _classify_proper_noun(candidate)
            if kind == "PERSON":
                self.persons[candidate] += 1
            else:
                self.gpes[candidate] += 1
        lower = text.lower()
        # Track which spans we've already consumed so longer multi-word
        # matches don't double-count their substrings.
        consumed: list[tuple[int, int]] = []
        for term in self._food_terms_sorted:
            start = 0
            while True:
                idx = lower.find(term, start)
                if idx == -1:
                    break
                end = idx + len(term)
                # boundary check — avoid matching inside another word
                before_ok = idx == 0 or not lower[idx - 1].isalnum()
                after_ok = end == len(lower) or not lower[end].isalnum()
                if before_ok and after_ok and not any(
                    cs <= idx and end <= ce for cs, ce in consumed
                ):
                    self.foods[term] += 1
                    consumed.append((idx, end))
                start = end

    def scan_messages(self, messages: Iterable[dict]) -> None:
        """Scan an iterable of ``{"role": ..., "content": ...}`` messages.

        Only ``assistant`` turns are scanned — the assistant is the side
        we're auditing for locale grounding. ``user`` turns reflect the
        learner's input and would skew the distribution.
        """
        for msg in messages:
            if (msg.get("role") or "").lower() != "assistant":
                continue
            content = msg.get("content") or ""
            self.scan_text(content)

    # ----- reporting ------------------------------------------------------

    def _top(self, counter: Counter[str], n: int) -> list[FreqEntry]:
        total = sum(counter.values())
        if total == 0:
            return []
        return [
            FreqEntry(item=item, count=cnt, share=cnt / total)
            for item, cnt in counter.most_common(n)
        ]

    def report(
        self,
        top_names: int = 30,
        top_cities: int = 20,
        top_foods: int = 20,
        warn_threshold: float = OVERREP_TOP_SHARE_THRESHOLD,
    ) -> DiversityReport:
        rpt = DiversityReport(
            total_documents=self.documents_scanned,
            total_person_mentions=sum(self.persons.values()),
            total_gpe_mentions=sum(self.gpes.values()),
            total_food_mentions=sum(self.foods.values()),
            top_names=self._top(self.persons, top_names),
            top_cities=self._top(self.gpes, top_cities),
            top_foods=self._top(self.foods, top_foods),
        )
        for label, entries in (
            ("name", rpt.top_names),
            ("city", rpt.top_cities),
            ("food", rpt.top_foods),
        ):
            if entries and entries[0].share > warn_threshold:
                rpt.warnings.append(
                    f"top {label} {entries[0].item!r} accounts for "
                    f"{entries[0].share * 100:.1f}% of mentions "
                    f"(threshold {warn_threshold * 100:.0f}%)"
                )
        return rpt
