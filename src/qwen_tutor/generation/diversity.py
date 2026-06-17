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
# Food names are not captured by spaCy as entities (no FOOD label), so they are
# counted by matching lowercase terms directly. Because this vocab is
# country-specific, it is not stored as a static list in the module; instead it
# is loaded from the ``food_terms`` field in ``config/locale.yaml``. If left
# empty, food counts will simply remain 0, and the PERSON / GPE sections of the
# diversity report (spaCy NER-based) will still work normally.
#
# If ``food_terms`` is populated in locale.yaml, that locale's representative
# foods are included in the diversity report's top_foods section. If
# ``avoid_food_terms`` is also provided, teacher-generated food terms leaking
# into the wrong default cuisine (e.g. Western items) will also be counted, so
# the tracker can surface a bad signal earlier.

from qwen_tutor.locale import LOCALES as _LOCALES_DV

# Compatibility: expose empty tuples so old imports of
# IRANIAN_FOOD_TERMS / WESTERN_FOOD_TERMS / DEFAULT_FOOD_TERMS do not break
# (locale.yaml is the source of truth).
IRANIAN_FOOD_TERMS: tuple[str, ...] = ()
WESTERN_FOOD_TERMS: tuple[str, ...] = ()

# Multi-locale: include the union of all locales' food_terms and avoid_food_terms
# as diversity counting targets. If a locale's representative food appears in a
# different locale's conversation, it will still be captured in the diversity report.
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
    total_category_examples: int = 0
    top_names: list[FreqEntry] = field(default_factory=list)
    top_cities: list[FreqEntry] = field(default_factory=list)
    top_foods: list[FreqEntry] = field(default_factory=list)
    top_categories: list[FreqEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_documents": self.total_documents,
            "total_person_mentions": self.total_person_mentions,
            "total_gpe_mentions": self.total_gpe_mentions,
            "total_food_mentions": self.total_food_mentions,
            "total_category_examples": self.total_category_examples,
            "top_names": [e.to_dict() for e in self.top_names],
            "top_cities": [e.to_dict() for e in self.top_cities],
            "top_foods": [e.to_dict() for e in self.top_foods],
            "top_categories": [e.to_dict() for e in self.top_categories],
            "warnings": list(self.warnings),
        }

    def format_text(self) -> str:
        lines = [
            f"Documents scanned: {self.total_documents}",
            f"  PERSON entities: {self.total_person_mentions}",
            f"  GPE entities:    {self.total_gpe_mentions}",
            f"  food mentions:   {self.total_food_mentions}",
            f"  category-tagged: {self.total_category_examples}",
        ]
        for label, entries in (
            ("Top names (PERSON)", self.top_names),
            ("Top cities (GPE)", self.top_cities),
            ("Top foods", self.top_foods),
            ("Categories", self.top_categories),
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
# Avoid external NER (spaCy) dependency by extracting proper nouns with regex.
# This also addresses spaCy failures on romanized Singapore/Chinese/Japanese names
# that fall outside the typical NER training domain.
#
# Drawback: it cannot perfectly distinguish PERSON from GPE. We use a heuristic
# that classifies single-word candidates as PERSON and multi-word candidates as
# GPE. It is not perfect, but it is sufficient for the single purpose of the
# diversity report: identifying which names/places appear too frequently.

# One to four consecutive capitalized words (people/place name pattern).
# Multi-word capitalized phrase: "Naqsh-e Jahan Square", "Times Square",
# "Lin Mei", "Tehran", ...
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]{1,}(?:[\s\-][A-Z][a-zA-Z'\-]+){0,3})\b"
)

# Common sentence-initial words that often start with a capital letter,
# used to filter false positives.
_PROPER_STOPWORDS: frozenset[str] = frozenset(
    s.lower()
    for s in (
        # days / months
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
        "Sunday", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        # times / seasons
        "Today", "Tomorrow", "Yesterday", "Tonight", "Morning", "Afternoon",
        "Evening", "Night", "Spring", "Summer", "Autumn", "Fall", "Winter",
        # articles / pronouns / frequently used sentence-initial words
        "The", "A", "An", "I", "You", "He", "She", "We", "They", "It",
        "This", "That", "These", "Those", "My", "Your", "His", "Her", "Our",
        "Their", "Some", "Any", "No", "Yes", "Oh", "Well", "OK", "Okay",
        "Sure", "Hi", "Hello", "Hey", "Bye", "Thanks", "Thank",
        # commonly occurring function words
        "When", "Where", "Why", "How", "What", "Who", "Which", "If", "Or",
        "And", "But", "So", "Then", "There", "Here",
    )
)


def _is_likely_proper_noun(candidate: str) -> bool:
    """Filter out false positives like common sentence-initial words."""
    if len(candidate) < 3:
        return False
    first_word = candidate.split()[0].lower()
    return first_word not in _PROPER_STOPWORDS


def _classify_proper_noun(candidate: str) -> str:
    """Simple heuristic: estimate PERSON/GPE by word count.

    1 word - PERSON (name)
    2+ words - GPE (city/place name; multi-word Singapore/Chinese names like
              "Lin Mei" may be misclassified as GPE, but this is sufficient
              for diversity counting purposes).
    """
    return "PERSON" if len(candidate.split()) == 1 else "GPE"


class DiversityTracker:
    """Accumulates name/city/food mention counts across a scan stream."""

    def __init__(
        self,
        food_terms: Iterable[str] | None = None,
        **_legacy: object,  # Compatibility: ignore old spacy_model kwarg
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
        # Per-life-domain example count, populated via ``track_category``.
        # Independent of the proper-noun / food scans because categories
        # come from metadata, not from the assistant text.
        self.categories: Counter[str] = Counter()

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

    def track_category(self, category: str | None) -> None:
        """Bump the per-life-domain example count.

        Callers pass ``example.metadata.category`` for each scanned
        example. Empty/None values are silently skipped so legacy data
        without the field doesn't pollute the report.
        """
        if not category:
            return
        self.categories[category] += 1

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
            total_category_examples=sum(self.categories.values()),
            top_names=self._top(self.persons, top_names),
            top_cities=self._top(self.gpes, top_cities),
            top_foods=self._top(self.foods, top_foods),
            # Show all categories — typically only ~10 so no truncation.
            top_categories=self._top(self.categories, len(self.categories) or 1),
        )
        for label, entries in (
            ("name", rpt.top_names),
            ("city", rpt.top_cities),
            ("food", rpt.top_foods),
            ("category", rpt.top_categories),
        ):
            if entries and entries[0].share > warn_threshold:
                rpt.warnings.append(
                    f"top {label} {entries[0].item!r} accounts for "
                    f"{entries[0].share * 100:.1f}% of mentions "
                    f"(threshold {warn_threshold * 100:.0f}%)"
                )
        return rpt
