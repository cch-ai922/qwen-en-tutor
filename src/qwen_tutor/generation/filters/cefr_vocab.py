"""CEFR vocabulary-band filter.

Loads a JSON file mapping each band (A1..C1) to a list of words, plus a
stopword list. For an SFTExample at target level X, computes the
fraction of *content words* in the assistant turns whose CEFR band is
strictly above X, ignoring stopwords and words not in any band
(charitable default for partial vocab files).

Fails if that fraction exceeds 5%.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from qwen_tutor.generation.filters.base import Filter, FilterableExample, FilterResult

BAND_ORDER = ("A1", "A2", "B1", "B2", "C1", "C2")
BAND_RANK = {b: i for i, b in enumerate(BAND_ORDER)}

DEFAULT_VOCAB_PATH = Path("config/cefr_vocab_bands.json")
DEFAULT_THRESHOLD = 0.05

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


class CEFRVocabFilter(Filter):
    name = "cefr_vocab"

    def __init__(
        self,
        vocab_path: str | Path = DEFAULT_VOCAB_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        with Path(vocab_path).open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        self.word_to_band: dict[str, str] = {}
        for band, words in doc.get("bands", {}).items():
            if band not in BAND_RANK:
                continue
            for w in words:
                lw = w.lower()
                # earliest-seen-band wins (in case of accidental duplicates)
                self.word_to_band.setdefault(lw, band)
        self.stopwords: set[str] = {s.lower() for s in doc.get("stopwords", [])}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]

    def _assistant_text(self, example: FilterableExample) -> str:
        # CEFRVocabFilter only applies to dialogues; for DPO/Eval there is
        # no single target level for the assistant turns, so return "".
        if hasattr(example, "messages") and hasattr(example, "metadata"):
            md = example.metadata
            if hasattr(md, "cefr_level"):
                turns = [
                    m.content for m in example.messages if m.role == "assistant"
                ]
                return "\n".join(turns)
        return ""

    async def check(self, example: FilterableExample) -> FilterResult:
        # only meaningful for SFTExample
        md = getattr(example, "metadata", None)
        cefr_level = getattr(md, "cefr_level", None) if md is not None else None
        if cefr_level is None or cefr_level not in BAND_RANK:
            return FilterResult(passed=True, reason="not a CEFR-bound example")
        if cefr_level == "C2":
            return FilterResult(
                passed=True, score=0.0, reason="C2 is open; nothing is above band"
            )
        target_rank = BAND_RANK[cefr_level]
        text = self._assistant_text(example)
        if not text:
            return FilterResult(passed=True, reason="no assistant text")

        tokens = self._tokenize(text)
        content_words = [t for t in tokens if t not in self.stopwords]
        known = [w for w in content_words if w in self.word_to_band]
        if not known:
            return FilterResult(
                passed=True,
                score=0.0,
                reason="no in-vocab content words",
                metadata={"total_tokens": len(tokens), "content_words": len(content_words)},
            )
        above: list[str] = [
            w for w in known if BAND_RANK[self.word_to_band[w]] > target_rank
        ]
        ratio = len(above) / len(known)
        passed = ratio <= self.threshold

        # collect top offending words for the diagnostic
        offending: dict[str, int] = {}
        for w in above:
            offending[w] = offending.get(w, 0) + 1
        top_offenders = sorted(offending.items(), key=lambda kv: -kv[1])[:10]

        return FilterResult(
            passed=passed,
            score=ratio,
            reason=(
                None
                if passed
                else (
                    f"{ratio * 100:.1f}% of content words exceed band "
                    f"{cefr_level} (threshold {self.threshold * 100:.0f}%)"
                )
            ),
            metadata={
                "target_level": cefr_level,
                "above_band_ratio": ratio,
                "known_content_words": len(known),
                "above_band_count": len(above),
                "top_offenders": top_offenders,
            },
        )
