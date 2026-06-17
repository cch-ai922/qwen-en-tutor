"""speaks_l1 sanity filter.

The ``speaks_l1`` variant of ``language_redirect`` requires one user turn
to be written in the learner's L1 (a non-Latin script). When the teacher
refuses to emit L1 and answers only in English, the resulting example is
degenerate and useless as training data.

This filter only inspects SFTExamples whose
``metadata.generation.language_trigger == "speaks_l1"`` and rejects them
in either of these cases:

  1. Every user turn is English (i.e. zero non-Latin characters anywhere)
     -> the teacher never produced the L1 turn. Degenerate.
  2. The assistant turn contains a "you could say in English: '<text>'"
     pattern that simply echoes the user's previous English turn
     -> because the teacher skipped the L1 turn, the tutor ends up
       "translating" English into English, which is nonsensical. Only
       reject when the echo similarity is >= 0.8, so genuine paraphrases
       still pass.

All other records (not speaks_l1) are passed through unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from qwen_tutor.generation.filters.base import (
    Filter,
    FilterableExample,
    FilterResult,
)
from qwen_tutor.schemas import SFTExample

# Non-Latin script detection (same set as NonLatinScriptFilter).
_NON_LATIN_RE = re.compile(
    "["
    "぀-ゟ゠-ヿㇰ-ㇿ　-〿㐀-䶿一-鿿豈-﫿＀-￯"  # CJK
    "가-힯ᄀ-ᇿ㄰-㆏"                              # Hangul
    "Ѐ-ӿԀ-ԯ"                                    # Cyrillic
    "؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿"                       # Arabic
    "֐-׿"                                        # Hebrew
    "ऀ-ॿঀ-৿਀-੿"                                # Devanagari/Bengali/Gurmukhi
    "฀-๿"                                        # Thai
    "]"
)

# Echo-pattern signatures: tutor sentences that introduce a quoted English
# version of "what the learner said". These on their own are NOT a problem
# (real recasts include them); they are a SIGNAL that the tutor is about
# to paraphrase. The degeneracy is when the paraphrase is nearly identical
# to the user's previous English turn.
_ECHO_INTRO_RE = re.compile(
    r"(?i)("
    r"you could say in english"
    r"|in english (we'?d|you'?d|you would) say"
    r"|let me say it in english"
    r"|in english(?: that)?(?: would be)?[:,]"
    r")"
)

# Extract the first quoted span after the echo intro. The tutor's
# paraphrase typically lives inside single, double, or smart quotes.
_QUOTED_RE = re.compile(r"['\"‘’“”]([^'\"‘’“”]{4,200})['\"‘’“”]")

# Echo-similarity threshold above which we consider the tutor's quote to
# be a verbatim echo of the user's prior English turn. SequenceMatcher
# ratio in [0, 1]; 0.85+ catches the degenerate "translate your own
# English back into English" pattern while leaving room for legitimate
# minor edits in real paraphrases.
_ECHO_SIMILARITY_THRESHOLD = 0.85

# Minimum length the user's prior turn must have before we trust the
# echo-similarity check. Very short turns ("yes", "thanks") trivially
# match each other.
_MIN_USER_TURN_CHARS = 12


def _normalize(s: str) -> str:
    """Lowercase + NFKC + strip punctuation/whitespace for similarity comparison."""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def _has_non_latin(text: str) -> bool:
    return bool(_NON_LATIN_RE.search(text or ""))


def _detect_degenerate_echo(messages: list) -> str | None:
    """Walk the dialogue looking for the "tutor echoes the user's prior
    English back" pattern. Returns a reason string if detected, else None.
    """
    for i, msg in enumerate(messages):
        if msg.role != "assistant" or i == 0:
            continue
        if not _ECHO_INTRO_RE.search(msg.content):
            continue
        # Find what the tutor quoted.
        m = _QUOTED_RE.search(msg.content)
        if not m:
            continue
        quoted = m.group(1).strip()
        if len(quoted) < _MIN_USER_TURN_CHARS:
            continue
        # The user's prior turn is the immediately preceding user message.
        prior_user_content = None
        for j in range(i - 1, -1, -1):
            if messages[j].role == "user":
                prior_user_content = messages[j].content
                break
        if prior_user_content is None:
            continue
        if len(prior_user_content) < _MIN_USER_TURN_CHARS:
            continue
        # If the prior user turn was L1 (has non-Latin), the tutor IS
        # supposed to paraphrase it in English — that's the correct flow,
        # not an echo. The degenerate case is paraphrase of already-English.
        if _has_non_latin(prior_user_content):
            continue
        sim = SequenceMatcher(
            None, _normalize(prior_user_content), _normalize(quoted)
        ).ratio()
        if sim >= _ECHO_SIMILARITY_THRESHOLD:
            return (
                f"assistant[{i}] echoed user[{i - 1}]'s English back as a "
                f"fake paraphrase (similarity {sim:.2f} >= {_ECHO_SIMILARITY_THRESHOLD:.2f})"
            )
    return None


class SpeaksL1SanityFilter(Filter):
    """Catch degenerate ``speaks_l1`` examples that slipped through generation.

    Two failure modes are flagged:
      1. No user turn contains non-Latin script (teacher refused to write L1).
      2. Tutor echoes the user's prior English back as if "translating" it
         (the "you could say in English: '<verbatim>'" pattern over already-
         English input).

    Records that are NOT ``speaks_l1`` pass through unchanged.
    """

    name = "speaks_l1_sanity"

    async def check(self, example: FilterableExample) -> FilterResult:
        # Only SFT examples have the language_trigger metadata.
        if not isinstance(example, SFTExample):
            return FilterResult(passed=True, reason="not an SFTExample")

        gen = (example.metadata.generation or {}) if example.metadata else {}
        trigger = gen.get("language_trigger")
        if trigger != "speaks_l1":
            return FilterResult(
                passed=True, reason="not a speaks_l1 record"
            )

        # Failure mode 1: no user turn has any non-Latin character.
        l1_user_turns = [
            i for i, m in enumerate(example.messages)
            if m.role == "user" and _has_non_latin(m.content)
        ]
        if not l1_user_turns:
            return FilterResult(
                passed=False,
                score=0.0,
                reason=(
                    "speaks_l1 example has NO user turn in L1 — teacher "
                    "refused to switch language"
                ),
                metadata={"failure_mode": "no_l1_turn"},
            )

        # Failure mode 2: tutor echoes user English back as fake paraphrase.
        echo_reason = _detect_degenerate_echo(list(example.messages))
        if echo_reason:
            return FilterResult(
                passed=False,
                score=0.0,
                reason=f"degenerate echo: {echo_reason}",
                metadata={"failure_mode": "english_echo"},
            )

        return FilterResult(
            passed=True,
            score=1.0,
            metadata={"l1_turn_count": len(l1_user_turns)},
        )


__all__ = ["SpeaksL1SanityFilter"]
