"""Mechanical-naturalness filter.

Scores the assistant turns of an SFTExample against four register
metrics, compares each against a target range for the example's CEFR
level, and returns a soft 0-1 score (mean of per-metric scores). Passes
if the score exceeds 0.6.

Metrics:
  * contraction_rate — contractions / (contractions + uncontracted forms)
  * discourse_markers_per_100 — discourse-marker tokens per 100 tokens
  * opener_variety — unique sentence-opening words / number of sentences
  * mean_sentence_length — tokens per sentence

Each metric is mapped to [0, 1] via a piecewise-linear band: 1.0 inside
the target range, falling off linearly outside it down to 0 at the
"hard" edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from qwen_tutor.generation.filters.base import Filter, FilterableExample, FilterResult
from qwen_tutor.schemas import SFTExample

DEFAULT_THRESHOLD = 0.6

# Filter is unreliable on very short text: the per-100 rates spike on a
# single discourse hit and contraction opportunities are sparse. Skip
# (pass-through) below this many assistant-side tokens.
MIN_ASSISTANT_TOKENS_FOR_SCORING = 40
MIN_CONTRACTION_OPPORTUNITIES = 3
MIN_TOKENS_FOR_DISCOURSE_METRIC = 80

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# Contraction inventory. Each entry is (contracted_form_pattern,
# uncontracted_form_pattern). Both patterns are case-insensitive and
# word-boundary anchored.
_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    (r"\bI'm\b", r"\bI am\b"),
    (r"\b(?:you|we|they)'re\b", r"\b(?:you|we|they) are\b"),
    (r"\b(?:he|she|it)'s\b", r"\b(?:he|she|it) is\b"),
    (r"\b(?:I|you|we|they)'ve\b", r"\b(?:I|you|we|they) have\b"),
    (r"\b(?:I|he|she|it|we|you|they)'ll\b", r"\b(?:I|he|she|it|we|you|they) will\b"),
    (r"\b(?:I|he|she|it|we|you|they)'d\b", r"\b(?:I|he|she|it|we|you|they) (?:would|had)\b"),
    (r"\bdon't\b", r"\bdo not\b"),
    (r"\bdoesn't\b", r"\bdoes not\b"),
    (r"\bdidn't\b", r"\bdid not\b"),
    (r"\bisn't\b", r"\bis not\b"),
    (r"\baren't\b", r"\bare not\b"),
    (r"\bwasn't\b", r"\bwas not\b"),
    (r"\bweren't\b", r"\bwere not\b"),
    (r"\bcan't\b", r"\bcannot\b"),
    (r"\bcouldn't\b", r"\bcould not\b"),
    (r"\bwouldn't\b", r"\bwould not\b"),
    (r"\bshouldn't\b", r"\bshould not\b"),
    (r"\bhaven't\b", r"\bhave not\b"),
    (r"\bhasn't\b", r"\bhas not\b"),
    (r"\bhadn't\b", r"\bhad not\b"),
    (r"\bwon't\b", r"\bwill not\b"),
)

# Compiled at module load.
_CONTRACTIONS_COMPILED = [
    (re.compile(c, re.IGNORECASE), re.compile(u, re.IGNORECASE))
    for c, u in _CONTRACTIONS
]

# Discourse markers, lowercased. Match as a phrase with word boundaries.
_DISCOURSE_MARKERS: tuple[str, ...] = (
    "actually", "honestly", "frankly", "to be honest", "by the way",
    "in fact", "for example", "for instance", "in other words",
    "on the other hand", "however", "though", "anyway", "anyhow",
    "well", "right", "okay", "ok", "oh", "ah", "hmm",
    "I mean", "you know", "you see", "I guess", "I suppose",
    "I think", "I'd say", "kind of", "sort of",
    "indeed", "moreover", "furthermore", "nevertheless", "still",
    "after all", "of course", "obviously", "apparently",
)
_DISCOURSE_PATTERNS = [
    re.compile(rf"(?<![A-Za-z]){re.escape(m)}(?![A-Za-z])", re.IGNORECASE)
    for m in _DISCOURSE_MARKERS
]


# ----- per-CEFR target ranges -------------------------------------------


@dataclass(frozen=True)
class MetricRange:
    soft_low: float
    target_low: float
    target_high: float
    soft_high: float

    def score(self, value: float) -> float:
        # Strict inequality on the soft edges so that a value sitting EXACTLY
        # on soft_low (or soft_high) gets evaluated against the target range
        # instead of being short-circuited to 0. This matters for levels where
        # ``soft_low == target_low`` (e.g. A1 contraction_rate at 0.0 — zero
        # contractions is in-target for A1, not a failure).
        if value < self.soft_low or value > self.soft_high:
            return 0.0
        if self.target_low <= value <= self.target_high:
            return 1.0
        if value < self.target_low:
            span = self.target_low - self.soft_low
            return max(0.0, (value - self.soft_low) / span) if span > 0 else 0.0
        # value > target_high
        span = self.soft_high - self.target_high
        return max(0.0, (self.soft_high - value) / span) if span > 0 else 0.0


@dataclass(frozen=True)
class LevelTargets:
    mean_sent_len: MetricRange
    discourse_per_100: MetricRange
    opener_variety: MetricRange
    contraction_rate: MetricRange


# All ranges treat opener_variety as monotone-positive: more variety is
# always at least as good as less (target_high = soft_high = 1.0). For
# contraction_rate, target_low at 0.0 lets dialogues with very few
# contraction opportunities score fairly. Mean sentence length tracks
# the level naturally; discourse_per_100 expects at least some markers
# at every level (zero markers reads as textbook-stilted).
LEVEL_TARGETS: dict[str, LevelTargets] = {
    "A1": LevelTargets(
        mean_sent_len=MetricRange(2.0, 3.0, 8.0, 14.0),
        discourse_per_100=MetricRange(0.0, 0.5, 4.0, 8.0),
        opener_variety=MetricRange(0.1, 0.4, 1.0, 1.0),
        contraction_rate=MetricRange(0.0, 0.0, 0.4, 0.8),
    ),
    "A2": LevelTargets(
        mean_sent_len=MetricRange(2.0, 4.0, 10.0, 18.0),
        discourse_per_100=MetricRange(0.0, 1.0, 5.0, 10.0),
        opener_variety=MetricRange(0.2, 0.5, 1.0, 1.0),
        contraction_rate=MetricRange(0.0, 0.2, 0.6, 0.9),
    ),
    "B1": LevelTargets(
        mean_sent_len=MetricRange(4.0, 6.0, 14.0, 24.0),
        discourse_per_100=MetricRange(0.0, 1.5, 6.0, 12.0),
        opener_variety=MetricRange(0.3, 0.6, 1.0, 1.0),
        contraction_rate=MetricRange(0.0, 0.3, 0.7, 0.95),
    ),
    "B2": LevelTargets(
        mean_sent_len=MetricRange(5.0, 8.0, 18.0, 30.0),
        discourse_per_100=MetricRange(0.5, 2.0, 7.0, 14.0),
        opener_variety=MetricRange(0.4, 0.65, 1.0, 1.0),
        contraction_rate=MetricRange(0.1, 0.4, 0.8, 0.97),
    ),
    "C1": LevelTargets(
        mean_sent_len=MetricRange(6.0, 10.0, 22.0, 36.0),
        discourse_per_100=MetricRange(1.0, 2.5, 9.0, 16.0),
        opener_variety=MetricRange(0.5, 0.7, 1.0, 1.0),
        contraction_rate=MetricRange(0.2, 0.5, 0.85, 0.98),
    ),
    "C2": LevelTargets(
        mean_sent_len=MetricRange(6.0, 10.0, 26.0, 42.0),
        discourse_per_100=MetricRange(1.0, 2.5, 10.0, 18.0),
        opener_variety=MetricRange(0.5, 0.7, 1.0, 1.0),
        contraction_rate=MetricRange(0.2, 0.5, 0.85, 0.98),
    ),
}


# ----- metric computation ------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _split_sentences(text: str) -> list[str]:
    chunks = _SENT_SPLIT_RE.split(text.strip())
    return [c.strip() for c in chunks if c.strip()]


def _contraction_rate(text: str) -> tuple[float, int, int]:
    c_total = 0
    u_total = 0
    for c_pat, u_pat in _CONTRACTIONS_COMPILED:
        c_total += len(c_pat.findall(text))
        u_total += len(u_pat.findall(text))
    denom = c_total + u_total
    rate = c_total / denom if denom > 0 else 0.0
    return rate, c_total, u_total


def _discourse_per_100(text: str, n_tokens: int) -> tuple[float, int]:
    hits = 0
    for pat in _DISCOURSE_PATTERNS:
        hits += len(pat.findall(text))
    rate = (hits / n_tokens * 100.0) if n_tokens > 0 else 0.0
    return rate, hits


def _opener_variety(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    first_words: list[str] = []
    for s in sentences:
        toks = _tokenize(s)
        if toks:
            first_words.append(toks[0].lower())
    if not first_words:
        return 0.0
    return len(set(first_words)) / len(first_words)


# ----- filter ------------------------------------------------------------


class NaturalnessFilter(Filter):
    name = "naturalness"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    async def check(self, example: FilterableExample) -> FilterResult:
        if not isinstance(example, SFTExample):
            return FilterResult(passed=True, reason="not an SFTExample")
        level = example.metadata.cefr_level
        targets = LEVEL_TARGETS.get(level)
        if targets is None:
            return FilterResult(passed=True, reason=f"no targets for level {level}")

        text = "\n".join(m.content for m in example.messages if m.role == "assistant")
        if not text.strip():
            return FilterResult(passed=False, reason="no assistant text")

        tokens = _tokenize(text)
        sentences = _split_sentences(text)
        n_tokens = len(tokens)
        n_sents = len(sentences)
        mean_len = n_tokens / n_sents if n_sents > 0 else 0.0
        c_rate, c_n, u_n = _contraction_rate(text)
        d_rate, d_hits = _discourse_per_100(text, n_tokens)
        variety = _opener_variety(sentences)

        # Skip on very short text — the rate-based metrics are too noisy
        # below ~40 tokens of assistant text to be meaningful.
        if n_tokens < MIN_ASSISTANT_TOKENS_FOR_SCORING:
            return FilterResult(
                passed=True,
                score=None,
                reason=(
                    f"assistant text too short ({n_tokens} tokens) for stable "
                    "naturalness scoring; skipped"
                ),
                metadata={
                    "level": level,
                    "n_tokens": n_tokens,
                    "n_sentences": n_sents,
                    "skipped": True,
                },
            )

        scores: dict[str, float] = {
            "mean_sentence_length": targets.mean_sent_len.score(mean_len),
            "opener_variety": targets.opener_variety.score(variety),
        }
        # Discourse metric only when there are enough tokens for the
        # per-100 rate to stabilize.
        if n_tokens >= MIN_TOKENS_FOR_DISCOURSE_METRIC:
            scores["discourse_per_100"] = targets.discourse_per_100.score(d_rate)
        # Contraction metric only when there are enough contractable
        # opportunities to compute a meaningful rate.
        if (c_n + u_n) >= MIN_CONTRACTION_OPPORTUNITIES:
            scores["contraction_rate"] = targets.contraction_rate.score(c_rate)

        composite = sum(scores.values()) / len(scores)
        passed = composite > self.threshold

        worst = sorted(scores.items(), key=lambda kv: kv[1])[:2]
        reason = None
        if not passed:
            worst_txt = ", ".join(f"{name}={val:.2f}" for name, val in worst)
            reason = (
                f"composite naturalness {composite:.2f} <= {self.threshold:.2f}; "
                f"weakest: {worst_txt}"
            )

        return FilterResult(
            passed=passed,
            score=composite,
            reason=reason,
            metadata={
                "level": level,
                "n_tokens": n_tokens,
                "n_sentences": n_sents,
                "mean_sentence_length": mean_len,
                "contraction_rate": c_rate,
                "contraction_counts": {"contracted": c_n, "uncontracted": u_n},
                "discourse_per_100": d_rate,
                "discourse_marker_hits": d_hits,
                "opener_variety": variety,
                "scored_metrics": list(scores.keys()),
                "scores": scores,
            },
        )
