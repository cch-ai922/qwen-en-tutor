"""Optional naturalness LLM-judge (spot-check).

The mechanical NaturalnessFilter scores a small set of register metrics
deterministically. This judge runs on a *sample* of examples (default
5%) and asks a cheap model for a holistic 1-5 naturalness score plus a
short justification. Use it as a sanity check that the mechanical
metrics aren't drifting away from what a fluent reader would call
natural.

Sampling is deterministic per example id (hash → keep/drop) so reruns
spot-check the same examples.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from qwen_tutor.generation.filters.base import Filter, FilterableExample, FilterResult
from qwen_tutor.generation.teacher import TeacherClient
from qwen_tutor.schemas import Message, SFTExample
from qwen_tutor.utils.runner import extract_first_json

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 0.05
DEFAULT_MIN_SCORE = 3.0


# Judge prompt — instructions ONLY. Dynamic data (target CEFR + transcript)
# arrives in the USER message constructed in ``NaturalnessLLMJudge.check``.
_JUDGE_PROMPT_RAW = (
    "You are reviewing an English-language conversation between a "
    "{country_adjective}\n"
    "{learner_description} (the \"user\") and an English tutor (the\n"
    "\"assistant\").\n"
    "\n"
    "The input — target CEFR level and the full transcript — is delivered\n"
    "as the USER message immediately following these instructions.\n"
    "\n"
    "Rate how natural the ASSISTANT's English sounds - does it read like a\n"
    "real conversation partner, or like a textbook robot? Score from 1 to 5:\n"
    "\n"
    "  1 = stilted, register-mismatched, overuses textbook patterns\n"
    "  2 = often unnatural, frequent over-formality or missing reactions\n"
    "  3 = mostly fine but occasionally robotic\n"
    "  4 = sounds like a real, register-appropriate speaker\n"
    "  5 = indistinguishable from a thoughtful native partner at this level\n"
    "\n"
    "Also note in one short sentence what most influenced your score.\n"
    "\n"
    "Output STRICT JSON ONLY, no markdown fences:\n"
    "\n"
    '{"score": <int 1-5>, "note": "<one short sentence>"}\n'
)


def _render_judge_prompt(locale_name: str | None = None) -> str:
    """Substitute locale placeholders for the given locale; no per-call
    placeholders remain — dynamic data goes in the USER message at call time.
    """
    from qwen_tutor.locale import get_locale

    loc = get_locale(locale_name)
    return (
        _JUDGE_PROMPT_RAW
        .replace("{country_adjective}", loc.country_adjective)
        .replace("{learner_description}", loc.learner_description)
    )


# Back-compat: default-locale-pre-rendered constant.
JUDGE_PROMPT = _render_judge_prompt()


def _render_transcript(example: SFTExample) -> str:
    lines: list[str] = []
    for m in example.messages:
        role = "User" if m.role == "user" else ("Tutor" if m.role == "assistant" else m.role.capitalize())
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def _hash_keep(example_id: str, sample_rate: float) -> bool:
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < sample_rate


class NaturalnessLLMJudge(Filter):
    name = "naturalness_judge"

    def __init__(
        self,
        judge: TeacherClient,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        min_score: float = DEFAULT_MIN_SCORE,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> None:
        self.judge = judge
        self.sample_rate = sample_rate
        self.min_score = min_score
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def check(self, example: FilterableExample) -> FilterResult:
        if not isinstance(example, SFTExample):
            return FilterResult(passed=True, reason="not an SFTExample")
        if not _hash_keep(example.id, self.sample_rate):
            return FilterResult(
                passed=True, reason="not sampled", metadata={"sampled": False}
            )

        transcript = _render_transcript(example)
        locale_name = getattr(example.metadata, "locale", None)
        system_prompt = _render_judge_prompt(locale_name)
        user_message = (
            f"Target CEFR level: {example.metadata.cefr_level}\n"
            f"\n"
            f"Transcript:\n"
            f"{transcript}"
        )
        try:
            raw = await self.judge.generate(
                system=system_prompt,
                messages=[Message(role="user", content=user_message)],
                cacheable_prefix=None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            parsed = extract_first_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("naturalness_judge: judge call failed for %s: %s", example.id, exc)
            return FilterResult(
                passed=True,
                reason=f"judge unavailable ({exc.__class__.__name__}); skipping",
                metadata={"sampled": True, "judge_error": str(exc)},
            )
        if not isinstance(parsed, dict) or "score" not in parsed:
            return FilterResult(
                passed=True,
                reason="judge returned malformed JSON; skipping",
                metadata={"sampled": True, "raw_truncated": str(parsed)[:200]},
            )
        try:
            score = float(parsed["score"])
        except (TypeError, ValueError):
            return FilterResult(
                passed=True,
                reason="judge score not a number; skipping",
                metadata={"sampled": True, "raw": parsed},
            )
        note = parsed.get("note", "")
        passed = score >= self.min_score
        return FilterResult(
            passed=passed,
            score=score,
            reason=(
                None
                if passed
                else f"judge naturalness {score:.1f} < {self.min_score:.1f} ({note!r})"
            ),
            metadata={"sampled": True, "judge_score": score, "judge_note": note},
        )
