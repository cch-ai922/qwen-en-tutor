"""Mode-consistency filter.

Different example types are expected to obey different output-shape
contracts:

  * SFTExample with ``scenario_type`` in {"normal", "redirect"} must
    contain plain conversation in the assistant turns. Any ``<think>``
    or ``</think>`` tag is a leak from the evaluation pipeline.

  * EvaluationExample's assistant turn must contain BOTH a
    ``<think>...</think>`` block AND a parseable JSON object after the
    closing tag.

  * Other examples (DPOExample, etc.) pass unconditionally — they have
    their own structural constraints checked elsewhere.
"""

from __future__ import annotations

import json
import re
from typing import Any

from qwen_tutor.generation.filters.base import Filter, FilterableExample, FilterResult
from qwen_tutor.schemas import EvaluationExample, SFTExample
from qwen_tutor.utils.runner import extract_first_json

_THINK_OPEN_RE = re.compile(r"<think\b", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


class ModeConsistencyFilter(Filter):
    name = "mode_consistency"

    async def check(self, example: FilterableExample) -> FilterResult:
        if isinstance(example, SFTExample):
            return self._check_sft(example)
        if isinstance(example, EvaluationExample):
            return self._check_eval(example)
        return FilterResult(passed=True)

    def _check_sft(self, example: SFTExample) -> FilterResult:
        st = example.metadata.scenario_type
        if st not in ("normal", "redirect"):
            return FilterResult(passed=True, reason=f"scenario_type={st} not gated")
        for i, msg in enumerate(example.messages):
            if msg.role != "assistant":
                continue
            if _THINK_OPEN_RE.search(msg.content) or _THINK_CLOSE_RE.search(msg.content):
                return FilterResult(
                    passed=False,
                    reason=(
                        f"SFT assistant turn {i} contains <think> tag "
                        f"(scenario_type={st})"
                    ),
                    metadata={"turn_index": i},
                )
        return FilterResult(passed=True)

    def _check_eval(self, example: EvaluationExample) -> FilterResult:
        assistant_turns = [m for m in example.messages if m.role == "assistant"]
        if not assistant_turns:
            return FilterResult(
                passed=False, reason="evaluation example has no assistant turn"
            )
        content = assistant_turns[-1].content
        has_open = bool(_THINK_OPEN_RE.search(content))
        has_close = bool(_THINK_CLOSE_RE.search(content))
        if not (has_open and has_close):
            return FilterResult(
                passed=False,
                reason="evaluation assistant turn missing <think>...</think> block",
                metadata={"has_open": has_open, "has_close": has_close},
            )
        # After </think>, there must be a parseable JSON object.
        close_match = _THINK_CLOSE_RE.search(content)
        assert close_match is not None
        after = content[close_match.end() :].strip()
        if not after:
            return FilterResult(
                passed=False, reason="no content after </think> tag"
            )
        try:
            parsed = extract_first_json(after)
        except json.JSONDecodeError as exc:
            return FilterResult(
                passed=False,
                reason=f"unparseable JSON after </think>: {exc.msg}",
            )
        if not isinstance(parsed, dict):
            return FilterResult(
                passed=False,
                reason=f"JSON after </think> is {type(parsed).__name__}, not object",
            )
        required = {"overall_cefr_estimate", "scores", "specific_feedback", "strengths", "suggested_practice"}
        missing = required - set(parsed.keys())
        if missing:
            return FilterResult(
                passed=False,
                reason=f"evaluation JSON missing required fields: {sorted(missing)}",
                metadata={"missing_fields": sorted(missing)},
            )
        return FilterResult(passed=True)
