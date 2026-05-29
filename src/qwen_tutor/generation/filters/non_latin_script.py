"""Non-Latin-script filter.

영어 conversational tutor 학습 데이터에는 CJK / Cyrillic / Arabic /
Devanagari 등 비-Latin 글자가 들어가면 안 됩니다 (대화 본문은 영어, 이름·
지명은 romanized form 이 원칙). 프롬프트에 명시했지만 teacher 가 가끔
무시하기 때문에 mechanical guard 로 한 번 더 잡아 줍니다.

이 필터는 SFT / DPO / Evaluation 예시 모두에서 다음 위치를 검사합니다:
  - user_role.name, model_role.name (있을 경우)
  - 모든 message 의 content
  - DPO 의 chosen / rejected message content
  - SFT 의 system_prompt 도 한 번 더 점검 (보통 안 문제이지만 안전 차원)

차단 결정: 매칭되는 비-Latin 문자가 단 하나라도 발견되면 fail. score 는
"발견된 비-Latin 문자의 개수" 의 역수로 계산 (정보용; 임계값에는 사용되지
않음 — pass/fail 은 binary).
"""

from __future__ import annotations

import re

from qwen_tutor.generation.filters.base import (
    Filter,
    FilterableExample,
    FilterResult,
)
from qwen_tutor.schemas import DPOExample, EvaluationExample, SFTExample

# 비-Latin script 유니코드 블록들. 좁게 시작해서 필요시 더 넓힐 수 있습니다.
# Latin-1/Latin Extended/공백/구두점/숫자는 제외하고, 영어 대화에 들어가서
# 곤란한 스크립트들만 잡습니다.
_NON_LATIN_RE = re.compile(
    "["
    "぀-ゟ"   # Hiragana
    "゠-ヿ"   # Katakana
    "ㇰ-ㇿ"   # Katakana Phonetic Extensions
    "　-〿"   # CJK Symbols and Punctuation
    "㐀-䶿"   # CJK Unified Ideographs Extension A
    "一-鿿"   # CJK Unified Ideographs
    "豈-﫿"   # CJK Compatibility Ideographs
    "＀-￯"   # Halfwidth and Fullwidth Forms
    "가-힯"   # Hangul Syllables
    "ᄀ-ᇿ"   # Hangul Jamo
    "㄰-㆏"   # Hangul Compatibility Jamo
    "Ѐ-ӿ"   # Cyrillic
    "Ԁ-ԯ"   # Cyrillic Supplement
    "؀-ۿ"   # Arabic
    "ݐ-ݿ"   # Arabic Supplement
    "ࢠ-ࣿ"   # Arabic Extended-A
    "ﭐ-﷿"   # Arabic Presentation Forms-A
    "ﹰ-﻿"   # Arabic Presentation Forms-B
    "֐-׿"   # Hebrew
    "ऀ-ॿ"   # Devanagari
    "ঀ-৿"   # Bengali
    "਀-੿"   # Gurmukhi
    "฀-๿"   # Thai
    "]"
)


def _scan(text: str) -> list[str]:
    """Return a deduplicated list of non-Latin characters found in ``text``.

    Empty / None inputs return an empty list. The list is bounded to the
    first 10 distinct offenders so error messages stay readable.
    """
    if not text:
        return []
    seen: dict[str, None] = {}
    for m in _NON_LATIN_RE.finditer(text):
        ch = m.group(0)
        if ch not in seen:
            seen[ch] = None
            if len(seen) >= 10:
                break
    return list(seen.keys())


class NonLatinScriptFilter(Filter):
    """Reject any record whose role names, messages, or system prompt
    contain non-Latin script characters (CJK / Cyrillic / Arabic /
    Devanagari / Hebrew / Thai / Bengali / Gurmukhi).

    English-language tutor data must be in Latin script — including
    romanized forms of foreign names (``Li Na`` not ``李娜``). The
    locale_instruction_block makes this rule explicit to the teacher,
    but small or distilled models sometimes ignore it. This filter is
    the mechanical guard.
    """

    name = "non_latin_script"

    async def check(self, example: FilterableExample) -> FilterResult:
        fields_to_scan: list[tuple[str, str]] = []

        # Metadata role names (when present — only SFT/DPO have them).
        md = getattr(example, "metadata", None)
        user_role = getattr(md, "user_role", None) if md is not None else None
        model_role = getattr(md, "model_role", None) if md is not None else None
        if user_role is not None and getattr(user_role, "name", None):
            fields_to_scan.append(("user_role.name", user_role.name))
        if model_role is not None and getattr(model_role, "name", None):
            fields_to_scan.append(("model_role.name", model_role.name))

        # System prompt (rendered, may contain locale strings — should still be Latin).
        sys_prompt = getattr(example, "system_prompt", None)
        if isinstance(sys_prompt, str):
            fields_to_scan.append(("system_prompt", sys_prompt))

        # ``speaks_l1`` SFT examples are SUPPOSED to have one user turn in
        # the learner's L1 (non-Latin script). The SpeaksL1SanityFilter
        # validates that turn separately, so here we exempt user turns
        # from the script check on these records and only enforce the
        # script rule on assistant + tutor content.
        gen_meta = (md.generation or {}) if md is not None and hasattr(md, "generation") else {}
        is_speaks_l1 = isinstance(gen_meta, dict) and gen_meta.get("language_trigger") == "speaks_l1"

        # Messages (SFTExample, EvaluationExample) — list of Message objects.
        msgs = getattr(example, "messages", None) or []
        for i, m in enumerate(msgs):
            if is_speaks_l1 and m.role == "user":
                continue  # L1 user turn is allowed/required for speaks_l1
            fields_to_scan.append((f"messages[{i}].content", m.content))

        # DPO-specific fields.
        if isinstance(example, DPOExample):
            for i, m in enumerate(example.prompt_messages):
                if is_speaks_l1 and m.role == "user":
                    continue
                fields_to_scan.append((f"prompt_messages[{i}].content", m.content))
            fields_to_scan.append(("chosen.content", example.chosen.content))
            fields_to_scan.append(("rejected.content", example.rejected.content))

        offenders: list[tuple[str, list[str]]] = []
        total = 0
        for label, text in fields_to_scan:
            hits = _scan(text)
            if hits:
                offenders.append((label, hits))
                total += len(hits)

        if not offenders:
            return FilterResult(
                passed=True,
                score=1.0,
                reason=None,
                metadata={"scanned_fields": len(fields_to_scan)},
            )

        # Build a short reason naming the first 3 offending fields.
        sample = "; ".join(
            f"{label}: {''.join(chars[:5])}" for label, chars in offenders[:3]
        )
        return FilterResult(
            passed=False,
            score=0.0,
            reason=(
                f"non-Latin script characters detected in {len(offenders)} "
                f"field(s): {sample}"
            ),
            metadata={
                "n_offending_fields": len(offenders),
                "n_distinct_chars": total,
            },
        )


__all__ = ["NonLatinScriptFilter"]
