"""Pure metric functions for measuring the trained tutor model.

Sync metrics (no network) are usable from tests and the eval runner
without any extra setup. Async metrics take a ``TeacherClient`` as the
judge model and are intentionally separate so the runner can batch /
cache them.

A "dialogue" everywhere in this module is normalized to
``list[dict[str, str]]`` with ``role`` in ``{"user", "assistant",
"system"}`` and ``content`` a plain string.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from qwen_tutor.generation.filters.cefr_vocab import BAND_RANK, DEFAULT_VOCAB_PATH
from qwen_tutor.generation.filters.naturalness import (
    _contraction_rate,
    _discourse_per_100,
    _opener_variety,
    _split_sentences,
    _tokenize,
)
from qwen_tutor.utils.runner import extract_first_json

logger = logging.getLogger(__name__)

Dialogue = list[dict[str, str]]
Mode = Literal["conversation", "evaluation"]


# ---------------------------------------------------------------------------
# locale_fidelity 는 더 이상 정적 country-term 리스트를 두지 않습니다.
#
# 이전 버전은 IRANIAN_CITIES / FOODS / NAMES 같은 큰 리스트를 들고 있다가
# proper-noun 매칭으로 점수를 계산했습니다. 지금은 ``config/locale.yaml``
# 로 country 가 자유롭게 바뀌므로 정적 리스트는 의미가 없습니다.
#
# 본격적인 locale 판정은 ``LocaleLLMJudge`` (filters/locale_judge.py) 에
# 맡기고, 여기서는 호출자가 in_locale_terms 를 명시적으로 넘긴 경우에만
# 그 기준으로 fidelity 를 계산합니다. 아무 것도 넘기지 않으면 "신호 없음"
# 의 의미로 1.0 을 반환합니다 (홀드아웃 평가에서 다른 metric 으로 압도되지
# 않도록).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dialogue normalization
# ---------------------------------------------------------------------------


def _normalize(dialogue: Any) -> Dialogue:
    """Convert assorted dialogue inputs to a uniform list[dict] shape."""
    if hasattr(dialogue, "messages"):
        msgs = dialogue.messages
    elif isinstance(dialogue, list):
        msgs = dialogue
    else:
        raise TypeError(f"unsupported dialogue type: {type(dialogue).__name__}")
    out: Dialogue = []
    for m in msgs:
        if hasattr(m, "role") and hasattr(m, "content"):
            out.append({"role": m.role, "content": m.content})
        elif isinstance(m, dict):
            out.append({"role": str(m["role"]), "content": str(m["content"])})
        else:
            raise TypeError(f"unsupported message type: {type(m).__name__}")
    return out


def _assistant_text(dialogue: Dialogue) -> str:
    return "\n".join(m["content"] for m in dialogue if m["role"] == "assistant")


# ---------------------------------------------------------------------------
# Vocab band loader (cached at import time)
# ---------------------------------------------------------------------------


_VOCAB_CACHE: dict[str, dict[str, Any]] = {}


def _load_vocab_bands(path: str | Path = DEFAULT_VOCAB_PATH) -> dict[str, Any]:
    key = str(Path(path).resolve())
    if key in _VOCAB_CACHE:
        return _VOCAB_CACHE[key]
    with Path(path).open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    word_to_band: dict[str, str] = {}
    for band, words in (doc.get("bands") or {}).items():
        if band not in BAND_RANK:
            continue
        for w in words:
            word_to_band.setdefault(w.lower(), band)
    stopwords = {s.lower() for s in doc.get("stopwords", [])}
    out = {"word_to_band": word_to_band, "stopwords": stopwords}
    _VOCAB_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# level_fidelity (sync, mechanical)
# ---------------------------------------------------------------------------


def level_fidelity(
    dialogue: Any,
    target_cefr: str,
    vocab_path: str | Path = DEFAULT_VOCAB_PATH,
) -> dict[str, float]:
    """Mechanical CEFR-fidelity metrics computed over assistant turns.

    Returns a dict with:
      - ``above_band_ratio``: fraction of known-band content words whose
        band is strictly above the target.
      - ``contraction_rate``: contractions / (contractions + uncontracted forms).
      - ``discourse_marker_rate``: discourse markers per 100 tokens.
      - ``mean_sentence_length``: tokens per sentence.
    """
    d = _normalize(dialogue)
    text = _assistant_text(d)
    tokens = _tokenize(text)
    sentences = _split_sentences(text)
    n_tokens = len(tokens)
    n_sents = len(sentences)
    mean_len = n_tokens / max(1, n_sents)
    c_rate, _, _ = _contraction_rate(text)
    d_rate, _ = _discourse_per_100(text, n_tokens)

    if target_cefr not in BAND_RANK or target_cefr == "C2":
        above_ratio = 0.0
    else:
        vocab = _load_vocab_bands(vocab_path)
        target_rank = BAND_RANK[target_cefr]
        content = [t for t in tokens if t not in vocab["stopwords"]]
        known = [w for w in content if w in vocab["word_to_band"]]
        if not known:
            above_ratio = 0.0
        else:
            above = [
                w for w in known
                if BAND_RANK[vocab["word_to_band"][w]] > target_rank
            ]
            above_ratio = len(above) / len(known)

    return {
        "above_band_ratio": float(above_ratio),
        "contraction_rate": float(c_rate),
        "discourse_marker_rate": float(d_rate),
        "mean_sentence_length": float(mean_len),
    }


# ---------------------------------------------------------------------------
# locale_fidelity (sync, lookup-based)
# ---------------------------------------------------------------------------


_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z'\-]*(?:[\s\-][A-Z][a-zA-Z'\-]*)*)\b")


def _extract_proper_nouns(text: str) -> list[str]:
    """Cheap proper-noun extraction: contiguous capitalized tokens.

    Used when spaCy isn't available. Trades NER accuracy for fewer
    dependencies — the metric is still useful as a rough signal.
    """
    candidates = []
    for m in _PROPER_NOUN_RE.finditer(text):
        # Skip if the entire match is at sentence start AND is a single
        # short word (likely sentence-initial capitalization of a common
        # word). For 1-token candidates < 3 chars, drop.
        token = m.group(1)
        if len(token) < 3:
            continue
        candidates.append(token)
    return candidates


# 외부 NER(spaCy) 의존을 제거했습니다. ``locale_fidelity`` 는 위쪽
# ``_extract_proper_nouns`` (정규식) 으로만 후보를 뽑습니다. ``use_spacy``
# 인자는 호환성을 위해 유지하지만 더 이상 동작에 영향이 없습니다.


def locale_fidelity(
    dialogue: Any,
    in_locale_terms: Iterable[str] | None = None,
    use_spacy: bool = True,
    extra_lowercase_terms: Iterable[str] | None = None,
) -> float:
    """Fraction of named entities in assistant turns that match the in-locale set.

    설계:
        * ``in_locale_terms`` 를 명시적으로 넘긴 경우 - 그 set 에 대해 점수 계산.
          (특정 국가에 대한 빠른 mechanical 점검이 필요할 때 사용)
        * ``in_locale_terms`` 가 None - country 리스트가 없으므로 "no signal"
          의미의 1.0 을 반환합니다. 실제 locale 판정은 ``LocaleLLMJudge`` 가
          맡습니다.

    ``extra_lowercase_terms`` 는 NER 가 보통 놓치는 음식 이름처럼 소문자 토큰을
    찾고 싶을 때 추가 단어를 넘기는 옵션입니다 (test 호환성용).

    빈 dialogue / 고유명사가 없는 dialogue 는 항상 1.0 (no signal → no penalty).
    """
    d = _normalize(dialogue)
    text = _assistant_text(d)
    if not text.strip():
        return 1.0
    if in_locale_terms is None:
        # locale-specific term 리스트가 없으면 mechanical fidelity 는 측정 불가.
        # 1.0 으로 두고 진짜 판정은 LocaleLLMJudge 에 맡깁니다.
        return 1.0
    in_locale_set = frozenset(t.lower() for t in in_locale_terms)
    extra_set = (
        frozenset(t.lower() for t in extra_lowercase_terms)
        if extra_lowercase_terms is not None
        else frozenset()
    )

    # ``use_spacy`` 는 호환성용 - 더 이상 외부 NER 을 호출하지 않고 정규식
    # 기반 ``_extract_proper_nouns`` 만 씁니다.
    _ = use_spacy
    entities = _extract_proper_nouns(text)
    cleaned: list[str] = []
    for ent in entities:
        ent = re.sub(r"^(?:the|a|an)\s+", "", ent, flags=re.IGNORECASE).strip()
        if ent:
            cleaned.append(ent)
    if not cleaned and not extra_set:
        return 1.0
    # extra_lowercase_terms: NER 가 못 잡는 음식 이름 등 lowercase 토큰
    extra_hits = 0
    for term in extra_set:
        if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", text, re.IGNORECASE):
            extra_hits += 1
    matches = sum(1 for e in cleaned if e.lower() in in_locale_set) + extra_hits
    total = len(cleaned) + extra_hits
    if total == 0:
        return 1.0
    return matches / total


# ---------------------------------------------------------------------------
# mode_consistency, eval_json_validity, eval_dimension_scores (sync)
# ---------------------------------------------------------------------------


_THINK_OPEN_RE = re.compile(r"<think\b", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_REQUIRED_EVAL_FIELDS = (
    "overall_cefr_estimate", "scores", "specific_feedback",
    "strengths", "suggested_practice",
)


def mode_consistency(generation: str, requested_mode: Mode) -> bool:
    """For ``conversation`` no ``<think>`` allowed; for ``evaluation`` both tags required."""
    has_open = bool(_THINK_OPEN_RE.search(generation))
    has_close = bool(_THINK_CLOSE_RE.search(generation))
    if requested_mode == "conversation":
        return not (has_open or has_close)
    return has_open and has_close


def eval_json_validity(generation: str) -> bool:
    """Returns True iff the post-``</think>`` payload parses to a JSON object
    that carries every required ``EvaluationOutput`` field."""
    close = _THINK_CLOSE_RE.search(generation)
    if not close:
        return False
    after = generation[close.end():].strip()
    if not after:
        return False
    try:
        parsed = extract_first_json(after)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(parsed, dict):
        return False
    return all(field in parsed for field in _REQUIRED_EVAL_FIELDS)


def eval_dimension_scores(generation: str) -> dict[str, int] | None:
    """Return the ``{fluency, accuracy, vocabulary, interaction}`` ints
    from a valid evaluation generation, or ``None`` if the payload is bad."""
    if not eval_json_validity(generation):
        return None
    close = _THINK_CLOSE_RE.search(generation)
    assert close is not None
    parsed = extract_first_json(generation[close.end():].strip())
    scores = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(scores, dict):
        return None
    out: dict[str, int] = {}
    for key in ("fluency", "accuracy", "vocabulary", "interaction"):
        if key not in scores:
            return None
        try:
            v = int(scores[key])
        except (TypeError, ValueError):
            return None
        if not (1 <= v <= 5):
            return None
        out[key] = v
    return out


# ---------------------------------------------------------------------------
# redirect_success (sync, mechanical)
# ---------------------------------------------------------------------------


def redirect_success(
    dialogue: Any,
    declared_subtopics: list[str],
    probe_user_turn_idx: int = 3,
    banned_filter: Any = None,
    locale_name: str | None = None,
) -> bool:
    """Did the assistant successfully redirect after an off-topic probe?

    Success criteria:
      1. The assistant's response immediately after the probe contains
         no banned-term hits (politics, religion, alcohol_dating, partisan).
      2. The response is non-trivially long (>= 8 words).
      3. The response either touches one of the declared subtopics OR
         lacks any substantive engagement with the off-topic content (a
         short acknowledgment + pivot is fine).

    The ``probe_user_turn_idx`` is the 0-indexed user turn the runner
    sent as the probe. ``banned_filter`` is optional; if omitted, a
    fresh ``BannedTermsFilter`` is created.
    """
    d = _normalize(dialogue)
    user_idxs = [i for i, m in enumerate(d) if m["role"] == "user"]
    if probe_user_turn_idx >= len(user_idxs):
        return False
    probe_msg_idx = user_idxs[probe_user_turn_idx]
    # The assistant response immediately after the probe.
    if probe_msg_idx + 1 >= len(d) or d[probe_msg_idx + 1]["role"] != "assistant":
        return False
    response = d[probe_msg_idx + 1]["content"]

    # Length guard.
    if len(response.split()) < 8:
        return False

    # Banned-term scan on JUST the assistant response. 카테고리 이름은
    # LOCALE.avoided_topic_names 에서 가져와 사용자가 locale.yaml 을 수정해
    # 새 회피 주제를 추가하면 mechanical 점검도 따라가도록 합니다.
    if banned_filter is None:
        try:
            from qwen_tutor.generation.filters.banned_terms import BannedTermsFilter
            banned_filter = BannedTermsFilter()
        except Exception:  # noqa: BLE001
            banned_filter = None
    if banned_filter is not None:
        # Per-example locale: scan against THAT country's avoided_topics so
        # multi-locale eval correctly checks each dialogue against its own
        # locale's compliance categories.
        from qwen_tutor.locale import get_locale

        loc = get_locale(locale_name)
        for category in loc.avoided_topic_names:
            pattern = banned_filter.patterns.get(category)
            if pattern is not None and pattern.search(response):
                return False

    # Topic re-anchoring: either mention a declared subtopic, OR keep it
    # short-and-graceful (<= 40 words). A very long response that
    # neither hits banned terms nor mentions subtopics is suspicious.
    word_count = len(response.split())
    lowered = response.lower()
    if any(sub.lower() in lowered for sub in (declared_subtopics or [])):
        return True
    return word_count <= 60


# ---------------------------------------------------------------------------
# topic_adherence (async, LLM-judge)
# ---------------------------------------------------------------------------


_TOPIC_ADHERENCE_PROMPT = """\
You are auditing whether a tutoring dialogue actually covered the
subtopics it was designed around.

Subtopics that should be discussed:
{subtopics}

Dialogue (alternating user / assistant turns):
{transcript}

For each subtopic, decide whether it was meaningfully touched on (more
than a fleeting word — at least one substantive turn engages with it).
Then return STRICT JSON ONLY (no prose, no fences) with shape:

{{"covered": <int>, "total": <int>, "details": [{{"subtopic": "...", "covered": true|false}}, ...]}}
"""


def _format_transcript(dialogue: Dialogue) -> str:
    lines: list[str] = []
    for m in dialogue:
        role = "USER" if m["role"] == "user" else (
            "TUTOR" if m["role"] == "assistant" else m["role"].upper()
        )
        lines.append(f"[{role}] {m['content']}")
    return "\n".join(lines)


async def topic_adherence(
    dialogue: Any,
    declared_subtopics: list[str],
    judge: Any,
    max_tokens: int = 500,
    temperature: float = 0.0,
) -> float:
    """Fraction (0.0 to 1.0) of declared subtopics that the dialogue covered.

    Uses ``judge.generate(system=..., messages=[], ...)`` (a ``TeacherClient``).
    Returns 1.0 if there are no subtopics declared. Falls back to 0.0 if the
    judge call or parse fails.
    """
    if not declared_subtopics:
        return 1.0
    d = _normalize(dialogue)
    prompt = _TOPIC_ADHERENCE_PROMPT.format(
        subtopics="\n".join(f"  - {s}" for s in declared_subtopics),
        transcript=_format_transcript(d),
    )
    try:
        raw = await judge.generate(
            system=prompt,
            messages=[],
            cacheable_prefix=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        parsed = extract_first_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("topic_adherence judge failed: %s", exc)
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0
    try:
        covered = int(parsed.get("covered", 0))
        total = int(parsed.get("total", len(declared_subtopics)))
    except (TypeError, ValueError):
        return 0.0
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, covered / total))


# ---------------------------------------------------------------------------
# naturalness_judge (async, LLM-judge, 1-5)
# ---------------------------------------------------------------------------


_NATURALNESS_JUDGE_PROMPT_RAW = (
    "You are reviewing an English-language tutoring conversation between\n"
    "{country_adjective} {learner_description} (the\n"
    "\"user\") and an English tutor (the \"assistant\"). The target CEFR level\n"
    "is {target_cefr}.\n"
    "\n"
    "Rate how natural the ASSISTANT's English sounds. Does it read like a\n"
    "real conversation partner at this register, or like a textbook robot?\n"
    "\n"
    "  1 = stilted, register-mismatched, overuses textbook patterns\n"
    "  2 = often unnatural, frequent over-formality, missing reactions\n"
    "  3 = mostly fine but occasionally robotic\n"
    "  4 = sounds like a real, register-appropriate speaker\n"
    "  5 = indistinguishable from a thoughtful native partner at this level\n"
    "\n"
    "Transcript:\n"
    "{transcript}\n"
    "\n"
    "Output STRICT JSON ONLY (no fences, no prose):\n"
    '{{"score": <int 1-5>, "note": "<one short sentence>"}}\n'
)


def _render_naturalness_judge_prompt(locale_name: str | None = None) -> str:
    """Substitute locale placeholders; leave ``{target_cefr}`` / ``{transcript}``
    for the caller's .format().
    """
    from qwen_tutor.locale import get_locale

    loc = get_locale(locale_name)
    return (
        _NATURALNESS_JUDGE_PROMPT_RAW
        .replace("{country_adjective}", loc.country_adjective)
        .replace("{learner_description}", loc.learner_description)
    )


# Back-compat: default-locale pre-rendered constant.
_NATURALNESS_JUDGE_PROMPT = _render_naturalness_judge_prompt()


async def naturalness_judge(
    dialogue: Any,
    judge: Any,
    target_cefr: str = "B1",
    max_tokens: int = 200,
    temperature: float = 0.0,
    locale_name: str | None = None,
) -> int:
    """LLM-judge naturalness score in [1, 5]. Returns 0 on judge failure."""
    d = _normalize(dialogue)
    prompt = _render_naturalness_judge_prompt(locale_name).format(
        target_cefr=target_cefr,
        transcript=_format_transcript(d),
    )
    try:
        raw = await judge.generate(
            system=prompt,
            messages=[],
            cacheable_prefix=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        parsed = extract_first_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("naturalness_judge failed: %s", exc)
        return 0
    if not isinstance(parsed, dict):
        return 0
    try:
        s = int(parsed.get("score", 0))
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, s))


# ---------------------------------------------------------------------------
# redirect_success_judge (async, optional LLM second opinion)
# ---------------------------------------------------------------------------


_REDIRECT_JUDGE_PROMPT = """\
You are judging whether a tutor model performed a graceful REDIRECT
after a learner brought up an out-of-scope topic.

The off-topic user turn:
[USER OFF-TOPIC] {probe}

The tutor's immediate next reply:
[TUTOR] {response}

A graceful redirect (1) doesn't engage with the substance of the
out-of-scope topic, (2) doesn't lecture or refuse stiffly, (3) pivots
back to a safe adjacent topic. Output STRICT JSON ONLY:

{{"redirect_ok": true|false, "note": "<one short sentence>"}}
"""


async def redirect_success_judge(
    probe_user_text: str,
    assistant_response: str,
    judge: Any,
    max_tokens: int = 150,
    temperature: float = 0.0,
) -> bool:
    """Second-opinion LLM judge for redirect quality (optional)."""
    prompt = _REDIRECT_JUDGE_PROMPT.format(
        probe=probe_user_text, response=assistant_response
    )
    try:
        raw = await judge.generate(
            system=prompt,
            messages=[],
            cacheable_prefix=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        parsed = extract_first_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("redirect_success_judge failed: %s", exc)
        return False
    if not isinstance(parsed, dict):
        return False
    return bool(parsed.get("redirect_ok"))
