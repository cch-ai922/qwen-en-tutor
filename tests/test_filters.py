"""Tests for mechanical filters: banned_terms, mode_consistency,
naturalness, non_latin_script.

LLM-judge filters (locale_judge, judge.py) are intentionally not
covered here — they require network mocks and live in separate tests.
"""

from __future__ import annotations

import pytest

from qwen_tutor.generation.filters.banned_terms import BannedTermsFilter
from qwen_tutor.generation.filters.mode_consistency import ModeConsistencyFilter
from qwen_tutor.generation.filters.naturalness import (
    LEVEL_TARGETS,
    MetricRange,
    NaturalnessFilter,
)
from qwen_tutor.generation.filters.non_latin_script import NonLatinScriptFilter
from qwen_tutor.schemas import (
    DPOExample,
    EvaluationExample,
    EvaluationMetadata,
    ExampleMetadata,
    Message,
    ModelRole,
    SFTExample,
    UserRole,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user_role() -> UserRole:
    return UserRole(name="Maryam", description="A2 Iranian learner.")


@pytest.fixture
def model_role() -> ModelRole:
    return ModelRole(name="Hassan", description="An Iranian tutor.")


def _make_sft(
    user_role: UserRole,
    model_role: ModelRole,
    *,
    messages: list[Message],
    cefr_level: str = "A2",
    scenario_type: str = "normal",
    locale: str = "iran",
    generation: dict | None = None,
) -> SFTExample:
    return SFTExample(
        id="sft-test",
        metadata=ExampleMetadata(
            topic="weekend plans",
            subtopics=["family", "food", "city"],
            user_role=user_role,
            model_role=model_role,
            cefr_level=cefr_level,
            scenario_type=scenario_type,
            locale=locale,
            generation=generation,
        ),
        system_prompt="Be a patient tutor.",
        messages=messages,
    )


# ---------------------------------------------------------------------------
# BannedTermsFilter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_banned_terms_passes_clean_text(user_role, model_role):
    ex = _make_sft(
        user_role, model_role,
        messages=[
            Message(role="user", content="I went to Tehran last weekend."),
            Message(role="assistant", content="That sounds great. What did you do there?"),
        ],
    )
    result = await BannedTermsFilter().check(ex)
    assert result.passed is True


@pytest.mark.asyncio
async def test_banned_terms_catches_politics_term_in_assistant(user_role, model_role):
    # "election" is in the politics category of banned_terms.yaml.
    ex = _make_sft(
        user_role, model_role,
        messages=[
            Message(role="user", content="What did you do yesterday?"),
            Message(role="assistant", content="I read about the election results."),
        ],
    )
    result = await BannedTermsFilter().check(ex)
    assert result.passed is False
    assert "banned-term hit" in (result.reason or "")
    assert result.metadata["hits"]


@pytest.mark.asyncio
async def test_banned_terms_exempts_user_turn_on_redirect(user_role, model_role):
    # In a redirect scenario the user is SCRIPTED to drop a banned term;
    # the tutor's job is to pivot. The filter must not penalize this.
    ex = _make_sft(
        user_role, model_role,
        scenario_type="redirect",
        messages=[
            Message(role="user", content="What do you think about the election?"),
            Message(role="assistant", content="Let's stick to your weekend instead. What did you cook?"),
        ],
    )
    result = await BannedTermsFilter().check(ex)
    assert result.passed is True, (
        f"redirect-scenario user turn should be exempt, but got: {result.reason}"
    )


@pytest.mark.asyncio
async def test_banned_terms_still_catches_assistant_leak_on_redirect(user_role, model_role):
    # Redirect exempts user turns, but the tutor must NEVER leak banned content.
    ex = _make_sft(
        user_role, model_role,
        scenario_type="redirect",
        messages=[
            Message(role="user", content="What about elections?"),
            Message(role="assistant", content="The communist parliament voted last week."),
        ],
    )
    result = await BannedTermsFilter().check(ex)
    assert result.passed is False


# ---------------------------------------------------------------------------
# ModeConsistencyFilter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mode_consistency_sft_passes_without_think(user_role, model_role):
    ex = _make_sft(
        user_role, model_role,
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="Hello! How are you today?"),
        ],
    )
    result = await ModeConsistencyFilter().check(ex)
    assert result.passed is True


@pytest.mark.asyncio
async def test_mode_consistency_sft_fails_on_leaked_think(user_role, model_role):
    ex = _make_sft(
        user_role, model_role,
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="<think>I should be friendly</think>Hello!"),
        ],
    )
    result = await ModeConsistencyFilter().check(ex)
    assert result.passed is False
    assert "think" in (result.reason or "").lower()


@pytest.mark.asyncio
async def test_mode_consistency_eval_passes_with_think_and_valid_json():
    eval_ex = EvaluationExample(
        id="eval-1",
        metadata=EvaluationMetadata(
            source_dialogue_id="sft-1", learner_cefr_target="A2",
        ),
        system_prompt="examiner.",
        messages=[
            Message(role="user", content="Transcript:\n[USER turn 0] I went there."),
            Message(
                role="assistant",
                content=(
                    "<think>The learner used past tense correctly.</think>\n"
                    '{"overall_cefr_estimate":"A2","scores":{"fluency":3,"accuracy":3,'
                    '"vocabulary":3,"interaction":3,"topic_adherence":3},'
                    '"specific_feedback":[],"strengths":[],"suggested_practice":"x"}'
                ),
            ),
        ],
    )
    result = await ModeConsistencyFilter().check(eval_ex)
    assert result.passed is True


@pytest.mark.asyncio
async def test_mode_consistency_eval_fails_without_think():
    eval_ex = EvaluationExample(
        id="eval-2",
        metadata=EvaluationMetadata(
            source_dialogue_id="sft-1", learner_cefr_target="A2",
        ),
        system_prompt="examiner.",
        messages=[
            Message(role="user", content="Transcript:"),
            Message(role="assistant", content='{"overall_cefr_estimate":"A2"}'),
        ],
    )
    result = await ModeConsistencyFilter().check(eval_ex)
    assert result.passed is False


@pytest.mark.asyncio
async def test_mode_consistency_eval_fails_on_missing_required_fields():
    eval_ex = EvaluationExample(
        id="eval-3",
        metadata=EvaluationMetadata(
            source_dialogue_id="sft-1", learner_cefr_target="A2",
        ),
        system_prompt="examiner.",
        messages=[
            Message(role="user", content="Transcript:"),
            Message(
                role="assistant",
                content='<think>x</think>\n{"overall_cefr_estimate":"A2"}',
            ),
        ],
    )
    result = await ModeConsistencyFilter().check(eval_ex)
    assert result.passed is False
    assert "missing" in (result.reason or "").lower()


# ---------------------------------------------------------------------------
# MetricRange.score — the recent boundary bug regression test
# ---------------------------------------------------------------------------


def test_metric_range_value_at_soft_low_inside_target_scores_one():
    # The bug fixed in naturalness.py: value EXACTLY on soft_low was returning
    # 0 even when soft_low == target_low (e.g. A1 contraction_rate where
    # zero contractions is in-target). Strict-inequality fix means 0.0 in a
    # [0.0, 0.4] target band scores 1.0.
    r = MetricRange(soft_low=0.0, target_low=0.0, target_high=0.4, soft_high=0.8)
    assert r.score(0.0) == 1.0
    assert r.score(0.4) == 1.0


def test_metric_range_value_in_band_scores_one():
    r = MetricRange(soft_low=0.0, target_low=0.2, target_high=0.6, soft_high=0.9)
    assert r.score(0.4) == 1.0


def test_metric_range_value_below_soft_low_scores_zero():
    r = MetricRange(soft_low=0.1, target_low=0.2, target_high=0.6, soft_high=0.9)
    assert r.score(0.05) == 0.0


def test_metric_range_value_above_soft_high_scores_zero():
    r = MetricRange(soft_low=0.0, target_low=0.2, target_high=0.6, soft_high=0.9)
    assert r.score(0.95) == 0.0


def test_metric_range_ramps_linearly_below_target():
    r = MetricRange(soft_low=0.0, target_low=0.4, target_high=0.6, soft_high=0.9)
    # 0.2 is halfway between soft_low (0.0) and target_low (0.4) → score 0.5
    assert r.score(0.2) == pytest.approx(0.5)


def test_metric_range_ramps_linearly_above_target():
    r = MetricRange(soft_low=0.0, target_low=0.4, target_high=0.6, soft_high=0.8)
    # 0.7 is halfway between target_high (0.6) and soft_high (0.8) → score 0.5
    assert r.score(0.7) == pytest.approx(0.5)


def test_a1_contraction_rate_zero_is_in_target():
    # End-to-end regression for the boundary fix using real LEVEL_TARGETS.
    # Without the fix, every A1 example with zero contractions failed.
    a1 = LEVEL_TARGETS["A1"]
    assert a1.contraction_rate.score(0.0) == 1.0


# ---------------------------------------------------------------------------
# NaturalnessFilter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_naturalness_skipped_on_short_text(user_role, model_role):
    # Under MIN_ASSISTANT_TOKENS_FOR_SCORING (40) → pass-through with reason.
    ex = _make_sft(
        user_role, model_role,
        cefr_level="A2",
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="Hello! How are you today?"),
        ],
    )
    result = await NaturalnessFilter().check(ex)
    assert result.passed is True
    assert "too short" in (result.reason or "")
    assert result.metadata["skipped"] is True


@pytest.mark.asyncio
async def test_naturalness_passes_balanced_b1_text(user_role, model_role):
    # Long enough for all metrics; mixes contracted and uncontracted forms
    # and uses a couple of discourse markers, with varied openers.
    natural = (
        "Honestly, the market was busier than I expected. I'm glad we arrived early, "
        "because by ten o'clock the queues stretched all the way down the street. "
        "You know, the vendors are friendlier on weekday mornings. We were not sure "
        "what to pick, so the fishmonger suggested a small piece of trout. "
        "Actually, that turned out to be the best choice. After lunch, we walked to "
        "the riverside and sat for a while. It is a calm place. I think we will go "
        "back next month, if the weather holds up."
    )
    ex = _make_sft(
        user_role, model_role,
        cefr_level="B1",
        messages=[
            Message(role="user", content="How was your day?"),
            Message(role="assistant", content=natural),
        ],
    )
    result = await NaturalnessFilter().check(ex)
    assert result.passed is True, f"expected pass; reason={result.reason} score={result.score}"


@pytest.mark.asyncio
async def test_naturalness_non_sft_passes_through(user_role, model_role):
    # DPOExample is not an SFTExample; filter should pass-through.
    dpo = DPOExample(
        id="dpo-1",
        metadata=ExampleMetadata(
            topic="t", subtopics=["a", "b", "c"],
            user_role=user_role, model_role=model_role,
            cefr_level="B1", scenario_type="normal",
        ),
        system_prompt="x",
        prompt_messages=[Message(role="user", content="hi")],
        chosen=Message(role="assistant", content="hello"),
        rejected=Message(role="assistant", content="yo"),
        rejection_axis="register_unnatural",
        rejection_note="too casual",
    )
    result = await NaturalnessFilter().check(dpo)
    assert result.passed is True
    assert "not an SFTExample" in (result.reason or "")


# ---------------------------------------------------------------------------
# NonLatinScriptFilter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_latin_passes_clean_english(user_role, model_role):
    ex = _make_sft(
        user_role, model_role,
        messages=[
            Message(role="user", content="I went to Isfahan."),
            Message(role="assistant", content="Naqsh-e Jahan Square is wonderful in spring."),
        ],
    )
    result = await NonLatinScriptFilter().check(ex)
    assert result.passed is True


@pytest.mark.asyncio
async def test_non_latin_fails_on_cjk_in_assistant(user_role, model_role):
    ex = _make_sft(
        user_role, model_role,
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="你好! Welcome to the lesson."),
        ],
    )
    result = await NonLatinScriptFilter().check(ex)
    assert result.passed is False
    assert "non-Latin" in (result.reason or "")


@pytest.mark.asyncio
async def test_non_latin_fails_on_arabic_in_user(user_role, model_role):
    ex = _make_sft(
        user_role, model_role,
        messages=[
            Message(role="user", content="مرحبا"),
            Message(role="assistant", content="Hello."),
        ],
    )
    result = await NonLatinScriptFilter().check(ex)
    assert result.passed is False


@pytest.mark.asyncio
async def test_non_latin_exempts_speaks_l1_user_turn(user_role, model_role):
    # The speaks_l1 language_redirect stream EXPECTS one user turn in L1.
    # NonLatinScriptFilter must exempt user turns on these records, leaving
    # the L1-validation to SpeaksL1SanityFilter.
    ex = _make_sft(
        user_role, model_role,
        generation={"language_trigger": "speaks_l1"},
        messages=[
            Message(role="user", content="我想学英语"),
            Message(role="assistant", content="Let's switch to English. What would you like to talk about?"),
        ],
    )
    result = await NonLatinScriptFilter().check(ex)
    assert result.passed is True, (
        f"speaks_l1 user turn must be exempt; got: {result.reason}"
    )


@pytest.mark.asyncio
async def test_non_latin_still_catches_assistant_leak_on_speaks_l1(user_role, model_role):
    # Exemption is user-only. An assistant turn with non-Latin script still fails.
    ex = _make_sft(
        user_role, model_role,
        generation={"language_trigger": "speaks_l1"},
        messages=[
            Message(role="user", content="我想学英语"),
            Message(role="assistant", content="好的, let's practice."),
        ],
    )
    result = await NonLatinScriptFilter().check(ex)
    assert result.passed is False
