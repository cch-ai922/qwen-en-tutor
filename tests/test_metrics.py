"""Tests for qwen_tutor.training.eval.metrics.

Sync metrics use synthetic dialogues; LLM-judge metrics use a tiny mock
``TeacherClient`` that returns canned JSON so the tests stay offline and
deterministic.
"""

from __future__ import annotations

import json

import pytest

from qwen_tutor.training.eval.metrics import (
    eval_dimension_scores,
    eval_json_validity,
    level_fidelity,
    locale_fidelity,
    mode_consistency,
    naturalness_judge,
    redirect_success,
    redirect_success_judge,
    topic_adherence,
)


# ---------------------------------------------------------------------------
# Mock judge client (mimics TeacherClient.generate signature)
# ---------------------------------------------------------------------------


class MockJudge:
    """Returns a pre-configured JSON-shaped string from ``generate``."""

    def __init__(self, payload: dict | str):
        self.payload = payload
        self.calls: list[str] = []

    async def generate(
        self,
        system: str,
        messages: list,
        cacheable_prefix: str | None = None,
        max_tokens: int = 0,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(system)
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


# ---------------------------------------------------------------------------
# Synthetic dialogues
# ---------------------------------------------------------------------------


@pytest.fixture
def natural_iranian_dialogue() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "Hi! How was your weekend?"},
        {
            "role": "assistant",
            "content": (
                "Oh, it was really nice. I went to Isfahan with my family. "
                "We walked around Naqsh-e Jahan Square and ate beryani."
            ),
        },
        {"role": "user", "content": "That sounds great. Did you take photos?"},
        {
            "role": "assistant",
            "content": (
                "Yes, lots of them near Si-o-se-pol at sunset. Maryam wanted "
                "faloodeh afterwards, so we walked over to a place in Jolfa."
            ),
        },
        {"role": "user", "content": "I love faloodeh in summer. How was the weather?"},
        {
            "role": "assistant",
            "content": (
                "Pretty hot during the day, actually. By the way, did you try "
                "the new chai house near Tajrish last week? Reza recommended it."
            ),
        },
    ]


@pytest.fixture
def stiff_dialogue() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "How are you?"},
        {
            "role": "assistant",
            "content": (
                "It is the case that I am well today. There are several things "
                "that I would like to mention. It is important to note that "
                "the weather is also good. One may also consider that this "
                "is a typical day. It is also notable that today is unremarkable."
            ),
        },
    ]


@pytest.fixture
def western_dialogue() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "Where did you go on vacation?"},
        {
            "role": "assistant",
            "content": (
                "I went to New York City with Sarah. We visited Times Square, "
                "Central Park, and the Statue of Liberty. We ate hot dogs and "
                "cheesecake. John joined us in Manhattan."
            ),
        },
    ]


@pytest.fixture
def redirect_dialogue() -> list[dict[str, str]]:
    """8 messages: 4 user / 4 assistant. Probe at user turn idx 2."""
    return [
        {"role": "user", "content": "Hi Maryam!"},
        {"role": "assistant", "content": "Hello! How are you today in Tabriz?"},
        {"role": "user", "content": "Good. What did you cook this weekend?"},
        {
            "role": "assistant",
            "content": "I made ghormeh sabzi with my mother. It came out really well.",
        },
        {"role": "user", "content": "What do you think about the election results?"},
        {
            "role": "assistant",
            "content": (
                "That's not something I want to get into. Speaking of cooking "
                "though, did you also try ash reshteh recently? I had some great "
                "ash reshteh at El Goli last week."
            ),
        },
        {"role": "user", "content": "OK, fair enough. Tell me about El Goli."},
        {
            "role": "assistant",
            "content": (
                "El Goli is a beautiful park in Tabriz with a small lake in the "
                "middle. Families go there for picnics in the summer."
            ),
        },
    ]


@pytest.fixture
def evaluation_generation() -> str:
    """Valid evaluation generation: <think>...</think> + JSON."""
    return (
        "<think>The learner used 'goed' instead of 'went' in user turn 0. "
        "Overall A2-appropriate.</think>\n"
        + json.dumps(
            {
                "overall_cefr_estimate": "A2",
                "scores": {"fluency": 3, "accuracy": 2, "vocabulary": 3, "interaction": 3},
                "specific_feedback": [],
                "strengths": ["Clear word order"],
                "suggested_practice": "Drill irregular past forms",
            }
        )
    )


# ---------------------------------------------------------------------------
# level_fidelity
# ---------------------------------------------------------------------------


def test_level_fidelity_returns_all_four_metrics(natural_iranian_dialogue):
    out = level_fidelity(natural_iranian_dialogue, target_cefr="A2")
    assert set(out.keys()) == {
        "above_band_ratio",
        "contraction_rate",
        "discourse_marker_rate",
        "mean_sentence_length",
    }
    for v in out.values():
        assert isinstance(v, float)


def test_level_fidelity_c2_is_open(natural_iranian_dialogue):
    out = level_fidelity(natural_iranian_dialogue, target_cefr="C2")
    assert out["above_band_ratio"] == 0.0


def test_level_fidelity_catches_overshoot():
    over_band = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": (
                "The juxtaposition of nuanced contemporary cultural discourse "
                "exacerbates the underlying ostensible paradoxical tendency."
            ),
        },
    ]
    out = level_fidelity(over_band, target_cefr="A1")
    # Most/all known content words are C1 → above-band ratio should be high.
    assert out["above_band_ratio"] >= 0.5


# ---------------------------------------------------------------------------
# locale_fidelity
# ---------------------------------------------------------------------------
#
# locale_fidelity 는 더 이상 정적 IRANIAN_* 리스트를 내장하지 않습니다.
# Country 가 ``config/locale.yaml`` 로 임의로 바뀔 수 있어, mechanical
# fidelity 점수가 필요할 때는 호출자가 in_locale_terms 를 직접 넘깁니다.
# 아래 set 은 fixture dialogue 가 사용하는 Iranian 어휘를 최소한으로 모아 둔
# 테스트 전용 리스트입니다.

_TEST_IRANIAN_TERMS = (
    "Isfahan", "Naqsh-e Jahan", "Si-o-se-pol", "Jolfa", "Tajrish",
    "Tabriz", "El Goli", "Maryam", "Reza",
)
_TEST_IRANIAN_LOWERCASE_TERMS = (
    "beryani", "faloodeh", "chai", "ghormeh sabzi", "ash reshteh",
)


def test_locale_fidelity_iranian_dialogue_high(natural_iranian_dialogue):
    score = locale_fidelity(
        natural_iranian_dialogue,
        in_locale_terms=_TEST_IRANIAN_TERMS,
        extra_lowercase_terms=_TEST_IRANIAN_LOWERCASE_TERMS,
        use_spacy=False,
    )
    assert score >= 0.7, f"expected high in-locale score, got {score}"


def test_locale_fidelity_western_dialogue_low(western_dialogue):
    score = locale_fidelity(
        western_dialogue,
        in_locale_terms=_TEST_IRANIAN_TERMS,
        extra_lowercase_terms=_TEST_IRANIAN_LOWERCASE_TERMS,
        use_spacy=False,
    )
    assert score <= 0.25, f"expected low in-locale score, got {score}"


def test_locale_fidelity_no_entities_returns_one():
    no_proper = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello, how are you today."},
    ]
    # 고유명사 없음 → in_locale_terms 가 있어도 1.0 (no signal → no penalty).
    assert (
        locale_fidelity(no_proper, in_locale_terms=_TEST_IRANIAN_TERMS, use_spacy=False)
        == 1.0
    )


def test_locale_fidelity_without_terms_returns_one(natural_iranian_dialogue):
    """in_locale_terms 를 안 넘기면 mechanical 판정을 포기하고 1.0 반환."""
    assert locale_fidelity(natural_iranian_dialogue, use_spacy=False) == 1.0


# ---------------------------------------------------------------------------
# mode_consistency
# ---------------------------------------------------------------------------


def test_mode_consistency_conversation_pass():
    assert mode_consistency("Hello, how are you today?", "conversation") is True


def test_mode_consistency_conversation_fail_on_think():
    assert mode_consistency("<think>oops</think>Hi.", "conversation") is False


def test_mode_consistency_evaluation_pass():
    assert mode_consistency("<think>...</think>\n{}", "evaluation") is True


def test_mode_consistency_evaluation_fail_missing_think():
    assert mode_consistency("Just JSON without think.", "evaluation") is False


# ---------------------------------------------------------------------------
# eval_json_validity + eval_dimension_scores
# ---------------------------------------------------------------------------


def test_eval_json_validity_pass(evaluation_generation):
    assert eval_json_validity(evaluation_generation) is True


def test_eval_json_validity_missing_think_fails():
    bad = json.dumps({"overall_cefr_estimate": "A2"})
    assert eval_json_validity(bad) is False


def test_eval_json_validity_missing_fields_fails():
    bad = "<think>x</think>\n" + json.dumps(
        {"overall_cefr_estimate": "A2", "scores": {}}
    )
    assert eval_json_validity(bad) is False


def test_eval_dimension_scores_returns_ints(evaluation_generation):
    scores = eval_dimension_scores(evaluation_generation)
    assert scores == {"fluency": 3, "accuracy": 2, "vocabulary": 3, "interaction": 3}


def test_eval_dimension_scores_invalid_returns_none():
    assert eval_dimension_scores("not an eval generation") is None


def test_eval_dimension_scores_out_of_range_returns_none():
    bad = "<think>x</think>\n" + json.dumps(
        {
            "overall_cefr_estimate": "A2",
            "scores": {"fluency": 7, "accuracy": 3, "vocabulary": 3, "interaction": 3},
            "specific_feedback": [],
            "strengths": [],
            "suggested_practice": "x",
        }
    )
    assert eval_dimension_scores(bad) is None


# ---------------------------------------------------------------------------
# redirect_success
# ---------------------------------------------------------------------------


def test_redirect_success_graceful_pivot(redirect_dialogue):
    # Probe at user-turn idx 2 ("What do you think about the election results?")
    assert redirect_success(
        redirect_dialogue,
        declared_subtopics=["cooking", "food", "park", "weekend"],
        probe_user_turn_idx=2,
    ) is True


def test_redirect_success_fails_when_assistant_engages_substantively():
    bad = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "What about the parliament debate?"},
        {
            "role": "assistant",
            "content": (
                "The parliament had a heated debate last week about the election. "
                "I think the politicians made several important points about the "
                "regime and how the government should handle reform."
            ),
        },
    ]
    assert redirect_success(
        bad, declared_subtopics=["weekend"], probe_user_turn_idx=3
    ) is False


def test_redirect_success_fails_on_trivial_response():
    bad = [
        {"role": "user", "content": "What's your favorite politician?"},
        {"role": "assistant", "content": "Hmm."},
    ]
    assert redirect_success(bad, declared_subtopics=[], probe_user_turn_idx=0) is False


# ---------------------------------------------------------------------------
# topic_adherence (async, with MockJudge)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topic_adherence_full_coverage(natural_iranian_dialogue):
    judge = MockJudge({"covered": 3, "total": 3, "details": []})
    score = await topic_adherence(
        natural_iranian_dialogue,
        declared_subtopics=["food", "places", "weather"],
        judge=judge,
    )
    assert score == 1.0
    assert len(judge.calls) == 1


@pytest.mark.asyncio
async def test_topic_adherence_partial(natural_iranian_dialogue):
    judge = MockJudge({"covered": 1, "total": 4})
    score = await topic_adherence(
        natural_iranian_dialogue,
        declared_subtopics=["food", "weather", "transport", "history"],
        judge=judge,
    )
    assert score == 0.25


@pytest.mark.asyncio
async def test_topic_adherence_empty_subtopics_returns_one(natural_iranian_dialogue):
    judge = MockJudge({"covered": 0, "total": 0})
    score = await topic_adherence(
        natural_iranian_dialogue, declared_subtopics=[], judge=judge
    )
    assert score == 1.0
    # No judge call needed
    assert judge.calls == []


@pytest.mark.asyncio
async def test_topic_adherence_judge_failure_returns_zero(natural_iranian_dialogue):
    class BrokenJudge:
        async def generate(self, **_kw):
            raise RuntimeError("network down")

    score = await topic_adherence(
        natural_iranian_dialogue,
        declared_subtopics=["food"],
        judge=BrokenJudge(),
    )
    assert score == 0.0


# ---------------------------------------------------------------------------
# naturalness_judge (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_naturalness_judge_returns_clipped_int(natural_iranian_dialogue):
    judge = MockJudge({"score": 4, "note": "natural"})
    s = await naturalness_judge(natural_iranian_dialogue, judge=judge, target_cefr="A2")
    assert s == 4


@pytest.mark.asyncio
async def test_naturalness_judge_clips_high(natural_iranian_dialogue):
    judge = MockJudge({"score": 99, "note": "out of range"})
    s = await naturalness_judge(natural_iranian_dialogue, judge=judge)
    assert s == 5


@pytest.mark.asyncio
async def test_naturalness_judge_judge_failure_returns_zero(natural_iranian_dialogue):
    class BrokenJudge:
        async def generate(self, **_kw):
            raise RuntimeError("nope")

    s = await naturalness_judge(natural_iranian_dialogue, judge=BrokenJudge())
    assert s == 0


# ---------------------------------------------------------------------------
# redirect_success_judge (async, optional)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redirect_success_judge_ok():
    judge = MockJudge({"redirect_ok": True, "note": "graceful"})
    ok = await redirect_success_judge(
        probe_user_text="what about politics?",
        assistant_response="Let's stick to food. Did you try ash reshteh?",
        judge=judge,
    )
    assert ok is True


@pytest.mark.asyncio
async def test_redirect_success_judge_fail():
    judge = MockJudge({"redirect_ok": False, "note": "engaged with topic"})
    ok = await redirect_success_judge(
        probe_user_text="politics?", assistant_response="...", judge=judge
    )
    assert ok is False
