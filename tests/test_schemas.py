"""Round-trip JSONL serialization tests for qwen_tutor.schemas."""

import pytest
from pydantic import ValidationError

from qwen_tutor.schemas import (
    DPOExample,
    EvaluationExample,
    EvaluationMetadata,
    EvaluationOutput,
    ExampleMetadata,
    Message,
    ModelRole,
    ScenarioSeed,
    ScoreBreakdown,
    SFTExample,
    TurnFeedback,
    UserRole,
)


@pytest.fixture
def user_role() -> UserRole:
    return UserRole(
        name="Maryam",
        description="An A2 Iranian learner curious about her city.",
    )


@pytest.fixture
def model_role() -> ModelRole:
    return ModelRole(
        name="English Tutor",
        description="A patient English conversation tutor.",
    )


@pytest.fixture
def example_metadata(user_role: UserRole, model_role: ModelRole) -> ExampleMetadata:
    return ExampleMetadata(
        topic="weekend plans",
        subtopics=["family", "city", "food"],
        user_role=user_role,
        model_role=model_role,
        cefr_level="A2",
        scenario_type="normal",
        generation={"provider": "anthropic", "model": "claude-opus-4-7"},
    )


@pytest.fixture
def sft_example(example_metadata: ExampleMetadata) -> SFTExample:
    return SFTExample(
        id="sft-001",
        metadata=example_metadata,
        system_prompt="You are a patient English tutor for an A2 Iranian learner.",
        messages=[
            Message(role="user", content="I went to Isfahan last weekend."),
            Message(
                role="assistant",
                content="That sounds great. Did you visit Naqsh-e Jahan Square?",
            ),
        ],
        quality_signals={"locale_check": "pass", "cefr_check": "pass"},
    )


@pytest.fixture
def dpo_example(example_metadata: ExampleMetadata) -> DPOExample:
    return DPOExample(
        id="dpo-001",
        metadata=example_metadata,
        system_prompt="You are a patient English tutor for an A2 Iranian learner.",
        prompt_messages=[
            Message(role="user", content="I went to Isfahan last weekend."),
        ],
        chosen=Message(
            role="assistant",
            content="That sounds great. Did you visit Naqsh-e Jahan Square?",
        ),
        rejected=Message(
            role="assistant",
            content="Cool, did you go to Times Square?",
        ),
        rejection_axis="locale_violation",
        rejection_note="Rejected reply uses an American landmark instead of an Iranian one.",
    )


@pytest.fixture
def evaluation_output() -> EvaluationOutput:
    return EvaluationOutput(
        overall_cefr_estimate="A2",
        scores=ScoreBreakdown(fluency=3, accuracy=2, vocabulary=3, interaction=3, topic_adherence=4),
        specific_feedback=[
            TurnFeedback(
                turn_index=0,
                user_text="I goed to Isfahan last weekend.",
                issue="Incorrect past tense of 'go'.",
                correction="I went to Isfahan last weekend.",
                level="A2",
                severity="minor",
            ),
        ],
        strengths=["Clear word order", "Topic is appropriate"],
        suggested_practice="Review irregular past-tense verbs (go/went, eat/ate, see/saw).",
    )


@pytest.fixture
def eval_example(evaluation_output: EvaluationOutput) -> EvaluationExample:
    assistant_content = (
        "<think>The learner used 'goed' instead of 'went'. Otherwise the turn "
        "is coherent and A2-appropriate.</think>\n"
        + evaluation_output.model_dump_json()
    )
    return EvaluationExample(
        id="eval-001",
        metadata=EvaluationMetadata(
            source_dialogue_id="sft-001",
            learner_cefr_target="A2",
        ),
        system_prompt="You are an examiner estimating the learner's CEFR level.",
        messages=[
            Message(
                role="user",
                content="Transcript:\nLearner: I goed to Isfahan last weekend.",
            ),
            Message(role="assistant", content=assistant_content),
        ],
    )


def test_sft_roundtrip(tmp_path, sft_example: SFTExample) -> None:
    path = tmp_path / "sft.jsonl"
    written = SFTExample.to_jsonl(path, [sft_example, sft_example])
    assert written == 2
    loaded = list(SFTExample.from_jsonl(path))
    assert loaded == [sft_example, sft_example]


def test_dpo_roundtrip(tmp_path, dpo_example: DPOExample) -> None:
    path = tmp_path / "dpo.jsonl"
    written = DPOExample.to_jsonl(path, [dpo_example])
    assert written == 1
    loaded = list(DPOExample.from_jsonl(path))
    assert loaded == [dpo_example]


def test_eval_roundtrip(tmp_path, eval_example: EvaluationExample) -> None:
    path = tmp_path / "eval.jsonl"
    EvaluationExample.to_jsonl(path, [eval_example])
    loaded = list(EvaluationExample.from_jsonl(path))
    assert loaded == [eval_example]


def test_evaluation_metadata_roundtrip_with_generation(tmp_path, evaluation_output: EvaluationOutput) -> None:
    metadata = EvaluationMetadata(
        source_dialogue_id="sft-001",
        learner_cefr_target="A2",
        locale="china",
        scenario_type="redirect",
        generation={"language_trigger": "speaks_l1"},
    )
    example = EvaluationExample(
        id="eval-002",
        metadata=metadata,
        system_prompt="You are an examiner estimating the learner's CEFR level.",
        messages=[
            Message(role="user", content="Transcript:\nLearner: 我喜欢学习英语。"),
            Message(role="assistant", content="<think>Reasoning...</think>..."),
        ],
    )
    path = tmp_path / "eval_generation.jsonl"
    EvaluationExample.to_jsonl(path, [example])
    loaded = list(EvaluationExample.from_jsonl(path))
    assert loaded == [example]


def test_evaluation_output_roundtrip(evaluation_output: EvaluationOutput) -> None:
    raw = evaluation_output.model_dump_json()
    rebuilt = EvaluationOutput.model_validate_json(raw)
    assert rebuilt == evaluation_output


def test_jsonl_skips_blank_lines(tmp_path, sft_example: SFTExample) -> None:
    path = tmp_path / "sft.jsonl"
    body = sft_example.model_dump_json()
    path.write_text(f"{body}\n\n   \n{body}\n", encoding="utf-8")
    loaded = list(SFTExample.from_jsonl(path))
    assert loaded == [sft_example, sft_example]


def test_extra_field_forbidden_on_user_role() -> None:
    with pytest.raises(ValidationError):
        UserRole.model_validate(
            {"name": "Maryam", "description": "An A2 learner.", "junk_field": 1}
        )


def test_extra_field_forbidden_on_sft_example(sft_example: SFTExample) -> None:
    payload = sft_example.model_dump()
    payload["surprise_field"] = "boom"
    with pytest.raises(ValidationError):
        SFTExample.model_validate(payload)


def test_scenario_seed_requires_min_subtopics(
    user_role: UserRole, model_role: ModelRole
) -> None:
    with pytest.raises(ValidationError):
        ScenarioSeed(
            topic="travel",
            subtopics=["bus", "ticket"],
            user_role=user_role,
            model_role=model_role,
            setting="A bus station in Tabriz at dawn.",
            cefr_level="B1",
        )


def test_scenario_seed_caps_max_subtopics(
    user_role: UserRole, model_role: ModelRole
) -> None:
    with pytest.raises(ValidationError):
        ScenarioSeed(
            topic="travel",
            subtopics=["a", "b", "c", "d", "e", "f"],
            user_role=user_role,
            model_role=model_role,
            setting="A bus station in Tabriz at dawn.",
            cefr_level="B1",
        )


def test_scenario_seed_valid(user_role: UserRole, model_role: ModelRole) -> None:
    seed = ScenarioSeed(
        topic="travel",
        subtopics=["bus", "ticket", "departure"],
        user_role=user_role,
        model_role=model_role,
        setting="A bus station in Tabriz at dawn.",
        cefr_level="B1",
    )
    assert seed.cefr_level == "B1"
    assert len(seed.subtopics) == 3


def test_score_breakdown_range_enforced() -> None:
    with pytest.raises(ValidationError):
        ScoreBreakdown(fluency=6, accuracy=3, vocabulary=3, interaction=3, topic_adherence=3)
    with pytest.raises(ValidationError):
        ScoreBreakdown(fluency=0, accuracy=3, vocabulary=3, interaction=3, topic_adherence=3)
    with pytest.raises(ValidationError):
        ScoreBreakdown(fluency=3, accuracy=3, vocabulary=3, interaction=3, topic_adherence=6)
    with pytest.raises(ValidationError):
        ScoreBreakdown(fluency=3, accuracy=3, vocabulary=3, interaction=3, topic_adherence=0)


def test_invalid_rejection_axis_rejected(dpo_example: DPOExample) -> None:
    payload = dpo_example.model_dump()
    payload["rejection_axis"] = "not_a_real_axis"
    with pytest.raises(ValidationError):
        DPOExample.model_validate(payload)


def test_invalid_cefr_level_rejected(
    user_role: UserRole, model_role: ModelRole
) -> None:
    with pytest.raises(ValidationError):
        ScenarioSeed(
            topic="travel",
            subtopics=["bus", "ticket", "departure"],
            user_role=user_role,
            model_role=model_role,
            setting="A bus station in Tabriz at dawn.",
            cefr_level="D1",  # type: ignore[arg-type]
        )
