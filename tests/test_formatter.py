"""Tests for qwen_tutor.training.formatter.ChatFormatter.

Covers:
  * /no_think prepending on conversation-mode user turns,
  * deployment-system-prompt rendering (overrides the example's stored
    ``system_prompt``),
  * evaluation-mode preservation of the pre-built <think>...</think>
    block,
  * assistant-only loss-mask construction,
  * end-to-end round-trip (encode → decode) with a Qwen3-shaped mock
    tokenizer (always runs) and with the real Qwen3 tokenizer when it's
    available locally (skipped otherwise).
"""

from __future__ import annotations

from typing import Any

import pytest

from qwen_tutor.schemas import (
    EvaluationExample,
    EvaluationMetadata,
    ExampleMetadata,
    Message,
    ModelRole,
    SFTExample,
    UserRole,
)
from qwen_tutor.training.formatter import (
    LOSS_IGNORE_INDEX,
    NO_THINK_TAG,
    ChatFormatter,
    EXPECTED_SPECIAL_TOKENS,
)


# ---------------------------------------------------------------------------
# Mock Qwen3-shaped tokenizer
# ---------------------------------------------------------------------------


class MockQwen3Tokenizer:
    """Minimal Qwen3-shaped tokenizer for offline formatter testing.

    Renders messages as ChatML strings and encodes character-by-character
    using Unicode code points. That keeps tokenization deterministic,
    prefix-stable across message counts, and round-trippable
    (``decode(encode(text)) == text``).
    """

    def __init__(self) -> None:
        self.additional_special_tokens = list(EXPECTED_SPECIAL_TOKENS)
        self.all_special_tokens = list(EXPECTED_SPECIAL_TOKENS) + ["<|endoftext|>"]
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|im_end|>"

    def get_vocab(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.all_special_tokens)}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
        **_: Any,
    ):
        parts: list[str] = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        if not tokenize:
            return text
        return [ord(c) for c in text]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deployment_template() -> str:
    return (
        "You are a patient English conversation tutor for Iranian adult "
        "learners. The learner is at CEFR level {cefr_level}.\n"
    )


@pytest.fixture
def evaluation_prompt() -> str:
    return "You are an English examiner. Produce <think>...</think> then JSON.\n"


@pytest.fixture
def mock_tokenizer() -> MockQwen3Tokenizer:
    return MockQwen3Tokenizer()


@pytest.fixture
def formatter(
    mock_tokenizer: MockQwen3Tokenizer,
    deployment_template: str,
    evaluation_prompt: str,
) -> ChatFormatter:
    return ChatFormatter(
        tokenizer=mock_tokenizer,
        deployment_system_prompt_template=deployment_template,
        evaluation_system_prompt=evaluation_prompt,
        max_seq_length=4096,
    )


@pytest.fixture
def sft_example() -> SFTExample:
    return SFTExample(
        id="sft-001",
        metadata=ExampleMetadata(
            topic="weekend plans",
            subtopics=["family", "city", "food"],
            user_role=UserRole(name="Maryam", description="A2 learner"),
            model_role=ModelRole(name="Tutor", description="patient tutor"),
            cefr_level="A2",
            scenario_type="normal",
        ),
        system_prompt="STORED PROMPT (should be ignored)",
        messages=[
            Message(role="user", content="How was your weekend?"),
            Message(role="assistant", content="It was great. I went to Isfahan."),
            Message(role="user", content="Did you take photos?"),
            Message(role="assistant", content="Yes, near Naqsh-e Jahan Square."),
        ],
    )


@pytest.fixture
def evaluation_example() -> EvaluationExample:
    return EvaluationExample(
        id="eval-001",
        metadata=EvaluationMetadata(
            source_dialogue_id="sft-001", learner_cefr_target="A2"
        ),
        system_prompt="STORED EVAL PROMPT (should be ignored)",
        messages=[
            Message(
                role="user",
                content="Transcript:\n[USER] I goed to Isfahan last weekend.",
            ),
            Message(
                role="assistant",
                content=(
                    "<think>Learner used 'goed' instead of 'went'.</think>\n"
                    '{"overall_cefr_estimate": "A2", "scores": {"fluency": 3, '
                    '"accuracy": 2, "vocabulary": 3, "interaction": 3}, '
                    '"specific_feedback": [], "strengths": [], '
                    '"suggested_practice": "drill irregular past forms"}'
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# format_sft_example: structural checks
# ---------------------------------------------------------------------------


def test_sft_uses_deployment_template_not_stored_system_prompt(
    formatter: ChatFormatter, sft_example: SFTExample
) -> None:
    chat = formatter.format_sft_example(sft_example)
    assert chat.messages[0]["role"] == "system"
    assert "CEFR level A2" in chat.messages[0]["content"]
    assert "STORED PROMPT" not in chat.messages[0]["content"]


def test_sft_prepends_no_think_to_first_user_turn(
    formatter: ChatFormatter, sft_example: SFTExample
) -> None:
    chat = formatter.format_sft_example(sft_example)
    user_messages = [m for m in chat.messages if m["role"] == "user"]
    assert user_messages[0]["content"].startswith(f"{NO_THINK_TAG}\n")
    # Subsequent user turns are not tagged.
    assert not user_messages[1]["content"].startswith(NO_THINK_TAG)


def test_sft_does_not_double_tag_no_think(formatter: ChatFormatter) -> None:
    pre_tagged = SFTExample(
        id="sft-002",
        metadata=ExampleMetadata(
            topic="t",
            subtopics=["a", "b", "c"],
            user_role=UserRole(name="X", description="x"),
            model_role=ModelRole(name="Y", description="y"),
            cefr_level="A1",
            scenario_type="normal",
        ),
        system_prompt="s",
        messages=[
            Message(role="user", content="/no_think\nHello"),
            Message(role="assistant", content="Hi there."),
        ],
    )
    chat = formatter.format_sft_example(pre_tagged)
    user_msg = next(m for m in chat.messages if m["role"] == "user")
    assert user_msg["content"].count(NO_THINK_TAG) == 1


def test_sft_enable_thinking_false(
    formatter: ChatFormatter, sft_example: SFTExample
) -> None:
    chat = formatter.format_sft_example(sft_example)
    assert chat.enable_thinking is False
    assert chat.example_type == "sft"


def test_sft_rejects_wrong_scenario_type(formatter: ChatFormatter) -> None:
    bad = SFTExample(
        id="sft-003",
        metadata=ExampleMetadata(
            topic="t",
            subtopics=["a", "b", "c"],
            user_role=UserRole(name="X", description="x"),
            model_role=ModelRole(name="Y", description="y"),
            cefr_level="A1",
            scenario_type="redirect",   # valid here
        ),
        system_prompt="s",
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hi"),
        ],
    )
    # Patch in an invalid scenario_type via model_copy:
    object.__setattr__(bad.metadata, "scenario_type", "not_a_real_type")
    with pytest.raises(ValueError, match="scenario_type"):
        formatter.format_sft_example(bad)


# ---------------------------------------------------------------------------
# format_evaluation_example: structural checks
# ---------------------------------------------------------------------------


def test_eval_uses_evaluation_system_prompt(
    formatter: ChatFormatter, evaluation_example: EvaluationExample
) -> None:
    chat = formatter.format_evaluation_example(evaluation_example)
    assert chat.messages[0]["role"] == "system"
    assert "examiner" in chat.messages[0]["content"].lower()
    assert "STORED EVAL PROMPT" not in chat.messages[0]["content"]


def test_eval_preserves_think_block_without_duplication(
    formatter: ChatFormatter, evaluation_example: EvaluationExample
) -> None:
    chat = formatter.format_evaluation_example(evaluation_example)
    assistant = next(m for m in chat.messages if m["role"] == "assistant")
    assert assistant["content"].count("<think>") == 1
    assert assistant["content"].count("</think>") == 1


def test_eval_enable_thinking_true(
    formatter: ChatFormatter, evaluation_example: EvaluationExample
) -> None:
    chat = formatter.format_evaluation_example(evaluation_example)
    assert chat.enable_thinking is True
    assert chat.example_type == "evaluation"


# ---------------------------------------------------------------------------
# tokenize: loss mask + round-trip
# ---------------------------------------------------------------------------


def test_tokenize_round_trip_decodes_to_template_text(
    formatter: ChatFormatter, sft_example: SFTExample
) -> None:
    tok = formatter.format_for_training(sft_example)
    # decode(input_ids) should equal the .text the formatter cached
    decoded = formatter.tokenizer.decode(tok.input_ids)
    assert decoded == tok.text


def test_tokenize_loss_mask_only_covers_assistant_turns(
    formatter: ChatFormatter, sft_example: SFTExample
) -> None:
    tok = formatter.format_for_training(sft_example)
    # Rebuild the assistant-only sequence from positions whose label != -100.
    kept_ids = [
        tok.input_ids[i] for i, lbl in enumerate(tok.labels) if lbl != LOSS_IGNORE_INDEX
    ]
    kept_text = formatter.tokenizer.decode(kept_ids)
    # Every assistant turn body should appear; user turn bodies should not.
    assert "It was great. I went to Isfahan." in kept_text
    assert "Yes, near Naqsh-e Jahan Square." in kept_text
    assert "How was your weekend?" not in kept_text
    assert "Did you take photos?" not in kept_text
    # And there should be at least one assistant role marker in there.
    assert "<|im_start|>assistant" in kept_text


def test_tokenize_eval_round_trip_and_mask(
    formatter: ChatFormatter, evaluation_example: EvaluationExample
) -> None:
    tok = formatter.format_for_training(evaluation_example)
    # Round trip.
    assert formatter.tokenizer.decode(tok.input_ids) == tok.text
    # The <think> block + JSON must survive in the assistant span.
    kept_ids = [
        tok.input_ids[i] for i, lbl in enumerate(tok.labels) if lbl != LOSS_IGNORE_INDEX
    ]
    kept = formatter.tokenizer.decode(kept_ids)
    assert "<think>" in kept and "</think>" in kept
    assert '"overall_cefr_estimate": "A2"' in kept
    # User-side transcript must NOT be in the assistant-only mask.
    assert "Transcript" not in kept


def test_tokenize_labels_and_input_ids_have_equal_length(
    formatter: ChatFormatter, sft_example: SFTExample
) -> None:
    tok = formatter.format_for_training(sft_example)
    assert len(tok.input_ids) == len(tok.labels) == len(tok.attention_mask)


def test_tokenize_attention_mask_is_all_ones(
    formatter: ChatFormatter, sft_example: SFTExample
) -> None:
    tok = formatter.format_for_training(sft_example)
    assert all(v == 1 for v in tok.attention_mask)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_template_missing_cefr_placeholder_raises(
    mock_tokenizer: MockQwen3Tokenizer, evaluation_prompt: str
) -> None:
    with pytest.raises(ValueError, match="{cefr_level}"):
        ChatFormatter(
            tokenizer=mock_tokenizer,
            deployment_system_prompt_template="missing placeholder",
            evaluation_system_prompt=evaluation_prompt,
        )


def test_validate_special_tokens_missing_raises(
    deployment_template: str, evaluation_prompt: str
) -> None:
    class BareTokenizer:
        additional_special_tokens: list[str] = []
        all_special_tokens: list[str] = []

        def get_vocab(self) -> dict[str, int]:
            return {}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(c) for c in text]

    with pytest.raises(ValueError, match="missing expected special tokens"):
        ChatFormatter(
            tokenizer=BareTokenizer(),
            deployment_system_prompt_template=deployment_template,
            evaluation_system_prompt=evaluation_prompt,
        )


# ---------------------------------------------------------------------------
# Real Qwen3 tokenizer round-trip (skipped if not installed locally)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_qwen3_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B", trust_remote_code=False
        )
    except Exception as exc:  # network / cache miss / model not available
        pytest.skip(f"Qwen3 tokenizer unavailable: {exc}")


@pytest.mark.slow
def test_real_qwen3_round_trip(
    real_qwen3_tokenizer,
    deployment_template: str,
    evaluation_prompt: str,
    sft_example: SFTExample,
    evaluation_example: EvaluationExample,
) -> None:
    fmt = ChatFormatter(
        tokenizer=real_qwen3_tokenizer,
        deployment_system_prompt_template=deployment_template,
        evaluation_system_prompt=evaluation_prompt,
    )
    sft_tok = fmt.format_for_training(sft_example)
    re_encoded = real_qwen3_tokenizer.apply_chat_template(
        fmt.format_sft_example(sft_example).messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    assert list(re_encoded)[: len(sft_tok.input_ids)] == sft_tok.input_ids

    eval_tok = fmt.format_for_training(evaluation_example)
    # The eval assistant content must still contain <think>...</think> after
    # encode → decode.
    decoded = real_qwen3_tokenizer.decode(eval_tok.input_ids, skip_special_tokens=False)
    assert "<think>" in decoded and "</think>" in decoded
    assert '"overall_cefr_estimate"' in decoded
