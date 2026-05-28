"""Pydantic v2 schemas for every artifact in the qwen-en-tutor data pipeline.

All models set ``extra="forbid"`` so a stray field — usually a typo in a config
or a drift between the generation and filter sides of the pipeline — raises a
clear ``ValidationError`` instead of silently passing through.

The Example types (SFTExample, DPOExample, EvaluationExample) carry
``from_jsonl`` / ``to_jsonl`` helpers via ``JSONLMixin`` for streaming reads
and writes against the on-disk corpora under ``data/``.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

CEFRLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
ScenarioType = Literal["normal", "redirect", "unleveled_natural"]
MessageRole = Literal["system", "user", "assistant"]
RejectionAxis = Literal[
    "register_unnatural",
    "cefr_mismatch",
    "locale_violation",
    "pedagogy_weak",
    "accuracy_error",
    "off_topic",
    "language_violation",
    "persona_break",
]
SeverityLevel = Literal["minor", "moderate", "major"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JSONLMixin:
    """Streaming JSONL read/write helpers for Example types."""

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Iterator[Self]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                yield cls.model_validate_json(line)  # type: ignore[attr-defined]

    @classmethod
    def to_jsonl(cls, path: str | Path, examples: Iterable[Self]) -> int:
        path = Path(path)
        count = 0
        with path.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(ex.model_dump_json())
                fh.write("\n")
                count += 1
        return count


class UserRole(_Strict):
    name: str
    description: str


class ModelRole(_Strict):
    name: str
    description: str


class ScenarioSeed(_Strict):
    # On-disk seeds carry an ``id`` string; declare it so strict-mode loading
    # doesn't reject existing data.
    id: str | None = None
    topic: str
    subtopics: list[str] = Field(min_length=3, max_length=5)
    user_role: UserRole
    model_role: ModelRole
    setting: str
    cefr_level: CEFRLevel
    # Which locale (config/locale.yaml entry) this seed is grounded in.
    # Older single-locale data on disk has no field; defaults to "china".
    locale: str = "china"


class Message(_Strict):
    role: MessageRole
    content: str


class ExampleMetadata(_Strict):
    """Shared scenario-level metadata for SFT and DPO examples."""

    topic: str
    subtopics: list[str] = Field(min_length=3, max_length=5)
    user_role: UserRole
    model_role: ModelRole
    cefr_level: CEFRLevel
    scenario_type: ScenarioType
    # Which locale (config/locale.yaml entry) this example is grounded in.
    # Older single-locale data on disk has no field; defaults to "china".
    locale: str = "china"
    generation: dict[str, Any] | None = None


class SFTExample(JSONLMixin, _Strict):
    id: str
    metadata: ExampleMetadata
    system_prompt: str
    messages: list[Message]
    quality_signals: dict[str, Any] | None = None


class DPOExample(JSONLMixin, _Strict):
    id: str
    metadata: ExampleMetadata
    system_prompt: str
    prompt_messages: list[Message]
    chosen: Message
    rejected: Message
    rejection_axis: RejectionAxis
    rejection_note: str


class ScoreBreakdown(_Strict):
    fluency: int = Field(ge=1, le=5)
    accuracy: int = Field(ge=1, le=5)
    vocabulary: int = Field(ge=1, le=5)
    interaction: int = Field(ge=1, le=5)


class TurnFeedback(_Strict):
    turn_index: int = Field(ge=0)
    user_text: str
    issue: str
    correction: str
    level: CEFRLevel
    severity: SeverityLevel


class EvaluationOutput(_Strict):
    """The structured JSON the assistant emits inside an EvaluationExample turn.

    In the on-disk EvaluationExample, this object is serialized to JSON and
    concatenated after a ``<think>...</think>`` block inside the assistant
    message's ``content`` string.
    """

    overall_cefr_estimate: CEFRLevel
    scores: ScoreBreakdown
    specific_feedback: list[TurnFeedback]
    strengths: list[str]
    suggested_practice: str


class EvaluationMetadata(_Strict):
    source_dialogue_id: str
    learner_cefr_target: CEFRLevel
    # Which locale (config/locale.yaml entry) this evaluation is grounded in.
    # Older single-locale data on disk has no field; defaults to "china".
    locale: str = "china"


class EvaluationExample(JSONLMixin, _Strict):
    id: str
    metadata: EvaluationMetadata
    system_prompt: str
    messages: list[Message]
