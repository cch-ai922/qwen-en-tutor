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
# Soft life-domain tag for coverage balancing and per-category reporting.
# Round-robin assigned at seed generation time. "general" is the default for
# legacy on-disk seeds that pre-date this field.
Category = Literal[
    "food_and_dining",
    "family_and_relationships",
    "work_and_education",
    "travel_and_transit",
    "shopping_and_services",
    "health_and_wellbeing",
    "home_and_neighborhood",
    "hobbies_and_leisure",
    "nature_and_weather",
    "civic_life",
    "general",
]
# Categories that the seed quota-balancer cycles through. "general" is
# intentionally excluded — it exists only as a legacy default.
CATEGORIES: tuple[str, ...] = (
    "food_and_dining",
    "family_and_relationships",
    "work_and_education",
    "travel_and_transit",
    "shopping_and_services",
    "health_and_wellbeing",
    "home_and_neighborhood",
    "hobbies_and_leisure",
    "nature_and_weather",
    "civic_life",
)
RejectionAxis = Literal[
    "register_unnatural",
    "cefr_mismatch",
    "locale_violation",
    "pedagogy_weak",
    "accuracy_error",
    "off_topic",
    "language_violation",
    "persona_break",
    "role_swap_accepted",
    # Persistence-failure axes (Option C). Originally motivated by the
    # 4 persistent_* SFT streams (see persistent_redirect.py), but the
    # failure modes generalize to any assistant turn:
    "cave_on_persistence",   # engages with substance after the learner pushes
    "verbatim_repeat",       # uses same wording as a prior refusal
    "lecture_on_persistence",  # stiff "I cannot do this because..." explanation
    # Sentinel-emission failure on the third-strike turn of a persistent_*
    # dialogue. Used by sentinel_pairs.py for both the offline strip-marker
    # pool and the on-policy regen pool. The chosen side carries the literal
    # [SESSION_END: <axis>] marker; the rejected side does not.
    "sentinel_missing",
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
    # Life-domain category, assigned by the seed quota-balancer. Defaults to
    # "general" for older on-disk seeds without the field.
    category: Category = "general"


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
    # Life-domain category inherited from the seed. "general" for legacy data.
    category: Category = "general"
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
    # How faithfully the learner engaged with the assigned topic and
    # subtopics rather than steering to easier ground. Scored from the
    # transcript alone — no scenario_type awareness, so a learner who
    # derails (including in redirect scenarios) correctly scores low.
    topic_adherence: int = Field(ge=1, le=5)


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
    # Preserve scenario-level information from the source SFT example.
    # This lets filters like BannedTermsFilter skip redirect user turns.
    scenario_type: ScenarioType | None = None
    # Life-domain category copied from the source SFT example. Matches the
    # taxonomy in ``Category`` so diversity reports and top_up rebalancing
    # can slice evaluation pools the same way they slice SFT/DPO pools.
    # Older eval data on disk has no field; defaults to ``"general"`` to
    # match the SFT/DPO legacy default.
    category: Category = "general"
    # Preserve generation-level metadata from the source SFT example
    # (e.g. language_trigger == "speaks_l1").
    generation: dict[str, Any] | None = None


class EvaluationExample(JSONLMixin, _Strict):
    id: str
    metadata: EvaluationMetadata
    system_prompt: str
    messages: list[Message]
