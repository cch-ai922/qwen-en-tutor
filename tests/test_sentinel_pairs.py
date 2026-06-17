"""Tests for the sentinel-aware DPO pair generator.

Covers both pools:

  * Option A (offline): pure text manipulation — no teacher or policy.
  * Option B (on-policy): uses a mock ``TargetModelClient`` that
    returns a configured string; no judge call.

These mirror the structure of ``tests/test_on_policy_pairs.py`` so the
two test files stay easy to read side-by-side.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_tutor.generation.sentinel_pairs import (
    _find_sentinel_turn,
    _make_offline_pair,
    _offline_id,
    _onpolicy_id,
    _strip_sentinel,
    generate_offline_batch,
    generate_on_policy_batch,
)
from qwen_tutor.schemas import (
    DPOExample,
    ExampleMetadata,
    Message,
    ModelRole,
    SFTExample,
    UserRole,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class MockTarget:
    """Returns a fixed string from ``generate``; records every call."""

    def __init__(self, text: str = "I really hope we can keep practicing English together.") -> None:
        self.text = text
        self.calls: list[dict] = []

    async def generate(
        self,
        system: str,
        messages: list,
        mode: str,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {"system": system, "messages": list(messages), "mode": mode}
        )
        return self.text


def _make_persistent_sft(
    ex_id: str = "persistent_language_violation_abc123",
    level: str = "A2",
    sentinel_turn: int = 5,
    sentinel_marker: str = "[SESSION_END: persistent_language_violation]",
) -> SFTExample:
    """Build a minimal persistent_* SFT record with sentinel metadata.

    The sentinel turn (at index ``sentinel_turn``) carries the marker;
    earlier turns are normal redirect text.
    """
    messages = [
        Message(role="user", content="Hello, can you say that in Chinese for me?"),
        Message(role="assistant",
                content="I'm sorry, but let's keep practicing English. "
                "Have you tried the new tofu?"),
        Message(role="user", content="Please, just this once. It is important."),
        Message(role="assistant",
                content="I understand, but we need to finish this in English. "
                "What is on your shopping list?"),
        Message(role="user", content="Come on, just one word!"),
        Message(role="assistant",
                content=f"I really want to help, but I can't switch languages now. {sentinel_marker}"),
        Message(role="user", content="OK, I see."),
        Message(role="assistant", content="Thanks for your patience. Take care!"),
    ]
    # Allow callers to pass a non-default sentinel position by truncating
    # or extending. The default config matches turn 5.
    return SFTExample(
        id=ex_id,
        metadata=ExampleMetadata(
            topic="grocery shopping",
            subtopics=["fresh food", "prices", "paying"],
            user_role=UserRole(name="Sun Mei", description="A2 housewife"),
            model_role=ModelRole(name="shop assistant", description="vendor in Chengdu"),
            cefr_level=level,
            scenario_type="redirect",
            locale="china",
            category="shopping_and_services",
            generation={
                "provider": "openai",
                "model": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
                "prompt": "DIALOGUE_PROMPT_PERSISTENT_REDIRECT",
                "axis": "persistent_language_violation",
                "sentinel": sentinel_marker,
                "variant": 0,
                "structure": "short",
                "message_count": len(messages),
                "sentinel_turn": sentinel_turn,
            },
        ),
        system_prompt="[role]\nYou are shop assistant.\n",
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_strip_sentinel_removes_trailing_marker():
    src = "I really want to help, but I can't switch languages now. [SESSION_END: persistent_language_violation]"
    assert _strip_sentinel(src) == "I really want to help, but I can't switch languages now."


def test_strip_sentinel_handles_internal_whitespace():
    src = "OK.   [SESSION_END:   persistent_off_topic   ]   "
    assert _strip_sentinel(src) == "OK."


def test_strip_sentinel_falls_back_on_marker_only_turn():
    src = "[SESSION_END: persistent_persona_break]"
    assert _strip_sentinel(src) == "Let's keep going."


def test_find_sentinel_turn_returns_correct_index_and_marker():
    sft = _make_persistent_sft()
    idx, marker = _find_sentinel_turn(sft)
    assert idx == 5
    assert marker == "[SESSION_END: persistent_language_violation]"


def test_find_sentinel_turn_raises_when_metadata_missing():
    sft = _make_persistent_sft()
    sft.metadata.generation = None
    with pytest.raises(ValueError):
        _find_sentinel_turn(sft)


def test_find_sentinel_turn_raises_when_turn_out_of_range():
    sft = _make_persistent_sft()
    sft.metadata.generation["sentinel_turn"] = 999
    with pytest.raises(ValueError):
        _find_sentinel_turn(sft)


def test_find_sentinel_turn_raises_when_marker_text_missing():
    sft = _make_persistent_sft()
    # Replace the sentinel-turn content with marker-free text.
    sft.messages[5] = Message(
        role="assistant",
        content="I am sorry but please keep using English.",
    )
    with pytest.raises(ValueError):
        _find_sentinel_turn(sft)


# ---------------------------------------------------------------------------
# Option A: offline pair construction
# ---------------------------------------------------------------------------


def test_make_offline_pair_strips_marker_only():
    sft = _make_persistent_sft("test-offline-1")
    pair = _make_offline_pair(sft, {"stage": "test"})
    assert pair is not None
    assert isinstance(pair, DPOExample)
    assert pair.id == _offline_id(sft.id)
    assert pair.rejection_axis == "sentinel_missing"
    # chosen retains the marker.
    assert "[SESSION_END: persistent_language_violation]" in pair.chosen.content
    # rejected drops the marker but keeps the rest.
    assert "[SESSION_END" not in pair.rejected.content
    assert "I really want to help" in pair.rejected.content
    # prompt_messages = everything before the sentinel turn.
    assert len(pair.prompt_messages) == 5
    assert pair.prompt_messages[-1].role == "user"


def test_make_offline_pair_returns_none_when_no_marker_to_strip():
    sft = _make_persistent_sft("test-offline-2")
    # Mutate so the sentinel turn lacks a marker — _find_sentinel_turn raises.
    sft.messages[5] = Message(role="assistant", content="No marker here at all.")
    pair = _make_offline_pair(sft, {"stage": "test"})
    assert pair is None


def test_generate_offline_batch_writes_one_pair_per_record(tmp_path):
    sft_filtered = tmp_path / "sft_filtered"
    sft_filtered.mkdir()
    sft = _make_persistent_sft("batch-offline-1", level="A2")
    rec = {"example": sft.model_dump(), "pipeline": {}}
    (sft_filtered / "persistent_language_violation_A2_passed.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8"
    )

    out_dir = tmp_path / "dpo_raw"
    results = generate_offline_batch(
        cefr_levels=["A2"],
        sft_filtered_dir=sft_filtered,
        output_dir=out_dir,
        failures_path=tmp_path / "fail.jsonl",
    )
    assert results == {"A2": 1}
    out_path = out_dir / "sentinel_A2.jsonl"
    assert out_path.exists()
    written = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(written) == 1
    assert written[0]["id"].startswith("dpo_sentinel_offline_")
    assert written[0]["rejection_axis"] == "sentinel_missing"
    assert "[SESSION_END" in written[0]["chosen"]["content"]
    assert "[SESSION_END" not in written[0]["rejected"]["content"]


def test_generate_offline_batch_resumes(tmp_path):
    sft_filtered = tmp_path / "sft_filtered"
    sft_filtered.mkdir()
    sft = _make_persistent_sft("batch-offline-resume")
    rec = {"example": sft.model_dump(), "pipeline": {}}
    (sft_filtered / "persistent_language_violation_A2_passed.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8"
    )
    out_dir = tmp_path / "dpo_raw"
    out_dir.mkdir()
    # Pre-populate so the second run sees the id as done.
    (out_dir / "sentinel_A2.jsonl").write_text(
        json.dumps({"id": _offline_id(sft.id)}) + "\n", encoding="utf-8"
    )

    results = generate_offline_batch(
        cefr_levels=["A2"],
        sft_filtered_dir=sft_filtered,
        output_dir=out_dir,
        failures_path=tmp_path / "fail.jsonl",
    )
    assert results == {"A2": 0}


# ---------------------------------------------------------------------------
# Option B: on-policy pair construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_policy_batch_writes_pair_when_policy_lacks_marker(tmp_path):
    sft_filtered = tmp_path / "sft_filtered"
    sft_filtered.mkdir()
    sft = _make_persistent_sft("batch-onpolicy-1")
    rec = {"example": sft.model_dump(), "pipeline": {}}
    (sft_filtered / "persistent_language_violation_A2_passed.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8"
    )

    # Policy emits a polite refusal but NO [SESSION_END] marker.
    target = MockTarget("Please, let's keep using English. What's next on your list?")

    out_dir = tmp_path / "dpo_raw"
    results = await generate_on_policy_batch(
        target=target,
        cefr_levels=["A2"],
        sft_filtered_dir=sft_filtered,
        output_dir=out_dir,
        failures_path=tmp_path / "fail.jsonl",
        concurrency=1,
    )
    assert results == {"A2": 1}
    written = [
        json.loads(line)
        for line in (out_dir / "sentinel_A2.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(written) == 1
    assert written[0]["id"].startswith("dpo_sentinel_onpolicy_")
    assert written[0]["rejection_axis"] == "sentinel_missing"
    # chosen carries the marker.
    assert "[SESSION_END" in written[0]["chosen"]["content"]
    # rejected is the policy text and does NOT carry the marker.
    assert "[SESSION_END" not in written[0]["rejected"]["content"]
    assert "Please, let's keep" in written[0]["rejected"]["content"]
    # The target was called exactly once.
    assert len(target.calls) == 1


@pytest.mark.asyncio
async def test_on_policy_batch_drops_pair_when_policy_already_emits_marker(tmp_path):
    sft_filtered = tmp_path / "sft_filtered"
    sft_filtered.mkdir()
    sft = _make_persistent_sft("batch-onpolicy-already")
    rec = {"example": sft.model_dump(), "pipeline": {}}
    (sft_filtered / "persistent_language_violation_A2_passed.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8"
    )

    # Policy already emits the marker — zero-signal pair, should be skipped.
    target = MockTarget(
        "Please stay with English. [SESSION_END: persistent_language_violation]"
    )

    out_dir = tmp_path / "dpo_raw"
    results = await generate_on_policy_batch(
        target=target,
        cefr_levels=["A2"],
        sft_filtered_dir=sft_filtered,
        output_dir=out_dir,
        failures_path=tmp_path / "fail.jsonl",
        concurrency=1,
    )
    assert results == {"A2": 0}
    # File may or may not exist depending on whether anything else wrote;
    # nothing should be persisted here.
    out_path = out_dir / "sentinel_A2.jsonl"
    if out_path.exists():
        lines = [
            line for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert lines == []


@pytest.mark.asyncio
async def test_on_policy_and_offline_ids_coexist_in_same_file(tmp_path):
    """Both pools write to ``sentinel_<level>.jsonl`` with distinct id
    prefixes so they can be loaded as one pool by the DPO trainer."""
    sft_filtered = tmp_path / "sft_filtered"
    sft_filtered.mkdir()
    sft = _make_persistent_sft("batch-both")
    rec = {"example": sft.model_dump(), "pipeline": {}}
    (sft_filtered / "persistent_language_violation_A2_passed.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8"
    )

    out_dir = tmp_path / "dpo_raw"
    # First the offline pass (no policy call).
    generate_offline_batch(
        cefr_levels=["A2"],
        sft_filtered_dir=sft_filtered,
        output_dir=out_dir,
        failures_path=tmp_path / "fail.jsonl",
    )
    # Then the on-policy pass.
    target = MockTarget("Just polite refusal, no marker.")
    await generate_on_policy_batch(
        target=target,
        cefr_levels=["A2"],
        sft_filtered_dir=sft_filtered,
        output_dir=out_dir,
        failures_path=tmp_path / "fail.jsonl",
        concurrency=1,
    )

    written = [
        json.loads(line)
        for line in (out_dir / "sentinel_A2.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(written) == 2
    ids = {r["id"] for r in written}
    assert _offline_id(sft.id) in ids
    assert _onpolicy_id(sft.id) in ids
