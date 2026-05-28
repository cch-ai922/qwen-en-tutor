"""Tests for the on-policy DPO generator.

Uses a mock ``TargetModelClient`` (returns a configured string) and a
mock ``TeacherClient`` for the judge (returns a configured JSON
verdict). No network, no GPU, no real model weights involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_tutor.generation.on_policy_pairs import (
    MIN_JUDGE_MARGIN,
    _generate_one_pair,
    _onpolicy_id,
    _parse_judge_verdict,
    _pick_assistant_turn_index,
    _teacher_is_a,
    generate_batch,
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
# Mocks
# ---------------------------------------------------------------------------


class MockTarget:
    """Returns a fixed reply, records every call."""

    def __init__(self, text: str = "Default policy reply.") -> None:
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


class MockJudge:
    """Returns a pre-configured JSON-shaped string from ``generate``."""

    class _Cfg:
        provider = "mock"
        model = "mock-judge"

    config = _Cfg()

    def __init__(self, verdict: dict | str = None) -> None:
        self.verdict = verdict
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
        if isinstance(self.verdict, str):
            return self.verdict
        return json.dumps(self.verdict or {})


def _make_sft(ex_id: str = "sft-001", level: str = "A2") -> SFTExample:
    return SFTExample(
        id=ex_id,
        metadata=ExampleMetadata(
            topic="weekend plans",
            subtopics=["family", "food", "city"],
            user_role=UserRole(name="Maryam", description="A2 learner"),
            model_role=ModelRole(name="Tutor", description="patient tutor"),
            cefr_level=level,
            scenario_type="normal",
        ),
        system_prompt="...",
        messages=[
            Message(role="user", content="How was your weekend?"),
            Message(role="assistant", content="It was great. I went to Isfahan with my family."),
            Message(role="user", content="Did you eat something special?"),
            Message(role="assistant", content="Yes, we had beryani near Naqsh-e Jahan Square."),
            Message(role="user", content="Sounds wonderful."),
            Message(role="assistant", content="It was, really. Have you been to Isfahan before?"),
        ],
    )


DEPLOYMENT_TEMPLATE = (
    "You are a patient tutor for an Iranian learner at CEFR level {cefr_level}."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_pick_assistant_turn_skips_first_when_possible():
    sft = _make_sft()
    idx = _pick_assistant_turn_index(sft)
    assistant_indices = [i for i, m in enumerate(sft.messages) if m.role == "assistant"]
    # Should NOT be the first assistant turn (idx 1).
    assert idx in assistant_indices[1:]


def test_pick_assistant_turn_deterministic():
    sft = _make_sft("seed-xyz")
    assert _pick_assistant_turn_index(sft) == _pick_assistant_turn_index(sft)


def test_teacher_is_a_deterministic_and_balanced():
    a = sum(1 for i in range(200) if _teacher_is_a(f"id_{i}"))
    # Hash-based; expect rough balance
    assert 60 < a < 140, f"unexpectedly skewed: {a}/200"
    # Determinism
    assert _teacher_is_a("id_42") == _teacher_is_a("id_42")


def test_parse_judge_verdict_valid():
    raw = json.dumps(
        {"winner": "A", "margin": 2, "reason_axis": "locale_violation", "note": "Western place"}
    )
    v = _parse_judge_verdict(raw)
    assert v == {
        "winner": "A", "margin": 2, "reason_axis": "locale_violation",
        "note": "Western place",
    }


def test_parse_judge_verdict_bad_axis_falls_back():
    raw = json.dumps(
        {"winner": "B", "margin": 3, "reason_axis": "made_up_axis", "note": "x"}
    )
    v = _parse_judge_verdict(raw)
    assert v["reason_axis"] == "register_unnatural"  # safe fallback


def test_parse_judge_verdict_invalid_winner_returns_none():
    raw = json.dumps({"winner": "neither", "margin": 2})
    assert _parse_judge_verdict(raw) is None


def test_parse_judge_verdict_unparseable_returns_none():
    assert _parse_judge_verdict("not json at all") is None


# ---------------------------------------------------------------------------
# _generate_one_pair: outcomes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pair_kept_when_teacher_wins_clearly(tmp_path):
    sft = _make_sft("won-by-teacher")
    target = MockTarget("It is the case that beryani is consumed near a square.")
    teacher_is_a = _teacher_is_a(sft.id)
    teacher_winner = "A" if teacher_is_a else "B"
    judge = MockJudge(
        {"winner": teacher_winner, "margin": 3, "reason_axis": "register_unnatural",
         "note": "policy was stiff"}
    )
    failures = tmp_path / "fail.jsonl"
    audit = tmp_path / "audit.jsonl"

    pair = await _generate_one_pair(
        sft=sft, target=target, judge=judge,
        deployment_system_prompt_template=DEPLOYMENT_TEMPLATE,
        min_margin=MIN_JUDGE_MARGIN,
        target_max_tokens=64, target_temperature=0.7,
        judge_max_tokens=200, judge_temperature=0.0,
        failures_path=failures, audit_path=audit,
        generation_meta_base={"provider": "mock", "model": "mock-judge"},
    )
    assert pair is not None
    assert isinstance(pair, DPOExample)
    assert pair.id == _onpolicy_id(sft.id)
    assert pair.rejection_axis == "register_unnatural"
    # chosen = teacher's turn (= one of sft.messages)
    assert pair.chosen.content in [m.content for m in sft.messages if m.role == "assistant"]
    # rejected = policy text
    assert "It is the case that" in pair.rejected.content
    # audit row exists
    audit_rows = audit.read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_rows) == 1
    audit_obj = json.loads(audit_rows[0])
    assert audit_obj["teacher_won"] is True
    assert audit_obj["margin"] == 3


@pytest.mark.asyncio
async def test_pair_skipped_on_tie(tmp_path):
    sft = _make_sft("tied")
    target = MockTarget("Some policy reply.")
    judge = MockJudge(
        {"winner": "tie", "margin": 0, "reason_axis": "register_unnatural", "note": "even"}
    )
    failures = tmp_path / "fail.jsonl"
    audit = tmp_path / "audit.jsonl"

    pair = await _generate_one_pair(
        sft=sft, target=target, judge=judge,
        deployment_system_prompt_template=DEPLOYMENT_TEMPLATE,
        min_margin=MIN_JUDGE_MARGIN,
        target_max_tokens=64, target_temperature=0.7,
        judge_max_tokens=200, judge_temperature=0.0,
        failures_path=failures, audit_path=audit,
        generation_meta_base={"provider": "mock", "model": "mock-judge"},
    )
    assert pair is None
    # audit still recorded
    assert audit.exists()


@pytest.mark.asyncio
async def test_pair_skipped_when_policy_wins(tmp_path):
    sft = _make_sft("policy-won")
    target = MockTarget("An unusually good policy reply.")
    teacher_is_a = _teacher_is_a(sft.id)
    policy_winner = "B" if teacher_is_a else "A"
    judge = MockJudge(
        {"winner": policy_winner, "margin": 3, "reason_axis": "register_unnatural",
         "note": "policy was better"}
    )
    failures = tmp_path / "fail.jsonl"
    audit = tmp_path / "audit.jsonl"

    pair = await _generate_one_pair(
        sft=sft, target=target, judge=judge,
        deployment_system_prompt_template=DEPLOYMENT_TEMPLATE,
        min_margin=MIN_JUDGE_MARGIN,
        target_max_tokens=64, target_temperature=0.7,
        judge_max_tokens=200, judge_temperature=0.0,
        failures_path=failures, audit_path=audit,
        generation_meta_base={"provider": "mock", "model": "mock-judge"},
    )
    # Should be skipped — policy beat teacher.
    assert pair is None
    # Audit still recorded with teacher_won=False
    audit_obj = json.loads(audit.read_text(encoding="utf-8").strip())
    assert audit_obj["teacher_won"] is False


@pytest.mark.asyncio
async def test_pair_skipped_when_margin_below_threshold(tmp_path):
    sft = _make_sft("slight-edge")
    target = MockTarget("Slightly stiff policy reply.")
    teacher_is_a = _teacher_is_a(sft.id)
    teacher_winner = "A" if teacher_is_a else "B"
    judge = MockJudge(
        {"winner": teacher_winner, "margin": 1, "reason_axis": "register_unnatural",
         "note": "slight edge"}
    )
    failures = tmp_path / "fail.jsonl"
    audit = tmp_path / "audit.jsonl"

    pair = await _generate_one_pair(
        sft=sft, target=target, judge=judge,
        deployment_system_prompt_template=DEPLOYMENT_TEMPLATE,
        min_margin=2,
        target_max_tokens=64, target_temperature=0.7,
        judge_max_tokens=200, judge_temperature=0.0,
        failures_path=failures, audit_path=audit,
        generation_meta_base={"provider": "mock", "model": "mock-judge"},
    )
    assert pair is None


@pytest.mark.asyncio
async def test_judge_returns_unparseable_records_failure(tmp_path):
    sft = _make_sft("bad-verdict")
    target = MockTarget("Some policy reply.")
    judge = MockJudge("totally not json")
    failures = tmp_path / "fail.jsonl"
    audit = tmp_path / "audit.jsonl"

    pair = await _generate_one_pair(
        sft=sft, target=target, judge=judge,
        deployment_system_prompt_template=DEPLOYMENT_TEMPLATE,
        min_margin=MIN_JUDGE_MARGIN,
        target_max_tokens=64, target_temperature=0.7,
        judge_max_tokens=200, judge_temperature=0.0,
        failures_path=failures, audit_path=audit,
        generation_meta_base={"provider": "mock", "model": "mock-judge"},
    )
    assert pair is None
    # Failure recorded.
    fail_text = failures.read_text(encoding="utf-8")
    assert "unparseable" in fail_text


# ---------------------------------------------------------------------------
# generate_batch: end-to-end with file IO
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_batch_writes_pair_from_filtered_sft(tmp_path):
    # Build a minimal data/sft_filtered/ tree the loader recognizes.
    sft_filtered = tmp_path / "sft_filtered"
    sft_filtered.mkdir()
    sft = _make_sft("batch-001", level="A2")
    rec = {"example": sft.model_dump(), "pipeline": {}}
    (sft_filtered / "normal_A2_passed.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8"
    )

    target = MockTarget("It is the case that pomegranates cost lots.")
    teacher_is_a = _teacher_is_a(sft.id)
    winner = "A" if teacher_is_a else "B"
    judge = MockJudge(
        {"winner": winner, "margin": 3, "reason_axis": "register_unnatural",
         "note": "stiff"}
    )

    out_dir = tmp_path / "dpo_raw"
    failures = tmp_path / "fail.jsonl"
    audit = tmp_path / "audit.jsonl"
    results = await generate_batch(
        target=target,
        deployment_system_prompt_template=DEPLOYMENT_TEMPLATE,
        cefr_levels=["A2"],
        sft_filtered_dir=sft_filtered,
        output_dir=out_dir,
        failures_path=failures,
        audit_path=audit,
        judge=judge,
        min_margin=2,
        max_per_level=None,
        concurrency=2,
    )
    assert results == {"A2": 1}
    out_path = out_dir / "on_policy_A2.jsonl"
    assert out_path.exists()
    written = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(written) == 1
    assert written[0]["id"].startswith("dpo_onpolicy_")
    assert written[0]["rejection_axis"] == "register_unnatural"


@pytest.mark.asyncio
async def test_generate_batch_resumes_from_existing_output(tmp_path):
    sft_filtered = tmp_path / "sft_filtered"
    sft_filtered.mkdir()
    sft = _make_sft("batch-002", level="A2")
    rec = {"example": sft.model_dump(), "pipeline": {}}
    (sft_filtered / "normal_A2_passed.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8"
    )
    # Pre-populate the output with this id so the second run sees it as done.
    out_dir = tmp_path / "dpo_raw"
    out_dir.mkdir()
    out_path = out_dir / "on_policy_A2.jsonl"
    out_path.write_text(json.dumps({"id": _onpolicy_id(sft.id)}) + "\n", encoding="utf-8")

    target = MockTarget("anything")
    judge = MockJudge({"winner": "A", "margin": 3, "reason_axis": "register_unnatural", "note": "x"})
    results = await generate_batch(
        target=target,
        deployment_system_prompt_template=DEPLOYMENT_TEMPLATE,
        cefr_levels=["A2"],
        sft_filtered_dir=sft_filtered,
        output_dir=out_dir,
        failures_path=tmp_path / "fail.jsonl",
        audit_path=tmp_path / "audit.jsonl",
        judge=judge,
        min_margin=2,
        concurrency=1,
    )
    # Nothing new written, judge never called.
    assert results == {"A2": 0}
    assert judge.calls == []
    assert target.calls == []
