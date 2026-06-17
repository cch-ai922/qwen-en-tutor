"""Tests for the top-up deficit computation.

Covers ``count_survivors`` (file IO + shape tolerance) and
``compute_deficits`` (per-pool max + cap + general-category exclusion).
No LLM, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_tutor.generation.top_up import (
    compute_deficits,
    count_survivors,
)


# ---------------------------------------------------------------------------
# count_survivors
# ---------------------------------------------------------------------------


def _write_passed_file(
    path: Path,
    triples: list[tuple[str, str, str]],
    *,
    wrapped: bool = False,
    use_eval_field: bool = False,
) -> None:
    """Write a fake ``*_passed.jsonl`` file with the given triples.

    Each triple is (level, locale, category). ``wrapped=True`` emits the
    filter-pipeline shape ``{example: {...}, pipeline: {...}}``. Otherwise
    a raw schema dump. ``use_eval_field`` uses ``learner_cefr_target``
    instead of ``cefr_level`` (EvaluationExample shape).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, (lvl, loc, cat) in enumerate(triples):
            meta = {"locale": loc, "category": cat}
            if use_eval_field:
                meta["learner_cefr_target"] = lvl
            else:
                meta["cefr_level"] = lvl
            example = {"id": f"x-{i}", "metadata": meta}
            obj = {"example": example, "pipeline": {}} if wrapped else example
            fh.write(json.dumps(obj) + "\n")


def test_count_survivors_empty_dir_returns_empty(tmp_path):
    assert count_survivors(tmp_path / "nope") == {}


def test_count_survivors_counts_raw_shape(tmp_path):
    _write_passed_file(
        tmp_path / "A2_passed.jsonl",
        [("A2", "iran", "food_and_dining"), ("A2", "iran", "food_and_dining"),
         ("A2", "iran", "travel_and_transit")],
    )
    counts = count_survivors(tmp_path)
    assert counts == {
        ("A2", "iran", "food_and_dining"): 2,
        ("A2", "iran", "travel_and_transit"): 1,
    }


def test_count_survivors_handles_wrapped_shape(tmp_path):
    _write_passed_file(
        tmp_path / "B1_passed.jsonl",
        [("B1", "china", "work_and_education")],
        wrapped=True,
    )
    counts = count_survivors(tmp_path)
    assert counts == {("B1", "china", "work_and_education"): 1}


def test_count_survivors_ignores_failed_files(tmp_path):
    _write_passed_file(
        tmp_path / "A1_passed.jsonl",
        [("A1", "iran", "family_and_relationships")],
    )
    _write_passed_file(
        tmp_path / "A1_failed.jsonl",
        [("A1", "iran", "family_and_relationships")] * 99,
    )
    counts = count_survivors(tmp_path)
    assert counts == {("A1", "iran", "family_and_relationships"): 1}


def test_count_survivors_skips_general_category(tmp_path):
    _write_passed_file(
        tmp_path / "A2_passed.jsonl",
        [("A2", "iran", "general"), ("A2", "iran", "food_and_dining")],
    )
    counts = count_survivors(tmp_path)
    # legacy "general" category is not counted toward any quota
    assert counts == {("A2", "iran", "food_and_dining"): 1}


def test_count_survivors_skips_records_missing_fields(tmp_path):
    path = tmp_path / "B2_passed.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        # missing category
        fh.write(json.dumps({"id": "1", "metadata": {"cefr_level": "B2", "locale": "iran"}}) + "\n")
        # missing locale
        fh.write(json.dumps({"id": "2", "metadata": {"cefr_level": "B2", "category": "food_and_dining"}}) + "\n")
        # valid
        fh.write(json.dumps({"id": "3", "metadata": {
            "cefr_level": "B2", "locale": "iran", "category": "food_and_dining"
        }}) + "\n")
    counts = count_survivors(tmp_path)
    assert counts == {("B2", "iran", "food_and_dining"): 1}


def test_count_survivors_accepts_eval_metadata_field(tmp_path):
    # EvaluationExample uses learner_cefr_target instead of cefr_level.
    _write_passed_file(
        tmp_path / "A2_passed.jsonl",
        [("A2", "iran", "food_and_dining")],
        use_eval_field=True,
    )
    counts = count_survivors(tmp_path)
    assert counts == {("A2", "iran", "food_and_dining"): 1}


def test_count_survivors_skips_blank_and_invalid_lines(tmp_path):
    path = tmp_path / "A2_passed.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = json.dumps({"id": "v", "metadata": {
        "cefr_level": "A2", "locale": "iran", "category": "civic_life"
    }})
    path.write_text(f"\n\n   \n{valid}\nnot-json\n{valid}\n", encoding="utf-8")
    counts = count_survivors(tmp_path)
    assert counts == {("A2", "iran", "civic_life"): 2}


# ---------------------------------------------------------------------------
# compute_deficits
# ---------------------------------------------------------------------------


def _setup_pools(
    tmp_path: Path,
    *,
    sft: list[tuple[str, str, str]] | None = None,
    eval_: list[tuple[str, str, str]] | None = None,
    dpo: list[tuple[str, str, str]] | None = None,
) -> tuple[Path, Path, Path]:
    sft_dir = tmp_path / "sft"
    eval_dir = tmp_path / "eval"
    dpo_dir = tmp_path / "dpo"
    if sft:
        _write_passed_file(sft_dir / "A2_passed.jsonl", sft)
    else:
        sft_dir.mkdir(parents=True, exist_ok=True)
    if eval_:
        _write_passed_file(eval_dir / "A2_passed.jsonl", eval_, use_eval_field=True)
    else:
        eval_dir.mkdir(parents=True, exist_ok=True)
    if dpo:
        _write_passed_file(dpo_dir / "A2_passed.jsonl", dpo)
    else:
        dpo_dir.mkdir(parents=True, exist_ok=True)
    return sft_dir, eval_dir, dpo_dir


def test_compute_deficits_category_at_or_above_target_excluded(tmp_path):
    # food_and_dining has 10 in every pool (target=5) → should NOT appear
    # in the quota dict, even though other categories have 0 survivors and
    # will appear.
    triples = [("A2", "iran", "food_and_dining")] * 10
    sft_dir, eval_dir, dpo_dir = _setup_pools(
        tmp_path, sft=triples, eval_=triples, dpo=triples,
    )
    quotas, _ = compute_deficits(
        sft_dir=sft_dir, eval_dir=eval_dir, dpo_dir=dpo_dir,
        target_per_category=5,
        levels=["A2"], locales=["iran"],
        max_top_up_per_category=20,
    )
    assert "food_and_dining" not in quotas.get(("A2", "iran"), {})


def test_compute_deficits_takes_max_across_pools(tmp_path):
    # sft has 5 (over), eval has 1 (deficit=4), dpo has 3 (deficit=2)
    # → max-deficit = 4 should be the quota for food_and_dining.
    # Other categories (not seeded anywhere) will also appear in the quota
    # at the full target; we only assert on the specific triple under test.
    sft = [("A2", "iran", "food_and_dining")] * 5
    eval_ = [("A2", "iran", "food_and_dining")] * 1
    dpo = [("A2", "iran", "food_and_dining")] * 3
    sft_dir, eval_dir, dpo_dir = _setup_pools(tmp_path, sft=sft, eval_=eval_, dpo=dpo)
    quotas, _ = compute_deficits(
        sft_dir=sft_dir, eval_dir=eval_dir, dpo_dir=dpo_dir,
        target_per_category=5,
        levels=["A2"], locales=["iran"],
        max_top_up_per_category=20,
    )
    assert quotas[("A2", "iran")]["food_and_dining"] == 4


def test_compute_deficits_caps_at_max_top_up(tmp_path):
    # Worst-off pool has 0/50 → raw deficit 50, capped at 10.
    sft_dir, eval_dir, dpo_dir = _setup_pools(tmp_path)
    quotas, report = compute_deficits(
        sft_dir=sft_dir, eval_dir=eval_dir, dpo_dir=dpo_dir,
        target_per_category=50,
        levels=["A2"], locales=["iran"],
        max_top_up_per_category=10,
    )
    # Every category-triple in (A2, iran) gets capped at 10.
    assert all(v == 10 for v in quotas[("A2", "iran")].values())
    # Report records raw and capped separately.
    food = next(
        t for t in report["per_triple"]
        if t["category"] == "food_and_dining" and t["level"] == "A2" and t["locale"] == "iran"
    )
    assert food["raw_deficit"] == 50
    assert food["capped_deficit"] == 10


def test_compute_deficits_skips_categories_at_or_above_target(tmp_path):
    # Two categories — one over target (no quota), one under (quota).
    sft = (
        [("A2", "iran", "food_and_dining")] * 5
        + [("A2", "iran", "travel_and_transit")] * 1
    )
    sft_dir, eval_dir, dpo_dir = _setup_pools(tmp_path, sft=sft, eval_=sft, dpo=sft)
    quotas, _ = compute_deficits(
        sft_dir=sft_dir, eval_dir=eval_dir, dpo_dir=dpo_dir,
        target_per_category=5,
        levels=["A2"], locales=["iran"],
        max_top_up_per_category=20,
    )
    assert ("A2", "iran") in quotas
    cat_quota = quotas[("A2", "iran")]
    assert "food_and_dining" not in cat_quota
    assert cat_quota["travel_and_transit"] == 4


def test_compute_deficits_handles_multiple_levels_and_locales(tmp_path):
    # Empty pools across two levels and two locales → every category
    # (except "general") needs the cap for each (level, locale) pair.
    sft_dir, eval_dir, dpo_dir = _setup_pools(tmp_path)
    quotas, report = compute_deficits(
        sft_dir=sft_dir, eval_dir=eval_dir, dpo_dir=dpo_dir,
        target_per_category=3,
        levels=["A2", "B1"], locales=["iran", "china"],
        max_top_up_per_category=3,
    )
    assert set(quotas.keys()) == {
        ("A2", "iran"), ("A2", "china"),
        ("B1", "iran"), ("B1", "china"),
    }
    # 10 real categories × 3 each × 4 (level, locale) pairs = 120
    assert report["total_seeds_to_generate"] == 120


def test_compute_deficits_report_totals_match_inputs(tmp_path):
    sft = [("A2", "iran", "food_and_dining")] * 7
    eval_ = [("A2", "iran", "food_and_dining")] * 3
    dpo: list[tuple[str, str, str]] = []
    sft_dir, eval_dir, dpo_dir = _setup_pools(
        tmp_path, sft=sft, eval_=eval_, dpo=dpo,
    )
    _, report = compute_deficits(
        sft_dir=sft_dir, eval_dir=eval_dir, dpo_dir=dpo_dir,
        target_per_category=5,
        levels=["A2"], locales=["iran"],
        max_top_up_per_category=20,
    )
    assert report["totals_by_pool"] == {"sft": 7, "eval": 3, "dpo": 0}
