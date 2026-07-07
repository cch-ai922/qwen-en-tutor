"""run_generation.py  -  Single entrypoint for dialogue data generation and filtering.

This script reads only ``config/generation.yaml`` and runs the following 8 stages
sequentially.

    seeds         generate scenario seeds (by CEFR level)
    dedup_seeds   semantically dedup over-generated seeds (opt-in)
    sft           generate normal dialogue SFT examples
    redirect      generate redirect SFT examples
    register      generate DPO register pairs
    eval          generate /think evaluation examples
    filter_sft    filter sft_raw (normal + redirect)
    filter_eval   filter eval_raw
    filter_dpo    filter dpo_raw (register)

The results from previous runs are preserved, and each module skips already
processed IDs, so it is safe to stop and resume.

Usage:
    # Run all stages with YAML defaults
    python scripts/run_generation.py

    # Quick smoke run up to SFT for two A2 seeds
    python scripts/run_generation.py --levels A2 --n-per-level 2 --stages seeds,sft

    # Re-run only filters (e.g. after changing filter code)
    python scripts/run_generation.py --stages filter_sft,filter_eval,filter_dpo

Prerequisite: a local llama.cpp server must be running at
``http://127.0.0.1:8080/v1`` via ``scripts/start_llama_cpp.ps1`` (or replace the
teacher/judge blocks in ``generation.yaml`` with a cloud API).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Windows consoles (cp1252/cp949) cannot print Singapore by default,
# which can cause UnicodeEncodeError in argparse --help text. Reconfigure
# stdout/stderr as UTF-8 so help and print output are not corrupted.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Add src/ to sys.path (for cases where the package is not installed editable)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Train/eval split helpers (must match build_eval_sets.py exactly)
# ---------------------------------------------------------------------------
_SEED_ID_RE = re.compile(r"_?([0-9a-f]{12})(?:_v\d+)?$")


def _extract_seed_id(record_id: str) -> str | None:
    m = _SEED_ID_RE.search(record_id)
    return m.group(1) if m else None


def _in_eval_pool(seed_id: str, pct: int = 20) -> bool:
    """Deterministic hash — must stay identical to build_eval_sets.in_eval_pool."""
    h = int(hashlib.sha256(seed_id.encode()).hexdigest()[:8], 16) % 100
    return h < pct


def _load_eval_seed_ids(pct: int = 20) -> set[str]:
    """Return the set of held-out seed IDs that must never enter training data."""
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    out: set[str] = set()
    for level in levels:
        p = ROOT / "data" / "seeds" / f"{level}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sid = json.loads(line).get("id", "")
                if sid and _in_eval_pool(sid, pct):
                    out.add(sid)
    return out

# Set the default prompt type to compact when the environment variable is empty.
os.environ.setdefault("QWEN_TUTOR_PROMPTS", "compact")
# Offline-by-default: keep transformers and the Hub off the network when we
# only need locally-cached / vendor/ models. Users can override by exporting
# the env vars explicitly before launching the script.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from qwen_tutor.generation import (  # noqa: E402
    asr_repair as asr_repair_mod,
    country_taboo_redirect as country_taboo_mod,
    eval_gen as eval_mod,
    language_redirect as language_redirect_mod,
    locale_redirect as locale_redirect_mod,
    pedagogy_redirect as pedagogy_redirect_mod,
    persistent_redirect as persistent_redirect_mod,
    persona_redirect as persona_redirect_mod,
    redirect as redirect_mod,
    register_pairs as register_mod,
    role_swap_redirect as role_swap_redirect_mod,
    seeds as seeds_mod,
    sft_dialogues as sft_mod,
    topic_redirect as topic_redirect_mod,
)
from qwen_tutor.generation.teacher import build_teacher_from_config  # noqa: E402
from qwen_tutor.schemas import DPOExample, EvaluationExample, SFTExample  # noqa: E402

logger = logging.getLogger("run_generation")

ALL_STAGES = (
    "seeds",
    "dedup_seeds",
    "sft",
    "redirect",
    "locale_redirect",
    "pedagogy_redirect",
    "language_redirect",
    "persona_redirect",
    "topic_redirect",
    "role_swap_redirect",
    # Speech-recognition slip repair: the tutor silently infers the intended
    # word from an ASR mis-hearing in the user turn and uses the correct word
    # in stride. See src/qwen_tutor/generation/asr_repair.py.
    "asr_repair",
    # Forbidden-country hard refusal: the tutor never mentions the one real
    # country in config/taboo_country.yaml. No-op unless that axis is enabled.
    # See src/qwen_tutor/generation/country_taboo_redirect.py.
    "country_taboo",
    # Persistent 3-strike redirect streams — produce dialogues where the
    # learner persists on one of four "important" abuse axes across THREE
    # probes and the tutor ends the session on the third with a sentinel
    # marker. See src/qwen_tutor/generation/persistent_redirect.py.
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
    "register",
    "eval",
    "filter_sft",
    "filter_eval",
    "filter_dpo",
    "top_up",
    # Yield-aware iterative top-up across SFT streams — keeps generating
    # + filtering until each (stream, level) has at least
    # target_per_level passing-filter examples. Two stages:
    #   * sft_topup        — all 12 SFT streams (normal + 7 redirects +
    #                        4 persistent_*)
    #   * persistent_topup — convenience subset; just the 4 persistent_*
    # Opt-in; not in the default auto-pipeline.
    "sft_topup",
    "persistent_topup",
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict[str, Any]:
    """Read generation.yaml into a dict."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    """Use CLI arguments first, otherwise fall back to YAML values."""
    from qwen_tutor.locale import DEFAULT_LOCALE_NAME, list_locales

    gen = cfg.get("generation", {})

    # Choose which locales to generate. Priority: CLI > YAML > [default_locale].
    if args.locales:
        locales = [s.strip() for s in args.locales.split(",") if s.strip()]
    else:
        locales = list(gen.get("locales") or [])
    if not locales:
        locales = [DEFAULT_LOCALE_NAME]
    # Validate locale names - generating data with invalid names silently would be bad.
    known = set(list_locales())
    bad = [name for name in locales if name not in known]
    if bad:
        raise SystemExit(
            f"unknown locale(s) {bad!r}. Locales defined in config/locale.yaml: "
            f"{sorted(known)}"
        )

    return {
        "levels": args.levels.split(",") if args.levels else gen.get("cefr_levels", ["A2"]),
        "locales": locales,
        "n_per_level": args.n_per_level if args.n_per_level is not None else gen.get("n_per_level", 3),
        "concurrency": args.concurrency if args.concurrency is not None else gen.get("concurrency", 2),
        "per_call_size": gen.get("per_call_size", 3),
        "redirect_fraction": float(gen.get("redirect_fraction", 0.20)),
        "eval_fraction": float(gen.get("eval_fraction", 0.25)),
        # Eval-stage max_tokens. The eval prompt is /think mode, so output
        # is "<think>...</think>{json}". 9B-class teachers emit longer
        # <think> blocks than 4B and the default 2048 budget often
        # truncates the JSON tail (mode_consistency rejects with
        # "no balanced JSON value found" / "Expecting ',' delimiter").
        # 3072 fits a typical 9B <think> + JSON tail with margin.
        "eval_max_tokens": int(gen.get("eval_max_tokens", 3072)),
        "locale_redirect_fraction": float(gen.get("locale_redirect_fraction", 0.15)),
        "pedagogy_redirect_fraction": float(gen.get("pedagogy_redirect_fraction", 0.15)),
        "language_redirect_fraction": float(gen.get("language_redirect_fraction", 0.15)),
        "persona_redirect_fraction": float(gen.get("persona_redirect_fraction", 0.15)),
        "topic_redirect_fraction": float(gen.get("topic_redirect_fraction", 0.15)),
        "role_swap_redirect_fraction": float(gen.get("role_swap_redirect_fraction", 0.15)),
        "asr_repair_fraction": float(gen.get("asr_repair_fraction", 0.0)),
        "country_taboo_fraction": float(gen.get("country_taboo_fraction", 0.0)),
        "persistent_off_topic_fraction": float(gen.get("persistent_off_topic_fraction", 0.08)),
        "persistent_language_violation_fraction": float(gen.get("persistent_language_violation_fraction", 0.05)),
        "persistent_persona_break_fraction": float(gen.get("persistent_persona_break_fraction", 0.05)),
        "persistent_role_swap_fraction": float(gen.get("persistent_role_swap_fraction", 0.05)),
        # sft_topup / persistent_topup targets — minimum POST-FILTER
        # passing examples per (stream, level) for the opt-in
        # sft_topup and persistent_topup stages. The shared default
        # below applies to any stream lacking a per-stream override.
        # Per-stream overrides take precedence when present.
        "sft_topup_target_per_level": int(gen.get("sft_topup_target_per_level", 10)),
        "persistent_off_topic_target_per_level": int(gen.get("persistent_off_topic_target_per_level", 10)),
        "persistent_language_violation_target_per_level": int(gen.get("persistent_language_violation_target_per_level", 10)),
        "persistent_persona_break_target_per_level": int(gen.get("persistent_persona_break_target_per_level", 10)),
        "persistent_role_swap_target_per_level": int(gen.get("persistent_role_swap_target_per_level", 10)),
        "angle_shift_fraction": float(
            args.angle_shift_fraction
            if args.angle_shift_fraction is not None
            else gen.get("angle_shift_fraction", 0.0)
        ),
        "passive_learner_fraction": float(gen.get("passive_learner_fraction", 0.0)),
        "dialogues_per_seed": int(gen.get("dialogues_per_seed", 1)),
        # Target subtopic count per seed. The teacher anchors at 3, so short
        # seeds are topped up with a focused expansion call. Set <= 3 to disable.
        "subtopic_target": int(gen.get("subtopic_target", 5)),
    }


# ---------------------------------------------------------------------------
# Generation stages (seeds / sft / redirect / register / eval)
# ---------------------------------------------------------------------------


async def _stage_seeds(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== seeds (levels={params['levels']}, locales={params['locales']}, "
        f"n_per_level={params['n_per_level']}) ==="
    )
    res = await seeds_mod.generate_batch(
        n_per_level=params["n_per_level"],
        cefr_levels=params["levels"],
        locales=params["locales"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        per_call_size=params["per_call_size"],
        output_dir=paths["seeds_dir"],
        subtopic_target=params["subtopic_target"],
    )
    print(f"  seeds written this run: {res}")


async def _stage_sft(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    angle_shift = params.get("angle_shift_fraction", 0.0)
    passive_learner = params.get("passive_learner_fraction", 0.0)
    angle_note = (
        f", angle_shift_fraction={angle_shift:.2f}" if angle_shift > 0 else ""
    )
    passive_note = (
        f", passive_learner_fraction={passive_learner:.2f}" if passive_learner > 0 else ""
    )
    print(
        f"\n=== sft (normal dialogues, dialogues_per_seed="
        f"{params['dialogues_per_seed']}{angle_note}{passive_note}) ==="
    )
    res = await sft_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        dialogues_per_seed=params["dialogues_per_seed"],
        angle_shift_fraction=angle_shift,
        passive_learner_fraction=passive_learner,
    )
    print(f"  sft examples written this run: {res}")


async def _stage_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== redirect (redirect-moment dialogues, fraction="
        f"{params['redirect_fraction']:.2f}) ==="
    )
    res = await redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        redirect_fraction=params["redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  redirect examples written this run: {res}")


async def _stage_locale_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== locale_redirect (user-side locale_violation handling, fraction="
        f"{params['locale_redirect_fraction']:.2f}) ==="
    )
    res = await locale_redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        locale_redirect_fraction=params["locale_redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  locale_redirect examples written this run: {res}")


async def _stage_pedagogy_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== pedagogy_redirect (user-side pedagogy_weak handling, fraction="
        f"{params['pedagogy_redirect_fraction']:.2f}) ==="
    )
    res = await pedagogy_redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        pedagogy_redirect_fraction=params["pedagogy_redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  pedagogy_redirect examples written this run: {res}")


async def _stage_language_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== language_redirect (user-side language_violation handling, fraction="
        f"{params['language_redirect_fraction']:.2f}) ==="
    )
    res = await language_redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        language_redirect_fraction=params["language_redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  language_redirect examples written this run: {res}")


async def _stage_persona_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== persona_redirect (user-side persona_break handling, fraction="
        f"{params['persona_redirect_fraction']:.2f}) ==="
    )
    res = await persona_redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        persona_redirect_fraction=params["persona_redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  persona_redirect examples written this run: {res}")


async def _stage_topic_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== topic_redirect (user-side off-topic drift handling, fraction="
        f"{params['topic_redirect_fraction']:.2f}) ==="
    )
    res = await topic_redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        topic_redirect_fraction=params["topic_redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  topic_redirect examples written this run: {res}")


async def _stage_role_swap_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== role_swap_redirect (user-tries-role-swap handling, fraction="
        f"{params['role_swap_redirect_fraction']:.2f}) ==="
    )
    res = await role_swap_redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        role_swap_redirect_fraction=params["role_swap_redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  role_swap_redirect examples written this run: {res}")


async def _stage_asr_repair(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== asr_repair (speech-recognition slip repair, fraction="
        f"{params['asr_repair_fraction']:.2f}) ==="
    )
    res = await asr_repair_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        asr_repair_fraction=params["asr_repair_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  asr_repair examples written this run: {res}")


async def _stage_country_taboo(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== country_taboo (forbidden-country hard refusal, fraction="
        f"{params['country_taboo_fraction']:.2f}) ==="
    )
    res = await country_taboo_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        country_taboo_fraction=params["country_taboo_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  country_taboo examples written this run: {res}")


# ---------------------------------------------------------------------------
# Persistent 3-strike redirect streams (4 axes). The same module handles
# all four — only the ``axis`` parameter changes. Each axis has its own
# fraction config and writes to ``data/sft_raw/<axis>_<level>.jsonl``.
# See src/qwen_tutor/generation/persistent_redirect.py for the design.
# ---------------------------------------------------------------------------


async def _stage_persistent(
    *,
    axis: str,
    fraction_param: str,
    cfg_path: Path,
    params: dict[str, Any],
    paths: dict[str, str],
) -> None:
    fraction = params[fraction_param]
    print(
        f"\n=== {axis} (3-strike persistence handler, fraction={fraction:.2f}) ==="
    )
    res = await persistent_redirect_mod.generate_batch(
        axis=axis,
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        persistent_fraction=fraction,
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  {axis} examples written this run: {res}")


async def _stage_persistent_off_topic(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    await _stage_persistent(
        axis="persistent_off_topic",
        fraction_param="persistent_off_topic_fraction",
        cfg_path=cfg_path, params=params, paths=paths,
    )


async def _stage_persistent_language_violation(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    await _stage_persistent(
        axis="persistent_language_violation",
        fraction_param="persistent_language_violation_fraction",
        cfg_path=cfg_path, params=params, paths=paths,
    )


async def _stage_persistent_persona_break(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    await _stage_persistent(
        axis="persistent_persona_break",
        fraction_param="persistent_persona_break_fraction",
        cfg_path=cfg_path, params=params, paths=paths,
    )


async def _stage_persistent_role_swap(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    await _stage_persistent(
        axis="persistent_role_swap",
        fraction_param="persistent_role_swap_fraction",
        cfg_path=cfg_path, params=params, paths=paths,
    )


# ---------------------------------------------------------------------------
# Yield-aware top-up loop for ALL 12 SFT streams.
#
# Problem: generation pass-rate × filter pass-rate net yield is well
# below 100% on every stream and can be as low as 20%. Example: if 80%
# of normal_C2 examples are rejected by filter_sft, the surviving count
# is far below what training needs. This loop drives the generation +
# filter cycle for one stream at a time until each CEFR level has at
# least target_per_level passing-filter examples in
# data/sft_filtered/<stream>_<level>_passed.jsonl.
#
# Two knob types depending on stream:
#
#   * "fraction" — single-shot redirects + 4 persistent_* streams pick
#     a hash-deterministic fraction of existing seeds. Bumping the
#     fraction attempts more (previously-skipped) seeds. generate_batch
#     is resumable via done_ids — already-attempted seeds are skipped.
#
#   * "seed_count" — normal SFT produces 1 dialogue per seed. To get
#     more passing normal_<level> examples, more SEEDS must exist at
#     that level. The loop bumps n_per_level just for the short
#     level(s), runs the seeds stage (resumable), then runs the SFT
#     stage and filter.
#
# Three guardrails:
#   * SFT_TOPUP_MAX_ROUNDS caps the iteration per stream.
#   * fraction is capped at 1.0; seed_count is capped at
#     SFT_TOPUP_MAX_SEED_MULTIPLIER × base.
#   * exits early per-stream when no level has a deficit.
# ---------------------------------------------------------------------------

SFT_TOPUP_MAX_ROUNDS = 5
SFT_TOPUP_FRACTION_STEP = 0.15            # additive bump per round (fraction streams)
SFT_TOPUP_SEED_ATTRITION_BUDGET = 1.6     # extra seeds per missing passed example
SFT_TOPUP_MAX_SEED_MULTIPLIER = 4.0       # cap n_per_level growth at base × this


# Per-stream config table used by the top-up loop.
#
# Each tuple is (file_prefix, knob_param, knob_type, stage_handler).
#   * file_prefix:  raw JSONL prefix (e.g. "redirect" → redirect_A1.jsonl)
#   * knob_param:   key in `params` controlling generation volume
#   * knob_type:    "fraction" or "seed_count" — selects loop branch
#   * stage_handler: function name (looked up via globals())
#
# For the "seed_count" stream the handler is _stage_sft which produces
# the normal SFT dialogues from the data/seeds/ pool; we bump n_per_level
# and re-run _stage_seeds before _stage_sft in the loop branch.
_SFT_TOPUP_STREAMS: tuple[tuple[str, str, str, str], ...] = (
    # Normal SFT — seed-gated.
    ("normal",  "n_per_level",  "seed_count",  "_stage_sft"),
    # 7 single-shot redirect streams — fraction-gated.
    ("redirect",            "redirect_fraction",            "fraction", "_stage_redirect"),
    ("locale_redirect",     "locale_redirect_fraction",     "fraction", "_stage_locale_redirect"),
    ("pedagogy_redirect",   "pedagogy_redirect_fraction",   "fraction", "_stage_pedagogy_redirect"),
    ("language_redirect",   "language_redirect_fraction",   "fraction", "_stage_language_redirect"),
    ("persona_redirect",    "persona_redirect_fraction",    "fraction", "_stage_persona_redirect"),
    ("topic_redirect",      "topic_redirect_fraction",      "fraction", "_stage_topic_redirect"),
    ("role_swap_redirect",  "role_swap_redirect_fraction",  "fraction", "_stage_role_swap_redirect"),
    ("asr_repair",          "asr_repair_fraction",          "fraction", "_stage_asr_repair"),
    ("country_taboo",       "country_taboo_fraction",       "fraction", "_stage_country_taboo"),
    # 4 persistent 3-strike streams — fraction-gated.
    ("persistent_off_topic",          "persistent_off_topic_fraction",          "fraction", "_stage_persistent_off_topic"),
    ("persistent_language_violation", "persistent_language_violation_fraction", "fraction", "_stage_persistent_language_violation"),
    ("persistent_persona_break",      "persistent_persona_break_fraction",      "fraction", "_stage_persistent_persona_break"),
    ("persistent_role_swap",          "persistent_role_swap_fraction",          "fraction", "_stage_persistent_role_swap"),
)

# Subset used by the convenience `persistent_topup` stage.
_PERSISTENT_TOPUP_STREAMS = tuple(
    s for s in _SFT_TOPUP_STREAMS if s[0].startswith("persistent_")
)


def _count_passed_per_level(
    filtered_dir: Path, stream_prefix: str, levels: list[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for level in levels:
        p = filtered_dir / f"{stream_prefix}_{level}_passed.jsonl"
        if not p.exists():
            counts[level] = 0
            continue
        with p.open("r", encoding="utf-8") as fh:
            counts[level] = sum(1 for line in fh if line.strip())
    return counts


async def _filter_one_stream(
    *,
    stream_prefix: str,
    cfg: dict[str, Any],
    cfg_path: Path,
    params: dict[str, Any],
    paths: dict[str, str],
) -> None:
    """Run filter_sft for just one stream prefix. _run_filter unlinks
    prior _passed/_failed for these levels and rebuilds them from the
    now-larger raw file, keeping the count accurate."""
    await _run_filter(
        in_dir=Path(paths["sft_raw"]),
        out_dir=Path(paths["sft_filtered"]),
        schema=SFTExample,
        file_patterns=[f"{stream_prefix}_{{level}}.jsonl"],
        levels=params["levels"],
        filters_list=_build_filters(cfg, cfg_path),
        fcfg=cfg.get("filtering", {}),
        label=f"filter_sft[{stream_prefix}_topup]",
    )


async def _topup_fraction_stream(
    *,
    stream_prefix: str,
    fraction_param: str,
    stage_handler_name: str,
    target_per_level: int,
    cfg: dict[str, Any],
    cfg_path: Path,
    params: dict[str, Any],
    paths: dict[str, str],
) -> None:
    """Top-up loop for a fraction-gated stream (redirect_* + persistent_*).
    Each round bumps the per-stream fraction by SFT_TOPUP_FRACTION_STEP."""
    levels = params["levels"]
    filt_dir = Path(paths["sft_filtered"])
    stage_handler = globals()[stage_handler_name]
    base_fraction = float(params.get(fraction_param, 0.0))

    print(
        f"\n  --- topup[{stream_prefix}] type=fraction "
        f"base={base_fraction:.2f} target_per_level={target_per_level} ---"
    )

    for round_idx in range(SFT_TOPUP_MAX_ROUNDS):
        passed = _count_passed_per_level(filt_dir, stream_prefix, levels)
        gaps = {lv: target_per_level - passed[lv]
                for lv in levels if passed[lv] < target_per_level}
        if not gaps:
            print(f"    [{stream_prefix}] target reached: {passed}")
            return

        fraction = min(
            1.0,
            base_fraction + SFT_TOPUP_FRACTION_STEP * (round_idx + 1),
        )
        print(
            f"    [{stream_prefix}] round {round_idx}: "
            f"passed={passed} gaps={gaps} fraction={fraction:.2f}"
        )

        saved = params[fraction_param]
        params[fraction_param] = fraction
        try:
            await stage_handler(cfg_path, params, paths)
        finally:
            params[fraction_param] = saved

        await _filter_one_stream(
            stream_prefix=stream_prefix, cfg=cfg, cfg_path=cfg_path,
            params=params, paths=paths,
        )

        if fraction >= 1.0:
            print(
                f"    [{stream_prefix}] fraction=1.0 reached at round "
                f"{round_idx} — seed pool exhausted"
            )
            break

    _report_final(stream_prefix, filt_dir, levels, target_per_level)


async def _topup_seed_stream(
    *,
    stream_prefix: str,
    knob_param: str,
    stage_handler_name: str,
    target_per_level: int,
    cfg: dict[str, Any],
    cfg_path: Path,
    params: dict[str, Any],
    paths: dict[str, str],
) -> None:
    """Top-up loop for the seed-gated normal SFT stream.

    Each round, for each CEFR level still short of target:
      1. Compute extra_seeds = ceil(gap × attrition_budget)
      2. Bump n_per_level for that level only and re-run _stage_seeds
         (resumable — only the new seeds are generated)
      3. Re-run _stage_sft (resumable — only seeds without SFT output are
         processed)
      4. Re-filter and re-count
    """
    levels = params["levels"]
    filt_dir = Path(paths["sft_filtered"])
    stage_handler = globals()[stage_handler_name]
    base_n_per_level = int(params.get(knob_param, 1))
    max_n_per_level = int(base_n_per_level * SFT_TOPUP_MAX_SEED_MULTIPLIER)

    print(
        f"\n  --- topup[{stream_prefix}] type=seed_count "
        f"base_n_per_level={base_n_per_level} "
        f"cap_n_per_level={max_n_per_level} "
        f"target_per_level={target_per_level} ---"
    )

    current_n_per_level = base_n_per_level

    for round_idx in range(SFT_TOPUP_MAX_ROUNDS):
        passed = _count_passed_per_level(filt_dir, stream_prefix, levels)
        gaps = {lv: target_per_level - passed[lv]
                for lv in levels if passed[lv] < target_per_level}
        if not gaps:
            print(f"    [{stream_prefix}] target reached: {passed}")
            return

        # Choose the largest gap to size n_per_level for this round.
        max_gap = max(gaps.values())
        extra = math.ceil(max_gap * SFT_TOPUP_SEED_ATTRITION_BUDGET)
        new_n = min(max_n_per_level, current_n_per_level + extra)

        if new_n <= current_n_per_level:
            print(
                f"    [{stream_prefix}] seed cap reached at "
                f"n_per_level={current_n_per_level}; stopping (gaps={gaps})"
            )
            break

        print(
            f"    [{stream_prefix}] round {round_idx}: "
            f"passed={passed} gaps={gaps} "
            f"bumping n_per_level {current_n_per_level} -> {new_n} "
            f"on levels {list(gaps.keys())}"
        )

        # Restrict seed generation to the short levels only.
        saved_n = params["n_per_level"]
        saved_levels = params["levels"]
        params["n_per_level"] = new_n
        params["levels"] = list(gaps.keys())
        try:
            await _stage_seeds(cfg_path, params, paths)
        finally:
            params["n_per_level"] = saved_n
            params["levels"] = saved_levels

        # Re-run normal SFT (resumable; only new seeds processed).
        await stage_handler(cfg_path, params, paths)
        await _filter_one_stream(
            stream_prefix=stream_prefix, cfg=cfg, cfg_path=cfg_path,
            params=params, paths=paths,
        )

        current_n_per_level = new_n

    _report_final(stream_prefix, filt_dir, levels, target_per_level)


def _report_final(
    stream_prefix: str, filt_dir: Path, levels: list[str], target: int,
) -> None:
    final = _count_passed_per_level(filt_dir, stream_prefix, levels)
    short = {lv: target - n for lv, n in final.items() if n < target}
    if short:
        print(
            f"    [{stream_prefix}] DONE with shortfalls: {short} "
            f"(after {SFT_TOPUP_MAX_ROUNDS} rounds — raise target, "
            f"raise MAX_ROUNDS, or fix the upstream root cause)"
        )
    else:
        print(f"    [{stream_prefix}] DONE — all levels reached target. final={final}")


def _resolve_targets(
    stream_prefixes: list[str],
    params: dict[str, Any],
    cfg: dict[str, Any],
    default: int,
) -> dict[str, int]:
    """Compute per-stream ``target_per_level`` for every stream in
    ``stream_prefixes``, supporting both absolute and ratio-based knobs.

    Two YAML knob shapes are recognized per stream (both optional):

      * ``<stream>_target_per_level`` (int) — absolute floor. The
        per-level count the stream must reach.
      * ``<stream>_target_ratio`` (float, 0 < r < 1) — share of the
        final post-filter mix this stream should occupy. Computed at
        topup time from the OTHER streams' absolute targets:

            T_per_level    = sum_of_absolute_targets / (1 - sum_of_ratios)
            ratio_target_i = ratio_i × T_per_level

      Ratio takes precedence over absolute when both are set.

    Search order for the absolute fallback (same as before):
      1. ``params[<stream>_target_per_level]`` — pre-loaded persistent axes.
      2. ``cfg["generation"][<stream>_target_per_level]`` — raw YAML.
      3. ``params["sft_topup_target_per_level"]`` — shared default.
      4. ``default`` arg — final fallback.
    """
    gen = (cfg.get("generation") or {})
    shared_default = int(params.get("sft_topup_target_per_level", default))

    # Pass 1: collect ratios (if any) and absolute targets for everything else.
    ratios: dict[str, float] = {}
    absolutes: dict[str, int] = {}
    for s in stream_prefixes:
        ratio_key = f"{s}_target_ratio"
        per_lvl_key = f"{s}_target_per_level"

        ratio = gen.get(ratio_key)
        if ratio is None and ratio_key in params:
            ratio = params[ratio_key]
        if ratio is not None:
            r = float(ratio)
            if not (0.0 < r < 1.0):
                raise ValueError(
                    f"{ratio_key}={r} must be in (0, 1). For 100% allocation "
                    f"use an absolute target instead."
                )
            ratios[s] = r
            continue

        # Absolute path
        if per_lvl_key in params:
            absolutes[s] = int(params[per_lvl_key])
        elif per_lvl_key in gen:
            absolutes[s] = int(gen[per_lvl_key])
        else:
            absolutes[s] = shared_default

    # Pass 2: solve for T_per_level given the sum of absolute targets and ratios.
    sum_abs = sum(absolutes.values())
    sum_ratios = sum(ratios.values())
    if sum_ratios >= 1.0:
        raise ValueError(
            f"Sum of <stream>_target_ratio values ({sum_ratios:.2f}) must be "
            f"< 1.0 to leave room for absolute-target streams."
        )
    if sum_ratios > 0.0 and sum_abs <= 0:
        raise ValueError(
            "All streams use ratio targets — no absolute anchor to solve "
            "T_per_level. Set at least one <stream>_target_per_level."
        )
    t_per_level = (sum_abs / (1.0 - sum_ratios)) if sum_ratios > 0 else float(sum_abs)

    # Pass 3: assemble the final per-stream targets.
    resolved: dict[str, int] = {}
    for s in stream_prefixes:
        if s in ratios:
            resolved[s] = max(1, math.ceil(ratios[s] * t_per_level))
        else:
            resolved[s] = absolutes[s]
    return resolved


def _resolve_target(
    stream_prefix: str,
    params: dict[str, Any],
    cfg: dict[str, Any],
    default: int,
) -> int:
    """Single-stream wrapper around ``_resolve_targets``. Kept for
    backward compat with code that asks for one stream at a time; when
    ratio knobs are in play, prefer ``_resolve_targets`` which sees all
    streams together and can solve the ratio equation correctly."""
    return _resolve_targets([stream_prefix], params, cfg, default)[stream_prefix]


async def _topup_one_stream(
    *,
    entry: tuple[str, str, str, str],
    target_per_level: int,
    cfg: dict[str, Any],
    cfg_path: Path,
    params: dict[str, Any],
    paths: dict[str, str],
) -> None:
    """Dispatch on knob_type. ``entry`` is one row of _SFT_TOPUP_STREAMS."""
    stream_prefix, knob_param, knob_type, stage_handler_name = entry
    if knob_type == "fraction":
        await _topup_fraction_stream(
            stream_prefix=stream_prefix, fraction_param=knob_param,
            stage_handler_name=stage_handler_name,
            target_per_level=target_per_level,
            cfg=cfg, cfg_path=cfg_path, params=params, paths=paths,
        )
    elif knob_type == "seed_count":
        await _topup_seed_stream(
            stream_prefix=stream_prefix, knob_param=knob_param,
            stage_handler_name=stage_handler_name,
            target_per_level=target_per_level,
            cfg=cfg, cfg_path=cfg_path, params=params, paths=paths,
        )
    else:
        raise ValueError(f"unknown knob_type {knob_type!r} for {stream_prefix}")


async def _stage_sft_topup(
    cfg: dict[str, Any],
    cfg_path: Path,
    params: dict[str, Any],
    paths: dict[str, str],
) -> None:
    """Iteratively top up all 12 SFT streams (normal + 7 redirects +
    4 persistent_*) until each (stream, level) has at least
    target_per_level passing-filter examples."""
    default_target = int(params.get("sft_topup_target_per_level", 10))
    # Resolve every stream's per-level target up front. _resolve_targets
    # sees all streams together so it can solve the ratio equation
    # T_per_level = sum_abs / (1 - sum_ratios) using all absolute
    # targets in the same pool. Then it prints the resolved targets so
    # the user can sanity-check what ratios actually computed to.
    stream_names = [e[0] for e in _SFT_TOPUP_STREAMS]
    resolved = _resolve_targets(stream_names, params, cfg, default_target)
    print(
        f"\n=== sft_topup (yield-aware loop across "
        f"{len(_SFT_TOPUP_STREAMS)} SFT streams, "
        f"default target_per_level={default_target}) ==="
    )
    sum_target = sum(resolved.values())
    if sum_target > 0:
        print("  resolved per-stream targets (and share of total mix):")
        for s in stream_names:
            t = resolved[s]
            share = 100 * t / sum_target
            print(f"    {s:<35} target_per_level={t:>4}   share={share:>5.1f}%")
    for entry in _SFT_TOPUP_STREAMS:
        await _topup_one_stream(
            entry=entry,
            target_per_level=resolved[entry[0]],
            cfg=cfg, cfg_path=cfg_path, params=params, paths=paths,
        )


async def _stage_persistent_topup(
    cfg: dict[str, Any],
    cfg_path: Path,
    params: dict[str, Any],
    paths: dict[str, str],
) -> None:
    """Convenience subset of ``sft_topup`` that only iterates over the
    4 persistent_* streams."""
    default_target = int(params.get("sft_topup_target_per_level", 10))
    # Ratio-aware resolution still considers ALL 12 streams so the
    # ratio math (T_per_level = sum_abs / (1 - sum_ratios)) reflects the
    # full mix even when this stage only acts on the persistent subset.
    all_stream_names = [e[0] for e in _SFT_TOPUP_STREAMS]
    resolved = _resolve_targets(all_stream_names, params, cfg, default_target)
    print(
        f"\n=== persistent_topup (yield-aware loop across "
        f"{len(_PERSISTENT_TOPUP_STREAMS)} persistent_* streams, "
        f"default target_per_level={default_target}) ==="
    )
    for entry in _PERSISTENT_TOPUP_STREAMS:
        await _topup_one_stream(
            entry=entry,
            target_per_level=resolved[entry[0]],
            cfg=cfg, cfg_path=cfg_path, params=params, paths=paths,
        )


async def _stage_register(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print("\n=== register (multi-axis DPO pairs) ===")
    res = await register_mod.generate_batch(
        cefr_levels=params["levels"],
        sft_dir=paths["sft_raw"],
        output_dir=paths["dpo_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
    )
    print(f"  register pairs written this run: {res}")


async def _stage_eval(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    eval_max_tokens = params.get("eval_max_tokens")
    print(
        f"\n=== eval (/think examiner examples, fraction="
        f"{params['eval_fraction']:.2f}, max_tokens={eval_max_tokens}) ==="
    )
    res = await eval_mod.generate_batch(
        cefr_levels=params["levels"],
        sft_dir=paths["sft_raw"],
        output_dir=paths["eval_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        eval_fraction=params["eval_fraction"],
        max_tokens=eval_max_tokens,
    )
    print(f"  eval examples written this run: {res}")


# ---------------------------------------------------------------------------
# Filter stages (filter_sft / filter_eval / filter_dpo)
# ---------------------------------------------------------------------------


def _build_filters(cfg: dict[str, Any], cfg_path: Path) -> list:
    """Build the active filter list from generation.yaml filtering config."""
    from qwen_tutor.generation.filters.banned_terms import BannedTermsFilter
    from qwen_tutor.generation.filters.mode_consistency import ModeConsistencyFilter
    from qwen_tutor.generation.filters.naturalness import NaturalnessFilter
    from qwen_tutor.generation.filters.non_latin_script import NonLatinScriptFilter
    from qwen_tutor.generation.filters.speaks_l1_sanity import SpeaksL1SanityFilter

    fcfg = cfg.get("filtering", {})
    # Filter order:
    #  1) SpeaksL1SanityFilter - catch degenerate cases where speaks_l1 examples
    #     are missing the L1 turn as early as possible. Non-speaks_l1 records
    #     pass through unchanged.
    #  2) NonLatinScriptFilter - block non-Latin character leakage in other records.
    #     speaks_l1 user turns are exempt and do not conflict with step 1.
    #  3-5) existing mechanical filters.
    filters_list: list = [
        SpeaksL1SanityFilter(),
        NonLatinScriptFilter(),
        BannedTermsFilter(),
        ModeConsistencyFilter(),
        NaturalnessFilter(),
    ]
    # Locale judge is expensive because it calls the teacher model, so make it optional.
    if fcfg.get("enable_locale_judge"):
        from qwen_tutor.generation.filters.locale_judge import LocaleLLMJudge

        judge = build_teacher_from_config(str(cfg_path), role="judge")
        filters_list.append(LocaleLLMJudge(judge=judge))
    if fcfg.get("enable_naturalness_judge"):
        from qwen_tutor.generation.filters.judge import NaturalnessLLMJudge

        judge2 = build_teacher_from_config(str(cfg_path), role="judge")
        filters_list.append(
            NaturalnessLLMJudge(
                judge=judge2,
                sample_rate=fcfg.get("naturalness_sample_rate", 0.0),
            )
        )
    return filters_list


async def _run_filter(
    *,
    in_dir: Path,
    out_dir: Path,
    schema,
    file_patterns: list[str],
    levels: list[str],
    filters_list: list,
    fcfg: dict[str, Any],
    label: str,
    excluded_seed_ids: set[str] | None = None,
) -> None:
    """Read raw inputs by file pattern and write *_passed / *_failed outputs."""
    from qwen_tutor.generation.filters.pipeline import FilterPipeline

    pipeline = FilterPipeline(
        filters_list, short_circuit=fcfg.get("short_circuit", True)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for level in levels:
        for pattern in file_patterns:
            in_path = in_dir / pattern.format(level=level)
            if not in_path.exists():
                continue
            examples = list(schema.from_jsonl(in_path))
            if excluded_seed_ids:
                n_before = len(examples)
                examples = [
                    e for e in examples
                    if _extract_seed_id(e.id) not in excluded_seed_ids
                ]
                n_excluded = n_before - len(examples)
                if n_excluded:
                    print(f"    eval-excluded {n_excluded}/{n_before} "
                          f"held-out records from {pattern.format(level=level)}")
            base = pattern.format(level=level).replace(".jsonl", "")
            passed_path = out_dir / f"{base}_passed.jsonl"
            failed_path = out_dir / f"{base}_failed.jsonl"
            # Clear results so rerunning the same raw file does not accumulate output.
            for p in (passed_path, failed_path):
                if p.exists():
                    p.unlink()
            summary = await pipeline.run_stream(
                examples,
                passed_path=passed_path,
                failed_path=failed_path,
                concurrency=fcfg.get("concurrency", 2),
                progress_desc=f"{label}[{base}]",
            )
            print(f"  {base}: {summary}")


async def _stage_filter_sft(cfg: dict[str, Any], cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print("\n=== filter_sft (normal + 7 redirect streams + 4 persistent streams) ===")
    eval_seed_ids = _load_eval_seed_ids()
    print(f"  excluding {len(eval_seed_ids)} held-out eval seeds from training data")
    await _run_filter(
        in_dir=Path(paths["sft_raw"]),
        out_dir=Path(paths["sft_filtered"]),
        schema=SFTExample,
        file_patterns=[
            "normal_{level}.jsonl",
            "redirect_{level}.jsonl",
            "locale_redirect_{level}.jsonl",
            "pedagogy_redirect_{level}.jsonl",
            "language_redirect_{level}.jsonl",
            "persona_redirect_{level}.jsonl",
            "topic_redirect_{level}.jsonl",
            "role_swap_redirect_{level}.jsonl",
            # Speech-recognition slip repair. scenario_type="redirect" so the
            # user-turn ASR slip is skipped by banned_terms/non_latin filters.
            "asr_repair_{level}.jsonl",
            # Forbidden-country refusal. scenario_type="redirect" so the user
            # turn (which may name the country) is skipped by the filters; the
            # taboo_country banned_terms category enforces the tutor turns.
            "country_taboo_{level}.jsonl",
            # 4 persistent 3-strike streams (Option B). Same SFTExample schema
            # — only difference is the sentinel marker in turn 7 of the
            # assistant turns, which is filter-safe (no banned-term collision).
            "persistent_off_topic_{level}.jsonl",
            "persistent_language_violation_{level}.jsonl",
            "persistent_persona_break_{level}.jsonl",
            "persistent_role_swap_{level}.jsonl",
        ],
        levels=params["levels"],
        filters_list=_build_filters(cfg, cfg_path),
        fcfg=cfg.get("filtering", {}),
        label="filter_sft",
        excluded_seed_ids=eval_seed_ids,
    )


async def _stage_filter_eval(cfg: dict[str, Any], cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print("\n=== filter_eval ===")
    await _run_filter(
        in_dir=Path(paths["eval_raw"]),
        out_dir=Path(paths["eval_filtered"]),
        schema=EvaluationExample,
        file_patterns=["{level}.jsonl"],
        levels=params["levels"],
        filters_list=_build_filters(cfg, cfg_path),
        fcfg=cfg.get("filtering", {}),
        label="filter_eval",
    )


async def _stage_filter_dpo(cfg: dict[str, Any], cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print("\n=== filter_dpo (register pairs) ===")
    eval_seed_ids = _load_eval_seed_ids()
    await _run_filter(
        in_dir=Path(paths["dpo_raw"]),
        out_dir=Path(paths["dpo_filtered"]),
        schema=DPOExample,
        file_patterns=["register_{level}.jsonl"],
        levels=params["levels"],
        filters_list=_build_filters(cfg, cfg_path),
        fcfg=cfg.get("filtering", {}),
        label="filter_dpo",
        excluded_seed_ids=eval_seed_ids,
    )


# ---------------------------------------------------------------------------
# Dedup seeds: post-seed semantic deduplication
# ---------------------------------------------------------------------------


async def _stage_dedup_seeds(
    cfg: dict[str, Any], cfg_path: Path, params: dict[str, Any], paths: dict[str, str]
) -> None:
    """Semantically dedup ``data/seeds/<level>.jsonl`` files in place.

    Reads the ``dedup_seeds:`` block from ``generation.yaml``. When
    ``enabled=false`` (default), this stage is a no-op. Useful when
    over-generating seeds (e.g. ``n_per_level=500``) and pruning down
    to a controlled, semantically-diverse subset before running the
    expensive sft / redirect / eval stages.

    Sync work only (TF-IDF + cosine over JSONL files), no teacher calls.
    """
    from qwen_tutor.generation.dedup_seeds import dedup_seeds_dir

    dcfg = cfg.get("dedup_seeds", {}) or {}
    if not dcfg.get("enabled", False):
        print("\n=== dedup_seeds (skipped, dedup_seeds.enabled=false) ===")
        return
    sim_threshold = float(dcfg.get("similarity_threshold", 0.75))
    raw_cap = dcfg.get("max_per_cell")
    max_per_cell = int(raw_cap) if raw_cap is not None else None
    backup = bool(dcfg.get("backup", True))
    print(
        f"\n=== dedup_seeds (threshold={sim_threshold}, "
        f"max_per_cell={max_per_cell}, backup={backup}) ==="
    )
    stats = dedup_seeds_dir(
        seeds_dir=Path(paths["seeds_dir"]),
        levels=params["levels"],
        similarity_threshold=sim_threshold,
        max_per_cell=max_per_cell,
        backup=backup,
    )
    print(
        f"  total: input={stats['input']} kept={stats['kept']} "
        f"dropped={stats['dropped']}"
    )
    for f in stats["files"]:
        if f["input"] == 0:
            continue
        ratio = f["kept"] / f["input"]
        print(f"  {Path(f['file']).name}: {f['input']} -> {f['kept']} ({ratio:.0%})")
    if stats["dropped"]:
        print(
            f"  dropped seeds written to {paths['seeds_dir']}/_dropped/ for inspection"
        )


# ---------------------------------------------------------------------------
# Top-up: post-filter category rebalancing
# ---------------------------------------------------------------------------


async def _stage_top_up(
    cfg: dict[str, Any], cfg_path: Path, params: dict[str, Any], paths: dict[str, str]
) -> None:
    """Generate extra seeds for under-represented (level, locale, category)
    triples after filter_sft / filter_eval / filter_dpo have finished.

    Single-shot: computes deficits across all three filtered pools, generates
    new seeds biased toward the deficit categories, and stops. The caller is
    expected to re-invoke ``run_generation.py`` (or include the downstream
    stages in ``--stages``) so the new seeds flow through sft / redirects /
    register / eval / filter_*. Top-up is intentionally NOT iterative —
    a category that keeps failing filtering needs a root-cause fix.
    """
    from qwen_tutor.generation.categories import load_category_names
    from qwen_tutor.generation.top_up import compute_deficits, log_deficit_report

    top_cfg = cfg.get("top_up", {})
    if not top_cfg.get("enabled", False):
        print("\n=== top_up (skipped, top_up.enabled=false) ===")
        return

    target_per_category = int(top_cfg.get("target_per_category", 5))
    max_top_up_per_category = int(top_cfg.get("max_top_up_per_category", 10))

    print(
        f"\n=== top_up (target={target_per_category}/category, "
        f"cap={max_top_up_per_category}) ==="
    )
    quotas, report = compute_deficits(
        sft_dir=Path(paths["sft_filtered"]),
        eval_dir=Path(paths["eval_filtered"]),
        dpo_dir=Path(paths["dpo_filtered"]),
        target_per_category=target_per_category,
        levels=params["levels"],
        locales=params["locales"],
        max_top_up_per_category=max_top_up_per_category,
        categories=load_category_names(str(cfg_path)),
    )
    log_deficit_report(report)

    total = report["total_seeds_to_generate"]
    if total == 0:
        print("  no deficits — top_up is a no-op")
        return

    print(f"  generating {total} new seeds across deficit categories...")
    res = await seeds_mod.generate_batch(
        # n_per_level is ignored when category_quotas is provided, but the
        # signature still requires it; pass 0 to be explicit.
        n_per_level=0,
        cefr_levels=params["levels"],
        locales=params["locales"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        per_call_size=params["per_call_size"],
        output_dir=paths["seeds_dir"],
        category_quotas=quotas,
        subtopic_target=params["subtopic_target"],
    )
    total_written = sum(res.values()) if isinstance(res, dict) else 0
    print(f"  top-up seeds written this run: {res}")

    auto_rerun = bool(top_cfg.get("auto_rerun_downstream", False))
    if total_written == 0:
        return
    if not auto_rerun:
        print(
            "  next step: re-run downstream stages so the new seeds flow through.\n"
            "  example: python scripts/run_generation.py "
            "--stages sft,redirect,locale_redirect,pedagogy_redirect,"
            "language_redirect,persona_redirect,topic_redirect,role_swap_redirect,"
            "register,eval,filter_sft,filter_eval,filter_dpo\n"
            "  OR set top_up.auto_rerun_downstream: true in generation.yaml "
            "to let run_generation.py do this automatically."
        )
        return

    # auto_rerun_downstream: process the new seeds end-to-end without
    # forcing the user to re-invoke run_generation.py manually. Each
    # downstream stage is resumable (skips IDs already on disk), so this
    # only generates the records derived from the new top-up seeds.
    print(
        f"\n=== top_up: auto-rerun downstream "
        f"({total_written} new seeds → sft/redirects/register/eval/filter_*) ==="
    )
    await _stage_sft(cfg_path, params, paths)
    await _stage_redirect(cfg_path, params, paths)
    await _stage_locale_redirect(cfg_path, params, paths)
    await _stage_pedagogy_redirect(cfg_path, params, paths)
    await _stage_language_redirect(cfg_path, params, paths)
    await _stage_persona_redirect(cfg_path, params, paths)
    await _stage_topic_redirect(cfg_path, params, paths)
    await _stage_role_swap_redirect(cfg_path, params, paths)
    await _stage_persistent_off_topic(cfg_path, params, paths)
    await _stage_persistent_language_violation(cfg_path, params, paths)
    await _stage_persistent_persona_break(cfg_path, params, paths)
    await _stage_persistent_role_swap(cfg_path, params, paths)
    await _stage_register(cfg_path, params, paths)
    await _stage_eval(cfg_path, params, paths)
    await _stage_filter_sft(cfg, cfg_path, params, paths)
    await _stage_filter_eval(cfg, cfg_path, params, paths)
    await _stage_filter_dpo(cfg, cfg_path, params, paths)
    print("\n=== top_up auto-rerun complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="qwen-en-tutor data generation and filtering pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config/generation.yaml")
    p.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help="Stages to run (comma-separated). Default: all.",
    )
    p.add_argument(
        "--levels",
        default=None,
        help="CEFR levels (comma-separated). Default: generation.cefr_levels in YAML.",
    )
    p.add_argument(
        "--locales",
        default=None,
        help="Locale names to generate (comma-separated, e.g. china,japan,italy). Default: generation.locales in YAML or default_locale in config/locale.yaml.",
    )
    p.add_argument(
        "--n-per-level",
        type=int,
        default=None,
        help="Number of seeds per level. Default: generation.n_per_level in YAML.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Concurrency level. Default: generation.concurrency in YAML.",
    )
    p.add_argument(
        "--angle-shift-fraction",
        type=float,
        default=None,
        help="Fraction of normal SFT variants where the learner approaches with a valid angle that differs from user_role.description. 0.0 = disabled (default). > 0 adds angle-shift normal dialogues for that fraction of seeds.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _validate_stages(stages: list[str]) -> list[str]:
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        raise SystemExit(f"Invalid stage(s): {bad!r}. Valid values: {list(ALL_STAGES)}")
    return stages


async def _main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")
    cfg = _load_config(cfg_path)
    paths = cfg.get("paths", {})
    params = _resolve(args, cfg)
    stages = _validate_stages([s.strip() for s in args.stages.split(",") if s.strip()])

    # Map each stage to its handler
    gen_handlers = {
        "seeds":              lambda: _stage_seeds(cfg_path, params, paths),
        "sft":                lambda: _stage_sft(cfg_path, params, paths),
        "redirect":           lambda: _stage_redirect(cfg_path, params, paths),
        "locale_redirect":    lambda: _stage_locale_redirect(cfg_path, params, paths),
        "pedagogy_redirect":  lambda: _stage_pedagogy_redirect(cfg_path, params, paths),
        "language_redirect":  lambda: _stage_language_redirect(cfg_path, params, paths),
        "persona_redirect":   lambda: _stage_persona_redirect(cfg_path, params, paths),
        "topic_redirect":     lambda: _stage_topic_redirect(cfg_path, params, paths),
        "role_swap_redirect": lambda: _stage_role_swap_redirect(cfg_path, params, paths),
        "asr_repair":         lambda: _stage_asr_repair(cfg_path, params, paths),
        "country_taboo":      lambda: _stage_country_taboo(cfg_path, params, paths),
        "persistent_off_topic":          lambda: _stage_persistent_off_topic(cfg_path, params, paths),
        "persistent_language_violation": lambda: _stage_persistent_language_violation(cfg_path, params, paths),
        "persistent_persona_break":      lambda: _stage_persistent_persona_break(cfg_path, params, paths),
        "persistent_role_swap":          lambda: _stage_persistent_role_swap(cfg_path, params, paths),
        "register":           lambda: _stage_register(cfg_path, params, paths),
        "eval":               lambda: _stage_eval(cfg_path, params, paths),
    }
    filter_handlers = {
        "dedup_seeds": lambda: _stage_dedup_seeds(cfg, cfg_path, params, paths),
        "filter_sft":  lambda: _stage_filter_sft(cfg, cfg_path, params, paths),
        "filter_eval": lambda: _stage_filter_eval(cfg, cfg_path, params, paths),
        "filter_dpo":  lambda: _stage_filter_dpo(cfg, cfg_path, params, paths),
        "top_up":      lambda: _stage_top_up(cfg, cfg_path, params, paths),
        "sft_topup":   lambda: _stage_sft_topup(cfg, cfg_path, params, paths),
        "persistent_topup": lambda: _stage_persistent_topup(cfg, cfg_path, params, paths),
    }

    for stage in stages:
        if stage in gen_handlers:
            await gen_handlers[stage]()
        elif stage in filter_handlers:
            await filter_handlers[stage]()

    print("\n=== generation pipeline complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
