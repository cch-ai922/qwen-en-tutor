"""Redirect-scenario SFT generation.

For each scenario seed, calls DIALOGUE_PROMPT_REDIRECT with a cycling
``redirect_axis`` so the resulting SFT corpus contains balanced coverage
across the redirect axes the user is expected to brush up against.

Tags ``scenario_type="redirect"`` in the metadata. The deployment system
prompt is rendered into ``system_prompt`` just like the normal SFT
stage, so the deployed model sees the same system context whether the
dialogue contains a redirect moment or not. Writes to
``data/sft_raw/redirect_{level}.jsonl``.
"""

from __future__ import annotations

import json
import logging
from itertools import cycle
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from qwen_tutor.generation._prompt_select import (
    render_scenario_deployment_system_prompt,
    render_level_spec,
    render_prompt,
    validate_prompt_has_locale_instruction,
)
from qwen_tutor.generation.seeds import iter_seeds
from qwen_tutor.generation.teacher import TeacherClient, build_teacher_from_config
from qwen_tutor.locale import LOCALE, get_locale
from qwen_tutor.schemas import ExampleMetadata, Message, ScenarioSeed, SFTExample
from qwen_tutor.utils.runner import (
    append_failure,
    append_jsonl,
    deterministic_sample,
    extract_first_json,
    gather_with_concurrency,
    load_existing_ids,
)

logger = logging.getLogger(__name__)

DEFAULT_SEEDS_DIR = Path("data/seeds")
DEFAULT_OUTPUT_DIR = Path("data/sft_raw")
DEFAULT_FAILURES_PATH = Path("data/redirect_failures.jsonl")

# REDIRECT_AXES is based on the default locale's avoided_topics.
# In multi-locale operation, each seed pulls its own locale directly, so this
# constant is only a backward-compatible fallback.
REDIRECT_AXES: tuple[str, ...] = LOCALE.avoided_topic_names


def _redirect_id(seed_id: str, axis: str, variant: int = 0) -> str:
    """Redirect example id. If variant > 0, it is another sample for the same seed+axis."""
    if variant == 0:
        return f"redirect_{axis}_{seed_id}"
    return f"redirect_{axis}_{seed_id}_v{variant}"


def _parse_messages(raw: str) -> list[Message]:
    data = extract_first_json(raw)
    if not isinstance(data, dict) or "messages" not in data:
        raise ValueError("redirect dialogue response missing top-level 'messages' field")
    msgs_raw = data["messages"]
    if not isinstance(msgs_raw, list):
        raise ValueError("'messages' is not a list")
    messages: list[Message] = []
    for m in msgs_raw:
        if not isinstance(m, dict):
            continue
        try:
            messages.append(Message.model_validate(m))
        except ValidationError as exc:
            logger.warning("dropping malformed message: %s", exc)
    from qwen_tutor.generation.prompts import MIN_TURNS

    if len(messages) < MIN_TURNS:
        raise ValueError(
            f"redirect dialogue too short ({len(messages)} turns, "
            f"need >= {MIN_TURNS})"
        )
    return messages


async def _generate_one(
    teacher: TeacherClient,
    seed_id: str,
    seed: ScenarioSeed,
    axis: str,
    level_spec: str,
    max_tokens: int,
    temperature: float,
    failures_path: Path,
    generation_meta_base: dict[str, Any],
    variant: int = 0,
) -> SFTExample | None:
    locale = seed.locale
    scenario_json = json.dumps(seed.model_dump(), ensure_ascii=False)
    template = render_prompt("dialogue_redirect", locale_name=locale)
    prompt = template.format(
        scenario_json=scenario_json,
        level=seed.cefr_level,
        level_spec_with_locale_instruction=level_spec,
        redirect_axis=axis,
    )
    validate_prompt_has_locale_instruction(prompt, locale_name=locale)
    try:
        raw = await teacher.generate(
            system=prompt,
            messages=[],
            cacheable_prefix=level_spec,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        messages = _parse_messages(raw)
    except Exception as exc:  # noqa: BLE001
        append_failure(
            failures_path,
            _redirect_id(seed_id, axis, variant),
            f"{type(exc).__name__}: {exc}",
            level=seed.cefr_level,
            stage="redirect",
            seed_id=seed_id,
            redirect_axis=axis,
            variant=variant,
        )
        return None

    metadata = ExampleMetadata(
        topic=seed.topic,
        subtopics=list(seed.subtopics),
        user_role=seed.user_role,
        model_role=seed.model_role,
        cefr_level=seed.cefr_level,
        scenario_type="redirect",
        locale=locale,
        category=seed.category,
        generation={**generation_meta_base, "redirect_axis": axis, "variant": variant},
    )
    return SFTExample(
        id=_redirect_id(seed_id, axis, variant),
        metadata=metadata,
        system_prompt=render_scenario_deployment_system_prompt(
            cefr_level=seed.cefr_level,
            locale_name=locale,
            topic=seed.topic,
            subtopics=seed.subtopics,
            user_role_name=seed.user_role.name,
            user_role_description=seed.user_role.description,
            model_role_name=seed.model_role.name,
            model_role_description=seed.model_role.description,
        ),
        messages=messages,
    )


async def generate_batch(
    cefr_levels: list[str] | None = None,
    seeds_dir: str | Path = DEFAULT_SEEDS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    failures_path: str | Path = DEFAULT_FAILURES_PATH,
    config_path: str | Path = "config/generation.yaml",
    concurrency: int = 20,
    max_tokens: int = 4096,
    temperature: float = 0.8,
    axes: tuple[str, ...] | None = None,
    teacher: TeacherClient | None = None,
    redirect_fraction: float = 0.20,
    dialogues_per_seed: int = 1,
) -> dict[str, int]:
    """Generate redirect dialogues for a fraction of seeds, cycling axes.

    ``redirect_fraction`` is the fraction of seeds that should produce a
    redirect variant (0.0–1.0).
    If ``dialogues_per_seed`` is 2 or more, the same (seed, axis) combination
    produces that many variants. Axes are assigned cyclically across the full
    set of (seed, variant) pairs so the axis distribution stays even.

    Returns ``{level: n_written}`` for this run.
    """
    if teacher is None:
        teacher = build_teacher_from_config(config_path, role="teacher")
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)

    generation_meta_base = {
        "provider": teacher.config.provider,
        "model": teacher.config.model,
        "prompt": "DIALOGUE_PROMPT_REDIRECT",
    }

    # Per-(level, locale) level_spec cache.
    level_spec_cache: dict[tuple[str, str], str] = {}

    def _level_spec(level: str, locale: str) -> str:
        key = (level, locale)
        if key not in level_spec_cache:
            level_spec_cache[key] = render_level_spec(level, locale_name=locale)
        return level_spec_cache[key]

    results: dict[str, int] = {}
    for level in cefr_levels:
        out_path = output_dir / f"redirect_{level}.jsonl"
        done_ids = load_existing_ids(out_path)

        # Select exactly redirect_fraction of the seed pool for each level,
        # using a deterministic sample.
        all_seeds = list(iter_seeds(seeds_dir, [level]))
        selected_seeds = deterministic_sample(
            all_seeds, redirect_fraction, key=lambda s: s[0]
        )
        logger.info(
            "[redirect:%s] fraction=%.2f → %d/%d seeds selected (x%d variants)",
            level, redirect_fraction, len(selected_seeds), len(all_seeds),
            dialogues_per_seed,
        )

        # Group selected seeds by locale so each locale gets its own axis
        # cycle from THAT locale's avoided_topics. This keeps the axis
        # distribution balanced within each locale.
        seeds_by_locale: dict[str, list[tuple[str, ScenarioSeed]]] = {}
        for sid, seed in selected_seeds:
            seeds_by_locale.setdefault(seed.locale, []).append((sid, seed))

        pending: list[tuple[str, ScenarioSeed, str, int]] = []
        for loc_name, seed_list in seeds_by_locale.items():
            axes_for_loc = (
                tuple(axes) if axes else get_locale(loc_name).avoided_topic_names
            )
            axis_cycle = cycle(axes_for_loc)
            for sid, seed in seed_list:
                for variant in range(dialogues_per_seed):
                    axis = next(axis_cycle)
                    if _redirect_id(sid, axis, variant) in done_ids:
                        continue
                    pending.append((sid, seed, axis, variant))

        if not pending:
            logger.info("[redirect:%s] nothing to do (%d done)", level, len(done_ids))
            results[level] = 0
            continue

        async def _run(item: tuple[str, ScenarioSeed, str, int]) -> SFTExample | None:
            sid, seed, axis, variant = item
            return await _generate_one(
                teacher=teacher,
                seed_id=sid,
                seed=seed,
                axis=axis,
                level_spec=_level_spec(seed.cefr_level, seed.locale),
                max_tokens=max_tokens,
                temperature=temperature,
                failures_path=failures_path,
                generation_meta_base=generation_meta_base,
                variant=variant,
            )

        completed = await gather_with_concurrency(
            [_run(item) for item in pending],
            concurrency=concurrency,
            desc=f"redirect[{level}]",
        )

        written = 0
        for ex in completed:
            if ex is None:
                continue
            append_jsonl(out_path, ex)
            written += 1
        results[level] = written
        logger.info("[redirect:%s] wrote %d new examples", level, written)
    return results
