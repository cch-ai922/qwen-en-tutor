"""Forbidden-country hard-refusal SFT generation (special refusal axis).

Special axis defined in ``config/taboo_country.yaml``: the tutor must never
mention ONE specific real country or anything associated with it (name,
nationality/adjective, cities, people, brands, history, events). When the
learner brings it up, the tutor refuses to name/confirm/discuss/compare it,
gives a brief non-committal deflection, and pivots to safe neutral ground --
WITHOUT naming the country anywhere in its own turn.

The learner turn may name the country (training masks user turns and the
mechanical filters skip them for ``scenario_type="redirect"``); the TUTOR turn
must not -- the ``taboo_country`` category in ``config/banned_terms.yaml``
mechanically rejects any example whose assistant turn leaks it.

Disabled by default: ``generate_batch`` is a no-op unless
``config/taboo_country.yaml`` has ``enabled: true`` with a real ``country``.

Writes to ``data/sft_raw/country_taboo_{level}.jsonl``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from qwen_tutor.generation._prompt_select import (
    render_scenario_deployment_system_prompt,
    render_level_spec,
    render_prompt,
    validate_prompt_has_locale_instruction,
)
from qwen_tutor.generation.prompts import taboo_country_enabled, taboo_country_fields
from qwen_tutor.generation.seeds import iter_seeds
from qwen_tutor.generation.teacher import TeacherClient, build_teacher_from_config
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
DEFAULT_FAILURES_PATH = Path("data/country_taboo_failures.jsonl")


def _country_taboo_id(seed_id: str, variant: int = 0) -> str:
    if variant == 0:
        return f"country_taboo_{seed_id}"
    return f"country_taboo_{seed_id}_v{variant}"


def _parse_messages(raw: str) -> list[Message]:
    data = extract_first_json(raw)
    if not isinstance(data, dict) or "messages" not in data:
        raise ValueError(
            "country_taboo dialogue response missing top-level 'messages' field"
        )
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
            f"country_taboo dialogue too short ({len(messages)} turns, "
            f"need >= {MIN_TURNS})"
        )
    return messages


async def _generate_one(
    teacher: TeacherClient,
    seed_id: str,
    seed: ScenarioSeed,
    taboo: dict[str, str],
    level_spec: str,
    max_tokens: int,
    temperature: float,
    failures_path: Path,
    generation_meta_base: dict[str, Any],
    variant: int = 0,
) -> SFTExample | None:
    locale = seed.locale
    scenario_json = json.dumps(seed.model_dump(), ensure_ascii=False)
    template = render_prompt("dialogue_country_taboo", locale_name=locale)
    prompt = template.format(
        scenario_json=scenario_json,
        level=seed.cefr_level,
        level_spec_with_locale_instruction=level_spec,
        taboo_country=taboo["country"],
        taboo_country_adjective=taboo["country_adjective"],
        taboo_pivot_hint=taboo["pivot_hint"],
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
            _country_taboo_id(seed_id, variant),
            f"{type(exc).__name__}: {exc}",
            level=seed.cefr_level,
            stage="country_taboo",
            seed_id=seed_id,
            variant=variant,
        )
        return None

    metadata = ExampleMetadata(
        topic=seed.topic,
        subtopics=list(seed.subtopics),
        user_role=seed.user_role,
        model_role=seed.model_role,
        cefr_level=seed.cefr_level,
        scenario_type="redirect",  # let banned_terms/non_latin filters skip the user turn
        locale=locale,
        category=seed.category,
        generation={
            **generation_meta_base,
            "taboo_country": taboo["country"],
            "variant": variant,
        },
    )
    return SFTExample(
        id=_country_taboo_id(seed_id, variant),
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
    teacher: TeacherClient | None = None,
    country_taboo_fraction: float = 0.0,
    dialogues_per_seed: int = 1,
) -> dict[str, int]:
    """Generate forbidden-country refusal dialogues for a fraction of seeds.

    No-op (returns {} with zeros) unless ``config/taboo_country.yaml`` is
    enabled with a real country. ``country_taboo_fraction`` is the portion of
    the seed pool turned into refusal variants.

    Returns ``{level: n_written}`` for this run.
    """
    if not taboo_country_enabled():
        logger.info(
            "[country_taboo] disabled (config/taboo_country.yaml enabled:false "
            "or country unset) -- skipping"
        )
        return {level: 0 for level in (cefr_levels or [])}

    taboo = taboo_country_fields()
    if teacher is None:
        teacher = build_teacher_from_config(config_path, role="teacher")
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)

    generation_meta_base = {
        "provider": teacher.config.provider,
        "model": teacher.config.model,
        "prompt": "DIALOGUE_PROMPT_COUNTRY_TABOO",
    }

    level_spec_cache: dict[tuple[str, str], str] = {}

    def _level_spec(level: str, locale: str) -> str:
        key = (level, locale)
        if key not in level_spec_cache:
            level_spec_cache[key] = render_level_spec(level, locale_name=locale)
        return level_spec_cache[key]

    results: dict[str, int] = {}
    for level in cefr_levels:
        out_path = output_dir / f"country_taboo_{level}.jsonl"
        done_ids = load_existing_ids(out_path)

        all_seeds = list(iter_seeds(seeds_dir, [level]))
        selected_seeds = deterministic_sample(
            all_seeds, country_taboo_fraction, key=lambda s: f"country_taboo::{s[0]}"
        )
        logger.info(
            "[country_taboo:%s] fraction=%.2f -> %d/%d seeds selected (x%d variants)",
            level, country_taboo_fraction, len(selected_seeds), len(all_seeds),
            dialogues_per_seed,
        )

        pending: list[tuple[str, ScenarioSeed, int]] = []
        for sid, seed in selected_seeds:
            for variant in range(dialogues_per_seed):
                if _country_taboo_id(sid, variant) in done_ids:
                    continue
                pending.append((sid, seed, variant))

        if not pending:
            logger.info("[country_taboo:%s] nothing to do (%d done)", level, len(done_ids))
            results[level] = 0
            continue

        async def _run(item: tuple[str, ScenarioSeed, int]) -> SFTExample | None:
            sid, seed, variant = item
            return await _generate_one(
                teacher=teacher,
                seed_id=sid,
                seed=seed,
                taboo=taboo,
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
            desc=f"country_taboo[{level}]",
        )

        written = 0
        for ex in completed:
            if ex is None:
                continue
            append_jsonl(out_path, ex)
            written += 1
        results[level] = written
        logger.info("[country_taboo:%s] wrote %d new examples", level, written)
    return results
