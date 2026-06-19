"""Persona-break user-side handler SFT generation.

For a fraction of scenario seeds, generate a multi-turn dialogue in which
the LEARNER (user) tests the tutor's persona in one of two ways and the
TUTOR (assistant) deflects in character without breaking the conversational
fiction:

  - ``asks_if_human``  : user asks if the tutor is a real person /
                         chatbot / AI ("Are you a real person?",
                         "Are you a chatbot?").
  - ``asks_about_model``: user asks meta-questions about origin /
                         capabilities ("Who made you?", "What model
                         are you?", "Are you ChatGPT?").

This is the user-side counterpart to ``register_pairs.py`` axis
``persona_break`` (assistant-side, where the tutor itself leaks AI
scaffolding like "As an AI..."). Together they teach the model both
"don't leak AI persona yourself" (DPO) and "stay in character when the
user probes" (SFT).

Cycles through ``PERSONA_TRIGGER_KINDS`` deterministically per seed.
``scenario_type="redirect"`` on the emitted SFTExample.
Writes to ``data/sft_raw/persona_redirect_{level}.jsonl``.
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
DEFAULT_FAILURES_PATH = Path("data/persona_redirect_failures.jsonl")

# Two user-side persona-probe patterns.
PERSONA_TRIGGER_KINDS: tuple[str, ...] = ("asks_if_human", "asks_about_model")


def _persona_redirect_id(seed_id: str, trigger: str, variant: int = 0) -> str:
    if variant == 0:
        return f"persona_redirect_{trigger}_{seed_id}"
    return f"persona_redirect_{trigger}_{seed_id}_v{variant}"


def _parse_messages(raw: str) -> list[Message]:
    data = extract_first_json(raw)
    if not isinstance(data, dict) or "messages" not in data:
        raise ValueError(
            "persona_redirect dialogue response missing top-level 'messages' field"
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
            f"persona_redirect dialogue too short ({len(messages)} turns, "
            f"need >= {MIN_TURNS})"
        )
    return messages


async def _generate_one(
    teacher: TeacherClient,
    seed_id: str,
    seed: ScenarioSeed,
    trigger: str,
    level_spec: str,
    max_tokens: int,
    temperature: float,
    failures_path: Path,
    generation_meta_base: dict[str, Any],
    variant: int = 0,
) -> SFTExample | None:
    locale = seed.locale
    scenario_json = json.dumps(seed.model_dump(), ensure_ascii=False)
    template = render_prompt("dialogue_persona_redirect", locale_name=locale)
    prompt = template.format(
        scenario_json=scenario_json,
        level=seed.cefr_level,
        level_spec_with_locale_instruction=level_spec,
        persona_trigger=trigger,
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
        # Extract violation_turn_idx from the same JSON the teacher emitted
        # (prompts now require it as a top-level field).
        _vti_data = extract_first_json(raw)
        violation_turn_idx: int | None = None
        if isinstance(_vti_data, dict):
            _vti_val = _vti_data.get("violation_turn_idx")
            if isinstance(_vti_val, (int, float)):
                try:
                    violation_turn_idx = int(_vti_val)
                except (TypeError, ValueError):
                    violation_turn_idx = None
    except Exception as exc:  # noqa: BLE001
        append_failure(
            failures_path,
            _persona_redirect_id(seed_id, trigger, variant),
            f"{type(exc).__name__}: {exc}",
            level=seed.cefr_level,
            stage="persona_redirect",
            seed_id=seed_id,
            persona_trigger=trigger,
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
        generation={
            **generation_meta_base,
            "persona_trigger": trigger,
            "variant": variant,
            "violation_turn_idx": violation_turn_idx,
        },
    )
    return SFTExample(
        id=_persona_redirect_id(seed_id, trigger, variant),
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
    triggers: tuple[str, ...] | None = None,
    teacher: TeacherClient | None = None,
    persona_redirect_fraction: float = 0.15,
    dialogues_per_seed: int = 1,
) -> dict[str, int]:
    """Generate persona-redirect dialogues for a fraction of seeds.

    Returns ``{level: n_written}`` for this run.
    """
    if teacher is None:
        teacher = build_teacher_from_config(config_path, role="teacher")
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    triggers = triggers or PERSONA_TRIGGER_KINDS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)

    generation_meta_base = {
        "provider": teacher.config.provider,
        "model": teacher.config.model,
        "prompt": "DIALOGUE_PROMPT_PERSONA_REDIRECT",
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
        out_path = output_dir / f"persona_redirect_{level}.jsonl"
        done_ids = load_existing_ids(out_path)

        all_seeds = list(iter_seeds(seeds_dir, [level]))
        selected_seeds = deterministic_sample(
            all_seeds, persona_redirect_fraction, key=lambda s: s[0]
        )
        logger.info(
            "[persona_redirect:%s] fraction=%.2f -> %d/%d seeds selected (x%d variants)",
            level, persona_redirect_fraction,
            len(selected_seeds), len(all_seeds), dialogues_per_seed,
        )

        trigger_cycle = cycle(triggers)
        pending: list[tuple[str, ScenarioSeed, str, int]] = []
        for sid, seed in selected_seeds:
            for variant in range(dialogues_per_seed):
                trigger = next(trigger_cycle)
                if _persona_redirect_id(sid, trigger, variant) in done_ids:
                    continue
                pending.append((sid, seed, trigger, variant))

        if not pending:
            logger.info(
                "[persona_redirect:%s] nothing to do (%d done)",
                level, len(done_ids),
            )
            results[level] = 0
            continue

        async def _run(item: tuple[str, ScenarioSeed, str, int]) -> SFTExample | None:
            sid, seed, trigger, variant = item
            return await _generate_one(
                teacher=teacher,
                seed_id=sid,
                seed=seed,
                trigger=trigger,
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
            desc=f"persona_redirect[{level}]",
        )

        written = 0
        for ex in completed:
            if ex is None:
                continue
            append_jsonl(out_path, ex)
            written += 1
        results[level] = written
        logger.info("[persona_redirect:%s] wrote %d new examples", level, written)
    return results
