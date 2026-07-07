"""SFT dialogue generation (normal, non-redirect scenarios).

For each scenario seed, calls DIALOGUE_PROMPT_NORMAL on the teacher
model and parses the result into an SFTExample. The deployment system
prompt (the prompt the deployed tutor model will see at inference) is
baked into ``SFTExample.system_prompt`` so training and inference share
identical system context.

Resumes from checkpoint: skips any scenario whose target SFTExample id
is already present in the output file. Writes to
``data/sft_raw/normal_{level}.jsonl``.
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
DEFAULT_FAILURES_PATH = Path("data/sft_dialogues_failures.jsonl")


def _sft_id(seed_id: str, variant: int = 0) -> str:
    """SFT example id. If ``variant > 0``, this is another sample for the same seed."""
    if variant == 0:
        return f"sft_{seed_id}"
    return f"sft_{seed_id}_v{variant}"


def _sft_angle_shift_id(seed_id: str, variant: int = 0) -> str:
    """Angle-shift normal SFT variant id.

    A normal dialogue whose learner view is a valid angle different from
    ``user_role.description``. It still uses ``scenario_type="normal"`` and
    is stored alongside normal dialogues in ``normal_<level>.jsonl``, but the
    different id prefix keeps resume behavior safe.
    """
    if variant == 0:
        return f"sft_angle_{seed_id}"
    return f"sft_angle_{seed_id}_v{variant}"


def _sft_passive_learner_id(seed_id: str, variant: int = 0) -> str:
    """Passive-learner normal SFT variant id.

    A normal dialogue where the learner is passive/minimal and the tutor
    proactively leads. Same ``scenario_type="normal"``, stored alongside
    normal dialogues in ``normal_<level>.jsonl``; the distinct id prefix
    keeps resume behavior safe.
    """
    if variant == 0:
        return f"sft_passive_{seed_id}"
    return f"sft_passive_{seed_id}_v{variant}"


def _parse_messages(raw: str) -> list[Message]:
    data = extract_first_json(raw)
    if not isinstance(data, dict) or "messages" not in data:
        raise ValueError("dialogue response missing top-level 'messages' field")
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
            f"dialogue too short ({len(messages)} turns, need >= {MIN_TURNS})"
        )
    return messages


async def _generate_one(
    teacher: TeacherClient,
    seed_id: str,
    seed: ScenarioSeed,
    level_spec: str,
    max_tokens: int,
    temperature: float,
    failures_path: Path,
    generation_meta: dict[str, Any],
    variant: int = 0,
    is_angle_shift: bool = False,
    is_passive_learner: bool = False,
) -> SFTExample | None:
    """Generate a normal SFT dialogue. ``is_angle_shift=True`` swaps in the
    angle-shift prompt variant, which asks the teacher to write the learner
    from a different valid angle than ``user_role.description`` (same topic,
    same model_role, no redirect). ``is_passive_learner=True`` swaps in the
    passive-learner prompt variant, where the learner is minimal/stuck and the
    tutor proactively leads the conversation. Output ``scenario_type`` stays
    ``"normal"`` for both -- these dialogues belong in the normal SFT stream,
    just with a broader learner-behavior distribution. The two flags are
    mutually exclusive; ``is_passive_learner`` takes precedence if both set.
    """
    locale = seed.locale
    scenario_json = json.dumps(seed.model_dump(), ensure_ascii=False)
    if is_passive_learner:
        prompt_name = "dialogue_normal_passive_learner"
    elif is_angle_shift:
        prompt_name = "dialogue_normal_angle_shift"
    else:
        prompt_name = "dialogue_normal"
    template = render_prompt(prompt_name, locale_name=locale)
    prompt = template.format(
        scenario_json=scenario_json,
        level=seed.cefr_level,
        level_spec_with_locale_instruction=level_spec,
    )
    validate_prompt_has_locale_instruction(prompt, locale_name=locale)
    if is_passive_learner:
        id_fn = _sft_passive_learner_id
        stage_name = "sft_normal_passive_learner"
    elif is_angle_shift:
        id_fn = _sft_angle_shift_id
        stage_name = "sft_normal_angle_shift"
    else:
        id_fn = _sft_id
        stage_name = "sft_normal"
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
            id_fn(seed_id, variant),
            f"{type(exc).__name__}: {exc}",
            level=seed.cefr_level,
            stage=stage_name,
            seed_id=seed_id,
            variant=variant,
            angle_shift=is_angle_shift,
            passive_learner=is_passive_learner,
        )
        return None

    if is_passive_learner:
        meta_prompt_name = "DIALOGUE_PROMPT_NORMAL_PASSIVE_LEARNER"
    elif is_angle_shift:
        meta_prompt_name = "DIALOGUE_PROMPT_NORMAL_ANGLE_SHIFT"
    else:
        meta_prompt_name = generation_meta.get("prompt", "DIALOGUE_PROMPT_NORMAL")
    metadata = ExampleMetadata(
        topic=seed.topic,
        subtopics=list(seed.subtopics),
        user_role=seed.user_role,
        model_role=seed.model_role,
        cefr_level=seed.cefr_level,
        scenario_type="normal",
        locale=locale,
        category=seed.category,
        generation={
            **generation_meta,
            "prompt": meta_prompt_name,
            "variant": variant,
            "angle_shift": is_angle_shift,
            "passive_learner": is_passive_learner,
        },
    )
    return SFTExample(
        id=id_fn(seed_id, variant),
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
    dialogues_per_seed: int = 1,
    angle_shift_fraction: float = 0.0,
    passive_learner_fraction: float = 0.0,
) -> dict[str, int]:
    """Generate normal SFT dialogues for every seed under ``seeds_dir``.

    If ``dialogues_per_seed`` is 1, that is the standard one-per-seed behavior.
    If it is 2 or higher, ``_generate_one`` is called that many times for the
    same seed to create variants. Each variant id is unique in the form
    ``sft_{seed_id}_v{variant}``.

    If ``angle_shift_fraction`` > 0, a deterministic fraction of seeds is
    selected in addition to normal generation, and angle-shift variants are
    generated. Angle-shift variants write a user-side turn with a valid angle
    different from ``user_role.description``, while the tutor still responds
    normally in-character (no redirect). This is optional data that teaches
    the model that ``[learner]`` is a soft hint. Default is 0.0 (disabled).
    The id prefix is ``sft_angle_``, so these variants can be stored in the
    same ``normal_<level>.jsonl`` file without conflicting.

    If ``passive_learner_fraction`` > 0, a deterministic fraction of seeds
    additionally get a passive-learner variant: the learner is minimal/stuck
    and the tutor proactively leads the conversation so it never stalls.
    scenario_type stays "normal"; the id prefix is ``sft_passive_``. Default
    0.0 (disabled).

    Returns ``{level: n_written}`` for this run.
    """
    if teacher is None:
        teacher = build_teacher_from_config(config_path, role="teacher")
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)

    generation_meta = {
        "provider": teacher.config.provider,
        "model": teacher.config.model,
        "prompt": "DIALOGUE_PROMPT_NORMAL",
    }

    # Per-(level, locale) level_spec cache so we don't re-read cefr_specs.yaml
    # for every pending seed.
    level_spec_cache: dict[tuple[str, str], str] = {}

    def _level_spec(level: str, locale: str) -> str:
        key = (level, locale)
        if key not in level_spec_cache:
            level_spec_cache[key] = render_level_spec(level, locale_name=locale)
        return level_spec_cache[key]

    results: dict[str, int] = {}
    for level in cefr_levels:
        out_path = output_dir / f"normal_{level}.jsonl"
        done_ids = load_existing_ids(out_path)
        all_seeds = list(iter_seeds(seeds_dir, [level]))

        # 1) standard normal variants.
        #    Tuple: (seed_id, seed, variant, is_angle_shift, is_passive_learner)
        pending: list[tuple[str, ScenarioSeed, int, bool, bool]] = []
        for sid, seed in all_seeds:
            for variant in range(dialogues_per_seed):
                if _sft_id(sid, variant) in done_ids:
                    continue
                pending.append((sid, seed, variant, False, False))

        # 2) optional angle-shift variants. Deterministic subset of seeds so
        #    re-runs hit the same selection. One angle-shift variant per
        #    selected seed (independent of dialogues_per_seed).
        if angle_shift_fraction > 0.0 and all_seeds:
            angle_seeds = deterministic_sample(
                all_seeds, angle_shift_fraction, key=lambda s: s[0]
            )
            logger.info(
                "[sft:%s] angle_shift_fraction=%.2f -> %d/%d additional angle-shift variants",
                level, angle_shift_fraction, len(angle_seeds), len(all_seeds),
            )
            for sid, seed in angle_seeds:
                if _sft_angle_shift_id(sid, 0) in done_ids:
                    continue
                pending.append((sid, seed, 0, True, False))

        # 3) optional passive-learner variants. Deterministic subset (own key
        #    salt so it does not coincide with the angle-shift selection). One
        #    passive-learner variant per selected seed.
        if passive_learner_fraction > 0.0 and all_seeds:
            passive_seeds = deterministic_sample(
                all_seeds, passive_learner_fraction, key=lambda s: f"passive::{s[0]}"
            )
            logger.info(
                "[sft:%s] passive_learner_fraction=%.2f -> %d/%d additional passive-learner variants",
                level, passive_learner_fraction, len(passive_seeds), len(all_seeds),
            )
            for sid, seed in passive_seeds:
                if _sft_passive_learner_id(sid, 0) in done_ids:
                    continue
                pending.append((sid, seed, 0, False, True))

        if not pending:
            logger.info("[sft:%s] nothing to do (%d already done)", level, len(done_ids))
            results[level] = 0
            continue

        async def _run(
            args: tuple[str, ScenarioSeed, int, bool, bool],
        ) -> SFTExample | None:
            sid, seed, variant, is_angle_shift, is_passive_learner = args
            return await _generate_one(
                teacher=teacher,
                seed_id=sid,
                seed=seed,
                level_spec=_level_spec(seed.cefr_level, seed.locale),
                max_tokens=max_tokens,
                temperature=temperature,
                failures_path=failures_path,
                generation_meta=generation_meta,
                variant=variant,
                is_angle_shift=is_angle_shift,
                is_passive_learner=is_passive_learner,
            )

        completed = await gather_with_concurrency(
            [_run(args) for args in pending],
            concurrency=concurrency,
            desc=f"sft_normal[{level}]",
        )

        written = 0
        for ex in completed:
            if ex is None:
                continue
            append_jsonl(out_path, ex)
            written += 1
        results[level] = written
        logger.info("[sft:%s] wrote %d new examples", level, written)
    return results
