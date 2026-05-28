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
    render_deployment_system_prompt,
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
    extract_first_json,
    gather_with_concurrency,
    load_existing_ids,
)

logger = logging.getLogger(__name__)

DEFAULT_SEEDS_DIR = Path("data/seeds")
DEFAULT_OUTPUT_DIR = Path("data/sft_raw")
DEFAULT_FAILURES_PATH = Path("data/sft_dialogues_failures.jsonl")


def _sft_id(seed_id: str, variant: int = 0) -> str:
    """SFT 예시 id. variant>0 이면 같은 seed 의 다른 sampling 결과."""
    if variant == 0:
        return f"sft_{seed_id}"
    return f"sft_{seed_id}_v{variant}"


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
) -> SFTExample | None:
    locale = seed.locale
    scenario_json = json.dumps(seed.model_dump(), ensure_ascii=False)
    template = render_prompt("dialogue_normal", locale_name=locale)
    prompt = template.format(
        scenario_json=scenario_json,
        level=seed.cefr_level,
        level_spec_with_locale_instruction=level_spec,
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
            _sft_id(seed_id, variant),
            f"{type(exc).__name__}: {exc}",
            level=seed.cefr_level,
            stage="sft_normal",
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
        scenario_type="normal",
        locale=locale,
        generation={**generation_meta, "variant": variant},
    )
    return SFTExample(
        id=_sft_id(seed_id, variant),
        metadata=metadata,
        system_prompt=render_deployment_system_prompt(
            seed.cefr_level, locale_name=locale
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
) -> dict[str, int]:
    """Generate normal SFT dialogues for every seed under ``seeds_dir``.

    ``dialogues_per_seed`` 가 1 이면 시드당 한 편(기존). 2 이상이면 같은
    시드에 대해 그 횟수만큼 ``_generate_one`` 을 호출해 변종을 만듭니다.
    각 변종 id 는 ``sft_{seed_id}_v{variant}`` 형태로 유일합니다.

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
        pending: list[tuple[str, ScenarioSeed, int]] = []
        for sid, seed in iter_seeds(seeds_dir, [level]):
            for variant in range(dialogues_per_seed):
                if _sft_id(sid, variant) in done_ids:
                    continue
                pending.append((sid, seed, variant))
        if not pending:
            logger.info("[sft:%s] nothing to do (%d already done)", level, len(done_ids))
            results[level] = 0
            continue

        async def _run(args: tuple[str, ScenarioSeed, int]) -> SFTExample | None:
            sid, seed, variant = args
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
