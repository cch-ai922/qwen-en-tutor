"""Speech-recognition (ASR) slip-repair SFT generation.

The tutor is used through speech: the learner talks, a speech-to-text system
transcribes them, and the tutor reads the transcript. STT frequently mis-hears
a word and writes a DIFFERENT real word that sounds similar ("buy" -> "by",
"here" -> "hear", "four" -> "for"). This stream trains the tutor to silently
infer the intended word from context, use the CORRECT word naturally in its
own reply, and keep the conversation going -- WITHOUT quizzing the learner or
turning the slip into a spelling lesson.

Each dialogue contains both parts:
  USER part  : 1-2 turns in the middle carry a plausible ASR slip of the
               assigned ``ASR_ERROR_KINDS`` kind (the intended word obvious
               from context).
  TUTOR part : a natural reply that uses the correct word in stride and
               continues on-topic -- an invisible repair, not a correction.

Cycle ``ASR_ERROR_KINDS`` deterministically across seeds and variants to
balance the error-kind distribution. Saved as ``scenario_type="redirect"`` so
``BannedTermsFilter`` / ``NonLatinScriptFilter`` skip the user turn (the slip
is an intended user-side artifact the model never emits).

Writes to ``data/sft_raw/asr_repair_{level}.jsonl``.
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
DEFAULT_FAILURES_PATH = Path("data/asr_repair_failures.jsonl")

# Kinds of speech-to-text mis-hearing the learner turn may carry. Each is a
# REAL English word the recognizer plausibly writes instead of the intended
# word, with the intended word obvious from context:
#
#   homophone:      identical sound, different word (buy/by, here/hear,
#                   know/no, weather/whether).
#   near_homophone: one or two phonemes off (bear/bare, dessert/desert,
#                   feel/fill).
#   word_boundary:  recognizer split or merged words wrong ("ice cream" vs
#                   "I scream", "a lot" vs "allot").
#   number_or_unit: a number/unit mis-heard (four/for, eight/ate, two/to).
#
# Cycled deterministically across (seed, variant) pairs for an even
# distribution.
ASR_ERROR_KINDS: tuple[str, ...] = (
    "homophone",
    "near_homophone",
    "word_boundary",
    "number_or_unit",
)


def _asr_repair_id(seed_id: str, kind: str, variant: int = 0) -> str:
    if variant == 0:
        return f"asr_repair_{kind}_{seed_id}"
    return f"asr_repair_{kind}_{seed_id}_v{variant}"


def _parse_messages(raw: str) -> list[Message]:
    data = extract_first_json(raw)
    if not isinstance(data, dict) or "messages" not in data:
        raise ValueError(
            "asr_repair dialogue response missing top-level 'messages' field"
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
            f"asr_repair dialogue too short ({len(messages)} turns, "
            f"need >= {MIN_TURNS})"
        )
    return messages


async def _generate_one(
    teacher: TeacherClient,
    seed_id: str,
    seed: ScenarioSeed,
    kind: str,
    level_spec: str,
    max_tokens: int,
    temperature: float,
    failures_path: Path,
    generation_meta_base: dict[str, Any],
    variant: int = 0,
) -> SFTExample | None:
    locale = seed.locale
    scenario_json = json.dumps(seed.model_dump(), ensure_ascii=False)
    template = render_prompt("dialogue_asr_repair", locale_name=locale)
    prompt = template.format(
        scenario_json=scenario_json,
        level=seed.cefr_level,
        level_spec_with_locale_instruction=level_spec,
        asr_error_kind=kind,
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
            _asr_repair_id(seed_id, kind, variant),
            f"{type(exc).__name__}: {exc}",
            level=seed.cefr_level,
            stage="asr_repair",
            seed_id=seed_id,
            asr_error_kind=kind,
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
            "asr_error_kind": kind,
            "variant": variant,
        },
    )
    return SFTExample(
        id=_asr_repair_id(seed_id, kind, variant),
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
    kinds: tuple[str, ...] | None = None,
    teacher: TeacherClient | None = None,
    asr_repair_fraction: float = 0.10,
    dialogues_per_seed: int = 1,
) -> dict[str, int]:
    """Generate ASR-repair dialogues for a fraction of seeds.

    ``asr_repair_fraction``: portion of the seed pool turned into ASR-repair
    variants. ``dialogues_per_seed`` >= 2 means multiple variants per
    (seed, kind). Error kind cycles deterministically across (seed, variant)
    pairs to keep the distribution flat.

    Returns ``{level: n_written}`` for this run.
    """
    if teacher is None:
        teacher = build_teacher_from_config(config_path, role="teacher")
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    kinds = kinds or ASR_ERROR_KINDS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)

    generation_meta_base = {
        "provider": teacher.config.provider,
        "model": teacher.config.model,
        "prompt": "DIALOGUE_PROMPT_ASR_REPAIR",
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
        out_path = output_dir / f"asr_repair_{level}.jsonl"
        done_ids = load_existing_ids(out_path)

        all_seeds = list(iter_seeds(seeds_dir, [level]))
        selected_seeds = deterministic_sample(
            all_seeds, asr_repair_fraction, key=lambda s: f"asr::{s[0]}"
        )
        logger.info(
            "[asr_repair:%s] fraction=%.2f -> %d/%d seeds selected (x%d variants)",
            level, asr_repair_fraction, len(selected_seeds), len(all_seeds),
            dialogues_per_seed,
        )

        kind_cycle = cycle(kinds)
        pending: list[tuple[str, ScenarioSeed, str, int]] = []
        for sid, seed in selected_seeds:
            for variant in range(dialogues_per_seed):
                kind = next(kind_cycle)
                if _asr_repair_id(sid, kind, variant) in done_ids:
                    continue
                pending.append((sid, seed, kind, variant))

        if not pending:
            logger.info("[asr_repair:%s] nothing to do (%d done)", level, len(done_ids))
            results[level] = 0
            continue

        async def _run(item: tuple[str, ScenarioSeed, str, int]) -> SFTExample | None:
            sid, seed, kind, variant = item
            return await _generate_one(
                teacher=teacher,
                seed_id=sid,
                seed=seed,
                kind=kind,
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
            desc=f"asr_repair[{level}]",
        )

        written = 0
        for ex in completed:
            if ex is None:
                continue
            append_jsonl(out_path, ex)
            written += 1
        results[level] = written
        logger.info("[asr_repair:%s] wrote %d new examples", level, written)
    return results
