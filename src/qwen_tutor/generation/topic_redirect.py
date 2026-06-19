"""Topic-drift user-side handler SFT generation.

5th redirect axis. Even if the learner suddenly brings up an off-topic subject
(weather/personal life/different setting/generic smalltalk), the tutor should
acknowledge it lightly without lecturing and smoothly return the conversation
to the topic. This pattern is taught as SFT data.

Each dialogue contains both parts:
  USER part  : one off-topic turn somewhere in the middle matching one of
               ``TOPIC_DRIFT_TRIGGER_KINDS``.
  TUTOR part : one sentence acknowledge + one sentence bridge-back + return
               to on-topic.

Cycle ``TOPIC_DRIFT_TRIGGER_KINDS`` deterministically across seeds and variants
to balance the trigger distribution. It is saved as
``scenario_type="redirect"`` so ``BannedTermsFilter`` skips the user turn
(because this redirect user turn is an "intended" off-topic move).

Writes to ``data/sft_raw/topic_redirect_{level}.jsonl``.
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
DEFAULT_FAILURES_PATH = Path("data/topic_redirect_failures.jsonl")

# HARD off-topic trigger categories. Brief weather small talk / one-line
# personal exchanges are considered NORMAL daily-life flow and the system
# prompt tells the model to just accept them in stride -- those are NOT in
# this menu. The triggers below are real abandonment of the scenario:
#
#   different_setting:         user starts describing a different place or
#                              activity totally unrelated to the scenario
#                              (e.g. shopping scenario -> talks about museum)
#   subject_swap:              user explicitly tries to change the topic
#                              ("let's talk about sports instead")
#   extended_personal_inquiry: user keeps pushing about the tutor's personal
#                              life beyond a single one-line exchange (3+
#                              consecutive personal probes)
#   off_domain_tangent:        user pivots to a high-stakes domain unrelated
#                              to the scenario (philosophy, politics-adjacent
#                              opinions, sustained life advice questions)
#
# Cycled deterministically across (seed, variant) pairs for an even
# distribution.
TOPIC_DRIFT_TRIGGER_KINDS: tuple[str, ...] = (
    "different_setting",
    "subject_swap",
    "extended_personal_inquiry",
    "off_domain_tangent",
)


def _topic_redirect_id(seed_id: str, trigger: str, variant: int = 0) -> str:
    if variant == 0:
        return f"topic_redirect_{trigger}_{seed_id}"
    return f"topic_redirect_{trigger}_{seed_id}_v{variant}"


def _parse_messages(raw: str) -> list[Message]:
    data = extract_first_json(raw)
    if not isinstance(data, dict) or "messages" not in data:
        raise ValueError(
            "topic_redirect dialogue response missing top-level 'messages' field"
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
            f"topic_redirect dialogue too short ({len(messages)} turns, "
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
    template = render_prompt("dialogue_topic_redirect", locale_name=locale)
    prompt = template.format(
        scenario_json=scenario_json,
        level=seed.cefr_level,
        level_spec_with_locale_instruction=level_spec,
        topic_drift_trigger=trigger,
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
            _topic_redirect_id(seed_id, trigger, variant),
            f"{type(exc).__name__}: {exc}",
            level=seed.cefr_level,
            stage="topic_redirect",
            seed_id=seed_id,
            topic_drift_trigger=trigger,
            variant=variant,
        )
        return None

    metadata = ExampleMetadata(
        topic=seed.topic,
        subtopics=list(seed.subtopics),
        user_role=seed.user_role,
        model_role=seed.model_role,
        cefr_level=seed.cefr_level,
        scenario_type="redirect",  # let banned_terms filter skip the user turn
        locale=locale,
        category=seed.category,
        generation={
            **generation_meta_base,
            "topic_drift_trigger": trigger,
            "variant": variant,
            "violation_turn_idx": violation_turn_idx,
        },
    )
    return SFTExample(
        id=_topic_redirect_id(seed_id, trigger, variant),
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
    topic_redirect_fraction: float = 0.20,
    dialogues_per_seed: int = 1,
) -> dict[str, int]:
    """Generate topic-redirect dialogues for a fraction of seeds.

    ``topic_redirect_fraction``: portion of the seed pool turned into
    topic-redirect variants. ``dialogues_per_seed`` >= 2 means multiple
    variants per (seed, trigger). Trigger cycles deterministically across
    (seed, variant) pairs to keep the distribution flat.

    Returns ``{level: n_written}`` for this run.
    """
    if teacher is None:
        teacher = build_teacher_from_config(config_path, role="teacher")
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    triggers = triggers or TOPIC_DRIFT_TRIGGER_KINDS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)

    generation_meta_base = {
        "provider": teacher.config.provider,
        "model": teacher.config.model,
        "prompt": "DIALOGUE_PROMPT_TOPIC_REDIRECT",
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
        out_path = output_dir / f"topic_redirect_{level}.jsonl"
        done_ids = load_existing_ids(out_path)

        all_seeds = list(iter_seeds(seeds_dir, [level]))
        selected_seeds = deterministic_sample(
            all_seeds, topic_redirect_fraction, key=lambda s: s[0]
        )
        logger.info(
            "[topic_redirect:%s] fraction=%.2f -> %d/%d seeds selected (x%d variants)",
            level, topic_redirect_fraction, len(selected_seeds), len(all_seeds),
            dialogues_per_seed,
        )

        trigger_cycle = cycle(triggers)
        pending: list[tuple[str, ScenarioSeed, str, int]] = []
        for sid, seed in selected_seeds:
            for variant in range(dialogues_per_seed):
                trigger = next(trigger_cycle)
                if _topic_redirect_id(sid, trigger, variant) in done_ids:
                    continue
                pending.append((sid, seed, trigger, variant))

        if not pending:
            logger.info(
                "[topic_redirect:%s] nothing to do (%d done)", level, len(done_ids)
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
            desc=f"topic_redirect[{level}]",
        )

        written = 0
        for ex in completed:
            if ex is None:
                continue
            append_jsonl(out_path, ex)
            written += 1
        results[level] = written
        logger.info("[topic_redirect:%s] wrote %d new examples", level, written)
    return results
