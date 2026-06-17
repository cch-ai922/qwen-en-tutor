"""Evaluation-example generation.

For each SFT example, calls EVALUATION_GENERATION_PROMPT to produce an
examiner-style assistant turn — a ``<think>...</think>`` reasoning block
immediately followed by a JSON ``EvaluationOutput`` object — and packs
it into an EvaluationExample with the source dialogue rendered as a
single user-turn transcript. Writes to ``data/eval_raw/{level}.jsonl``.
Resumes from existing output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qwen_tutor.generation._prompt_select import (
    render_prompt,
    validate_prompt_has_locale_instruction,
)
from qwen_tutor.generation.prompts import render_evaluation_system_prompt
from qwen_tutor.generation.teacher import TeacherClient, build_teacher_from_config
from qwen_tutor.schemas import (
    EvaluationExample,
    EvaluationMetadata,
    Message,
    SFTExample,
)
from qwen_tutor.utils.runner import (
    append_failure,
    append_jsonl,
    deterministic_sample,
    gather_with_concurrency,
    load_existing_ids,
)

logger = logging.getLogger(__name__)

DEFAULT_SFT_DIR = Path("data/sft_raw")
DEFAULT_OUTPUT_DIR = Path("data/eval_raw")
DEFAULT_FAILURES_PATH = Path("data/evaluation_failures.jsonl")


def _eval_id(sft_id: str) -> str:
    return f"eval_{sft_id}"


def _render_subtopics_block(subtopics: list[str]) -> str:
    return "\n".join(f"- {s}" for s in subtopics)


def _render_transcript(sft: SFTExample) -> str:
    # Roles + topic + subtopics are prepended so the deployed /think model sees
    # the same context at inference as the teacher saw at generation. Roles let
    # the judge score topic_adherence and interaction against what's appropriate
    # for the LEARNER/TUTOR personas, not just the bare topic.
    #
    # CEFR header: the EVALUATION_SYSTEM_PROMPT says "Given the transcript and
    # a target CEFR level, produce ..." — but historically the level was never
    # actually injected into the user content. Without it the model can only
    # guess the learner's *absolute* level, and ``overall_cefr_estimate`` loses
    # its meaning ("how does the learner compare to the target?"). Always emit
    # the level as the first line so training and deploy both surface it.
    md = sft.metadata
    lines: list[str] = [
        f"Target CEFR level: {md.cefr_level}",
        f"Tutor role: {md.model_role.name} -- {md.model_role.description}",
        f"Learner role: {md.user_role.description}",
        f"Assigned topic: {md.topic}",
        "Assigned subtopics:",
        _render_subtopics_block(md.subtopics),
        "",
        "Transcript:",
    ]
    user_idx = 0
    for m in sft.messages:
        if m.role == "user":
            lines.append(f"[USER turn {user_idx}] {m.content}")
            user_idx += 1
        elif m.role == "assistant":
            lines.append(f"[TUTOR] {m.content}")
        else:
            lines.append(f"[{m.role.upper()}] {m.content}")
    return "\n".join(lines)


def _iter_sft_examples(
    sft_dir: Path, levels: list[str]
) -> list[tuple[str, SFTExample]]:
    out: list[tuple[str, SFTExample]] = []
    for level in levels:
        # 8-way SFT sources: normal + redirect + 6 user-side redirect streams
        # (locale / pedagogy / language / persona / topic / role_swap). Each
        # stream contains a different kind of graceful response, which makes
        # eval(/think) grading more diverse.
        for prefix in (
            "normal",
            "redirect",
            "locale_redirect",
            "pedagogy_redirect",
            "language_redirect",
            "persona_redirect",
            "topic_redirect",
            "role_swap_redirect",
        ):
            path = sft_dir / f"{prefix}_{level}.jsonl"
            if not path.exists():
                continue
            for ex in SFTExample.from_jsonl(path):
                out.append((level, ex))
    return out


async def _generate_one(
    teacher: TeacherClient,
    sft: SFTExample,
    max_tokens: int,
    temperature: float,
    failures_path: Path,
) -> EvaluationExample | None:
    locale = sft.metadata.locale
    try:
        # Three-way alignment: the teacher sees the same prompt shape the
        # trained student will see at train and deploy time.
        #   - system: instructions only (rubric, JSON schema, output format)
        #   - user:   _render_transcript(sft), which carries the target CEFR
        #             level + roles + topic + subtopics + transcript
        # Previously the dialogue was embedded inside the system prompt and
        # the user turn was empty; small teachers (4B Q4) interpreted the
        # empty user turn as "Begin." and wasted their <think> budget
        # hunting for where the transcript actually was.
        system_prompt = render_prompt("evaluation_generation", locale_name=locale)
        validate_prompt_has_locale_instruction(system_prompt, locale_name=locale)
        transcript_payload = _render_transcript(sft)
        raw = await teacher.generate(
            system=system_prompt,
            messages=[Message(role="user", content=transcript_payload)],
            cacheable_prefix=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Some models like gpt-oss may send reasoning separately over a harmony
        # channel and omit the `<think>` tags. In that case, we wrap the prose
        # reasoning into `<think>...</think>`. If the reasoning is effectively
        # empty (e.g. Qwen3 outputs only `<think>\n</think>`), it has no value
        # as training data, so we treat it as a failure and regenerate.
        if "<think>" not in raw or "</think>" not in raw:
            # There must be a JSON body; otherwise this is a real failure.
            if "{" not in raw:
                raise ValueError(
                    "evaluation response missing both <think> block and JSON"
                )
            json_start = raw.index("{")
            head = raw[:json_start].strip()
            tail = raw[json_start:]
            if not head:
                raise ValueError(
                    "evaluation response is JSON-only (no reasoning); "
                    "ensure teacher is in /think mode"
                )
            # If prose exists, use it as the reasoning body (e.g. gpt-oss harmony
            # final channel prose).
            raw = f"<think>\n{head}\n</think>\n{tail}"
        else:
            # The think block exists but is empty (`<think>\n</think>`) - Qwen3
            # may have responded with /no_think or duplicated the close tag.
            # This is worthless as training data, so reject it.
            t_open = raw.index("<think>") + len("<think>")
            t_close = raw.index("</think>")
            if t_close <= t_open:
                raise ValueError(
                    "evaluation response has malformed <think>...</think> "
                    "(close before open?)"
                )
            think_body = raw[t_open:t_close].strip()
            if len(think_body) < 30:
                # If it is under 30 chars, treat it as effectively empty.
                raise ValueError(
                    f"evaluation response has empty/trivial <think> block "
                    f"({len(think_body)} chars); ensure teacher is in /think mode"
                )
        if "{" not in raw[raw.rindex("</think>") :]:
            raise ValueError("evaluation response missing JSON after </think>")
        # If the close tag is duplicated, keep only the last one.
        last_close = raw.rindex("</think>")
        first_close = raw.index("</think>")
        if last_close != first_close:
            # `<think>...</think>...</think>{json}` → `<think>...</think>{json}`
            head_chunk = raw[: first_close + len("</think>")]
            tail_chunk = raw[last_close + len("</think>") :]
            raw = head_chunk + tail_chunk
    except Exception as exc:  # noqa: BLE001
        append_failure(
            failures_path,
            _eval_id(sft.id),
            f"{type(exc).__name__}: {exc}",
            level=sft.metadata.cefr_level,
            stage="evaluation",
            sft_id=sft.id,
        )
        return None

    return EvaluationExample(
        id=_eval_id(sft.id),
        metadata=EvaluationMetadata(
            source_dialogue_id=sft.id,
            learner_cefr_target=sft.metadata.cefr_level,
            locale=locale,
            scenario_type=sft.metadata.scenario_type,
            category=sft.metadata.category,
            generation=sft.metadata.generation,
        ),
        system_prompt=render_evaluation_system_prompt(locale_name=locale),
        messages=[
            Message(role="user", content=_render_transcript(sft)),
            Message(role="assistant", content=raw.strip()),
        ],
    )


async def generate_batch(
    cefr_levels: list[str] | None = None,
    sft_dir: str | Path = DEFAULT_SFT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    failures_path: str | Path = DEFAULT_FAILURES_PATH,
    config_path: str | Path = "config/generation.yaml",
    role: str = "judge",
    concurrency: int = 20,
    max_tokens: int | None = 2048,
    temperature: float = 0.3,
    teacher: TeacherClient | None = None,
    eval_fraction: float = 0.25,
) -> dict[str, int]:
    """Generate EvaluationExample for a fraction of SFT examples.

    ``eval_fraction`` is the fraction of the SFT pool to use for /think eval
    examples (0.0–1.0). Matching this to the SFT/eval mix ratio in
    training.yaml minimizes the amount that ``mix_and_split`` discards. The
    same subset is always chosen for a given SFT id, so partial resume is
    safe.

    By default uses the ``judge`` role from ``generation.yaml`` (typically
    a smaller / cheaper examiner model). Returns ``{level: n_written}``.
    """
    if teacher is None:
        teacher = build_teacher_from_config(config_path, role=role)
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)
    # max_tokens=None means "use the default 2048". Callers that need a
    # bigger budget for /think output (e.g. 9B teachers whose <think> is
    # verbose enough to truncate the trailing JSON) pass an explicit int.
    if max_tokens is None:
        max_tokens = 2048

    sft_examples = _iter_sft_examples(Path(sft_dir), cefr_levels)
    by_level: dict[str, list[SFTExample]] = {lvl: [] for lvl in cefr_levels}
    for level, ex in sft_examples:
        by_level[level].append(ex)

    results: dict[str, int] = {}
    for level in cefr_levels:
        out_path = output_dir / f"{level}.jsonl"
        done_ids = load_existing_ids(out_path)

        # Deterministically select only eval_fraction of the SFT pool for eval generation.
        pool = by_level[level]
        selected = deterministic_sample(
            pool, eval_fraction, key=lambda ex: ex.id
        )
        logger.info(
            "[eval:%s] fraction=%.2f → %d/%d SFT examples selected",
            level, eval_fraction, len(selected), len(pool),
        )
        pending = [ex for ex in selected if _eval_id(ex.id) not in done_ids]
        if not pending:
            logger.info("[eval:%s] nothing to do (%d done)", level, len(done_ids))
            results[level] = 0
            continue

        async def _run(ex: SFTExample) -> EvaluationExample | None:
            return await _generate_one(
                teacher=teacher,
                sft=ex,
                max_tokens=max_tokens,
                temperature=temperature,
                failures_path=failures_path,
            )

        completed = await gather_with_concurrency(
            [_run(ex) for ex in pending],
            concurrency=concurrency,
            desc=f"eval[{level}]",
        )

        written = 0
        for ev in completed:
            if ev is None:
                continue
            append_jsonl(out_path, ev)
            written += 1
        results[level] = written
        logger.info("[eval:%s] wrote %d new examples", level, written)
    return results
