"""Evaluation-example generation.

For each SFT example, calls EVALUATION_GENERATION_PROMPT to produce an
examiner-style assistant turn — a ``<think>...</think>`` reasoning block
immediately followed by a JSON ``EvaluationOutput`` object — and packs
it into an EvaluationExample with the source dialogue rendered as a
single user-turn transcript. Writes to ``data/eval_raw/{level}.jsonl``.
Resumes from existing output.
"""

from __future__ import annotations

import json
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


def _render_transcript(sft: SFTExample) -> str:
    lines: list[str] = ["Transcript:"]
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
        # 6-way SFT 출처: normal + redirect + 4 user-side redirect 스트림
        # (locale / pedagogy / language / persona). 각 스트림이 다른 종류의
        # graceful 응답을 담고 있어 eval(/think 채점)도 다양해집니다.
        for prefix in (
            "normal",
            "redirect",
            "locale_redirect",
            "pedagogy_redirect",
            "language_redirect",
            "persona_redirect",
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
        dialogue_payload = json.dumps(
            {"messages": [m.model_dump() for m in sft.messages]},
            ensure_ascii=False,
        )
        template = render_prompt("evaluation_generation", locale_name=locale)
        prompt = template.format(
            full_dialogue_json=dialogue_payload,
            target_cefr=sft.metadata.cefr_level,
        )
        validate_prompt_has_locale_instruction(prompt, locale_name=locale)
        raw = await teacher.generate(
            system=prompt,
            messages=[],
            cacheable_prefix=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # gpt-oss 같은 모델은 harmony 채널로 reasoning 을 따로 보내고 `<think>`
        # 태그를 안 쓸 수 있습니다. 그런 경우 reasoning 텍스트를 prose 에서
        # 끌어와 ``<think>...</think>`` 형식으로 래핑합니다. 단, reasoning 이
        # 사실상 비어 있으면 (Qwen3 가 ``<think>\n</think>`` 만 찍은 경우 등)
        # 학습 데이터로서 가치가 없으므로 실패로 처리해서 재생성하게 만듭니다.
        if "<think>" not in raw or "</think>" not in raw:
            # JSON 본문은 있어야 합니다 - 없으면 진짜 실패.
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
            # prose 가 있으면 그게 reasoning 본문 (e.g. gpt-oss harmony final
            # channel 의 prose).
            raw = f"<think>\n{head}\n</think>\n{tail}"
        else:
            # think 블록은 있는데 비어 있는 경우 (`<think>\n</think>`) - Qwen3
            # 가 /no_think 으로 응답했거나, 같은 close tag 가 중복으로 찍힌
            # 경우. 학습 데이터로 무가치하니 reject.
            t_open = raw.index("<think>") + len("<think>")
            t_close = raw.index("</think>")
            if t_close <= t_open:
                raise ValueError(
                    "evaluation response has malformed <think>...</think> "
                    "(close before open?)"
                )
            think_body = raw[t_open:t_close].strip()
            if len(think_body) < 30:
                # 30 chars 미만이면 사실상 비어 있다고 간주
                raise ValueError(
                    f"evaluation response has empty/trivial <think> block "
                    f"({len(think_body)} chars); ensure teacher is in /think mode"
                )
        if "{" not in raw[raw.rindex("</think>") :]:
            raise ValueError("evaluation response missing JSON after </think>")
        # 같은 close tag 가 중복 찍힌 경우 마지막 것만 남깁니다.
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
    max_tokens: int = 2048,
    temperature: float = 0.3,
    teacher: TeacherClient | None = None,
    eval_fraction: float = 0.25,
) -> dict[str, int]:
    """Generate EvaluationExample for a fraction of SFT examples.

    ``eval_fraction`` 는 SFT 풀 중 /think eval 예시를 만들 비율(0.0~1.0).
    training.yaml 의 SFT/eval mix ratio 와 맞춰 두면 ``mix_and_split`` 이
    잘라 버리는 양이 거의 없어집니다. 같은 SFT id 에 대해 항상 같은
    부분집합이 선택되어 부분 resume 도 안전합니다.

    By default uses the ``judge`` role from ``generation.yaml`` (typically
    a smaller / cheaper examiner model). Returns ``{level: n_written}``.
    """
    if teacher is None:
        teacher = build_teacher_from_config(config_path, role=role)
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)

    sft_examples = _iter_sft_examples(Path(sft_dir), cefr_levels)
    by_level: dict[str, list[SFTExample]] = {lvl: [] for lvl in cefr_levels}
    for level, ex in sft_examples:
        by_level[level].append(ex)

    results: dict[str, int] = {}
    for level in cefr_levels:
        out_path = output_dir / f"{level}.jsonl"
        done_ids = load_existing_ids(out_path)

        # SFT 풀에서 eval_fraction 만큼만 deterministic 하게 골라 eval 생성.
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
