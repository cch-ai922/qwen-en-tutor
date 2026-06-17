"""On-policy DPO pair generation.

For each filtered SFT example, regenerate one assistant turn using the
CURRENT POLICY (the model being trained, post-SFT) and have a judge
model compare it against the teacher's original turn. If the teacher
wins by a meaningful margin, emit a DPO pair where:

    chosen   = the teacher's (filtered) SFT turn
    rejected = the policy's regenerated turn

This is "on-policy" in the sense that the rejected sample comes from
the current model's own distribution — DPO can therefore correct drift
modes that the offline register_pairs pipeline cannot see.

Ties and policy-wins are SKIPPED (not inverted into pairs): preferring
policy outputs over the vetted teacher risks teaching the model to
reinforce its own biases. The count of skipped cases is logged in the
failures file so you can audit whether the policy is genuinely matching
the teacher.

The judge classifies WHY the loser lost into one of the existing
``RejectionAxis`` values (``register_unnatural``, ``cefr_mismatch``,
``locale_violation``, ``pedagogy_weak``, ``accuracy_error``, ``language_violation``,
``persona_break``,
``off_topic``) so DPO training sees the same axis taxonomy as the
offline register pairs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, get_args

from qwen_tutor.generation.prompts import (
    render_scenario_deployment_system_prompt,
)
from qwen_tutor.generation.teacher import TeacherClient, build_teacher_from_config
from qwen_tutor.schemas import (
    DPOExample,
    ExampleMetadata,
    Message,
    RejectionAxis,
    SFTExample,
)
from qwen_tutor.utils.runner import (
    append_failure,
    append_jsonl,
    extract_first_json,
    gather_with_concurrency,
    load_existing_ids,
)
from qwen_tutor.utils.thinking import with_thinking_directive

logger = logging.getLogger(__name__)

DEFAULT_SFT_FILTERED_DIR = Path("data/sft_filtered")
DEFAULT_OUTPUT_DIR = Path("data/dpo_raw")
DEFAULT_FAILURES_PATH = Path("data/on_policy_pairs_failures.jsonl")
DEFAULT_AUDIT_PATH = Path("data/on_policy_pairs_audit.jsonl")

VALID_AXES: tuple[str, ...] = get_args(RejectionAxis)
MIN_JUDGE_MARGIN = 2  # 1=slight, 2=clear, 3=large — keep only clear or larger


# ---------------------------------------------------------------------------
# Target-model protocol (avoids depending on the evaluation module)
# ---------------------------------------------------------------------------


class TargetModelClient(Protocol):
    """Anything that can generate from the current policy."""

    async def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        mode: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        ...


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------


# Judge prompt — instructions ONLY. The per-call data (CEFR level, dialogue
# context, the two candidate assistant turns) arrives in the USER message
# constructed by ``_format_judge_user_message``. Splitting like this avoids
# the small-model confusion we hit in eval_gen (model wastes its <think>
# budget hunting for the data when system=everything / user=empty).
_JUDGE_PROMPT_RAW = (
    "You are judging two candidate assistant turns for an English-tutor\n"
    "conversation with {country_adjective} {learner_description}. The two\n"
    "candidates were produced for the same prompt context; one is from a\n"
    "strong reference model, the other from a model currently being trained.\n"
    "You do not know which is which - judge purely on output quality.\n"
    "\n"
    "The input — target CEFR level, the dialogue context, and the two\n"
    "candidate assistant turns — is delivered as the USER message\n"
    "immediately following these instructions.\n"
    "\n"
    "Evaluate the two candidates against ALL of these criteria simultaneously:\n"
    "\n"
    "  - locale: {country_adjective}-grounded (names, places, foods,\n"
    "    transit, currency, cultural items). "
    "{avoid_cultures_phrase} defaults are a major demerit.\n"
    "  - cefr_register: vocabulary, sentence length, grammar appropriate for\n"
    "    the target CEFR level shown in the USER message. Over-shoot and\n"
    "    under-shoot are both demerits.\n"
    "  - naturalness: conversational, register-appropriate. Textbook-stiff\n"
    "    phrasing is a demerit.\n"
    "  - pedagogy: engages the learner, asks follow-ups, gently recasts at\n"
    "    lower levels.\n"
    "  - accuracy: grammatical, factually consistent with the dialogue.\n"
    "  - on_topic: stays in the scenario; {avoided_topics_sentence}\n"
    "\n"
    "Choose a winner. If they are essentially tied, say so. If one wins,\n"
    "estimate the margin (1 = slight edge, 2 = clearly better, 3 = much\n"
    "better). Then classify the PRIMARY reason the loser lost, choosing\n"
    "ONE of these axis labels (the same taxonomy used elsewhere in the\n"
    "project):\n"
    "\n"
    "  register_unnatural    | stiff / textbook / register-mismatched register\n"
    "  cefr_mismatch         | wrong CEFR band (too hard or too easy)\n"
    "  locale_violation      | {avoid_cultures_phrase} names, places, foods, brands\n"
    "  pedagogy_weak         | doesn't engage, asks nothing, mechanical, lectures\n"
    "  accuracy_error        | grammar / agreement / tense errors\n"
    "  off_topic             | drifts out of scenario, touches banned topics\n"
    "  language_violation    | tutor leaves English (translates to L1 or code-switches)\n"
    "  persona_break         | tutor leaks AI / chatbot scaffolding, breaks role\n"
    "  cave_on_persistence   | engages with substance after the user persists (was meant to refuse)\n"
    "  verbatim_repeat       | uses a generic boilerplate refusal that ignores the specific scenario\n"
    "  lecture_on_persistence| stiff cold formal refusal citing 'guidelines' / 'policy' / 'my role'\n"
    "\n"
    "Output STRICT JSON only, no fences, no prose:\n"
    "\n"
    '{"winner": "A" | "B" | "tie",\n'
    '  "margin": 1 | 2 | 3,\n'
    '  "reason_axis": "<one of the eight axis labels above>",\n'
    '  "note": "<one short sentence>"}\n'
)


def _render_judge_prompt(locale_name: str | None = None) -> str:
    """Substitute locale placeholders for the given locale. No per-call
    placeholders remain — the dynamic data goes in the USER message via
    ``_format_judge_user_message``.
    """
    from qwen_tutor.locale import get_locale

    loc = get_locale(locale_name)
    return (
        _JUDGE_PROMPT_RAW
        .replace("{country_adjective}", loc.country_adjective)
        .replace("{learner_description}", loc.learner_description)
        .replace("{avoid_cultures_phrase}", loc.avoid_cultures_phrase)
        .replace("{avoided_topics_sentence}", loc.avoided_topics_sentence)
    )


def _format_judge_user_message(
    cefr_level: str, context: str, candidate_a: str, candidate_b: str
) -> str:
    """Build the USER message that pairs with the judge system prompt.

    Mirrors the (instructions, data) split used in eval_gen so small judge
    models don't waste their <think> budget hunting for inputs.
    """
    return (
        f"Target CEFR level: {cefr_level}\n"
        f"\n"
        f"Dialogue context (up to the turn under judgment):\n"
        f"{context}\n"
        f"\n"
        f"Candidate A:\n"
        f"{candidate_a}\n"
        f"\n"
        f"Candidate B:\n"
        f"{candidate_b}"
    )


# Back-compat: default-locale pre-rendered constant.
_JUDGE_PROMPT = _render_judge_prompt()


# ---------------------------------------------------------------------------
# IDs + helpers
# ---------------------------------------------------------------------------


def _onpolicy_id(sft_id: str) -> str:
    return f"dpo_onpolicy_{sft_id}"


def _pick_assistant_turn_index(sft: SFTExample) -> int:
    """Deterministic pick of an assistant turn to challenge.

    Hash-based so a re-run picks the same turn for the same SFT id.
    Skips the very first assistant turn (which is usually a greeting
    that's hard to differentiate) when more than one is available.

    For ``persistent_*`` dialogues, also excludes the sentinel-bearing
    turn (recorded as ``generation["sentinel_turn"]``). The sentinel
    turn has a separate DPO pool (``sentinel_pairs.py``) where the
    chosen/rejected contrast is the literal ``[SESSION_END: <axis>]``
    marker rather than the judge-evaluated register quality on which
    this module operates. Mixing the two contrasts inside a single
    judge-mediated pair would muddy the signal: the judge does not score
    sentinel emission, only response quality. The 3 earlier refusal
    turns of a persistent dialogue are still eligible here and carry
    the register signal we want to reinforce.
    """
    idxs = [i for i, m in enumerate(sft.messages) if m.role == "assistant"]
    if not idxs:
        raise ValueError("SFT example has no assistant turn")
    sentinel_turn = None
    gen = sft.metadata.generation
    if isinstance(gen, dict):
        st = gen.get("sentinel_turn")
        if isinstance(st, int):
            sentinel_turn = st
    candidates = idxs[1:] if len(idxs) > 1 else idxs
    if sentinel_turn is not None:
        filtered = [i for i in candidates if i != sentinel_turn]
        if filtered:
            candidates = filtered
    h = int(hashlib.sha256(sft.id.encode("utf-8")).hexdigest()[:8], 16)
    return candidates[h % len(candidates)]


def _teacher_is_a(sft_id: str) -> bool:
    """Deterministic A/B order randomization keyed on the SFT id."""
    h = int(hashlib.sha256(("ab:" + sft_id).encode("utf-8")).hexdigest()[:8], 16)
    return h % 2 == 0


def _format_context(messages: list[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        role = "User" if m.role == "user" else ("Tutor" if m.role == "assistant" else m.role.capitalize())
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) or "(no prior turns)"


def _parse_judge_verdict(raw: str) -> dict[str, Any] | None:
    try:
        parsed = extract_first_json(raw)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(parsed, dict):
        return None
    winner = parsed.get("winner")
    if winner not in ("A", "B", "tie"):
        return None
    try:
        margin = int(parsed.get("margin", 0))
    except (TypeError, ValueError):
        margin = 0
    axis = parsed.get("reason_axis")
    if axis not in VALID_AXES:
        axis = "register_unnatural"  # safe fallback
    note = str(parsed.get("note", ""))[:240]
    return {"winner": winner, "margin": margin, "reason_axis": axis, "note": note}


# ---------------------------------------------------------------------------
# Loading filtered SFT examples
# ---------------------------------------------------------------------------


def _iter_filtered_sft(
    sft_filtered_dir: Path, cefr_levels: list[str]
) -> list[tuple[str, SFTExample]]:
    """Walk passed-only outputs from the filter pipeline.

    Reads every SFT prefix the filter writes — not just ``normal`` and
    ``redirect`` — so on-policy DPO sees the full range of natural
    assistant register that the policy needs to prefer over spoiled
    register. The 4 ``persistent_*`` streams are included because each
    contains 4-5 natural refusal turns whose register matters at deploy;
    ``_pick_assistant_turn_index`` excludes the sentinel-bearing turn
    so we never reinforce a position-bound sentinel emission.
    """
    out: list[tuple[str, SFTExample]] = []
    prefixes = (
        "normal",
        "redirect",
        "locale_redirect",
        "pedagogy_redirect",
        "language_redirect",
        "persona_redirect",
        "topic_redirect",
        "role_swap_redirect",
        "persistent_off_topic",
        "persistent_language_violation",
        "persistent_persona_break",
        "persistent_role_swap",
    )
    for level in cefr_levels:
        for prefix in prefixes:
            path = sft_filtered_dir / f"{prefix}_{level}_passed.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    example = rec.get("example") if isinstance(rec, dict) else None
                    if example is None:
                        example = rec
                    if not isinstance(example, dict):
                        continue
                    try:
                        ex = SFTExample.model_validate(example)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("on_policy_pairs: dropping malformed record: %s", exc)
                        continue
                    out.append((level, ex))
    return out


# ---------------------------------------------------------------------------
# Single-pair generation
# ---------------------------------------------------------------------------


async def _generate_one_pair(
    sft: SFTExample,
    target: TargetModelClient,
    judge: TeacherClient,
    *,
    min_margin: int,
    target_max_tokens: int,
    target_temperature: float,
    judge_max_tokens: int,
    judge_temperature: float,
    failures_path: Path,
    audit_path: Path,
    generation_meta_base: dict[str, Any],
) -> DPOExample | None:
    try:
        turn_idx = _pick_assistant_turn_index(sft)
    except ValueError as exc:
        append_failure(failures_path, _onpolicy_id(sft.id), str(exc),
                       level=sft.metadata.cefr_level, stage="on_policy", sft_id=sft.id)
        return None

    teacher_turn = sft.messages[turn_idx]
    prefix_messages = sft.messages[:turn_idx]
    locale_name = sft.metadata.locale
    # Per-SFT-example locale: pick the right deployment system prompt for
    # this example's country so the policy regenerates with the correct
    # grounding.
    system_prompt = render_scenario_deployment_system_prompt(
        cefr_level=sft.metadata.cefr_level,
        locale_name=locale_name,
        topic=sft.metadata.topic,
        subtopics=sft.metadata.subtopics,
        user_role_name=sft.metadata.user_role.name,
        user_role_description=sft.metadata.user_role.description,
        model_role_name=sft.metadata.model_role.name,
        model_role_description=sft.metadata.model_role.description,
    )

    # Have the policy regenerate the assistant turn given the same prefix.
    try:
        policy_text = await target.generate(
            system=system_prompt,
            messages=[{"role": m.role, "content": m.content} for m in prefix_messages],
            mode="conversation",
            max_new_tokens=target_max_tokens,
            temperature=target_temperature,
        )
    except Exception as exc:  # noqa: BLE001
        append_failure(failures_path, _onpolicy_id(sft.id),
                       f"target.generate raised {type(exc).__name__}: {exc}",
                       level=sft.metadata.cefr_level, stage="on_policy", sft_id=sft.id)
        return None
    policy_text = (policy_text or "").strip()
    if not policy_text:
        append_failure(failures_path, _onpolicy_id(sft.id), "policy produced empty output",
                       level=sft.metadata.cefr_level, stage="on_policy", sft_id=sft.id)
        return None

    # Randomized A/B order (deterministic by sft.id).
    teacher_is_a = _teacher_is_a(sft.id)
    cand_a = teacher_turn.content if teacher_is_a else policy_text
    cand_b = policy_text if teacher_is_a else teacher_turn.content

    # Per-locale judge prompt — judge weighs "is this {country}-grounded?"
    # against the example's locale, not against a baked-in default. System
    # carries instructions only; the dynamic data (CEFR, context, candidates)
    # goes in the USER message so the judge model doesn't waste its <think>
    # budget hunting for inputs (same fix applied to eval_gen).
    system_prompt_judge = with_thinking_directive(
        _render_judge_prompt(locale_name), role="judge"
    )
    judge_user_message = _format_judge_user_message(
        cefr_level=sft.metadata.cefr_level,
        context=_format_context(prefix_messages),
        candidate_a=cand_a,
        candidate_b=cand_b,
    )
    try:
        raw = await judge.generate(
            system=system_prompt_judge,
            messages=[Message(role="user", content=judge_user_message)],
            cacheable_prefix=None,
            max_tokens=judge_max_tokens,
            temperature=judge_temperature,
        )
    except Exception as exc:  # noqa: BLE001
        append_failure(failures_path, _onpolicy_id(sft.id),
                       f"judge.generate raised {type(exc).__name__}: {exc}",
                       level=sft.metadata.cefr_level, stage="on_policy", sft_id=sft.id)
        return None

    verdict = _parse_judge_verdict(raw)
    if verdict is None:
        append_failure(failures_path, _onpolicy_id(sft.id),
                       "judge returned unparseable verdict",
                       level=sft.metadata.cefr_level, stage="on_policy", sft_id=sft.id,
                       raw_excerpt=raw[:300])
        return None

    # Map A/B back to teacher/policy and record audit row regardless of outcome.
    winner_side = verdict["winner"]
    if winner_side == "A":
        teacher_won = teacher_is_a
    elif winner_side == "B":
        teacher_won = not teacher_is_a
    else:
        teacher_won = None  # tie
    audit_record = {
        "sft_id": sft.id,
        "cefr_level": sft.metadata.cefr_level,
        "turn_index": turn_idx,
        "teacher_is_a": teacher_is_a,
        "winner": winner_side,
        "teacher_won": teacher_won,
        "margin": verdict["margin"],
        "reason_axis": verdict["reason_axis"],
        "note": verdict["note"],
    }
    append_jsonl(audit_path, audit_record)

    # Decision: keep only teacher-wins above the margin threshold.
    if winner_side == "tie":
        return None
    if not teacher_won:
        # Policy beat teacher (rare). Skipped: see module docstring.
        return None
    if verdict["margin"] < min_margin:
        return None

    metadata = ExampleMetadata(
        topic=sft.metadata.topic,
        subtopics=list(sft.metadata.subtopics),
        user_role=sft.metadata.user_role,
        model_role=sft.metadata.model_role,
        cefr_level=sft.metadata.cefr_level,
        scenario_type=sft.metadata.scenario_type,
        locale=sft.metadata.locale,
        category=sft.metadata.category,
        generation={
            **generation_meta_base,
            "source_sft_id": sft.id,
            "chosen_turn_index": turn_idx,
            "judge_margin": verdict["margin"],
            "judge_note": verdict["note"],
        },
    )
    rejection_note = (
        f"On-policy DPO: policy regeneration lost vs teacher at margin "
        f"{verdict['margin']} on axis {verdict['reason_axis']!r}. "
        f"Judge note: {verdict['note']}"
    )
    return DPOExample(
        id=_onpolicy_id(sft.id),
        metadata=metadata,
        system_prompt=sft.system_prompt,
        prompt_messages=list(prefix_messages),
        chosen=teacher_turn,
        rejected=Message(role="assistant", content=policy_text),
        rejection_axis=verdict["reason_axis"],
        rejection_note=rejection_note,
    )


# ---------------------------------------------------------------------------
# Top-level batch
# ---------------------------------------------------------------------------


async def generate_batch(
    target: TargetModelClient,
    *,
    cefr_levels: list[str] | None = None,
    sft_filtered_dir: str | Path = DEFAULT_SFT_FILTERED_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    failures_path: str | Path = DEFAULT_FAILURES_PATH,
    audit_path: str | Path = DEFAULT_AUDIT_PATH,
    config_path: str | Path = "config/generation.yaml",
    judge: TeacherClient | None = None,
    min_margin: int = MIN_JUDGE_MARGIN,
    max_per_level: int | None = None,
    concurrency: int = 4,
    target_max_tokens: int = 320,
    target_temperature: float = 0.7,
    judge_max_tokens: int = 2048,
    judge_temperature: float = 0.0,
) -> dict[str, int]:
    """Generate on-policy DPO pairs from the current policy.

    Returns ``{level: n_written}`` for this run. Pairs are written to
    ``data/dpo_raw/on_policy_{level}.jsonl`` so they are picked up by
    the existing ``filter_data.py dpo`` and ``train_dpo`` steps
    alongside the offline ``register_*.jsonl`` files.
    """
    if judge is None:
        judge = build_teacher_from_config(config_path, role="judge")
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)
    audit_path = Path(audit_path)

    generation_meta_base = {
        "provider": getattr(judge.config, "provider", "unknown"),
        "model": getattr(judge.config, "model", "unknown"),
        "prompt": "on_policy_pair_judge",
        "min_margin": min_margin,
    }

    sft_examples = _iter_filtered_sft(Path(sft_filtered_dir), cefr_levels)
    by_level: dict[str, list[SFTExample]] = {lvl: [] for lvl in cefr_levels}
    for level, ex in sft_examples:
        by_level[level].append(ex)

    results: dict[str, int] = {}
    for level in cefr_levels:
        out_path = output_dir / f"on_policy_{level}.jsonl"
        done_ids = load_existing_ids(out_path)
        pool = by_level.get(level, [])
        if max_per_level is not None:
            pool = pool[:max_per_level]
        pending = [ex for ex in pool if _onpolicy_id(ex.id) not in done_ids]
        if not pending:
            logger.info("[on_policy:%s] nothing to do (%d done, %d in pool)",
                        level, len(done_ids), len(pool))
            results[level] = 0
            continue

        async def _run(ex: SFTExample) -> DPOExample | None:
            return await _generate_one_pair(
                sft=ex,
                target=target,
                judge=judge,
                min_margin=min_margin,
                target_max_tokens=target_max_tokens,
                target_temperature=target_temperature,
                judge_max_tokens=judge_max_tokens,
                judge_temperature=judge_temperature,
                failures_path=failures_path,
                audit_path=audit_path,
                generation_meta_base=generation_meta_base,
            )

        completed = await gather_with_concurrency(
            [_run(ex) for ex in pending],
            concurrency=concurrency,
            desc=f"on_policy[{level}]",
        )
        written = 0
        for pair in completed:
            if pair is None:
                continue
            append_jsonl(out_path, pair)
            written += 1
        results[level] = written
        logger.info("[on_policy:%s] wrote %d pairs (from %d candidates)",
                    level, written, len(pending))
    return results
