"""Sentinel-aware DPO pair generation for the four ``persistent_*`` SFT streams.

Two pools, both written to ``data/dpo_raw/sentinel_<level>.jsonl``:

  * **Option A (offline)** — :func:`generate_offline_batch`. ``rejected``
    is the teacher's sentinel-firing response with the
    ``[SESSION_END: <axis>]`` marker stripped. Reachable from the SFT
    student's distribution (it is normal refusal text), so DPO loss is
    stable. Zero teacher / policy calls — purely deterministic text
    rewriting over the filtered persistent_* SFT records.

  * **Option B (on-policy)** — :func:`generate_on_policy_batch`.
    ``rejected`` is the SFT-trained student's regeneration of the
    sentinel-firing turn. The judge step is skipped because the marker's
    presence is mechanical ground truth: the teacher response carries
    ``[SESSION_END: <axis>]``, the student's regeneration almost never
    does, and the contrast on that single bracketed substring is the
    signal we want to teach. Pairs where the student *already* emits
    the marker are dropped (zero-signal).

Both pools target the same problem: ``[SESSION_END: ...]`` appears in
roughly 3–5 % of assistant turns in the SFT mix (only the third-strike
turn of the four persistent_* streams), so its per-token cross-entropy
gradient is weak. A student trained on SFT alone reliably learns to
*end* with a polite refusal but only sometimes emits the literal marker
that the external dispatcher matches on. The contrast in these pairs
(marker present vs absent, everything else equal in Option A, or
realistic SFT-student text in Option B) pushes the marker's conditional
probability up without disrupting other behaviors.

Both functions are resumable: ids already present in the output JSONL
are skipped, and the two id schemes (``dpo_sentinel_offline_<sft_id>``
and ``dpo_sentinel_onpolicy_<sft_id>``) coexist in the same file.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Protocol

from qwen_tutor.generation.prompts import (
    render_scenario_deployment_system_prompt,
)
from qwen_tutor.schemas import (
    DPOExample,
    ExampleMetadata,
    Message,
    SFTExample,
)
from qwen_tutor.utils.runner import (
    append_failure,
    append_jsonl,
    gather_with_concurrency,
    load_existing_ids,
)

logger = logging.getLogger(__name__)

DEFAULT_SFT_FILTERED_DIR = Path("data/sft_filtered")
DEFAULT_OUTPUT_DIR = Path("data/dpo_raw")
DEFAULT_FAILURES_PATH = Path("data/sentinel_pairs_failures.jsonl")

# The four persistent_* SFT stream prefixes that emit sentinels. Kept in
# sync with persistent_redirect.py and on_policy_pairs.py.
PERSISTENT_PREFIXES: tuple[str, ...] = (
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
)

# Greedy on the axis label; tolerates extra whitespace inside the brackets.
# We match a leading whitespace cluster too, so the trailing space before
# the marker is collapsed alongside the marker itself.
_SENTINEL_RE = re.compile(r"\s*\[SESSION_END:\s*[A-Za-z_]+\s*\]\s*")


# ---------------------------------------------------------------------------
# Target-model protocol (matches on_policy_pairs.TargetModelClient)
# ---------------------------------------------------------------------------


class TargetModelClient(Protocol):
    """Anything that can generate from the SFT-trained student policy."""

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
# ID schemes
# ---------------------------------------------------------------------------


def _offline_id(sft_id: str) -> str:
    return f"dpo_sentinel_offline_{sft_id}"


def _onpolicy_id(sft_id: str) -> str:
    return f"dpo_sentinel_onpolicy_{sft_id}"


# ---------------------------------------------------------------------------
# Sentinel-turn lookup + marker stripping
# ---------------------------------------------------------------------------


def _find_sentinel_turn(sft: SFTExample) -> tuple[int, str]:
    """Return ``(turn_index, sentinel_marker)`` for the sentinel-firing turn.

    Reads ``metadata.generation.sentinel_turn`` (the absolute message
    index) and ``metadata.generation.sentinel`` (the literal marker
    string e.g. ``[SESSION_END: persistent_language_violation]``).
    Raises :class:`ValueError` if either is missing or the indexed turn
    is not an assistant turn containing a ``[SESSION_END`` substring
    (defensive — keeps malformed records out of the DPO pool).
    """
    gen = sft.metadata.generation or {}
    turn_idx = gen.get("sentinel_turn")
    marker = gen.get("sentinel")
    if not isinstance(turn_idx, int) or not isinstance(marker, str):
        raise ValueError(
            "SFT record missing metadata.generation.{sentinel_turn,sentinel}"
        )
    if not (0 <= turn_idx < len(sft.messages)):
        raise ValueError(f"sentinel_turn={turn_idx} out of message range")
    turn = sft.messages[turn_idx]
    if turn.role != "assistant":
        raise ValueError(
            f"sentinel turn at index {turn_idx} has role {turn.role!r}, not 'assistant'"
        )
    if "[SESSION_END" not in turn.content:
        raise ValueError(
            f"sentinel turn at index {turn_idx} lacks a SESSION_END marker"
        )
    return turn_idx, marker


def _strip_sentinel(content: str) -> str:
    """Remove every ``[SESSION_END: ...]`` marker from ``content``.

    Collapses whitespace produced by the substitution. Returns the
    stripped text. If the result would be empty (a sentinel-only turn),
    falls back to a brief politeness so the rejected message is still
    non-empty (DPO trainer dislikes empty assistant strings).
    """
    stripped = _SENTINEL_RE.sub(" ", content)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if not stripped:
        return "Let's keep going."
    return stripped


# ---------------------------------------------------------------------------
# Persistent SFT loading
# ---------------------------------------------------------------------------


def _iter_persistent_sft(
    sft_filtered_dir: Path, cefr_levels: list[str]
) -> list[tuple[str, SFTExample]]:
    """Walk the four persistent_* SFT prefixes (passed-only)."""
    out: list[tuple[str, SFTExample]] = []
    for level in cefr_levels:
        for prefix in PERSISTENT_PREFIXES:
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
                        logger.warning(
                            "sentinel_pairs: dropping malformed record: %s", exc
                        )
                        continue
                    out.append((level, ex))
    return out


def _build_metadata(
    sft: SFTExample, generation_base: dict[str, Any], extra: dict[str, Any]
) -> ExampleMetadata:
    return ExampleMetadata(
        topic=sft.metadata.topic,
        subtopics=list(sft.metadata.subtopics),
        user_role=sft.metadata.user_role,
        model_role=sft.metadata.model_role,
        cefr_level=sft.metadata.cefr_level,
        scenario_type=sft.metadata.scenario_type,
        locale=sft.metadata.locale,
        category=sft.metadata.category,
        generation={**generation_base, **extra, "source_sft_id": sft.id},
    )


# ---------------------------------------------------------------------------
# Option A: offline (strip-marker)
# ---------------------------------------------------------------------------


def _make_offline_pair(
    sft: SFTExample, generation_base: dict[str, Any]
) -> DPOExample | None:
    """Build one Option-A pair from a persistent SFT record.

    Returns ``None`` if the record is malformed (no sentinel marker /
    sentinel_turn out of range) or if stripping the marker produces a
    zero-contrast pair (defensive — should not happen in practice).
    """
    try:
        turn_idx, marker = _find_sentinel_turn(sft)
    except ValueError:
        return None
    chosen_msg = sft.messages[turn_idx]
    rejected_text = _strip_sentinel(chosen_msg.content)
    if rejected_text.strip() == chosen_msg.content.strip():
        return None
    return DPOExample(
        id=_offline_id(sft.id),
        metadata=_build_metadata(
            sft,
            generation_base,
            {
                "sentinel_marker": marker,
                "chosen_turn_index": turn_idx,
                "mode": "offline",
            },
        ),
        system_prompt=sft.system_prompt,
        prompt_messages=list(sft.messages[:turn_idx]),
        chosen=chosen_msg,
        rejected=Message(role="assistant", content=rejected_text),
        rejection_axis="sentinel_missing",
        rejection_note=(
            "Sentinel-DPO (offline): the chosen response carries the "
            f"{marker} marker on the third-strike turn; rejected is the "
            "same response with the marker stripped, simulating the "
            "common SFT-only failure mode of producing a polite refusal "
            "that never closes the session."
        ),
    )


def generate_offline_batch(
    cefr_levels: list[str] | None = None,
    sft_filtered_dir: str | Path = DEFAULT_SFT_FILTERED_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    failures_path: str | Path = DEFAULT_FAILURES_PATH,
) -> dict[str, int]:
    """Generate offline (strip-marker) sentinel DPO pairs.

    No teacher / policy calls — purely text manipulation over the
    filtered ``persistent_*`` SFT records. Writes to
    ``data/dpo_raw/sentinel_<level>.jsonl``; resumable via id lookup.

    Returns ``{level: n_written}`` for this run.
    """
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)
    sft_filtered_dir = Path(sft_filtered_dir)
    generation_base = {"stage": "sentinel_pairs", "mode": "offline"}

    sft_examples = _iter_persistent_sft(sft_filtered_dir, cefr_levels)
    by_level: dict[str, list[SFTExample]] = {lvl: [] for lvl in cefr_levels}
    for level, ex in sft_examples:
        by_level[level].append(ex)

    results: dict[str, int] = {}
    for level in cefr_levels:
        out_path = output_dir / f"sentinel_{level}.jsonl"
        done = load_existing_ids(out_path)
        pending = [ex for ex in by_level[level] if _offline_id(ex.id) not in done]
        written = 0
        for ex in pending:
            pair = _make_offline_pair(ex, generation_base)
            if pair is None:
                append_failure(
                    failures_path,
                    _offline_id(ex.id),
                    "offline sentinel pair build failed (no marker / out-of-range)",
                    level=level,
                    stage="sentinel_pairs_offline",
                    sft_id=ex.id,
                )
                continue
            append_jsonl(out_path, pair)
            written += 1
        results[level] = written
        logger.info(
            "[sentinel_offline:%s] wrote %d new pairs (pool=%d, already done=%d)",
            level, written, len(by_level[level]), len(done),
        )
    return results


# ---------------------------------------------------------------------------
# Option B: on-policy (regen, no judge)
# ---------------------------------------------------------------------------


async def _generate_on_policy_pair(
    sft: SFTExample,
    target: TargetModelClient,
    generation_base: dict[str, Any],
    *,
    target_max_tokens: int,
    target_temperature: float,
    failures_path: Path,
) -> DPOExample | None:
    """Build one Option-B pair from a persistent SFT record.

    Returns ``None`` if the record is malformed, the target raises, or
    the regenerated policy response already contains a SESSION_END
    marker (in which case the pair would be zero-signal).
    """
    try:
        turn_idx, marker = _find_sentinel_turn(sft)
    except ValueError:
        return None
    chosen_msg = sft.messages[turn_idx]
    prefix_messages = sft.messages[:turn_idx]
    system_prompt = render_scenario_deployment_system_prompt(
        cefr_level=sft.metadata.cefr_level,
        locale_name=sft.metadata.locale,
        topic=sft.metadata.topic,
        subtopics=sft.metadata.subtopics,
        user_role_name=sft.metadata.user_role.name,
        user_role_description=sft.metadata.user_role.description,
        model_role_name=sft.metadata.model_role.name,
        model_role_description=sft.metadata.model_role.description,
    )
    try:
        policy_text = await target.generate(
            system=system_prompt,
            messages=[
                {"role": m.role, "content": m.content} for m in prefix_messages
            ],
            mode="conversation",
            max_new_tokens=target_max_tokens,
            temperature=target_temperature,
        )
    except Exception as exc:  # noqa: BLE001
        append_failure(
            failures_path,
            _onpolicy_id(sft.id),
            f"target.generate raised {type(exc).__name__}: {exc}",
            level=sft.metadata.cefr_level,
            stage="sentinel_pairs_onpolicy",
            sft_id=sft.id,
        )
        return None
    policy_text = (policy_text or "").strip()
    if not policy_text:
        append_failure(
            failures_path,
            _onpolicy_id(sft.id),
            "policy produced empty output",
            level=sft.metadata.cefr_level,
            stage="sentinel_pairs_onpolicy",
            sft_id=sft.id,
        )
        return None
    # If the policy already emits the marker on its own, there is no
    # gradient to add — skip rather than build a near-zero-contrast pair.
    if "[SESSION_END" in policy_text:
        return None
    return DPOExample(
        id=_onpolicy_id(sft.id),
        metadata=_build_metadata(
            sft,
            generation_base,
            {
                "sentinel_marker": marker,
                "chosen_turn_index": turn_idx,
                "mode": "on_policy",
            },
        ),
        system_prompt=sft.system_prompt,
        prompt_messages=list(prefix_messages),
        chosen=chosen_msg,
        rejected=Message(role="assistant", content=policy_text),
        rejection_axis="sentinel_missing",
        rejection_note=(
            "Sentinel-DPO (on-policy): policy regeneration of the sentinel-"
            f"firing turn omits the {marker} marker; teacher response with "
            "the marker is the chosen side. Judge step skipped because the "
            "marker's presence is mechanical ground truth."
        ),
    )


async def generate_on_policy_batch(
    target: TargetModelClient,
    *,
    cefr_levels: list[str] | None = None,
    sft_filtered_dir: str | Path = DEFAULT_SFT_FILTERED_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    failures_path: str | Path = DEFAULT_FAILURES_PATH,
    concurrency: int = 4,
    target_max_tokens: int = 320,
    target_temperature: float = 0.7,
) -> dict[str, int]:
    """Generate on-policy sentinel DPO pairs.

    Requires the SFT-trained student as ``target``. For each filtered
    ``persistent_*`` SFT record, regenerates the sentinel-firing turn
    from the student and pairs it against the teacher's marker-bearing
    turn. The judge step from ``on_policy_pairs`` is skipped because
    the marker's presence is mechanical ground truth: the teacher wins
    by construction whenever the policy fails to emit the marker.
    Pairs where the policy already emits the marker are dropped.

    Returns ``{level: n_written}`` for this run.
    """
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)
    sft_filtered_dir = Path(sft_filtered_dir)
    generation_base = {"stage": "sentinel_pairs", "mode": "on_policy"}

    sft_examples = _iter_persistent_sft(sft_filtered_dir, cefr_levels)
    by_level: dict[str, list[SFTExample]] = {lvl: [] for lvl in cefr_levels}
    for level, ex in sft_examples:
        by_level[level].append(ex)

    results: dict[str, int] = {}
    for level in cefr_levels:
        out_path = output_dir / f"sentinel_{level}.jsonl"
        done = load_existing_ids(out_path)
        pending = [ex for ex in by_level[level] if _onpolicy_id(ex.id) not in done]
        if not pending:
            logger.info(
                "[sentinel_onpolicy:%s] nothing to do (%d already done)",
                level, len(done),
            )
            results[level] = 0
            continue

        async def _run(ex: SFTExample) -> DPOExample | None:
            return await _generate_on_policy_pair(
                ex,
                target,
                generation_base,
                target_max_tokens=target_max_tokens,
                target_temperature=target_temperature,
                failures_path=failures_path,
            )

        completed = await gather_with_concurrency(
            [_run(ex) for ex in pending],
            concurrency=concurrency,
            desc=f"sentinel_onpolicy[{level}]",
        )
        written = 0
        for pair in completed:
            if pair is None:
                continue
            append_jsonl(out_path, pair)
            written += 1
        results[level] = written
        logger.info(
            "[sentinel_onpolicy:%s] wrote %d new pairs (from %d candidates)",
            level, written, len(pending),
        )
    return results
