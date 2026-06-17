"""Offline rewrite-based DPO pair generation (6-axis, axis-cycling).

For each SFT example (both normal and redirect variants), this stage:

  1. Deterministically picks one assistant turn (hash-of-id selection, so
     re-runs against the same SFT id pick the same turn);
  2. Picks ONE spoil axis from ``SPOIL_AXES`` via ``hash(sft.id) % 6`` so
     coverage of all six ``RejectionAxis`` values is spread evenly across
     the SFT pool without exploding total volume;
  3. For ``cefr_mismatch`` further picks a direction (``too_hard`` for
     A1-A2, ``too_easy`` for C1-C2, hash-based for B1/B2);
  4. Sends the natural turn + context + per-axis spoil instructions to
     ``SPOIL_REWRITE_PROMPT`` and emits a DPOExample with the rewrite as
     ``rejected`` and the original turn as ``chosen`` under the chosen
     ``rejection_axis``.

Writes to ``data/dpo_raw/register_{level}.jsonl`` (filename retained for
pipeline-stage compatibility; contents now span all 6 axes, not just
``register_unnatural``).

Resumes from any existing output.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from qwen_tutor.generation._prompt_select import (
    AXIS_SPOIL_INSTRUCTIONS,
    REJECTION_NOTES,
    SPOIL_AXES,
    render_prompt,
    validate_prompt_has_locale_instruction,
)
from qwen_tutor.generation.teacher import TeacherClient, build_teacher_from_config
from qwen_tutor.schemas import DPOExample, ExampleMetadata, Message, SFTExample
from qwen_tutor.utils.runner import (
    append_failure,
    append_jsonl,
    extract_first_json,
    gather_with_concurrency,
    load_existing_ids,
)

logger = logging.getLogger(__name__)

DEFAULT_SFT_DIR = Path("data/sft_filtered")
DEFAULT_OUTPUT_DIR = Path("data/dpo_raw")
DEFAULT_FAILURES_PATH = Path("data/register_pairs_failures.jsonl")

# Smart axis fallback: if the first axis is a no-op or parse fails, retry
# once with the next axis in rotation. We cap this at three attempts because
# too many retries would blow up teacher token usage.
# (3 attempts is half of the 6 total axes.)
MAX_AXIS_ATTEMPTS = 3

# Per-axis maximum similarity. Reject the rewrite if its char-level
# SequenceMatcher ratio against the chosen turn is >= this axis's threshold.
#
# Each axis has a different natural "footprint":
#   register_unnatural  - stiff/textbook rewrite restructures word choice,
#                         similarity is naturally 0.4-0.7. Anything > 0.85
#                         is likely a no-op.
#   cefr_mismatch       - vocab swap is substantial. > 0.80 = no-op.
#   locale_violation    - proper-noun swap keeps most structure. > 0.92.
#   pedagogy_weak       - lecture mode adds bullets / restructures. > 0.85.
#   off_topic           - drift goes to an unrelated subject; should be
#                         very different. > 0.70 = the teacher kept most
#                         of the original content (this is the v6 bug).
#   accuracy_error      - real grammar errors are 1-3 char swaps in place,
#                         legitimate similarity is 0.95-0.99. Only byte-
#                         identical can be auto-rejected; weak tail-addition
#                         no-ops have to be caught by the few-shot prompt.
AXIS_SIMILARITY_LIMITS: dict[str, float] = {
    "register_unnatural":  0.85,
    "cefr_mismatch":       0.80,
    "locale_violation":    0.92,
    "pedagogy_weak":       0.85,
    "off_topic":           0.70,
    "accuracy_error":      1.00,
    # Translation / heavy code-switching naturally has VERY low char similarity
    # to English (often <0.30). Anything > 0.85 means the teacher kept too
    # much English and only swapped a token or two - that's a no-op for this
    # axis.
    "language_violation":  0.85,
    # persona_break inserts a recognizable "As an AI..." phrase or refusal,
    # restructuring at least one sentence. Anything > 0.88 means the teacher
    # only tagged on a trailing disclaimer without rewriting.
    "persona_break":       0.88,
    # role_swap_accepted: tutor flips its frame of reference (sells -> buys,
    # advises -> asks for advice). That's substantial restructuring; > 0.85
    # means the teacher mostly kept the original turn and just added a
    # surrender clause.
    "role_swap_accepted":  0.85,
}
# Fallback when an unknown axis is passed in (e.g. future axis additions).
DEFAULT_SIMILARITY_LIMIT = 0.92


def _rewrite_similarity(a: str, b: str) -> float:
    """Char-level SequenceMatcher ratio. 1.0 = identical."""
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()


def _is_near_no_op(rewritten: str, chosen: str, axis: str) -> tuple[bool, float, float]:
    """Returns (is_no_op, similarity, threshold) for the given axis."""
    sim = _rewrite_similarity(rewritten, chosen)
    threshold = AXIS_SIMILARITY_LIMITS.get(axis, DEFAULT_SIMILARITY_LIMIT)
    return sim >= threshold, sim, threshold


def _existing_axis_counts(out_path: Path) -> dict[str, int]:
    """Count the axes of pairs already written to disk.

    Used as the initial quota balance when resuming generation.
    """
    counts = {ax: 0 for ax in SPOIL_AXES}
    if not out_path.exists():
        return counts
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ax = rec.get("rejection_axis")
            if ax in counts:
                counts[ax] += 1
    return counts


def _assign_axes_quota_balanced(
    pending: list[SFTExample],
    initial_counts: dict[str, int],
    axes: tuple[str, ...],
) -> list[tuple[SFTExample, str]]:
    """Assign one starting axis per SFT to push the batch toward uniform.

    Walk the SFTs in id order and choose the applicable axis with the lowest
    count so far. Ties are broken by the order in ``axes`` to keep the result
    deterministic. ``initial_counts`` reflects the axis distribution of pairs
    already written to disk, so reruns and resumes stay balanced.

    Returns a list of ``(sft, assigned_axis)`` pairs. The actual axis used in
    generation may differ if the chosen axis fails as a no-op and
    ``_generate_one`` rotates to another axis, but the overall distribution
    stays very balanced.
    """
    counts = {ax: initial_counts.get(ax, 0) for ax in axes}
    axes_order = {ax: i for i, ax in enumerate(axes)}
    out: list[tuple[SFTExample, str]] = []
    for sft in sorted(pending, key=lambda s: s.id):
        try:
            turn_idx = _pick_assistant_turn_index(sft)
        except ValueError:
            # Abnormal case: the SFT has no assistant turn. Start with the
            # first axis and let _generate_one handle the actual failure.
            out.append((sft, axes[0]))
            continue
        chosen_text = sft.messages[turn_idx].content
        applicable = _axes_applicable_to_turn(chosen_text, axes)
        axis = min(applicable, key=lambda a: (counts[a], axes_order[a]))
        counts[axis] += 1
        out.append((sft, axis))
    return out


def _dpo_id(sft_id: str) -> str:
    return f"dpo_register_{sft_id}"


def _hash_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _pick_axis(sft_id: str, axes: tuple[str, ...]) -> str:
    return axes[_hash_int(sft_id) % len(axes)]


def _pick_cefr_direction(sft_id: str, cefr_level: str) -> str:
    """Returns ``"too_hard"`` or ``"too_easy"`` for the ``cefr_mismatch`` axis.

    Low levels (A1-A2) always get ``too_hard`` (models tend to over-explain).
    High levels (C1-C2) always get ``too_easy`` (models dumb down). B1/B2 mix
    50/50 by hash for variety.
    """
    if cefr_level in ("A1", "A2"):
        return "too_hard"
    if cefr_level in ("C1", "C2"):
        return "too_easy"
    return "too_hard" if _hash_int(sft_id + ":dir") % 2 == 0 else "too_easy"


def _pick_assistant_turn_index(sft: SFTExample) -> int:
    """Pick the LONGEST assistant turn (hash as tiebreaker, for determinism).

    Earlier versions hash-picked an arbitrary assistant turn, which often
    landed on a 3-word acknowledgment (\"Yes. Thanks!\") and starved the
    accuracy_error / locale_violation spoil axes of raw material. Picking
    the longest assistant turn gives every axis enough text to work with.

    DPO-safety: the chosen side is always the natural turn (correct), so
    biasing toward longer chosens does NOT teach the model \"long = bad\";
    rejected variants on the same turn are spoiled in axis-specific ways.
    """
    idxs = [i for i, m in enumerate(sft.messages) if m.role == "assistant"]
    if not idxs:
        raise ValueError("SFT example has no assistant turns")
    h = int(hashlib.sha256(sft.id.encode("utf-8")).hexdigest()[:8], 16)
    # Sort by (word count desc, deterministic hash-derived tiebreaker asc)
    return sorted(
        idxs,
        key=lambda i: (-len(sft.messages[i].content.split()), (h + i) & 0xFFFF),
    )[0]


# A turn is "rich enough" for accuracy_error / locale_violation only if it has
# the raw material those axes need to spoil. The thresholds are loose - the
# spoil prompt still gets the no-op check as a safety net.
_MIN_WORDS_FOR_ACCURACY = 8
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")

# Sentence-initial common words that the proper-noun regex would otherwise
# match. Same list as locale_judge keeps for cheap NER pre-filtering.
_PROPER_NOUN_STOPWORDS: frozenset[str] = frozenset(
    s.lower()
    for s in (
        "Yes", "No", "Oh", "Well", "OK", "Okay", "Sure", "Hi", "Hello", "Hey",
        "Bye", "Thanks", "Thank", "Goodbye", "Please", "Sorry",
        "When", "Where", "Why", "How", "What", "Who", "Which", "If", "Or",
        "And", "But", "So", "Then", "There", "Here", "This", "That", "These",
        "Those", "The", "A", "An", "I", "You", "He", "She", "We", "They",
        "It", "My", "Your", "His", "Her", "Our", "Their", "Some", "Any",
        "Today", "Tomorrow", "Yesterday", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday",
    )
)


def _has_real_proper_noun(text: str) -> bool:
    """True if ``text`` has at least one capitalized token that is NOT a
    common sentence-initial stopword. Used to gate locale_violation."""
    for m in _PROPER_NOUN_RE.finditer(text):
        if m.group(0).lower() not in _PROPER_NOUN_STOPWORDS:
            return True
    return False


def _axes_applicable_to_turn(turn_text: str, axes: tuple[str, ...]) -> tuple[str, ...]:
    """Return the subset of ``axes`` whose spoils have raw material in this turn.

    Used as a pre-screen so the cycle does not waste its first attempt on an
    obviously-unsuitable axis. The fallback loop still tries remaining axes
    if the chosen one fails for other reasons.
    """
    words = turn_text.split()
    has_enough_words = len(words) >= _MIN_WORDS_FOR_ACCURACY
    has_proper_noun = _has_real_proper_noun(turn_text)
    out: list[str] = []
    for axis in axes:
        if axis == "accuracy_error" and not has_enough_words:
            continue
        if axis == "locale_violation" and not has_proper_noun:
            continue
        out.append(axis)
    return tuple(out) if out else axes  # safety fallback: never return empty


def _format_context(messages: list[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        role = "User" if m.role == "user" else ("Tutor" if m.role == "assistant" else m.role.capitalize())
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def _parse_rewrite(raw: str) -> str:
    data = extract_first_json(raw)
    if not isinstance(data, dict) or "rewritten" not in data:
        raise ValueError("rewrite response missing top-level 'rewritten' field")
    rewritten = data["rewritten"]
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("'rewritten' is empty or not a string")
    return rewritten


def _iter_sft_examples(
    sft_dir: Path, levels: list[str]
) -> list[tuple[str, SFTExample]]:
    """Walk all SFT prefixes for the requested levels. Each is a source of
    natural assistant turns that DPO can spoil along the rejection axes.

    Reads from the FILTERED SFT directory (``*_passed.jsonl``) so the
    ``chosen`` assistant turns we derive from these examples already
    passed every SFT filter (banned_terms, non_latin_script, etc.).
    This mirrors what ``on_policy_pairs.py`` does, and means the
    downstream ``filter_dpo`` stage is largely redundant on
    ``register_*`` records — the only content we add at this stage
    is the ``rejected`` (the deliberately-spoiled rewrite), which the
    DPO filter is configured to skip on purpose.

    The pipeline supports a filter-pipeline wrapper ``{"example": ...,
    "pipeline": ...}`` from the filter stage as well as raw SFT JSONL,
    so callers can also point this at ``data/sft_raw/`` directly if
    they want to operate on unfiltered SFT.
    """
    out: list[tuple[str, SFTExample]] = []
    for level in levels:
        for prefix in (
            "normal",
            "redirect",
            "locale_redirect",
            "pedagogy_redirect",
            "language_redirect",
            "persona_redirect",
            "topic_redirect",
            "role_swap_redirect",
            # 4 persistent 3-strike streams (Option B). The new
            # cave_on_persistence / verbatim_repeat / lecture_on_persistence
            # axes (Option C) are particularly meaningful when applied to
            # the refusal turns in these examples, but they also work on
            # any other assistant turn — the axis selector cycles by
            # hash(sft.id) and accepts whichever pair the teacher can spoil.
            "persistent_off_topic",
            "persistent_language_violation",
            "persistent_persona_break",
            "persistent_role_swap",
        ):
            # Try filtered shape first (``<prefix>_<level>_passed.jsonl``),
            # fall back to raw shape (``<prefix>_<level>.jsonl``) so this
            # helper still works if pointed at ``data/sft_raw/``.
            candidates = [
                sft_dir / f"{prefix}_{level}_passed.jsonl",
                sft_dir / f"{prefix}_{level}.jsonl",
            ]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Unwrap the filter-pipeline ``{"example": ...,
                    # "pipeline": ...}`` envelope when present.
                    if isinstance(rec, dict) and isinstance(rec.get("example"), dict):
                        rec = rec["example"]
                    try:
                        out.append((level, SFTExample.model_validate(rec)))
                    except Exception:  # noqa: BLE001
                        continue
    return out


async def _try_one_axis(
    *,
    teacher: TeacherClient,
    sft: SFTExample,
    axis: str,
    idx: int,
    chosen_msg: Message,
    context_msgs: list[Message],
    max_tokens: int,
    temperature: float,
) -> tuple[str, str | None]:
    """Run a single axis attempt. Returns ``(rewritten_text, direction)`` on
    success or raises ``ValueError`` on no-op / parse failure / teacher error.

    Caller is responsible for the per-axis try/except in the fallback loop.
    """
    if axis == "cefr_mismatch":
        direction = _pick_cefr_direction(sft.id, sft.metadata.cefr_level)
        instruction_key = f"cefr_mismatch_{direction}"
        axis_label = f"cefr_mismatch ({direction})"
    else:
        direction = None
        instruction_key = axis
        axis_label = axis

    axis_instructions = AXIS_SPOIL_INSTRUCTIONS[instruction_key].format(
        cefr_level=sft.metadata.cefr_level,
    )
    locale = sft.metadata.locale
    template = render_prompt("spoil_rewrite", locale_name=locale)
    prompt = template.format(
        natural_turn=chosen_msg.content,
        cefr_level=sft.metadata.cefr_level,
        context=_format_context(context_msgs) or "(no prior turns)",
        axis_label=axis_label,
        axis_instructions=axis_instructions,
    )
    validate_prompt_has_locale_instruction(prompt, locale_name=locale)
    raw = await teacher.generate(
        system=prompt,
        messages=[],
        cacheable_prefix=None,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    rewritten = _parse_rewrite(raw)
    # Reject near-no-op rewrites. The per-axis threshold lets accuracy_error
    # keep its legitimate 1-word in-place grammar errors (sim ~0.97) while
    # still catching off_topic / locale_violation no-ops where the teacher
    # only tweaked a trailing word (the v6 live-data bug).
    is_no_op, sim, threshold = _is_near_no_op(
        rewritten, chosen_msg.content, axis
    )
    if is_no_op:
        raise ValueError(
            f"axis={axis} produced a near-no-op rewrite "
            f"(similarity {sim:.3f} >= axis threshold {threshold:.2f})"
        )
    return rewritten, direction


async def _generate_one(
    teacher: TeacherClient,
    sft: SFTExample,
    max_tokens: int,
    temperature: float,
    failures_path: Path,
    generation_meta_base: dict[str, Any],
    axes: tuple[str, ...] = SPOIL_AXES,
    max_axis_attempts: int = MAX_AXIS_ATTEMPTS,
    assigned_axis: str | None = None,
) -> DPOExample | None:
    """Generate ONE DPO pair, falling back to other axes if the first fails.

    Starting axis is either ``assigned_axis`` (when caller did quota-balanced
    pre-assignment) or hash-deterministic from ``sft.id``. On no-op / parse /
    teacher error, the next axis in the applicable cycle is tried, up to
    ``max_axis_attempts`` times. This rescues the cases where a short clean
    turn defeats axis-specific spoils (e.g. accuracy_error on a 5-word
    acknowledgment), keeping pair volume high without producing zero-signal
    no-op pairs.
    """
    try:
        idx = _pick_assistant_turn_index(sft)
    except ValueError as exc:
        append_failure(
            failures_path,
            _dpo_id(sft.id),
            f"{type(exc).__name__}: {exc}",
            level=sft.metadata.cefr_level,
            stage="register_pairs",
            sft_id=sft.id,
        )
        return None
    chosen_msg = sft.messages[idx]
    context_msgs = list(sft.messages[:idx])

    # Pre-screen: drop axes the chosen turn cannot meaningfully spoil so the
    # first attempt is not wasted on a structurally-unsuitable axis. The full
    # applicable list is also the fallback pool below.
    applicable = _axes_applicable_to_turn(chosen_msg.content, axes)
    # Starting axis: caller's quota-balanced assignment if provided + applicable,
    # else hash-deterministic. Fall back to hash if the assigned axis is no
    # longer applicable (defensive).
    if assigned_axis is not None and assigned_axis in applicable:
        start_idx = applicable.index(assigned_axis)
    else:
        start_idx = _hash_int(sft.id) % len(applicable)

    n_tries = min(max_axis_attempts, len(applicable))
    attempt_failures: list[str] = []
    for attempt in range(n_tries):
        axis = applicable[(start_idx + attempt) % len(applicable)]
        try:
            rewritten, direction = await _try_one_axis(
                teacher=teacher,
                sft=sft,
                axis=axis,
                idx=idx,
                chosen_msg=chosen_msg,
                context_msgs=context_msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            attempt_failures.append(f"axis={axis}: {type(exc).__name__}: {exc}")
            continue

        # success — emit a DPOExample on this axis. Propagate locale and
        # key SFT generation flags (e.g., language_trigger) from the source
        # SFT so downstream filters can make the correct exemptions.
        sft_gen_meta = (getattr(sft.metadata, "generation", None) or {})
        generation_fields = {
            **generation_meta_base,
            "source_sft_id": sft.id,
            "chosen_turn_index": idx,
            "spoil_axis": axis,
            "spoil_direction": direction,
            "axis_attempt": attempt,
            "axis_attempts_failed": attempt_failures,
        }
        # Propagate language_trigger when present (e.g., speaks_l1), so
        # script-check filters can exempt user turns appropriately.
        if isinstance(sft_gen_meta, dict) and "language_trigger" in sft_gen_meta:
            generation_fields["language_trigger"] = sft_gen_meta["language_trigger"]

        metadata = ExampleMetadata(
            topic=sft.metadata.topic,
            subtopics=list(sft.metadata.subtopics),
            user_role=sft.metadata.user_role,
            model_role=sft.metadata.model_role,
            cefr_level=sft.metadata.cefr_level,
            scenario_type=sft.metadata.scenario_type,
            locale=sft.metadata.locale,
            category=sft.metadata.category,
            generation=generation_fields,
        )
        return DPOExample(
            id=_dpo_id(sft.id),
            metadata=metadata,
            system_prompt=sft.system_prompt,
            prompt_messages=list(context_msgs),
            chosen=chosen_msg,
            rejected=Message(role="assistant", content=rewritten),
            rejection_axis=axis,
            rejection_note=REJECTION_NOTES[axis],
        )

    # All fallback attempts exhausted — log and give up.
    append_failure(
        failures_path,
        _dpo_id(sft.id),
        f"all {n_tries} axis attempts failed: " + " | ".join(attempt_failures),
        level=sft.metadata.cefr_level,
        stage="register_pairs",
        sft_id=sft.id,
        attempts=n_tries,
    )
    return None


async def generate_batch(
    cefr_levels: list[str] | None = None,
    sft_dir: str | Path = DEFAULT_SFT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    failures_path: str | Path = DEFAULT_FAILURES_PATH,
    config_path: str | Path = "config/generation.yaml",
    concurrency: int = 20,
    max_tokens: int = 1024,
    temperature: float = 0.4,
    teacher: TeacherClient | None = None,
    axes: tuple[str, ...] = SPOIL_AXES,
) -> dict[str, int]:
    """Generate one rewrite-based DPO pair per SFT example, cycling axes.

    ``axes`` is the list of spoil axes to use. Defaults to the 6 axes in
    ``SPOIL_AXES``. Each SFT example is assigned an axis by
    ``hash(sft.id) % len(axes)``. If you want to use only one axis, pass
    ``axes=("register_unnatural",)``.

    Returns ``{level: n_written}`` for this run.
    """
    if teacher is None:
        teacher = build_teacher_from_config(config_path, role="teacher")
    cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_path)

    generation_meta_base = {
        "provider": teacher.config.provider,
        "model": teacher.config.model,
        "prompt": "SPOIL_REWRITE_PROMPT",
        "axes": list(axes),
    }

    results: dict[str, int] = {}
    sft_examples = _iter_sft_examples(Path(sft_dir), cefr_levels)
    by_level: dict[str, list[SFTExample]] = {lvl: [] for lvl in cefr_levels}
    for level, ex in sft_examples:
        by_level[level].append(ex)

    for level in cefr_levels:
        out_path = output_dir / f"register_{level}.jsonl"
        done_ids = load_existing_ids(out_path)
        pending = [ex for ex in by_level[level] if _dpo_id(ex.id) not in done_ids]
        if not pending:
            logger.info("[register:%s] nothing to do (%d done)", level, len(done_ids))
            results[level] = 0
            continue

        # Quota-balanced axis assignment. Start with the axis counts of pairs
        # already on disk, then choose the applicable axis with the lowest
        # current count for each pending SFT. Smart fallback only activates if
        # that axis fails.
        initial_counts = _existing_axis_counts(out_path)
        assignments = _assign_axes_quota_balanced(pending, initial_counts, axes)
        logger.info(
            "[register:%s] quota-balanced axes (incl. existing): %s",
            level,
            {ax: sum(1 for _, a in assignments if a == ax) + initial_counts[ax] for ax in axes},
        )

        async def _run(ex: SFTExample, assigned: str) -> DPOExample | None:
            return await _generate_one(
                teacher=teacher,
                sft=ex,
                max_tokens=max_tokens,
                temperature=temperature,
                failures_path=failures_path,
                generation_meta_base=generation_meta_base,
                axes=axes,
                assigned_axis=assigned,
            )

        completed = await gather_with_concurrency(
            [_run(ex, assigned) for ex, assigned in assignments],
            concurrency=concurrency,
            desc=f"register[{level}]",
        )

        written = 0
        for dpo in completed:
            if dpo is None:
                continue
            append_jsonl(out_path, dpo)
            written += 1
        results[level] = written
        logger.info("[register:%s] wrote %d new pairs", level, written)
    return results
