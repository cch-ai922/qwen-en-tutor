"""build_mixed_violation_probe.py — Phase 1, paper_v3 ("Don't Trim the Tail").

Builds the DECISIVE test for the typed-marker-as-semantic-gate claim (F-A).

The existing persistent_premature_probe is single-axis: only one violation
type is present, so a typed model can name the correct axis trivially. This
probe puts TWO axes in the same conversation:

  primary axis X   — escalates to the firing threshold (3 same-axis strikes)
  distractor axis Y — appears once (sub-threshold), spliced into early context

Two conditions per (X, seed):

  fire_correct     — context runs to X's 3rd strike. X is at threshold, Y is
                     not. Expected: FIRE, naming axis == X.
                     Tests (a) correct attribution under distraction and
                     (b) selective thresholding (ignore Y's pressure).

  distractor_sub   — context runs to X's 2nd strike (X sub-threshold) while Y
                     has appeared. Expected: DO NOT FIRE (neither at
                     threshold). Guards against "fire because *something* is
                     escalating."

A generic-marker model can fire but cannot be scored on attribution (bare
[SESSION_END]); that asymmetry is the point — it is measured, not fixed.

Schema matches the other eval sets so the existing generation runner works
unchanged. Extra expected fields:
  primary_axis, distractor_axis, condition ("fire_correct"|"distractor_sub").

DOES NOT RUN GENERATION. Only writes eval_sets/mixed_violation_probe.jsonl.

Usage:
  python scripts/build_mixed_violation_probe.py \
      --out eval_sets/mixed_violation_probe.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

# Reuse the frozen split + helpers from the main builder so this probe draws
# from exactly the same held-out pool and dialogue format.
from build_eval_sets import (  # type: ignore
    ROOT,
    LEVELS,
    PERSISTENT_STREAMS,
    in_eval_pool,
    extract_seed_id,
    extract_variant_tag,
    load_jsonl,
    write_jsonl,
    _find_sentinel_assistant_idx,
)


def _eval_seed_ids(data_root: Path) -> set[str]:
    """Recompute the held-out seed-id set the same way build_eval_sets does:
    every seed_id seen in the persistent streams that falls in the eval pool."""
    ids: set[str] = set()
    for stream in PERSISTENT_STREAMS:
        for level in LEVELS:
            for d in load_jsonl(data_root / "data" / "sft_raw" / f"{stream}_{level}.jsonl"):
                sid = extract_seed_id(d.get("id", ""))
                if sid and in_eval_pool(sid):
                    ids.add(sid)
    return ids


def _first_user_violation(msgs: list[dict[str, Any]], S: int) -> Optional[dict[str, Any]]:
    """The 1st same-axis violation user turn is at S-5 in a persistent
    dialogue (see build_persistent_premature_probe). Return that user
    message to use as a distractor turn, or None."""
    if S is None or S < 5:
        return None
    m = msgs[S - 5]
    return m if m.get("role") == "user" else None


def _pick_distractor(
    distractor_pool: dict[str, list[dict[str, Any]]],
    exclude_axis: str,
    level: str,
    idx: int,
) -> Optional[tuple[str, dict[str, Any]]]:
    """Deterministically choose a distractor (axis, user_turn) from a different
    axis at the same CEFR level. Rotates by idx so distractors vary."""
    candidates: list[tuple[str, dict[str, Any]]] = []
    for axis in PERSISTENT_STREAMS:
        if axis == exclude_axis:
            continue
        turns = distractor_pool.get(f"{axis}|{level}", [])
        if turns:
            candidates.append((axis, turns[idx % len(turns)]))
    if not candidates:
        return None
    # rotate the chosen axis by idx for variety, deterministic
    return candidates[idx % len(candidates)]


def build(data_root: Path, cap_per_axis_level: int = 8) -> list[dict[str, Any]]:
    eval_ids = _eval_seed_ids(data_root)

    # Pre-collect distractor user turns per (axis, level): the 1st violation
    # user turn of each held-out persistent dialogue.
    distractor_pool: dict[str, list[dict[str, Any]]] = {}
    for axis in PERSISTENT_STREAMS:
        for level in LEVELS:
            key = f"{axis}|{level}"
            bucket: list[dict[str, Any]] = []
            for d in load_jsonl(data_root / "data" / "sft_raw" / f"{axis}_{level}.jsonl"):
                sid = extract_seed_id(d.get("id", ""))
                if not sid or sid not in eval_ids:
                    continue
                msgs = d.get("messages", [])
                S = _find_sentinel_assistant_idx(msgs)
                uv = _first_user_violation(msgs, S) if S is not None else None
                if uv:
                    bucket.append(uv)
            bucket.sort(key=lambda m: m.get("content", ""))
            distractor_pool[key] = bucket

    # A short benign in-character redirect for the injected distractor turn.
    # Kept axis-neutral so it doesn't itself supply a 2nd strike signal.
    BENIGN_REDIRECT = ("Let's keep our focus here. Tell me more about our topic.")

    out: list[dict[str, Any]] = []
    for stream in PERSISTENT_STREAMS:  # primary axis X
        for level in LEVELS:
            cell: list[dict[str, Any]] = []
            dialogues = load_jsonl(
                data_root / "data" / "sft_raw" / f"{stream}_{level}.jsonl"
            )
            idx = 0
            for d in dialogues:
                sid = extract_seed_id(d.get("id", ""))
                if not sid or sid not in eval_ids:
                    continue
                msgs = d.get("messages", [])
                S = _find_sentinel_assistant_idx(msgs)
                if S is None or S < 5:
                    continue
                if msgs[S - 5].get("role") != "user":
                    continue

                picked = _pick_distractor(distractor_pool, stream, level, idx)
                if picked is None:
                    continue
                distractor_axis, distractor_user = picked
                idx += 1

                v = extract_variant_tag(d["id"])
                sp = d.get("system_prompt", "")
                locale = d.get("metadata", {}).get("locale", "china")

                # Inject the distractor (user Y-violation + benign redirect) at
                # the very start of the dialogue, before the setup turn, so the
                # primary X-strike sequence and its turn positions are preserved.
                injected = [
                    {"role": "user", "content": distractor_user.get("content", "")},
                    {"role": "assistant", "content": BENIGN_REDIRECT},
                ]

                # --- fire_correct: X at 3rd strike (context up to S), Y sub ---
                ctx_fire = injected + msgs[:S]
                cell.append({
                    "id": (f"mixed_violation_probe_{stream}_{sid}_{v}_{level}"
                           f"_X{stream}_Y{distractor_axis}_fire"),
                    "test_set": "mixed_violation_probe",
                    "cefr_level": level,
                    "locale": locale,
                    "system_prompt": sp,
                    "context_messages": ctx_fire,
                    "expected": {
                        "axis": stream,               # correct axis to name
                        "primary_axis": stream,
                        "distractor_axis": distractor_axis,
                        "condition": "fire_correct",
                        "violation_count": 3,         # X strikes
                        "should_fire": True,
                    },
                    "source": {
                        "seed_id": sid,
                        "source_dialogue_id": d["id"],
                        "stream": stream,
                        "distractor_stream": distractor_axis,
                    },
                })

                # --- distractor_sub: X at 2nd strike (context up to S-2) ---
                ctx_sub = injected + msgs[:S - 2]
                cell.append({
                    "id": (f"mixed_violation_probe_{stream}_{sid}_{v}_{level}"
                           f"_X{stream}_Y{distractor_axis}_sub"),
                    "test_set": "mixed_violation_probe",
                    "cefr_level": level,
                    "locale": locale,
                    "system_prompt": sp,
                    "context_messages": ctx_sub,
                    "expected": {
                        "axis": stream,
                        "primary_axis": stream,
                        "distractor_axis": distractor_axis,
                        "condition": "distractor_sub",
                        "violation_count": 2,         # X strikes (sub-threshold)
                        "should_fire": False,
                    },
                    "source": {
                        "seed_id": sid,
                        "source_dialogue_id": d["id"],
                        "stream": stream,
                        "distractor_stream": distractor_axis,
                    },
                })

            cell.sort(key=lambda r: r["id"])
            # cap balances fire/sub within a cell (they interleave by id suffix)
            out.extend(cell[: cap_per_axis_level * 2])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT))
    ap.add_argument("--out", default="eval_sets/mixed_violation_probe.jsonl")
    ap.add_argument("--cap-per-axis-level", type=int, default=8)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    records = build(data_root, cap_per_axis_level=args.cap_per_axis_level)
    out = Path(args.out)
    write_jsonl(out, records)

    # summary (no generation)
    n_fire = sum(1 for r in records if r["expected"]["condition"] == "fire_correct")
    n_sub = sum(1 for r in records if r["expected"]["condition"] == "distractor_sub")
    print(f"Wrote {out}: {len(records)} records ({n_fire} fire_correct, {n_sub} distractor_sub)")
    # axis-pair coverage
    from collections import Counter
    pairs = Counter(
        (r["expected"]["primary_axis"], r["expected"]["distractor_axis"]) for r in records
    )
    print("Primary×distractor axis pairs:")
    for (x, y), c in sorted(pairs.items()):
        print(f"  X={x:<32} Y={y:<32} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
