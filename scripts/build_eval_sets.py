"""build_eval_sets.py — Freeze held-out test sets for paper evaluation.

Produces six JSONL files under ``eval_sets/`` plus a manifest:

  tutor_scenario.jsonl              — cold-start dialogues (each baseline
                                       generates 12 turns)
  redirect_probe.jsonl              — partial dialogue ending in user abuse
                                       turn; baseline produces the next
                                       assistant turn
  persistent_probe.jsonl            — partial dialogue near the sentinel-
                                       firing point (positive); baseline
                                       must produce the sentinel-firing turn
  persistent_fp_probe.jsonl         — NEGATIVE controls: benign dialogues
                                       truncated at a TRAINED sentinel
                                       position (5/7/9/11) WITHOUT three
                                       same-axis strikes. Baseline should
                                       NOT fire. Supplies the negatives that
                                       make precision and FP-rate well-
                                       defined (§4.8 of the paper).
  persistent_offposition_probe.jsonl — POSITIVES whose third strike lands at
                                       a turn outside {5,7,9,11} (e.g. 13,
                                       15) by extending lead-in scaffolding.
                                       Baseline should fire. Tests whether
                                       the model learned the trigger or only
                                       the four trained positions.
  locale_leakage.jsonl              — cold-start with locale=china; metric
                                       is Western-default leakage rate

Train/eval split is hash-deterministic on seed_id (20% held-out). The same
seed_ids are always in the same pool, so this script is idempotent: re-run
it after generating more data and only the held-out subset grows.

Schema (one record per line):

    {
      "id": "<unique test record id>",
      "test_set": "tutor_scenario"|"redirect_probe"|"persistent_probe"|
                  "persistent_fp_probe"|"persistent_offposition_probe"|
                  "locale_leakage",
      "cefr_level": "A2",
      "locale": "china",
      "system_prompt": "<full tutor system prompt>",
      "context_messages": [{"role":"user","content":"..."}, ...],
      "expected": {                       # optional ground-truth labels
        "axis": "topic",                  # redirect_probe only
        "sentinel_turn": 7,               # persistent_probe / fp / offpos
        "should_fire": true,              # fp/offpos: True for positive,
                                          # False for FP negative
        "off_position": true              # offposition probe only
      },
      "source": {                         # provenance
        "seed_id": "...",
        "source_dialogue_id": "...",
        "stream": "redirect"
      }
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
REDIRECT_STREAMS = [
    "redirect",
    "locale_redirect",
    "pedagogy_redirect",
    "language_redirect",
    "persona_redirect",
    "topic_redirect",
    "role_swap_redirect",
]
PERSISTENT_STREAMS = [
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
]
# Map stream prefix -> the redirect axis label we expect the baseline to handle.
REDIRECT_AXIS = {
    "redirect": "generic",
    "locale_redirect": "locale",
    "pedagogy_redirect": "pedagogy",
    "language_redirect": "language",
    "persona_redirect": "persona",
    "topic_redirect": "topic",
    "role_swap_redirect": "role_swap",
}

# Seed-id regex: 12 hex chars, optionally suffixed with _vN
SEED_ID_RE = re.compile(r"_?([0-9a-f]{12})(?:_v\d+)?$")


def in_eval_pool(seed_id: str, pct: int = 20) -> bool:
    """Deterministic hash assignment: bottom ``pct``% goes to eval pool."""
    h = int(hashlib.sha256(seed_id.encode()).hexdigest()[:8], 16) % 100
    return h < pct


def extract_seed_id(record_id: str) -> str | None:
    m = SEED_ID_RE.search(record_id)
    return m.group(1) if m else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cap_per_cell(records: list[dict[str, Any]], cap: int, key) -> list[dict[str, Any]]:
    """Keep up to ``cap`` records per cell (cell = key(rec))."""
    by_cell: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_cell[key(r)].append(r)
    out: list[dict[str, Any]] = []
    for cell, recs in by_cell.items():
        # Deterministic order: sort by id so the same set always lands in eval
        recs.sort(key=lambda r: r["id"])
        out.extend(recs[:cap])
    return out


# ---------------------------------------------------------------------------
# Test set builders
# ---------------------------------------------------------------------------


def build_tutor_scenario(eval_seed_ids: set[str], data_root: Path) -> list[dict[str, Any]]:
    """One record per held-out seed × level.

    Each record is a cold start: empty context_messages. The baseline is
    expected to produce the full dialogue when given (system_prompt, seed).
    The seed itself is embedded so the baseline can construct the same
    instruction the teacher saw.
    """
    out: list[dict[str, Any]] = []
    for level in LEVELS:
        seeds_path = data_root / "data" / "seeds" / f"{level}.jsonl"
        for seed in load_jsonl(seeds_path):
            sid = seed["id"]
            if sid not in eval_seed_ids:
                continue
            rec = {
                "id": f"tutor_scenario_{sid}_{level}",
                "test_set": "tutor_scenario",
                "cefr_level": level,
                "locale": seed.get("locale", "china"),
                "seed": seed,
                "context_messages": [],
                "expected": {},
                "source": {"seed_id": sid, "stream": "seed_only"},
            }
            out.append(rec)
    return out


def build_locale_leakage(eval_seed_ids: set[str], data_root: Path) -> list[dict[str, Any]]:
    """Same cold-start scenarios as tutor_scenario but tagged for the
    locale-leakage metric (mechanical scan for Western-default entities)."""
    out: list[dict[str, Any]] = []
    for level in LEVELS:
        seeds_path = data_root / "data" / "seeds" / f"{level}.jsonl"
        for seed in load_jsonl(seeds_path):
            sid = seed["id"]
            if sid not in eval_seed_ids:
                continue
            if seed.get("locale", "china") != "china":
                continue
            rec = {
                "id": f"locale_leakage_{sid}_{level}",
                "test_set": "locale_leakage",
                "cefr_level": level,
                "locale": "china",
                "seed": seed,
                "context_messages": [],
                "expected": {},
                "source": {"seed_id": sid, "stream": "seed_only"},
            }
            out.append(rec)
    return out


def build_redirect_probe(eval_seed_ids: set[str], data_root: Path,
                         cap_per_axis_level: int = 10) -> list[dict[str, Any]]:
    """For each (stream, level), pick held-out SFT dialogues. Extract the
    context up to and including the user's abuse turn; the baseline must
    produce the assistant's redirect turn next.

    For single-shot redirect streams: the abuse turn is the LAST user turn
    in the dialogue (the immediately-preceding context shapes the abuse).
    """
    out: list[dict[str, Any]] = []
    for stream in REDIRECT_STREAMS:
        axis = REDIRECT_AXIS[stream]
        for level in LEVELS:
            p = data_root / "data" / "sft_raw" / f"{stream}_{level}.jsonl"
            dialogues = load_jsonl(p)
            cell_records: list[dict[str, Any]] = []
            for d in dialogues:
                sid = extract_seed_id(d["id"])
                if not sid or sid not in eval_seed_ids:
                    continue
                msgs = d.get("messages", [])
                # Find the last user turn — that's the abuse turn we want
                # the baseline to respond to.
                last_user_idx = None
                for i, m in enumerate(msgs):
                    if m["role"] == "user":
                        last_user_idx = i
                if last_user_idx is None or last_user_idx < 1:
                    continue
                context = msgs[: last_user_idx + 1]
                rec = {
                    "id": f"redirect_probe_{stream}_{sid}_{level}",
                    "test_set": "redirect_probe",
                    "cefr_level": level,
                    "locale": d.get("metadata", {}).get("locale", "china"),
                    "system_prompt": d.get("system_prompt", ""),
                    "context_messages": context,
                    "expected": {"axis": axis, "stream": stream},
                    "source": {
                        "seed_id": sid,
                        "source_dialogue_id": d["id"],
                        "stream": stream,
                    },
                }
                cell_records.append(rec)
            cell_records.sort(key=lambda r: r["id"])
            out.extend(cell_records[:cap_per_axis_level])
    return out


_SENTINEL_PREFIXES = (
    "[session_end",        # matches "[SESSION_END: <axis>]" axis-agnostic
    "[ended_by_tutor",     # legacy
    "<session_end",        # legacy
)


def _find_sentinel_assistant_idx(msgs: list[dict[str, Any]]) -> int | None:
    """Return the first assistant turn index whose content carries a
    sentinel marker. ``None`` if no marker is found. Marker matching is
    axis-agnostic (`[SESSION_END:` prefix), so the persistent generator's
    axis-tagged sentinels are detected correctly.
    """
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        c = m.get("content", "").lower()
        if any(prefix in c for prefix in _SENTINEL_PREFIXES):
            return i
    return None


def build_persistent_probe(eval_seed_ids: set[str], data_root: Path,
                           cap_per_axis_level: int = 10) -> list[dict[str, Any]]:
    """For each persistent (stream, level), pick held-out dialogues. The
    sentinel is in turn 5/7/9/11 depending on the variant (Option 2a
    4-variant design). We give the baseline the context UP TO turn
    sentinel_turn-1 and ask it to produce the next turn(s).

    Detection of sentinel turn: look for the sentinel marker in the
    assistant turn. The marker is a magic string set by persistent_redirect.py.
    """
    out: list[dict[str, Any]] = []
    for stream in PERSISTENT_STREAMS:
        for level in LEVELS:
            p = data_root / "data" / "sft_raw" / f"{stream}_{level}.jsonl"
            dialogues = load_jsonl(p)
            cell_records: list[dict[str, Any]] = []
            for d in dialogues:
                sid = extract_seed_id(d["id"])
                if not sid or sid not in eval_seed_ids:
                    continue
                msgs = d.get("messages", [])
                # Find the assistant turn carrying the sentinel marker.
                # Markers are emitted by persistent_redirect.py as
                # `[SESSION_END: <axis>]`. We match the prefix
                # `[SESSION_END` (axis-agnostic, no trailing `]`) plus the
                # `[ENDED_BY_TUTOR]` literal for legacy records. If neither
                # is found we DROP the record rather than fall back to the
                # last assistant turn — that fallback silently tagged the
                # post-sentinel goodbye as the sentinel slot in older
                # eval-set builds.
                sentinel_turn_idx = _find_sentinel_assistant_idx(msgs)
                if sentinel_turn_idx is None or sentinel_turn_idx < 2:
                    continue
                context = msgs[:sentinel_turn_idx]
                rec = {
                    "id": f"persistent_probe_{stream}_{sid}_{level}",
                    "test_set": "persistent_probe",
                    "cefr_level": level,
                    "locale": d.get("metadata", {}).get("locale", "china"),
                    "system_prompt": d.get("system_prompt", ""),
                    "context_messages": context,
                    "expected": {
                        "axis": stream,
                        "sentinel_turn": sentinel_turn_idx,
                        "should_fire": True,
                    },
                    "source": {
                        "seed_id": sid,
                        "source_dialogue_id": d["id"],
                        "stream": stream,
                    },
                }
                cell_records.append(rec)
            cell_records.sort(key=lambda r: r["id"])
            out.extend(cell_records[:cap_per_axis_level])
    return out


# ---------------------------------------------------------------------------
# Persistent-FP-Probe (negatives)
# ---------------------------------------------------------------------------

# Trained sentinel positions in user-opens 0-indexed convention. Must match
# the values in persistent_redirect._STRUCTURE_VARIANTS.
TRAINED_SENTINEL_POSITIONS = (5, 7, 9, 11)


def build_persistent_fp_probe(eval_seed_ids: set[str], data_root: Path,
                              cap_per_pos_level: int = 5) -> list[dict[str, Any]]:
    """Negative controls: benign normal-stream dialogues truncated at each
    trained sentinel position. The baseline should NOT fire the sentinel
    because no three same-axis strikes have occurred.

    For each (level, position) cell we sample up to ``cap_per_pos_level``
    held-out normal records. We keep the first ``position`` messages as
    context; the baseline produces the turn that would land at index
    ``position`` (an odd-indexed tutor turn).

    Source: ``data/sft_raw/normal_{level}.jsonl`` (held-out subset only).
    The normal stream is plain scaffolded tutor dialogue with no strikes.
    """
    out: list[dict[str, Any]] = []
    for pos in TRAINED_SENTINEL_POSITIONS:
        for level in LEVELS:
            p = data_root / "data" / "sft_raw" / f"normal_{level}.jsonl"
            dialogues = load_jsonl(p)
            cell: list[dict[str, Any]] = []
            for d in dialogues:
                sid = extract_seed_id(d["id"])
                if not sid or sid not in eval_seed_ids:
                    continue
                msgs = d.get("messages", [])
                # We need the dialogue to be at least ``pos+1`` messages
                # long so msgs[:pos] is non-empty and the baseline has a
                # well-defined slot to produce at index ``pos``.
                if len(msgs) <= pos:
                    continue
                context = msgs[:pos]
                # Sanity: parity of last context turn must be USER (even
                # index) so the baseline's produced turn is the TUTOR at
                # ``pos`` (odd index). For pos in {5,7,9,11}, pos is odd
                # and context length is even → last context turn is at
                # index pos-1 (odd) = TUTOR. That's wrong: the slot the
                # baseline fills is then a USER turn, not a sentinel slot.
                # We want pos to be a TUTOR turn, which under user-opens
                # 0-indexed means context[:pos] ends on a TUTOR turn
                # (odd index pos-1), and the baseline produces USER pos.
                # Wait — sentinel firing turns in training are at indices
                # 5,7,9,11, which are TUTOR turns under user-opens (USER
                # at even, TUTOR at odd). So msgs[:5] is 5 messages
                # (indices 0..4 = U,T,U,T,U) and msgs[5] is the TUTOR
                # turn we want the model to produce. Correct.
                rec = {
                    "id": f"persistent_fp_probe_normal_{sid}_{level}_p{pos}",
                    "test_set": "persistent_fp_probe",
                    "cefr_level": level,
                    "locale": d.get("metadata", {}).get("locale", "china"),
                    "system_prompt": d.get("system_prompt", ""),
                    "context_messages": context,
                    "expected": {
                        "axis": "none",
                        "sentinel_turn": pos,
                        "should_fire": False,
                    },
                    "source": {
                        "seed_id": sid,
                        "source_dialogue_id": d["id"],
                        "stream": "normal",
                    },
                }
                cell.append(rec)
            cell.sort(key=lambda r: r["id"])
            out.extend(cell[:cap_per_pos_level])
    return out


# ---------------------------------------------------------------------------
# Persistent-OffPosition-Probe (positives at untrained positions)
# ---------------------------------------------------------------------------


def _normal_prefix_pairs(level: str, locale: str, data_root: Path,
                        n_pairs: int) -> list[dict[str, str]] | None:
    """Pull the first ``2*n_pairs`` messages of a held-out normal dialogue
    matching ``level`` and ``locale``. Returns a list of message dicts or
    None if no suitable source is found.

    Used as a plausible lead-in prefix to prepend to a held-out persistent
    record, shifting the sentinel to an off-grid position by exactly
    ``2*n_pairs`` turns.
    """
    p = data_root / "data" / "sft_raw" / f"normal_{level}.jsonl"
    for d in load_jsonl(p):
        if d.get("metadata", {}).get("locale", "china") != locale:
            continue
        msgs = d.get("messages", [])
        if len(msgs) < 2 * n_pairs:
            continue
        # Take the first 2*n_pairs messages. By parity convention the first
        # is a USER turn and the second is a TUTOR turn, so the prefix
        # length is always even and the parity downstream is preserved.
        prefix = msgs[: 2 * n_pairs]
        if prefix[0].get("role") != "user":
            continue
        return prefix
    return None


def build_persistent_offposition_probe(eval_seed_ids: set[str],
                                       data_root: Path,
                                       cap_per_axis_level: int = 5
                                       ) -> list[dict[str, Any]]:
    """Positives whose third strike lands OFF the trained {5,7,9,11} grid.

    Construction: take a held-out persistent dialogue (any variant), find
    its sentinel turn, and prepend ``2*n_pairs`` plausible normal-stream
    scaffolding turns to shift the sentinel from its original position to
    an off-grid position. Off-grid odd positions reachable from the
    existing variants:

      original 11 + 2 = 13  (V4 + 1 prepended pair)
      original 11 + 4 = 15  (V4 + 2 prepended pairs)
      original 9  + 4 = 13  (V3 + 2 prepended pairs; redundant with above)

    We target positions {13, 15} using V4 records (original sentinel at 11)
    and 1 or 2 prepended normal pairs. The third strike is still the third
    same-axis strike in the dialogue — the principle is unchanged, only
    the absolute position varies.

    Skip records whose original sentinel is not at position 11 (we want
    a deterministic shift to an off-grid target, easiest from the longest
    variant). With ~25% of persistent records at V4, this yields ~1/4 the
    base pool; cap_per_axis_level controls per-cell quantity.
    """
    # Each entry: (prepended_pairs, target_off_position).
    OFF_GRID_PLAN = [(1, 13), (2, 15)]
    out: list[dict[str, Any]] = []
    for stream in PERSISTENT_STREAMS:
        for level in LEVELS:
            p = data_root / "data" / "sft_raw" / f"{stream}_{level}.jsonl"
            dialogues = load_jsonl(p)
            cell: list[dict[str, Any]] = []
            for d in dialogues:
                sid = extract_seed_id(d["id"])
                if not sid or sid not in eval_seed_ids:
                    continue
                msgs = d.get("messages", [])
                # Find the assistant turn carrying the sentinel marker
                # (axis-agnostic prefix match — see _find_sentinel_assistant_idx).
                sentinel_turn_idx = _find_sentinel_assistant_idx(msgs)
                # We only build off-position positives from V4 (sentinel
                # originally at 11). Other variants would land us back on
                # the trained grid after a shift.
                if sentinel_turn_idx != 11:
                    continue
                locale = d.get("metadata", {}).get("locale", "china")
                for n_pairs, target_pos in OFF_GRID_PLAN:
                    prefix = _normal_prefix_pairs(level, locale, data_root,
                                                  n_pairs)
                    if prefix is None:
                        continue
                    new_msgs = prefix + msgs
                    new_sentinel_idx = sentinel_turn_idx + 2 * n_pairs
                    if new_sentinel_idx != target_pos:
                        # Defensive: target_pos must match the arithmetic
                        continue
                    if new_msgs[new_sentinel_idx].get("role") != "assistant":
                        continue
                    context = new_msgs[:new_sentinel_idx]
                    rec = {
                        "id": (f"persistent_offposition_probe_{stream}_"
                               f"{sid}_{level}_p{target_pos}"),
                        "test_set": "persistent_offposition_probe",
                        "cefr_level": level,
                        "locale": locale,
                        "system_prompt": d.get("system_prompt", ""),
                        "context_messages": context,
                        "expected": {
                            "axis": stream,
                            "sentinel_turn": target_pos,
                            "should_fire": True,
                            "off_position": True,
                        },
                        "source": {
                            "seed_id": sid,
                            "source_dialogue_id": d["id"],
                            "stream": stream,
                            "prepended_pairs": n_pairs,
                            "original_sentinel_turn": sentinel_turn_idx,
                        },
                    }
                    cell.append(rec)
            cell.sort(key=lambda r: r["id"])
            out.extend(cell[:cap_per_axis_level])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build held-out eval sets for paper evaluation."
    )
    parser.add_argument("--data-root", default=str(ROOT),
                        help="Project root containing data/ and where eval_sets/ "
                             "will be written.")
    parser.add_argument("--eval-pct", type=int, default=20,
                        help="Held-out percentage (default 20).")
    parser.add_argument("--cap-per-cell", type=int, default=10,
                        help="Max records per (stream, level) cell for the "
                             "probe sets.")
    parser.add_argument("--output-dir", default="eval_sets")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = data_root / args.output_dir

    # 1) Compute the held-out seed set across all levels.
    all_seed_ids: set[str] = set()
    seeds_by_level: dict[str, list[str]] = {}
    for level in LEVELS:
        p = data_root / "data" / "seeds" / f"{level}.jsonl"
        ids = [s["id"] for s in load_jsonl(p)]
        seeds_by_level[level] = ids
        all_seed_ids.update(ids)
    eval_seed_ids = {sid for sid in all_seed_ids if in_eval_pool(sid, args.eval_pct)}

    print(f"Total seeds: {len(all_seed_ids)}; held-out ({args.eval_pct}%): "
          f"{len(eval_seed_ids)}")

    # 2) Build each test set.
    tutor = build_tutor_scenario(eval_seed_ids, data_root)
    leakage = build_locale_leakage(eval_seed_ids, data_root)
    redirect = build_redirect_probe(eval_seed_ids, data_root, args.cap_per_cell)
    persistent = build_persistent_probe(eval_seed_ids, data_root, args.cap_per_cell)
    persistent_fp = build_persistent_fp_probe(
        eval_seed_ids, data_root, args.cap_per_cell
    )
    persistent_offpos = build_persistent_offposition_probe(
        eval_seed_ids, data_root, args.cap_per_cell
    )

    write_jsonl(out_dir / "tutor_scenario.jsonl", tutor)
    write_jsonl(out_dir / "locale_leakage.jsonl", leakage)
    write_jsonl(out_dir / "redirect_probe.jsonl", redirect)
    write_jsonl(out_dir / "persistent_probe.jsonl", persistent)
    write_jsonl(out_dir / "persistent_fp_probe.jsonl", persistent_fp)
    write_jsonl(out_dir / "persistent_offposition_probe.jsonl", persistent_offpos)

    # 3) Write the split manifest for reproducibility.
    manifest = {
        "eval_pct": args.eval_pct,
        "cap_per_cell": args.cap_per_cell,
        "total_seeds": len(all_seed_ids),
        "held_out_seeds": sorted(eval_seed_ids),
        "seeds_per_level": {lv: len(ids) for lv, ids in seeds_by_level.items()},
        "test_set_sizes": {
            "tutor_scenario": len(tutor),
            "locale_leakage": len(leakage),
            "redirect_probe": len(redirect),
            "persistent_probe": len(persistent),
            "persistent_fp_probe": len(persistent_fp),
            "persistent_offposition_probe": len(persistent_offpos),
        },
        "redirect_probe_by_axis_level": _count_axis_level(redirect, REDIRECT_STREAMS),
        "persistent_probe_by_axis_level": _count_axis_level(persistent, PERSISTENT_STREAMS),
        "persistent_fp_probe_by_position": _count_fp_by_position(persistent_fp),
        "persistent_offposition_probe_by_position": _count_fp_by_position(
            persistent_offpos
        ),
    }
    with (out_dir / "_split_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    # 4) Summary print.
    print(f"\nWrote 6 test sets to {out_dir}/:")
    print(f"  tutor_scenario.jsonl              : {len(tutor)} records")
    print(f"  locale_leakage.jsonl              : {len(leakage)} records")
    print(f"  redirect_probe.jsonl              : {len(redirect)} records")
    print(f"  persistent_probe.jsonl            : {len(persistent)} records")
    print(f"  persistent_fp_probe.jsonl         : {len(persistent_fp)} records")
    print(f"  persistent_offposition_probe.jsonl: {len(persistent_offpos)} records")
    print(f"\nManifest: {out_dir}/_split_manifest.json")
    return 0


def _count_fp_by_position(records: list[dict[str, Any]]) -> dict[int, int]:
    c: dict[int, int] = {}
    for r in records:
        pos = r.get("expected", {}).get("sentinel_turn")
        if isinstance(pos, int):
            c[pos] = c.get(pos, 0) + 1
    return dict(sorted(c.items()))


def _count_axis_level(records: list[dict[str, Any]],
                      stream_order: list[str]) -> dict[str, dict[str, int]]:
    c: dict[str, dict[str, int]] = {s: {lv: 0 for lv in LEVELS} for s in stream_order}
    for r in records:
        stream = r["source"]["stream"]
        lv = r["cefr_level"]
        if stream in c:
            c[stream][lv] += 1
    return c


if __name__ == "__main__":
    sys.exit(main())
