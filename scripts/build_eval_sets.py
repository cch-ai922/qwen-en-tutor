"""build_eval_sets.py — Freeze held-out test sets for paper evaluation.

Produces four JSONL files under ``eval_sets/`` plus a manifest:

  tutor_scenario.jsonl    — cold-start dialogues (each baseline generates 12 turns)
  redirect_probe.jsonl    — partial dialogue ending in user abuse turn;
                            baseline produces the next assistant turn
  persistent_probe.jsonl  — partial dialogue near the sentinel-firing point;
                            baseline must produce the sentinel-firing turn
  locale_leakage.jsonl    — cold-start with locale=china;
                            metric is Western-default leakage rate

Train/eval split is hash-deterministic on seed_id (20% held-out). The same
seed_ids are always in the same pool, so this script is idempotent: re-run
it after generating more data and only the held-out subset grows.

Schema (one record per line):

    {
      "id": "<unique test record id>",
      "test_set": "tutor_scenario"|"redirect_probe"|"persistent_probe"|"locale_leakage",
      "cefr_level": "A2",
      "locale": "china",
      "system_prompt": "<full tutor system prompt>",
      "context_messages": [{"role":"user","content":"..."}, ...],
      "expected": {                       # optional ground-truth labels
        "axis": "topic",                  # redirect_probe only
        "sentinel_turn": 7                # persistent_probe only
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


def build_persistent_probe(eval_seed_ids: set[str], data_root: Path,
                           cap_per_axis_level: int = 10) -> list[dict[str, Any]]:
    """For each persistent (stream, level), pick held-out dialogues. The
    sentinel is in turn 5/7/9/11 depending on the variant (Option 2a
    4-variant design). We give the baseline the context UP TO turn
    sentinel_turn-1 and ask it to produce the next turn(s).

    Detection of sentinel turn: look for the sentinel marker in the
    assistant turn. The marker is a magic string set by persistent_redirect.py.
    """
    SENTINEL_MARKERS = [
        "[SESSION_END]",
        "[ENDED_BY_TUTOR]",
        "<SESSION_END>",
        "ending this session",
        "I have to end",
    ]
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
                sentinel_turn_idx = None
                for i, m in enumerate(msgs):
                    if m["role"] == "assistant":
                        content = m.get("content", "")
                        if any(mk.lower() in content.lower() for mk in SENTINEL_MARKERS):
                            sentinel_turn_idx = i
                            break
                if sentinel_turn_idx is None:
                    # Fallback: assume last assistant turn is the sentinel
                    for i in reversed(range(len(msgs))):
                        if msgs[i]["role"] == "assistant":
                            sentinel_turn_idx = i
                            break
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

    write_jsonl(out_dir / "tutor_scenario.jsonl", tutor)
    write_jsonl(out_dir / "locale_leakage.jsonl", leakage)
    write_jsonl(out_dir / "redirect_probe.jsonl", redirect)
    write_jsonl(out_dir / "persistent_probe.jsonl", persistent)

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
        },
        "redirect_probe_by_axis_level": _count_axis_level(redirect, REDIRECT_STREAMS),
        "persistent_probe_by_axis_level": _count_axis_level(persistent, PERSISTENT_STREAMS),
    }
    with (out_dir / "_split_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    # 4) Summary print.
    print(f"\nWrote 4 test sets to {out_dir}/:")
    print(f"  tutor_scenario.jsonl   : {len(tutor)} records")
    print(f"  locale_leakage.jsonl   : {len(leakage)} records")
    print(f"  redirect_probe.jsonl   : {len(redirect)} records")
    print(f"  persistent_probe.jsonl : {len(persistent)} records")
    print(f"\nManifest: {out_dir}/_split_manifest.json")
    return 0


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
