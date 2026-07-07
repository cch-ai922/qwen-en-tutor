"""copy_sft_raw_to_a5.py — Copy persistent records from sft_raw into sft_raw_a5_persistent.

Keeps turn-7 records as-is.
Trims turn-9 records by removing the first 2 messages (sentinel shifts 9→7).
Trims turn-11 records by removing the first 4 messages (sentinel shifts 11→7).
Excludes turn-5 records (would need to add turns, not possible).

Deduplicates against IDs already in sft_raw_a5_persistent.
Appends '_t7' suffix to trimmed record IDs to mark them as derived.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SFT_RAW = ROOT / "data" / "sft_raw"
A5_DIR = ROOT / "data" / "sft_raw_a5_persistent"

AXES = [
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
]
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

# Messages to remove from the front for each sentinel_turn value
TRIM = {7: 0, 9: 2, 11: 4}


def _trim_record(r: dict, trim: int) -> dict:
    """Return a new record with the first `trim` messages removed and metadata updated."""
    import copy
    r2 = copy.deepcopy(r)
    msgs = r2.get("messages", [])
    r2["messages"] = msgs[trim:]
    gen = r2.setdefault("metadata", {}).setdefault("generation", {})
    old_st = gen.get("sentinel_turn", 7)
    old_mc = gen.get("message_count", len(msgs))
    gen["sentinel_turn"] = old_st - trim
    gen["message_count"] = old_mc - trim
    r2["id"] = r2["id"] + "_t7"
    return r2


def main() -> None:
    # Collect existing IDs in a5_persistent (to avoid duplicates)
    existing_ids: set[str] = set()
    for axis in AXES:
        for level in LEVELS:
            f = A5_DIR / f"{axis}_{level}.jsonl"
            if f.exists():
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        existing_ids.add(json.loads(line).get("id", ""))

    print(f"Existing a5_persistent IDs: {len(existing_ids)}")

    total_added = 0
    for axis in AXES:
        axis_added = 0
        for level in LEVELS:
            src = SFT_RAW / f"{axis}_{level}.jsonl"
            dst = A5_DIR / f"{axis}_{level}.jsonl"
            if not src.exists():
                continue

            new_lines: list[str] = []
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                st = r.get("metadata", {}).get("generation", {}).get("sentinel_turn")
                if st not in TRIM:
                    continue  # sentinel_turn=5 or unknown — skip

                trim = TRIM[st]
                r2 = _trim_record(r, trim) if trim > 0 else r

                if r2["id"] in existing_ids:
                    continue

                # Sanity check: verify sentinel is at position 7 after trimming
                msgs = r2.get("messages", [])
                sentinel_in_meta = r2.get("metadata", {}).get("generation", {}).get("sentinel")
                if sentinel_in_meta and len(msgs) > 7:
                    if sentinel_in_meta not in msgs[7].get("content", ""):
                        continue  # sentinel not at expected position — skip

                existing_ids.add(r2["id"])
                new_lines.append(json.dumps(r2, ensure_ascii=False))

            if new_lines:
                with open(dst, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(new_lines) + "\n")
                axis_added += len(new_lines)
                total_added += len(new_lines)
                print(f"  {axis}_{level}: +{len(new_lines)}")

        print(f"  {axis}: +{axis_added} total")

    print(f"\nDone. Total added: {total_added}")
    print(f"New a5_persistent total: {len(existing_ids)}")


if __name__ == "__main__":
    main()
