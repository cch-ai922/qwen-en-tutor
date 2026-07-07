"""trim_persistent_post_sentinel.py — Drop messages after the sentinel.

For each persistent_*_passed.jsonl in sft_filtered_a5_persistent/ and
sft_filtered_a6_persistent/, truncate `example.messages` so that the last
message is the assistant turn containing the sentinel string. Also update
`example.metadata.generation.message_count` to reflect the new length.

Why: at deployment the model emits the sentinel and the session ends. Any
turns after the sentinel in the training data teach the model to keep
generating past it, which dilutes the stop signal. Removing them also
shortens sequences and speeds up training.

Run after data is materialized into the sft_filtered_*_persistent dirs;
hardlinks under sft_filtered_a5/ and sft_filtered_a6/ pick up the changes
automatically (same inode).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRS = [
    ROOT / "data" / "sft_filtered_a5_persistent",
    ROOT / "data" / "sft_filtered_a6_persistent",
    ROOT / "data" / "sft_filtered_a7_persistent",
]


def _trim(rec: dict) -> tuple[dict, int, int, bool]:
    ex = rec.get("example") or rec  # tolerate both layouts
    msgs = ex.get("messages", [])
    sentinel = ex.get("metadata", {}).get("generation", {}).get("sentinel")
    orig = len(msgs)
    if not sentinel or not msgs:
        return rec, orig, orig, False
    sent_idx = None
    for i, m in enumerate(msgs):
        if sentinel in m.get("content", ""):
            sent_idx = i
            break
    if sent_idx is None:
        return rec, orig, orig, False
    ex["messages"] = msgs[: sent_idx + 1]
    gen = ex.setdefault("metadata", {}).setdefault("generation", {})
    gen["message_count"] = len(ex["messages"])
    return rec, orig, len(ex["messages"]), True


def main() -> None:
    grand_total_records = 0
    grand_total_orig_msgs = 0
    grand_total_new_msgs = 0
    for d in DIRS:
        if not d.exists():
            print(f"skip (missing): {d}")
            continue
        files = sorted(d.glob("persistent_*_passed.jsonl"))
        print(f"\n=== {d.name} ({len(files)} files) ===")
        dir_records = 0
        dir_trimmed = 0
        dir_orig = 0
        dir_new = 0
        for f in files:
            lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            out_lines = []
            file_trimmed = 0
            file_orig = 0
            file_new = 0
            for line in lines:
                rec = json.loads(line)
                rec2, orig, new, did = _trim(rec)
                file_orig += orig
                file_new += new
                if did:
                    file_trimmed += 1
                out_lines.append(json.dumps(rec2, ensure_ascii=False))
            f.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            avg_o = file_orig / max(1, len(lines))
            avg_n = file_new / max(1, len(lines))
            print(f"  {f.name}: {len(lines)} recs (trimmed {file_trimmed}); avg msgs {avg_o:.2f} -> {avg_n:.2f}")
            dir_records += len(lines)
            dir_trimmed += file_trimmed
            dir_orig += file_orig
            dir_new += file_new
        if dir_records:
            print(f"  TOTAL {d.name}: {dir_records} recs (trimmed {dir_trimmed}); "
                  f"avg msgs {dir_orig/dir_records:.2f} -> {dir_new/dir_records:.2f}; "
                  f"tokens-relative -{100*(1 - dir_new/dir_orig):.1f}%")
        grand_total_records += dir_records
        grand_total_orig_msgs += dir_orig
        grand_total_new_msgs += dir_new

    if grand_total_records:
        print(f"\nGrand total: {grand_total_records} records; "
              f"messages {grand_total_orig_msgs} -> {grand_total_new_msgs} "
              f"({100*(1 - grand_total_new_msgs/grand_total_orig_msgs):.1f}% reduction)")


if __name__ == "__main__":
    main()
