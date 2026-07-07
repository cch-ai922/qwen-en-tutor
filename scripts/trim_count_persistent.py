"""trim_count_persistent.py — trim the count-annotated (A8) persistent records
to end at the [SESSION_END: axis, strike=3] marker. paper_v3 Phase 2 (A8-trim).

Mirrors trim_persistent_post_sentinel.py but targets the A8 count dir and
detects the count-form marker. Truncates each record's messages so the last
message is the assistant turn containing the strike=3 SESSION_END; the
[STRIKE: axis, N] tags on earlier turns are retained (they are before the
marker). CPU only. Idempotent.

Usage:
  python scripts/trim_count_persistent.py \
      --src data/sft_filtered_a8_count_persistent \
      --dst data/sft_filtered_a8_count_trim_persistent
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "[SESSION_END"  # matches the strike=3 form; STRIKE tags don't contain it


def trim_file(src: Path, dst: Path) -> tuple[int, int]:
    n = trimmed = 0
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ex = rec.get("example") or rec
        msgs = ex.get("messages", [])
        idx = next((i for i, m in enumerate(msgs)
                    if MARKER in (m.get("content") or "")), None)
        n += 1
        if idx is not None and idx < len(msgs) - 1:
            ex["messages"] = msgs[: idx + 1]
            gen = ex.setdefault("metadata", {}).setdefault("generation", {})
            gen["message_count"] = len(ex["messages"])
            trimmed += 1
        out.append(json.dumps(rec, ensure_ascii=False))
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return n, trimmed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/sft_filtered_a8_count_persistent")
    ap.add_argument("--dst", default="data/sft_filtered_a8_count_trim_persistent")
    a = ap.parse_args()
    src, dst = ROOT / a.src, ROOT / a.dst
    if not src.exists():
        raise SystemExit(f"source dir missing: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    tot_n = tot_t = 0
    for f in sorted(src.glob("*_passed.jsonl")):
        n, t = trim_file(f, dst / f.name)
        print(f"  {f.name}: {n} recs, trimmed {t}")
        tot_n += n; tot_t += t
    print(f"TOTAL: {tot_n} recs, trimmed {tot_t} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
