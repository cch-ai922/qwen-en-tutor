"""build_untrim_a5a6a7.py — materialize UN-TRIMMED A5/A6/A7 SFT data.

No teacher regen, no truncation. Purely mechanical:

  A5-untrim persistent = the passed A5 records (fixed-turn-7, typed) with their
      TRIMMED messages replaced by the UN-trimmed messages from the raw regen
      output (data/sft_raw_a5_persistent), matched by id. (Verified: filtered
      messages are an exact prefix of the raw messages.)
  A7-untrim persistent = convert_a1_to_a7 on A1's UN-trimmed persistent
      (data/sft_filtered), generic sentinel + generic system prompt.
  A6-untrim persistent = convert_a5_to_a6 on A5-untrim persistent.

Then assemble full condition dirs by copying the existing sft_filtered_a{5,6,7}
dir (which carries the shared non-persistent streams) and OVERWRITING the
persistent_*_passed.jsonl files with the un-trimmed versions.

Does NOT touch the existing trimmed dirs, does NOT write configs, does NOT
train. Writes:
  data/sft_filtered_a{5,6,7}_untrim_persistent/   (persistent only)
  data/sft_filtered_a{5,6,7}_untrim/              (full training dir)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AXES = ["persistent_off_topic", "persistent_language_violation",
        "persistent_persona_break", "persistent_role_swap"]
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def build_a5_untrim_persistent() -> Path:
    src_filt = ROOT / "data" / "sft_filtered_a5_persistent"
    src_raw = ROOT / "data" / "sft_raw_a5_persistent"
    dst = ROOT / "data" / "sft_filtered_a5_untrim_persistent"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    n_rec = n_swap = n_missing = 0
    for axis in AXES:
        for lvl in LEVELS:
            fp = src_filt / f"{axis}_{lvl}_passed.jsonl"
            if not fp.exists():
                continue
            rp = src_raw / f"{axis}_{lvl}.jsonl"
            raw = {}
            if rp.exists():
                for line in rp.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        raw[r.get("id")] = r
            out = []
            for line in fp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ex = rec.get("example") or rec
                r = raw.get(ex.get("id"))
                n_rec += 1
                if r is not None:
                    ex["messages"] = r["messages"]          # un-trimmed
                    gen = ex.setdefault("metadata", {}).setdefault("generation", {})
                    gen["message_count"] = len(r["messages"])
                    n_swap += 1
                else:
                    n_missing += 1
                out.append(json.dumps(rec, ensure_ascii=False))
            (dst / f"{axis}_{lvl}_passed.jsonl").write_text(
                "\n".join(out) + "\n", encoding="utf-8")
    print(f"[A5] persistent records: {n_rec} | un-trimmed from raw: {n_swap} | "
          f"no-raw-match (left trimmed): {n_missing}")
    return dst


def run_convert(script: str, src: str, dst: str) -> None:
    print(f"\n[{script}] {src} -> {dst}")
    r = subprocess.run(
        [sys.executable, f"scripts/{script}", "--src-dir", src, "--dst-dir", dst],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        raise SystemExit(f"{script} failed (rc={r.returncode})")


def assemble(cond: str, untrim_persist: Path) -> Path:
    base = ROOT / f"data/sft_filtered_{cond}"
    dst = ROOT / f"data/sft_filtered_{cond}_untrim"
    if not base.exists():
        raise SystemExit(f"base condition dir missing: {base}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(base, dst)
    n = 0
    for f in sorted(untrim_persist.glob("*_passed.jsonl")):
        shutil.copy(f, dst / f.name)
        n += 1
    print(f"[assemble {cond}] copied {base.name} -> {dst.name}, overwrote {n} persistent files")
    return dst


def main() -> int:
    print("=== 1) A5 un-trimmed persistent (from raw) ===")
    a5p = build_a5_untrim_persistent()

    print("\n=== 2) A7 un-trimmed persistent (convert_a1_to_a7 on A1 un-trimmed) ===")
    run_convert("convert_a1_to_a7.py", "data/sft_filtered",
                "data/sft_filtered_a7_untrim_persistent")

    print("\n=== 3) A6 un-trimmed persistent (convert_a5_to_a6 on A5-untrim) ===")
    run_convert("convert_a5_to_a6.py", str(a5p),
                "data/sft_filtered_a6_untrim_persistent")

    print("\n=== 4) assemble full condition dirs ===")
    assemble("a5", a5p)
    assemble("a6", ROOT / "data/sft_filtered_a6_untrim_persistent")
    assemble("a7", ROOT / "data/sft_filtered_a7_untrim_persistent")

    print("\nDONE. Built data/sft_filtered_a{5,6,7}_untrim/ (NO configs, NO training).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
