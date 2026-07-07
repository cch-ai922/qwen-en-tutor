"""eval_multiseed_s7_now.py — eval-only driver for seed-7 ckpt-600 (training was
stopped early; both checkpoint-600 adapters are already on disk). Runs the 4
evals + assemble + score, no training. Detached/background-safe.
"""
from __future__ import annotations
import os, sys, time, shutil, json, subprocess
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "logs" / "multiseed_s7.log"
EVOUT = ROOT / "outputs" / "paper_v2" / "eval_multiseed_s7"


def log(m: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(tag, cmd):
    log(f"START {tag}: {' '.join(cmd)}")
    pl = ROOT / "logs" / f"multiseed_s7_{tag}.log"
    with pl.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ),
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"{'OK' if rc == 0 else 'FAIL'} {tag} (rc={rc}) [log: {pl.name}]")
    return rc == 0


def eval_b(baseline, ts):
    return run(f"eval_{baseline}_{ts}",
               [PY, "scripts/run_paper_eval.py", "--baseline", baseline,
                "--test-set", ts, "--output-dir", str(EVOUT)])


def assemble(dest, srcdir):
    ddir = ROOT / "outputs/paper_v2/phase0_trim_study" / dest
    ddir.mkdir(parents=True, exist_ok=True)
    copied = []
    for probe in ("persistent_probe.jsonl", "persistent_premature_probe.jsonl"):
        s = srcdir / probe
        if s.exists():
            shutil.copy2(s, ddir / probe); copied.append(probe)
    (ddir / "_SOURCE.json").write_text(json.dumps({
        "dest": dest, "note": "seed-7 ckpt-600 (1.51ep); trim training stopped early, ckpt-600 complete",
        "copied": copied}), encoding="utf-8")
    log(f"assembled {dest}: {copied}")


def main():
    log("=== multiseed s7 EVAL-ONLY run START (training stopped early) ===")
    ok = {}
    for b in ("v3_a1_untrim_s7", "v3_a1_trim_s7"):
        ok[(b, "pos")] = eval_b(b, "persistent_probe")
        ok[(b, "prem")] = eval_b(b, "persistent_premature_probe")
    if ok[("v3_a1_untrim_s7", "pos")] or ok[("v3_a1_untrim_s7", "prem")]:
        assemble("A1_untrim_s7", EVOUT / "v3_a1_untrim_s7")
    if ok[("v3_a1_trim_s7", "pos")] or ok[("v3_a1_trim_s7", "prem")]:
        assemble("A1_trim_s7", EVOUT / "v3_a1_trim_s7")
    run("score_s7", [PY, "scripts/score_multiseed_s7.py"])
    log("=== multiseed s7 EVAL-ONLY run COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
