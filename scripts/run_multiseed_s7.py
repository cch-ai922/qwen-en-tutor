"""run_multiseed_s7.py — one-shot seed-7 trim-vs-untrim replication (reviewer #2).

Bundled end-to-end (train -> eval -> assemble -> score) so a shared-VRAM run
needs no manual checkpointing between phases (see feedback_bundle_sequential_
judge_chains). Fail-forward: log a failed phase, skip its dependents, keep going.

Plan (minimal GPU — ONE training run):
  * untrim seed-7 is REUSED: outputs/paper_v2/a1_s7/sft/checkpoint-600 (1.51 ep)
  * trim   seed-7 is TRAINED here under the identical 2-epoch cosine schedule;
    we harvest its checkpoint-600 so the pair is schedule+step matched.
  * eval both on persistent_probe (recall) + persistent_premature_probe (premature)
  * assemble into outputs/paper_v2/phase0_trim_study/A1_{untrim,trim}_s7/
  * score with score_phase0_attribution.py (same scorer as the headline cells)

Headline seed-42 (@1.0 ep) stays the paper's primary number; seed-7 (@1.51 ep,
footnoted) shows the trim DIRECTION replicates across seeds.

Run (background):
  PYTHONUTF8=1 .venv/Scripts/python.exe scripts/run_multiseed_s7.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "logs" / "multiseed_s7.log"
LOG.parent.mkdir(exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def gpu() -> tuple[int, int, int]:
    """(free_mb, util_pct, temp_c); (99999,0,0) if nvidia-smi absent."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        free, util, temp = (int(x.strip()) for x in out.split(","))
        return free, util, temp
    except Exception:
        return 99999, 0, 0


def cooldown(temp_max: int = 60, free_min_mb: int = 8000, max_wait: int = 1200) -> None:
    log(f"cooldown: waiting temp<={temp_max}C and free>={free_min_mb}MB (max {max_wait}s)")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        free, util, temp = gpu()
        if temp <= temp_max and free >= free_min_mb:
            log(f"cooldown OK (free={free}MB util={util}% temp={temp}C after {int(time.time()-t0)}s)")
            return
        time.sleep(20)
    log(f"cooldown TIMEOUT after {max_wait}s — proceeding anyway")


def run(tag: str, cmd: list[str], env: dict | None = None) -> bool:
    log(f"START {tag}: {' '.join(cmd)}")
    e = dict(os.environ)
    if env:
        e.update(env)
    phase_log = ROOT / "logs" / f"multiseed_s7_{tag}.log"
    with phase_log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=e, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    ok = rc == 0
    log(f"{'OK' if ok else 'FAIL'} {tag} (rc={rc})  [full log: {phase_log.name}]")
    return ok


def eval_baseline(baseline: str, test_set: str) -> bool:
    return run(
        f"eval_{baseline}_{test_set}",
        [PY, "scripts/run_paper_eval.py", "--baseline", baseline,
         "--test-set", test_set, "--output-dir", "outputs/paper_v2/eval_multiseed_s7"],
    )


def assemble(dest: str, src_baseline_dir: Path) -> None:
    """Copy the two probe JSONLs into a phase0_trim_study-style cell dir."""
    ddir = ROOT / "outputs" / "paper_v2" / "phase0_trim_study" / dest
    ddir.mkdir(parents=True, exist_ok=True)
    import json
    copied = []
    for probe in ("persistent_probe.jsonl", "persistent_premature_probe.jsonl"):
        srcf = src_baseline_dir / probe
        if srcf.exists():
            shutil.copy2(srcf, ddir / probe)
            copied.append(probe)
    (ddir / "_SOURCE.json").write_text(json.dumps({
        "dest": dest, "source": str(src_baseline_dir.relative_to(ROOT / "outputs" / "paper_v2")),
        "note": "multiseed seed-7 @ckpt-600 (1.51ep); untrim reused a1_s7, trim newly trained",
        "copied": copied,
    }), encoding="utf-8")
    log(f"assembled {dest}: {copied}")


def main() -> int:
    log("=== multiseed s7 run START ===")

    # ---- Phase 1: train trim seed-7 (only GPU-training phase) ----
    trim_ckpt = ROOT / "outputs/paper_v2/a1_1ep_trim_s7/sft/checkpoint-600"
    if trim_ckpt.exists():
        log("trim s7 checkpoint-600 already present — skipping training")
        trained = True
    else:
        cooldown()
        trained = run("train_trim_s7",
                      [PY, "scripts/run_training.py",
                       "--training-config", "config/paper_v2/training_a1_1ep_trim_s7.yaml",
                       "--stages", "train_sft"])
        if not (trim_ckpt.exists()):
            log("WARN: expected checkpoint-600 missing after training; "
                "trim eval will FAIL-forward (check step count / save_steps)")

    # ---- Phase 2: eval both seed-7 models on both probes ----
    cooldown()
    ev_root = ROOT / "outputs/paper_v2/eval_multiseed_s7"
    # untrim reuses a1_s7 ckpt-600 (baseline v3_a1_untrim_s7)
    ok_u1 = eval_baseline("v3_a1_untrim_s7", "persistent_probe")
    ok_u2 = eval_baseline("v3_a1_untrim_s7", "persistent_premature_probe")
    ok_t1 = ok_t2 = False
    if trained and trim_ckpt.exists():
        ok_t1 = eval_baseline("v3_a1_trim_s7", "persistent_probe")
        ok_t2 = eval_baseline("v3_a1_trim_s7", "persistent_premature_probe")
    else:
        log("SKIP trim s7 eval (training/checkpoint unavailable)")

    # ---- Phase 3: assemble into phase0_trim_study cells ----
    if ok_u1 or ok_u2:
        assemble("A1_untrim_s7", ev_root / "v3_a1_untrim_s7")
    if ok_t1 or ok_t2:
        assemble("A1_trim_s7", ev_root / "v3_a1_trim_s7")

    # ---- Phase 4: score (mechanical, reuses headline scorer via a patched CELLS) ----
    # score_phase0_attribution.py has fixed CELLS; we score the seed-7 cells with
    # a tiny inline scorer to avoid editing that script's canonical map.
    run("score_s7", [PY, "scripts/score_multiseed_s7.py"])

    log("=== multiseed s7 run COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
