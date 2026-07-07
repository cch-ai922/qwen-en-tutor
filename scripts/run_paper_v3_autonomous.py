"""run_paper_v3_autonomous.py — unattended, fail-forward runner for all of
paper_v3's remaining phases, with GPU cooldown between heavy stages.

Runs WITHOUT supervision: each phase is an independent subprocess; a failure is
logged and its dependents are skipped, but the chain continues (fail-forward).
Completed phases are checkpointed (.done files) so a re-run RESUMES rather than
repeats. A GPU cooldown gate waits for temp + free memory to recover between
training/eval stages so the card isn't hammered.

PHASES (Phase 0/1 already done -> skipped as their outputs exist):
  phase3  — off-tutor/family generalization (Llama support-chat trim vs untrim)
  phase2  — count-marker H3: build data, train A8 + A8-trim, eval, score
  phase4  — mechanism figure: P(sentinel) vs depth, A1-trim vs A1-untrim
  build   — rebuild the full paper (md/pdf/docx)

Each phase declares needs_gpu; a cooldown runs AFTER each GPU phase.

Usage (fire-and-forget):
  python scripts/run_paper_v3_autonomous.py            # run all pending phases
  python scripts/run_paper_v3_autonomous.py --only phase2,phase4
  python scripts/run_paper_v3_autonomous.py --force    # ignore .done checkpoints
  python scripts/run_paper_v3_autonomous.py --cooldown-temp 58 --cooldown-free-mb 2000

Cooldown gate: after a GPU phase, poll nvidia-smi until
  temperature <= --cooldown-temp AND free_memory >= --cooldown-free-mb
(or --cooldown-max-wait seconds elapse), so models are unloaded and the card
has cooled before the next heavy stage.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "outputs" / "paper_v3" / ".autorun"
LOG = ROOT / "outputs" / "paper_v3" / "autorun.log"
PY = sys.executable
LLAMA = "./vendor/models/Llama-3.2-1B-Base"


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def gpu_stats() -> tuple[int, int, int]:
    """(free_mb, util_pct, temp_c). Returns (99999,0,0) if nvidia-smi absent."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        free, util, temp = (int(x.strip()) for x in out.split(","))
        return free, util, temp
    except Exception:
        return 99999, 0, 0


def cooldown(temp_max: int, free_min_mb: int, max_wait: int, poll: int = 20) -> None:
    log(f"cooldown: waiting for temp<={temp_max}C and free>={free_min_mb}MB "
        f"(max {max_wait}s)")
    waited = 0
    while waited < max_wait:
        free, util, temp = gpu_stats()
        if temp <= temp_max and free >= free_min_mb:
            log(f"cooldown OK (free={free}MB util={util}% temp={temp}C after {waited}s)")
            return
        time.sleep(poll)
        waited += poll
    free, util, temp = gpu_stats()
    log(f"cooldown TIMEOUT after {max_wait}s (free={free}MB temp={temp}C) — proceeding")


def run_cmd(cmd: list[str], phase: str, extra_env: dict | None = None) -> bool:
    """Run one phase as a subprocess. Returns True on success. Never raises —
    fail-forward: log and return False."""
    import os
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"   # Windows cp1252 guard (TRL chat templates)
    if extra_env:
        env.update(extra_env)
    log(f"START {phase}: {' '.join(cmd)}")
    phase_log = ROOT / "outputs" / "paper_v3" / f"autorun_{phase}.log"
    try:
        with phase_log.open("w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        if rc == 0:
            log(f"OK {phase} (rc=0)  [full log: {phase_log.name}]")
            return True
        log(f"FAIL {phase} (rc={rc})  [see {phase_log.name}] — fail-forward, continuing")
        return False
    except Exception as e:  # noqa: BLE001
        log(f"ERROR {phase}: {e!r} — fail-forward, continuing")
        return False


# ---- phase implementations (each returns True/False, never raises) ----

def do_phase3(a) -> bool:
    # build (idempotent) then train+score on Llama. If a run is already in flight
    # its outputs land here; we only (re)run if the result json is missing.
    res = ROOT / "outputs/paper_v3/phase3/phase3_result.json"
    if res.exists() and not a.force:
        log("phase3 result exists — skipping")
        return True
    ok = run_cmd([PY, "scripts/phase3_synthetic_trim.py", "build",
                  "--out-dir", "outputs/paper_v3/phase3"], "phase3_build")
    ok &= run_cmd([PY, "scripts/phase3_synthetic_trim.py", "run",
                   "--out-dir", "outputs/paper_v3/phase3",
                   "--base", LLAMA, "--epochs", "3"], "phase3_run")
    return ok


def do_phase2(a) -> bool:
    # count-marker H3: build count data (untrimmed + trimmed), assemble both,
    # train A8 + A8-trim, eval both on the premature + positive probes.
    # --- CPU data prep (dependency for everything below) ---
    prep = True
    prep &= run_cmd([PY, "scripts/convert_typed_to_count.py",
                     "--src-dir", "data/sft_filtered",
                     "--dst-dir", "data/sft_filtered_a8_count_persistent"],
                    "phase2_convert")
    prep &= run_cmd([PY, "scripts/assemble_a8_count.py"], "phase2_assemble_untrim")
    prep &= run_cmd([PY, "scripts/trim_count_persistent.py",
                     "--src", "data/sft_filtered_a8_count_persistent",
                     "--dst", "data/sft_filtered_a8_count_trim_persistent"],
                    "phase2_trim_count")
    prep &= run_cmd([PY, "scripts/assemble_a8_count.py", "--trim"],
                    "phase2_assemble_trim")
    if not prep:
        log("phase2: CPU data prep failed — skipping A8 training (fail-forward)")
        return False

    ok = True
    # --- GPU: train A8 (untrimmed) then A8-trim, cooldown between ---
    cond_env: dict[str, str] = {}
    cond_env["QWEN_TUTOR_SENTINEL_FORMAT"] = "count"

    for cfg, name, adapter_flag in (
        ("training_a8_count_sentinel.yaml", "phase2_train_a8", "a8_count"),
        ("training_a8_count_sentinel_trim.yaml", "phase2_train_a8_trim", "a8_count_trim"),
    ):
        saved = ROOT / f"outputs/paper_v2/{adapter_flag}/sft/adapter_model.safetensors"
        if saved.exists() and not a.force:
            log(f"{name}: adapter already trained — skipping")
            continue
        t_ok = run_cmd([PY, "scripts/run_training.py",
                        "--training-config", f"config/paper_v2/{cfg}",
                        "--stages", "train_sft"], name, extra_env=cond_env or None,)
        ok &= t_ok
        if t_ok:
            cooldown(a.cooldown_temp, a.cooldown_free_mb, a.cooldown_max_wait)

    # --- GPU: eval both A8 variants on premature + positive probes ---
    for baseline in ("paper_a8", "paper_a8_trim"):
        e_ok = run_cmd([PY, "scripts/run_paper_eval.py", "--baseline", baseline,
                        "--test-set", "persistent_premature_probe",
                        "--output-dir", "outputs/paper_v3/eval_a8"],
                       f"phase2_eval_{baseline}")
        ok &= e_ok
        run_cmd([PY, "scripts/run_paper_eval.py", "--baseline", baseline,
                 "--test-set", "persistent_probe",
                 "--output-dir", "outputs/paper_v3/eval_a8"],
                f"phase2_eval_{baseline}_pos")
        if e_ok:
            cooldown(a.cooldown_temp, a.cooldown_free_mb, a.cooldown_max_wait)
    return ok


def do_phase4(a) -> bool:
    ok = run_cmd([PY, "scripts/score_sentinel_logprob_vs_depth.py",
                  "--base", "./vendor/models/Qwen_3.5_0.8B-Base",
                  "--trim-adapter", "outputs/paper_v2/a1_1ep_trim/sft",
                  "--untrim-adapter", "outputs/paper_v2/a1_1ep/sft",
                  "--probe", "eval_sets/persistent_premature_probe.jsonl",
                  "--out", "outputs/paper_v3/score/sentinel_logprob_vs_depth.json"],
                 "phase4_logprob")
    cooldown(a.cooldown_temp, a.cooldown_free_mb, a.cooldown_max_wait)
    return ok


def do_build(a) -> bool:
    return run_cmd([PY, "scripts/build_paper_v3.py"], "build")


PHASES = [
    ("phase3", do_phase3, True),
    ("phase2", do_phase2, True),
    ("phase4", do_phase4, True),
    ("build",  do_build,  False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of phase names to run")
    ap.add_argument("--force", action="store_true", help="ignore .done checkpoints")
    ap.add_argument("--cooldown-temp", type=int, default=60)
    ap.add_argument("--cooldown-free-mb", type=int, default=8000)
    ap.add_argument("--cooldown-max-wait", type=int, default=1200)
    a = ap.parse_args()

    CKPT.mkdir(parents=True, exist_ok=True)
    only = set(x.strip() for x in a.only.split(",") if x.strip())
    log(f"=== paper_v3 autonomous run START (only={only or 'ALL'} force={a.force}) ===")

    summary = {}
    for name, fn, _gpu in PHASES:
        if only and name not in only:
            continue
        done = CKPT / f"{name}.done"
        if done.exists() and not a.force:
            log(f"SKIP {name} (checkpoint {done.name} exists)")
            summary[name] = "skipped(done)"
            continue
        ok = fn(a)
        summary[name] = "OK" if ok else "FAIL"
        if ok:
            done.write_text(datetime.now().isoformat(), encoding="utf-8")

    log("=== paper_v3 autonomous run COMPLETE ===")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
