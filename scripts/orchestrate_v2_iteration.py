"""Paper v2 iteration orchestrator.

Conservative scope for an unattended overnight run:
  Phase 1  rebuild eval_sets (Stage 0) — safe, fast
  Phase 2  setup_paper_ablation_data.py --conditions a1,a3,a5 (Stage 2)
  Phase 3  SFT retrain A1, A3, A5 sequentially (Stage 3)

Stages 1 (data regen with new target counts), 4 (eval gen), 5
(judging), and 6 (LoRA r=32 ablation) are deferred — they require
either tooling adaptation (Stage 1 target counts) or compute beyond
the overnight window (Stages 4-6 depend on Stage 3 finishing). The
user reviews progress when they return and decides next steps.

Fail-forward semantics: if a phase fails (e.g., a single condition's
SFT crashes), log it and proceed to the next condition. The
orchestrator never blocks the whole pipeline on one failure.

Trigger: polls logs/orchestrate_pairwise_followup.log for the marker
'PAIRWISE-PREFERENCE FOLLOWUP COMPLETE' written by the upstream
pairwise orchestrator. Sleeps 30s between polls.
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"c:\project\conversationFactory\qwen-en-tutor")
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
ORCH_LOG = LOGS / "orchestrate_v2_iteration.log"
UPSTREAM_LOG = LOGS / "orchestrate_pairwise_followup.log"

# Training configs per condition (paper-named yaml files)
CONDITIONS = [
    ("a1", "config/paper/training_a1_full.yaml"),
    ("a3", "config/paper/training_a3_no_specialized.yaml"),
    ("a5", "config/paper/training_a5_fixed_turn_7.yaml"),
]


def log(msg: str) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {msg}"
    print(line, flush=True)
    with ORCH_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_for_marker(path: Path, marker: str, poll_s: int = 30) -> None:
    log(f"watching {path.name} for '{marker}'")
    while True:
        if path.exists():
            try:
                if marker in path.read_text(encoding="utf-8", errors="ignore"):
                    log("marker detected")
                    return
            except OSError:
                pass
        time.sleep(poll_s)


def kill_llama_server() -> None:
    """SFT training and llama-server can't coexist on 12 GB. Ensure
    server is down before training launches."""
    try:
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"],
                       check=False, capture_output=True, text=True)
    except FileNotFoundError:
        pass
    time.sleep(4)


def run(cmd: list[str], log_path: Path, phase: str) -> int:
    log(f"phase: {phase}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    with log_path.open("a", encoding="utf-8") as fout:
        fout.write(f"\n=== {phase} ({_dt.datetime.now():%Y-%m-%d %H:%M:%S}) ===\n")
        fout.flush()
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                stdout=fout, stderr=subprocess.STDOUT).returncode
        except FileNotFoundError as e:
            log(f"phase {phase} FAILED to launch: {e}")
            return -1
    log(f"phase {phase} rc={rc}")
    return rc


# --------------------------------------------------------------------------
# PHASES
# --------------------------------------------------------------------------

def phase1_rebuild_eval_sets() -> None:
    log("=== PHASE 1: rebuild eval_sets (Stage 0) ===")
    run([sys.executable, "scripts/build_eval_sets.py"],
        LOGS / "v2_build_eval_sets.log",
        "build_eval_sets")
    log("phase 1 complete")


def phase2_setup_ablation_data() -> None:
    log("=== PHASE 2: setup ablation data for A1/A3/A5 (Stage 2) ===")
    run([sys.executable, "scripts/setup_paper_ablation_data.py",
         "--conditions", "a1,a3,a5"],
        LOGS / "v2_setup_ablation_data.log",
        "setup_ablation_data a1,a3,a5")
    log("phase 2 complete")


def phase3_sft_retrain() -> None:
    """SFT retrain each condition sequentially. Each ~10-13h on 12GB
    GPU. fail-forward across conditions."""
    log("=== PHASE 3: SFT retrain A1 -> A3 -> A5 (Stage 3) ===")
    kill_llama_server()  # ensure no server competes for VRAM
    for cond_name, cfg_path in CONDITIONS:
        if not (ROOT / cfg_path).exists():
            log(f"  skip {cond_name}: config not found ({cfg_path})")
            continue
        log(f"  starting SFT retrain for {cond_name}")
        rc = run(
            [sys.executable, "scripts/run_training.py",
             "--training-config", cfg_path,
             "--stages", "train_sft"],
            LOGS / f"v2_sft_{cond_name}.log",
            f"sft_retrain {cond_name}",
        )
        if rc != 0:
            log(f"  {cond_name} SFT failed (rc={rc}); continuing to next condition")
        else:
            log(f"  {cond_name} SFT complete")
    log("phase 3 complete")


def main() -> int:
    log(f"v2 orchestrator pid={os.getpid()} starting")
    wait_for_marker(UPSTREAM_LOG, "PAIRWISE-PREFERENCE FOLLOWUP COMPLETE")

    try:
        phase1_rebuild_eval_sets()
    except Exception as e:
        log(f"PHASE 1 FAILED: {type(e).__name__}: {e}")
        # continue — Stage 2 doesn't strictly need rebuilt eval_sets
    try:
        phase2_setup_ablation_data()
    except Exception as e:
        log(f"PHASE 2 FAILED: {type(e).__name__}: {e}")
        # continue — Stage 3 uses per-condition dirs that should already exist
    try:
        phase3_sft_retrain()
    except Exception as e:
        log(f"PHASE 3 FAILED: {type(e).__name__}: {e}")
        return 3

    log("=== V2 ITERATION ORCHESTRATOR COMPLETE ===")
    log("Stages remaining for user-driven follow-up: 1 (data regen with new "
        "target counts), 4 (eval generation), 5 (judging + aggregate), "
        "6 optional (A1 LoRA r=32 ablation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
