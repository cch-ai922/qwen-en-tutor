"""Redirect-axis judging refresh, scheduled to run AFTER the
mechanical re-eval finishes.

Background. The redirect-axis judge prompt in run_paper_score.py was
silently classifying responses with an EMPTY user-context for every
record across every baseline. The generation-side code never wrote a
`last_user_turn` field that the score-side defensively read with a
"" fallback. We fixed this on 2026-06-24 by:

  - build_eval_sets.py: emit `expected.violation_turn_idx` and
    `expected.violation_turn_content` into redirect_probe records.
  - run_paper_eval.py: propagate `expected` into the saved generation
    record so the scorer can read the violation content.
  - run_paper_score.py: read `expected.violation_turn_content` (with
    eval_set lookup fallback) instead of empty.

This script rebuilds the eval set, regenerates paper_eval responses
for B1-B4 redirect_probe (so they carry the new field), then re-judges
redirect_axis for ALL 9 baselines with all 3 judges (Prometheus,
Llama-3.1, Gemma-2). A1-A5 responses don't need re-generation -- the
scorer reads the violation content from the eval_set lookup -- only
the JUDGE step needs to re-run for them with proper context.

Trigger. Polls logs/orchestrate_b14_mechanical.log for the marker
"=== MECHANICAL REFRESH COMPLETE ===" written by
orchestrate_b14_mechanical_refresh.py at the end of its phase 4.
Sleeps 30s between polls.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"c:\project\conversationFactory\qwen-en-tutor")
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
ORCH_LOG = LOGS / "orchestrate_redirect_axis_refresh.log"
MECH_LOG = LOGS / "orchestrate_b14_mechanical.log"
LLAMA_CPP = ROOT / "vendor" / "llama_cpp"

B14 = ["qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct",
       "qwen3_5_4b_instruct", "qwen3_5_9b_teacher"]
ALL_9 = ["paper_a1", "paper_a2", "paper_a3_sft", "paper_a4_sft", "paper_a5_sft",
         "qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct", "qwen3_5_4b_instruct",
         "qwen3_5_9b_teacher"]
EVAL_OUT = ROOT / "outputs" / "paper" / "eval"
JUDGED_ROOT = ROOT / "outputs" / "paper" / "score" / "judged"

JUDGE_SERVERS = {
    "prometheus_7b_judge": {
        "gguf": "prometheus-7b-v2.0.Q4_K_M.gguf",
        "ctx": 32768,
        "model_id_match": "prometheus",
    },
    "llama31_8b_judge": {
        "gguf": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "ctx": 32768,
        "model_id_match": "Llama-3.1",
    },
    "gemma2_9b_judge": {
        "gguf": "gemma-2-9b-it-Q5_K_M.gguf",
        "ctx": 16384,
        "model_id_match": "gemma-2",
    },
}
TEACHER_SERVER = {
    "gguf": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
    "ctx": 32768,
    "model_id_match": "Qwen3.5",
}


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
    try:
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"],
                       check=False, capture_output=True, text=True)
    except FileNotFoundError:
        pass
    time.sleep(4)


def start_llama_server(spec: dict, log_path: Path) -> None:
    log(f"starting llama-server with {spec['gguf']}")
    fout = log_path.open("a", encoding="utf-8", buffering=1)
    subprocess.Popen(
        [str(LLAMA_CPP / "llama-server.exe"),
         "-m", f"../models/GGUF/{spec['gguf']}",
         "-ngl", "80",
         "-c", str(spec["ctx"]),
         "--host", "0.0.0.0",
         "--port", "8080",
         "--log-prefix"],
        cwd=str(LLAMA_CPP),
        stdout=fout, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8080/v1/models", timeout=3
            ) as r:
                body = json.load(r)
                mid = body.get("data", [{}])[0].get("id", "")
                if spec["model_id_match"].lower() in mid.lower():
                    log(f"llama-server ready: {mid}")
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"server didn't ready in 5min ({spec['gguf']})")


def run_subprocess(cmd: list[str], log_path: Path, phase: str) -> int:
    log(f"phase: {phase}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    with log_path.open("a", encoding="utf-8") as fout:
        fout.write(f"\n=== {phase} ({_dt.datetime.now():%Y-%m-%d %H:%M:%S}) ===\n")
        fout.flush()
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                            stdout=fout, stderr=subprocess.STDOUT).returncode
    log(f"phase {phase} rc={rc}")
    return rc


# --------------------------------------------------------------------------
# PHASES
# --------------------------------------------------------------------------

def phase1_rebuild_eval_sets() -> None:
    log("=== PHASE 1: rebuild eval_sets so redirect_probe carries violation_turn_idx ===")
    run_subprocess([sys.executable, "scripts/build_eval_sets.py"],
                   LOGS / "build_eval_sets_post_fix.log",
                   "build_eval_sets")
    log("phase 1 complete")


def phase2_regen_b14_redirect_probe() -> None:
    """Regenerate B1-B4 redirect_probe with the new code path so the saved
    generation records carry expected.violation_turn_content. A1-A5
    records will use the eval_set lookup fallback in the scorer -- no
    regen needed for them."""
    log("=== PHASE 2: regen B1-B4 redirect_probe (writes violation_turn_content) ===")
    # Delete B1-B4 redirect_probe paper_eval files (they were written with
    # the old paper_eval code, lacking the new field).
    for b in B14:
        fp = EVAL_OUT / b / "redirect_probe.jsonl"
        if fp.exists():
            bak = fp.with_suffix(".jsonl.before_violation_field.bak")
            fp.replace(bak)
            log(f"  backed up {b}/redirect_probe.jsonl")

    # B1, B2, B3 local HF (no server needed)
    kill_llama_server()
    for b in ["qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct",
              "qwen3_5_4b_instruct"]:
        run_subprocess(
            [sys.executable, "scripts/run_paper_eval.py",
             "--baseline", b, "--test-set", "redirect_probe"],
            LOGS / f"regen_redirect_{b}.log",
            f"regen {b} redirect_probe",
        )
    # B4 via teacher GGUF
    start_llama_server(TEACHER_SERVER, LOGS / "teacher_server.log")
    run_subprocess(
        [sys.executable, "scripts/run_paper_eval.py",
         "--baseline", "qwen3_5_9b_teacher", "--test-set", "redirect_probe"],
        LOGS / "regen_redirect_qwen3_5_9b_teacher.log",
        "regen qwen3_5_9b_teacher redirect_probe",
    )
    kill_llama_server()
    log("phase 2 complete")


def phase3_redirect_axis_judge_chain() -> None:
    """Re-judge redirect_axis for all 9 baselines with all 3 judges.
    Deletes existing judge output files first so resume doesn't skip
    them."""
    log("=== PHASE 3: redirect_axis re-judge x 3 judges x 9 baselines ===")
    for judge in JUDGE_SERVERS:
        # Delete stale judged outputs for redirect_axis under this judge
        log(f"  deleting stale {judge} redirect_probe judged files")
        for b in ALL_9:
            fp = JUDGED_ROOT / judge / b / "redirect_probe__redirect_axis.jsonl"
            if fp.exists():
                fp.unlink()
                log(f"    rm {b}/redirect_probe__redirect_axis.jsonl")
        # Swap server to this judge
        kill_llama_server()
        start_llama_server(JUDGE_SERVERS[judge],
                           LOGS / f"{judge}_server_redirect.log")
        for b in ALL_9:
            run_subprocess(
                [sys.executable, "scripts/run_paper_score.py",
                 "--baseline", b, "--metrics", "judged",
                 "--judge", judge, "--rubric", "generic",
                 "--metric-filter", "redirect_axis"],
                LOGS / f"score_redirect_{judge}.log",
                f"{judge} {b} redirect_axis",
            )
    kill_llama_server()
    log("phase 3 complete")


def phase4_aggregate() -> None:
    log("=== PHASE 4: re-aggregate across 3 judges x 9 baselines ===")
    baselines = ",".join(ALL_9)
    judges = ",".join(JUDGE_SERVERS.keys())
    run_subprocess(
        [sys.executable, "scripts/run_paper_score.py",
         "--aggregate", "--baselines", baselines, "--judges", judges],
        LOGS / "aggregate_post_redirect.log",
        "aggregate post-redirect-fix",
    )
    log("phase 4 complete")


def main() -> int:
    log(f"redirect-axis refresh watchdog pid={os.getpid()} starting")
    wait_for_marker(MECH_LOG, "MECHANICAL REFRESH COMPLETE")
    try:
        phase1_rebuild_eval_sets()
    except Exception as e:
        log(f"PHASE 1 FAILED: {type(e).__name__}: {e}")
        return 1
    try:
        phase2_regen_b14_redirect_probe()
    except Exception as e:
        log(f"PHASE 2 FAILED: {type(e).__name__}: {e}")
        # B4 may have failed (teacher GGUF flake); B1-B3 may still be valid
    try:
        phase3_redirect_axis_judge_chain()
    except Exception as e:
        log(f"PHASE 3 FAILED: {type(e).__name__}: {e}")
        return 3
    try:
        phase4_aggregate()
    except Exception as e:
        log(f"PHASE 4 FAILED: {type(e).__name__}: {e}")
        return 4
    log("=== REDIRECT-AXIS REFRESH COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
