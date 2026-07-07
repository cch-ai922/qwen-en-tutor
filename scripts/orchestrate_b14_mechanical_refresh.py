"""B1-B4 mechanical re-eval after Option A augment.

Background: in the prior orchestration we only regenerated
`tutor_scenario` for B1-B4 with the augmented system prompt. The
probe + locale_leakage test sets still hold pre-augment responses,
so mechanical metrics (persistent F1, locale leakage, etc.) for
B1-B4 are inconsistent with the augmented experimental design.

This orchestrator re-generates B1-B4 paper_eval for the 5 non-
tutor_scenario test sets with the augmented prompt (via
`augment_system_for_offshelf` -- already gated to B1-B4 only),
then re-scores mechanical metrics, then re-aggregates so the
existing aggregator picks up the refreshed mechanical fields.

A1-A5 are untouched -- they never get the augment, and their
responses haven't changed.
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
ORCH_LOG = LOGS / "orchestrate_b14_mechanical.log"
LLAMA_CPP = ROOT / "vendor" / "llama_cpp"
EVAL_OUT = ROOT / "outputs" / "paper" / "eval"

B14 = ["qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct",
       "qwen3_5_4b_instruct", "qwen3_5_9b_teacher"]
ALL_9 = ["paper_a1", "paper_a2", "paper_a3_sft", "paper_a4_sft", "paper_a5_sft",
         "qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct", "qwen3_5_4b_instruct",
         "qwen3_5_9b_teacher"]
# 5 test sets that need fresh B1-B4 responses (tutor_scenario already done).
TEST_SETS_TO_REGEN = [
    "redirect_probe",
    "persistent_probe",
    "persistent_fp_probe",
    "persistent_offposition_probe",
    "locale_leakage",
]

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


def kill_llama_server() -> None:
    try:
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"],
                       check=False, capture_output=True, text=True)
    except FileNotFoundError:
        pass
    time.sleep(4)


def start_llama_server(gguf_name: str, ctx: int, model_id_match: str,
                       log_path: Path) -> subprocess.Popen:
    log(f"starting llama-server with {gguf_name}")
    fout = log_path.open("a", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        [str(LLAMA_CPP / "llama-server.exe"),
         "-m", f"../models/GGUF/{gguf_name}",
         "-ngl", "80",
         "-c", str(ctx),
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
                if model_id_match.lower() in mid.lower():
                    log(f"llama-server ready: {mid}")
                    return proc
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"server did not become ready in 5min ({gguf_name})")


def run(cmd: list[str], log_path: Path, phase: str) -> int:
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


def backup_stale_b14_files() -> None:
    """Move pre-augment B1-B4 paper_eval files to *.before_augment_mech.bak so
    paper_eval re-runs from scratch instead of resuming from stale records."""
    log("backing up pre-augment B1-B4 paper_eval files for 5 test sets")
    n = 0
    for b in B14:
        for ts in TEST_SETS_TO_REGEN:
            fp = EVAL_OUT / b / f"{ts}.jsonl"
            if fp.exists():
                bak = fp.with_suffix(".jsonl.before_augment_mech.bak")
                fp.replace(bak)
                log(f"  -> {b}/{ts}.jsonl")
                n += 1
    log(f"backed up {n} files")


def regen_one_baseline(baseline: str, test_sets: list[str], log_path: Path) -> None:
    """Run paper_eval for one baseline across a list of test sets.

    Using --test-set all also works (paper_eval skips already-done records via
    resume), but a per-set loop is more transparent in the log when triaging
    failures."""
    for ts in test_sets:
        run([sys.executable, "scripts/run_paper_eval.py",
             "--baseline", baseline, "--test-set", ts],
            log_path, f"regen {baseline} {ts}")


def phase1_regen_local() -> None:
    """B1, B2, B3 - local HF, no server needed. Sequential on the 12GB GPU."""
    log("=== PHASE 1: regen B1-B3 mechanical test sets (local HF) ===")
    kill_llama_server()
    for b in ["qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct", "qwen3_5_4b_instruct"]:
        regen_one_baseline(b, TEST_SETS_TO_REGEN,
                           LOGS / f"regen_mech_{b}.log")
    log("phase 1 complete")


def phase2_regen_b4() -> None:
    """B4 needs teacher GGUF on llama-server."""
    log("=== PHASE 2: regen B4 mechanical test sets (teacher GGUF) ===")
    kill_llama_server()
    start_llama_server(TEACHER_SERVER["gguf"], TEACHER_SERVER["ctx"],
                       TEACHER_SERVER["model_id_match"],
                       LOGS / "teacher_server.log")
    regen_one_baseline("qwen3_5_9b_teacher", TEST_SETS_TO_REGEN,
                       LOGS / "regen_mech_qwen3_5_9b_teacher.log")
    kill_llama_server()
    log("phase 2 complete")


def phase3_score_mechanical() -> None:
    """Run mechanical scoring for B1-B4 on the freshly-regen'd test sets.

    For A1-A5 we re-run mechanical too (cheap, just to keep aggregator inputs
    consistent in the new round). Their responses are unchanged so numbers
    will match prior values."""
    log("=== PHASE 3: mechanical scoring (all 9 baselines) ===")
    for b in ALL_9:
        run([sys.executable, "scripts/run_paper_score.py",
             "--baseline", b, "--metrics", "mechanical"],
            LOGS / f"score_mech_{b}.log",
            f"mechanical {b}")
    log("phase 3 complete")


def phase4_aggregate() -> None:
    """Re-aggregate so the per-baseline JSONs pick up the refreshed
    mechanical fields (persistent.*, locale_leakage_rate)."""
    log("=== PHASE 4: re-aggregate across 3 judges x 9 baselines ===")
    baselines = ",".join(ALL_9)
    judges = "prometheus_7b_judge,llama31_8b_judge,gemma2_9b_judge"
    run([sys.executable, "scripts/run_paper_score.py",
         "--aggregate",
         "--baselines", baselines,
         "--judges", judges],
        LOGS / "aggregate_post_mech.log",
        "aggregate")
    log("phase 4 complete")


def main() -> int:
    log(f"orchestrator pid={os.getpid()} starting")
    backup_stale_b14_files()
    try:
        phase1_regen_local()
    except Exception as e:
        log(f"PHASE 1 FAILED: {type(e).__name__}: {e}")
        return 1
    try:
        phase2_regen_b4()
    except Exception as e:
        log(f"PHASE 2 FAILED: {type(e).__name__}: {e}")
        # continue -- B1-B3 numbers still valuable
    try:
        phase3_score_mechanical()
    except Exception as e:
        log(f"PHASE 3 FAILED: {type(e).__name__}: {e}")
        return 3
    try:
        phase4_aggregate()
    except Exception as e:
        log(f"PHASE 4 FAILED: {type(e).__name__}: {e}")
        return 4
    log("=== MECHANICAL REFRESH COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
