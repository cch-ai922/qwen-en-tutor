"""Full orchestration after Option A prompt augment for B1-B4.

Sequence:
  Phase 1  regen tutor_scenario for B1-B4 with augmented system prompt
           - B1, B2, B3: local HF models (no server needed)
           - B4: needs teacher GGUF on llama-server

  Phase 2  delete stale judged tutor_scenario for B1-B4 across all judges
           and rubric variants (generic + tutor + tutor_v2)

  Phase 3  Prometheus pass: judge B1-B4 tutor_scenario with generic + tutor
           + tutor_v2

  Phase 4  Llama-3.1 pass: judge B1-B4 tutor_scenario with generic + tutor
           + tutor_v2 (skips already-done A1-A5 done in earlier chain)

  Phase 5  Gemma-2 pass: judge all 9 baselines tutor_scenario with
           generic + tutor + tutor_v2

  Phase 6  Aggregate across all 9 baselines x 3 judges x 3 rubric variants.

Server swaps happen at phase boundaries. Each phase logs to its own
file under logs/ so failures can be triaged independently.

Designed to run unattended overnight. Logs heartbeat to
logs/orchestrate_b14_refresh.log.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"c:\project\conversationFactory\qwen-en-tutor")
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
ORCH_LOG = LOGS / "orchestrate_b14_refresh.log"
LLAMA_CPP = ROOT / "vendor" / "llama_cpp"
MODELS_GGUF = ROOT / "vendor" / "models" / "GGUF"

B14 = ["qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct",
       "qwen3_5_4b_instruct", "qwen3_5_9b_teacher"]
ALL_9 = ["paper_a1", "paper_a2", "paper_a3_sft", "paper_a4_sft", "paper_a5_sft",
         "qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct", "qwen3_5_4b_instruct",
         "qwen3_5_9b_teacher"]
RUBRICS = ["generic", "tutor_v2"]            # variants we re-run
DELETE_VARIANTS_FOR_B14 = ["generic", "tutor", "tutor_v2"]  # stale post-augment

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


def delete_judged_for_b14(judge: str, variants: list[str]) -> None:
    """Remove stale judged outputs for B1-B4 on tutor_scenario for the
    given rubric variants. Records are tied to (response, rubric_variant);
    new responses + same rubric must regenerate."""
    judged_root = ROOT / "outputs" / "paper" / "score" / "judged" / judge
    for b in B14:
        for variant in variants:
            for metric in ("cefr_adherence", "naturalness"):
                if variant == "generic":
                    fp = judged_root / b / f"tutor_scenario__{metric}.jsonl"
                else:
                    fp = judged_root / b / f"tutor_scenario__{metric}__{variant}.jsonl"
                if fp.exists():
                    fp.unlink()
                    log(f"  deleted {fp.relative_to(ROOT)}")
    # Also delete redirect_axis B1-B4 (responses regen would affect, but we
    # didn't regen redirect_probe; leave it)


def delete_judged_a5_contam(judge: str) -> None:
    """paper_a5_sft tutor for Llama-3.1 was judged after my mid-flight
    rubric edit. Delete so it gets re-judged with v1 cleanly."""
    if judge != "llama31_8b_judge":
        return
    base = ROOT / "outputs" / "paper" / "score" / "judged" / judge / "paper_a5_sft"
    for fname in ("tutor_scenario__cefr_adherence__tutor.jsonl",
                  "tutor_scenario__naturalness__tutor.jsonl"):
        fp = base / fname
        if fp.exists():
            fp.unlink()
            log(f"  deleted {fp.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# PHASES
# --------------------------------------------------------------------------

def phase1_regen_b14() -> None:
    log("=== PHASE 1: regen B1-B4 tutor_scenario with augmented prompt ===")
    kill_llama_server()
    # B1, B2, B3 local HF — no server
    for baseline in ["qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct",
                     "qwen3_5_4b_instruct"]:
        log(f"regen {baseline}")
        run([sys.executable, "scripts/run_paper_eval.py",
             "--baseline", baseline, "--test-set", "tutor_scenario"],
            LOGS / f"regen_{baseline}.log",
            f"regen {baseline} tutor_scenario")
    # B4 needs teacher GGUF on llama-server
    log("regen B4 (teacher) — needs teacher GGUF on port 8080")
    teacher_log = LOGS / "teacher_server.log"
    start_llama_server(TEACHER_SERVER["gguf"], TEACHER_SERVER["ctx"],
                       TEACHER_SERVER["model_id_match"], teacher_log)
    run([sys.executable, "scripts/run_paper_eval.py",
         "--baseline", "qwen3_5_9b_teacher", "--test-set", "tutor_scenario"],
        LOGS / "regen_qwen3_5_9b_teacher.log",
        "regen qwen3_5_9b_teacher tutor_scenario")
    kill_llama_server()
    log("phase 1 complete")


def phase2_delete_stale() -> None:
    log("=== PHASE 2: delete stale judged tutor_scenario for B1-B4 ===")
    for judge in JUDGE_SERVERS:
        delete_judged_for_b14(judge, DELETE_VARIANTS_FOR_B14)
    # Special: paper_a5_sft Llama-3.1 tutor v1 contaminated by mid-flight edit
    delete_judged_a5_contam("llama31_8b_judge")
    log("phase 2 complete")


def run_judge_chain(judge: str, log_path: Path,
                    baselines: list[str], rubrics: list[str]) -> None:
    for variant in rubrics:
        for b in baselines:
            metric_filter = (["--metric-filter", "cefr_adherence,naturalness"]
                             if variant in ("tutor", "tutor_v2") else [])
            run([sys.executable, "scripts/run_paper_score.py",
                 "--baseline", b, "--metrics", "judged",
                 "--judge", judge, "--rubric", variant] + metric_filter,
                log_path, f"{judge} {b} {variant}")


def phase3_prometheus() -> None:
    log("=== PHASE 3: Prometheus B1-B4 (generic + tutor + tutor_v2) ===")
    kill_llama_server()
    start_llama_server(JUDGE_SERVERS["prometheus_7b_judge"]["gguf"],
                       JUDGE_SERVERS["prometheus_7b_judge"]["ctx"],
                       JUDGE_SERVERS["prometheus_7b_judge"]["model_id_match"],
                       LOGS / "prometheus_server.log")
    run_judge_chain("prometheus_7b_judge",
                    LOGS / "prometheus_b14_refresh.log",
                    B14, RUBRICS)
    log("phase 3 complete")


def phase4_llama31() -> None:
    log("=== PHASE 4: Llama-3.1 full chain (skips done; runs B1-B4 fresh + v2 for all) ===")
    kill_llama_server()
    start_llama_server(JUDGE_SERVERS["llama31_8b_judge"]["gguf"],
                       JUDGE_SERVERS["llama31_8b_judge"]["ctx"],
                       JUDGE_SERVERS["llama31_8b_judge"]["model_id_match"],
                       LOGS / "llama31_server_p4.log")
    log_path = LOGS / "llama31_chain.log"
    # B4 generic (B4 responses changed after augment) + tutor_v2 for all 9.
    # tutor v1 dropped per user request.
    run_judge_chain("llama31_8b_judge", log_path,
                    ["qwen3_5_9b_teacher"], ["generic"])
    run_judge_chain("llama31_8b_judge", log_path,
                    ALL_9, ["tutor_v2"])
    # Write marker so existing watchdog detects completion and triggers gemma-2.
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n=== LLAMA-3.1 CHAIN COMPLETE ===\n")
    log("phase 4 complete; marker written for gemma-2 watchdog")


def main() -> int:
    log(f"orchestrator pid={os.getpid()} starting")
    try:
        phase1_regen_b14()
    except Exception as e:
        log(f"PHASE 1 FAILED: {type(e).__name__}: {e}")
        return 1
    try:
        phase2_delete_stale()
    except Exception as e:
        log(f"PHASE 2 FAILED: {type(e).__name__}: {e}")
        return 2
    try:
        phase3_prometheus()
    except Exception as e:
        log(f"PHASE 3 FAILED: {type(e).__name__}: {e}")
        # don't stop — Llama-3.1 + Gemma-2 still valuable
    try:
        phase4_llama31()
    except Exception as e:
        log(f"PHASE 4 FAILED: {type(e).__name__}: {e}")
        return 4
    log("=== ORCHESTRATOR DONE; gemma-2 watchdog will fire next ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
