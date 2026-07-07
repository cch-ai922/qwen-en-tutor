"""Regenerate A2-SFT and A3-SFT responses on the cleaned probe, then
re-run the pairwise pipeline.

Order:
  1. kill_llama_server (just in case)
  2. run_paper_eval.py --baseline paper_a2     --test-set redirect_probe
  3. run_paper_eval.py --baseline paper_a3_sft --test-set redirect_probe
  4. for each of (prometheus, llama31, gemma2):
       start llama-server -> score_pairwise_preference.py --judge X
  5. kill_llama_server, emit RERUN COMPLETE marker.
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
ORCH_LOG = LOGS / "orchestrate_probe_clean_rerun.log"
LLAMA_CPP = ROOT / "vendor" / "llama_cpp"

JUDGES = [
    ("prometheus_7b_judge", "prometheus-7b-v2.0.Q4_K_M.gguf",
     32768, "prometheus"),
    ("llama31_8b_judge", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     32768, "Llama-3.1"),
    ("gemma2_9b_judge", "gemma-2-9b-it-Q5_K_M.gguf",
     16384, "gemma-2"),
]

ADAPTER_BASELINES = ["paper_a2", "paper_a3_sft"]


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


def start_llama_server(gguf: str, ctx: int, model_match: str,
                       log_path: Path) -> None:
    log(f"starting llama-server with {gguf}")
    fout = log_path.open("a", encoding="utf-8", buffering=1)
    subprocess.Popen(
        [str(LLAMA_CPP / "llama-server.exe"),
         "-m", f"../models/GGUF/{gguf}",
         "-ngl", "80", "-c", str(ctx),
         "--host", "0.0.0.0", "--port", "8080",
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
                if model_match.lower() in mid.lower():
                    log(f"server ready: {mid}")
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"server didn't ready in 5min ({gguf})")


def run_paper_eval_baseline(baseline: str, log_path: Path) -> int:
    log(f"run_paper_eval --baseline {baseline} --test-set redirect_probe")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    with log_path.open("a", encoding="utf-8") as fout:
        fout.write(f"\n=== run_paper_eval {baseline} ({_dt.datetime.now():%Y-%m-%d %H:%M:%S}) ===\n")
        fout.flush()
        rc = subprocess.run(
            [sys.executable, "scripts/run_paper_eval.py",
             "--baseline", baseline,
             "--test-set", "redirect_probe"],
            cwd=str(ROOT), env=env,
            stdout=fout, stderr=subprocess.STDOUT,
        ).returncode
    log(f"run_paper_eval {baseline} rc={rc}")
    return rc


def run_pairwise(judge_name: str, log_path: Path) -> int:
    log(f"running pairwise --judge {judge_name}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    with log_path.open("a", encoding="utf-8") as fout:
        fout.write(f"\n=== pairwise {judge_name} ({_dt.datetime.now():%Y-%m-%d %H:%M:%S}) ===\n")
        fout.flush()
        rc = subprocess.run(
            [sys.executable, "scripts/score_pairwise_preference.py",
             "--judge", judge_name],
            cwd=str(ROOT), env=env,
            stdout=fout, stderr=subprocess.STDOUT,
        ).returncode
    log(f"pairwise {judge_name} rc={rc}")
    return rc


def main() -> int:
    log(f"probe-clean-rerun orchestrator pid={os.getpid()} starting")
    kill_llama_server()

    # Phase 1: regenerate A2-SFT and A3-SFT responses on the cleaned probe
    for baseline in ADAPTER_BASELINES:
        rc = run_paper_eval_baseline(
            baseline, LOGS / f"clean_rerun_eval_{baseline}.log"
        )
        if rc != 0:
            log(f"FATAL: run_paper_eval {baseline} failed rc={rc}")
            return rc

    log("=== Phase 1 complete: A2-SFT and A3-SFT responses regenerated ===")

    # Phase 2: 3-judge pairwise pass with server swaps
    for judge_name, gguf, ctx, model_match in JUDGES:
        kill_llama_server()
        start_llama_server(gguf, ctx, model_match,
                           LOGS / f"clean_rerun_server_{judge_name}.log")
        run_pairwise(judge_name, LOGS / f"clean_rerun_pairwise_{judge_name}.log")

    kill_llama_server()
    log("=== PROBE-CLEAN RERUN COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
