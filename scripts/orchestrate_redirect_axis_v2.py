"""Minimal-fix re-judge of redirect_axis after eval_set rebuild.

After the 2026-06-24 minimal-fix changes:
- build_eval_sets drops heuristic-detected records (169 -> 140)
- build_eval_sets emits expected.scenario_context per record
- run_paper_score's REDIRECT_AXIS_PROMPT is enriched with scenario context
- Aggregator already drops Prometheus from redirect_axis (done 2026-06-24)

This script swaps llama-server through Llama-3.1 then Gemma-2 and
re-runs redirect_axis judging for all 9 baselines, then aggregates.
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
ORCH_LOG = LOGS / "orchestrate_redirect_axis_v2.log"
LLAMA_CPP = ROOT / "vendor" / "llama_cpp"

ALL_9 = ["paper_a1", "paper_a2", "paper_a3_sft", "paper_a4_sft", "paper_a5_sft",
         "qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct", "qwen3_5_4b_instruct",
         "qwen3_5_9b_teacher"]

JUDGES = [
    ("llama31_8b_judge", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     32768, "Llama-3.1"),
    ("gemma2_9b_judge", "gemma-2-9b-it-Q5_K_M.gguf",
     16384, "gemma-2"),
]


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


def start_llama_server(gguf: str, ctx: int, model_id_match: str,
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
                if model_id_match.lower() in mid.lower():
                    log(f"server ready: {mid}")
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"server didn't ready in 5min ({gguf})")


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


def main() -> int:
    log(f"orchestrator pid={os.getpid()} starting")
    for judge_name, gguf, ctx, model_match in JUDGES:
        kill_llama_server()
        start_llama_server(gguf, ctx, model_match,
                           LOGS / f"{judge_name}_server_v2.log")
        for b in ALL_9:
            run(
                [sys.executable, "scripts/run_paper_score.py",
                 "--baseline", b, "--metrics", "judged",
                 "--judge", judge_name, "--rubric", "generic",
                 "--metric-filter", "redirect_axis"],
                LOGS / f"{judge_name}_chain_v2.log",
                f"{judge_name} {b} redirect_axis",
            )
    kill_llama_server()

    log("aggregating across all 9 baselines")
    baselines = ",".join(ALL_9)
    judges_arg = "prometheus_7b_judge,llama31_8b_judge,gemma2_9b_judge"
    run(
        [sys.executable, "scripts/run_paper_score.py",
         "--aggregate", "--baselines", baselines, "--judges", judges_arg],
        LOGS / "aggregate_redirect_v2.log",
        "aggregate post-minimal-fix",
    )
    log("=== REDIRECT-AXIS V2 REFRESH COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
