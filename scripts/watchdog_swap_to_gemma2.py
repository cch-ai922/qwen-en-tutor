"""Watchdog: when Llama-3.1 judge chain finishes, swap to Gemma-2 and run the
identical chain, then aggregate all three judges across all 9 baselines.

Detects Llama-3.1 completion by polling the chain log for the
"LLAMA-3.1 CHAIN COMPLETE" marker line written by run_llama31 chain.

Process-level operations (kill llama-server, start a new one) use
subprocess + signal portably; on Windows we rely on tasklist/taskkill.
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
LLAMA_LOG = ROOT / "logs" / "llama31_chain.log"
GEMMA_LOG = ROOT / "logs" / "gemma2_chain.log"
SERVER_LOG = ROOT / "logs" / "gemma2_server.log"
WATCHDOG_LOG = ROOT / "logs" / "watchdog_gemma2.log"
LLAMA_CPP = ROOT / "vendor" / "llama_cpp"
GGUF_REL = "../models/GGUF/gemma-2-9b-it-Q5_K_M.gguf"

BASELINES_TUTOR = [
    "paper_a1", "paper_a2", "paper_a3_sft", "paper_a4_sft", "paper_a5_sft",
    "qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct", "qwen3_5_4b_instruct",
]
AGG_BASELINES = ",".join(BASELINES_TUTOR + ["qwen3_5_9b_teacher"])
AGG_JUDGES = "prometheus_7b_judge,llama31_8b_judge,gemma2_9b_judge"


def log(msg: str) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_for_marker(path: Path, marker: str, poll_s: int = 30) -> None:
    log(f"watching {path} for marker '{marker}'")
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
    # Windows: taskkill any llama-server.exe
    log("stopping llama-server processes")
    try:
        subprocess.run(
            ["taskkill", "/IM", "llama-server.exe", "/F"],
            check=False, capture_output=True, text=True,
        )
    except FileNotFoundError:
        log("taskkill not available; skipping (server may persist)")
    time.sleep(5)


def start_gemma_server() -> subprocess.Popen:
    log("starting gemma-2 llama-server")
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    fout = SERVER_LOG.open("a", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        [
            str(LLAMA_CPP / "llama-server.exe"),
            "-m", GGUF_REL,
            "-ngl", "80",
            "-c", "16384",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--log-prefix",
        ],
        cwd=str(LLAMA_CPP),
        stdout=fout,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    log(f"gemma-2 server pid={proc.pid}")
    return proc


def wait_ready(timeout_s: int = 300) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8080/v1/models", timeout=3
            ) as r:
                body = json.load(r)
                mid = body.get("data", [{}])[0].get("id", "")
                if "gemma-2" in mid.lower():
                    log(f"gemma-2 server ready: {mid}")
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def run_score(args: list[str], phase: str) -> None:
    log(f"phase: {phase}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    with GEMMA_LOG.open("a", encoding="utf-8") as fout:
        fout.write(f"\n=== {phase} ({_dt.datetime.now():%Y-%m-%d %H:%M:%S}) ===\n")
        fout.flush()
        rc = subprocess.run(
            [sys.executable, "scripts/run_paper_score.py"] + args,
            cwd=str(ROOT),
            env=env,
            stdout=fout, stderr=subprocess.STDOUT,
        ).returncode
    log(f"phase {phase} rc={rc}")


def main() -> int:
    log("watchdog armed")
    wait_for_marker(LLAMA_LOG, "LLAMA-3.1 CHAIN COMPLETE")

    kill_llama_server()
    proc = start_gemma_server()
    if not wait_ready(300):
        log("ERROR: gemma-2 server failed to start within 5min")
        try:
            proc.terminate()
        except Exception:
            pass
        return 1

    # --- Gemma-2 chain (matches Prometheus / Llama-3.1 scope) ---
    GEMMA_LOG.write_text(
        f"=== gemma-2 chain start {_dt.datetime.now():%Y-%m-%d %H:%M:%S} ===\n",
        encoding="utf-8",
    )
    started = time.time()

    # Gemma-2 pass: generic + tutor_v2 only. tutor v1 dropped per user.
    run_score(
        ["--baseline", "qwen3_5_9b_teacher", "--metrics", "judged",
         "--judge", "gemma2_9b_judge", "--rubric", "generic"],
        "B4 (teacher) generic",
    )
    for b in BASELINES_TUTOR + ["qwen3_5_9b_teacher"]:
        run_score(
            ["--baseline", b, "--metrics", "judged",
             "--judge", "gemma2_9b_judge", "--rubric", "tutor_v2",
             "--metric-filter", "cefr_adherence,naturalness"],
            f"{b} tutor_v2",
        )

    with GEMMA_LOG.open("a", encoding="utf-8") as f:
        f.write("=== GEMMA-2 CHAIN COMPLETE ===\n")
    elapsed = (time.time() - started) / 60
    log(f"gemma-2 chain done in {elapsed:.1f} min")

    # --- Aggregation across all 3 judges + 9 baselines ---
    log("aggregating all 3 judges x 9 baselines")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    agg_log = ROOT / "logs" / "aggregate.log"
    with agg_log.open("a", encoding="utf-8") as fout:
        fout.write(f"=== aggregate {_dt.datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
        rc = subprocess.run(
            [sys.executable, "scripts/run_paper_score.py",
             "--aggregate", "--baselines", AGG_BASELINES,
             "--judges", AGG_JUDGES],
            cwd=str(ROOT),
            env=env,
            stdout=fout, stderr=subprocess.STDOUT,
        ).returncode
    log(f"aggregate rc={rc}")
    log("=== WATCHDOG COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
