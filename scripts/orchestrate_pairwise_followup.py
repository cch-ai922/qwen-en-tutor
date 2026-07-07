"""Pairwise-preference orchestrator (§5.4.5).

Waits for the current redirect_axis re-judge orchestrator to write
"REDIRECT-AXIS V2 REFRESH COMPLETE", then:
  1. Llama-3.1 already loaded -> run pairwise persona/role_swap preference
  2. Swap to Gemma-2 -> run pairwise again
  3. Aggregate, summarize, write outputs/paper/score/pairwise_preference.json
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
ORCH_LOG = LOGS / "orchestrate_pairwise_followup.log"
UPSTREAM_LOG = LOGS / "orchestrate_redirect_axis_v2.log"
LLAMA_CPP = ROOT / "vendor" / "llama_cpp"

JUDGES = [
    ("prometheus_7b_judge", "prometheus-7b-v2.0.Q4_K_M.gguf",
     32768, "prometheus"),
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
    log(f"pairwise orchestrator pid={os.getpid()} starting")
    wait_for_marker(UPSTREAM_LOG, "REDIRECT-AXIS V2 REFRESH COMPLETE")

    for judge_name, gguf, ctx, model_match in JUDGES:
        kill_llama_server()
        start_llama_server(gguf, ctx, model_match,
                           LOGS / f"{judge_name}_pairwise_server.log")
        run_pairwise(judge_name, LOGS / f"pairwise_{judge_name}.log")

    kill_llama_server()
    log("=== PAIRWISE-PREFERENCE FOLLOWUP COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
