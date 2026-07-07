"""run_eval_queue.py — post-training EVAL queue (runs after the multi-seed
training queue finishes its last job, a3_s7). Three phases, IN ORDER:

  PHASE 1  Un-trimmed A5/A6/A7 premature-firing eval (§5.3.1 trim closure).
           Eval-gen the un-trimmed A5/A6/A7 adapters + copy A1 (paper_a2,
           already un-trimmed) into ONE fresh dir, then score_sentinel_2x2.
           Mechanical, no judge servers. Compares un-trim vs the existing
           trimmed numbers (outputs/paper_v2/score/sentinel_2x2.json).

  PHASE 2  Multi-seed A1/A3 eval (3-seed mean +/- s.d. for §5.4 / §6.1).
           Delegates to the tested run_multiseed_eval.py (eval-gen +
           mechanical + judged + aggregate; it re-gates on the 4 seed
           adapters + free GPU, which pass instantly here).

  PHASE 3  Tier 1 cross-family PROMPT-ONLY baseline: Llama-3.1-8B-Instruct
           across the persistence prompting-ladder matched to the Qwen teacher
           (§5.3) — zero-shot, P-few (worked exemplars @320), P-cot (output-
           forcing scaffold = prompted CoT @768) — scored by mechanical
           sentinel recall; plus withholding (redirect_probe -> Llama + Gemma
           judges). P-think (native Qwen /think) is omitted: Llama has no native
           thinking mode. Rebuts "prompt-only failure is Qwen-specific" (§6.2).

GATE: waits until a3_s7's adapter exists AND the GPU is free (training truly
done), so this can be launched DETACHED NOW and will idle until the training
queue finishes.

Detached launch:
  PYTHONUTF8=1 nohup python scripts/run_eval_queue.py >> logs/eval_queue.log 2>&1 &

Idempotent + fail-forward: each step skips when its output already exists; a
failing phase is logged and the next phase still runs (matches the training
queue's fail-forward discipline). Self-contained (does not import the other
scripts' internals) so it survives an unattended overnight run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP = ROOT / "vendor" / "llama_cpp"
GGUF_DIR = "../models/GGUF"           # relative to LLAMA_CPP (cwd of llama-server)
PY = sys.executable
SENT = re.compile(r"\[SESSION_END")   # generic or axis-specific

# The final training job whose adapter marks "training queue complete".
FINAL_ADAPTER = ROOT / "outputs" / "paper_v2" / "a3_s7" / "sft" / "adapter_model.safetensors"

# Premature test sets for the trim analysis (mechanical; no judge).
PREM_SETS = "persistent_probe,persistent_premature_probe"
UNTRIM_DIR = ROOT / "outputs" / "paper_v2" / "eval_untrim"

# Tier 1
LLAMA_GGUF = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
GEMMA_GGUF = "gemma-2-9b-it-Q5_K_M.gguf"
TIER1_DIR = ROOT / "outputs" / "paper_v2" / "eval_llama31_baseline"
TIER1_BASE = "llama31_8b_instruct"
SCORE_DIR = ROOT / "outputs" / "paper_v2" / "score"

# Thermal cooldown between GPU-heavy stages, mirroring the training queue's
# 5-min inter-run pause (the RTX 3060 throttles ~83C). Applied at the
# sustained-load boundaries: after training, between phases, and between
# judge-server swaps. Tunable via env (set EVAL_GPU_COOLDOWN_S=0 to disable).
GPU_COOLDOWN_S = int(os.environ.get("EVAL_GPU_COOLDOWN_S", "300"))


def _log(m: str) -> None:
    print(f"[eval-queue {time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# GPU / llama-server helpers (self-contained)
# ---------------------------------------------------------------------------
def _gpu_free(threshold_mib: int = 2000) -> bool:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        return int(r.stdout.strip().split("\n")[0]) <= threshold_mib
    except Exception:
        return True


def _kill_server() -> None:
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True, text=True)
    time.sleep(4)


def _start_server(gguf: str, ctx: int, match: str, timeout_s: int = 360) -> bool:
    """Start llama-server on :8080 with `gguf`, wait until /v1/models reports
    a model id containing `match`."""
    _kill_server()
    (ROOT / "logs").mkdir(exist_ok=True)
    log_f = open(ROOT / "logs" / f"eval_queue_server_{match}.log", "a",
                 encoding="utf-8", buffering=1)
    subprocess.Popen(
        [str(LLAMA_CPP / "llama-server.exe"), "-m", f"{GGUF_DIR}/{gguf}",
         "-ngl", "80", "-c", str(ctx), "--host", "0.0.0.0", "--port", "8080",
         "--log-prefix"],
        cwd=str(LLAMA_CPP), stdout=log_f, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=3) as r:
                mid = json.load(r).get("data", [{}])[0].get("id", "").lower()
                if match.lower() in mid:
                    _log(f"  server ready: {mid}")
                    return True
        except Exception:
            pass
        time.sleep(3)
    _log(f"  server did not ready in {timeout_s}s ({gguf})")
    return False


def _run(cmd: list[str], label: str) -> int:
    _log(f"  run: {label}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    if rc != 0:
        _log(f"  !! nonzero rc={rc} for: {label} (continuing — fail-forward)")
    return rc


def _cooldown(reason: str) -> None:
    """Release the GPU (kill any server) and idle so the 3060 cools between
    sustained-load stages. No-op if EVAL_GPU_COOLDOWN_S=0."""
    _kill_server()
    if GPU_COOLDOWN_S <= 0:
        return
    _log(f"  GPU cooldown {GPU_COOLDOWN_S}s ({reason})")
    time.sleep(GPU_COOLDOWN_S)


def _recall(path: Path) -> float | None:
    if not path.exists():
        return None
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        return None
    return sum(1 for r in recs if SENT.search(r.get("generation", "") or "")) / len(recs)


# ---------------------------------------------------------------------------
# PHASE 1 — un-trimmed A5/A6/A7 premature firing (§5.3.1 trim closure)
# ---------------------------------------------------------------------------
def phase1_untrim_premature() -> None:
    _log("PHASE 1: un-trimmed A5/A6/A7 premature-firing eval (mechanical)")
    _kill_server()  # free VRAM for HF student inference

    # A5 registry path is STALE (points at the old trimmed outputs/paper/a5/sft),
    # so override --adapter-path to the un-trimmed paper_v2 adapter. A6/A7 registry
    # entries already point at paper_v2 and carry requires_env=generic.
    jobs = [
        ("paper_a5_sft", ["--adapter-path", "outputs/paper_v2/a5/sft"]),
        ("paper_a6_sft", []),
        ("paper_a7_sft", []),
    ]
    for base, extra in jobs:
        done = (UNTRIM_DIR / base / "persistent_premature_probe.jsonl").exists() and \
               (UNTRIM_DIR / base / "persistent_probe.jsonl").exists()
        if done:
            _log(f"  SKIP eval {base} (exists in eval_untrim)")
            continue
        _run([PY, "scripts/run_paper_eval.py", "--baseline", base, *extra,
              "--test-set", PREM_SETS, "--output-dir", str(UNTRIM_DIR)],
             f"eval untrim {base}")

    # A1 (paper_a2) is already un-trimmed; copy its premature+persistent
    # generations into the same dir so score_sentinel_2x2 assembles the full
    # 2x2 (it resolves every condition from one --eval-dir).
    src = ROOT / "outputs" / "paper_v2" / "eval" / "paper_a2"
    dst = UNTRIM_DIR / "paper_a2"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("persistent_probe.jsonl", "persistent_premature_probe.jsonl"):
        s = src / f
        if s.exists() and not (dst / f).exists():
            shutil.copyfile(s, dst / f)
            _log(f"  copied A1 {f} into eval_untrim/paper_a2")
        elif not s.exists():
            _log(f"  WARN: A1 source {s} missing — 2x2 will show A1 as n/a")

    out = SCORE_DIR / "sentinel_2x2_untrim.json"
    _run([PY, "scripts/score_sentinel_2x2.py",
          "--eval-dir", str(UNTRIM_DIR), "--out", str(out)],
         "score_sentinel_2x2 (untrim)")
    _log(f"  PHASE 1 done — untrim scores at {out}")
    _log("  compare premature_rate vs trimmed (outputs/paper_v2/score/sentinel_2x2.json:"
         " A5 0.560 / A6 0.642 / A7 0.692)")


# ---------------------------------------------------------------------------
# PHASE 2 — multi-seed A1/A3 (delegate to the tested run_multiseed_eval.py)
# ---------------------------------------------------------------------------
def phase2_multiseed() -> None:
    _log("PHASE 2: multi-seed A1/A3 eval (delegating to run_multiseed_eval.py)")
    # run_multiseed_eval re-gates on the 4 seed adapters + free GPU; both hold
    # here (phase 1's student process has exited, freeing VRAM). It also runs
    # the a1_1ep decorrelation eval first if missing, then eval-gen + mechanical
    # + judged (Llama/Gemma/Prometheus, batched) + aggregate mean+/-s.d.
    _run([PY, "scripts/run_multiseed_eval.py"], "run_multiseed_eval")
    _log("  PHASE 2 done — see outputs/paper_v2/score/multiseed_summary.json")


# ---------------------------------------------------------------------------
# PHASE 3 — Tier 1 cross-family prompt-only baseline (Llama-3.1-8B-Instruct)
# ---------------------------------------------------------------------------
def phase3_tier1_llama() -> None:
    _log("PHASE 3: Tier 1 — Llama-3.1-8B-Instruct prompt-only (matched ladder)")
    summary: dict = {"baseline": TIER1_BASE, "gguf": LLAMA_GGUF}
    d = TIER1_DIR / TIER1_BASE

    # 3a. Generate with Llama served on :8080. Matched to the Qwen teacher's
    #     persistence prompting-ladder (§5.3): SAME probe files + SAME budgets
    #     the teacher used (read from its generations' params.max_new_tokens):
    #       zero-shot  persistent_probe        @320  (recall)
    #                  redirect_probe          @320  (withholding probe)
    #       P-few      persistent_probe_pfew   @320  (worked 3-strike exemplars)
    #       P-cot      persistent_probe_pcot   @768  (output-forcing REASONING/
    #                                                 TALLY/DECISION/REPLY scaffold
    #                                                 = prompted CoT)
    #     P-think (native Qwen /think) is intentionally OMITTED: Llama-3.1 has no
    #     native thinking mode, so the deepest CoT rung it can reach is the
    #     prompted scaffold (P-cot). The api client's /no_think injection is a
    #     no-op on Llama.
    #     Withholding is scored on the canonical n=63 pedagogy_withholding_probe
    #     (21 redirect + 42 extra), matching §5.4 — not the 21-only redirect_probe.
    gen = [
        ("persistent_probe,pedagogy_withholding_probe", 320,
         ("persistent_probe.jsonl", "pedagogy_withholding_probe.jsonl")),
        ("persistent_probe_pfew", 320, ("persistent_probe_pfew.jsonl",)),
        ("persistent_probe_pcot", 768, ("persistent_probe_pcot.jsonl",)),
    ]
    need = [(ts, bud) for ts, bud, outs in gen
            if not all((d / o).exists() for o in outs)]
    if not need:
        _log("  SKIP Llama gen (all ladder outputs exist)")
    else:
        if not _start_server(LLAMA_GGUF, 32768, "llama"):
            _log("  Llama server failed to start — skipping PHASE 3")
            return
        for ts, bud in need:
            _run([PY, "scripts/run_paper_eval.py", "--baseline", TIER1_BASE,
                  "--test-set", ts, "--output-dir", str(TIER1_DIR),
                  "--max-new-tokens", str(bud)],
                 f"Llama gen {ts} @{bud}")

    # 3b. Mechanical persistence recall per rung — judge-free, the clean
    #     cross-family headline (does Llama fire the 3rd-strike sentinel with
    #     the full protocol in-prompt / with exemplars / with a scaffold?).
    #     Qwen 9B teacher reference (§5.3): zero 0.025, P-few 0.013, P-cot
    #     0.057, P-think-native 0.63 — trained A1 0.83.
    summary["persistence_recall"] = {
        "zero_shot": _recall(d / "persistent_probe.jsonl"),
        "p_few": _recall(d / "persistent_probe_pfew.jsonl"),
        "p_cot": _recall(d / "persistent_probe_pcot.jsonl"),
    }
    _log(f"  Llama persistence recall ladder = {summary['persistence_recall']}")
    _log("  (Qwen teacher ref: zero 0.025 / P-few 0.013 / P-cot 0.057 / "
         "P-think-native 0.63 ; trained A1 0.83)")

    # 3c. Withholding (zero-shot redirect_probe) — Llama judge (reuse server;
    #     the baseline GGUF IS the Llama-judge GGUF, so no swap), then Gemma.
    #     Self-preference caveat on the Llama-judged number; Gemma independent.
    wh: dict = {}
    if _start_server(LLAMA_GGUF, 32768, "llama"):
        out_ll = SCORE_DIR / "withhold_llama31base_llama.json"
        if not out_ll.exists():
            _run([PY, "scripts/score_withholding_rate.py",
                  "--judge", "llama31_8b_judge", "--eval-dir", str(TIER1_DIR),
                  "--baselines", TIER1_BASE,
                  "--probe-file", "pedagogy_withholding_probe.jsonl",
                  "--out", str(out_ll)], "withhold n=63 Llama-judge (self, caveat)")
        wh["llama_selfjudge"] = _withhold_of(out_ll, TIER1_BASE)
    else:
        _log("  Llama judge server failed; skipping Llama withholding")

    if _start_server(GEMMA_GGUF, 16384, "gemma"):
        out_gm = SCORE_DIR / "withhold_llama31base_gemma.json"
        if not out_gm.exists():
            _run([PY, "scripts/score_withholding_rate.py",
                  "--judge", "gemma2_9b_judge", "--eval-dir", str(TIER1_DIR),
                  "--baselines", TIER1_BASE,
                  "--probe-file", "pedagogy_withholding_probe.jsonl",
                  "--out", str(out_gm)], "withhold n=63 Gemma-judge (independent)")
        wh["gemma_judge"] = _withhold_of(out_gm, TIER1_BASE)
    else:
        _log("  Gemma judge server failed; skipping Gemma withholding")
    _kill_server()

    summary["withholding"] = wh
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    (SCORE_DIR / "tier1_llama_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"  PHASE 3 done — Tier1 summary: {summary}")


def _withhold_of(path: Path, baseline: str) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    for r in d.get("results", []):
        if r.get("baseline") == baseline:
            return r.get("withhold_rate")
    return None


# ---------------------------------------------------------------------------
def _wait_for_training_done() -> None:
    _log("eval-queue start; waiting for a3_s7 adapter (training queue done) + free GPU")
    waited = 0
    while True:
        if FINAL_ADAPTER.exists() and _gpu_free():
            break
        time.sleep(120)
        waited += 120
        if waited % 1800 == 0:
            _log(f"  still waiting: a3_s7 adapter={FINAL_ADAPTER.exists()}, "
                 f"gpu_free={_gpu_free()} ({waited // 60} min)")
    # small settle so the training process fully releases the device
    time.sleep(15)
    _log("a3_s7 present and GPU free — starting eval phases")


def main() -> int:
    _wait_for_training_done()
    # Cool after training (a3_s7 just finished hot) before eval hammers the card.
    _cooldown("post-training, before eval")
    phases = [phase1_untrim_premature, phase2_multiseed, phase3_tier1_llama]
    for i, phase in enumerate(phases):
        try:
            phase()
        except Exception as exc:  # fail-forward: one phase dying must not kill the rest
            _log(f"!! {phase.__name__} raised {type(exc).__name__}: {exc} — continuing")
        if i < len(phases) - 1:
            _cooldown(f"after {phase.__name__}")
    _log("=== EVAL QUEUE COMPLETE ===")
    _log("Next: fill §5.3.1 (untrim vs trim premature), replace §5.4/§6.1 (exp) "
         "with multiseed_summary.json, add Tier-1 Llama row (§6.2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
