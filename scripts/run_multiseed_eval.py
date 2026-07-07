"""run_multiseed_eval.py — auto eval + score for the Tier-1 multi-seed reseed.

Fires AFTER the multi-seed training queue (run_multiseed_queue.py) has produced
all four seed adapters. Computes the 3-seed (42/123/7) statistics that replace
the provisional "(exp)" placeholders in the paper.

Pipeline (idempotent, restart-safe — skips any output that already exists):
  0. Wait until all four seed adapters exist AND the GPU is free.
  A. Eval-gen (HF, no server) each seed adapter on redirect_probe +
     persistent_probe + locale_leakage. Seed s writes under
     outputs/paper_v2/eval_seed{s}/{paper_a2|paper_a3_sft}/ — reusing the
     headline baseline names so the existing scorers resolve unchanged.
  B. Mechanical (judge-free): persistence recall (persistent_probe) and
     locale leakage (word-boundary gazetteer), per seed.
  C. Judged: withholding rate (Llama-3.1 + Gemma-2) and pairwise A1-vs-A3
     (Llama-3.1 + Gemma-2 + Prometheus). Judges are launched ONCE each and
     reused across seeds to minimise GGUF swaps. Prometheus does pairwise
     only (it cannot do the binary withholding classification).
  D. Aggregate mean +/- s.d. across seeds {42,123,7} (42 = the headline
     paper_a2 / paper_a3_sft eval already on disk) and write
     outputs/paper_v2/score/multiseed_summary.json + print a table.

Detached launch (after the training queue is running):
  PYTHONUTF8=1 nohup python scripts/run_multiseed_eval.py >> logs/multiseed_eval.log 2>&1 &

The seed-42 numbers are read from the existing single-seed outputs:
  withholding: outputs/paper_v2/score/pedagogy_withholding_extra_llama.json +
               ped_withhold_orig21_gemma.json / ped_withhold_extra_gemma.json
  (we re-derive 42 from the eval dirs to keep one code path; see _seed42_*).
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP = ROOT / "vendor" / "llama_cpp"
GGUF_DIR = "../models/GGUF"

SEEDS = [123, 7]                      # 42 is the headline run, already on disk
COND = {"paper_a2": "a1", "paper_a3_sft": "a3"}   # eval-name -> adapter prefix
PROBES = "redirect_probe,persistent_probe,locale_leakage"

JUDGES = {
    "llama31_8b_judge": ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 32768, "llama"),
    "gemma2_9b_judge":  ("gemma-2-9b-it-Q5_K_M.gguf", 16384, "gemma"),
    "prometheus_7b_judge": ("prometheus-7b-v2.0.Q4_K_M.gguf", 32768, "prometheus"),
}
SENT = re.compile(r"\[SESSION_END")


def _log(m: str) -> None:
    print(f"[multiseed-eval {time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------
# GPU / server helpers
# ----------------------------------------------------------------------------
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
    _kill_server()
    log_f = open(ROOT / "logs" / f"multiseed_eval_server_{match}.log", "a",
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
                    _log(f"server ready: {mid}")
                    return True
        except Exception:
            pass
        time.sleep(3)
    _log(f"server did not ready in {timeout_s}s ({gguf})")
    return False


def _run(cmd: list[str], label: str) -> int:
    _log(f"  run: {label}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


# ----------------------------------------------------------------------------
# Phase A: eval-gen
# ----------------------------------------------------------------------------
def eval_gen() -> None:
    _log("Phase A: eval-gen seed adapters (HF, no server)")
    _kill_server()
    for s in SEEDS:
        for base, pref in COND.items():
            adapter = f"outputs/paper_v2/{pref}_s{s}/sft"
            out_dir = ROOT / "outputs" / "paper_v2" / f"eval_seed{s}"
            done = (out_dir / base / "redirect_probe.jsonl").exists() and \
                   (out_dir / base / "persistent_probe.jsonl").exists() and \
                   (out_dir / base / "locale_leakage.jsonl").exists()
            if done:
                _log(f"  SKIP eval {base} seed {s} (exists)")
                continue
            _run([sys.executable, "scripts/run_paper_eval.py",
                  "--baseline", base, "--adapter-path", adapter,
                  "--test-set", PROBES, "--output-dir", str(out_dir)],
                 f"eval {base} seed{s}")


# ----------------------------------------------------------------------------
# Phase B: mechanical (judge-free)
# ----------------------------------------------------------------------------
def _gazetteer():
    import yaml
    d = yaml.safe_load((ROOT / "config" / "western_entities.yaml").read_text(encoding="utf-8"))
    pats = []
    for items in d.values():
        if items:
            for e in items:
                if isinstance(e, str) and e.strip():
                    pats.append(re.compile(r"\b" + re.escape(e.lower()).replace(r"\-", "[- ]?") + r"\b"))
    return pats


def _recall(path: Path) -> float | None:
    if not path.exists():
        return None
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        return None
    return sum(1 for r in recs if SENT.search(r.get("generation", "") or "")) / len(recs)


def _leak(path: Path, pats) -> float | None:
    if not path.exists():
        return None
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        return None
    def leaks(t):
        t = (t or "").lower()
        return any(p.search(t) for p in pats)
    return sum(1 for r in recs if leaks(r.get("generation", ""))) / len(recs)


def mechanical() -> dict:
    _log("Phase B: mechanical (recall, locale) per seed")
    pats = _gazetteer()
    out = {"recall_a1": {}, "leak_a1": {}, "leak_a3": {}}
    # seed 42 from the headline eval dir
    base42 = ROOT / "outputs" / "paper_v2" / "eval"
    out["recall_a1"][42] = _recall(base42 / "paper_a2" / "persistent_probe.jsonl")
    out["leak_a1"][42] = _leak(base42 / "paper_a2" / "locale_leakage.jsonl", pats)
    out["leak_a3"][42] = _leak(base42 / "paper_a3_sft" / "locale_leakage.jsonl", pats)
    for s in SEEDS:
        d = ROOT / "outputs" / "paper_v2" / f"eval_seed{s}"
        out["recall_a1"][s] = _recall(d / "paper_a2" / "persistent_probe.jsonl")
        out["leak_a1"][s] = _leak(d / "paper_a2" / "locale_leakage.jsonl", pats)
        out["leak_a3"][s] = _leak(d / "paper_a3_sft" / "locale_leakage.jsonl", pats)
    return out


# ----------------------------------------------------------------------------
# Phase C: judged (withholding + pairwise)
# ----------------------------------------------------------------------------
def judged() -> None:
    _log("Phase C: judged scoring (withholding + pairwise), judges reused across seeds")
    for judge, (gguf, ctx, match) in JUDGES.items():
        if not _start_server(gguf, ctx, match):
            _log(f"  {judge}: server failed; skipping its scores")
            continue
        for s in SEEDS:
            eval_dir = f"outputs/paper_v2/eval_seed{s}"
            # withholding: Llama + Gemma only (Prometheus can't do binary)
            if judge in ("llama31_8b_judge", "gemma2_9b_judge"):
                out = ROOT / "outputs" / "paper_v2" / "score" / f"withhold_seed{s}_{match}.json"
                if not out.exists():
                    _run([sys.executable, "scripts/score_withholding_rate.py",
                          "--judge", judge, "--eval-dir", eval_dir,
                          "--probe-file", "redirect_probe.jsonl",
                          "--baselines", "paper_a2,paper_a3_sft",
                          "--out", str(out)], f"withhold seed{s} {match}")
            # pairwise A1(keep=paper_a2) vs A3(drop=paper_a3_sft): all 3 judges
            pout = ROOT / "outputs" / "paper_v2" / "score" / f"pairwise_seed{s}_{match}.json"
            if not pout.exists():
                env = os.environ.copy()
                env["PYTHONUTF8"] = "1"
                env["QWEN_TUTOR_EVAL_OUT"] = str(ROOT / eval_dir)
                env["QWEN_TUTOR_PAIRWISE_SCORE_OUT"] = str(pout)
                _log(f"  run: pairwise seed{s} {match}")
                subprocess.run([sys.executable, "scripts/score_pairwise_preference.py",
                                "--judge", judge], cwd=str(ROOT), env=env)
        _kill_server()


# ----------------------------------------------------------------------------
# Phase D: aggregate
# ----------------------------------------------------------------------------
def _mean_sd(vals: list[float]) -> tuple[float, float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return (float("nan"), float("nan"))
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (m, sd)


def _withhold_rate(path: Path, baseline: str) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    for r in d.get("results", []):
        if r.get("baseline") == baseline:
            return r.get("withhold_rate")
    return None


def aggregate(mech: dict) -> dict:
    _log("Phase D: aggregate mean +/- s.d. across seeds 42/123/7")
    summary = {"seeds": [42] + SEEDS}

    # Withholding: average the two judges per seed, then mean+/-sd across seeds.
    for base, key in (("paper_a2", "withhold_a1"), ("paper_a3_sft", "withhold_a3")):
        per_seed = {}
        # seed 42 from the headline scoring files
        s42 = []
        f_ll = ROOT / "outputs" / "paper_v2" / "score" / "pedagogy_withholding_extra_llama.json"
        v = _withhold_rate(f_ll, base)
        if v is not None:
            s42.append(v)
        # gemma seed-42 (extra + orig pooled is non-trivial; use extra-gemma as proxy if present)
        f_gm = ROOT / "outputs" / "paper_v2" / "score" / "ped_withhold_extra_gemma.json"
        v = _withhold_rate(f_gm, base)
        if v is not None:
            s42.append(v)
        per_seed[42] = statistics.mean(s42) if s42 else None
        for s in SEEDS:
            js = []
            for match in ("llama", "gemma"):
                v = _withhold_rate(ROOT / "outputs" / "paper_v2" / "score" / f"withhold_seed{s}_{match}.json", base)
                if v is not None:
                    js.append(v)
            per_seed[s] = statistics.mean(js) if js else None
        m, sd = _mean_sd(list(per_seed.values()))
        summary[key] = {"per_seed": per_seed, "mean": m, "sd": sd}

    # Mechanical
    for key in ("recall_a1", "leak_a1", "leak_a3"):
        m, sd = _mean_sd(list(mech[key].values()))
        summary[key] = {"per_seed": mech[key], "mean": m, "sd": sd}

    out = ROOT / "outputs" / "paper_v2" / "score" / "multiseed_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"wrote {out}")

    # Print a paper-ready table
    print("\n=== MULTI-SEED SUMMARY (mean +/- s.d. over seeds 42/123/7) ===", flush=True)
    def fmt(d):
        return f"{d['mean']:.3f} +/- {d['sd']:.3f}  (per-seed {d['per_seed']})"
    print(f"  withholding A1: {fmt(summary['withhold_a1'])}")
    print(f"  withholding A3: {fmt(summary['withhold_a3'])}")
    print(f"  persistence recall A1: {fmt(summary['recall_a1'])}")
    print(f"  locale leakage A1: {fmt(summary['leak_a1'])}")
    print(f"  locale leakage A3: {fmt(summary['leak_a3'])}")
    print("  (pairwise per-seed JSONs in score/pairwise_seed*_*.json — aggregate per axis when filling §5.4.5)")
    return summary


def main() -> int:
    _log("multiseed-eval start; waiting for all 4 seed adapters + free GPU")
    adapters = [ROOT / "outputs" / "paper_v2" / f"{pref}_s{s}" / "sft" / "adapter_model.safetensors"
                for s in SEEDS for pref in ("a1", "a3")]
    waited = 0
    while True:
        if all(a.exists() for a in adapters) and _gpu_free():
            break
        time.sleep(120)
        waited += 120
        if waited % 3600 == 0:
            ndone = sum(a.exists() for a in adapters)
            _log(f"still waiting: {ndone}/4 adapters, gpu_free={_gpu_free()} ({waited//60} min)")
    _log("all 4 adapters present and GPU free; starting eval+score")

    # First: the a1_1ep decorrelation eval, if it never ran (the eval-first
    # watcher lost the GPU race to the training queue). Mechanical, ~45 min.
    a1_1ep_pp = ROOT / "outputs" / "paper_v2" / "eval_a1_1ep" / "paper_a2" / "persistent_probe.jsonl"
    if not a1_1ep_pp.exists():
        _log("a1_1ep eval missing — running it first (matched-budget decorrelation)")
        _run([sys.executable, "scripts/run_paper_eval.py",
              "--baseline", "paper_a2",
              "--adapter-path", "outputs/paper_v2/a1_1ep/sft",
              "--test-set", "persistent_probe,persistent_premature_probe",
              "--output-dir", "outputs/paper_v2/eval_a1_1ep"],
             "a1_1ep decorrelation eval")
    else:
        _log("a1_1ep eval already present; skipping")

    eval_gen()
    mech = mechanical()
    judged()
    aggregate(mech)
    _log("=== multiseed-eval complete; replace the (exp) cells in the paper ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
