"""Master orchestrator for v2 final retrain.

Runs the full pipeline end-to-end, unattended:

  Stage 1  backup_and_delete         fail-stop
  Stage 2  full_regen                fail-stop  (teacher server started here)
  Stage 3  smoke_quality_check       fail-stop (catches prompt bugs)
  Stage 4a-d  sft_retrain_a{1,3,4,5} fail-forward across conditions
  Stage 5  dpo_a1                    fail-forward
  Stage 6a sft_eval_gen              fail-forward per (baseline, test_set)
  Stage 6b score_mechanical          fail-forward
  Stage 6c score_per_axis_f1         fail-forward
  Stage 6d score_pairwise            fail-forward
  Stage 6e score_naturalness         fail-forward
  END marker

Checkpoints in outputs/paper_v2/.checkpoints/{stage}.done. Each stage's
checkpoint file is written ONLY on success; relaunch resumes from the
last completed stage. No marker-polling across orchestrators (the
previous orchestrator chain's collision bug is by-design eliminated --
this is a single process driving everything).

Output goes under outputs/paper_v2/ to preserve v1 outputs/paper/ as
the comparison baseline.

Resource management: all llama-server lifecycle (teacher + judges) is
managed here.  The teacher (Qwen3.5-9B-UD-Q4_K_XL.gguf) and all judge
GGUFs share the same RTX 3060 12GB GPU with SFT/DPO training, so only
one server is ever running at a time:

  Stage 2 regen:     teacher server UP,  kill before Stage 4
  Stage 4-5 train:   no server
  Stage 6 HF eval:   no server (teacher killed; HF models loaded directly)
  Stage 6 teach eval: teacher server UP, kill after
  Stage 6c-f score:  judge GGUFs, one at a time, swapped per scoring round
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
ORCH_LOG = LOGS / "orchestrate_v2_final.log"

PAPER_V2 = ROOT / "outputs" / "paper_v2"
CHECKPOINTS = PAPER_V2 / ".checkpoints"
EVAL_OUT = PAPER_V2 / "eval"
SCORE_OUT = PAPER_V2 / "score"
LLAMA_CPP = ROOT / "vendor" / "llama_cpp"

SPECIALIZED_STREAMS = (
    "locale_redirect", "language_redirect", "pedagogy_redirect",
    "persona_redirect", "role_swap_redirect", "topic_redirect",
)
PERSISTENT_STREAMS = (
    "persistent_off_topic", "persistent_language_violation",
    "persistent_persona_break", "persistent_role_swap",
)
# v2 condition set (revised 2026-06-26 per user direction):
#   - DROP A4 (no persistent) — secondary claim, not paper-critical
#   - DROP A1's DPO step — paper claims are SFT-data-design claims;
#                          DPO can be retrained as a follow-up
#   - ADD A6 (generic [SESSION_END], fixed turn-7 persistent)
#   - ADD A7 (generic [SESSION_END], 4-variant persistent)
# Together A2/A5/A6/A7 form a clean 2x2:
#                        4-variant persistent     fixed turn-7
#   axis-specific:             A2                       A5
#   generic [SESSION_END]:     A7                       A6
# A1's SFT step still runs (its adapter IS the "paper_a2" SFT-only
# baseline); we just skip the DPO step. So orchestrator stages are:
#   stage4_sft_a1  -> trains the SFT adapter shared by paper_a2
#   stage4_sft_a3  -> SFT-only ablation (no specialized OR persistent)
#   stage4_sft_a5  -> SFT-only ablation (fixed-turn-7 persistent)
#   stage4_sft_a6  -> SFT-only ablation (fixed-turn-7 + generic sentinel)
#   stage4_sft_a7  -> SFT-only ablation (4-variant + generic sentinel)
SFT_CONDITIONS = ("a1", "a3", "a6", "a7", "a5")

# Adapter baselines (HF). Paper convention (revised v2):
#   paper_a2     = A1 SFT only       (SFT adapter from stage4_sft_a1)
#   paper_a3_sft = A3 SFT only       (drops specialized + persistent)
#   paper_a5_sft = A5 SFT only       (fixed-turn-7 persistent)
#   paper_a6_sft = A6 SFT only       (fixed-turn-7 + generic [SESSION_END])
#   paper_a7_sft = A7 SFT only       (4-variant + generic [SESSION_END])
# Note: paper_a1 (SFT+DPO) and paper_a4_sft are DROPPED for v2.
ADAPTER_BASELINES = ("paper_a2", "paper_a3_sft",
                     "paper_a5_sft", "paper_a6_sft", "paper_a7_sft")
# Off-the-shelf zero-shot baselines. The 3 small ones are HF (no
# llama-server needed). The 9B teacher runs via the local llama-server
# (this orchestrator starts/stops it automatically).
HF_OFFSHELF_BASELINES = ("qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct",
                         "qwen3_5_4b_instruct")
TEACHER_BASELINE = "qwen3_5_9b_teacher"
ALL_EVAL_BASELINES = (*ADAPTER_BASELINES, *HF_OFFSHELF_BASELINES,
                      TEACHER_BASELINE)
TEST_SETS = (
    "tutor_scenario", "locale_leakage", "redirect_probe",
    "persistent_probe", "persistent_fp_probe", "persistent_offposition_probe",
)

# Teacher GGUF (local — same RTX 3060 as training; must be killed before SFT/DPO)
TEACHER_GGUF   = "Qwen3.5-9B-UD-Q4_K_XL.gguf"
TEACHER_CTX    = 32768
TEACHER_MATCH  = "Qwen3.5"   # substring checked in /v1/models response

# Judge GGUFs for scoring
JUDGES = (
    ("prometheus_7b_judge", "prometheus-7b-v2.0.Q4_K_M.gguf",         32768, "prometheus"),
    ("llama31_8b_judge",    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 32768, "Llama-3.1"),
    ("gemma2_9b_judge",     "gemma-2-9b-it-Q5_K_M.gguf",              16384, "gemma-2"),
)

TIMESTAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = ROOT / f"data_backup_pre_v2_{_dt.datetime.now():%Y%m%d}"


# ----------------------------------------------------------------------
# Logging + checkpoint helpers
# ----------------------------------------------------------------------

def log(msg: str) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {msg}"
    print(line, flush=True)
    with ORCH_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def checkpoint_path(stage: str) -> Path:
    return CHECKPOINTS / f"{stage}.done"


def is_done(stage: str) -> bool:
    return checkpoint_path(stage).exists()


def mark_done(stage: str, extra: dict | None = None) -> None:
    payload = {"stage": stage, "completed": _dt.datetime.now().isoformat()}
    if extra:
        payload.update(extra)
    checkpoint_path(stage).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_subprocess(cmd: list[str], log_path: Path, label: str,
                   extra_env: dict | None = None) -> int:
    log(f"  {label}: starting")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"  # so subprocess output streams to log in real time
    # v2 routing: point scoring scripts at outputs/paper_v2/
    env["QWEN_TUTOR_EVAL_OUT"] = str(EVAL_OUT)
    env["QWEN_TUTOR_SCORE_OUT"] = str(SCORE_OUT)
    env["QWEN_TUTOR_MECH_SCORE_OUT"] = str(SCORE_OUT / "redirect_mechanical.json")
    env["QWEN_TUTOR_PAIRWISE_SCORE_OUT"] = str(SCORE_OUT / "pairwise_preference.json")
    # SAFETY: scrub sentinel/variant env vars from the inherited parent
    # env. If a developer's shell has these set (e.g. from running a
    # convert/regen script earlier in the same terminal), they would
    # leak into A1/A2/A3/A5 subprocesses and corrupt their training data
    # / eval system prompts.  Only A6's stage4 / paper_a6_sft eval are
    # allowed to set them via extra_env below.
    for leaky in ("QWEN_TUTOR_SENTINEL_FORMAT",
                  "QWEN_TUTOR_PERSISTENT_FORCED_VARIANT"):
        env.pop(leaky, None)
    if extra_env:
        env.update(extra_env)
        # Audit log so any per-condition override is visible in the
        # orchestrator log (catches accidentally-leaving-it-on bugs).
        for k, v in extra_env.items():
            if k.startswith("QWEN_TUTOR_"):
                log(f"    {label}: env override {k}={v}")
    with log_path.open("a", encoding="utf-8") as fout:
        fout.write(f"\n=== {label} ({_dt.datetime.now():%Y-%m-%d %H:%M:%S}) ===\n")
        fout.flush()
        try:
            rc = subprocess.run(
                cmd, cwd=str(ROOT), env=env,
                stdout=fout, stderr=subprocess.STDOUT,
            ).returncode
        except FileNotFoundError as e:
            log(f"  {label}: launch FAILED: {e}")
            return -1
    log(f"  {label}: rc={rc}")
    return rc


def kill_llama_server() -> None:
    try:
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"],
                       check=False, capture_output=True, text=True)
    except FileNotFoundError:
        pass
    time.sleep(4)


def start_llama_server(gguf: str, ctx: int, model_match: str,
                       log_path: Path) -> bool:
    """Return True if server is up and serving the expected model."""
    log(f"  starting llama-server with {gguf}")
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
                    log(f"  server ready: {mid}")
                    return True
        except Exception:
            pass
        time.sleep(3)
    log(f"  llama-server did not ready in 5 min ({gguf})")
    return False


# ----------------------------------------------------------------------
# Stage implementations
# ----------------------------------------------------------------------

def stage1_backup_and_delete() -> int:
    """Snapshot data/, delete 6 specialized SFT streams + DPO pairs."""
    log("Stage 1: backup data/ and delete stale specialized streams")
    if not BACKUP_DIR.exists():
        log(f"  copying data/ -> {BACKUP_DIR.name}/ (this may take a few minutes)")
        shutil.copytree(ROOT / "data", BACKUP_DIR)
    else:
        log(f"  backup {BACKUP_DIR.name} exists; not recopying")

    # Delete the 6 specialized streams from all sft_raw + filtered dirs
    deleted = 0
    for sub in ("sft_raw", "sft_filtered", "sft_filtered_a4", "sft_filtered_a5"):
        d = ROOT / "data" / sub
        if not d.exists():
            continue
        for stream in SPECIALIZED_STREAMS:
            for f in d.glob(f"{stream}_*.jsonl"):
                f.unlink()
                deleted += 1
    log(f"  deleted {deleted} specialized SFT files")

    # Delete DPO pairs (rebuilt from new SFT student)
    dpo_dir = ROOT / "data" / "dpo_pairs"
    if dpo_dir.exists():
        shutil.rmtree(dpo_dir)
        log("  deleted data/dpo_pairs/")
    dpo_filtered = ROOT / "data" / "dpo_filtered"
    if dpo_filtered.exists():
        shutil.rmtree(dpo_filtered)
        log("  deleted data/dpo_filtered/")

    return 0


def stage2_full_regen() -> int:
    """Run the full data-generation pipeline for the 6 specialized streams +
    persistent topup. Starts the teacher server (local llama-server with the
    9B Qwen3.5 GGUF) and kills it when done so SFT training in Stage 4 gets
    full VRAM."""
    log("Stage 2: full regen via run_generation.py")
    kill_llama_server()   # clean slate
    if not start_llama_server(TEACHER_GGUF, TEACHER_CTX, TEACHER_MATCH,
                              LOGS / "v2_final_stage2_teacher_server.log"):
        log("Stage 2: teacher server failed to start — aborting regen")
        return 1
    stages_str = ",".join((*SPECIALIZED_STREAMS, *PERSISTENT_STREAMS,
                           "filter_sft", "top_up", "persistent_topup"))
    rc = run_subprocess(
        [sys.executable, "scripts/run_generation.py",
         "--stages", stages_str],
        LOGS / "v2_final_stage2_regen.log", "stage2_full_regen",
    )
    kill_llama_server()   # free VRAM for Stage 3 smoke check + Stage 4 SFT
    return rc


def stage2b_setup_ablation_data() -> int:
    """Refresh per-condition filtered dirs after regen.

    sft_filtered_a3, sft_filtered_a4, sft_filtered_a5 use hardlinks into
    sft_filtered/. After filter_sft unlinks + recreates each _passed.jsonl
    (new inode), those hardlinks point to stale old-data inodes. This stage
    clears and re-links all per-condition dirs from the freshly filtered
    sft_filtered/ so every training condition trains on current data.

    Fail-stop: if this fails, per-condition dirs are stale and A3/A4/A5
    training would use old data.
    """
    log("Stage 2b: refresh per-condition ablation data dirs (setup_paper_ablation_data.py)")
    # v2 condition set: a3 (no redirects), a5 (fixed-turn-7), a6 (fixed-turn-7
    # + generic), a7 (4-variant + generic). a4 was dropped; a1/a2 use
    # sft_filtered/ directly.
    rc = run_subprocess(
        [sys.executable, "scripts/setup_paper_ablation_data.py",
         "--conditions", "a3,a5,a6,a7"],
        LOGS / "v2_final_stage2b_ablation_setup.log", "setup_ablation_data",
    )
    return rc


def stage3_smoke_quality_check() -> int:
    """Automated quality gate on the regenerated data. Halts pipeline if
    new prompts produce bad output."""
    log("Stage 3: automated quality check on regenerated SFT data")
    rc = run_subprocess(
        [sys.executable, "scripts/smoke_quality_check.py"],
        LOGS / "v2_final_stage3_quality.log", "stage3_quality_check",
    )
    return rc


def stage4_sft_retrain(cond: str) -> int:
    """SFT retrain for one condition. Reads config/paper_v2/training_<cond>_*.yaml.
    llama-server must be DOWN before training starts (VRAM)."""
    log(f"Stage 4: SFT retrain {cond}")
    kill_llama_server()
    cfg_map = {
        "a1": "config/paper_v2/training_a1_full.yaml",
        "a3": "config/paper_v2/training_a3_no_specialized.yaml",
        "a5": "config/paper_v2/training_a5_fixed_turn_7.yaml",
        "a6": "config/paper_v2/training_a6_generic_sentinel.yaml",
        "a7": "config/paper_v2/training_a7_generic_sentinel.yaml",
    }
    if cond not in cfg_map:
        log(f"  unknown condition {cond}; skip")
        return 1
    # Per-condition env vars. A6 training MUST set
    # QWEN_TUTOR_SENTINEL_FORMAT=generic because the training formatter
    # (formatter.py:222) re-renders the deployment system prompt from
    # record metadata — it does NOT use the baked-in
    # example.system_prompt. Without this, A6 records would be served
    # with the axis-specific [persistence] block, mismatched against the
    # generic [SESSION_END] in the assistant turns.
    cond_env: dict[str, str] = {}
    if cond in ("a6", "a7"):
        cond_env["QWEN_TUTOR_SENTINEL_FORMAT"] = "generic"
    rc = run_subprocess(
        [sys.executable, "scripts/run_training.py",
         "--training-config", cfg_map[cond],
         "--stages", "train_sft"],
        LOGS / f"v2_final_stage4_sft_{cond}.log", f"sft_retrain_{cond}",
        extra_env=cond_env or None,
    )
    return rc


def stage5_dpo_a1() -> int:
    """DPO retrain for A1.

    Split into two sub-runs to handle the VRAM constraint:
      - on_policy_gen: HF student (0.8B) + teacher API call — both fit in
        12GB together (~8-9GB total), so teacher server must be UP.
      - filter_dpo: CPU-side; teacher server can stay up (locale judge
        disabled), but we kill it here as a clean boundary.
      - train_dpo: gradient pass needs full VRAM — NO server allowed.
    """
    log("Stage 5: DPO retrain A1")

    # --- 5a: on-policy generation + filter (teacher server UP) ---
    kill_llama_server()  # clean slate
    if not start_llama_server(TEACHER_GGUF, TEACHER_CTX, TEACHER_MATCH,
                              LOGS / "v2_final_stage5_teacher_server.log"):
        log("Stage 5: teacher server failed to start — on_policy_gen will fail")
        # fail-forward: run anyway so filter_dpo/train_dpo can still proceed
    rc_gen = run_subprocess(
        [sys.executable, "scripts/run_training.py",
         "--training-config", "config/paper_v2/training_a1_full.yaml",
         "--stages", "on_policy_gen,filter_dpo"],
        LOGS / "v2_final_stage5a_on_policy_filter.log", "dpo_on_policy_filter",
    )
    kill_llama_server()  # free VRAM before gradient training

    # --- 5b: DPO training (no server) ---
    rc_train = run_subprocess(
        [sys.executable, "scripts/run_training.py",
         "--training-config", "config/paper_v2/training_a1_full.yaml",
         "--stages", "train_dpo"],
        LOGS / "v2_final_stage5b_train_dpo.log", "dpo_train",
    )
    return max(rc_gen, rc_train)


def _eval_test_sets_for(baseline: str) -> str:
    """Per-baseline test-set selection (v2 paper §5 scope, see EVAL_SESSION_v2_HANDOFF.md §3).

    Untrained base / instruct models (B1, B2) cannot emit \"[SESSION_END]\" at
    all — they were never trained on the sentinel and have no \"third strike\"
    concept. Their persistent_* probe results are degenerate (0% fire, 0% FP,
    0% off-position, 0% premature) and not informative. Skip all persistent_*
    probes for them; report \"n/a\" in §5.3 prose.

    Larger instructed baselines (B3, B4) DO have the sentinel instruction in
    their deployment prompt and may attempt to fire — keep persistent_probe,
    persistent_fp_probe, and persistent_premature_probe for them. Drop
    persistent_offposition_probe (was a v1 §5.3.2 finding; not central to
    v2's 2x2 design).

    Adapter baselines (paper_a*) always run \"all\" — their full results are
    load-bearing for both §5.3 (2x2 sentinel design) and §5.4 (pedagogy
    withholding rate). Already-done baselines hit the checkpoint and are
    skipped in the loop above regardless of this setting.
    """
    # B1, B2: no persistent training — skip ALL persistent_* probes
    if baseline in ("qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct"):
        return "tutor_scenario,locale_leakage,redirect_probe"
    # B3, B4 (and 9B teacher): drop only persistent_offposition_probe
    if baseline in ("qwen3_5_4b_instruct", "qwen3_5_9b_teacher"):
        return ("tutor_scenario,locale_leakage,redirect_probe,"
                "persistent_probe,persistent_fp_probe,persistent_premature_probe")
    return "all"


def stage6_eval_all_baselines() -> int:
    """Eval generations for all 9 baselines.

    HF baselines (8): run with no llama-server in VRAM.
    Teacher baseline (1, qwen3_5_9b_teacher): start teacher llama-server,
    run eval, kill server. Server swap happens inside this stage so a
    checkpoint resume doesn't require manual server management.

    Per-baseline test-set scope is set by _eval_test_sets_for() — B1/B2
    skip all persistent_* probes; B3/B4/teacher skip
    persistent_offposition_probe; adapter baselines run all 7.
    """
    log("Stage 6: eval generations for all 9 baselines")
    kill_llama_server()  # ensure clean VRAM for HF eval
    adapter_paths = {
        "paper_a2":     str(PAPER_V2 / "a1" / "sft"),   # a1's SFT step IS A2's adapter
        "paper_a3_sft": str(PAPER_V2 / "a3" / "sft"),
        "paper_a5_sft": str(PAPER_V2 / "a5" / "sft"),
        "paper_a6_sft": str(PAPER_V2 / "a6" / "sft"),
        "paper_a7_sft": str(PAPER_V2 / "a7" / "sft"),
    }
    overall_rc = 0

    # ---- Phase A: HF baselines (no server needed) ----
    hf_baselines = [b for b in ALL_EVAL_BASELINES if b != TEACHER_BASELINE]
    for baseline in hf_baselines:
        if is_done(f"stage6_eval_{baseline}"):
            log(f"  {baseline}: checkpoint exists; skip")
            continue
        ts = _eval_test_sets_for(baseline)
        log(f"  {baseline}: test sets = {ts}")
        cmd = [sys.executable, "scripts/run_paper_eval.py",
               "--baseline", baseline, "--test-set", ts,
               "--output-dir", str(EVAL_OUT)]
        ap = adapter_paths.get(baseline)
        if ap:
            if not Path(ap).exists():
                log(f"  {baseline}: v2 adapter {ap} missing; FALLING BACK to v1 default")
            else:
                cmd += ["--adapter-path", ap]
        rc = run_subprocess(
            cmd,
            LOGS / f"v2_final_stage6_eval_{baseline}.log",
            f"eval_{baseline}",
        )
        if rc == 0:
            mark_done(f"stage6_eval_{baseline}")
        else:
            overall_rc = max(overall_rc, rc)
            log(f"  {baseline}: rc={rc} (fail-forward)")

    # ---- Phase B: teacher baseline (needs llama-server with 9B GGUF) ----
    if not is_done(f"stage6_eval_{TEACHER_BASELINE}"):
        if not start_llama_server(TEACHER_GGUF, TEACHER_CTX, TEACHER_MATCH,
                                  LOGS / "v2_final_stage6_teacher_server.log"):
            log(f"  {TEACHER_BASELINE}: teacher server failed to start; skip")
            overall_rc = max(overall_rc, 1)
        else:
            ts = _eval_test_sets_for(TEACHER_BASELINE)
            log(f"  {TEACHER_BASELINE}: test sets = {ts}")
            rc = run_subprocess(
                [sys.executable, "scripts/run_paper_eval.py",
                 "--baseline", TEACHER_BASELINE, "--test-set", ts,
                 "--output-dir", str(EVAL_OUT)],
                LOGS / f"v2_final_stage6_eval_{TEACHER_BASELINE}.log",
                f"eval_{TEACHER_BASELINE}",
            )
            if rc == 0:
                mark_done(f"stage6_eval_{TEACHER_BASELINE}")
            else:
                overall_rc = max(overall_rc, rc)
                log(f"  {TEACHER_BASELINE}: rc={rc} (fail-forward)")
            kill_llama_server()  # free VRAM for scoring judges
    else:
        log(f"  {TEACHER_BASELINE}: checkpoint exists; skip")

    return overall_rc


def wait_for_data_review() -> None:
    """Block until the user creates REVIEW_APPROVED.txt.

    Writes WAITING_FOR_REVIEW.txt with instructions, then polls every 30s.
    To resume from a terminal:
      echo approved > outputs/paper_v2/REVIEW_APPROVED.txt
    """
    review_gate = PAPER_V2 / "WAITING_FOR_REVIEW.txt"
    approved    = PAPER_V2 / "REVIEW_APPROVED.txt"
    # If already approved (pre-created before relaunch), skip the gate immediately.
    if approved.exists():
        log("=== DATA REVIEW GATE: pre-approved, skipping ===")
        review_gate.unlink(missing_ok=True)
        return
    approved.unlink(missing_ok=True)  # clear any stale approval
    review_gate.write_text(
        "SFT data review gate — pipeline is paused. GPU is free.\n"
        "\n"
        "Step 1 — verify SFT data sentinel formats:\n"
        "  python scripts/verify_v2_sft_data.py\n"
        "\n"
        "Step 2 — run premature-fire probe on v1 A5 (positional shortcut diagnostic):\n"
        "  python scripts/run_paper_eval.py --baseline paper_a5_sft"
        " --test-set persistent_premature_probe\n"
        "  Then inspect: outputs/paper/eval/paper_a5_sft/persistent_premature_probe.jsonl\n"
        "  Key question: does viol2_p7 fire rate exceed viol2_p3/p5/p9?\n"
        "\n"
        "When satisfied with both, resume with:\n"
        "  echo approved > outputs/paper_v2/REVIEW_APPROVED.txt\n",
        encoding="utf-8",
    )
    log("=== DATA REVIEW GATE: pipeline paused before SFT (GPU is free) ===")
    log("  1. Run:  python scripts/verify_v2_sft_data.py")
    log("  2. Run:  python scripts/run_paper_eval.py --baseline paper_a5_sft "
        "--test-set persistent_premature_probe")
    log(f"  3. Then: echo approved > {approved}")
    while not approved.exists():
        time.sleep(30)
    review_gate.unlink(missing_ok=True)
    log("=== DATA REVIEW GATE: approved — proceeding to SFT ===")


def stage6c_score_mechanical() -> int:
    """Mechanical scoring across all baselines. The script iterates baselines
    internally. Env vars QWEN_TUTOR_EVAL_OUT and QWEN_TUTOR_MECH_SCORE_OUT
    redirect it to outputs/paper_v2/."""
    log("Stage 6c: mechanical scoring")
    rc = run_subprocess(
        [sys.executable, "scripts/score_redirect_mechanical.py"],
        LOGS / "v2_final_stage6c_score_mechanical.log", "score_mechanical",
    )
    return rc


def _run_sentinel_2x2() -> int:
    """§5.3 mechanical sentinel-firing 2x2 + premature-firing analysis.
    Covers ALL conditions including A5/A6/A7 (this is what they exist for).
    No judge needed — pure substring detection of the sentinel marker."""
    log("Stage 6 (sentinel 2x2): mechanical sentinel firing + premature")
    return run_subprocess(
        [sys.executable, "scripts/score_sentinel_2x2.py",
         "--eval-dir", str(EVAL_OUT),
         "--out", str(SCORE_OUT / "sentinel_2x2.json")],
        LOGS / "v2_final_stage6_sentinel_2x2.log", "sentinel_2x2",
    )


ALL_SCORING_BASELINES = (*ADAPTER_BASELINES,
                         "qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct",
                         "qwen3_5_4b_instruct", "qwen3_5_9b_teacher")


def _score_judged_for_metric(metric_filter: str, judges: tuple) -> int:
    """Run run_paper_score.py per (baseline, judge) for one judged metric.
    Loads each judge ONCE then iterates all baselines (saves server swaps).
    Env vars route the script to outputs/paper_v2/."""
    overall_rc = 0
    for judge_name, gguf, ctx, mid in judges:
        kill_llama_server()
        if not start_llama_server(gguf, ctx, mid,
                                   LOGS / f"v2_final_server_{judge_name}_{metric_filter}.log"):
            log(f"  {metric_filter}/{judge_name}: server didn't start; skip")
            overall_rc = max(overall_rc, 1)
            continue
        for baseline in ALL_SCORING_BASELINES:
            ck = f"stage6_score_{metric_filter}_{judge_name}_{baseline}"
            if is_done(ck):
                log(f"    {metric_filter}/{judge_name}/{baseline}: skip (done)")
                continue
            rc = run_subprocess(
                [sys.executable, "scripts/run_paper_score.py",
                 "--baseline", baseline,
                 "--metrics", "judged",
                 "--metric-filter", metric_filter,
                 "--judge", judge_name],
                LOGS / f"v2_final_stage6_score_{metric_filter}_{judge_name}.log",
                f"score_{metric_filter}_{judge_name}_{baseline}",
            )
            if rc == 0:
                mark_done(ck)
            else:
                overall_rc = max(overall_rc, rc)
    # Aggregate at the end (env vars route output to paper_v2 too)
    kill_llama_server()
    judge_names = ",".join(j[0] for j in judges)
    rc = run_subprocess(
        [sys.executable, "scripts/run_paper_score.py",
         "--baselines", ",".join(ALL_SCORING_BASELINES),
         "--judges", judge_names,
         "--aggregate"],
        LOGS / f"v2_final_stage6_aggregate_{metric_filter}.log",
        f"aggregate_{metric_filter}",
    )
    if rc != 0:
        overall_rc = max(overall_rc, rc)
    return overall_rc


# NOTE: stage6d_score_per_axis_f1 and stage6e_score_pairwise have been
# RETIRED in v2 per paper/EVAL_SESSION_v2_HANDOFF.md §3.5.
#
# - F1 (was stage6d) conflates response-shape classification with quality:
#   saturates near 0.75 on persona/role_swap regardless of training and
#   floors near 0.30 on locale/language/topic at n=13. Replaced by
#   group-appropriate metrics (mechanical for locale/lang/topic;
#   withholding-rate for pedagogy; v1 pairwise reused for persona/role_swap).
#
# - Pairwise (was stage6e) is unchanged across v2 conditions vs v1 (same
#   conditions on persona/role_swap probes). Reuse the v1 §5.4.5 result
#   (49% / 38% parity); no v2 re-run.
#
# The retired functions are kept in source but no longer called from main().


def stage6f_score_naturalness() -> int:
    log("Stage 6f: naturalness scoring (all 3 judges)")
    return _score_judged_for_metric("naturalness", JUDGES)


def stage6g_score_withholding_rate() -> int:
    """v2 §5.4 contribution-1 load-bearing experiment: pedagogy withholding
    rate under matched prompt. For each pedagogy_redirect record, a judge
    binarily classifies whether the tutor withheld the direct answer and
    scaffolded vs gave the answer directly. Rate per condition.

    Scored ONLY on the necessity-contrast conditions (per user directive
    2026-06-30): A1 (=paper_a2, keeps pedagogy stream), A3 (drops it),
    and B1-B4 (instruct-only). A5/A6/A7 are excluded from all judge evals
    — they exist solely for the §5.3 sentinel 2x2 (mechanical). ~21
    pedagogy records × 6 conditions ≈ 126 binary calls, ~8-10 min.
    Cross-judge agreement on this binary classification is not load-bearing
    for the §5.4 claim; the contrast is condition-vs-condition under a
    single judge, which removes per-judge bias as a confound.
    """
    log("Stage 6g: pedagogy withholding rate (Llama-3.1 8B judge, single pass)")
    judge_name, gguf, ctx, mid = next(j for j in JUDGES if j[0] == "llama31_8b_judge")
    # §5.4 necessity contrast only — exclude A5/A6/A7 (judge-free conditions).
    withholding_baselines = ",".join((
        "paper_a2", "paper_a3_sft",
        "qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct",
        "qwen3_5_4b_instruct", "qwen3_5_9b_teacher",
    ))
    kill_llama_server()
    if not start_llama_server(gguf, ctx, mid,
                               LOGS / "v2_final_server_withholding_judge.log"):
        log("  withholding: judge server didn't start")
        return 1
    rc = run_subprocess(
        [sys.executable, "scripts/score_withholding_rate.py",
         "--judge", judge_name,
         "--eval-dir", str(EVAL_OUT),
         "--baselines", withholding_baselines,
         "--out", str(SCORE_OUT / "pedagogy_withholding_rate.json")],
        LOGS / "v2_final_stage6g_withholding.log",
        "withholding_rate",
    )
    kill_llama_server()
    return rc


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run_stage(stage_name: str, fn, fail_stop: bool = False) -> int:
    if is_done(stage_name):
        log(f"=== Stage {stage_name} already done; skipping ===")
        return 0
    log(f"=== STAGE START: {stage_name} ===")
    try:
        rc = fn()
    except Exception as e:
        log(f"=== STAGE EXCEPTION {stage_name}: {type(e).__name__}: {e} ===")
        rc = 1
    if rc == 0:
        mark_done(stage_name)
        log(f"=== STAGE COMPLETE: {stage_name} ===")
    else:
        log(f"=== STAGE FAILED: {stage_name} rc={rc} (fail_stop={fail_stop}) ===")
        if fail_stop:
            log("FAIL-STOP semantics: halting pipeline.")
            sys.exit(rc)
    return rc


def main() -> int:
    PAPER_V2.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    SCORE_OUT.mkdir(parents=True, exist_ok=True)

    log(f"V2 FINAL ORCHESTRATOR pid={os.getpid()} starting (TS={TIMESTAMP})")

    # ---- fail-stop chain (data prep + smoke gate) ----
    run_stage("stage1_backup_and_delete",   stage1_backup_and_delete,   fail_stop=True)
    run_stage("stage2_full_regen",          stage2_full_regen,          fail_stop=True)
    run_stage("stage2b_setup_ablation_data", stage2b_setup_ablation_data, fail_stop=True)
    run_stage("stage3_smoke_quality_check", stage3_smoke_quality_check, fail_stop=True)

    # ---- user data review gate (before first SFT) ----
    # Only blocks if no SFT condition has started yet (skip on resume).
    if not any(is_done(f"stage4_sft_{c}") for c in SFT_CONDITIONS):
        wait_for_data_review()

    # ---- fail-forward training ----
    # v2: SFT only (A2, A3, A5, A6). No DPO — see SFT_CONDITIONS note.
    for i, cond in enumerate(SFT_CONDITIONS):
        stage_name = f"stage4_sft_{cond}"
        was_done_before = is_done(stage_name)
        run_stage(stage_name, lambda c=cond: stage4_sft_retrain(c))
        # GPU cooldown after each SFT stage that actually ran (prevents thermal
        # throttling — extended continuous load on the RTX 3060 degrades step
        # time by ~2x). Skip if this stage was already-done (no work happened)
        # or if it was the last SFT condition.
        if not was_done_before and i < len(SFT_CONDITIONS) - 1:
            log("Cooling GPU 5 min before next SFT condition")
            time.sleep(300)

    # ---- fail-forward eval + scoring ----
    # v2 scoring per paper/EVAL_SESSION_v2_HANDOFF.md §3.5 (revised 2026-06-30):
    #   - mechanical (sentinel firing 2x2 + premature; locale leakage;
    #     lang/topic rates) — covers ALL conditions incl. A5/A6/A7, since
    #     the sentinel score is what A5/A6/A7 exist for.
    #   - withholding rate (single-judge binary; §5.4 contribution-1) —
    #     scored on A1(=paper_a2), A3, B1-B4 only (the necessity contrast).
    #
    # RETIRED / SKIPPED in v2:
    #   - F1 (was stage6d): conflates shape-classification with quality.
    #   - pairwise (was stage6e): reuse v1 §5.4.5 parity result.
    #   - naturalness (was stage6f): v1 naturalness scores are stable and
    #     confident; A1-A7 share base model + recipe so naturalness does
    #     not change materially across v2 conditions. Reuse v1 §5.2 N3
    #     numbers. (Re-enable stage6f_score_naturalness if a reviewer
    #     specifically requests fresh naturalness on v2 generations.)
    run_stage("stage6_eval_all_baselines", stage6_eval_all_baselines)
    run_stage("stage6c_score_mechanical", stage6c_score_mechanical)
    run_stage("stage6_sentinel_2x2", lambda: _run_sentinel_2x2())
    run_stage("stage6g_score_withholding_rate", stage6g_score_withholding_rate)

    kill_llama_server()
    log("=== V2 FINAL PIPELINE COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
