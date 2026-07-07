"""run_multiseed_queue.py — sequential multi-seed training queue.

Waits for the GPU to be free (the in-flight a1_1ep run, or any other python
training, to exit), then trains the four Tier-1 reseed configs ONE AT A TIME
(never two on the 12 GB GPU). Each run is checkpointed every 100 steps and
resumes if interrupted (run_training.py / SFTTrainer resume_from_checkpoint).

Idempotent: skips a config whose adapter (adapter_model.safetensors) already
exists in its output_dir, so re-launching after a crash continues the queue.

Run (detached, survives the session):
  PYTHONUTF8=1 nohup python scripts/run_multiseed_queue.py >> logs/multiseed_queue.log 2>&1 &
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

QUEUE = [
    # Un-trimmed A5/A6/A7 (1 epoch, seed 42) — completes the trim analysis
    # (§5.3.1): does un-trimming reduce premature firing in every design cell,
    # as it did for A1 (0.783 trimmed -> 0.208 un-trimmed)? Trains on the now
    # un-trimmed data/sft_filtered_a{5,6,7} into outputs/paper_v2/a{5,6,7}/sft
    # (the prior TRIMMED adapters were cleared; their eval results are kept in
    # outputs/paper_v2/eval/paper_a{5,6,7}_sft/). Quick (~4h each), so first.
    ("config/paper_v2/training_a5_fixed_turn_7.yaml",     "outputs/paper_v2/a5/sft"),
    ("config/paper_v2/training_a6_generic_sentinel.yaml", "outputs/paper_v2/a6/sft"),
    ("config/paper_v2/training_a7_generic_sentinel.yaml", "outputs/paper_v2/a7/sft"),
    # Multi-seed A1/A3 (seeds 123, 7) — §4.10 robustness.
    ("config/paper_v2/training_a1_s123.yaml", "outputs/paper_v2/a1_s123/sft"),
    ("config/paper_v2/training_a1_s7.yaml",   "outputs/paper_v2/a1_s7/sft"),
    ("config/paper_v2/training_a3_s123.yaml", "outputs/paper_v2/a3_s123/sft"),
    ("config/paper_v2/training_a3_s7.yaml",   "outputs/paper_v2/a3_s7/sft"),
]


def _log(msg: str) -> None:
    print(f"[multiseed-queue {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _gpu_busy(threshold_mib: int = 2000) -> bool:
    """True if the GPU has a large allocation (a training run in flight).
    Used only for the INITIAL wait on the in-flight a1_1ep run — the queue's
    own runs are launched blocking/sequentially, so they never overlap."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        used = int(r.stdout.strip().split("\n")[0])
        return used > threshold_mib
    except Exception:
        return False  # if nvidia-smi unavailable, don't block


def _adapter_done(out_dir: str) -> bool:
    p = ROOT / out_dir / "adapter_model.safetensors"
    return p.exists()


def main() -> int:
    _log("queue start; waiting for GPU to free (in-flight a1_1ep to exit)")
    # Require the GPU to read free for two consecutive checks (avoid racing
    # the moment a1_1ep releases VRAM before its process fully exits).
    waited = 0
    free_streak = 0
    while free_streak < 2:
        if _gpu_busy():
            free_streak = 0
        else:
            free_streak += 1
        time.sleep(60)
        waited += 60
        if waited % 1800 == 0:
            _log(f"still waiting for GPU after {waited//60} min")
    _log("GPU free; starting queue")

    for cfg, out_dir in QUEUE:
        if _adapter_done(out_dir):
            _log(f"SKIP {cfg} — adapter already exists at {out_dir}")
            continue
        _log(f"START {cfg} -> {out_dir}")
        env_prefix = {"PYTHONUTF8": "1"}
        import os
        env = os.environ.copy(); env.update(env_prefix)
        rc = subprocess.run(
            [sys.executable, "scripts/run_training.py",
             "--training-config", cfg, "--stages", "train_sft"],
            cwd=str(ROOT), env=env,
        ).returncode
        if rc == 0 and _adapter_done(out_dir):
            _log(f"DONE  {cfg} (rc=0, adapter written)")
        else:
            _log(f"FAIL  {cfg} (rc={rc}, adapter_present={_adapter_done(out_dir)}) "
                 f"— continuing queue (fail-forward)")
        # GPU cooldown between back-to-back SFTs: the original A1/A3 runs
        # thermal-throttled (~2-3x slowdown) under sustained load. A ~5-min
        # idle gap lets the 3060 cool from ~70C+ back toward idle so the next
        # run starts with full thermal headroom.
        _log("cooling GPU ~5 min before next run")
        time.sleep(300)

    _log("=== multi-seed queue complete ===")
    _log("next: eval each adapter on redirect_probe + persistent_probe + "
         "locale_leakage, then judge withholding + pairwise, then mean±SD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
