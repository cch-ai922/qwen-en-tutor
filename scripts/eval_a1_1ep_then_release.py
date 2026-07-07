"""eval_a1_1ep_then_release.py — run a1_1ep's mechanical eval the moment its
training finishes, BEFORE the multi-seed queue grabs the GPU.

a1_1ep is the matched-budget decorrelation diagnostic (§5.3.1). Its eval is
judge-free (sentinel firing on persistent_probe + persistent_premature_probe),
so it only needs ~30-45 min of HF inference. The multi-seed queue
(run_multiseed_queue.py) is GPU-gated and will start a1_s123 as soon as the
GPU frees; this watcher detects a1_1ep's adapter the instant it is written and
launches the eval fast enough (20 s poll) to re-occupy the GPU before the
queue's slower gate (2 free checks at 60 s) commits — so the eval runs first.

Detached launch:
  PYTHONUTF8=1 nohup python scripts/eval_a1_1ep_then_release.py >> logs/eval_a1_1ep.log 2>&1 &
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "outputs" / "paper_v2" / "a1_1ep" / "sft" / "adapter_model.safetensors"
OUT_DIR = ROOT / "outputs" / "paper_v2" / "eval_a1_1ep"
TEST_SETS = "persistent_probe,persistent_premature_probe"


def _log(m: str) -> None:
    print(f"[a1_1ep-eval {time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def main() -> int:
    _log(f"watching for a1_1ep adapter at {ADAPTER}")
    while not ADAPTER.exists():
        time.sleep(20)
    _log("a1_1ep adapter present (training done); launching eval immediately")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    # paper_a2 baseline spec (same base/recipe as A1) with the 1-epoch adapter.
    rc = subprocess.run(
        [sys.executable, "scripts/run_paper_eval.py",
         "--baseline", "paper_a2",
         "--adapter-path", "outputs/paper_v2/a1_1ep/sft",
         "--test-set", TEST_SETS,
         "--output-dir", str(OUT_DIR)],
        cwd=str(ROOT), env=env,
    ).returncode
    _log(f"a1_1ep eval rc={rc}; outputs in {OUT_DIR}")
    _log("GPU released — the multi-seed queue may now proceed.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
