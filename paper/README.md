# Paper reproduction guide

End-to-end runbook for reproducing the paper from scratch on a single
RTX 3060 12 GB machine. Every script lives in `qwen-en-tutor/scripts/`
and reads its config from `qwen-en-tutor/config/`.

For the LaTeX paper build itself (the `.tex` → `.pdf` step), see
[latex/README.md](latex/README.md). This document covers everything
upstream of that: data generation, training, evaluation, scoring, and
the eventual PDF rebuild.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [TL;DR sequence](#2-tldr-sequence)
3. [Stage 1 — Data generation](#3-stage-1--data-generation)
4. [Stage 2 — Ablation data fan-out](#4-stage-2--ablation-data-fan-out)
5. [Stage 3 — Train each condition](#5-stage-3--train-each-condition)
6. [Stage 4 — Freeze evaluation sets](#6-stage-4--freeze-evaluation-sets)
7. [Stage 5 — Generate baseline responses](#7-stage-5--generate-baseline-responses)
8. [Stage 6 — Score the generations](#8-stage-6--score-the-generations)
9. [Stage 7 — (Optional) GGUF export for deployment](#9-stage-7--optional-gguf-export-for-deployment)
10. [Stage 8 — Rebuild the paper PDF](#10-stage-8--rebuild-the-paper-pdf)
11. [What to skip if you only want fresh numbers](#11-what-to-skip-if-you-only-want-fresh-numbers)

---

## 1. Prerequisites

| Requirement | How to satisfy |
|---|---|
| GPU + driver | NVIDIA RTX 3060 12 GB (or larger). CUDA 12.x. |
| Python venv | Set up per main [README.md](../README.md#2-one-time-offline-bundle-on-a-connected-machine) |
| Teacher model GGUF | `vendor/models/GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf` served via `llama-server` on `127.0.0.1:8080` |
| Student base | `vendor/models/Qwen_3.5_0.8B-Base` (HuggingFace format, for training) |
| Baseline checkpoints | `vendor/models/Qwen_3.5_0.8B`, `Qwen_3.5_4B` (for B2 / B3 zero-shot eval) |
| Judge ensemble GGUFs (§4.7) | `vendor/models/GGUF/prometheus-7b-v2.0.Q4_K_M.gguf`, `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`, `gemma-2-9b-it-Q5_K_M.gguf`. Cross-family ensemble; no Qwen judges (eliminates teacher self-preference bias). |
| Pandoc + MiKTeX | For the LaTeX paper build; see [latex/README.md](latex/README.md). Installers vendored under [../vendor/installers/](../vendor/installers/) |

All commands below assume you are in the `qwen-en-tutor/` directory
with the project venv activated:

```powershell
cd qwen-en-tutor
.\.venv\Scripts\Activate.ps1
```

**Windows-specific environment**: every training and generation launch
must set `PYTHONUTF8=1` (TRL ≥ 1.5 reads `deepseekv3.jinja` without an
encoding kwarg and dies on Windows `cp1252`):

```powershell
$env:PYTHONUTF8 = "1"
```

**Teacher launch command (canonical)** — from `vendor/llama_cpp/`:

```powershell
llama-server.exe -m ..\models\GGUF\Qwen3.5-9B-UD-Q4_K_XL.gguf `
  -ngl 80 -c 32768 --host 0.0.0.0 --port 8080 --log-verbose --log-prefix
```

---

## 2. TL;DR sequence

| # | Script | Why | Wall-clock |
|---|---|---|---|
| 1 | `run_generation.py` | Build SFT + DPO + eval corpora from teacher | ~24–48 h cumulative |
| 1b | `regen_redirect_streams.py` | Re-emit single-shot redirect streams so `metadata.generation.violation_turn_idx` is present (used by Stage 4 to ground-truth the violation turn instead of relying on a pivot heuristic) | ~3–4 h |
| 2 | `setup_paper_ablation_data.py` | Materialize A3/A4/A5 subset dirs (hardlinks) | seconds |
| 2b | `regen_persistent_a5.py` | A5 only: regen persistent streams with `QWEN_TUTOR_PERSISTENT_FORCED_VARIANT=1` (V2-only, sentinel fixed at turn 7) | ~2 h |
| 3 | `run_training.py` (× 4) + `run_sentinel_chain.py` (× 3) | Train A1 / A3 / A4 / A5 adapters | ~5–8 h per condition |
| 4 | `build_eval_sets.py` | Freeze **six** held-out test sets (Tutor-Scenario, Redirect-Probe, Persistent-Probe, **Persistent-FP-Probe**, **Persistent-OffPosition-Probe**, Locale-Leakage) | minutes |
| 5 | `run_paper_eval.py` (× 9 baselines: A1, A2, A3, A4, A5, B1, B2, B3, B4) | Generate responses on test sets | ~6–10 h total |
| 6 | `run_paper_score.py` (mechanical + judged) | Score with cross-family ensemble (Prometheus + Llama-3.1 + Gemma-2) and aggregate | ~6–10 h (judge swaps dominate) |
| 7 | `run_merge.py` + `run_gguf_export.py` | (Optional) deploy artifact | minutes |
| 8 | `scripts/build_paper_latex.ps1` | Rebuild PDF with real numbers | minutes |

Worst-case total from scratch: about a week of wall-clock on RTX 3060,
mostly idle waiting for GPU work. With overnight scheduling, four
calendar days is realistic.

---

## 3. Stage 1 — Data generation

Generates seeds + 12 SFT streams + register-DPO + eval examples +
filter cascade. Edits to `config/generation.yaml` control which locale,
which CEFR levels, and target counts per stream.

The teacher must be served before running this stage. The teacher and
trainer cannot co-reside on 12 GB VRAM; the orchestrator stops the
server for training and restarts it for generation phases.

```powershell
# Start the 9B teacher in a separate terminal (blocks GPU during generation):
# llama-server -m vendor/models/Qwen3.5-9B-UD-Q4_K_XL.gguf --port 8080 -ngl 80 -c 32768

# Full generation (8 stages, fully resumable):
python scripts/run_generation.py

# Or run partial stages — see ALL_STAGES in scripts/run_generation.py:87:
python scripts/run_generation.py --stages seeds,dedup_seeds
python scripts/run_generation.py --stages sft,redirect,locale_redirect,pedagogy_redirect,language_redirect,persona_redirect,topic_redirect,role_swap_redirect
python scripts/run_generation.py --stages persistent_off_topic,persistent_language_violation,persistent_persona_break,persistent_role_swap
python scripts/run_generation.py --stages register,eval
python scripts/run_generation.py --stages filter_sft,filter_eval,filter_dpo
python scripts/run_generation.py --stages sft_topup,persistent_topup    # yield-aware top-up
```

Outputs land in:

- `data/sft_raw/`, `data/sft_filtered/` — 12-stream SFT corpora
- `data/dpo_raw/`, `data/dpo_filtered/` — register DPO pairs
- `data/eval_raw/`, `data/eval_filtered/` — `<think>`-mode evaluator examples

Every stage skips records already in its output JSONL, so a killed run
loses no work — just rerun the same command.

---

## 4. Stage 2 — Ablation data fan-out

Materializes per-condition subset directories from `data/sft_filtered/`.
Uses hardlinks where possible (zero disk cost).

```powershell
python scripts/setup_paper_ablation_data.py --conditions a3,a4,a5
# -> data/sft_filtered_a3/   (normal + generic redirect + persistent only)
# -> data/sft_filtered_a4/   (normal + 7 single-shot redirects, no persistent)
# -> data/sft_filtered_a5/   (all 12 streams, but persistent come from
#                             data/sft_filtered_a5_persistent/ — V2-only)
```

A1 and A2 use `data/sft_filtered/` directly — no setup needed.

**A5 pre-step (V2-only persistent regen)**. A5 keeps all 12 streams but
forces every persistent dialogue to use the V2 ("medium") structure
with sentinel at turn 7 — this isolates the trigger-position
decorrelation contribution (§3.4). The setup script reads persistent
streams from `data/sft_filtered_a5_persistent/`, which is built by:

```powershell
# Teacher must be up. PYTHONUTF8=1 and the forced-variant env var are
# set at the top of regen_persistent_a5.py automatically.
python scripts/regen_persistent_a5.py
# -> data/sft_raw_a5_persistent/<axis>_<level>.jsonl  (V2 only)
# -> data/sft_filtered_a5_persistent/<axis>_<level>_passed.jsonl
```

Then run `setup_paper_ablation_data.py --conditions a5` to hardlink
everything into `data/sft_filtered_a5/`.

The script is idempotent: re-run it any time `data/sft_filtered/`
changes (e.g. after a top-up round).

---

## 5. Stage 3 — Train each condition

Per-condition configs are in `config/paper/training_<condition>.yaml`.
The driver reads `config/training.yaml` by default; for ablations,
pass the paper config explicitly with `--config`.

**Critical:** stop the `llama-server` teacher process before each
training run. Teacher (~4 GB) + trainer (~7.7 GB) = ~11.7 GB, which
exceeds 12 GB device capacity.

### Condition A1 — full system (SFT + DPO + sentinel chain)

```powershell
# SFT
python scripts/run_training.py --config config/paper/training_a1_full.yaml --stages train_sft

# Sentinel-aware DPO pool generation (offline; no teacher needed)
python scripts/run_sentinel_chain.py --condition a1 `
    --sft-adapter outputs/paper/a1/sft `
    --sft-filtered-dir data/sft_filtered

# Restart teacher, then run on-policy DPO generation + filter
# llama-server -m vendor/models/Qwen3.5-9B-UD-Q4_K_XL.gguf --port 8080 -ngl 80 -c 32768
python scripts/run_training.py --config config/paper/training_a1_full.yaml --stages on_policy_gen,filter_dpo

# Stop teacher, then DPO training + final eval
python scripts/run_training.py --config config/paper/training_a1_full.yaml --stages train_dpo,eval_final
```

### Condition A3 — no specialized redirects

```powershell
python scripts/run_training.py --config config/paper/training_a3_no_specialized.yaml --stages train_sft

python scripts/run_sentinel_chain.py --condition a3 `
    --sft-adapter outputs/paper/a3/sft `
    --sft-filtered-dir data/sft_filtered_a3

python scripts/run_training.py --config config/paper/training_a3_no_specialized.yaml --stages on_policy_gen,filter_dpo,train_dpo
```

### Condition A4 — no persistent streams

No sentinel chain needed because the persistent streams are dropped.

```powershell
python scripts/run_training.py --config config/paper/training_a4_no_persistent.yaml --stages train_sft,on_policy_gen,filter_dpo,train_dpo
```

### Condition A5 — fixed-turn-7 persistent (decorrelation ablation)

A5 trains on all 12 streams but the persistent streams come from
the V2-only regen (`data/sft_filtered_a5/` per Stage 2). Same training
recipe as A1 — the only difference is the input data. **A1 vs A5 is
the direct test of the trigger-position decorrelation contribution
(§3.4) — A1 vs A4 is the weaker "does persistence training help at all?"
contrast.**

```powershell
python scripts/run_training.py --config config/paper/training_a5_fixed_turn_7.yaml --stages train_sft

python scripts/run_sentinel_chain.py --condition a5 `
    --sft-adapter outputs/paper/a5/sft `
    --sft-filtered-dir data/sft_filtered_a5

python scripts/run_training.py --config config/paper/training_a5_fixed_turn_7.yaml --stages on_policy_gen,filter_dpo,train_dpo
```

### Condition A2 — SFT-only

A2 reuses A1's SFT adapter; no separate training step is needed.
Evaluation in Stage 5 points at `outputs/paper/a1/sft` directly via
the `paper_a2` baseline alias.

### Timings

- `train_sft`: ~3–5 h on RTX 3060 for the ~3 000-record corpus
- `train_dpo`: ~1–2 h
- `on_policy_gen`: ~30–60 min (with teacher running)
- `eval_final`: ~30 min

Plan one overnight per condition.

---

## 6. Stage 4 — Freeze evaluation sets

Builds the six held-out JSONL files every baseline is scored on. The
sentinel-related probe set was expanded from one to three to make
precision and false-positive rate well-defined (§4.8).

```powershell
python scripts/build_eval_sets.py
# -> eval_sets/tutor_scenario.jsonl                (N=224)
# -> eval_sets/redirect_probe.jsonl                (N=143; uses
#                                                   metadata.violation_turn_idx
#                                                   from Stage 1b regen)
# -> eval_sets/persistent_probe.jsonl              (positives, target ~140)
# -> eval_sets/persistent_fp_probe.jsonl           (negatives at trained
#                                                   sentinel positions
#                                                   {5,7,9,11}, target ~240)
# -> eval_sets/persistent_offposition_probe.jsonl  (positives at off-grid
#                                                   positions {13,15}, target ~60)
# -> eval_sets/locale_leakage.jsonl                (N=224, scenario-overlapping)
# -> eval_sets/_split_manifest.json
```

The split is hash-deterministic on `seed_id` (`sha256(seed_id)[:8] % 100 < 20`),
so this is a once-per-corpus operation. Re-run only if the seed pool
changes, OR after Stage 1b regen lands new `violation_turn_idx`
metadata in `data/sft_raw/`.

Each probe record's ID now includes a variant tag (`_v<N>`) so that
SFT records sharing a seed but differing in variant don't collapse to
the same probe ID. Records may carry `expected.detection_reason` ∈
{`metadata`, `pivot`, `fallback`} indicating how the redirect-probe
violation turn was identified.

---

## 7. Stage 5 — Generate baseline responses

Each baseline takes one process. Run sequentially — they swap-load
VRAM on RTX 3060.

```powershell
# Zero-shot baselines (inference only, no training required)
python scripts/run_paper_eval.py --baseline qwen3_5_0_8b_base --test-set all
python scripts/run_paper_eval.py --baseline qwen3_5_0_8b_instruct --test-set all
python scripts/run_paper_eval.py --baseline qwen3_5_4b_instruct --test-set all
python scripts/run_paper_eval.py --baseline qwen3_5_9b_teacher --test-set all   # needs teacher served

# Trained conditions
python scripts/run_paper_eval.py --baseline paper_a1 --test-set all
python scripts/run_paper_eval.py --baseline paper_a2 --test-set all
python scripts/run_paper_eval.py --baseline paper_a3 --test-set all
python scripts/run_paper_eval.py --baseline paper_a4 --test-set all
python scripts/run_paper_eval.py --baseline paper_a5 --test-set all
```

Outputs land in `eval_results/<baseline>/<test_set>.jsonl`. Re-runnable:
already-generated records are skipped, so a crashed run resumes cleanly.

Per-baseline wall-clock:

- 0.8B / instruct baselines: 30–60 min
- 4B-instruct: 60–90 min
- 9B teacher (API calls): 60–90 min

For all 8: budget a full day.

To list all available baselines: `python scripts/run_paper_eval.py --baseline list`.

---

## 8. Stage 6 — Score the generations

Two scoring passes per baseline.

### Mechanical pass (cheap, no judge)

Computes the full sentinel suite (recall, FP-rate, precision, F1,
OffPosition recall, per-position fire-rate breakdowns) over
Persistent-Probe ∪ Persistent-FP-Probe ∪ Persistent-OffPosition-Probe,
plus locale-leakage rate via gazetteer regex.

```powershell
foreach ($b in 'qwen3_5_0_8b_base','qwen3_5_0_8b_instruct','qwen3_5_4b_instruct','qwen3_5_9b_teacher','paper_a1','paper_a2','paper_a3','paper_a4','paper_a5') {
    python scripts/run_paper_score.py --baseline $b --metrics mechanical
}
```

### Judged pass (slow, needs judge ensemble)

Naturalness, CEFR adherence, and redirect-axis F1 are scored by a
**cross-family** 3-judge ensemble — Prometheus 7B (rubric protocol),
Llama-3.1-8B-Instruct (instruct JSON), and Gemma-2-9B-it (instruct
JSON). The ensemble is deliberately Qwen-free to eliminate
self-preference bias against the Qwen-family teacher (§4.7).

Each judge has its own GGUF. Only one judge fits in VRAM at a time,
so swap models between rounds. Use `--limit 30` for a quick first-pass
smoke before committing to the full ~600-record judging per judge.

```powershell
# Round 1: Prometheus. Stop existing llama-server, then:
# llama-server -m vendor/models/GGUF/prometheus-7b-v2.0.Q4_K_M.gguf -ngl 80 -c 32768 --host 0.0.0.0 --port 8080
foreach ($b in 'qwen3_5_0_8b_base','qwen3_5_0_8b_instruct','qwen3_5_4b_instruct','qwen3_5_9b_teacher','paper_a1','paper_a2','paper_a3','paper_a4','paper_a5') {
    python scripts/run_paper_score.py --baseline $b --metrics judged --judge prometheus_7b_judge
}

# Round 2: Llama-3.1. Stop llama-server, then:
# llama-server -m vendor/models/GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -ngl 80 -c 32768 --host 0.0.0.0 --port 8080
foreach ($b in '<same list>') {
    python scripts/run_paper_score.py --baseline $b --metrics judged --judge llama31_8b_judge
}

# Round 3: Gemma-2. Stop llama-server, then:
# llama-server -m vendor/models/GGUF/gemma-2-9b-it-Q5_K_M.gguf -ngl 80 -c 32768 --host 0.0.0.0 --port 8080
foreach ($b in '<same list>') {
    python scripts/run_paper_score.py --baseline $b --metrics judged --judge gemma2_9b_judge
}
```

The scorer probes `/v1/models` before each round and warns if the
loaded GGUF doesn't match the judge name — catches accidental
model-loaded-vs-judge-asked mismatches.

### Aggregate

Combines per-judge scores into the ensemble median + Krippendorff's
alpha used in Table 7.

```powershell
python scripts/run_paper_score.py --aggregate `
    --baselines paper_a1,paper_a2,paper_a3,paper_a4,paper_a5,qwen3_5_0_8b_base,qwen3_5_0_8b_instruct,qwen3_5_4b_instruct,qwen3_5_9b_teacher
```

Output: `outputs/paper/score/aggregated/<baseline>.json` — the source
for Table 7 numbers in
[sections/05_results_scaffold.md](sections/05_results_scaffold.md).

---

## 9. Stage 7 — (Optional) GGUF export for deployment

If you want to actually deploy A1 via `llama-server`:

```powershell
python scripts/run_merge.py --adapter outputs/paper/a1/dpo --merged-dir outputs/paper/a1/merged_dpo
python scripts/run_gguf_export.py --merged-dir outputs/paper/a1/merged_dpo --quantize Q4_K_M
# -> vendor/models/paper_a1.Q4_K_M.gguf
```

Then start the deployment server (see main [README.md §13](../README.md#13-deploying-the-trained-tutor)).

This stage is not required for the paper itself.

---

## 10. Stage 8 — Rebuild the paper PDF

Once `scores/aggregate.json` is populated, replace placeholder values
in [sections/05_results_scaffold.md](sections/05_results_scaffold.md)
with the real ones, then rebuild.

### Fast preview (Chrome headless, no LaTeX)

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --headless=new --disable-gpu --no-pdf-header-footer `
  --print-to-pdf="paper\build\paper.pdf" `
  "file:///$((Resolve-Path paper\build\paper.html).Path -replace '\\','/')"
```

Note: the HTML version at `paper/build/paper.html` is currently
hand-written; tables and the abstract are LaTeX-ish inline HTML. Edits
to `paper/sections/*.md` do not automatically flow through to it.

### Submission-grade LaTeX (pandoc + MiKTeX)

This is the canonical path for venue submission. See
[latex/README.md](latex/README.md) for the install + style-file setup.

```powershell
pwsh ./scripts/build_paper_latex.ps1 -Open
# -> paper/latex/generated/main.pdf
```

The LaTeX path:

1. Runs `pandoc` on each `sections/*.md` to produce `.tex`
2. Stages everything (including `acl.sty`, `acl_natbib.bst`,
   `references.bib`) into `paper/latex/generated/`
3. Runs `pdflatex → bibtex → pdflatex → pdflatex` (the standard
   four-pass dance for cross-refs and bibliography)
4. Opens the resulting `paper/latex/generated/main.pdf`

---

## 11. What to skip if you only want fresh numbers

If the data and adapters from a previous run are already on disk, you
only need Stages 4 → 6:

```powershell
$env:PYTHONUTF8 = "1"
python scripts/build_eval_sets.py

# 9 baselines × 6 test sets generation; teacher GPU resident for B4 only
foreach ($b in 'qwen3_5_0_8b_base','qwen3_5_0_8b_instruct','qwen3_5_4b_instruct','qwen3_5_9b_teacher','paper_a1','paper_a2','paper_a3','paper_a4','paper_a5') {
    python scripts/run_paper_eval.py --baseline $b --test-set all
    python scripts/run_paper_score.py --baseline $b --metrics mechanical
}

# Judged scoring: one llama-server-swap per judge round (see Stage 6).
# Then for each round (after starting llama-server with the right GGUF):
foreach ($b in '<list above>') {
    python scripts/run_paper_score.py --baseline $b --metrics judged --judge <judge_name>
}

python scripts/run_paper_score.py --aggregate `
    --baselines paper_a1,paper_a2,paper_a3,paper_a4,paper_a5,qwen3_5_0_8b_base,qwen3_5_0_8b_instruct,qwen3_5_4b_instruct,qwen3_5_9b_teacher
```

Then rebuild the PDF per Stage 8. Wall-clock for the full B1–B4 +
A1–A5 fill-in: about 16–24 hours of mostly overnight work (the extra
condition + the expanded eval set add ~25% vs. the old 4-condition
single-probe matrix).
