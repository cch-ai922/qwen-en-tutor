# Reproducing Paper 1

**Paper 1** — *What Must Be Trained and What Can Be Prompted: A Per-Capability Study
of Tutor Redirect Behavior*
(full version in `paper/`, 8-page condensation in `paper_workshop/`)

Read [`README.md`](README.md) first for the shared pipeline, release policy, and
environment gotchas. This guide maps each headline claim to the exact command,
config, and expected number.

---

## Conditions used by Paper 1

| Cond. | Meaning | Data dir | Training config |
|-------|---------|----------|-----------------|
| **A1** | Full 12-stream SFT + DPO (the full system) | `data/sft_filtered/` | `config/paper/training_a1_full.yaml` |
| **A3** | Drop 6 specialized redirect streams (generic-SFT baseline) | `data/sft_filtered_a3/` | `config/paper/training_a3_no_specialized.yaml` |
| **A4** | Drop 4 persistent streams | `data/sft_filtered_a4/` | `config/paper/training_a4_no_persistent.yaml` |
| **A5** | Persistence fixed at turn 7 (position decorrelation) | `data/sft_filtered_a5/` | `config/paper/training_a5_fixed_turn_7.yaml` |
| **B1–B4** | Prompt-only baselines (no training): 0.8B base, 0.8B instruct, 4B instruct, 9B teacher | — | matched deployment system prompt |
| Cross-family | Llama-3.2-1B-**Instruct** A1/A3 (the reported cross-family result) | as above | `config/paper/training_llamaINST_a{1,3}_*.yaml` |

> **Cross-family uses Llama-3.2-1B-Instruct, not Base.** Paper 1's cross-family
> replication (§6.3) trains from **Llama-3.2-1B-Instruct**: the matched untrained
> control reaches recall 0.25, full SFT on that identical base reaches 0.91,
> isolating training as the sole variable. An earlier attempt on
> Llama-3.2-1B-*Base* could not learn clean turn-termination in one epoch (see
> `paper/sections/06_discussion.md`), so Base was dropped for Paper 1. The two
> `config/paper/training_llama_a{1,3}_*.yaml` (Base) configs are retained only as
> that failed-attempt record and back **no** reported Paper 1 number — do not use
> them to reproduce Paper 1. (Paper 2's off-tutor H-gen result legitimately uses
> Llama-3.2-1B-Base on a *different* synthetic task; see `REPRODUCE_paper2.md`.)

Base model: **Qwen3.5-0.8B-Base**, QLoRA r=16/α=32, 2 epochs SFT + 1 epoch DPO,
seeds {42, 123, 7}.

---

## Full reproduction — command sequence

> Prefix every command with `PYTHONUTF8=1` on Windows. Start the teacher/judge
> server first (`scripts/start_llama_cpp.ps1`) for Stages 1 and 4; stop it before
> Stage 3 (VRAM).

```bash
# Stage 1 — generate the base SFT/DPO/eval data (teacher model)
python scripts/run_generation.py               # → data/sft_filtered/, data/dpo_filtered/

# Stage 2 — materialize ablation subsets
python scripts/setup_paper_ablation_data.py --conditions a3 a4 a5

# Stage 3 — train each condition (repeat per seed via make_multiseed_configs.py)
python scripts/run_training.py --training-config config/paper/training_a1_full.yaml
python scripts/run_training.py --training-config config/paper/training_a3_no_specialized.yaml --stages train_sft
python scripts/run_training.py --training-config config/paper/training_a4_no_persistent.yaml --stages train_sft
python scripts/run_training.py --training-config config/paper/training_a5_fixed_turn_7.yaml   --stages train_sft

# Stage 4 — evaluate + score each baseline against the frozen eval sets
python scripts/run_paper_eval.py  --baseline paper_a1 --test-set all
python scripts/run_paper_score.py --baseline paper_a1 --aggregate
#   ... repeat --baseline for paper_a3_sft, paper_a4_sft, paper_a5_sft,
#       qwen3_5_0_8b_base, qwen3_5_0_8b_instruct, qwen3_5_4b_instruct, qwen3_5_9b_teacher

# Metric-specific scorers
python scripts/score_withholding_rate.py        # → outputs/paper_v2/score/pedagogy_withholding_rate.json
python scripts/score_pairwise_preference.py     # → outputs/paper/score/pairwise_preference.json
python scripts/run_multiseed_eval.py            # → outputs/paper_v2/score/multiseed_summary.json (mean ± s.d.)

# Build the paper + figures
python scripts/build_paper_merged.py            # → paper/build/paper_full.{md,pdf,docx}
```

**Fast verification (Path B):** skip Stages 1–3; run only the `run_paper_score.py`
/ `score_*` steps against the committed `outputs/**/score/*.json` + `eval_sets/`,
then `build_paper_merged.py`, and confirm the numbers below.

---

## Claim → evidence map

The `outputs/.../score/*.json` files are committed, so every number below can be
checked without retraining.

| # | Claim (as stated in paper) | Metric / script | Score file | Expected value |
|---|----------------------------|-----------------|------------|----------------|
| 1 | **Persistence is not promptable.** Prompt-only baselines barely fire; trained model fires reliably. | sentinel recall — `run_paper_score.py score_sentinel_firing()` | `outputs/paper/score/aggregated/paper_a1.json` (`persistent.recall`) | trained A1 **≈ 0.77–0.83**; prompt-only B1–B4 **≤ 0.06** (zero/few-shot); native CoT on 9B teacher **≈ 0.63** |
| 2 | **Pedagogical withholding is not promptable.** Trained model withholds the direct answer under a matched prompt; instruct/teacher baselines do not. | withholding-rate — `score_withholding_rate.py` | `outputs/paper_v2/score/pedagogy_withholding_rate.json` | trained A1 **≈ 0.57**; A3 **≈ 0.10**; all instruct-only ≤ 0.43 incl. 9B teacher **≈ 0.45** |
| 3 | **Redirect-axis F1 is context-blind and masks the boundary; pairwise eval recovers the specialized-data effect.** | pairwise preference — `score_pairwise_preference.py` | `outputs/paper/score/pairwise_preference.json`, `outputs/paper_v2/score/pairwise_all_axes_a1_vs_a3.json` | A1 beats A3 on all 6 axes (mean ≈ 0.69); **role_swap win-rate ≈ 0.87** |
| 4 | **Position decorrelation (A5) removes positional recall bias but not premature firing** → threshold-laxity, not positional shortcut. | premature-firing rate — `score_phase0_attribution.py` | `outputs/paper_v2/score/phase0_attribution.json` | premature firing **unchanged** by fixing turn 7 (see Paper 2 for the full trim analysis) |
| 5 | **Results are seed-robust** (seeds 42/123/7). | multi-seed aggregation — `run_multiseed_eval.py` | `outputs/paper_v2/score/multiseed_summary.json`, `tier1_llama_summary.json` | withholding A1 **0.63 ± 0.08** / A3 **0.13 ± 0.01**; recall **0.85 ± 0.04** |
| 6 | **Cross-family:** the direction holds on Llama-3.2-1B; magnitude is family-dependent. | Tier-1 Llama eval | `outputs/paper_v2/score/` Llama runs | direction-holds, magnitude-varies |

> Naming note: in the v2 output tree, the file label `paper_a2` corresponds to the
> full-system A1 condition (a naming carryover). The withholding value 0.57 is that
> full-system row. Cross-check IDs in `per_record` if in doubt.

---

## Where each paper table/figure comes from

| Paper artifact | Produced by | Reads |
|----------------|-------------|-------|
| Train-vs-prompt boundary matrix (2×5) | `build_paper_merged.py` (table in `paper/sections/05_results.md`) | aggregated score JSONs |
| Statistical summary table | `build_paper_merged.py` | `multiseed_summary.json` |
| Persistence prompting ladder (zero/few/CoT) | `paper/sections/05_results.md` | persistence recall JSONs |
| SFT corpus composition table | `build_paper_merged.py` | `data/sft_filtered/` counts |
