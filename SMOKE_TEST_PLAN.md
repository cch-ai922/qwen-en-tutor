# Smoke test plan

A 30-minute, ~$5 end-to-end validation run. The aim is to verify every
stage of the pipeline works against real data and real models before
committing to a full data-generation run that costs $100-500.

## Profile

Defined in [config/pipeline_smoke.yaml](config/pipeline_smoke.yaml).

- **50 SFT scenarios per level × 3 levels** (A2, B1, B2) = 150 base scenarios
- Filter pipeline runs with `enable_locale_judge: false` (mechanical only, no LLM-judge cost)
- 50 SFT training steps, 25 DPO steps (just enough to verify gradients flow + checkpoints write)
- 9 holdout scenarios for each of the two eval passes
- Comparison threshold loosened to 0.10 (smoke runs are noisy)

## Expected cost

Approximate, with `claude-haiku-4-5` as the teacher (override in `config/generation.yaml`):

| Stage              | Calls | In-tokens  | Out-tokens | USD (haiku) |
|--------------------|------:|-----------:|-----------:|------------:|
| seeds              |    15 |   165,000  |    45,000  |       $0.39 |
| sft normal         |   150 | 1,800,000  |   300,000  |       $3.30 |
| sft redirect       |   150 | 1,800,000  |   300,000  |       $3.30 |
| eval gen           |   300 | 3,600,000  |   450,000  |       $5.85 |
| DPO register       |   300 | 1,200,000  |    90,000  |       $1.65 |
| **total**          | **915** | **8.5M** | **1.2M**   |    **~$14** |

Add another **~$3-5** for the holdout eval, which simulates 9 scenarios × 8 turns × 2 LLMs (target + learner). On `gpt-4o-mini` the total is closer to **~$3-5**. On `claude-opus-4-7` the smoke run is **~$80-100** — only worth it for production validation.

For a true $5 smoke, set both teacher and judge to `claude-haiku-4-5` or `gpt-4o-mini` and cut `n_per_level` to 25.

## Prerequisites

1. `make install` — Python 3.11+, `uv sync` (or `pip install -e ".[dev]"`)
2. `make spacy` — downloads `en_core_web_sm` for the diversity tracker + locale filter
3. **API keys** — set `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` (depending on `config/generation.yaml`'s `provider`)
4. **GPU** — required for the training + eval stages. The smoke run needs ~24 GB VRAM (Qwen3 8B in 4-bit + LoRA + a small batch). On a single A100 40GB or RTX 4090, expect ~15-20 minutes for 50 SFT steps + 25 DPO steps.

## Step-by-step

```bash
# 1. Wire up API + spaCy
make install
make spacy
export ANTHROPIC_API_KEY=sk-ant-...
export WANDB_API_KEY=...           # optional; comment out report_to in training.yaml to skip

# 2. (Optional) Confirm the file plumbing first — no APIs, no GPU
make dry-run

# 3. The real smoke run (sequential, with checkpoints):
make smoke-test

# 4. Read the report
cat data/_pipeline_state/pipeline-<timestamp>/report.md
```

## What "success" looks like

The final `report.md` should show all 15 stages with status either `completed` or `dry-run`, and:

- **seeds**: ~150 scenarios written across 3 levels
- **sft_gen / redirect_gen / eval_gen / dpo_gen**: each writes JSONL of roughly the expected size
- **filter_sft / filter_eval / filter_dpo**: pass rate > 60% on each (drops on stiff register, leftover Western names, mode bleed)
- **train_sft**: a LoRA adapter materializes under `outputs/smoke_sft/`. wandb (if enabled) shows the loss curve dropping in the first 20-30 steps.
- **eval_intermediate**: produces `data/eval_results/pipeline-*_intermediate/summary.json`. Mode consistency, JSON validity, and locale fidelity should all be ≥ 0.7 on average against the SFT-only adapter.
- **on_policy_gen / filter_dpo_on_policy**: writes `data/dpo_raw/on_policy_<level>.jsonl` (one entry per filtered SFT example where the teacher won by margin ≥ 2). Inspect `data/on_policy_pairs_audit.jsonl` to see the policy-vs-teacher win-rate split — at SFT-only stage it should heavily favor teacher.
- **train_dpo**: a LoRA adapter materializes under `outputs/smoke_dpo/`. It trains on BOTH the offline register pairs AND the on-policy pairs.
- **eval_final**: produces `data/eval_results/pipeline-*_final/summary.json` against the DPO adapter.
- **compare**: writes `compare_report.json` listing the metric deltas between intermediate and final. Smoke runs are noisy, so 1-3 regressions at threshold=0.10 is acceptable.

## Failure modes the smoke test is designed to catch

| Symptom                                | Stage              | Likely root cause |
|----------------------------------------|--------------------|-------------------|
| KeyError loading `generation.yaml`     | seeds              | missing provider section or wrong env var |
| Empty `data/sft_raw/normal_*.jsonl`    | sft_gen            | teacher rejecting the prompt; check `data/sft_dialogues_failures.jsonl` |
| 0% pass rate in `filter_sft`           | filter_sft         | banned-terms catching `{cefr_level}` placeholder or similar — inspect first record |
| `eval_gen` produces records without `<think>` | eval_gen     | teacher ignoring the prompt's instructions; usually a too-small max_tokens |
| Training crashes on CUDA OOM           | train_sft          | reduce `per_device_train_batch_size` to 1 + raise `gradient_accumulation_steps` |
| Training crashes on tokenizer special-token check | train_sft | base model isn't a Qwen3 variant — confirm `base_model.model_id` |
| `eval_intermediate` returns 0 valid JSON | eval_intermediate | model never learned `/think` mode — verify your data mix had eval examples |
| `compare` flags a regression on `locale_fidelity` | compare | locale_judge wasn't run during filtering, so Western leakage slipped through |

## After the smoke run

If everything looks good:

1. Bump `n_per_level` in `config/pipeline.yaml` to 500-1000.
2. Re-enable `enable_locale_judge: true` in `filtering:` (essential before production).
3. Switch teacher to `claude-opus-4-7` if the haiku output quality looked thin.
4. Run `make full-pipeline` and check `data/_pipeline_state/<run_id>/report.md` periodically.
