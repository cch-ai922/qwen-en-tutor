# Reproducibility Package

This directory documents how to reproduce the results in the two papers produced
from this repository. It is written for **reviewers and independent researchers**:
it maps every headline claim to the exact script, config, and expected number, and
explains what is released, what is withheld, and why.

> **TL;DR for reviewers.** The trained model checkpoints are **not** distributed
> (they are LoRA deltas over public base models, and are large). Everything needed
> to *regenerate* them — data-generation scripts, training configs, seeds,
> evaluation sets, judge prompts, and scoring scripts — **is** released. Follow the
> per-paper guide to rebuild any result end to end.

---

## 1. The two papers and how they relate

Both papers are produced by **one shared pipeline**. They differ only in *which
experimental conditions they consume* and *which analysis they run* — not in the
underlying code.

| Paper | Title | Question | Guide |
|-------|-------|----------|-------|
| **Paper 1** (`paper/`, condensed as `paper_workshop/`) | *What Must Be Trained and What Can Be Prompted: A Per-Capability Study of Tutor Redirect Behavior* | Which tutor behaviors need fine-tuning vs. can be prompted? | [`REPRODUCE_paper1.md`](REPRODUCE_paper1.md) |
| **Paper 2** (`paper_v3/`) | *Don't Trim the Tail: Sequence Truncation Weakens Threshold Learning for Rare Control Tokens* | How does trimming SFT sequences change when a rare control marker fires? | [`REPRODUCE_paper2.md`](REPRODUCE_paper2.md) |

`paper_workshop/` is an 8-page condensation of Paper 1 with no new experiments — it
reuses Paper 1's outputs. It does not need a separate reproduction guide.

**Why they are not split into two repos.** Both papers share the same data
generation (`scripts/run_generation.py`), training (`scripts/run_training.py`), and
evaluation/judge (`scripts/run_paper_eval.py`, `scripts/run_paper_score.py`) code.
Paper 1 consumes conditions **A1, A3, A4, A5** (2-epoch, with prompt-only baselines
B1–B4); Paper 2 consumes **A1, A5, A6, A7** across **trimmed vs. untrimmed** data
(1-epoch, matched budget). Forking the code would duplicate it and drift. Instead,
the shared pipeline lives once at the repo root, and each paper's guide names the
subset it uses.

---

## 2. The shared pipeline (four stages)

```
  ┌─ Stage 1: DATA GENERATION ────────────────────────────────────────────┐
  │  teacher LLM (Qwen3.5-9B, served by llama.cpp) generates synthetic      │
  │  tutoring dialogues across 12 streams × 6 CEFR levels, then filters.    │
  │  scripts/run_generation.py  →  data/sft_filtered/                       │
  └────────────────────────────────────────────────────────────────────────┘
                                   │
  ┌─ Stage 2: PER-CONDITION DATA SHAPING ─────────────────────────────────┐
  │  materialize A3/A4/A5/A6/A7 subsets & trim/untrim/typed/generic/count   │
  │  variants from the base data.                                          │
  │  scripts/setup_paper_ablation_data.py, regen_persistent_a5.py, ...     │
  └────────────────────────────────────────────────────────────────────────┘
                                   │
  ┌─ Stage 3: TRAINING (QLoRA SFT [+ DPO for Paper 1 A1]) ─────────────────┐
  │  scripts/run_training.py --training-config config/.../training_*.yaml   │
  │  →  outputs/.../{condition}/sft  (LoRA adapters — NOT released)         │
  └────────────────────────────────────────────────────────────────────────┘
                                   │
  ┌─ Stage 4: EVALUATION + LLM-JUDGE SCORING ─────────────────────────────┐
  │  generate on frozen eval_sets/, then score (mechanical + LLM judge).   │
  │  scripts/run_paper_eval.py  →  scripts/run_paper_score.py               │
  │  →  outputs/.../score/*.json  →  paper tables/figures                   │
  └────────────────────────────────────────────────────────────────────────┘
```

**Entry-point scripts (all under `scripts/`):**

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `run_generation.py` | Teacher-generated dialogues → raw → filtered SFT/DPO/eval data |
| 2 | `setup_paper_ablation_data.py` | Materialize A3/A4/A5/A6/A7 data subsets |
| 2 | `regen_persistent_a5.py`, `regen_persistent_a6.py`, `convert_a1_to_a7.py` | Persistent-stream variants (turn-7 / generic / A7) |
| 2 | `trim_persistent_post_sentinel.py`, `build_untrim_a5a6a7.py` | Trim / untrim variants (Paper 2) |
| 2 | `convert_typed_to_count.py`, `assemble_a8_count.py` | Count-marker variant (Paper 2, H3 — deferred) |
| 3 | `run_training.py` | SFT (+ optional DPO) via a `--training-config` YAML |
| 3 | `make_multiseed_configs.py` | Auto-generate seed-123 / seed-7 configs |
| 4 | `run_paper_eval.py` | Generate responses on a frozen eval set |
| 4 | `run_paper_score.py` | Mechanical + LLM-judge scoring; `--aggregate` for summaries |
| 4 | `score_withholding_rate.py`, `score_pairwise_preference.py`, `score_phase0_attribution.py` | Metric-specific scorers |
| 4 | `run_multiseed_eval.py` | Aggregate seeds {42, 123, 7} → mean ± s.d. |
| — | `build_paper_merged.py`, `build_paper_v3.py`, `make_paper_figures.py` | Compile papers + figures |

---

## 3. What is released vs. withheld

| Artifact | Released? | Where / why |
|----------|-----------|-------------|
| Data-generation, training, eval, scoring **scripts** | ✅ Yes | `scripts/`, `src/` (in git) |
| Training **configs** (all conditions, all seeds) | ✅ Yes | `config/paper/`, `config/paper_v2/` (in git) |
| **Generation config + judge prompts** | ✅ Yes | `config/generation.yaml`; prompts inline in `scripts/run_paper_score.py`, `score_withholding_rate.py` |
| **Frozen evaluation sets** | ✅ Yes | `eval_sets/*.jsonl` (in git) |
| **Generated SFT/DPO training data** | ⚠️ On request / regenerable | `data/` is gitignored (large + teacher-derived); fully regenerable via Stage 1. See §4. |
| **Score outputs** (the numbers behind tables) | ✅ Yes | `outputs/**/score/*.json` — small JSON, committed as evidence |
| **Trained model checkpoints** (LoRA adapters, merged, GGUF) | ❌ Withheld | Large; LoRA deltas over public bases; available from authors on request |
| **Base models** (Qwen3.5, Llama-3.2) | ❌ Not ours | Download from HuggingFace (see per-paper guide) |
| **Teacher/judge GGUF binaries** | ❌ Not ours | Download from HuggingFace / llama.cpp releases |

**Reproducibility statement to paste into both papers:**

> *All data-generation, training, and evaluation code, together with the training
> configurations, random seeds, frozen evaluation sets, and judge prompts, are
> released at <REPO_URL> to enable full reproduction. Synthetic training data is
> regenerable from the released generation scripts and is additionally available
> from the authors on request. Trained LoRA adapters are low-rank deltas over the
> publicly available base models (Qwen3.5-0.8B-Base, Llama-3.2-1B) and are
> available from the authors on request. Per-condition score outputs (the JSON
> underlying every reported table) are included in the repository.*

---

## 4. Two ways to reproduce

**Path A — Full reproduction from scratch (regenerate everything).**
Run Stage 1 to regenerate the SFT data with the teacher model, then Stages 2–4.
This reproduces the *entire* result including data. Requires a GPU and the teacher
GGUF. Expect run-to-run variation in the generated data (different teacher sampling)
but the *qualitative and statistical* conclusions are stable across seeds — that is
itself a reported result (multi-seed §, seeds 42/123/7).

**Path B — Verify from released data + scores (fast).**
Skip Stage 1: use the committed `eval_sets/` and `outputs/**/score/*.json` to
re-run the *scoring* and *table/figure build* and confirm the numbers match the
paper. This checks the analysis without a multi-day GPU run.

Each per-paper guide gives the exact commands for both paths.

---

## 5. Environment (critical gotchas)

These silently break reproduction if missed — document them for reviewers.

| Requirement | Why |
|-------------|-----|
| `PYTHONUTF8=1` (Windows) | TRL reads jinja chat templates as UTF-8; Windows cp1252 default crashes. Prefix **every** training/generation command. |
| `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | Pipeline is offline-by-default; avoids Hub calls mid-run. |
| Teacher/judge served via `llama-server.exe` on `127.0.0.1:8080/v1` | Generation + judging call an OpenAI-compatible endpoint. Launch command in `scripts/start_llama_cpp.ps1`. Cannot co-reside with training on 12 GB VRAM — stop the server before training. |
| LoRA EOS: pass **both** `<|im_end|>` and `<|endoftext|>` at generation | Qwen3.5 LoRA can still emit the doc-level EOS; missing this truncates/hangs eval. |
| `python -m spacy download en_core_web_sm` | Used by generation filters. |
| Python 3.10+, `pip install -e .` (see `pyproject.toml`); offline bundle via `scripts/setup_offline.py` | Dependency install. |
| pandoc + xelatex | Paper/figure build only. |

See the repo root `OFFLINE_SETUP.md` and `README.md` for full environment setup.

---

## 6. Files in this directory

- `README.md` — this file (shared overview + release policy).
- `REPRODUCE_paper1.md` — Paper 1 claim → script → config → expected number.
- `REPRODUCE_paper2.md` — Paper 2 claim → script → config → expected number.
