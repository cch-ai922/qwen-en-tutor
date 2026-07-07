# Reproducing Paper 2

**Paper 2** — *Don't Trim the Tail: Sequence Truncation Weakens Threshold Learning
for Rare Control Tokens* (`paper_v3/`)

Read [`README.md`](README.md) first for the shared pipeline, release policy, and
environment gotchas. This guide maps each hypothesis to the exact command, config,
and expected number.

---

## The 2×2×2 design

Paper 2 crosses three binary factors over the persistent-sentinel behavior:

- **Marker:** typed `[SESSION_END: <axis>]` vs. generic `[SESSION_END]`
- **Position:** 4-variant {5,7,9,11} vs. fixed turn 7
- **Trim:** untrimmed (sequence continues past the marker) vs. trimmed (sequence ends at the marker)

| Cell | Position | Marker | Training config (base: `config/paper_v2/`) |
|------|----------|--------|--------------------------------------------|
| **A1** | 4-variant | typed | `training_a1_1ep.yaml` (untrim) · `training_a1_1ep_trim.yaml` (trim) |
| **A5** | fixed-7 | typed | `training_a5_fixed_turn_7.yaml` |
| **A6** | fixed-7 | generic | `training_a6_generic_sentinel.yaml` |
| **A7** | 4-variant | generic | `training_a7_generic_sentinel.yaml` |

Base model: **Qwen3.5-0.8B-Base**, QLoRA, **1 epoch** (matched budget), seeds {42, 123, 7}.
Off-tutor generalization (H-gen) uses **Llama-3.2-1B-Base**.

---

## Full reproduction — command sequence

> Prefix every command with `PYTHONUTF8=1` on Windows.

```bash
# Stage 1–2 — reuse the base data from Paper 1's Stage 1, then build trim/untrim variants
python scripts/setup_paper_ablation_data.py --conditions a5 a6 a7
python scripts/build_untrim_a5a6a7.py                 # untrimmed A5/A6/A7 (canonical)
python scripts/trim_persistent_post_sentinel.py       # trimmed variants

# Stage 3 — train each cell, untrimmed and trimmed (1 epoch)
python scripts/run_training.py --training-config config/paper_v2/training_a1_1ep.yaml       --stages train_sft
python scripts/run_training.py --training-config config/paper_v2/training_a1_1ep_trim.yaml  --stages train_sft
python scripts/run_training.py --training-config config/paper_v2/training_a5_fixed_turn_7.yaml   --stages train_sft
python scripts/run_training.py --training-config config/paper_v2/training_a6_generic_sentinel.yaml --stages train_sft
python scripts/run_training.py --training-config config/paper_v2/training_a7_generic_sentinel.yaml --stages train_sft
#   (+ trimmed variants of A5/A6/A7; + reseed via make_multiseed_configs.py for {123,7})

# Stage 4 — score premature firing + axis attribution + mechanism
python scripts/run_paper_eval.py  --baseline paper_a1 --test-set persistent_premature_probe
python scripts/score_phase0_attribution.py            # → outputs/paper_v2/score/phase0_attribution.json
python scripts/score_mixed_violation_probe.py         # → outputs/paper_v2/score/mixed_violation.json (H2b)
python scripts/score_sentinel_logprob_vs_depth.py \
    --trim-adapter   outputs/paper_v2/a1_1ep_trim/sft \
    --untrim-adapter outputs/paper_v2/a1_1ep/sft \
    --probe eval_sets/persistent_premature_probe.jsonl \
    --out outputs/paper_v3/score/sentinel_logprob_vs_depth.json   # mechanism (H1-mech)

# Off-tutor generalization (H-gen) on Llama-3.2-1B-Base — report EPOCH 1 only
python scripts/phase3_synthetic_trim.py build --out-dir outputs/paper_v3/phase3
python scripts/phase3_synthetic_trim.py run   --out-dir outputs/paper_v3/phase3 \
    --base ./vendor/models/Llama-3.2-1B-Base --epochs 3

# Build paper + figures
python scripts/make_paper_figures.py                  # → paper_v3/figures/fig{1-4}.png
python scripts/build_paper_v3.py                      # → paper_v3/build/paper_full.{md,pdf,docx}
```

**Fast verification (Path B):** `score_phase0_attribution.py` and
`make_paper_figures.py` read the committed
`outputs/paper_v2/score/phase0_attribution.json`, so the headline table + Figure 3
can be regenerated and checked without retraining.

---

## Hypothesis → evidence map

| H | Claim | Metric / script | Score file | Expected value |
|---|-------|-----------------|------------|----------------|
| **H1** (headline) | **Trimming raises premature firing in every (position × marker) cell**, dwarfing position/marker main effects. | premature-firing rate — `score_phase0_attribution.py` | `outputs/paper_v2/score/phase0_attribution.json` | A1 0.207→0.783 (Δ+0.576) · A5 0.119→0.560 (Δ+0.440) · A6 0.157→0.641 (Δ+0.484) · A7 0.214→0.692 (Δ+0.478) |
| **H1-mech** | Mechanism is **threshold-laxity**, not positional shortcut: trimming deletes escalation negatives, so P(sentinel) stops tracking violation depth. | logprob-vs-depth — `score_sentinel_logprob_vs_depth.py` | `outputs/paper_v3/score/sentinel_logprob_vs_depth.json` | untrimmed P(sentinel) rises with violation count; trimmed is flat/high |
| **H2** | **Typed markers are semantic gates:** correct-axis attribution 0.94–0.99; generic markers carry no axis signal; attribution is robust to trim even where timing collapses. | axis-attribution accuracy — `score_phase0_attribution.py` | `phase0_attribution.json` | typed **0.94–0.99**; generic undefined |
| **H2b** | Under distraction (primary axis at threshold, distractor sub-threshold), typed models name the correct axis. | mixed-violation — `score_mixed_violation_probe.py` | `outputs/paper_v2/score/mixed_violation.json` | correct-axis **≈ 0.975 / 0.963**; distractor **< 2%** |
| **H-gen** | The trim→premature effect **transfers off-tutor** to Llama-3.2-1B-Base. | `phase3_synthetic_trim.py run` | `outputs/paper_v3/phase3/` | **epoch 1 only:** 0.683→0.830 (Δ+0.148), recall 1.0 |
| **H3** | Count-annotated markers `[SESSION_END: STRIKE=3: axis]` make firing trim-robust. | count-marker — `assemble_a8_count.py` + eval | `outputs/paper_v3/eval_a8/` | **INCONCLUSIVE — deferred to future work (§5.5).** A8 under-fires (recall ≈ 0.06–0.13). Do not report as a positive result. |

### Statistics (in the build)

Fisher / McNemar tests on the trim effect: **p < 1e-32**; Wilson 95% CIs for
untrimmed vs. trimmed are **disjoint** in every cell. Computed by
`score_phase0_attribution.py` (significance block) →
`outputs/paper_v2/score/phase0_significance.json`.

---

## Checkpoint-availability caveat (important for reviewers)

Trained adapters are withheld (see release policy). Among the internal checkpoints,
**untrimmed A1/A5/A6/A7 and trimmed A1** were retained; **trimmed A5/A6/A7 adapters
were deleted** — their *data and configs exist*, so they are retrainable in one
`run_training.py` call each, but the exact adapters are not on hand. The reported
trimmed-A5/A6/A7 numbers come from `phase0_attribution.json` (committed). State this
in the paper's reproducibility appendix (`paper_v3/sections/08_appendix_repro.md`).

---

## Where each paper figure comes from

| Figure | Produced by | Reads |
|--------|-------------|-------|
| `fig1_sequences.png` — untrimmed vs. trimmed record | `make_paper_figures.py` | (illustrative) |
| `fig2_mechanism.png` — threshold-laxity diagram | `make_paper_figures.py` | (illustrative) |
| `fig3_premature.png` — premature firing, 4 cells × trim | `make_paper_figures.py` | `outputs/paper_v2/score/phase0_attribution.json` (real data) |
| `fig4_logit.png` — P(sentinel) vs. violation count | `make_paper_figures.py` | logprob-vs-depth report |
