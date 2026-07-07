# 4. Experimental Setup

<!-- paper_v3 — "Don't Trim the Tail" -->
<!-- Grounded in the actual paper_v2 pipeline. Script/config/data paths are
     verbatim so results are reproducible. -->

## 4.1 Base model, training regime

All conditions fine-tune the same **Qwen3.5 0.8B base** with QLoRA
[@dettmers2023qlora; @hu2022lora] SFT (no DPO [@rafailov2023dpo]),
**1 epoch**, **seed 42**, via `scripts/run_training.py --stages train_sft`
reading a per-condition YAML under `config/paper_v2/`. Using SFT-only at a single
matched budget removes training-budget and DPO as confounds, isolating the
data-shape factors. (The paper_v2 primary A1 was 2 epochs; we do not use it
here — we use the matched 1-epoch retrain `training_a1_1ep.yaml`, so every cell
in this paper shares budget and seed.)

## 4.2 The eight cells (2×2×2)

The design crosses **position** (fixed-7 / 4-variant), **marker**
(typed / generic), and **trim** (trimmed / untrimmed). Cell tags follow
paper_v2:

| Cell | position | marker | config |
|------|----------|--------|--------|
| A1 | 4-variant | typed   | `training_a1_1ep.yaml` |
| A5 | fixed-7   | typed   | `training_a5_fixed_turn_7.yaml` |
| A6 | fixed-7   | generic | `training_a6_generic_sentinel.yaml` |
| A7 | 4-variant | generic | `training_a7_generic_sentinel.yaml` |

The trim factor is a data manipulation applied to each cell's persistent
streams (§4.4).

## 4.3 Data

SFT data lives in `data/`. Each condition trains from a `data/sft_filtered_*`
directory holding all twelve streams (normal + generic redirect + 6 specialized
redirects + 4 persistent). Only the **four persistent streams**
(`persistent_off_topic`, `persistent_language_violation`,
`persistent_persona_break`, `persistent_role_swap`) differ across cells; the
other eight streams are shared, so any cell-to-cell difference is attributable to
the persistent-stream manipulation alone. Persistent records are ~1/5 of the
corpus; the remaining ~4/5 are deep benign dialogues that end normally without a
marker — so "conversation depth" alone is already decorrelated from firing by the
shared majority data (relevant to the mechanism, §6).

## 4.4 How each cell's persistent data is produced (exact provenance)

The trim/marker/position variants are produced by mechanical transforms of a
common source, not independent regenerations — so a cell-to-cell contrast is a
clean single-factor manipulation. The chains, from the actual build scripts:

**Marker (typed → generic):**
- `scripts/convert_a1_to_a7.py` — takes A1's 4-variant **typed** persistent
  data (`data/sft_filtered`), string-replaces `[SESSION_END: persistent_<axis>]`
  → `[SESSION_END]`, and re-renders the deployment system prompt with the
  generic `[persistence]` block (`QWEN_TUTOR_SENTINEL_FORMAT=generic`). Produces
  A7's persistent data. Holds position (4-variant) fixed.
- `scripts/convert_a5_to_a6.py` — same transform on A5's fixed-7 typed data →
  A6. Holds position (fixed-7) fixed. So A5↔A6 and A1↔A7 each isolate exactly the
  marker factor.

**Trim (untrimmed → trimmed):**
- `scripts/trim_persistent_post_sentinel.py` — truncates each persistent record
  so its last message is the assistant turn containing the marker, and updates
  `metadata.generation.message_count`. Applied to A5/A6/A7 persistent dirs. This
  is the *native* trim state of A5/A6/A7.
- A1's native persistent data (`data/sft_filtered`) is **untrimmed** (keeps the
  benign post-marker continuation). Its trimmed counterpart is a separate copy,
  `data/sft_filtered_a1_trim`, used by `training_a1_1ep_trim.yaml`.

**Reconstructing the untrimmed counterparts of A5/A6/A7** (for the trim contrast):
- `scripts/build_untrim_a5a6a7.py` rebuilds untrimmed persistent data without
  re-running the teacher:
  - **A5-untrim** = the passed A5 records with their trimmed `messages` replaced
    by the untrimmed `messages` from the raw regen output
    (`data/sft_raw_a5_persistent`), matched by id. The script verifies the
    filtered (trimmed) messages are an **exact prefix** of the raw (untrimmed)
    messages — the fidelity guarantee that makes the trim contrast clean.
  - **A7-untrim** = `convert_a1_to_a7` applied to A1's *untrimmed* persistent
    data.
  - **A6-untrim** = `convert_a5_to_a6` applied to A5-untrim.
  Output: `data/sft_filtered_a{5,6,7}_untrim/`.

We document these chains explicitly because the untrimmed A5/A6/A7 are
*reconstructed*, not trained-from-scratch matched pairs; the exact-prefix
verification is the central check that the only thing changing between a
trimmed and untrimmed cell is the presence of the post-marker continuation.

## 4.5 Evaluation

Generations are produced by `scripts/run_paper_eval.py`, one checkpoint at a
time, over the frozen held-out probes in `eval_sets/` (20% hash-deterministic
seed split, `scripts/build_eval_sets.py`), written to
`outputs/paper_v2/eval*/{condition}/{test_set}.jsonl`. Firing is mechanical: a
generation "fires" iff it contains a `[SESSION_END...]` marker. The axis, if
present, is the token inside `[SESSION_END: <axis>]`.

**Disambiguated trim-study layout.** Because the paper_v2 output dirs encode
trim status confusingly (for A5/A6/A7 the *trimmed*-model generations live under
`outputs/paper_v2/eval/`, the untrimmed under `eval_untrim/`; A1's pair lives
under `eval_a1_1ep/` and `eval_a1_1ep_trim/`), we copy them into a
self-documenting tree, `outputs/paper_v2/phase0_trim_study/<CELL>_<variant>/`,
each dir carrying a `_SOURCE.json` provenance record.

## 4.6 Scoring scripts

- `scripts/score_sentinel_2x2.py` — recall / FP / premature (paper_v2, mechanical).
- `scripts/score_phase0_attribution.py` — this paper's F-A/F-B scorer: premature
  by (design × marker × trim) cell, vc-stratified, plus axis-attribution accuracy
  and the emitted-vs-true axis confusion matrix. Output:
  `outputs/paper_v2/score/phase0_attribution.json`.
- `scripts/score_mixed_violation_probe.py` — Phase 1 scorer for attribution
  under distraction.

## 4.7 Probes

- **persistent_probe** — true third-strike positives (recall / attribution).
- **persistent_premature_probe** — single-axis sub-threshold contexts, n=318,
  stratified by (violation_count, premature_turn). Built by
  `build_eval_sets.py::build_persistent_premature_probe`.
- **mixed_violation_probe** (Phase 1) — `scripts/build_mixed_violation_probe.py`.
  Primary axis X escalates to threshold while distractor axis Y appears
  sub-threshold; n≈282 (141 fire_correct / 141 distractor_sub), all 12 X×Y pairs.
  Generated on the typed checkpoints (A1, A5) and scored by
  `score_mixed_violation_probe.py` (§5.4; output
  `outputs/paper_v3/score/mixed_violation.json`).

## 4.8 Statistical note

All numbers are single training seed (seed 42). We additionally retrain the
primary A1 trim/untrim pair at an independent seed (seed 7) and reproduce the
trim effect (§5.1). The trim effect (~+0.5, §5) is far larger than plausible seed
variance; the attribution effect (typed ~0.96 vs generic undefined) is structural.
