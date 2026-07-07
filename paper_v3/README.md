# paper_v3 — "Don't Trim the Tail"

**Working title:** *Don't Trim the Tail: Sequence Truncation Weakens Threshold
Learning for Rare Control Tokens*

Second paper split off from `paper/` (the train-vs-prompt boundary paper, v2).
This one is a **data-curation / shortcut-learning** paper: the *shape* of SFT
data — not model scale — governs when and how a fine-tuned LM emits a rare
control marker.

## Thesis

Two data-shape choices causally control rare-marker emission in fine-tuned LMs:

- **F-B (headline) — the terminal-position / trim artifact.** Truncating
  training sequences to end at a rare marker deletes the negative examples of
  the marker's *trigger feature* (here: escalation), leaving the trigger
  perfectly predictive of firing and destroying the model's ability to
  threshold on the trigger's *magnitude* (strike count). Result: **premature
  emission**. Retaining post-marker continuation restores the counterexamples
  and the threshold.

- **F-A (supporting) — typed markers as semantic gates.** Emitting a *typed*
  marker (`[SESSION_END: <axis>]`) forces the model to attribute the firing to
  a specific violated invariant, yielding a correct, actionable classification;
  a generic marker (`[SESSION_END]`) carries no such signal and cannot be
  scored on attribution at all.

- **Remedy (constructive, Phase 2 — TODO) — count-annotated markers.** Putting
  the strike count into the marker (`[SESSION_END: <axis>, strike=3]`)
  supervises the latent counter explicitly and is predicted to neutralize the
  trim vulnerability.

## Evidence base (Phase 0 — established from existing generations, no new compute)

| Finding | Result | Source |
|---|---|---|
| F-B trim → premature | +0.44 to +0.58 across **all 4** (design×marker) cells | `phase0_attribution.json` |
| F-A typed attribution | typed 0.94–0.99 correct axis; generic 0% (undefined); 0 contentless fires for typed | same |
| Mechanism | vc-stratified: trim inflates P(fire\|escalation), breaks strike-count threshold | same |

**Disambiguated data tree:** `../outputs/paper_v2/phase0_trim_study/<COND>_<variant>/`
(each dir has `_SOURCE.json` provenance). NOTE the trap: for A5/A6/A7 the
*trimmed*-model generations live under `eval/` and *untrimmed* under
`eval_untrim/`; the copies fix this naming.

**Scorer:** `../scripts/score_phase0_attribution.py`
**Scored output:** `../outputs/paper_v2/score/phase0_attribution.json`

## Status / plan

- [x] Phase 0 — F-A + F-B from existing data (DONE; numbers verified against
      `phase0_attribution.json`)
- [x] All 7 sections drafted (§1–§7) with [PROVEN]/[EXPECTED] tags
- [x] Phase 1 probe BUILT (not generated): `scripts/build_mixed_violation_probe.py`
      + scorer `scripts/score_mixed_violation_probe.py`
- [x] Phase 2 count-marker (Design B) BUILT (not run): convert + assemble +
      2 configs (see runbook)
- [ ] Phase 1 RUN — generate mixed_violation_probe on A1/A5/A6/A7, score
- [ ] Phase 2 RUN — build A8 data, train A8 + A8-trim, eval, score
- [x] Phase 3 harness BUILT (data + probe done, CPU): `scripts/phase3_synthetic_trim.py`
      — synthetic count-to-3 STOP-marker task, zero tutoring semantics; trimmed
      vs untrimmed. `build` done (4000 seqs/variant + 400 sub-threshold probe).
- [ ] Phase 3 RUN — wire `run` to a tiny causal LM, train both variants, score
      premature STOP-emission. Prediction: trimmed > untrimmed (same DIRECTION
      as tutor §5.1) → proves trim→premature is data-shape, not corpus artifact.
- [ ] Phase 4 — mechanism figure: P(sentinel token | context) vs depth/escalation,
      trimmed vs untrimmed (forward-pass logprobs, NOT sampling; new harness
      `score_sentinel_logprob_vs_depth.py`). Reuses persistent_premature_probe
      contexts. Two x-axes: raw turn-depth (expect flat for both) vs
      accumulated escalation at fixed depth (expect trimmed to diverge upward at
      strike<3). This is the single best figure for the §6.1 mechanism.
- [ ] Cross-family magnitude check (DEFERRED, §6.5 item 6): re-run ONLY the A1
      trim/untrim contrast on a small full-attention non-Qwen base
      (Llama-3.2-1B or SmolLM2-360M) to test whether the +0.5 trim Δ is specific
      to Qwen's linear-attention state. NOT load-bearing — Phase 3 already covers
      *existence*; this tests *magnitude*. To run: download the base, clone
      `training_a1_1ep.yaml` + `training_a1_1ep_trim.yaml` changing only
      `base_model.model_id`, retrain both, eval + score_phase0_attribution. One
      cell, not the 2×2×2.
- [ ] Surgery on `paper/` (v2): strip the F-A/F-B mechanism analysis, keep the
      2×2 only as a budget/design robustness check, add forward-reference here

## Resume runbook (tomorrow — all GPU steps need the training pipeline free)

**Phase 1 (no training):**
```
# 1. build the frozen probe (CPU)
python scripts/build_mixed_violation_probe.py --out eval_sets/mixed_violation_probe.jsonl
# 2. generate on the TYPED checkpoints (GPU) — attribution is typed-only,
#    so A1 + A5 are the load-bearing runs. A6/A7 are OPTIONAL and only feed the
#    secondary distractor_sub premature check (generic markers have no axis to
#    attribute), so skip them unless you want that secondary number.
python scripts/run_paper_eval.py --baseline v3_a1_untrim --test-set mixed_violation_probe --output-dir outputs/paper_v3/eval_mixed
python scripts/run_paper_eval.py --baseline v3_a5_untrim --test-set mixed_violation_probe --output-dir outputs/paper_v3/eval_mixed
# optional (secondary premature-under-distraction only):
# python scripts/run_paper_eval.py --baseline v3_a6_untrim --test-set mixed_violation_probe --output-dir outputs/paper_v3/eval_mixed
# python scripts/run_paper_eval.py --baseline v3_a7_untrim --test-set mixed_violation_probe --output-dir outputs/paper_v3/eval_mixed
# 3. score
python scripts/score_mixed_violation_probe.py --eval-dir outputs/paper_v3/eval_mixed \
    --out outputs/paper_v3/score/mixed_violation.json
# 4. paste correct-axis / wrong-axis into 05_results.md §5.4 (replace [EXPECTED])
```

**Phase 2 (count-marker, Design B):**
```
# 1. count-annotate A1's typed persistent (CPU)
python scripts/convert_typed_to_count.py \
    --src-dir data/sft_filtered --dst-dir data/sft_filtered_a8_count_persistent
# 2. assemble full untrimmed A8 dir (CPU)
python scripts/assemble_a8_count.py
# 3. train A8 untrimmed (GPU)
python scripts/run_training.py --config config/paper_v2/training_a8_count_sentinel.yaml \
    --stages train_sft
# 4. for A8-trim: trim the count persistent, assemble --trim, train the _trim config
#    (trim_persistent_post_sentinel.py matches "[SESSION_END" so strike=3 truncates)
# 5. eval both on persistent_probe + persistent_premature_probe, score with
#    score_phase0_attribution.py (add A8/A8-trim cells), paste trim-Δ into §5.5
```

## Files created for paper_v3 (this session)

Scripts (all CPU-verified where non-GPU; none run against the GPU):
- `scripts/build_mixed_violation_probe.py` — Phase 1 probe builder (dry-verified: 282 recs)
- `scripts/score_mixed_violation_probe.py` — Phase 1 scorer
- `scripts/convert_typed_to_count.py` — Design B count-marker data (dry-verified: 614 recs tagged)
- `scripts/assemble_a8_count.py` — assemble full A8 SFT dir
- `scripts/score_phase0_attribution.py` — Phase 0 scorer (already run)

Configs:
- `config/paper_v2/training_a8_count_sentinel.yaml` — A8 (count, untrimmed)
- `config/paper_v2/training_a8_count_sentinel_trim.yaml` — A8-trim

Data (disambiguated):
- `outputs/paper_v2/phase0_trim_study/<CELL>_<variant>/` + `_SOURCE.json` each
- `outputs/paper_v2/score/phase0_attribution.json`

## Checkpoint availability (IMPORTANT — constrains what is re-runnable)

Available adapters (exact paths, per user):
- **untrimmed:** A1 = `outputs/paper_v2/a1_1ep/sft`, A5 = `outputs/paper_v2/a5/sft`,
  A6 = `outputs/paper_v2/a6/sft`, A7 = `outputs/paper_v2/a7/sft`
- **trimmed:** A1-trim = `outputs/paper_v2/a1_1ep_trim/sft` ONLY
- **DELETED:** A5-trim, A6-trim, A7-trim

These are wired into `run_paper_eval.py` as dedicated baselines `v3_a1_untrim`,
`v3_a1_trim`, `v3_a5_untrim`, `v3_a6_untrim`, `v3_a7_untrim` (separate from the
paper_v2 `paper_a2/a5_sft` baselines, which point at the 2-epoch A1 and are left
unchanged). Phase-1/4 use these v3_* baselines. Generations write to
`outputs/paper_v3/eval_mixed/<baseline>/`.

Consequences:
- **H1 (§5.1) is SAFE** — its numbers come from saved *generations*
  (`phase0_trim_study/*/`, `phase0_attribution.json`), not live checkpoints, so
  deleting the trimmed A5/A6/A7 adapters does NOT un-prove it. But those trimmed
  A5/A6/A7 outputs can no longer be *regenerated* (e.g. for new probes) without
  retraining.
- **Phase 4 mechanism figure** uses the **A1 pair** (a1_1ep_trim vs a1_1ep) —
  the surviving trimmed checkpoint, and the largest trim Δ. Do not point it at
  a5/a6/a7-trim (gone).
- **Phase 1** (A1, A5 untrimmed) and **Phase 2** (A8 new trainings) are
  unaffected.
- If a reviewer wants the trimmed A5/A6/A7 re-run, they must be *retrained*
  (configs + data still exist: `training_a{5,6,7}_*.yaml`, trimmed data dirs).

## Building the full paper (md / docx / pdf)

```
python scripts/build_paper_v3.py
```

Merges `paper_v3/sections/01..07` (title + abstract are parsed from
`01_introduction.md`), resolves `[@key]` citations against
`paper_v3/references.bib` via pandoc `--citeproc`, appends a `# References`
list, and writes `paper_v3/build/paper_full.{md,pdf,docx}`. Requires pandoc +
xelatex (both present on this machine). Re-run after editing any section or the
bib. Citations use pandoc-native `[@key]` markdown; add new entries to
`references.bib` (it is self-contained — reuses the relevant paper_v2 entries).

## Conventions carried from v2

- Firing = generation contains a `[SESSION_END...]` marker (see scorer regex).
- Cell tags: A1=4var+typed, A5=fix7+typed, A6=fix7+generic, A7=4var+generic.
- All Phase-0 numbers are single training seed (seed 42); reseed 123/7 pending
  (§4.10 of v2). The trim effect (~+0.5) is far too large for seed variance.
