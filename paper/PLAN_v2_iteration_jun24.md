# Paper iteration v2 — regen + retrain + re-eval plan

Captured 2026-06-23 after diagnostics from the current iteration (B1-B4
prompt-augment + tutor_v2 rubric pass) surfaced four structural issues
in the data and one in the eval coverage. This plan addresses all five.

## Issues found in current iteration

| # | issue | evidence | fix |
|---|---|---|---|
| 1 | 6 specialized redirect streams have ~7 records per CEFR level (44 total each) | `data/sft_filtered/{locale,language,persona,pedagogy,role_swap,topic}_redirect_*_passed.jsonl` | boost to ~35/level (210 each) |
| 2 | C1/C2 generic-redirect drops to ~20 records (vs ~168 at B2) | `data/sft_filtered/redirect_C{1,2}_passed.jsonl` | refill C1/C2 to match B2 volume |
| 3 | Length-style mismatch between teacher's training-data turns (short) and B-baseline eval responses (long) | response-length analysis: A1 161 chars vs B4 329 chars at eval time, vs B4's own training turns at A1 = 73 chars | bake `[response shape]` directive into canonical deployment system prompt template |
| 4 | A5 (fixed-turn-7) shows degraded recall but **not** false positives (Pers-FP=0.00) — §3.4 hypothesis only partly demonstrated | aggregate from Prometheus pass: A5 Pers-FP=0.00, OffPos-Recall=0.55 | boost persistent ×1.5 — gives A5 best shot at FP emergence at trained position |
| 5 | `persistent_offposition_probe` has only 60 records at 2 positions ({13, 15}); §3.4 generalization tested only at ±2/±4 from the grid | `eval_sets/persistent_offposition_probe.jsonl` line count = 60 | expand to ±6/±8/±10/±12 (180 records, 4 new positions) |
| 6 | **redirect_axis macro F1 metric was broken by four independent bugs** at v1; the metric is supposed to test the §3.3 / N2 taxonomy contribution (does adding 6 specialized redirect streams produce axis-appropriate repairs?). Until these are fixed, no N2 conclusion can be drawn from this metric. | see "Redirect-axis macro F1 bug stack" below | regenerate sources with metadata; enrich judge prompt with scenario context; drop Prometheus from this metric; rebuild eval set with metadata-only `violation_turn_idx` |

## Redirect-axis macro F1 bug stack (found 2026-06-24)

The macro F1 metric is the N2 (taxonomy contribution) test: it measures
whether the model produced the **axis-appropriate repair** for each
of 7 redirect axes. Four independent bugs corrupted v1 results:

| # | bug | impact | v2 fix |
|---|---|---|---|
| A | Judge prompt's `{context}` was always empty: `_last_user_turn(record)` read a field nothing wrote. | Judges classified responses with no idea what the user said. | Done in code 2026-06-24 (build_eval_sets emits `violation_turn_content`; paper_eval propagates it; run_paper_score reads it). |
| B | Prometheus is rubric-trained, ignores the categorical-JSON instruct prompt, fills `max_new_tokens=160` with feedback prose, parses to `axis="none"` on 98.4% of records. | Three-judge median was polluted by Prometheus voting `none`. Ties between Llama-3.1 and Gemma-2 were broken by Prometheus's bogus `none`. | Done in code 2026-06-24: aggregator skips Prometheus for `redirect_axis`. Ensemble for this metric is Llama-3.1 + Gemma-2 only. |
| C | `violation_turn_idx` is teacher-emitted metadata for 78.7% of records but heuristic/fallback for 21.3% (36 records). Audit of 8 heuristic records: 5/8 (~62%) point at a **benign post-redirect turn** instead of the actual violation, because the source dialogue continues past the teacher's redirect. | ~13% of every baseline's F1 is computed on records whose probe context contains the actual violation + teacher's redirect + a benign follow-up turn, with the model asked to redirect the benign turn. | Regenerate the redirect source dialogues so every record carries `metadata.generation.violation_turn_idx` from the teacher (Option 2). Drop heuristic-detected records from probe construction. |
| D | Judge prompt receives only the violation turn + tutor response, NOT the scenario topic / subtopics / user role / tutor role / locale / CEFR level. | Judge can't tell whether mentioning "Costco" is a `locale` miss (depends on knowing the locale), or whether a topic shift is `topic` redirect (depends on knowing the original topic). Systematic confusion across axes. | Enrich `REDIRECT_AXIS_PROMPT` to include the rendered scenario context (or the system_prompt fields). |

Bugs A and B are fixed in code already and are picked up on the next
aggregation pass; bugs C and D need the v2 source-regen and prompt
work below.

What this means for the paper draft: the N2 contribution claim
(`A2 macro F1 > A3-SFT macro F1`) cannot be supported by the
current metric until bugs C and D are also fixed. The post-fix
numbers (A and B applied) show the direction flipped (A3-SFT 0.390 >
A2 0.373). We hold any §5.4 narrative rewrite until v2 closes bugs
C and D.

## Already fixed this iteration (do not redo)

- **B4 thinking-mode bug**: APIBaseline.generate appends `/no_think` (was emitting `<think>...</think>` with empty content)
- **B1-B4 system-prompt augment (Option A)**: temporary runtime patch in `augment_system_for_offshelf()` ([scripts/run_paper_eval.py](../scripts/run_paper_eval.py)); next iteration replaces this with the same text baked into the canonical deployment template, so all baselines and trainees see it uniformly
- **Probe ID dedup, redirect violation_turn_idx metadata, per-condition DPO data dirs**

## Rubric decision (paper uses generic only)

We do NOT use `tutor_v2` (the calibrated rubric with CEFR length anchors and few-shot examples) in the paper. Reasons:

- Generic rubric is the industry-standard 1-5 evaluation prompt — no methodological defense required.
- A calibrated rubric can be challenged as "you tuned the rubric to favor your model." The story we tell with the generic rubric is more defensible.
- Generic rubric already shows A1 beats B1/B2/B3 on naturalness at matched model size; CEFR-adherence gap is honestly acknowledged and explained via length-asymmetry + model-size confounds in §6.

The `tutor_v2` rubric code stays in [scripts/run_paper_score.py](../scripts/run_paper_score.py) so it can be re-run if a reviewer asks for it; the v2 iteration runs and aggregates only the generic rubric.

## Sentinel-architecture clarification (don't re-test)

The sentinel `[SESSION_END: <axis>]` is the model's signal to the
external dispatcher. Once the model emits it, the conversation system
terminates the session — the model never sees another user turn after
sentinel. A "post-sentinel continuation" eval is therefore meaningless;
the situation does not occur in deployment.

## Data budget

| stream | current | target | per-level | new records |
|---|---:|---:|---:|---:|
| normal | 1,795 | **3,000** | 500 | +1,205 |
| redirect (generic) | 646 | **900** | 150 (A1-B2), 100 (C1-C2) | +254 |
| 6 specialized × 1 each | ~44 | **210 each** | 35/level | +166 each, +996 total |
| 4 persistent × 1 each | ~168 | **250 each** (1.5×) | 42/level | +82 each, +328 total |
| **TOTAL** | ~6,055 | ~6,160 | — | **~2,783 new** |

Curriculum check: normal stays at 49% of training mix, specialized
aggregate 20%, persistent aggregate 16%, generic redirect 15%. Normal
remains dominant — no curriculum flip.

## Eval-set additions

| eval set | change | size | new records | teacher time |
|---|---|---:|---:|---:|
| `persistent_offposition_probe` | expand from {13,15} to {13,15,17,19,21,23} (30 each) | 60 → **180** | +120 | ~7 min |
| `redirect_escalation_probe` (NEW) | 2nd-strike user turn after a 1st-strike redirect already produced; model must produce a SHORTER, differently-worded 2nd-strike redirect | — → **60** | +60 | ~4 min |

Total new eval records: ~180. Paper §4.6 (Eval test sets) must be
updated to describe both.

## Five-stage pipeline

### Stage 0 — Eval-set additions and redirect-axis fixes (~30 min)

**0a. Probe-set additions** (existing plan):

- Update [scripts/build_eval_sets.py](../scripts/build_eval_sets.py) to add positions {17, 19, 21, 23} to the off-position builder and a new `build_redirect_escalation_probe` builder.
- Run `python scripts/build_eval_sets.py` to materialize the new eval files.
- Verify counts: `persistent_offposition_probe` = 180, `redirect_escalation_probe` = 60.

**0b. Redirect-axis metric fixes** (added 2026-06-24 after bug stack found, see "Issue 6" above):

- Stage 1's source regen for the 7 redirect streams must ensure **every** redirect dialogue carries `metadata.generation.violation_turn_idx` (Option 2). Drop any source dialogue that doesn't carry the field rather than fall back to the heuristic detector — losing ~10-20% of source records is acceptable; ~62% wrong heuristic detection is not.
- `build_redirect_probe` in `build_eval_sets.py` must require `metadata.violation_turn_idx` — refuse to include any record whose detection_reason would be `pivot` or `fallback`. This will shrink the probe from 169 → ~133 records but guarantees ground-truth correctness.
- Enrich `REDIRECT_AXIS_PROMPT` in `run_paper_score.py` to include the scenario context (topic, subtopics, user_role, model_role, locale_country, cefr_level). The judge needs this to distinguish e.g. `locale` from `topic` redirects. Carry these through paper_eval generation records (they're already in `system_prompt`; extract or store them alongside).
- Keep `prometheus_7b_judge` out of the `redirect_axis` ensemble (already in code on 2026-06-24).

### Stage 1 — Data regen (~3h teacher + ~10 min filter)

- Total new records: ~2,783 at ~3.5 sec/record via teacher API ≈ **~2.7h**.
- Streams: spawn 4 parallel generator processes (normal, redirect, specialized batch, persistent batch) — teacher API handles concurrent requests.
- Bake the `[response shape]` directive (currently the Option-A runtime addendum) into [_SCENARIO_DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE](../src/qwen_tutor/generation/prompts.py) *before* this stage so:
  - Student SFT data carries it implicitly via teacher's generated turns
  - B1-B4 zero-shot eval inherits it at inference time without runtime augment (clean methodology — remove `augment_system_for_offshelf` once baked in)
- After regen: run `clean_sft_train_data.py` filter; merge into `data/sft_filtered/`.

### Stage 2 — Per-condition data preparation (~5 min)

- **A1**: full mix (all streams)
- **A3**: full mix except 6 specialized (keeps generic redirect)
- **A5**: full mix but all persistent records forced to sentinel_turn=7 (re-run `regen_persistent_a5.py` with new data, `QWEN_TUTOR_PERSISTENT_FORCED_VARIANT=1`)
- Re-run `setup_paper_ablation_data.py --conditions a1,a3,a5`.
- A2, A4 are not retrained — outcomes already known.

### Stage 3 — SFT retrain A1/A3/A5 only (~30-40h)

- SFT-only (no DPO — register-pool leakage across conditions remains unresolved).
- Per condition: ~10-13h on 12GB GPU at LoRA r=16 (data ~1.7× current; current was ~6-8h).
- Sequential on the GPU; total **~30-40h SFT**.
- Outputs to `outputs/paper_v2/{a1,a3,a5}/sft/`.

### Stage 4 — Eval generation (~3h)

- Regenerate paper_eval for all 9 baselines across all test sets including the 2 new/expanded eval sets.
- With `[response shape]` baked in, no runtime augment needed — remove `augment_system_for_offshelf` call site in [scripts/run_paper_eval.py](../scripts/run_paper_eval.py).
- Mechanical scoring runs immediately after generation per baseline (no GPU needed).

### Stage 5 — Judging + aggregate (~4h)

- Same 3-judge cross-family ensemble: Prometheus 7B, Llama-3.1 8B, Gemma-2 9B.
- Rubric: **generic only** (tutor v1 and tutor_v2 both dropped — see rubric decision above).
- Reuse `orchestrate_b14_refresh.py` + `watchdog_swap_to_gemma2.py` skeleton; update phase scope to all 9 baselines × {generic}.
- Add **inter-judge Cohen's κ** computation on generic scores after aggregate completes to back the cross-family-ensemble claim in §6.

### Stage 6 (optional) — A1 LoRA rank ablation (~13h + ~1h eval/judge)

- Run after Stage 5 main results are in.
- Retrain only A1 with `lora.r=32, lora_alpha=64` (keep alpha/r ratio = 2).
- Same data, same training schedule, same hyperparams otherwise.
- Eval + judge the new checkpoint as a 10th baseline (`paper_a1_r32`).
- Compare against A1 r=16 (Stage 3) on identical eval sets.
- Adds **~10-13h SFT + ~1h eval/judge** to budget.
- If results show meaningful gain (>2 points on generic judged scores OR a clear F1 shift), add a §6 capacity-vs-data trade-off paragraph; otherwise drop from paper.
- Decision criterion: don't widen rank for *all* conditions just because A1 r=32 helped — would be a separate iteration v3 question.

## Predicted outcomes (what to watch for)

| signal | meaning |
|---|---|
| **A5 Pers-FP ≥ 0.05** | §3.4 trigger-position decorrelation confirmed at FP level (headline) |
| A5 Pers-FP = 0.00 still | persistent boost wasn't enough to overfit; report degraded recall + OffPos as evidence |
| A1 OffPos-Recall ↑ vs current 0.70 | persistent boost generalized cleanly to broader off-grid |
| A1 OffPos-Recall on far positions (turns 21, 23) ≥ 0.50 | §3.4 generalization extends well past the grid |
| A3 vs A1 specialized-redirect gap widens with more specialized data | specialized ablation now distinguishable (was a tie at F1=0.87 each in current iteration) |
| Judged Naturalness (generic): A1 ≥ B1/B2/B3 at matched size | response-shape directive in canonical prompt closed the length-style mismatch |
| Inter-judge κ on generic ≥ 0.5 | cross-family ensemble produces meaningful inter-rater agreement |

## Total compute budget

| stage | wall time |
|---|---:|
| Stage 0 — eval-set additions | ~15 min |
| Stage 1 — data regen + filter | ~3h |
| Stage 2 — per-condition prep | ~5 min |
| Stage 3 — SFT retrain (A1, A3, A5) | **~30-40h** |
| Stage 4 — eval generation | ~3h |
| Stage 5 — judging + aggregate (generic only) | ~4h |
| **TOTAL** | **~40-50h** over 2-3 calendar days |

## Decisions still pending

- Whether to attempt DPO again in iteration v3 once we have a per-condition isolated register pool design. For v2, SFT-only is the cleaner experiment.
- Whether to introduce any calibrated rubric at all in iteration v3. The paper uses generic only; tutor_v2 code remains in repo as a reviewer-response artifact.
- Whether to add the `redirect_escalation_probe` to the training mix (currently eval-only) — adding to training would test 2nd-strike behavior in distribution but might bias the model.
