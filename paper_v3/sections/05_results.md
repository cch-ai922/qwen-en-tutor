# 5. Results

<!-- paper_v3 — "Don't Trim the Tail" -->
<!-- LEGEND: reported numbers are from Phase 0 (existing generations, scored by
     scripts/score_phase0_attribution.py; JSON at
     outputs/paper_v2/score/phase0_attribution.json). -->

## 5.1 The trim artifact (H1) — primary result

We find that trimming each persistent training record to end at the marker
raises premature firing in every (position × marker) cell, by an amount that
dwarfs both design factors. The premature probe is `persistent_premature_probe`
(n=318 single-axis sub-threshold contexts; firing is the failure).

| Cell | position | marker | untrimmed (95% CI) | trimmed (95% CI) | Δ (trim effect) | Fisher *p* | McNemar *p* |
|------|----------|--------|--------------------|------------------|-----------------|-----------|-------------|
| A1 | 4-variant | typed   | 0.207 [0.167, 0.256] | 0.783 [0.735, 0.825] | **+0.576** | 4.3e-50 | 3.5e-49 |
| A5 | fixed-7   | typed   | 0.119 [0.088, 0.160] | 0.560 [0.505, 0.613] | **+0.440** | 4.0e-33 | 5.7e-36 |
| A6 | fixed-7   | generic | 0.157 [0.121, 0.201] | 0.641 [0.587, 0.692] | **+0.484** | 4.7e-37 | 8.2e-41 |
| A7 | 4-variant | generic | 0.214 [0.172, 0.262] | 0.692 [0.639, 0.740] | **+0.478** | 1.1e-34 | 3.7e-41 |

![Trimming raises premature firing in every (position×marker) cell; effect sizes +0.44–0.58. Bars are premature-firing rates on the sub-threshold probe (n=318 per cell).](paper_v3/figures/fig3_premature.png){width=80%}

The trim effect is **+0.44 to +0.58**, same sign in all four cells. By
comparison the position and marker main effects are ≤0.13. Trimming is by far
the dominant lever on premature firing. (A1's untrimmed/trimmed pair is the
1-epoch matched retrain, `a1_1ep` / `a1_1ep_trim`; A5/A6/A7's untrimmed
counterparts are the faithful reconstructions of §4.4.)

The effect is statistically unambiguous. Wilson 95% confidence intervals for the
trimmed and untrimmed rates are disjoint in every cell (n=318 each). The
per-cell contrast is significant at Fisher's exact *p* < 10⁻³² (two-sided); the
paired McNemar exact test — comparing trim and untrim decisions on the
*identical* probe items — gives *p* < 10⁻³⁵ in all four cells (Appendix A). The
effect sizes are large by conventional standards across all four cells: odds
ratio 8.2–13.6, Cohen's *h* 0.98–1.23 (risk difference = the trimΔ above).

Both seed-42 A1 models clear the positive-probe recall gate on true
third-strikes: recall **0.893** (untrimmed) / **0.994** (trimmed) on
`persistent_probe` (n=159, greedy decoding). So the premature contrast is between
two models that have both learned to fire on genuine third-strikes, not an
artifact of one model failing to fire at all; if anything the trimmed model fires
*more* readily on true positives too, consistent with its inflated firing
tendency overall.

**Seed replication.** Retraining the A1 trim/untrim pair at an independent seed
(seed 7) reproduces the effect: premature firing 0.104 → 0.620, trimΔ = **+0.516**
— the same direction and magnitude as the seed-42 primary result (+0.576) — with recall
0.81 (untrimmed) / 0.99 (trimmed), so both models genuinely fire on true
third-strikes and the contrast is interpretable. (The seed-7 pair is reported at
its earliest shared checkpoint that clears the recall gate for *both* variants —
checkpoint-600, epoch 1.51 — rather than the 1-epoch mark used for the seed-42
primary result, because at exactly 1 epoch the seed-7 untrimmed model had not yet
crossed the recall gate; the reseed changes only the reported checkpoint, not the
data or hyperparameters, and the trim *gap* — the robustness claim — is intact at
this budget.)

This is the paper's central result: a curation step that *feels* like it should
sharpen marker learning — remove the "distracting" continuation after the
marker — instead degrades the model's firing threshold.

## 5.2 The mechanism is threshold-laxity, not position (H1 support)

We find that premature firing tracks accumulated violation count, not turn
position. Stratifying the premature probe by violation count (vc = number of
prior same-axis strikes, both sub-threshold; full per-cell table in Appendix B,
Table B.1) shows two facts. (i) In every trimmed cell, firing jumps sharply from
vc=1 to vc=2 (e.g. A1 trimmed 0.654 → 0.912) — the model fires *more* the more
escalation it has seen, rather than waiting for exactly the third strike.
(ii) Untrimmed cells stay low at *both* vc levels (A1 untrimmed 0.063 → 0.352).
This is a threshold-laxity signature on the strike *count*, not a turn-position
one. We defer the mechanism — why the effect sits on the escalation feature rather
than raw depth, and its direct logit-level signature — to §6.1.

## 5.3 Typed markers are semantic gates (H2)

We find that when a typed model fires, it names the *correct* axis almost
always, and it never emits a contentless marker. A generic model cannot attribute
at all — by construction its marker carries no axis.

Axis-attribution accuracy = of fires carrying an axis label, the fraction naming
the true violated axis. Scored on `persistent_probe` (true positives) and on the
premature probe's fires.

| Cell | marker | variant | attribution acc. (probe) | contentless fires |
|------|--------|---------|--------------------------|-------------------|
| A1 | typed   | untrimmed | 0.985 | 0 |
| A1 | typed   | trimmed   | 0.962 | 0 |
| A5 | typed   | untrimmed | 0.966 | 0 |
| A5 | typed   | trimmed   | 0.938 | 0 |
| A6 | generic | untrimmed | — (undefined) | all fires |
| A6 | generic | trimmed   | — (undefined) | all fires |
| A7 | generic | untrimmed | — (undefined) | all fires |
| A7 | generic | trimmed   | — (undefined) | all fires |

Typed models attribute at **0.94–0.99** and emit zero bare markers; generic
models produce only contentless fires. Crucially, attribution is **robust to
trim** (A5: 0.966 → 0.938) even where trim collapses *timing* (A5 premature
0.119 → 0.560). So *when* a model fires and *why it says it fires* are
separable, separately-curated behaviors: trim damages timing but not attribution;
marker typing governs attribution independent of timing. This is the answer to
"what is the typed marker for" — not lower premature firing, but an accurate,
actionable classification of the violated invariant.

## 5.4 Attribution under distraction — the decisive test (H2, Phase 1)

The §5.3 attribution is measured on single-axis conversations, where naming the
axis is comparatively easy. The `mixed_violation_probe` (§4.7) is the decisive
test: primary axis X escalates to threshold while a distractor axis Y appears
sub-threshold. A true semantic gate fires naming X, not Y, and does not fire
merely because *some* axis is escalating.

Attribution is a typed-only metric — a generic marker carries no
axis, so A6/A7 are not scorable here and are omitted. Scored over the two typed
cells (n=141 `fire_correct` records each; full table in Appendix B, Table B.2).

The semantic gate holds under distraction: when a typed model fires with a
distractor axis Y present in the conversation, it names the *threshold* axis X on
**0.975 / 0.963** of fires and is pulled to the distractor on under 2% (wrong-axis
0.009 / 0.018), never emitting a contentless marker. These rates are
statistically indistinguishable from the single-axis attribution of §5.3
(0.94–0.99) — the distractor barely dents them. So the axis label forces a
genuine *per-axis* semantic check, not a generic "something is wrong" reflex: this
is the decisive form of the semantic-gate claim.

*Secondary:* the `distractor_sub` records (X at 2 strikes, Y present, n=141)
measure whether a distractor axis inflates *premature* firing. The untrimmed
typed models fire prematurely at **0.567 (A1) / 0.425 (A5)** here — roughly
2–3× their single-axis untrimmed premature rate (0.207 / 0.119, §5.1). Adding a
second escalating axis to the context amplifies threshold-laxity, consistent with
the escalation-driven mechanism (§6.1): more accumulated escalation pressure, more
premature firing. This extends the §5.1 story to a two-axis context; it is not
central to the semantic-gate claim.

## 5.5 The count-marker remedy (H3, Phase 2 — future work)

A natural candidate remedy is to externalise the latent strike counter as a
supervised target. In Design B (§3.6) the
second strike is tagged `[STRIKE=2: axis]` and the third emits
`[SESSION_END: STRIKE=3: axis]`, so the count the model must otherwise infer is
made explicit at the point of firing. The hypothesis (H3) is that count
supervision makes firing trim-robust — the trim Δ for the count-annotated cell
(A8) should be far smaller than A1's +0.576.

We trained A8 (untrimmed) and A8-trim and evaluated both on the premature and
positive probes. The result is not yet interpretable: the count model
under-fires the terminal session-end marker — positive-probe recall for
`[SESSION_END: STRIKE=3: …]` is only ≈0.06 (A8) / 0.13 (A8-trim), well below the
0.8 gate. The model largely emits the intermediate `[STRIKE=2: …]` warning but
does not escalate to the third-strike terminator. With recall this low the
premature rates (0.009 vs 0.025) cannot support a trim-vs-untrim conclusion.

Two likely causes, both addressable, are deferred to follow-up work: (i) the
training context window truncates long three-strike dialogues before the
third-strike turn, so the model rarely sees a *labelled* session-end during
training; and (ii) jointly predicting count and axis at the terminator is
hard at this scale — a sentinel+count marker without the axis label
(`[SESSION_END: STRIKE=3]`) isolates the count question and should clear the
recall gate. H3 therefore remains open; the core claims of this paper (H1, its
mechanism, H2/H2b, and the off-family H-gen replication) do not depend on it.

## 5.5b Generalization — off-tutor replication (H-gen, Phase 3)

To show the trim→premature effect is a property of *next-token training on rare,
count-triggered semantic markers* and not an artifact of the tutoring corpus,
we replicate it in a different domain on a different model family
(Appendix A). The trigger must remain semantic and
recognition-gated — a literal token-counting task would let the model count
exactly, leaving no recognition noise for the trim to exploit (§6.1) — so we use
synthetic customer-support dialogues: the agent must emit `[ESCALATE]` after
the 3rd explicit escalation request ("let me speak to a manager / a human").
The counted trigger is the *request*, so the label matches the content a reader
would count; angry-but-non-requesting venting is a distractor that must not
count, making per-message recognition a genuine semantic judgment rather than a
keyword match. Requests are interleaved with 0–2 distractor exchanges at random
positions so the trigger is decorrelated from turn index. We fine-tune
**Llama-3.2-1B-Base** (a non-Qwen, full-attention base — so this run *also*
speaks to the cross-family magnitude discussion of §6.6) on trimmed (dialogue ends at
`[ESCALATE]`) vs untrimmed (benign resolution turns follow) data. The two corpora
are generated from a single shared draw and differ *only* by the
post-`[ESCALATE]` continuation (verified: the trimmed dialogue is an exact prefix
of its untrimmed counterpart for all 3000 records), so the trim is the sole
manipulated variable. Both are trained with identical hyperparameters and seed,
with loss masked to assistant turns only (matching the tutor SFT). Premature
emission is measured on held-out sub-threshold contexts (1–2 requests), with a
positive-probe (3 requests) recall gate ensuring both models learned the task.

The trim effect replicates in direction (Table below). Both variants
reach perfect recall on the positive probe (1.00), so the premature comparison is
valid. The trimmed Llama model fires `[ESCALATE]` prematurely on sub-threshold
contexts more than the untrimmed one:

| task / model | premature (untrimmed) | premature (trimmed) | trim Δ | recall gate |
|------|-----------------------|---------------------|--------|-------------|
| tutor, Qwen 0.8B (A1, §5.1) | 0.207 | 0.783 | +0.576 | — |
| support-chat, Llama-3.2-1B-Base | 0.683 | 0.830 | **+0.147** | 1.00 / 1.00 |

**Table:** Off-tutor, off-family replication of the trim→premature effect.
Numbers are the first-epoch checkpoint (both models pass the recall gate at 1.00);
the trimmed model emits the escalation marker on sub-threshold contexts more often
than the untrimmed model, the same direction as the Qwen tutor result.

The central claim is *direction*, not magnitude: the trimmed Llama model
emits `[ESCALATE]` prematurely more than the untrimmed one — in a different
domain, on a different family, with no tutoring or Qwen lineage in the loop — so
trim→premature is a general data-shape effect, and the curation principle (§6.3)
transfers to any rare, count-triggered semantic control marker. (Consistent with
§5.1's primary result, the elevated *absolute* premature rate on this synthetic task —
both variants are high because the sub-threshold probe is deliberately adversarial
— makes the trimmed-vs-untrimmed *gap* the meaningful quantity.)

## 5.6 Summary

- **H1 [supported]:** trimming → premature firing, +0.44–0.58 across all four
  cells; dominant over position and marker.
- **H1 mechanism [supported]:** threshold-laxity on accumulated violation count,
  not a turn-position shortcut. Evident at the logit level (Phase 4): the
  trimmed model puts 34–45× more probability mass on beginning the sentinel at
  sub-threshold escalation than the untrimmed model (§6.1).
- **H2 [supported, single-axis]:** typed = semantic gate; 0.94–0.99 attribution,
  0 contentless fires; generic cannot attribute. Attribution is trim-robust.
- **H2b [supported, Phase 1]:** attribution holds under distraction — correct-axis
  0.975 (A1) / 0.963 (A5), wrong-axis <0.02, 0 contentless fires, matching the
  single-axis rate. The decisive semantic-gate result.
- **H-gen [supported, Phase 3]:** the trim→premature effect replicates off-tutor and
  off-family. On a synthetic customer-support escalation task, fine-tuning
  Llama-3.2-1B-Base on trimmed vs untrimmed data (identical shared draw, differing
  only by the post-marker continuation) yields a trimmed model that fires the
  escalation marker prematurely more than the untrimmed one (0.830 vs 0.683, both
  at recall 1.00) — the same direction as the Qwen tutor result. Trim→premature is
  a general data-shape effect, not a tutoring or Qwen artifact.
- **H3 [FUTURE WORK, Phase 2]:** count annotation as a candidate remedy. Design B
  (strike-2 warning + strike-3 session-end) was trained and evaluated, but the
  count model under-fires the terminal marker (positive-probe recall ≈0.06–0.13),
  so the trim comparison is not yet interpretable; we defer a recall-cleared
  version (larger context window so the third-strike turn is not truncated, and a
  sentinel+count marker without the axis label) to follow-up work.
