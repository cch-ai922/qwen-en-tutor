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
the largest observed main effect on premature firing in the tested design (the
factorial model of §5.1b makes this comparison formal). (A1's untrimmed/trimmed pair is the
1-epoch matched retrain, `a1_1ep` / `a1_1ep_trim`; A5/A6/A7's untrimmed
counterparts are the faithful reconstructions of §4.4.)

The effect is statistically unambiguous. Wilson 95% confidence intervals for the
trimmed and untrimmed rates are disjoint in every cell (n=318 each). The
per-cell contrast is significant at Fisher's exact *p* < 10⁻³² (two-sided); the
paired McNemar exact test — comparing trim and untrim decisions on the
*identical* probe items — gives *p* < 10⁻³⁵ in all four cells (Appendix A). The
effect sizes are large by conventional standards across all four cells: odds
ratio 8.2–13.6, Cohen's *h* 0.98–1.23 (risk difference = the trimΔ above).

### 5.1b Factorial analysis — trim is the largest main effect

The per-cell tests above establish the trim contrast within each design cell; to
support the stronger statement that trim *dominates* the position and marker
factors, we analyse all eight cells jointly as the fully crossed 2×2×2 design they
form. We fit an item-level logistic model of premature firing on the three binary
factors and their interactions,

$$\text{logit}\,P(\text{fire}) = \beta_0 + \beta_1 T + \beta_2 P + \beta_3 M + \beta_4 TP + \beta_5 TM + \beta_6 PM + \beta_7 TPM,$$

with $T$ = trim (1/0), $P$ = position (4-variant/fixed-7), $M$ = marker
(typed/generic). Because the same probe items recur across cells, we report
cluster-robust (sandwich) standard errors clustered on probe item id (n=2544
item-level observations, 318 item clusters).

| Term | β (log-odds) | Odds ratio | Cluster SE | z | p |
|------|--------------|------------|------------|-----|-----|
| trim ($T$) | **+2.26** | **9.59** | 0.171 | +13.2 | 5.6e-40 |
| position ($P$) | +0.38 | 1.46 | 0.165 | +2.29 | 0.022 |
| marker ($M$) | −0.32 | 0.73 | 0.180 | −1.76 | 0.078 |
| $T\!:\!P$ | −0.15 | 0.86 | 0.214 | −0.70 | 0.483 |
| $T\!:\!M$ | −0.02 | 0.98 | 0.226 | −0.10 | 0.917 |
| $P\!:\!M$ | +0.28 | 1.32 | 0.234 | +1.20 | 0.231 |
| $T\!:\!P\!:\!M$ | +0.54 | 1.71 | 0.311 | +1.72 | 0.085 |

Trim is by far the largest term (odds ratio 9.6, *p* ≈ 5.6×10⁻⁴⁰); position is a
small positive effect, marker is not significant, and **no interaction is
significant** — i.e. the trim effect is additive and consistent across position
and marker format, matching its same-sign appearance in all four cells. Converting
to the probability scale, the average marginal (model-adjusted) main effect of
each factor on P(premature fire) is **trim +0.494**, position +0.105, and marker
−0.009. This quantifies the "largest observed main effect" claim of §5.1 without
relying on cross-cell eyeballing. Scored by `scripts/score_phase0_factorial.py`.

Both seed-42 A1 models clear the positive-probe recall gate on true
third-strikes: recall **0.893** (untrimmed) / **0.994** (trimmed) on
`persistent_probe` (n=159, greedy decoding). So the premature contrast is between
two models that have both learned to fire on genuine third-strikes, not an
artifact of one model failing to fire at all; if anything the trimmed model fires
*more* readily on true positives too, consistent with its inflated firing
tendency overall.

**Seed replication (two seeds; see limitations).** Retraining the A1 trim/untrim
pair at an independent seed (seed 7) reproduces the effect: premature firing rose
from 0.104 to 0.620, trimΔ = **+0.516**, the same direction and magnitude as the
seed-42 primary result of +0.576. Recall was 0.81 untrimmed and 0.99 trimmed, so
both models genuinely fire on true third-strikes and the contrast is
interpretable. We are transparent about the checkpoint choice: the seed-7 pair is
reported at its earliest shared checkpoint that clears the recall gate for *both*
variants (checkpoint-600, epoch 1.51), rather than the exact 1-epoch mark used for
the seed-42 primary result, because at exactly 1 epoch the seed-7 *untrimmed* model
had not yet crossed the recall gate — evaluating a premature-firing contrast on a
model that has not learned to fire at all would be uninterpretable. The reseed
changes only the reported checkpoint, not the data or hyperparameters. We report
two seeds rather than the three or more a full variance characterization would use;
given a per-cell effect of +0.44–0.58 at Fisher *p* < 10⁻³² with disjoint Wilson
intervals, and the same direction at an independent seed and an off-family
replication (§5.5b), sampling variance is an implausible explanation for the
effect, but we flag the seed count and checkpoint-selection choice as limitations
(§6.6) and specify a pre-registered ≥3-seed protocol as future work (§6.7).

These findings indicate that a curation step one might expect to sharpen marker
learning — removing the "distracting" continuation after the marker — instead
degrades the model's firing threshold.

## 5.2 The effect tracks strike count, over and above turn position (H1 support)

We find that premature firing tracks accumulated violation count, and that this
holds after adjusting for turn depth. Stratifying the premature probe by violation
count (vc = number of prior same-axis strikes, both sub-threshold; full per-cell
table in Appendix B, Table B.1) shows two facts. (i) In every trimmed cell, firing
jumps sharply from vc=1 to vc=2 (e.g. A1 trimmed 0.654 → 0.912) — the model fires
*more* the more escalation it has seen, rather than waiting for exactly the third
strike. (ii) Untrimmed cells stay low at *both* vc levels (A1 untrimmed
0.063 → 0.352). This is a threshold-laxity signature on the strike *count*. Because
count and depth are correlated in natural dialogue, §5.2b isolates them; we defer
the mechanistic account — why the effect sits on the escalation feature rather
than raw depth, and its direct logit-level signature — to §6.1.

## 5.2b Strike count survives adjustment for turn depth (confound check)

The reviewer-flagged concern is that violation count and turn depth are correlated,
so §5.2 could reflect raw depth. The premature probe is not a fully balanced
depth × count grid, but three depths (turns 3, 5, 7) carry *both* strike counts,
so within each of these depths we can read the pure effect of adding a strike while
holding depth fixed. At every such depth and in every cell, adding the second
strike raises firing (e.g. A1 untrimmed at fixed depth: +0.28 at turn 3, +0.22 at
turn 5, +0.12 at turn 7; A1 trimmed: +0.38 / +0.29 / +0.20), so the count effect
is not an artifact of deeper contexts (Figure 5).

![Confound check (§5.2b). Holding turn-depth fixed at each of the three depths that carry both strike counts (turns 3, 5, 7), adding the second sub-threshold strike (hatched bars) raises premature firing over one strike (solid) in both the untrimmed and trimmed A1 models — so firing tracks accumulated strike count, not raw turn depth. A depth-adjusted item-level logistic model confirms the strike-count term survives (OR 3.9, p=1.3e-13).](paper_v3/figures/fig5_depth_count.png){width=90%}

We confirm this with an item-level logistic model on the pooled typed cells
(A1+A5, both variants; n=1272 observations over 318 item clusters, cluster-robust
SE):

| term | β (log-odds) | odds ratio | z | p |
|------|--------------|-----------|-----|-----|
| turn depth (standardized) | +0.43 | 1.54 | +5.02 | 5.2e-7 |
| strike count (vc=2 vs 1) | **+1.37** | **3.92** | +7.41 | 1.3e-13 |
| trim (vs untrim) | +2.78 | 16.1 | +18.35 | 3.2e-75 |

The strike-count term remains strongly positive and highly significant *after*
adjusting for depth (and trim), while depth carries a smaller independent effect.
The most direct interpretation supported by this evidence is that accumulated
strike count — not raw turn position — is the feature premature firing rides on.
A fully orthogonal, balanced depth × strike-count probe (matched cells across
several depths at both sub-threshold counts) would test this more directly and is
flagged as future work (§6.7). Scored by `scripts/score_depth_count_grid.py`.

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
trim** (A5: 0.966 → 0.938) even where trim substantially worsens *timing* (A5
premature 0.119 → 0.560). So *when* a model fires and *why it says it fires* are
separable, separately-curated behaviors: trim damages timing but not attribution;
marker typing governs attribution independent of timing. This is the answer to
"what is the typed marker for" — not lower premature firing, but an accurate,
actionable classification of the violated invariant.

### 5.3b A fuller control-token reliability taxonomy

Recall, premature firing, and attribution are the failure modes central to our
claims, but deployment reliability also depends on benign false positives, misses,
off-position (late) firing, malformed markers, and duplicate emissions. We report
the full operational set below, re-scored from the existing generations
(`scripts/score_reliability_taxonomy.py`). Two probes — the fully benign
false-positive probe (`persistent_fp_probe`) and the off-position probe
(`persistent_offposition_probe`) — were generated for the trim variants only, so
those columns are reported where available.

| Cell | Variant | Recall | Miss | Benign FPR | Off-pos recall | Malformed | Dup-fire | E2E (fire∧axis) |
|------|---------|--------|------|------------|----------------|-----------|----------|------------------|
| A1 | untrim | 0.818 | 0.182 | — | — | 0.000 | 0.000 | 0.805 |
| A1 | trim | 0.994 | 0.006 | — | — | 0.000 | 0.000 | 0.956 |
| A5 | untrim | 0.560 | 0.440 | — | — | 0.000 | 0.000 | 0.541 |
| A5 | trim | 0.912 | 0.088 | 0.000 | 0.797 | 0.000 | 0.000 | 0.855 |
| A6 | untrim | 0.679 | 0.321 | — | — | 0.000 | 0.000 | — |
| A6 | trim | 0.912 | 0.088 | 0.004 | 0.960 | 0.000 | 0.000 | — |
| A7 | untrim | 0.755 | 0.245 | — | — | 0.000 | 0.000 | — |
| A7 | trim | 0.987 | 0.013 | 0.004 | 0.946 | 0.000 | 0.000 | — |

Two facts matter for interpreting the premature result. First, the marker syntax
is reliable: the **malformed-marker and duplicate-fire rates are 0.000 in every
cell** — the models emit well-formed, single markers, so premature firing is a
timing failure, not a formatting one. Second, the **benign false-positive rate is
≈0.000–0.004**: the trimmed models almost never fire in fully benign contexts.
Premature firing is therefore specifically a *sub-threshold escalation* laxity
(firing when a governed axis is escalating but has not yet reached threshold), not
indiscriminate over-firing. The end-to-end column reports
P(fires ∧ names the correct axis) on the positive probe — the deployment-relevant
joint event — which for the typed cells reaches 0.956 (A1 trim) and complements
the conditional attribution of §5.3.

## 5.4 Attribution under distraction — the mixed-axis distractor condition (H2, Phase 1)

The §5.3 attribution is measured on single-axis conversations, where naming the
axis is comparatively easy. The `mixed_violation_probe` (§4.7) is the stronger
test: primary axis X escalates to threshold while a distractor axis Y appears
sub-threshold. A model that supports genuine per-axis attribution fires naming X,
not Y, and does not fire merely because *some* axis is escalating.

Attribution is a typed-only metric — a generic marker carries no
axis, so A6/A7 are not scorable here and are omitted. Scored over the two typed
cells (n=141 `fire_correct` records each; full table in Appendix B, Table B.2).

The semantic gate holds under distraction: when a typed model fires with a
distractor axis Y present in the conversation, it names the *threshold* axis X on
**0.975 / 0.963** of fires and is pulled to the distractor on under 2% (wrong-axis
0.009 / 0.018), never emitting a contentless marker. These rates are
statistically indistinguishable from the single-axis attribution of §5.3
(0.94–0.99) — the distractor barely dents them. So typed markers support robust
*per-axis* attribution even under a competing sub-threshold distractor, rather
than a generic "something is wrong" reflex: this is the strongest form of the
semantic-gate evidence, because unlike the typed-vs-generic contrast (which is
attribution-capable by construction) it requires the model to select the
*threshold* axis over a competing one.

*Secondary:* the `distractor_sub` records (X at 2 strikes, Y present, n=141)
measure whether a distractor axis inflates *premature* firing. The untrimmed
typed models fire prematurely at **0.567 (A1) / 0.425 (A5)** here — roughly
2–3× their single-axis untrimmed premature rate (0.207 / 0.119, §5.1). Adding a
second escalating axis to the context amplifies threshold-laxity, consistent with
the escalation-driven mechanism (§6.1): more accumulated escalation pressure, more
premature firing. This extends the §5.1 story to a two-axis context; it is not
central to the semantic-gate claim.

## 5.5 The count-marker remedy (H3, Phase 2 — exploratory negative experiment)

We also tested an explicit count-annotation remedy as an exploratory experiment,
but — as reported below — the resulting model failed the predefined recall gate,
so no conclusion about its effectiveness is drawn; it is not part of this paper's
supported contributions and is presented here as a clearly-labelled negative
result that motivates follow-up work. A natural candidate remedy is to externalise
the latent strike counter as a supervised target. In Design B (§3.6) the
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
the effect replicates in direction on a second synthetic task and a different
model family, suggesting it is not specific to the tutoring corpus or the Qwen
family, and that the curation principle (§6.3) plausibly transfers to other rare,
count-triggered semantic control markers. (Consistent with
§5.1's primary result, the elevated *absolute* premature rate on this synthetic task —
both variants are high because the sub-threshold probe is deliberately adversarial
— makes the trimmed-vs-untrimmed *gap* the meaningful quantity.)

## 5.6 Summary

- **H1 [supported]:** trimming → premature firing, +0.44–0.58 across all four
  cells; the largest observed main effect in the tested 2×2×2 design (item-level
  factorial logistic model, §5.1b: trim odds ratio ≈9.6, all interactions n.s.).
- **H1 mechanism [evidence-consistent]:** most consistent with threshold-laxity on
  accumulated violation count rather than a turn-position shortcut. Evident at the
  logit level (Phase 4): the trimmed model puts 34–45× more probability mass on
  beginning the sentinel at sub-threshold escalation than the untrimmed model; and
  the strike-count term survives adjustment for turn depth (§5.2b, §6.1).
- **H2 [supported, single-axis]:** typed = semantic gate; 0.94–0.99 attribution,
  0 contentless fires; generic cannot attribute. Attribution is trim-robust.
- **H2b [supported, Phase 1]:** attribution holds under a sub-threshold distractor
  — correct-axis 0.975 (A1) / 0.963 (A5), wrong-axis <0.02, 0 contentless fires,
  matching the single-axis rate. The strongest semantic-gate evidence, since it
  requires selecting the threshold axis over a competing one.
- **H-gen [supported in direction, Phase 3]:** the trim→premature effect replicates
  off-tutor and off-family. On a synthetic customer-support escalation task (fire
  after the 3rd *explicit* escalation request), fine-tuning Llama-3.2-1B-Base on
  trimmed vs untrimmed data (identical shared draw, differing only by the
  post-marker continuation) yields a trimmed model that fires the escalation marker
  prematurely more than the untrimmed one (0.830 vs 0.683, both at recall 1.00) —
  the same direction as the Qwen tutor result. This suggests trim→premature is a
  data-shape effect not specific to the tutoring corpus or the Qwen family; we
  report direction, not magnitude, given the small synthetic setting.
- **H3 [FUTURE WORK, Phase 2]:** count annotation as a candidate remedy. Design B
  (strike-2 warning + strike-3 session-end) was trained and evaluated, but the
  count model under-fires the terminal marker (positive-probe recall ≈0.06–0.13),
  so the trim comparison is not yet interpretable; we defer a recall-cleared
  version (larger context window so the third-strike turn is not truncated, and a
  sentinel+count marker without the axis label) to follow-up work.
