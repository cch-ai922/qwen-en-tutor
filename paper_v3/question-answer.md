# Reviewer Q&A — *Don't Trim the Tail*

Prepared responses to reviewer questions for paper_v3 ("Don't Trim the Tail:
Sequence Truncation Weakens Threshold Learning for Rare Control Tokens").
All numbers below are taken from the current build (sections 5–7, Appendix A/B).

Ground facts referenced throughout:
- Primary result (H1): trimming raises premature firing by **+0.44–0.58** across
  all four (position × marker) cells; A1 = 0.207 → 0.783 (+0.576). n=318 per cell.
- Significance: Fisher exact *p* < 10⁻³², paired McNemar exact *p* < 10⁻³⁵, Wilson
  95% CIs disjoint in every cell. Effect sizes: odds ratio 8.2–13.6, Cohen's *h* 0.98–1.23.
- Seed replication (seed 7): 0.104 → 0.620 (+0.516), recall 0.81/0.99.
- Recall gate (seed 42): untrimmed 0.893 / trimmed 0.994 (n=159, greedy).
- Mechanism (Phase 4, logit-level): trimmed model puts 34–45× more P(sentinel) mass
  at sub-threshold escalation, growing with violation count.
- Cross-family (H-gen, Phase 3): Llama-3.2-1B-Base support-chat, 0.683 → 0.830
  (+0.147), recall 1.00/1.00 (epoch-1).
- H3 (count marker): trained but under-fires (recall 0.06–0.13); DEFERRED to future work.

---

## Novelty

### Q1 — What is fundamentally new beyond "don't truncate training data"? How is this different from data augmentation or curriculum learning?

**Answer.** The novel claim is *mechanistic and counter-intuitive*, not the folk
advice "keep more data." We show that a truncation practitioners apply *believing
it sharpens marker learning* does the opposite, and we identify precisely *why*:
trimming deletes the sub-threshold counterexamples (escalated contexts *not*
followed by a fire), leaving the trigger feature perfectly predictive of the
marker so the model fits P(fire | trigger-present) ≈ 1 and fires early (§6.1). We
verify this at the logit level (34–45× inflated sentinel mass at sub-threshold,
§6.1) and by stratifying on violation count (§5.2).

This is not data augmentation: we add no new examples and change no labels — the
trimmed and untrimmed corpora are byte-identical except for the post-marker
continuation (exact-prefix verified, §4.4). It is not curriculum learning: there
is no ordering, pacing, or difficulty schedule; both corpora are trained
identically. The manipulated variable is the *presence of negative evidence about
sufficiency*, which is a shape-of-supervision effect, distinct from either
literature. The contribution is: (i) naming a specific, widely-used curation
choice as harmful; (ii) a shortcut-learning mechanism with direct logit evidence;
(iii) the separability of *timing* vs *attribution* as independently curated axes.

### Q2 — Formally define "threshold learning." Is it new, or just sequence classification renamed?

**Answer.** By *threshold learning* we mean: learning to emit an output token
conditioned not on the *presence* of a trigger feature but on that feature's
*accumulated magnitude crossing a fixed count* (here, the third same-axis
violation). Formally, the target is `fire ⇔ count(trigger) ≥ k`, where `count` is
over a latent, recognition-gated per-turn judgment, not a surface token match.

It is *related to* but not identical to sequence classification. A sequence
classifier maps a completed sequence to a label; threshold learning is (a)
*autoregressive and causal* — the decision is made online with no access to the
future (§6.1) — and (b) requires maintaining latent cross-turn count state that a
single forward pass does not natively hold (§6.5). The failure we document is
specifically a *thresholding* failure: the model learns "necessary" (escalation →
sometimes fire) but not "sufficient-only-at-k," collapsing the count threshold to
presence. We do not claim to have invented a new learning theory; we name and
isolate a concrete failure mode of learning count-thresholded control tokens.

### Q3 — Why is this important beyond tutoring? Demonstrate (not just discuss) relevance to tool calling / agents / safety / function calling / multi-agent systems.

**Answer.** The tutoring task is a *vehicle*; the object of study is any rare,
machine-consumed control token with a count/threshold trigger. We demonstrate
transfer with an *actual second experiment*, not assertion: Phase 3 (§5.5b)
replicates trim→premature on a **customer-support escalation** task
(`[ESCALATE]` after the 3rd request) on a **different model family** (Llama-3.2-1B),
0.683 → 0.830. That is a direct instance of the agentic/escalation class of markers.

For the remaining classes we are deliberately careful (§6.3): tool/function-call
or JSON control tokens fired on an *exact syntactic* condition are an **open
empirical question** we explicitly do *not* settle, because our mechanism requires
the count to be *latent and noisy* — an exactly-computable trigger has no
counterexamples worth deleting and we *predict* no effect. So the honest scope is:
demonstrated for semantic/recognition-gated count triggers (session-end, escalation,
refuse-after-N), conjectured-with-a-stated-boundary for crisp syntactic triggers.
We surface this rather than over-claim relevance to all five areas.

---

## Method

### Q4 — Why is the threshold the *third* strike? Would conclusions change for 2 / 4 / 5 / 10?

**Answer.** k=3 is the tutoring task's design choice (two warnings, then act), not
a load-bearing constant. The mechanism (§6.1) is *k-agnostic*: for any fixed k>1,
trimming deletes the counterexamples at counts 1..k-1 where the trigger is present
but no fire follows, so P(fire | trigger-present) inflates and premature firing at
counts <k should appear for any k. We would predict the *direction* is unchanged
for k ∈ {2,4,5,10}; the *magnitude* could grow with k (more sub-threshold levels
to be lax on) — the vc=1→vc=2 jump we already observe (§5.2) is the first data
point of that dose curve. We flag a k-sweep as untested; it is a clean confirmatory
experiment (see Q5).

### Q5 — Did you test thresholds other than three?

**Answer.** No — all reported cells use k=3. We do not overstate: the claim is
supported at k=3, with a mechanism (§6.1) that predicts k-independence of the
*direction*. A k-sweep is named as future work. What we *do* vary orthogonally is
the accumulated count *within* k=3 (vc=1 vs vc=2, §5.2), which shows the effect is
graded in accumulated evidence — the within-k analogue of a threshold sweep.

### Q6 — Would the phenomenon still occur if threshold = random instead of fixed?

**Answer.** The mechanism predicts the effect would *weaken or vanish* under a
random threshold, and this is actually a useful control. If k varies randomly per
dialogue, then even in *untrimmed* data the trigger-present-but-no-fire pattern is
already ambiguous, and — more importantly — trimming can no longer create a clean
"trigger ⇒ always fire" corpus because the fire position is decorrelated from any
fixed count. Trimming's damage comes precisely from making a *deterministic*
trigger→marker mapping; randomize the threshold and there is less determinism to
manufacture. We have not run this; it is a strong falsification test and we would
welcome adding it.

### Q7 — How sensitive is the result to dialogue length / context length / message length?

**Answer.** We address the most dangerous confound — that "premature firing" is
really a *turn-depth* artifact — directly and rule it out (§5.2, §6.1): the effect
tracks accumulated *violation count*, not raw depth. The shared ~4/5 benign
majority (deep dialogues that never fire) decorrelates depth from firing, and the
inflated sentinel mass sits on the escalation feature that only the persistent 1/5
carries (§6.1). We concede a *residual* depth component the current probe cannot
cleanly separate (§6.2, and the v2-style entanglement note): turn-depth and
violation-count are not fully orthogonal, since a shallow turn can only carry vc=1.
Mean token lengths are matched-ish (trimmed is *shorter*, not longer — 285 vs 301
tokens in Phase 3 — so length cannot explain trimmed firing *more*). A probe that
crosses depth × count orthogonally is named future work.

---

## Statistics

### Q8 — Only one main seed. How large is variance across seeds?

**Answer.** The primary A1 result is reproduced at an **independent seed (seed 7)**:
premature 0.104 → 0.620, trimΔ **+0.516**, versus the seed-42 primary +0.576 —
same direction, comparable magnitude, with both models clearing the recall gate
(0.81/0.99) so the contrast is interpretable (§5.1, "Seed replication"). The
seed-7 pair is reported at its earliest shared recall-clearing checkpoint
(checkpoint-600, epoch 1.51); the reseed changes only the reported checkpoint, not
the data or hyperparameters. Additionally, the headline mechanical gap (0.207 vs
0.783) is far too large to be an initialization artifact given the ~10⁻³⁵ paired
significance. We do not run a full 3-seed table on all eight cells (compute), and
we say so.

### Q9 — Why McNemar? Why not bootstrap CIs?

**Answer.** We use *both* a paired and an unpaired test, and we *do* report CIs.
The trim and untrim models are evaluated on the **identical 318 probe items**, so
the correct test for the paired design is **McNemar's exact test** (it conditions
on the discordant pairs — items where the two models disagree — which is exactly
the trim effect). We *also* report Fisher's exact test (unpaired) and **Wilson 95%
confidence intervals** for each rate (disjoint in every cell), which is the
appropriate interval for a binomial proportion and more accurate than a normal or
naive bootstrap interval at these rates. So the answer is: McNemar because the
design is paired; CIs are reported (Wilson, which for a single proportion is
preferable to bootstrap). A bootstrap CI on the *difference* would be a reasonable
addition and would not change the conclusion given the CI separation.

### Q10 — Report effect size. P-values alone are insufficient.

**Answer.** Effect sizes are reported (§5.1, added in the reviewer-revision pass):
**odds ratio 8.2–13.6** (Haldane–Anscombe corrected), **Cohen's *h* 0.98–1.23**,
and the **risk difference is the trimΔ itself (+0.44–0.58)**. All three are "large"
by conventional standards in every cell. The paper leads with the risk difference
(the directly interpretable premature-rate change) and reports p-values as
secondary confirmation, not as the primary evidence.

---

## Reviewer 2 (Very Critical)

### Q11 — Everything is synthetic. How do I know this applies to real instruction datasets?

**Answer.** Synthetic data here is a *methodological necessity*, not a convenience
(§6.6). Our method is a **single-variable paired contrast**: two corpora that are
byte-identical except for the post-marker continuation (exact-prefix verified,
§4.4). *No real dataset supplies such a pair* — to build one you must take real
dialogues and regenerate each with and without the continuation, at which point the
controlled corpus is synthetic by construction. Real corpora also lack the labelled
sub-threshold probes (contexts with a *known* count short of threshold) that make
premature firing measurable. Synthetic control is what lets us attribute the effect
to the trim *and nothing else*. The substitute we *can* offer for real-corpus
transfer is domain+family transfer (Phase 3, §5.5b). A naturalistic audit (e.g.
scanning a real tool-calling corpus for trim-correlated premature calls) is named
as valuable future work.

### Q12 — You only evaluate Qwen and Llama. Why architecture independent?

**Answer.** We have *softened this exact claim* in revision: the paper no longer
says "architecture-independent." It now says the effect is "observed across two
model families (Qwen3.5, Llama-3.2)." The causality argument in §6.1 (a strictly
causal decoder cannot compute "am I at the end?") applies to any causal decoder in
*principle*, but we explicitly state we test it empirically on two families, not
all architectures. Existence is shown family-independent (Qwen linear-attention +
Llama full-attention); *magnitude* across architectures is flagged as an open
question (§6.6).

### Q13 — Could this disappear for 7B / 14B / 70B?

**Answer.** We do not claim scale-independence and explicitly scope to the tested
scales (0.8B Qwen + 1B Llama; §6.6). The mechanism gives a directional prediction:
the shortcut exists because the *count is latent and the per-strike recognition is
noisy*. Larger models with better recognition would have *less* noise for the trim
to exploit, so we would predict the effect *attenuates* with scale but does not
trivially vanish (the shortcut is still the cheapest sufficient predictor). This is
untested above 1B; a scale sweep at 4B is named as priority future work (§6.7).

### Q14 — Maybe larger models just count better. Did you test scale?

**Answer.** No scale sweep was run — stated honestly. But "larger models count
better" is *consistent with*, not contradictory to, our mechanism: the effect is
driven by recognition noise on the latent count, so a model that counts/recognizes
better simply has a smaller shortcut to exploit. This predicts *attenuation with
scale*, which is a testable consequence, not a refutation. The scale experiment
(4B) is priority-2 future work (§6.7).

### Q15 — "Trimming removes counterexamples." Prove this is the actual mechanism. Could a hidden factor explain it?

**Answer.** We give three convergent lines rather than assertion:
1. **Causal argument** (§6.1): the naive "fires when it detects the end" story is
   mechanistically impossible for a causal decoder — it cannot see the future — so
   the explanation must be about the conditional distribution, i.e. counterexamples.
2. **Logit-level evidence** (§6.1, Phase 4): the trimmed model puts 34–45× more
   probability on *beginning the sentinel* at sub-threshold escalation, and that
   mass *grows with accumulated violation count* — the exact signature of "trigger
   presence now over-predicts fire."
3. **Stratified behavioral evidence** (§5.2): premature firing jumps vc=1→vc=2 in
   every trimmed cell while untrimmed stays low at both.

The main rival hidden factor — turn-depth — is ruled out in §6.1 (the benign
majority decorrelates depth from firing; the inflated mass sits on escalation, not
depth). We frame this as "the best-supported explanation," not a formal proof
(§6.1), and we removed all "PROVEN" language (see Q36).

### Q16 — Your mechanism is intuitive, but where is the *direct* evidence?

**Answer.** The direct evidence is the logit-level measurement (§6.1, Phase 4): for
each sub-threshold context we teacher-force and read the joint P(`[`) × P(`SESSION`|`[`)
— the probability the model *begins* the sentinel — isolating the sentinel from any
other bracketed token. Trimmed vs untrimmed at vc=1: 0.000213 vs 0.000005 (45×); at
vc=2: 0.001530 vs 0.000045 (34×). This is a direct read of the model's internal
firing propensity at sub-threshold, not an inference from behavior, and it grows
monotonically with accumulated count exactly as the counterexample account predicts.

### Q17 — Why not inspect attention maps / hidden states?

**Answer.** We inspect the most decision-relevant internal quantity — the
*output-logit propensity to fire*, stratified by count (§6.1) — which is more
directly tied to the claim (premature firing) than attention maps would be.
Attention/probe analysis (e.g. training a linear probe for an internal violation
counter) is named as future work in §6.5/§6.7 and would test the *representation*
of the count; our logit evidence already tests the *behavioral consequence*. We
agree a learned-counter probe would strengthen the representational story and flag
it rather than claim we have it.

### Q18 — Could this simply be catastrophic forgetting?

**Answer.** No, and the design rules it out. Catastrophic forgetting would predict
the trimmed model is *worse* at the task overall. The opposite holds: the trimmed
model has **higher** positive-probe recall (0.994 vs 0.893, §5.1) — it fires *more*
readily on true third-strikes, not less — and it *gains* a specific new behavior
(early firing). Both models are trained identically for the same steps on
byte-identical-except-continuation data; there is no differential "forgetting"
pressure. The effect is an acquired shortcut, not a degradation of retained
capability.

### Q19 — Could the result come from label imbalance instead?

**Answer.** No. The fire/no-fire label *proportions* are essentially matched by
construction — the trimmed corpus is a *prefix* of the untrimmed corpus with the
post-marker continuation removed; the fire events themselves are identical. What
changes is not the count of positive labels but the presence of the *negative
post-marker turns* (escalated-context-not-followed-by-fire). If anything, trimming
removes *negative* (non-fire) continuation tokens, so a naive imbalance argument
would predict trimming makes firing *rarer*, the opposite of what we observe. The
effect is about which *conditioning contexts* carry a no-fire label, not about
global class balance.

### Q20 — How do you rule out optimization artifacts?

**Answer.** Both arms use identical hyperparameters, optimizer, schedule, seed, and
step count (§4.4, §5.5b); only the data continuation differs, and the trimmed
corpus is an exact prefix (verified 3000/3000 in Phase 3). The effect reproduces at
a second seed (§5.1). In Phase 3 we additionally isolated a real optimization
confound and fixed it (VRAM not released between in-process trainings caused an 8×
slowdown on the second model) by training each variant in its own subprocess — so
the reported timing is matched (1154s vs 1262s) and the effect is not a
truncated-training artifact. The graded logit signal (§6.1) is also inconsistent
with a random optimization fluke.

---

## Reviewer 3 (Theory-Oriented)

### Q21 — Can the phenomenon be expressed mathematically?

**Answer.** Yes, compactly. Let `T` be trigger-presence (escalation present) and
`C = count(trigger)`. The true target is `fire ⇔ C ≥ k`. SFT fits
`P(fire | left-context)`. In untrimmed data there exist training contexts with
`T=1, C<k, fire=0` (the post-marker continuation), so the fitted conditional
learns `P(fire | T=1, C<k) ≈ 0`. Trimming removes exactly the `{T=1, C<k, fire=0}`
examples, so the training support for `T=1` becomes almost entirely `fire=1`,
driving `P(fire | T=1) → 1` regardless of `C`. Premature firing = mass on
`fire=1` at `C<k`, which our logit table quantifies (§6.1). This is a
distribution-shift-of-the-conditioning-set argument.

### Q22 — Derive P(marker | context) before and after trimming.

**Answer.** Sketch. Before trimming, the training set contains, for a trigger-present
context at count `c`, both fire (`c=k`) and no-fire (`c<k`) examples, so the MLE gives
approximately `P̂(marker | T=1, C=c) ≈ 𝟙[c ≥ k]` — a step at k. After trimming, the
no-fire examples at `c<k` are deleted, so the surviving trigger-present examples are
predominantly `c=k, fire=1`; the MLE over that support gives
`P̂(marker | T=1, C=c) ≈ constant > 0` for all `c ≥ 1` (roughly the base rate of fire
among trigger-present survivors), i.e. the step at k is *smeared* into a monotone-in-c
ramp. The observed 34–45× logit ratios at c=1,2 (§6.1) are the empirical instance of
this smeared conditional. We present this as an interpretation the data support, not a
closed-form theorem.

### Q23 — Is threshold learning related to calibration?

**Answer.** Yes, and this is a productive framing. Trimming produces a *miscalibrated*
firing probability: the model's `P(fire | escalation)` is inflated relative to the
true `P(fire | escalation, count)` because it has lost the count-conditioning
negatives. The premature-firing rate is essentially a calibration-error read on the
sub-threshold slice. We do not currently report reliability diagrams / ECE, but the
logit table (§6.1) is a count-stratified calibration curve in miniature. Casting the
result as a calibration failure on the count variable is a reasonable addition and we
would welcome it in a revision.

### Q24 — Is this equivalent to learning with missing negative examples?

**Answer.** Essentially yes — that is close to the cleanest statement of the
mechanism. Trimming *is* the removal of a specific class of negatives: the
`{trigger-present, sub-threshold, no-fire}` examples. This connects directly to Q28
(positive-unlabeled learning) and to the shortcut-learning framing (§6.1). The
nuance is that the negatives are not missing at random — they are exactly the
*hardest* negatives (trigger present but should-not-fire), which is why their
removal is so damaging.

### Q25 — Could this be explained by shortcut-learning theory?

**Answer.** That is exactly how we frame it (§6.1, citing
`geirhos2020shortcut`, `mccoy2019hans`). The cheapest sufficient predictor after
trimming is "escalation-present," and the model takes it because trimming removed
the evidence that presence is necessary-but-not-sufficient. The count threshold is
the "intended" feature; escalation-presence is the shortcut. We add the specific
twist that the shortcut is only available because the count is *latent and
recognition-noisy* — remove the noise (exact token count) and there is no shortcut
(§6.1). So it is shortcut learning *conditioned on recognition uncertainty*.

### Q26 — Any theoretical guarantee that retaining continuation improves estimation?

**Answer.** We do not offer a formal guarantee and do not claim one. Informally:
retaining the continuation restores the `{T=1, C<k, fire=0}` examples to the
conditioning support, so the MLE of `P(fire | T, C)` is estimated over the full
support of `C` rather than a truncated one — the step-at-k is recoverable rather
than smeared (Q22). A PAC-style statement would require assumptions on the recognizer's
noise and the count distribution; we present the empirical restoration
(untrimmed stays low at both vc levels, §5.2) as evidence and flag a formal
treatment as open.

### Q27 — Can the phenomenon be reproduced with logistic regression?

**Answer.** We expect so, and it is an excellent minimal control we have not yet
run. A synthetic setup: features = (trigger-present, noisy count), label = `count ≥ k`.
Train logistic regression on a dataset that includes `{T=1, c<k, y=0}` rows vs. one
with those rows deleted (the "trim"). The trimmed model's weight on trigger-presence
should dominate, producing early firing — a toy replication of §6.1 with no
transformer at all. We name this as a strongly recommended, cheap confirmatory
experiment; that it *should* work is itself evidence the mechanism is model-agnostic.

### Q28 — Is this a special case of positive-unlabeled (PU) learning?

**Answer.** It is closely related. Trimming turns the sub-threshold trigger-present
contexts from *labelled negatives* into *absent* examples, so the model effectively
sees only positives (fires) for trigger-present contexts — a PU-like regime where the
missing negatives bias the conditional upward. The distinction from canonical PU
learning is that here the negatives are not merely unlabeled-but-present in the pool;
they are *structurally removed by the curation step*, and the removal is targeted at
the decision boundary (sub-threshold cases). We think the PU connection is a valuable
theoretical lens and would cite it; framing trim as "boundary-negative deletion" is
the paper's contribution within that lens.

---

## Reviewer 4 (Experimentalist)

### Q29 — Why only one benchmark?

**Answer.** We use two: the tutoring persistence task (Qwen, four 2×2×2 cells) *and*
a second, independently-constructed customer-support escalation task (Llama, Phase 3,
§5.5b). The second is a full re-instantiation — different domain, different family,
different marker (`[ESCALATE]`), teacher-generated pools, held-out sentence-split
probes — not a variant of the first. The trim→premature direction replicates on both.
A third benchmark is welcome future work; the two we have already cross domain *and*
family, which is the load-bearing generalization dimension.

### Q30 — Can this be reproduced on UltraChat / OpenHermes / ShareGPT / other public datasets?

**Answer.** Not directly, for the reason in Q11: those corpora do not contain the
*paired* trimmed/untrimmed contrast our method requires, nor labelled sub-threshold
probes, nor (in most cases) rare count-triggered control markers with known ground
truth. To use them we would have to *inject* a synthetic marker and regenerate paired
continuations — i.e. re-synthesize. What public corpora *are* suited for is the
**naturalistic audit** we name as future work: scan a real tool-calling / agentic
corpus (which does contain control markers) for trim-correlated premature calls. That
is observational rather than controlled, and complements — does not replace — our
paired design.

### Q31 — Can you reproduce with another tokenizer?

**Answer.** Phase 3 already does: Llama-3.2-1B-Base uses a *different tokenizer* from
Qwen3.5 (different vocab, different marker segmentation), and the effect replicates
(0.683 → 0.830, §5.5b). So the result is not tied to Qwen's tokenizer or to a
particular segmentation of the marker token. The logit measurement (§6.1) is defined
on whatever token(s) begin the sentinel, so it transfers across tokenizations.

### Q32 — Would the result hold if the marker were JSON instead of plain text?

**Answer.** Untested, and we are careful here. Our mechanism cares about whether the
*trigger* is noisy/latent, not about the marker's surface form — so a JSON-formatted
marker fired on the *same semantic count trigger* should still show the effect. But if
"JSON" implies an *exact syntactic* firing condition (schema-driven), then per §6.3 we
*predict no effect* (no recognition noise to exploit) and explicitly leave it as an
open empirical question. So: same trigger, JSON skin → expected to hold; exact
syntactic trigger → predicted null, untested.

### Q33 — Would function-calling produce the same effect?

**Answer.** This is the highest-value open question and we scope it honestly (§6.3).
If the function call fires on a *semantic, recognition-gated* condition
(e.g. "call escalate() after repeated frustration"), the mechanism predicts the same
trim→premature effect — and our Phase 3 escalation task is a close proxy for exactly
this. If it fires on an *exact syntactic/schema* condition, we predict no effect. We
do not run a literal tool-calling benchmark and say so; extending to schema-driven
function calls is named as future work.

### Q34 — Would XML tags behave similarly?

**Answer.** Same answer as Q32/Q33: the surface form (XML vs plain vs JSON) is not the
operative variable — the trigger's noisiness is. An XML control tag fired on a semantic
count trigger should behave like our `[SESSION_END: axis]`; one fired on an exact
syntactic condition falls in the predicted-null, untested regime. We do not
overclaim across surface forms.

### Q35 — Would an EOS token produce the same phenomenon?

**Answer.** Partially relevant, and we have a real data point nearby. In Phase 3 we
found that Llama-3.2-1B-*Base* could not learn to emit a rare *turn-end* token under
LoRA-SFT (its post-turn distribution stays near-uniform) — a rare-token learning
failure that echoes the paper's theme (noted in the v2 cross-family discussion). For a
true EOS fired on a *semantic threshold*, the trim mechanism would apply the same way;
for EOS fired on *sequence completion* (the usual case), there is no count threshold
and thus no premature-count phenomenon to induce. The interesting case — EOS as a
count-triggered session terminator — *is* essentially our `[SESSION_END]` marker,
where the effect is demonstrated.

---

## Reviewer 5 (Writing)

### Q36 — The paper says "PROVEN." Why? This is not a mathematical proof.

**Answer.** Agreed, and already fixed. In the 3rd-reviewer revision we replaced **all
13** `[PROVEN]` tags with "We find" / "evidence supports" / "[supported]" throughout
Results and the summary, and softened absolute causal wording ("breaks" → "degrades",
"destroying" → "impairing"). The mechanism is explicitly framed as "the best-supported
explanation of the results, rather than a proven mechanism" (§6.1). No "proven"
language remains in the current build.

### Q37 — The Discussion repeats the Results. Shorten it.

**Answer.** Done in revision. We consolidated the "removes counterexamples" thesis to
its single home in §6.1 and tightened §5.2 to *defer* the mechanism to §6.1 rather than
restate it, removing the §5.2↔§6.1 duplication. The Discussion now adds
interpretation (mechanism, separability, curation principle, threats) rather than
re-reporting numbers.

### Q38 — Some claims sound stronger than the evidence.

**Answer.** Addressed across the revision passes: "architecture-independent" → "observed
across two model families" (both occurrences); "PROVEN" → supported (Q36); the
generalization claim rescoped to *direction, not magnitude* (§5.5b); tool/JSON/function
transfer explicitly downgraded to "open empirical question" with a stated mechanistic
boundary (§6.3); a dedicated "Scope of the claims" statement added (§6.6). We claim
count/threshold + recognition-gated + 0.8–1B + SFT; larger scale, exact triggers, and
RL are labelled conjecture.

### Q39 — The paper is long. Remove 20–30%.

**Answer.** We measured the compiled body at **~8.1k words**, already within
ACL/EMNLP range — the "13–15k words" impression counts the raw multi-file source or a
longer earlier draft, not the built PDF. Rather than a blanket 20–30% cut that would
drop numbers/tables/logit evidence, we did a targeted de-duplication pass (merged the
double "not a depth artifact" argument, compressed §6.6 from 6 items to 4, moved
scripts to Appendix A). We are open to further cuts but flag that the density is
already conference-appropriate and the numbers are load-bearing.

### Q40 — Several figures repeat the same message. Consolidate.

**Answer.** The four figures are non-redundant: fig1 = sequence/trim diagram (§3.3,
what the manipulation *is*); fig2 = mechanism (§6.1, *why* it happens, with the
deleted-counterexamples box); fig3 = premature-firing bars (§5.1, the *behavioral*
result, real data); fig4 = logit signature (§6.1, the *internal* result, real data,
log scale). fig3 and fig4 look adjacent but carry distinct evidence (behavior vs
logits). We can merge fig1/fig2 into one mechanism panel if space demands; the
result/mechanism figures should stay separate because they answer "what" vs "why."

---

## Meta-review Questions

### Q41 — If I remember ONE sentence, what should it be?

**Answer.** *Trimming SFT sequences to end at a rare control marker — a curation step
that feels like it sharpens marker learning — instead deletes the sub-threshold
counterexamples the model needs and makes it fire prematurely, by +0.44–0.58.*

### Q42 — Scientific contribution, not engineering?

**Answer.** The scientific contribution is identifying a *shortcut-learning failure
mode specific to count/threshold-triggered control tokens*: trimming converts a
latent-count threshold into a trigger-presence classifier by removing boundary
negatives, evidenced at the logit level. The engineering payoff (don't trim; retain a
benign continuation) *follows from* the science but is not the point — the point is
that the *shape of supervision*, not scale, determines whether a model learns a count
threshold or a presence shortcut.

### Q43 — What new scientific principle has been discovered?

**Answer.** That *learning a count/threshold-triggered emission requires
sub-threshold negative continuations in the training support*, and a common truncation
silently removes exactly those, so the model regresses from thresholding-on-count to
firing-on-presence. More generally: for rare control tokens, **the negatives near the
firing boundary are the load-bearing training signal**, and curation that maximizes
marker-terminality correlation destroys them.

### Q44 — Still interesting if the tutor application disappeared?

**Answer.** Yes — the tutor is a vehicle. Phase 3 already removes the tutor entirely
(customer-support escalation, Llama) and the effect persists. The finding is about
next-token training on rare count-triggered semantic markers, a class that includes
escalation, refusal-after-N, stop-after-goal, and (conjecturally) agentic control
tokens. The principle stands with no tutoring in the loop.

### Q45 — How should future datasets change because of your findings?

**Answer.** Concretely (§6.4): (1) do *not* trim training sequences to end at a rare
marker — retain a benign post-marker continuation; (2) treat where a sequence is
truncated as a *documented experimental variable*, not silent preprocessing (it can
move premature firing by +0.5); (3) evaluate *timing* separately from *semantics* with
a dedicated sub-threshold probe, since fluency/quality metrics miss premature firing;
(4) where the trigger is a count, supervise the count explicitly in the output.

---

## "Fatal Questions"

### F1 — How do you know trimming is the cause, not a hidden variable?

**Answer.** The two corpora are byte-identical except the post-marker continuation —
the trimmed corpus is an *exact prefix* of the untrimmed (verified 3000/3000 in Phase 3;
§4.4 exact-prefix guarantee for the tutor cells). Identical hyperparameters, optimizer,
schedule, seed, steps. So trim is the *only* manipulated variable by construction. The
main rival hidden variable — turn-depth — is ruled out (§6.1: the benign majority
decorrelates depth from firing; the inflated logit mass sits on escalation, not depth).
Label imbalance (Q19) and forgetting (Q18) are also ruled out. The effect reproduces at
a second seed (§5.1).

### F2 — Why generalize beyond synthetic datasets?

**Answer.** Because the mechanism (§6.1) is about the *structure of the conditional
distribution* (boundary negatives deleted), which is dataset-agnostic — and we test that
prediction on a second, independently-built synthetic domain and a different family
(Phase 3). Synthetic is a *necessity* for the paired contrast (Q11), not a limitation of
scope. The mechanism would produce a logistic-regression replication with no transformer
(Q27), underscoring generality. A real-corpus *audit* is named as future work.

### F3 — How do you know this is not specific to Qwen?

**Answer.** Phase 3 replicates the effect on **Llama-3.2-1B-Base** — different family,
different tokenizer, different attention type (full-attention vs Qwen's
linear-attention/gated-recurrent), different domain — 0.683 → 0.830 (§5.5b). Existence is
therefore demonstrated family-independent. We are careful to add that *magnitude* may be
family-dependent and flag a matched-magnitude cross-family run as open (§6.6/§6.7).

### F4 — Why should practitioners care?

**Answer.** Because the harmful practice is *common and feels correct* — practitioners
trim to end at the marker believing it sharpens learning — and the cost is a +0.5
premature-firing rate on exactly the tokens that trigger real system actions (session
end, tool call, escalation, refusal). The fix is nearly free (keep a benign continuation;
corpus bookkeeping only). High-leverage, low-cost, and invisible to standard
fluency/quality evals (§6.4).

### F5 — What existing paper does this contradict or extend?

**Answer.** It *extends* the shortcut-learning literature (`geirhos2020shortcut`,
`mccoy2019hans`) to the curation of rare control tokens, and connects to
process-supervision / scratchpad work (`nye2021scratchpad`, `wei2022cot`,
`lightman2024verify`) via the count-annotation remedy. It *contradicts* the implicit
folk practice — reflected in many tool-calling / structured-output SFT pipelines
(`liu2024apigen`, `liu2024toolace`, and grammar-constrained decoding work
`dong2024xgrammar`, `park2024grammaraligned`) — that truncating at the control token is a
harmless or beneficial preprocessing step. It complements agentic benchmarks
(`yao2024taubench`, `zheng2024processbench`) by supplying a data-side failure mode they
would observe but not explain.

### F6 — If correct, why hasn't anyone reported this before?

**Answer.** Three reasons. (1) It is *invisible to standard metrics*: fluency, quality,
and even positive-probe recall all look *fine or better* under trimming (trimmed recall
is actually higher, 0.994 vs 0.893) — you only see it with a dedicated *sub-threshold*
probe, which is not standard. (2) The manipulated variable is usually a *silent
preprocessing default*, not a logged experimental knob, so it is never ablated. (3) The
effect requires a *count/threshold trigger with recognition noise*; much control-token
work uses exact syntactic triggers where (per our prediction) the effect is absent — so
the regime where it bites is under-studied.

### F7 — Could the result disappear on GPT-4-class models?

**Answer.** We do not test that scale and say so. Directional prediction (Q13/Q14): the
shortcut shrinks as recognition improves, so we expect *attenuation* with scale but not
trivial disappearance — the trimmed conditional is still the cheapest sufficient
predictor. Whether a frontier model's recognition is good enough to erase the shortcut is
an empirical open question; a scale sweep is priority future work (§6.7).

### F8 — If OpenAI trains with trillions of tokens, would it still matter?

**Answer.** Scale is about *pretraining* breadth; the effect is a property of the
*fine-tuning corpus shape* for a *specific rare marker*, which trillions of pretraining
tokens do not fix — the model has still never seen sub-threshold-trigger-then-no-fire for
*that* marker unless the SFT curator retains it. So the fix lives in SFT curation
regardless of pretraining scale. Better recognition from scale may *reduce* magnitude
(Q13), but the curation lever remains the direct control and costs almost nothing to get
right.

---

## Response to the reviewer's four-topic prediction

The reviewer predicts ~80% of discussion is generalization, mechanism, scientific
framing, and claim scope. Our current build addresses each:

- **Generalization** — Phase 3 off-domain + off-family replication (§5.5b); scope
  stated (§6.6); naturalistic audit named as future work.
- **Mechanism** — causal argument + logit-level 34–45× evidence + count-stratified
  behavior (§6.1, §5.2); rivals (depth, forgetting, imbalance, optimization) ruled out.
- **Scientific framing** — positioned as a shortcut-learning / boundary-negative-deletion
  principle (§6.1, Q42–Q43), not a data-curation trick.
- **Claim scope** — "PROVEN" removed, "architecture-independent" removed, transfer to
  exact-syntactic triggers explicitly left open (§6.3, §6.6).
