# 6. Discussion

<!-- paper_v3 — "Don't Trim the Tail" -->

## 6.1 Why trimming causes premature firing (the mechanism we find most consistent with the evidence)

![The mechanism. Untrimmed data contains counterexamples — escalated contexts *not* followed by a fire — so the model learns escalation is necessary but not sufficient and fires on the third strike. Trimming deletes those counterexamples, leaving escalation perfectly predictive of the marker, so the model fires early.](paper_v3/figures/fig2_mechanism.png){width=90%}

The naive story — "trimming teaches the model the marker sits at the end of the
sequence, so it fires when it detects the end" — is not compatible with a literal
future-aware explanation: an autoregressive, strictly causal decoder generating
token *t* conditions only on tokens 1..t-1 and has no access to whether more
tokens will follow, so "am I at the end?" is not a feature it can compute at
generation time. (This argument is about causality and so applies to any strictly
causal decoder — linear-attention or gated-recurrent variants included — though we
test the effect empirically on two families, not all architectures.) We note that
present-position and other contextual termination cues remain *possible*
alternative correlates; the account we advance below is the one most directly
supported by the present evidence, not the only conceivable one.

Our evidence is most consistent with an account about which *counterexamples*
trimming removes — a shortcut-learning story [@geirhos2020shortcut; @mccoy2019hans]
in which the cheapest sufficient predictor wins, framed as the best-supported
explanation rather than a proven mechanism. SFT optimizes P(marker | left-context).
The marker's true trigger is a *count* (third same-axis strike); the cheap
left-visible correlate is *escalation-presence*. In untrimmed records, the turns
*after* the marker show an escalated context **not** followed by a fire — direct
supervision that escalation is necessary but not *sufficient* — and trimming
deletes exactly those turns, so the model fits P(fire | escalation-present) ≈ 1 and
fires after one or two strikes rather than counting to three. This is precisely the
vc=1→vc=2 rise, depth-adjusted, that §5.2/§5.2b document.

**Semantic recognition is a precondition, not incidental.** The shortcut is
available only because the model's *count* of same-axis strikes is uncertain, and
it is uncertain because each strike must first be *recognized* — "is this user
turn a role-swap attempt?" is a fuzzy semantic judgment, not a token match. The
trim does not teach "fire on a marker token seen at the end"; it teaches "fire
once the *noisy accumulated evidence* of the semantic trigger is high," because
trimming removed the examples showing that high evidence short of the exact
threshold is not yet a fire. This is why the effect requires a semantic,
recognition-gated trigger: a task that counts an *explicit, unambiguous* token
would let the model count exactly and leave no recognition noise for the trim to
exploit — no shortcut, no premature firing. The generalization test (§5.5b) is
therefore constructed on a *different-domain but still semantic* trigger (escalation
after repeated *angry* customer messages, where "angry" is recognition-gated),
not on literal token counting, precisely because the recognition noise is the
substrate the trim shortcut operates on.

**The logit-level signature (Phase 4).** The evidence supports this account
directly at the logit level: for each sub-threshold context we read the probability the model *begins
the sentinel* — the joint P(`[`) $\times$ P(`SESSION` | `[`), a two-token
teacher-forced measurement (not sampling; Appendix A) that isolates the sentinel
from any other bracketed token — for the A1 trim/untrim
pair over the same 318 sub-threshold contexts, stratified by accumulated
escalation (violation count):

| violation count | untrimmed P(sentinel) | trimmed P(sentinel) | ratio |
|-----------------|-----------------------|---------------------|-------|
| 1 (one prior strike)  | 0.000005 | 0.000213 | **45$\times$** |
| 2 (two prior strikes) | 0.000045 | 0.001530 | **34$\times$** |

![Logit-level signature (A1). At sub-threshold escalation the trimmed model places 34–45× more probability mass on beginning the sentinel than the untrimmed model, and that mass grows with accumulated violation count (log scale).](paper_v3/figures/fig4_logit.png){width=60%}

The graded internal signal supports the prose mechanism directly: trimming inflates
the model's conditional probability of firing given escalation, well before the
threshold (untrimmed mass stays near-zero until the true third strike; the trimmed
model's is pushed toward the boundary at every sub-threshold level). And this sits
on the escalation *feature*, not raw depth: the shared ~4/5 benign majority already
decorrelates depth from firing, and holding depth fixed the strike-count term
survives adjustment (§5.2b). The most direct interpretation is that escalation
count, not raw depth, drives the effect — though a fully orthogonal balanced probe
would test it more directly (§6.6).

## 6.2 Timing and attribution are separable

Trim substantially worsens *timing* (premature firing) but leaves *attribution*
almost untouched (A5 attribution 0.966 → 0.938 while premature 0.119 → 0.560). Marker
typing governs attribution independent of timing. So "a control marker" is
really two learned behaviors — a trigger (when) and a classifier (why) — with
independent curation levers: the trim (or post-marker continuation) governs the
trigger threshold; the marker's semantic content governs the classifier. A
practitioner can get one right and the other wrong; both must be curated.

## 6.3 A curation principle for rare control tokens

Generalizing beyond this task: **when training a model to emit a rare marker on
a threshold/count trigger, do not trim training sequences to end at the marker.**
Trimming maximizes the correlation between the marker and sequence-terminality/
trigger-presence, deleting the negative examples the model needs to learn that
the trigger's *presence* is not its *threshold*. Retain a benign continuation
after the marker. Where the trigger is a count, additionally supervise the count
explicitly in the output (§6.5). We conjecture — but do not demonstrate — that
this principle extends to other rare, machine-consumed markers whose trigger is a
*noisy, recognition-gated* count or threshold: refusal-after-N,
escalate-after-repetition, stop-after-goal. We deliberately scope the claim this
way. The mechanism (§6.1) requires that the count be *latent and uncertain* for
trimming to have counterexamples worth deleting; where the trigger is exactly
computable from surface tokens, there is no noise to exploit and we predict no
effect (a literal token-count trigger; see the mechanism argument in §6.1). We do
not run this control experiment; it follows from the mechanism rather than being
established here. Whether the artifact appears for
markers with crisp, non-semantic triggers — e.g. schema-driven tool/function-call
or JSON control tokens emitted on an exact syntactic condition — is therefore an
open empirical question we do not settle here.

## 6.4 Practical recommendations

The findings translate into concrete, low-cost guidance for anyone building SFT
corpora that contain rare machine-consumed markers:

1. **Retain a benign continuation after the marker.** Do not trim training
   sequences to end at the marker; keep the post-marker turns that show an
   escalated context *not* followed by another fire (§5.1, §6.1). This is the
   highest-leverage curation choice we identify, and substantially reduces
   premature firing.
2. **Treat sequence shaping as a documented experimental variable,** not a silent
   preprocessing detail — where a sequence is truncated can move premature firing
   by +0.5 (§5.1).
3. **Evaluate timing separately from semantics.** Conventional fluency/quality
   metrics do not detect premature firing; a dedicated sub-threshold probe does
   (§5.2). *When* and *why* a marker fires are separately curated behaviors
   (§6.2) and must be measured separately.
4. **Where the trigger is a count, supervise the count** (§6.5) rather than
   leaving it a latent variable the trim can corrupt.

These cost little beyond corpus bookkeeping. We demonstrate them at a single
small scale (0.8B) and on two model families; whether the magnitude holds at much
larger scale is left to future work (§6.6, §6.7).

## 6.5 The count-annotated marker (Design B)

Trim breaks a *latent* counter; the direct remedy is to stop keeping it latent.
Design B tags every strike with its running count, converting the count from an
unsupervised intermediate variable into a supervised output target. To fire
prematurely the model would now have to emit a wrong number (strike=2 where it
should say strike=3), which the loss penalizes — so the count annotation should
make firing contingent on the actual tally and, in particular, trim-robust
(H3, §5.5). This is chain-of-thought/process-supervision
[@nye2021scratchpad; @wei2022cot; @lightman2024verify] compressed to a single
tag: no separate reasoning trace, negligible inference cost, and (unlike native
CoT) no delivery-at-budget failure mode. The trade-off is a changed deployment
contract — the dispatcher now sees `[STRIKE: axis, N]` on non-terminal turns —
which we consider acceptable because those tags are independently useful
(per-turn abuse telemetry).

## 6.6 Threats to validity

**Scope of the claims.** We claim, and support with a controlled paired
experiment, a specific effect: for a rare marker whose trigger is a *noisy,
recognition-gated* count, trimming training sequences to end at the marker
inflates premature firing. Because the trimmed and untrimmed corpora are
byte-identical except for the post-marker continuation (§4.4, and the exact
removed-tail accounting of §4.5), the trim *manipulation* is a clean causal
intervention. The *explanation* we offer for it — deletion of the sub-threshold
counterexamples inducing threshold-laxity (§6.1) — is the account most consistent
with the evidence (including the logit-level signature and the depth-adjusted
strike-count model), not a proven mechanism to the exclusion of all contextual
alternatives. We do *not* claim this is a universal property of all control
tokens, all curation pipelines, or all model scales. In particular the effect is
demonstrated at 0.8B (Qwen3.5) plus an off-family 1B base (Llama-3.2-1B-Base, §5.5b), on count/threshold
triggers, and with SFT; extrapolation to much larger models, to exactly-computable
triggers (§6.3), or to RL/preference post-training is conjecture. The remaining
bullets enumerate the specific axes along which our evidence is thin.

- **Single family for the main 2×2×2 — a *secondary* threat here.** All eight
  cells use Qwen3.5 0.8B, but our claims are about the *training signal* (which
  counterexamples trimming deletes), not a specific model's capability, and the
  off-family, off-domain Llama-3.2-1B-Base replication (§5.5b) already shows the
  effect's *existence* is family- and domain-independent. What it does not settle
  is *magnitude*: Qwen3.5's linear-attention state could plausibly amplify trim
  sensitivity relative to a full-attention model, so a matched A1 trim/untrim
  contrast on a small full-attention non-Qwen base is the open cross-family
  *magnitude* question (§6.7).
- **Reconstructed untrimmed cells.** The untrimmed A5/A6/A7 are rebuilt by
  reconstruction (Appendix A), not trained-from-scratch matched pairs. The
  exact-prefix verification (§4.4) is the guarantee that only the post-marker
  continuation differs; still, an independent from-scratch regeneration would be
  stronger.
- **Seed count.** The main eight-cell design is trained at a single seed (42),
  with the primary A1 trim/untrim pair additionally replicated at seed 7 — two
  seeds in total for the primary contrast. Two seeds do not fully characterize
  training variance. We rely instead on effect *magnitude* (per-cell +0.44–0.58,
  Fisher *p* < 10⁻³², disjoint Wilson intervals) plus same-direction reproduction
  at an independent seed and an off-family model to argue the result is not
  seed-driven; a pre-registered ≥3-seed run at fixed checkpoints (seeds 7/42/123,
  five if budget permits) remains the proper test and is specified in §6.7.
- **Checkpoint selection for the seed-7 pair.** The seed-7 pair is reported at the
  earliest shared checkpoint clearing the recall gate for *both* variants
  (checkpoint-600, epoch 1.51), not the exact 1-epoch budget of the seed-42
  primary, because at 1 epoch the seed-7 untrimmed model had not yet learned to
  fire and a premature-firing contrast on a non-firing model is uninterpretable.
  This is a principled, pre-specified gate (recall ≥ threshold for both variants),
  not an outcome-maximizing search over checkpoints, but we flag it because a
  reviewer could read it as checkpoint selection; the fixed-checkpoint multi-seed
  protocol (§6.7) removes the ambiguity.
- **Token-budget confound (bounded, not eliminated).** Trimming removes the
  post-marker continuation, which also removes a small number of supervised tokens
  (§4.5: mean ≈14 assistant tokens, exactly one turn, per persistent dialogue).
  In principle the effect could ride on "fewer supervised tokens" rather than "the
  missing counterexamples" specifically. Two facts bound this: the removed tokens
  are ≈1% of the persistent-stream supervised tokens and a far smaller fraction of
  the whole corpus, and the depth-adjusted count model (§5.2b) locates the effect
  on the escalation feature rather than on generic sequence length. A definitive
  separation is a *token-matched control* — adding the same number of extra
  supervised tokens as benign continuation *without* the high-escalation/no-fire
  structure — which we specify in §6.7.
- **Single dataset / task — addressed by Phase 3.** The trim principle is
  demonstrated primarily on the tutoring persistence task; the §5.5b Llama
  replication (a different domain and family) is what extends it beyond this
  corpus, keeping the trigger semantic and recognition-gated because recognition
  noise is the substrate the shortcut requires (§6.1).
- **Synthetic data — a methodological necessity, not merely a convenience.** Both
  datasets are synthetic. Our method is a *single-variable paired contrast* — a
  trimmed and untrimmed corpus byte-identical except for the post-marker
  continuation (exact prefix, §4.4) — and no pre-existing real dataset supplies
  such a pair; constructing one requires regenerating real dialogues with and
  without the continuation, at which point the controlled corpus is synthetic by
  construction. Real corpora also lack the labelled sub-threshold probes that make
  premature firing measurable. Synthetic control is thus what lets us attribute the
  effect to the trim alone; a naturalistic study (auditing an existing tool-calling
  corpus for trim-correlated premature calls) remains valuable future work.

## 6.7 Future work

The most informative next experiments, in priority order (the first three
directly answer the strongest reviewer requests and are the ones we would run
before any broadening of scope):

1. **Multi-seed at fixed checkpoints.** Train the primary A1 trim/untrim pair at
   ≥3 pre-registered seeds (7, 42, 123; five if budget permits), evaluate all at
   *identical* fixed checkpoints, and report mean ± SD with learning curves for
   both recall and premature firing. This replaces the current two-seed result and
   removes the checkpoint-selection ambiguity (§6.6).
2. **Counterexample dose-response and token-matched control.** Vary the *retained*
   post-marker continuation across C0/C25/C50/C75/C100 (0/25/50/75/100% of the
   tail kept) and plot premature firing against retained counterexample supply, to
   show the effect is continuous in counterexample dose rather than a binary
   switch. In addition, a *token-matched* control adds the same number of extra
   supervised tokens as C100 but as benign continuation *without* the
   high-escalation/no-fire structure; if only the genuine counterexample-bearing
   tail reduces premature firing, the mechanism (§6.1) is established rather than
   merely evidence-consistent, and the token-budget confound (§6.6) is closed.
3. **Orthogonal depth × strike-count probe.** Build a balanced grid that
   independently varies turn depth (e.g. 5/7/9/11) and strike count (1/2) at fixed
   semantic axis, with matched cell sizes, and estimate P(fire) = f(count, depth,
   trim). §5.2b already shows count survives depth on the *existing* (unbalanced)
   probe; a purpose-built balanced grid would settle the depth-vs-count question
   directly.
4. **Cross-family magnitude** — re-run the A1 contrast on a small full-attention
   non-Qwen base (Llama-3.2-1B / SmolLM2-360M) to test whether the +0.5 magnitude
   is specific to Qwen's linear-attention state; direction is already
   family-independent via Phase 3 (§5.5b).
5. **Scale** — repeat the contrast at 4B (and larger) to test whether the effect
   size is scale-dependent; the present study fixes scale and does not claim scale
   independence.
6. **Close H3** — the count-annotated remedy under-fires at 0.8B (§5.5); larger
   context and dropping the axis label are the two fixes to test.
