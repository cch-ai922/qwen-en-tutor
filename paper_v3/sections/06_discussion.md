# 6. Discussion

<!-- paper_v3 — "Don't Trim the Tail" -->

## 6.1 Why trimming causes premature firing (the mechanism)

![The mechanism. Untrimmed data contains counterexamples — escalated contexts *not* followed by a fire — so the model learns escalation is necessary but not sufficient and fires on the third strike. Trimming deletes those counterexamples, leaving escalation perfectly predictive of the marker, so the model fires early.](paper_v3/figures/fig2_mechanism.png){width=90%}

The naive story — "trimming teaches the model the marker sits at the end of the
sequence, so it fires when it detects the end" — is mechanistically incoherent:
an autoregressive, strictly causal decoder generating token *t* conditions only
on tokens 1..t-1 and has no access to whether more tokens will follow.
"Am I at the end?" is not a feature it can compute at generation time. (This
argument is about causality and so applies to any strictly causal decoder —
linear-attention or gated-recurrent variants included — though we test the effect
empirically on two families, not all architectures.)

Our evidence is most consistent with an account about which *counterexamples*
trimming removes from the conditional distribution the model fits — a
shortcut-learning story [@geirhos2020shortcut; @mccoy2019hans] in which the
cheapest sufficient predictor wins. We frame this as the best-supported
explanation of the results below rather than a proven mechanism. SFT optimizes
P(marker | left-context).
The marker's true trigger is a *count* (third same-axis strike); the left-visible
correlate the model can cheaply read is *escalation-presence* (the user is
repeating a governed axis and the tutor is redirecting with escalating brevity).
In untrimmed persistent records, the turns *after* the marker show an escalated
context that is **not** followed by another fire — direct supervision that
escalation-presence is necessary but not *sufficient*. Trimming deletes exactly
those turns. After trimming, every escalated context in training terminates at a
fire, so the model fits P(fire | escalation-present) ≈ 1 and fires as soon as it
detects escalation — after one or two strikes — rather than counting to three.

This predicts, and the data confirm (§5.2), that premature firing rises with
accumulated violation count (vc=1 → vc=2 jumps in every trimmed cell) while
untrimmed cells stay low at both — the signature of escalation-thresholding
failure, not a turn-position shortcut.

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

At sub-threshold escalation the trimmed model places **34–45$\times$ more
probability mass on beginning the sentinel** than the untrimmed model, and that
mass grows with accumulated escalation. The untrimmed model keeps this mass
near-zero until the true third strike (consistent with its 0.207 premature rate);
the trimmed model's is pushed toward the firing boundary at every sub-threshold
level (consistent with its 0.783). The graded internal signal thus confirms the
prose mechanism directly: trimming inflates the model's conditional probability of
firing given escalation, well before the threshold. This is *not* a depth
artifact. Raw turn-depth rises in lockstep with escalation (deeper turns can only
carry more strikes), but the shared ~4/5 benign majority — deep dialogues that
never fire — already decorrelates depth from firing; the inflated conditional
therefore sits on the escalation feature specifically, which only the persistent
1/5 carries and which trimming strips of its negative half. Escalation, not
depth, is the clean read.

## 6.2 Timing and attribution are separable

Trim collapses *timing* (premature firing) but leaves *attribution* almost
untouched (A5 attribution 0.966 → 0.938 while premature 0.119 → 0.560). Marker
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
   single highest-leverage curation choice we identify.
2. **Treat sequence shaping as a documented experimental variable,** not a silent
   preprocessing detail — where a sequence is truncated can move premature firing
   by +0.5 (§5.1).
3. **Evaluate timing separately from semantics.** Conventional fluency/quality
   metrics do not detect premature firing; a dedicated sub-threshold probe does
   (§5.2). *When* and *why* a marker fires are separately curated behaviors
   (§6.2) and must be measured separately.
4. **Where the trigger is a count, supervise the count** (§6.5) rather than
   leaving it a latent variable the trim can corrupt.

These cost little beyond corpus bookkeeping and are independent of model scale
and architecture (§6.1).

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

**Scope of the claims.** We claim, and support with controlled experiments, a
specific causal effect: for a rare marker whose trigger is a *noisy,
recognition-gated* count, trimming training sequences to end at the marker
inflates premature firing, via deletion of the sub-threshold counterexamples
(§6.1). We do *not* claim this is a universal property of all control tokens,
all curation pipelines, or all model scales. In particular the effect is
demonstrated at 0.8B (Qwen3.5) plus an off-family 1B base (Llama-3.2-1B-Base, §5.5b), on count/threshold
triggers, and with SFT; extrapolation to much larger models, to exactly-computable
triggers (§6.3), or to RL/preference post-training is conjecture. The remaining
bullets enumerate the specific axes along which our evidence is thin.

- **Single family for the main 2×2×2 — and why it is a *secondary* threat here.**
  All eight cells of the main design use Qwen3.5 0.8B. This matters less for
  paper_v3 than for a prompt-vs-train boundary claim, because our claims are about
  the *training signal* (which counterexamples trimming deletes; what supervising
  an axis label forces the model to represent), not about a specific model's
  capability. The generality test we run is therefore an *off-family, off-domain*
  replication rather than a within-corpus ablation: Phase 3 (§5.5b) fine-tunes
  **Llama-3.2-1B-Base** — a non-Qwen, full-attention pretrained base — on a
  synthetic customer-support escalation task whose trigger remains *semantic and
  recognition-gated* (escalate after the 3rd explicit request; angry-but-non-
  requesting venting is a distractor that must not count). The trim→premature
  effect replicates in direction there (0.683 → 0.830, both at recall 1.00), so
  the effect's *existence* is already family- and domain-independent. Because that
  run uses a full-attention model, it *also* speaks to the magnitude question:
  Qwen3.5 uses linear-attention/gated-recurrent layers that compress history into
  a fixed-size state, which could plausibly make cross-turn counting more fragile
  and thus *amplify* trim sensitivity relative to a full-attention model. A tighter
  cross-family *magnitude* comparison — re-running the exact A1 trim/untrim contrast
  on a small full-attention non-Qwen base (e.g. Llama-3.2-1B or SmolLM2-360M) with
  matched data and hyperparameters — would isolate whether the +0.5 magnitude is
  architecture-specific; we flag that as the open cross-family *magnitude* question
  while reporting *existence* as family-independent.
- **Reconstructed untrimmed cells.** The untrimmed A5/A6/A7 are rebuilt by
  reconstruction (Appendix A), not trained-from-scratch matched pairs. The
  exact-prefix verification (§4.4) is the guarantee that only the post-marker
  continuation differs; still, an independent from-scratch regeneration would be
  stronger.
- **Single dataset / task — addressed by Phase 3.** The trim principle is
  demonstrated primarily on the tutoring persistence task. Phase 3 (§5.5b)
  replicates it in a *different domain* (customer-support escalation after the
  3rd angry message) on a *different model family* (Llama-3.2-1B), showing the
  effect is a property of next-token training on rare count-triggered semantic
  markers, not of this corpus or the Qwen family. The trigger is kept semantic
  and recognition-gated (not literal token counting) because recognition noise is
  the substrate the trim shortcut requires (§6.1).
- **Synthetic data — a methodological necessity, not merely a convenience.** Both
  our datasets are synthetic, which invites the question of whether the effect
  would appear on a real instruction-tuning corpus. The core of our method is a
  *single-variable paired contrast*: a trimmed and an untrimmed corpus that are
  byte-identical except for the presence of the post-marker continuation (verified
  as an exact prefix, §4.4). No pre-existing real dataset supplies such a pair —
  constructing one requires taking real dialogues and regenerating each with and
  without the continuation, at which point the controlled corpus is synthetic by
  construction. Real corpora also lack the labelled sub-threshold probes (contexts
  with a *known* strike count short of the threshold) that make premature firing
  measurable. Synthetic control is therefore what lets us attribute the effect to
  the trim and nothing else; the generalization evidence we *can* give — a
  different domain and a different model family (§5.5b) — is the appropriate
  substitute for a real-corpus replication that the paired design forecloses. We
  none the less regard a naturalistic study (e.g. auditing an existing tool-calling
  corpus for trim-correlated premature calls) as valuable future work.

## 6.7 Future work

The most informative next experiments, in priority order:

1. **Dose-response** — vary the retained post-marker continuation
   (0/25/50/75/100%) to show the trim effect is continuous in counterexample
   supply, not a binary switch.
2. **Cross-family magnitude** — re-run the A1 contrast on a small full-attention
   non-Qwen base (Llama-3.2-1B / SmolLM2-360M) to test whether the +0.5 magnitude
   is specific to Qwen's linear-attention state; existence is already
   family-independent via Phase 3 (§5.5b).
3. **Scale** — repeat the contrast at 4B for scale-(in)dependence.
4. **Close H3** — the count-annotated remedy under-fires at 0.8B (§5.5); larger
   context and dropping the axis label are the two fixes to test.
