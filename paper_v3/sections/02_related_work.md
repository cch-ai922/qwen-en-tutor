# 2. Related Work

<!-- paper_v3 — "Don't Trim the Tail" -->

**Control tokens and structured generation.** Fine-tuned LMs are routinely
trained to emit special tokens that gate downstream machinery — end-of-turn and
stop tokens, tool-call and function-call delimiters, refusal/safety triggers,
and routing tags. Recent function-calling work focuses on *scaling and verifying*
the SFT data that teaches these tokens [@schick2023toolformer; @qin2024toolllm; @liu2024apigen; @liu2024toolace],
and deployed agent benchmarks show how much reliable control-token emission
matters in multi-turn tool use [@yao2024taubench]. A parallel line enforces
structure at *decode* time via grammar-constrained or guided generation
[@willard2023guidance; @dong2024xgrammar; @park2024grammaraligned] — though
constraining the decoder distorts the learned distribution, motivating our focus
on shaping the *training data* instead. Most of this work treats the markers as
generation *targets* and studies *whether* the model emits them; we instead study
how the *shape* of the training sequences around a rare marker determines *when*
(timing/threshold) and *with what content* (attribution) the model emits it.

**Sequence termination, EOS, and stop-token learning.** A rare control marker
that ends a session is closely related to the end-of-sequence token: both are
learned signals that terminate generation on a learned condition. Work on
end-of-sequence and stop-token behavior notes that models can acquire biased
termination tendencies from the length and position statistics of training
sequences — over- or under-terminating relative to the intended condition
[@newman2020eos; @stern2019insertion]. <!-- PLACEHOLDER cites: verify/replace -->
Our trim manipulation is a controlled instance of this: truncating every
marker-bearing training sequence *at* the marker maximizes the correlation between
the marker and sequence-end, and we measure the resulting shift in the emission
threshold directly.

**Truncation, loss masking, and preprocessing artifacts in SFT.** How each
training example is preprocessed — where it is truncated to fit the context
window, whether loss is masked to response tokens only, and how examples are
packed — is known to affect fine-tuning outcomes, yet is frequently left as an
undocumented pipeline detail [@raffel2020t5; @muennighoff2023scaling]. <!-- PLACEHOLDER: add packing/masking cites -->
Response-only loss masking and sequence packing in particular change which tokens
supply gradient and what context each target is conditioned on
[@wolf2020transformers]. <!-- PLACEHOLDER: replace with a packing/masking-specific reference -->
We treat one such choice — post-marker truncation — as a first-class experimental
factor rather than an incidental preprocessing step, and mask loss to assistant
turns throughout so masking is held constant across cells.

**Premature and false-positive control actions.** In deployed agentic systems the
operational failure our probe measures — firing a control marker *before* its
trigger condition — appears as premature or spurious tool invocation, over-eager
function calls, and mis-calibrated refusal/safety triggers, all of which degrade
reliability even when surrounding text is fluent [@yao2024taubench]. <!-- PLACEHOLDER: add tool-over-calling and refusal-FP cites -->
Rare-event and selective-prediction work frames the same tension as calibrating
*when to abstain or act* on a low-base-rate trigger [@geifman2017selective; @el2010foundations]. <!-- PLACEHOLDER: verify abstention/selective-prediction cites -->
We connect this deployment-level failure to a specific, controllable data-curation
cause.

**Shortcut learning and spurious correlations.** Models minimize loss via the
cheapest sufficient predictor, latching onto features that are predictive in
the training distribution but not causal for the task. Our trim result is a
control-token instance: trimming makes *sequence-terminality / escalation-
presence* perfectly predictive of the marker, and the model binds to that cheap
cue instead of the true count-based trigger
[@geirhos2020shortcut; @mccoy2019hans; @gururangan2018artifacts].

**Data curation for fine-tuning.** A large body of work studies which *examples*
to include (quality filtering, dedup, mixture weights). Less attention is paid
to how each example is *shaped* — where it is truncated, what is masked, whether
post-target continuation is retained. We isolate one such choice (post-marker
trimming) and show it has a first-order effect on behavior, larger than the
architectural/position design choices it is usually bundled with
[@zhou2023lima; @muennighoff2023scaling].

**Counting and multi-turn state in LMs.** Emitting a marker "on the third
strike" requires maintaining a count across turns. Prior work shows LMs struggle
with exact counting and that making intermediate state explicit (scratchpads,
chain-of-thought) helps. The candidate count-annotated marker we outline as future
work (§6.6) is a minimal, inference-cheap form of this — supervising the running
count directly in the output rather than in a separate reasoning trace — in the
spirit of process-supervision work that supervises intermediate steps rather than
only the final answer
[@bhattamishra2020ability; @nye2021scratchpad; @wei2022cot; @lightman2024verify; @zheng2024processbench].

**The gap we address.** These threads — instruction tuning, alignment,
control-token utilization, termination bias, and preprocessing artifacts — are
individually well studied, but to our knowledge they have not been brought together
to isolate the effect of *how each training sequence is shaped around a rare
marker* — specifically whether truncating at the marker changes the learned
emission threshold — while holding token design, position, and optimization fixed.
As a result it is, as far as we are aware, not yet established whether premature
control-token emission originates from token design, optimization, or sequence
structure. (We make this a bounded claim rather than an assertion of absolute
novelty, pending a more exhaustive survey.) We address this by making post-marker
trim an explicit factor in a fully-crossed 2×2×2 design (§3), so its effect is
measured independently of the position and marker-format choices it is usually
bundled with.

**Scope of this study.** We take a multi-turn persistence marker — a session-end
sentinel that must fire on the third same-axis violation — as a controlled testbed,
because it is a rare, count-triggered, machine-consumed control token of exactly the
kind whose curation we study. Prior work establishes that such multi-turn behaviors
are acquired through supervised fine-tuning rather than prompting alone; we take
that acquisition as given and ask a distinct, downstream question: once training is
committed to, which *data-shape* choices — where a sequence is trimmed, whether the
marker is axis-typed, whether the count is supervised — govern *when* the marker
fires and *what reason* it encodes. Our claims are therefore about the training
signal for control-token emission, and are independent of the tutoring domain we
draw the testbed from.
