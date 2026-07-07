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
chain-of-thought) helps. Our count-annotated marker (Design B) is a minimal,
inference-cheap form of this: it supervises the running count directly in the
output rather than requiring a separate reasoning trace, in the spirit of recent
process-supervision work that supervises intermediate steps rather than only the
final answer
[@bhattamishra2020ability; @nye2021scratchpad; @wei2022cot; @lightman2024verify; @zheng2024processbench].

**The gap we address.** Across these threads, instruction tuning, alignment, and
control-token utilization are all well studied, yet no work isolates the effect of
*how each training sequence is shaped around a rare marker* — specifically whether
truncating at the marker changes the learned emission threshold — while holding
token design, position, and optimization fixed. It therefore remains unclear
whether premature control-token emission originates from token design,
optimization, or sequence structure. We close this gap by making post-marker trim
an explicit factor in a fully-crossed design (§3), so its effect is measured
independently of the position and marker-format choices it is usually bundled
with.

**Relation to the companion paper.** This paper is a data-curation study split
off from a companion boundary paper (the train-vs-prompt boundary for tutoring
behaviors). There, multi-turn persistence and pedagogical withholding are shown
to be *not promptable* — acquired only by SFT. Here we take the persistence
marker as a controlled testbed and ask which *data-shape* factors govern its
emission once training is committed to. The two papers share the pipeline and
the persistence task but make disjoint claims: the companion paper on
prompt-vs-train acquisition, this one on trim/typing/count curation.
