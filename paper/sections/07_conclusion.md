# 7. Conclusion

We presented a data-generation pipeline for English-tutor dialogue
that goes beyond a flat "good behavior" corpus to systematically
cover the long tail of tutor-side behavior. Two of its contributions
are stated as transferable design principles, and one is the pipeline
that operationalizes them at consumer scale.

The first principle is an **invariant-based decomposition** of tutor
redirect behavior: a tutor is the keeper of a small enumerated set of
interaction invariants, a learner violation breaks exactly one
invariant, and the correct redirect is the minimal repair that
restores it — with axes distinguished iff their repairs differ in
shape. This turns "a generic redirect stream is insufficient" from an
assertion into a falsifiable per-axis prediction. The second
principle is **trigger-position decorrelation**: when a rare
structured marker must fire on a semantic trigger but is positionally
regular in the training data, the model learns position as a proxy;
resampling the marker position — under the determinism and
parity constraints the pipeline imposes — forces it to learn the
trigger instead. We instantiate the first as a 12-stream SFT taxonomy
and the second as a 4-variant persistent design, and wrap both in a
locale-aware, yield-aware pipeline whose LLM-judge filter component we
audit and patch.

Empirically, we demonstrate the pipeline at the smallest practical
scale (a 0.8B-parameter student trained on a single RTX 3060
12GB consumer GPU). The decorrelation claim is tested by a fully
mechanical metric — sentinel firing, where the full system sharply
outperforms a same-base ablation trained without the 4-variant
design — and the decomposition claim by a per-axis redirect F1
ablation, where the full system sharply outperforms both the
generic-redirect ablation on specialized axes and a same-size
off-the-shelf instruct model. On the mechanical locale-leakage metric
the full system again outperforms the same-size baseline, and on
naturalness it approaches the 9B-teacher distillation upper bound.
(All figures in §5; single-seed and single-locale caveats in §6.)

Future work includes (i) multi-locale empirical evaluation extending
beyond `china` to `japan`/`italy`/etc.; (ii) larger-student
experiments to test the framework's scaling prediction — that the
decomposition benefit shrinks with capacity while the decorrelation
benefit holds; (iii) a learned naturalness filter to close the
C1 / C2 naturalness gap; (iv) multi-family distillation studies (e.g.
a Llama-3 student from a Qwen3.5 teacher) to test the intra-family
distillation assumption embedded in our setup; and (v) transfer of
the trigger-position decorrelation construction to other rare,
semantically-triggered markers such as refusal-token and tool-call
emission.
