# 7. Conclusion

<!-- paper_v3 — "Don't Trim the Tail" -->

We studied how the *shape* of SFT data governs a fine-tuned LM's emission of a
rare, machine-consumed control marker, using a four-axis session-ending sentinel
as a controlled testbed.

Our headline result is counter-intuitive and robust: **trimming training
sequences to end at the marker — a curation step that feels like it should
sharpen marker learning — instead induces premature firing.** Across every
(position × marker) cell, trimming raised the premature-firing rate by
+0.44 to +0.58, a main effect far larger than the position or marker design
choices it is usually bundled with. The mechanism is not a turn-position
shortcut and not an incoherent "detects the end" story; it is threshold-laxity:
trimming deletes the training examples in which an escalated context is *not*
followed by a fire, leaving escalation-presence perfectly predictive of the
marker and impairing the model's ability to threshold on the strike count.
Retaining the benign post-marker continuation restores correctly-timed firing.

We further showed that *when* a model fires and *why it says it fires* are
separable, separately-curated behaviors: **typed markers act as semantic gates**,
yielding 0.94–0.99 correct-axis attribution and zero contentless fires, where
generic markers cannot attribute at all — and attribution is robust to the trim
that collapses timing. The gate holds under a distractor axis (correct-axis
0.96–0.98, pulled to the distractor <2%), so the label forces a genuine per-axis
semantic check. Finally, the trim effect is **not corpus-specific**: it replicates
off-family and off-domain — fine-tuning Llama-3.2-1B-Base on a synthetic
customer-support escalation task reproduces trim→premature (trimmed 0.830 vs
untrimmed 0.683, both at full recall) — establishing it as a general property of
next-token training on rare, count-triggered semantic markers. As a candidate
remedy we sketch **count-annotated markers** — supervising the running strike
count in the output, predicted to make firing trim-robust — and leave a
recall-cleared evaluation of them to follow-up work.

The practical takeaway is a curation principle for any rare control token with a
count/threshold trigger: **do not trim to the marker; retain a continuation
after it; and where the trigger is a count, supervise the count explicitly.**
These are cheap, low-overhead data choices with first-order effects on
control-token reliability, and the effect is observed across two model families
(Qwen3.5, Llama-3.2).

<!-- Status: H1 (trim→premature), H1-mechanism (Phase 4 logit), H2 single-axis and
     H2b under-distraction (Phase 1), and H-gen off-family replication (Phase 3)
     are all supported by experiments and in the build. H3 count-remedy (Phase 2) under-fires the
     terminal marker and is deferred to future work (§5.5). -->
