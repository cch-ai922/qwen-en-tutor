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
marker and destroying the model's ability to threshold on the strike count.
Retaining the benign post-marker continuation restores correctly-timed firing.

We further showed that *when* a model fires and *why it says it fires* are
separable, separately-curated behaviors: **typed markers act as semantic gates**,
yielding 0.94–0.99 correct-axis attribution and zero contentless fires, where
generic markers cannot attribute at all — and attribution is robust to the trim
that collapses timing. Finally, we proposed **count-annotated markers** as a
direct remedy: supervising the running strike count in the output should make
firing trim-robust, closing the loop between the trim pathology and its cause
(an under-supervised latent counter).

The practical takeaway is a curation principle for any rare control token with a
count/threshold trigger: **do not trim to the marker; retain a continuation
after it; and where the trigger is a count, supervise the count explicitly.**
These are cheap, architecture-independent data choices with first-order effects
on control-token reliability.

<!-- Status: H1 (trim→premature) and H2 single-axis (typed=gate) are PROVEN from
     existing generations (Phase 0). H2-under-distraction (Phase 1) and H3
     count-remedy (Phase 2) are built and specified, awaiting GPU. -->
