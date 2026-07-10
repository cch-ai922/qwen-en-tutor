# 7. Conclusion

<!-- paper_v3 — "Don't Trim the Tail" -->

We studied how the *shape* of SFT data governs a fine-tuned LM's emission of a
rare, machine-consumed control marker, using a four-axis session-ending sentinel
as a controlled testbed.

Our primary result is counter-intuitive and robust: **trimming training
sequences to end at the marker — a curation step one might expect to sharpen
marker learning — instead induces premature firing.** Across every
(position × marker) cell, trimming raised the premature-firing rate by
+0.44 to +0.58, the largest observed main effect in the tested design. The
evidence is most consistent with a threshold-laxity mechanism, rather than a
turn-position shortcut or a literal "detects the end" account (which is not
available to a strictly causal decoder): trimming deletes the training examples in
which an escalated context is *not* followed by a fire, leaving escalation-presence
nearly always predictive of the marker and impairing the model's ability to
threshold on the strike count. Retaining the benign post-marker continuation
substantially reduces premature firing.

We further showed that *when* a model fires and *why it says it fires* are
separable, separately-curated behaviors: **typed markers act as semantic gates**,
yielding 0.94–0.99 correct-axis attribution and zero contentless fires, where
generic markers cannot attribute at all — and attribution is robust to the trim
that worsens timing. Typed markers support robust per-axis attribution under a
sub-threshold distractor (correct-axis 0.96–0.98, pulled to the distractor <2%).
Finally, the trim effect replicates *in direction* off-family and off-domain —
fine-tuning Llama-3.2-1B-Base on a synthetic customer-support escalation task
reproduces trim→premature (trimmed 0.830 vs untrimmed 0.683, both at full recall)
— suggesting it is not specific to the tutoring corpus or the Qwen family. We also
outline **count-annotated markers** — supervising the strike count directly in the
marker — as a candidate remedy for the trim artifact, and leave its evaluation to
future work (§6.6).

The practical takeaway is a curation principle for any rare control token with a
count/threshold trigger: **do not trim to the marker; retain a continuation
after it; and where the trigger is a count, supervise the count explicitly.**
These are cheap, low-overhead data choices with first-order effects on
control-token reliability, observed across two model families (Qwen3.5, Llama-3.2).

<!-- Status: H1 (trim→premature), H1-mechanism (logit-level analysis), H2 single-axis and
     H2b under-distraction, and H-gen off-family replication
     are all supported by experiments and in the build. Count-annotated remedy is
     outlined as future work only (§6.6). -->
