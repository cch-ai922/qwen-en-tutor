# 6. Discussion and Conclusion

Under a matched-prompt protocol, tutor redirect behaviors separate along an
interpretable line: **promptable** when a single clause both *describes* and
*elicits* the behavior, and **not promptable** when it requires cross-turn
state-tracking (persistence) or the suppression of a strong competing prior
(withholding) — things a clause can name but not produce. The boundary tells
a deployer of a small model which behaviors a prompt gives for free (locale,
role-swap, topic) and which require the pipeline (persistence, withholding).
Reaching it cleanly required retiring the conventional context-blind
redirect-axis F1 — a *type* classifier that ties a 0.8B student with a 9B
teacher — for matched per-capability instruments, which recover the largest
specialized-data effect in the study (role-swap quality, win-rate 0.87) on an
axis F1 called saturated parity.

**Cross-family replication.** The boundary is not a Qwen artifact: it
replicates in a second *trained* family. A Llama-3.2-1B student, evaluated
against its *own* untrained base under the identical prompt, lifts both
not-promptable behaviors far above prompt-only — persistence recall 0.25→0.91,
withholding 0.11→0.50 (two judges) — isolating training from scale, since
student and control share one base. A larger Llama-3.1-8B prompt-only probe
stays low even with chain-of-thought (persistence 0.27→0.55; withholding
0.22–0.32). The boundary's *direction* is robust across families; the
*magnitude* is family-dependent (Llama trained withholding 0.50 vs Qwen 0.61,
and Llama attains more prompt-only persistence than the Qwen teacher). One
asymmetry: the Llama student trains from an instruct checkpoint, since
Llama-3.2-1B-*Base* could not learn the rare turn-end token under LoRA-SFT.

**Threats to validity.** *Single locale*
does not threaten the central boundary: persistence and withholding are
structural behaviors independent of the locale backdrop. *Judged-metric
power*: the trained-vs-teacher withholding gap is directional at n=63; the
strong claim is the teacher's absolute sub-50% compliance. *Decorrelation* is
partially validated (fixes positional recall, not premature firing).

**Future work**, in priority order: (i) **broader cross-family replication**
(a third family, a base-checkpoint student, multi-locale) — the trained Llama
student above establishes the boundary in a second family; breadth remains;
(ii) **larger-scale decorrelation**, where the positional route is
cheaper relative to the semantic one; (iii) a **better-powered pedagogy
teacher comparison**; and (iv) transfer of the matched-prompt methodology to
other rare, semantically-triggered markers (refusal-token and tool-call
emission), where the same "describable but not promptable" question applies.

**Conclusion.** Which tutor behaviors must be trained and which can be
prompted is answerable per capability, and the answer is sharp: a prompt
clause installs a behavior only when it both describes *and* elicits it.
Persistence and pedagogical withholding are described by the prompt but
installed only by demonstration — the central finding, established on a single
consumer GPU where the distinction matters most.

## Ethics and data statement

All training and evaluation data are model-generated (distilled from a
locally-served teacher) and contain no personal data or PII; no human
subjects were involved. The intended use is research on data curation and the
train-versus-prompt boundary for small task-specific models.
