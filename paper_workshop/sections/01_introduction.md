# 1. Introduction

**Problem.** We ask, for each behavior a deployed English tutor needs,
whether a sufficiently explicit system prompt is enough or the behavior must
be demonstrated in fine-tuning. A deployed tutor is governed by a long
system prompt that already *states* most of what it should do — stay in
character, ground culture in the learner's locale, refuse role swaps, give
one short example rather than a grammar lecture, escalate to a session-end
signal under sustained abuse. If stating a behavior were sufficient, the
data-side problem would be trivial.

**Why it matters.** It is not: the behaviors split cleanly into ones a
prompt clause elicits and ones it only describes, and knowing which is which
is exactly the decision a practitioner faces when building a task-specific
model on consumer hardware — spend the data budget only where a prompt will
not do. The tutor setting makes the question answerable because each target
behavior corresponds to a clause already present in a realistic deployment
prompt (§3): the language of instruction, lesson topic, role structure,
persona, pedagogical contract, locale frame, and general appropriateness.

**Gap.** The prompting-versus-fine-tuning trade-off has been studied at the
level of *general* alignment — that a small demonstration set installs broad
instruction-following [@zhou2023lima; @ouyang2022instructgpt], and that
in-context demonstrations often convey format more than new capability
[@min2022rethinking] — but not resolved *per behavior* for a deployed task
model. Synthetic-instruction pipelines and tutor-LLM datasets (§2) target
general instruction-following with a uniform recipe; none asks, per
behavior, whether the deployment prompt already elicits what the data
teaches.

**Approach and contributions.** We supply the same fully-specified
deployment prompt at evaluation to a fine-tuned 0.8B student and to a ladder
of prompt-only baselines (0.8B base, 0.8B instruct, 4B instruct, and the 9B
teacher that generated the training data; Figure~\ref{fig:design}). Because
the instruction is present for every condition, a prompt-only failure cannot
be blamed on under-specification. Our contributions:

1. **A per-capability map of the train-versus-prompt boundary**
   (Figure~\ref{fig:boundary}): promptable when a single clause both
   *describes* and *elicits* the behavior (locale fidelity, role-swap and
   topic re-anchoring), not promptable when it requires cross-turn
   state-tracking (persistence) or the suppression of a strong competing
   prior (pedagogical withholding).
2. **An evidenced negative result about the conventional metric**: per-axis
   redirect F1 is a context-blind *type* classifier that ties a 0.8B student
   with the 9B teacher; we replace it with matched per-capability
   instruments (§5).
3. **Reusable apparatus**: a locale-aware, yield-aware generation pipeline
   (§3) and a *partially validated* trigger-position decorrelation
   construction for the persistence sentinel (§5).

All experiments run on one RTX 3060 12GB GPU — the regime where the boundary
matters most, since a large general-purpose model is not deployable there. We
evaluate one family (Qwen; a genuine limitation, §6) and one locale (which
does not limit the persistence/withholding boundary, both structural
behaviors independent of the locale backdrop).
