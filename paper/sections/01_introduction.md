# 1. Introduction

**Problem.** We ask, for each behavior a deployed English tutor needs,
whether a sufficiently explicit system prompt is enough or the behavior
must be demonstrated in fine-tuning. A deployed tutor is governed by a
long system prompt that already *states* most of what it should do —
stay in character, ground culture in the learner's locale, refuse role
swaps, give one short example rather than a grammar lecture, escalate to
a session-end signal under sustained abuse. If stating a behavior were
sufficient, the data-side problem would be trivial.

**Why it matters.** It is not trivial: the behaviors split cleanly into
ones a prompt clause elicits and ones it only describes, and knowing
which is which is exactly the decision a practitioner faces when building
a task-specific model on consumer hardware — spend the data-collection
and training budget only where a prompt will not do. The tutor setting
makes this question both unavoidable and answerable, because each target
behavior corresponds to a clause already present in a realistic
deployment prompt (§3.5 reproduces ours): pedagogical appropriateness is
level-specific, redirect behavior spans several *interaction invariants*
(language of instruction, lesson topic, role structure, persona,
pedagogical contract, locale frame), and cultural fit must resist the
Western-default references pretraining drives toward. This lets us ask,
clause by clause, whether the clause suffices.

**Gap in prior work.** The prompting-versus-fine-tuning question has been
studied at the level of general alignment — that a small, high-quality
demonstration set can install broad instruction-following
[@zhou2023lima; @ouyang2022instructgpt], and that in-context demonstrations
often convey format more than new capability [@min2022rethinking] — but not
resolved *per behavior* for a deployed task model. Synthetic-instruction
pipelines (§2.1) and tutor-LLM datasets (§2.3) target general
instruction-following with a uniform recipe; none asks, per behavior,
whether the deployment prompt already elicits what the data teaches.
Single-turn safety and persona work (§2.4) treats redirection as
one-prompt-one-refusal and does not address multi-turn persistence or the
per-axis promptability of redirects.

We fill this gap with a **matched-prompt per-capability evaluation**: the
same fully-specified deployment prompt — every redirect-axis instruction
and the full three-strike persistence protocol — is supplied at evaluation
to a fine-tuned 0.8B student and to a ladder of prompt-only baselines (0.8B
base, 0.8B instruct, 4B instruct, and the 9B teacher that generated the
training data). Holding the instruction fixed across conditions is what
makes a prompt-only failure interpretable: it isolates whether the behavior
is *promptable* at all (the full logic is in §5.1).

**Contribution.** Our contribution is a single one: **a per-capability map
of the train-versus-prompt boundary** for tutor redirect behavior,
established under a matched-prompt protocol that makes prompt-only failures
interpretable. The behaviors separate along an interpretable line:
*promptable* when a single clause both *describes* and *elicits* the
behavior (locale fidelity, role-swap and topic re-anchoring at the level of
response type — prompt-only reaches parity with the trained student), and
*not promptable* when the behavior requires cross-turn state-tracking
(multi-turn persistence) or the suppression of a strong competing prior
(pedagogical withholding), which a clause can name but not produce.

In support of that map — not as separate contributions — we also report an
evidenced negative result on the conventional metric (context-blind
redirect-axis F1 is a *type* classifier that ties a 0.8B student with the
9B teacher at 0.409, while a quality-aware pairwise eval finds the student
preferred on every axis; §5.5, Appendix A), and release the reusable
apparatus the study is built on: a locale-aware, yield-aware generation
pipeline (§3, with a `locale_judge` FP audit, §6.4) and a *partially
validated* trigger-position decorrelation construction (§5.3.1).

**Results.** Under the matched prompt, the two not-promptable behaviors
fail prompt-only and are installed by SFT. Persistence resists zero-shot
and few-shot prompting on the 9B teacher (recall $\leq 0.06$) and is only
partially recovered by native chain-of-thought (0.63, still below the
trained student's 0.83 and at 1.6–3.2k reasoning tokens per turn), while
the no-specialized-data ablation fires 0% of the time. Withholding stays at
0.09–0.45 prompt-only (9B teacher 0.45) against the trained student's 0.61
(two judges, n=63), collapsing to near the untrained-base rate when the
pedagogy stream is ablated. The promptable axes reach prompt-only parity.

**The boundary replicates in a second trained family.** A Llama-3.2-1B
student, evaluated against its *own* untrained base, lifts *both*
not-promptable behaviors far above prompt-only — persistence 0.25→0.91 and
withholding 0.11→0.50 — so the effect is training, not scale (student and
control share one base). A larger Llama-3.1-8B prompt-only probe stays low
even with chain-of-thought; the direction is robust across families, the
magnitude family-dependent (§6.2). The load-bearing conditions (A1, A3) are
reported over three seeds with mean$\pm$s.d., the withholding contrasts
carry per-judge two-proportion tests, and the pairwise win-rates carry
bootstrap CIs (§4.8, Table~\ref{tab:stat-summary}, §6.1).

**Scope and non-goals.** We focus on the *data side* and treat the
training recipe as fixed (QLoRA SFT). We do not contribute to
CEFR-leveling itself. We evaluate with a single base/teacher family
(Qwen) — a genuine limitation (§6.2) — and at a single locale
(`china`), which does *not* limit the central claim: persistence and
withholding are structural behaviors independent of the locale backdrop,
so the boundary for them is locale-independent by construction (§6.2).
The 0.8B student on RTX 3060 12GB is a deliberate choice: the boundary is
most consequential precisely where a large general-purpose model is not
deployable.

**Paper organization.** §2 surveys related work. §3 presents the
pipeline and reproduces the deployment prompt (§3.5). §4 specifies the
matched-prompt experimental setup. §5 reports the per-capability boundary
results. §6 discusses limitations, confounds, and methodological lessons.
§7 concludes.
