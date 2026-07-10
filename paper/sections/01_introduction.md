# 1. Introduction

**Problem.** We ask, for each behavior a deployed English tutor needs,
whether a sufficiently explicit system prompt is enough or the behavior
must be demonstrated in fine-tuning. A deployed tutor is governed by a
long system prompt that already *states* most of what it should do —
stay in character, ground culture in the learner's locale, refuse role
swaps, give one short example rather than a grammar lecture, escalate to
a session-end signal under sustained abuse. If stating a behavior were
sufficient, the data-side problem would be trivial.

**Why it matters.** Knowing which behaviors a prompt clause elicits and which it
only describes is exactly the decision a practitioner faces when building a
task-specific model on consumer hardware — spend the data-collection and training
budget only where a prompt will not do. The tutor setting makes this question both
unavoidable and answerable, because each target behavior corresponds to a clause
already present in a realistic deployment prompt (§3.5): redirect behavior spans
several *interaction invariants* (language, topic, role, persona, pedagogical
contract, locale), each a distinct clause we can test in isolation.

**Gap in prior work.** The prompting-versus-fine-tuning question has been
studied at the level of general alignment — a small demonstration set can install
broad instruction-following [@zhou2023lima; @ouyang2022instructgpt], and in-context
demonstrations often convey format more than new capability [@min2022rethinking] —
but not resolved *per behavior* for a deployed task model. Synthetic-instruction
pipelines (§2.1), tutor-LLM datasets (§2.3), and single-turn safety/persona work
(§2.4) target general instruction-following or one-prompt-one-refusal redirection;
none asks, per behavior, whether the deployment prompt already elicits what the
data teaches, nor addresses multi-turn persistence.

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
interpretable. Under the tested small-model and prompting regimes, the behaviors
separate into **three operational regimes** rather than a binary: **(A)
prompt-sufficient** — a single clause both describes and elicits the behavior and
data adds little (locale-leakage fidelity); **(B) prompt-elicitable but
data-refinable** — the behavior *type* is elicited by the clause (role-swap and
topic re-anchoring reach prompt-only type-parity), yet specialized data materially
improves its *quality*; and **(C) training-dependent under the tested regimes** —
the behavior requires cross-turn state-tracking (multi-turn persistence) or the
suppression of a strong competing prior (pedagogical withholding), which a clause
names but the tested prompts do not reliably elicit. Regime B is what a binary
promptable/not-promptable framing hides.

In support of that map — not as separate contributions — we report an evidenced
negative result on the conventional metric (context-blind redirect-axis F1 ties
student and 9B teacher at 0.409, while a quality-aware pairwise eval prefers the
student on every axis; §5.5, Appendix A), and release the reusable generation
pipeline the study is built on (§3, with a `locale_judge` FP audit, §6.4) plus a
*partially validated* trigger-position decorrelation construction (§5.3.1).

**Results.** Under the matched prompt, the two regime-C behaviors fail
prompt-only in the tested regimes and are installed by SFT. Persistence resists
zero-shot and few-shot prompting on the 9B teacher (recall $\leq 0.06$) and is only
partially recovered by native chain-of-thought (0.63, still below the trained
student's 0.83 and at 1.6–3.2k reasoning tokens per turn), while the
no-persistence-data ablation A3 fires 0% of the time (A3 trains on neither the
specialized nor the persistent streams, §5.1). Withholding stays at 0.09–0.45
prompt-only (9B teacher 0.45) against the trained student's 0.61 (two judges,
n=63), collapsing to near the untrained-base rate when the specialized streams are
ablated. The trained student's persistence survives a full error taxonomy
(precision 0.795, balanced accuracy 0.885, MCC 0.758, benign FPR 0.000; §5.3), so
its recall is genuine discrimination, not over-firing. The regime-A/B axes reach
prompt-only type-parity.

**The regime-C effects replicate in a second trained family.** A Llama-3.2-1B
student, evaluated against its *own* untrained base, lifts *both* regime-C
behaviors far above prompt-only — persistence 0.25→0.91 and withholding 0.11→0.50.
Because the student and its control share one base and differ only in training,
this separates *training from scale within that base* (it is not a claim about
scale independence across model sizes, which we do not test). A larger
Llama-3.1-8B prompt-only probe stays low even with chain-of-thought; the direction
is robust across families, the magnitude family-dependent (§6.2). The load-bearing
conditions (A1, A3) are reported over three seeds with mean$\pm$s.d., the
withholding contrasts carry per-judge paired McNemar tests and paired bootstrap
CIs, and the pairwise win-rates carry bootstrap CIs (§4.8,
Table~\ref{tab:stat-summary}, §6.1).

**Scope and non-goals.** We focus on the *data side* and treat the
training recipe as fixed (QLoRA SFT). We do not contribute to
CEFR-leveling itself. We evaluate with a single base/teacher family
(Qwen) — a genuine limitation (§6.2) — and at a single locale
(`china`). We *hypothesize* the persistence and withholding boundaries are less
locale-dependent than the redirect axes, because their formal trigger definitions
(count same-axis strikes; withhold the requested answer) contain no locale-specific
variables; but single-locale evaluation still limits external validity, and we do
not claim locale-independence as established (§6.2).
The 0.8B student on RTX 3060 12GB is a deliberate choice: the boundary is
most consequential precisely where a large general-purpose model is not
deployable.

**Paper organization.** §2 surveys related work. §3 presents the
pipeline and reproduces the deployment prompt (§3.5). §4 specifies the
matched-prompt experimental setup. §5 reports the per-capability boundary
results. §6 discusses limitations, confounds, and methodological lessons.
§7 concludes.
