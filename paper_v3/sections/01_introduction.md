# Don't Trim the Tail: Sequence Truncation Weakens Threshold Learning for Rare Control Tokens

## Abstract

Fine-tuned language models are increasingly trained to emit rare, machine-consumed
control markers — tokens that end a session, invoke a tool, or trigger a refusal —
on a semantic *threshold* condition. We show that the *shape* of the supervised
fine-tuning data, not model scale, governs both *when* such a marker fires and
*why* the model says it fires, and that two common curation choices have large,
sometimes counter-intuitive effects. First, *trimming* each training sequence to
end at the marker — a practice that intuitively should sharpen marker learning —
instead induces *premature firing*: across every position×marker design we test,
trimming raises the premature-firing rate by 0.44–0.58, a main effect far larger
than the design choices it is usually bundled with. We find the mechanism is not a
turn-position shortcut but threshold-laxity: trimming removes the training examples
in which the marker's trigger feature is present but *not* followed by firing,
leaving the trigger nearly always predictive and impairing the model's ability to
threshold on its magnitude; retaining a benign post-marker continuation restores
correctly-timed firing. Second, a *typed* marker acts as a *semantic gate*: models
trained to emit an axis-labeled marker attribute the correct violated invariant on
the large majority of fires and never emit a contentless marker, whereas generic
markers carry no such signal — and attribution survives the trim that collapses
timing, showing *when* and *why* a marker fires are separable, separately-curated
behaviors, even when a second sub-threshold violation is present as a distractor.
Third, the effect is not corpus-specific: it replicates in a different domain on a
different model family (Llama-3.2), so the vulnerability is a general property of
next-token training on trimmed sequences at a rare, count-triggered semantic
marker, not an artifact of one dataset. These are cheap, low-overhead
data-curation principles for reliable control-token emission.

*Keywords:* large language models; supervised fine-tuning; data curation; control
tokens; sequence truncation; threshold learning; tool calling; agentic systems.

---

# 1. Introduction

<!-- paper_v3 — "Don't Trim the Tail" -->

Large language models increasingly serve as decision-making components inside
larger software systems rather than as standalone text generators, coordinating
external tools, agents, retrieval pipelines, and dialogue workflows through
**rare structured control markers** — tokens that signal an external system to
take an action: end a session, hand off to a tool, refuse, or escalate. Unlike
ordinary generation, a control marker is an *execution signal*: it has a *trigger
condition* (fire when, and only when, some semantic threshold is met) and its
value is consumed by machinery, not read by a human. Emitting one too early, too
late, or under the wrong conditions can alter system behavior even when the
surrounding text remains fluent — so getting the *timing* wrong and the *content*
wrong (firing without a usable reason) are distinct failure modes with distinct
costs.

This paper shows that both failure modes are governed by the **shape of the
SFT data**, not by model scale, and that two common curation choices have large,
measurable, and in one case counter-intuitive effects.

**Contribution 1: the terminal-position / trim artifact.** A common
practice when training a model to emit a rare marker is to *trim* each training
sequence to end at the marker — it feels like it should sharpen marker learning
by removing distracting continuation. We show it does the opposite. Trimming
deletes every training example in which the marker's trigger feature is present
but *not* followed by firing, leaving the trigger perfectly predictive of the
marker and impairing the model's ability to threshold on the trigger's
magnitude. On a four-axis session-ending task, trimming raises premature-firing
rate by **+0.44 to +0.58 across every design tested** — a main effect far larger
than any other lever we study. Retaining the benign post-marker continuation
restores correctly-timed firing. <!-- F-B: proven Phase 0 -->

**Contribution 2: typed markers as semantic gates.** A *typed* marker
(`[SESSION_END: <axis>]`) forces the model to attribute firing to a specific
violated invariant; a *generic* marker (`[SESSION_END]`) does not. Typed-trained
models name the correct axis on **94–99%** of fires and never emit a contentless
marker, turning session-ending into an accurate, actionable classification;
generic-trained models carry no such signal by construction. Attribution is
robust to the trim manipulation even where timing collapses, establishing that
*when* a model fires and *why* it says it fires are separable, separately
curated behaviors. The gate survives the decisive test: under a *distractor*
axis (a second, sub-threshold violation in the conversation) typed models still
name the axis at threshold on 96–98% of fires and are pulled to the distractor
under 2% of the time. <!-- F-A: proven Phase 0 + Phase 1 -->

**Contribution 3 (generalization): the effect is not corpus-specific.** We
replicate trim→premature in a **different domain on a different model family** —
fine-tuning Llama-3.2-1B-Base on a synthetic customer-support escalation task —
where the trimmed model fires the escalation marker prematurely more than the
untrimmed one (0.830 vs 0.683, both at full recall), the same direction as the
tutor result. The vulnerability is a general property of next-token training on
trimmed sequences at a rare, count-triggered semantic marker, not an artifact of
one dataset or one model lineage.

We situate these as **data-curation principles for rare control tokens**. We also
sketch a candidate remedy — annotating the marker with its strike count to make
the latent counter an explicit supervised target — but leave a recall-cleared
evaluation of it to follow-up work (§5.5).
