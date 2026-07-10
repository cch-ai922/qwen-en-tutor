# Don't Trim the Tail: Sequence Truncation Weakens Threshold Learning for Rare Control Tokens

## Abstract

Fine-tuned language models are increasingly trained to emit rare,
machine-consumed control markers — tokens that end a session, invoke a tool, or
trigger a refusal — on a semantic threshold condition. We show that the shape of
the supervised fine-tuning data, with model family and training budget held fixed,
has a first-order effect on both when such a marker fires and what reason the
model encodes for firing. In a fully crossed factorial study of marker position,
marker format, and sequence trimming, we find that trimming each training sequence
to end at the marker — a practice one might expect to sharpen marker learning —
instead raises premature firing by 0.44 to 0.58 across every design, the largest
main effect in the study. The evidence is most consistent with a threshold-laxity
mechanism: trimming deletes the training turns that show the trigger present but
not yet followed by firing, so the model stops thresholding on the trigger's
magnitude, and retaining a benign post-marker continuation substantially reduces
premature firing. Separately, an axis-typed marker acts as a semantic gate,
attributing the correct violated invariant on the large majority of fires,
including under a distractor, whereas a generic marker carries no such signal. The
effect replicates in direction in a second domain and model family. Together these
yield cheap data-curation principles for reliable control-token emission.

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

This paper shows that both failure modes are strongly influenced by the **shape of
the SFT data** — with model family and training budget held fixed — and that two
common curation choices have large, measurable, and in one case counter-intuitive
effects.

**Contribution 1: the terminal-position / trim artifact.** A common
practice when training a model to emit a rare marker is to *trim* each training
sequence to end at the marker; one might expect this to sharpen marker learning
by removing distracting continuation. We show it does the opposite. Trimming
deletes every training example in which the marker's trigger feature is present
but *not* followed by firing, leaving the trigger perfectly predictive of the
marker and impairing the model's ability to threshold on the trigger's
magnitude. In a fully crossed 2×2×2 (position × marker-format × trim) design on a
four-axis session-ending task, trimming raises premature-firing rate by **+0.44 to
+0.58 across every design tested** — the largest observed main effect in the tested
design, confirmed by an item-level factorial logistic model in which trim is the
largest term by a wide margin (odds ratio ≈9.6) and every interaction with
position and marker format is non-significant. Retaining the benign post-marker continuation
substantially reduces premature firing. <!-- F-B: primary trim study -->

**Contribution 2: typed markers as semantic gates.** A *typed* marker
(`[SESSION_END: <axis>]`) forces the model to attribute firing to a specific
violated invariant; a *generic* marker (`[SESSION_END]`) does not. Typed-trained
models name the correct axis on **94–99%** of fires and never emit a contentless
marker, turning session-ending into an accurate, actionable classification;
generic-trained models carry no such signal by construction. Attribution is
robust to the trim manipulation even where timing is substantially worsened,
establishing that *when* a model fires and *why* it says it fires are separable,
separately curated behaviors. Typed markers support robust per-axis attribution
under a sub-threshold distractor: with a second, sub-threshold violation present
in the conversation, typed models still name the axis at threshold on 96–98% of
fires and are pulled to the distractor under 2% of the time. <!-- F-A: trim study + distractor test -->

**Contribution 3 (generalization): the effect is not specific to the tutoring
corpus.** We replicate trim→premature *in direction* in a **different domain on a
different model family** — fine-tuning Llama-3.2-1B-Base on a synthetic
customer-support escalation task, where the agent must emit an escalation marker
after the third *explicit* escalation request (angry-but-non-requesting messages
are distractors that must not count). The trimmed model fires the escalation
marker prematurely more than the untrimmed one (0.830 vs 0.683, both at full
recall), the same direction as the tutor result. This suggests the vulnerability
is a property of next-token training on trimmed sequences at a rare,
count-triggered semantic marker, not an artifact of one dataset or one model
lineage; we make the narrower directional claim rather than a claim of universal
magnitude, since both settings are small synthetic tasks (§6.6).

Together these results form **data-curation principles for rare control tokens**.
We additionally sketch a candidate remedy — annotating the marker with its strike
count to make the latent counter an explicit supervised target — and leave its
evaluation to future work (§6.6).
