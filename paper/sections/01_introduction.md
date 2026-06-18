# 1. Introduction

Building a dialogue model that can act as a competent English tutor
is a problem with three properties that make standard
instruction-tuning data inadequate. First, **pedagogical
appropriateness is level-specific**: vocabulary, grammar, and
scaffolding shape that work for a C1 learner are wrong for an A1
learner, and vice versa. Second, **safe-and-graceful redirect
behavior is multi-axis**: a tutor is the keeper of several
*interaction invariants* — the language of instruction, the lesson
topic, the role structure, the tutor persona, the pedagogical
contract, the locale frame — and a learner can violate any one of
them, each calling for a structurally different repair. Third,
**cultural fit matters**: a tutor deployed to learners in a
non-Western locale must avoid Western-default references that the
model's general pretraining corpus drives it towards.

A small but visible body of recent work has produced English-tutor
SFT datasets, typically as a flat collection of "good" tutor
dialogues at one or two CEFR levels. None of the public corpora we
are aware of separately covers the multi-axis redirect behaviors a
deployed tutor must handle, and most do not address locale fidelity
at all. The result is that a model fine-tuned on such corpora is
pedagogically competent in the common case but brittle in the long
tail: it fails (or fails gracefully but inconsistently) when the
learner deviates from the canonical script.

We present a data-generation pipeline aimed squarely at this long
tail. The pipeline produces three artefacts from a single
locally-served teacher model: a SFT corpus stratified across six
CEFR levels and twelve behavior streams; a DPO preference-pair
corpus mixing register-style pairs with student-vs-teacher
on-policy pairs; and a `<think>`-mode evaluator-example corpus.

Two of our contributions are stated as *transferable design
principles* that the tutor pipeline happens to instantiate, because
we believe they apply beyond tutoring; the third is the practical
pipeline that operationalizes them at consumer scale.

## Contributions

1. **An invariant-based taxonomy of tutor redirect behavior,
   instantiated as a 12-stream SFT corpus.** We model a tutor not as
   a producer of undifferentiated "good dialogue" but as the keeper
   of a set of *interaction invariants*. A learner violation breaks
   exactly one invariant, and the correct redirect is the *minimal
   repair* that restores it; two violations occupy distinct axes
   **iff their minimal repairs differ in shape**. This criterion
   fixes the granularity of the taxonomy rather than leaving it to
   intuition, and it turns the central claim — that a single generic
   "redirect" stream is insufficient — into a *prediction* (distinct
   invariants need distinct repairs) that we test directly in §5.4
   rather than merely assert. We instantiate the taxonomy as one
   normal stream, seven single-shot redirect streams (one per
   invariant: locale, pedagogy, language, persona, topic, role swap,
   and a generic appropriateness catch-all), and four persistent
   3-strike streams (off-topic, language violation, persona break,
   role swap), all stratified across CEFR levels A1–C2.

2. **A trigger-position decorrelation construction for learning
   rare structured markers, instantiated as a 4-variant persistent
   design.** When a model must emit a rare structured marker
   conditional on a *semantic* trigger, but that marker is
   *positionally regular* in the training data, the model learns the
   position as a proxy for the trigger — an instance of the
   shortcut-learning failure mode (§2.4). Our persistent 3-strike
   streams have exactly this structure: a sentinel must fire on the
   third same-axis violation, but a fixed-turn design teaches "fire
   at turn 7" instead. We decorrelate marker position from trigger
   by drawing the sentinel turn from a *seeded pseudo-random
   function of the record id* over the feasible, parity-constrained
   position set {5, 7, 9, 11}. The seeding is not incidental: it is
   forced by three pipeline invariants — resumability,
   train/eval-split coherence, and turn-parity — that rule out naive
   randomization (§3.4). The construction transfers to any setting
   with a rare, semantically-triggered, positionally-regular marker
   (refusal triggers, tool-call emission, agentic stop conditions).

3. **A locale-aware, yield-aware generation pipeline that
   operationalizes the taxonomy on consumer hardware.** The pipeline
   contributes several practical components that we report but do
   not claim as conceptual novelty: a two-variant locale-instruction
   block (a strict-Latin default and an allow-L1 variant for the
   `language_redirect` stream, without which that stream has a ~0%
   pass rate); a six-filter cascade combining cheap mechanical
   filters with an LLM-judge `locale_judge`; and a declarative,
   yield-aware top-up loop that lets the operator specify per-stream
   mix shares as either ratios or absolute floors without
   recomputing arithmetic as yields shift. We further surface one
   reusable methodological lesson from this pipeline: an audit of the
   `locale_judge` reveals systematic false positives (~85% of its
   rejections) on common English sentence-initial words and
   locally-canonical landmarks, remediable with static allowlists
   (global pass rate 70.1% → 88.4%; §6.5). We single this out because
   it generalises to any capitalization-based entity filter, not
   because the fix is deep.

We demonstrate the pipeline empirically by training a 0.8B-parameter
student on RTX 3060 12GB consumer hardware. Held-out evaluation
across four test sets (Tutor-Scenario, Redirect-Probe,
Persistent-Probe, and Locale-Leakage) shows that the full pipeline
yields a student that (i) on the **fully-mechanical sentinel-firing
metric** — our strongest evidence, since it consults no judge —
sharply outperforms a same-size ablation trained without the
4-variant persistent design; (ii) on **locale-leakage rate**, also
mechanical, sharply outperforms a same-size off-the-shelf instruct
model; (iii) on **redirect-axis F1** sharply outperforms the same
baseline; and (iv) on **naturalness on Tutor-Scenario** approaches
the 9B-teacher upper bound. (Numbers in §5.)

## Scope and explicit non-goals

We focus on the *data side* of the tutor problem. We treat the
training recipe as fixed (QLoRA SFT followed by DPO) and do not
explore alternative training objectives. We do not contribute to
the CEFR-leveling problem itself — we assume the teacher's CEFR
adherence is well-calibrated and let the filter cascade catch
egregious violations. We evaluate at a single locale (`china`); the
pipeline supports multi-locale generation via `config/locale.yaml`
but we leave the multi-locale empirical study to future work.

The student size and hardware are also a deliberate scope choice:
we run all experiments on a 0.8B base model trained with QLoRA on
RTX 3060 12GB. This is a feature rather than a constraint. We
believe a real argument for taxonomy-based tutor-data pipelines is
that they let small consumer-deployable models do work that
otherwise requires a much larger general-purpose model. We measure
distillation effectiveness by comparing the trained student against
the 9B teacher used to produce its training data.

## Paper organisation

§2 surveys related work on synthetic instruction data, LLM-judge
filtering, persona/safety adversarial data, shortcut learning, and
prior tutor-dataset work. §3 presents the pipeline. §4 specifies the
experimental setup. §5 reports results and ablations. §6 discusses
limitations and methodological lessons. §7 concludes.
