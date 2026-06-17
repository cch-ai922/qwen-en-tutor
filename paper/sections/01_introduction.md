# 1. Introduction

Building a dialogue model that can act as a competent English tutor
is a problem with three properties that make standard
instruction-tuning data inadequate. First, **pedagogical
appropriateness is level-specific**: vocabulary, grammar, and
scaffolding shape that work for a C1 learner are wrong for an A1
learner, and vice versa. Second, **safe-and-graceful redirect
behavior is multi-axis**: a tutor must respond differently when the
learner introduces a locale-violating cultural reference, when the
learner code-switches into L1, when the learner attempts to break
the tutor's persona, when the learner persists in off-topic drift,
and so on. Third, **cultural fit matters**: a tutor deployed to
learners in a non-Western locale must avoid Western-default
references that the model's general pretraining corpus drives it
towards.

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

## Contributions

1. **A 12-stream SFT taxonomy** that decomposes tutor-side dialogue
   into one normal stream, seven *single-shot redirect* streams
   (one per violation axis: locale, pedagogy, language, persona,
   topic, role swap, and a generic catch-all), and four
   *persistent 3-strike* streams (off-topic, language violation,
   persona break, role swap). The taxonomy lets us train a tutor
   that responds with the correct *shape* of redirect for each axis
   rather than collapsing all redirects into a single learned
   pattern.

2. **A 4-variant structural design for persistent abuse handling**
   that defeats the positional shortcut a student would otherwise
   learn from fixed-turn sentinel training. Each persistent dialogue
   places the sentinel-firing turn at one of four positions (5, 7,
   9, or 11), chosen hash-deterministically from the seed id. This
   forces the student to learn "third strike on the same axis"
   rather than "turn 7."

3. **A yield-aware top-up loop with declarative ratio targets.**
   The operator specifies per-stream targets as either an absolute
   per-level floor or a percentage-of-mix ratio; the loop
   computes the absolute target for ratio-specified streams via
   $T_{\text{per\_level}} = \sum_{s \in \text{abs}} t_s \big/ (1 -
   \sum_{r \in \text{ratio}} r_r)$ and iterates per-cell until the
   target is reached or `MAX_ROUNDS` is hit. The same equation lets
   the operator declare "I want `normal` to be exactly half the
   mix" without recomputing arithmetic when other streams' floors
   change.

4. **Locale-aware prompt engineering** with two variants of the
   locale-instruction block: a strict-Latin variant for streams
   where learner turns must be in English, and an allow-L1 variant
   for the `language_redirect` stream where the learner intentionally
   code-switches. Without this split, the language_redirect stream
   has a ~0% pass rate because its content contradicts the strict-
   Latin rule the default block imposes.

5. **A six-filter cascade** combining cheap mechanical filters with
   an LLM-judge `locale_judge`. We further note an engineering
   caveat for the locale_judge in §6 — a capitalization-based entity
   extractor produces systematic false positives on common English
   sentence-initial words, easily remediated with an allowlist — but
   we do not consider this a research contribution.

We demonstrate the pipeline empirically by training a 0.8B-parameter
student on RTX 3060 12GB consumer hardware. Held-out evaluation
across four test sets (Tutor-Scenario, Redirect-Probe,
Persistent-Probe, and Locale-Leakage) shows that the full pipeline
yields a student that (i) on the **fully-mechanical sentinel-firing
metric** sharply outperforms a same-size ablation trained without
the 4-variant persistent design; (ii) on **redirect-axis F1** sharply
outperforms a same-size off-the-shelf instruct model; (iii) on
**naturalness on Tutor-Scenario** approaches the 9B-teacher upper
bound; and (iv) on **locale-leakage rate** sharply outperforms the
same-size off-the-shelf instruct model. (Numbers in §5.)

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
filtering, persona/safety adversarial data, and prior tutor-dataset
work. §3 presents the pipeline. §4 specifies the experimental
setup. §5 reports results and ablations. §6 discusses limitations
and methodological lessons. §7 concludes.
