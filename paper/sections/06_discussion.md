# 6. Discussion

## 6.1 The naturalness gap at C1 / C2

The pipeline's clearest remaining weakness is C2 naturalness. In
§5.1 we report that the naturalness filter is responsible for
113 of the 363 post-audit rejections at C2 (~31% of C2 rejections)
and 50 of 178 rejections at C1. The 9B teacher itself produces
stilted-sounding C2 dialogues at a non-trivial rate; the filter
catches the worst offenders but the corpus quality at C2 is
materially lower than at A2.

Three forces drive this. First, **C2 dialogue requires the teacher
to switch into low-grade native speech**, and the teacher (a
post-trained chat model) is biased toward over-formal speech that
reads as stilted in casual contexts. Second, **the rendered system
prompt is the same length at every CEFR level**, which means the
share of context spent on level instruction shrinks as the desired
sophistication of the output increases. Third, **the naturalness
filter is heuristic** (perplexity bands, repetition heuristics)
rather than learned; some C2 turns that are perfectly natural to a
human reader fall outside the filter's bands because C2 vocabulary
has lower base-rate perplexity than A1 vocabulary.

A possible mitigation is to train a learned naturalness filter on
human-rated transcripts and add it as a seventh filter. We leave
this to future work.

## 6.2 Single-seed reporting and statistical rigor

We report a single training seed for each of the four trained
conditions. Standard practice for SFT/DPO ablation studies is three
or more seeds with mean $\pm$ std and a paired significance test. We
explicitly do not have those numbers. The compute cost of three
seeds for four conditions on RTX 3060 is large enough that we
chose to spend that budget on the four conditions instead of three
seeds of one condition.

The honest reading is that our reported numbers are point
estimates whose seed-to-seed variance is unknown. For the
mechanical metrics (sentinel firing, locale leakage) the variance
is bounded by the test-set bootstrap CIs we report. For judged
metrics the variance is partially captured by the inter-judge
agreement we report. Reviewers reading this should treat the
mechanical metrics with more confidence than the judged ones, and
the locale_judge FP audit (§6.5) as the strongest finding
independent of single-seed concerns. We note that our two headline
*conceptual* claims rest on the firmest evidence available here:
the trigger-position decorrelation claim is tested by a fully
mechanical metric (sentinel firing), and the invariant-decomposition
claim is a falsifiable per-axis prediction whose direction (A1 $\gg$ A3
on unseen axes, A1 $\approx$ A3 on the shared generic axis) does not depend
on absolute judged scores.

## 6.3 Threats to validity

**Single locale.** All experiments are at `locale=china`. The
pipeline supports multi-locale generation but we have not
empirically demonstrated that the locale-aware prompt engineering
transfers to e.g. `japan` or `italy`. The mechanical filter
(locale-leakage gazetteer) would need to be re-built for each new
locale.

**Single base model and teacher.** The student is Qwen3.5-0.8B-Base
and the teacher is Qwen3.5-9B-UD-Q4_K_XL. Both come from the same
model family, which means our distillation results reflect
intra-family distillation and may not transfer to e.g.
Llama-3 $\to$ Qwen3.5 or vice versa. Multi-family experiments are
left to future work.

**Self-preference judge bias (flagged, mitigated, not eliminated).**
The self-preference failure mode
[@panickssery2024selfpreference] — a model scoring its own family's
outputs more favourably than an out-of-family judge would — is a
well-documented threat for any work that uses an LLM ensemble to
evaluate a small student. We mitigate at the ensemble level: the
three judges in §4.7 (Prometheus-7B-v2, Mistral lineage;
Llama-3.1-8B-Instruct, Meta; Gemma-2-9B-it, Google) are drawn from
three model families *all distinct from the teacher's Qwen family*,
so the student is never scored by a checkpoint that shares its
pre-training corpus, tokenizer, or post-training recipe with the
teacher that produced its supervision data. The Qwen3.5-9B teacher
appears in the paper only as the data generator and as zero-shot
upper-bound baseline B4 — never as a judge. This is a stricter
judge-independence than is standard practice in tutor-LLM
evaluations, where at least one same-family judge is typical.

Two residual considerations remain, which we flag rather than
claim away. (i) Cross-family judges still carry their own
stylistic priors — Prometheus's rubric protocol prefers
explicit-criterion language, Llama-3.1 tends to reward longer
responses, Gemma-2 has its own register preferences — and the
ensemble median controls only the *family-correlated* component of
bias, not these individual priors. (ii) All three judges are
post-trained on instruction-following corpora that overlap
non-trivially with the data sources our teacher itself was trained
on, so "no shared family" does not imply "no shared training data."
We treat the cross-family ensemble as the strongest practical
mitigation available within the RTX-3060 12 GB compute envelope
and report inter-judge Krippendorff $\alpha$ and 100-sample
human-validation correlation (§5.7) as transparency on the
residual. The mechanical metrics (sentinel firing, locale leakage)
remain entirely judge-free and so untouched by this concern.

**Repair-shape vs intent detection.** The redirect-axis F1 metric
classifies the *produced response* by repair shape (§4.5), which is
a proxy for whether the model produced the correct minimal repair.
A model could in principle name the right axis while producing a
mis-shaped repair, or vice versa; we mitigate by classifying the
response rather than any explicit intent label, but the proxy is
not perfect and per-axis results (§5.4) should be read accordingly.

**Hardware-constrained scope.** All training fits on RTX 3060 12GB
via QLoRA. We argue this is a deployment-relevant choice: the
result is a 0.8B model on consumer hardware. A larger student
trained on the same data might capture more of the teacher's
capability; we have not measured this.

## 6.4 What we would do differently with more compute

Given a 40-hour A100-class budget rather than RTX 3060, the changes
that would most improve the paper are, in priority order:

1. **Three training seeds per condition** with paired significance
   tests on every judged metric.
2. **Larger student** (4B or 7B) to measure how each contribution
   scales with student capacity. The invariant framing makes a
   concrete prediction here: the redirect-taxonomy contribution
   should *shrink* at larger capacities, because a larger model can
   infer axis-specific repair shapes from a generic redirect stream,
   whereas the trigger-position decorrelation contribution should
   *hold*, because positional shortcut-learning is a data-structure
   artifact rather than a capacity limitation. The §5.3.3
   per-CEFR stratification adds a quantitative prediction for the
   *recall* dimension: the V3 / V4 dip (60% vs ~87% at V1 / V2)
   is uniform across CEFR and well under the SFT max-length cap, so
   it reads as a 0.8B capacity-at-distance limit rather than a data
   artefact. We predict a 4B student on the identical 4-variant
   corpus should recover most of that gap, closing V3 / V4 recall
   from 60% to $\geq$75% without any change to data or recipe; FP-rate
   should remain at zero and OffPosition recall at or above 70%.
3. **A learned naturalness filter** to address the C2 gap.
4. **Multi-locale empirical evaluation**, repeating the pipeline
   end-to-end for `japan` and `italy`.

We name these explicitly so that a reviewer's "but what about X"
intuition is met with a concrete answer rather than silence.

## 6.5 Engineering caveat: locale_judge false positives

Our `locale_judge` filter uses a capitalization-based proper-noun
extractor: it scans for sentence-initial capitalised tokens and
multi-token capitalised phrases and asks the teacher to classify
each candidate as in- or out-of-locale. In our initial run this
extractor produced a high false-positive rate on common English
sentence-initial words (`Plus`, `Line`, `Will`, `May`), on
locally-canonical landmarks (`West Lake`, `Drum Tower`, `Muslim
Quarter`), and on universal tools (`Python`, `Google Maps`). We
remediated by extending two static allowlists — one for common
English sentence-initial words and one for known-in-locale entities
— and pass rate rose from 70.1% to 88.4%.

We report this as an engineering caveat rather than as a research
contribution. The failure mode is a consequence of the extractor
choice; a different extractor (a learned NER model, or asking the
LLM to enumerate entities directly) would not have these specific
false positives. The reusable lesson is not the fix but the
discipline: practitioners building similar pipelines should budget
time to audit their entity-extractor rejection log regardless of
which extractor they choose, because a filter that silently discards
~57% of records — most of them good — can quietly halve a corpus
before anyone inspects the rejections.
