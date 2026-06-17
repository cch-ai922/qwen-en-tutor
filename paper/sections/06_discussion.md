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
or more seeds with mean ± std and a paired significance test. We
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
the locale_judge FP audit (§5.1) as the strongest finding
independent of single-seed concerns.

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
Llama-3 → Qwen3.5 or vice versa. Multi-family experiments are
left to future work.

**Self-preference judge bias.** Three of our three judges come
from the Qwen family. When judging Qwen-family outputs, this
introduces a known bias toward Qwen-style responses
[@panickssery2024selfpreference]. The mechanical metrics (sentinel
firing, locale
leakage) do not have this bias because they do not consult a
judge. The judged metrics (CEFR adherence, redirect F1,
naturalness) should be read with this caveat.

**Hardware-constrained scope.** All training fits on RTX 3060 12GB
via QLoRA. We argue this is a deployment-relevant choice: the
result is a 0.8B model on consumer hardware. A larger student
trained on the same data might capture more of the teacher's
capability; we have not measured this.

## 6.4 What we would do differently with more compute

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
false positives. Practitioners building similar pipelines should
budget time to audit their entity-extractor rejections regardless
of which extractor they choose, but the lesson is not novel.

Given a 40-hour A100-class budget rather than RTX 3060, the changes
that would most improve the paper are, in priority order:

1. **Three training seeds per condition** with paired significance
   tests on every judged metric.
2. **Larger student** (4B or 7B) to measure how the pipeline's
   benefits scale with student capacity. The hypothesis is that
   the redirect-axis taxonomy contribution shrinks at larger
   capacities (because a larger model can pick up axis-specific
   patterns from a generic redirect stream) but the persistent
   4-variant contribution holds.
3. **A learned naturalness filter** to address the C2 gap.
4. **Multi-locale empirical evaluation**, repeating the pipeline
   end-to-end for `japan` and `italy`.

We name these explicitly so that a reviewer's "but what about X"
intuition is met with a concrete answer rather than silence.
