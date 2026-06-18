# 5. Results

This section reports results across four sub-questions, each tied
to one of the research contributions in §1:

- §5.1: Pipeline yield — dataset statistics
- §5.2: Main results table (all ten conditions on all four test
  sets)
- §5.3: Sentinel firing (the fully-mechanical metric) and the
  4-variant ablation
- §5.4: Redirect-axis F1 and the taxonomy ablation
- §5.5: Locale leakage and the locale-aware-prompt contribution
- §5.6: Per-CEFR-level breakdown: honest reporting of where the
  pipeline is strong and where it is not

Throughout, we use one-tailed bootstrap CIs with 1000 resamples for
ranges over judged metrics; mechanical metrics are point estimates.

---

## 5.1 Pipeline yield and dataset statistics

Table 3 (TODO) reports per-stream, per-CEFR-level filter pass rates
for the 12-stream SFT corpus. Global pass rate is 88.4%
(2753 / 3116 records); per-CEFR pass rates are A1 99%, A2 93%, B1
94%, B2 94%, C1 79%, C2 68%. The C1 / C2 drop is driven by the
`naturalness` filter, which we discuss in §6.

An engineering note on the `locale_judge` filter is deferred to §6.4.

---

## 5.2 Main results

Table 4 (TODO) reports all ten conditions on the four test sets.

| Condition | Tutor-Scenario (naturalness) | Redirect-Probe (F1) | Persistent-Probe (sentinel F1) | Locale-Leakage (rate ↓) |
| --- | --- | --- | --- | --- |
| B1 — 0.8B-Base zero-shot | _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |
| B2 — 0.8B-Instruct zero-shot | _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |
| B3 — 4B-Instruct zero-shot | _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |
| B4 — 9B teacher (Q4) | _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |
| A4 — − persistent | _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |
| A5 — fixed-turn-7 (no decorrelation) | _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |
| A3 — − specialised redirects | _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |
| A2 — SFT only | _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |
| **A1 — full system** | _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |

Headlines to fill in once numbers land:

- N1 (sentinel F1, A1 vs A5): full system X% vs fixed-turn-7 Y%
  (gap ≥ 25pp expected). This is the decorrelation comparison;
  A1 vs A4 (vs no-persistent) is reported alongside as the weaker
  "persistence helps at all" contrast.
- N2 (redirect F1, A1 vs B2): full system X% vs same-size
  off-the-shelf instruct Y% (gap ≥ 15pp expected).
- N3 (naturalness, A1 vs B4): full system within 0.3 of the 9B
  teacher upper bound (distillation effectiveness).
- N4 (locale-leakage, A1 vs B2): full system X% vs same-size
  off-the-shelf Y% (gap ≥ 20pp expected).

---

## 5.3 Sentinel firing and the decorrelation ablation

The sentinel-firing metrics are the only fully-mechanical evaluation
in this paper, which makes them the strongest evidence we report. The
central comparison is **A1 (4-variant) vs A5 (fixed-turn-7)**: both
train on the persistent streams, so any gap isolates the
trigger-position decorrelation principle (§3.4) rather than the mere
presence of persistence training (which A1-vs-A4 covers). We report
three things, because uniform fire-rate alone is necessary but not
sufficient (§4.8):

1. **Recall on Persistent-Probe** and **F1** (precision computed over
   Persistent-Probe ∪ Persistent-FP-Probe). A1 expected ≫ A5.
2. **False-positive rate on Persistent-FP-Probe.** This is the
   sharpest discriminator: A5 (position shortcut) is expected to fire
   on benign turn-5/7/9/11 utterances; A1 (trigger-detector) is not.
3. **Firing on Persistent-OffPosition-Probe**, where the third strike
   lands off the trained grid. A1 expected to fire; A5 and a
   position-memoriser expected not to.

Figure 2 (TODO) shows fire-rate by expected sentinel position for A1
vs A5 vs A4. If A1 has learned "third strike on the same axis," its
positive fire-rate should be approximately uniform across {5, 7, 9,
11} *and* its FP-rate near zero; A5 should show position-peaked
firing and a high benign FP-rate; A4 (no persistent training) should
fire near the prior on everything.

Numbers to fill in:

- A1 recall: _placeholder_; FP-rate: _placeholder_; F1: _placeholder_.
- A5 recall: _placeholder_; FP-rate: _placeholder_; F1: _placeholder_.
- A4 recall: _placeholder_; FP-rate: _placeholder_; F1: _placeholder_.
- A1 vs A5 OffPosition firing rate: _placeholder_ / _placeholder_.
- Positive fire-rate by position {5, 7, 9, 11}, per condition:
  per-position table _placeholder_.

---

## 5.4 Redirect-axis F1 and taxonomy ablation

This subsection tests the prediction made by the invariant framing of
§3.3: because distinct invariants require distinct *minimal repairs*,
a model trained on a single generic redirect stream should fail to
produce the axis-appropriate repair on axes it never saw separately.
Redirect-Probe records are tagged with one of seven axes. Each
baseline produces a response; we classify the *repair shape* of that
response with the multi-judge ensemble (a proxy for repair
correctness, per §4.5). F1 is computed per axis and macro-averaged.

We expect A1 to substantially outperform A3 on axes that A3 does
not train on (`locale_redirect`, `pedagogy_redirect`,
`language_redirect`, `persona_redirect`, `topic_redirect`,
`role_swap_redirect`) but to be comparable on the generic
`redirect` axis that both train on. If instead A3 matched A1 across
the specialized axes, the invariant decomposition would be
unsupported — the prediction is falsifiable in this direction. The
4B-Instruct and 9B-teacher zero-shot baselines are expected to have
moderate F1 on the generic axis but unstable F1 on the specialised
axes because nothing in their post-training specifically targets the
distinct repair shapes.

Numbers to fill in:

- A1 macro-F1: _placeholder_
- A3 macro-F1: _placeholder_
- B2 macro-F1: _placeholder_
- Per-axis F1 table: _placeholder_

---

## 5.5 Locale leakage and the locale-aware prompt contribution

Locale-Leakage measures the rate at which the produced response
mentions a Western-default entity from a fixed gazetteer. This is
mechanical and exact.

Expected pattern:

- B1 / B2: high leakage (no locale-aware training; defaults to
  general Western pretraining priors).
- B3 / B4: lower than B1/B2 (more capacity to follow a tutor system
  prompt) but still non-negligible.
- A1: lowest leakage (trained on the locale-aware locale
  instruction block).

Numbers to fill in:

- A1 leakage rate: _placeholder_
- B2 leakage rate: _placeholder_
- B4 leakage rate: _placeholder_

---

## 5.6 Per-CEFR-level breakdown

We report each metric per CEFR level so the reader sees both where
the pipeline is strong and where it is weak. Specifically we expect:

- **A1–B1**: full pipeline strongest; pass rates ≥ 90% at filter,
  naturalness and locale fidelity close to teacher.
- **C1–C2**: dominant remaining failure mode is naturalness. The
  filter pass-rate audit identified naturalness as the second
  largest filter rejection class (26.6%) and showed naturalness
  rejections concentrated at C1 / C2.

We argue that honest per-level reporting is important because the
C2 gap is a real limitation of the pipeline that the global mean
would hide. We discuss this in §6.

---

## 5.7 Judge ensemble reliability

Table 5 (TODO) reports inter-judge agreement (Krippendorff's α) per
metric and per condition, along with the human-validation Pearson
correlation on the 100-sample subset. Metrics with α < 0.6 or
human-correlation < 0.6 are flagged.
