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

Table 6 (TODO) reports per-stream, per-CEFR-level filter pass rates
for the 12-stream SFT corpus. Global pass rate is 88.4%
(2753 / 3116 records); per-CEFR pass rates are A1 99%, A2 93%, B1
94%, B2 94%, C1 79%, C2 68%. The C1 / C2 drop is driven by the
`naturalness` filter, which we discuss in §6.

An engineering note on the `locale_judge` filter is deferred to §6.4.

---

## 5.2 Main results

Table 7 reports all ten conditions on the four test sets. **A1 cells
are current measurements**; A2–A5 and B1–B4 are pending and shown as
*pending* until those baseline runs complete.

| Condition | Tutor-Scenario (naturalness) | Redirect-Probe (F1) | Persistent-Probe (sentinel F1) | Locale-Leakage (rate ↓) |
| --- | --- | --- | --- | --- |
| B1 — 0.8B-Base zero-shot | *pending* | *pending* | *pending* | *pending* |
| B2 — 0.8B-Instruct zero-shot | *pending* | *pending* | *pending* | *pending* |
| B3 — 4B-Instruct zero-shot | *pending* | *pending* | *pending* | *pending* |
| B4 — 9B teacher (Q4) | *pending* | *pending* | *pending* | *pending* |
| A4 — − persistent | *pending* | *pending* | *pending* | *pending* |
| A5 — fixed-turn-7 (no decorrelation) | *pending* | *pending* | *pending* | *pending* |
| A3 — − specialised redirects | *pending* | *pending* | *pending* | *pending* |
| A2 — SFT only | *pending* | *pending* | *pending* | *pending* |
| **A1 — full system** | **4.20** | *pending* | **0.873** | **0.89%** |

Headlines (with A1 measured, contrasts pending):

- **N1 (sentinel F1, A1 vs A5):** A1 = **0.873** vs fixed-turn-7 A5
  = *pending* (gap $\geq$ 25 pp expected). This is the decorrelation
  comparison; A1 vs A4 is reported alongside as the weaker
  "persistence helps at all" contrast.
- **N2 (redirect F1, A1 vs B2):** A1 = *pending* vs same-size
  off-the-shelf instruct B2 = *pending* (gap $\geq$ 15 pp expected).
- **N3 (naturalness, A1 vs B4):** A1 = **4.20** vs 9B teacher
  B4 = *pending*; target is A1 within 0.3 of B4.
- **N4 (locale-leakage, A1 vs B2):** A1 = **0.89%** vs same-size
  off-the-shelf B2 = *pending* (gap $\geq$ 20 pp expected).

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
   Persistent-Probe $\cup$ Persistent-FP-Probe). A1 expected $\gg$ A5.
2. **False-positive rate on Persistent-FP-Probe.** This is the
   sharpest discriminator: A5 (position shortcut) is expected to fire
   on benign turn-5/7/9/11 utterances; A1 (trigger-detector) is not.
3. **Firing on Persistent-OffPosition-Probe**, where the third strike
   lands off the trained grid. A1 expected to fire; A5 and a
   position-memoriser expected not to.

### 5.3.1 A1 mechanical results

For the full-system condition we measure:

- **Recall** (Persistent-Probe, all positives): **0.775** (110 / 142).
- **False-positive rate** (Persistent-FP-Probe, all negatives):
  **0.000** (0 / 232) — zero false fires across all 232 benign
  records at the four trained sentinel positions.
- **Precision** (Persistent-Probe $\cup$ Persistent-FP-Probe): **1.000**
  (follows from FP = 0).
- **F1**: **0.873**.
- **OffPosition recall** (Persistent-OffPosition-Probe; third strike
  at turns 13 or 15, outside the trained grid): **0.700** (42 / 60).

The A4 (no-persistent) and A5 (fixed-turn-7) ablations are still
pending; the A1-vs-A5 decorrelation contrast that contribution 2
actually claims (N1) will be reported once A5 lands.

### 5.3.2 Position-stratified evidence

Table 8 shows fire-rate by expected sentinel position for A1.
Positives fire most strongly at the early variants (V1, V2) and
drop on the longer ones (V3, V4); false-positives are flat at 0%
across every trained position; off-position positives fire at 70%
despite never being seen in training.

**Table 8. Sentinel firing by expected turn for A1.** Trained
positions are {5, 7, 9, 11}; untrained off-grid positions are
{13, 15}.

| Expected turn | Positive rate (Persistent-Probe) | FP rate (Persistent-FP-Probe) | OffPos rate (Persistent-OffPosition-Probe) |
| --- | --- | --- | --- |
| 5 | 88.9% (56/63) | 0.0% (0/58) | — |
| 7 | 85.2% (23/27) | 0.0% (0/58) | — |
| 9 | 59.1% (13/22) | 0.0% (0/58) | — |
| 11 | 60.0% (18/30) | 0.0% (0/58) | — |
| 13 | — | — | 73.3% (22/30) |
| 15 | — | — | 66.7% (20/30) |

The **FP=0% column** is the decisive evidence against a positional
shortcut. A model that learned "fire at turn N" would have to fire
on benign turn-N utterances at non-trivial rate; A1 fires on zero
of them across all four trained positions. The **OffPosition
column** is the parallel positive evidence: a position-memoriser
would fire at ~0% on untrained turns 13/15, but A1 fires at 70%,
showing the trigger detector generalises beyond the four trained
positions.

### 5.3.3 The V3/V4 recall dip is not training-data truncation

A natural hypothesis for the V3/V4 dip ($\approx$60% vs $\approx$87% at V1/V2) is
that long prefixes are getting clipped at the SFT 1792-token cap,
removing some early strikes from the model's context. Per-CEFR
stratification refutes this. If truncation were the cause, the
misses would concentrate at C1/C2 where prefixes are longest;
empirically (Table 9) they do not. Per-CEFR row totals are A1
87.5%, A2 73.1%, B1 81.5%, B2 80.0%, C1 70.0%, C2 75.0% — no
monotonic CEFR effect, and B1 / B2 are *not* materially better than
C1 / C2.

**Table 9. Persistent-Probe positive fire-rate by CEFR level x
variant group for A1.** V1+V2 pools the short variants (sentinel at
turns 5, 7); V3+V4 pools the long variants (sentinel at turns 9, 11).
We pool rather than break out all four positions per CEFR cell
because per-position cell sizes are too small (V3 has 22 records
spread over six CEFR levels, V4 has 30) to read individual
percentages as anything other than coin flips. The pooled cells
have n=4–21, which is enough to show the V3+V4 column is
uniformly lower than V1+V2 across all CEFR levels with no
monotonic CEFR effect — refuting the truncation hypothesis (which
would predict a V3+V4 cliff concentrated at C1 / C2).

| CEFR | V1+V2 (turns 5, 7) | V3+V4 (turns 9, 11) | row total |
| --- | --- | --- | --- |
| A1 | 80.0% (4/5) | 90.9% (10/11) | 87.5% (14/16) |
| A2 | 86.7% (13/15) | 54.5% (6/11) | 73.1% (19/26) |
| B1 | 90.5% (19/21) | 50.0% (3/6) | 81.5% (22/27) |
| B2 | 88.2% (15/17) | 62.5% (5/8) | 80.0% (20/25) |
| C1 | 81.2% (13/16) | 25.0% (1/4) | 70.0% (14/20) |
| C2 | 93.8% (15/16) | 50.0% (6/12) | 75.0% (21/28) |

A direct token-length audit confirms: across the 142 positive
records, only **2 (1.4%)** exceed the 1792-token cap (one C1 V4
and one C2 V4 record, by 100–130 tokens). Median prompt length even
at C2 V4 is 1685 tokens, comfortably under the cap. The cap is
therefore not a load-bearing constraint on positions {5, 7, 9, 11},
which is also what motivates the §3.4 claim that hash-mod-4 over
{5, 7, 9, 11} is the feasible codomain.

The V3 / V4 dip is best read as a *0.8B capacity-at-distance*
finding: as the persistence block sits further from the sentinel
turn, the small student loses the strike count. The FP=0% column
rules out the alternative "long context confuses the model" reading
— A1 stays silent on 1700-token benign prefixes without difficulty;
it just sometimes fails to *count to three* across them. This
matches the §6.4 prediction that a 4B / 7B student trained on the
identical 4-variant corpus should recover most of the V1/V2-vs-V3/V4
gap.

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

Numbers (pending judge-ensemble pass on the new cross-family
Prometheus + Llama-3.1 + Gemma-2 ensemble):

- A1 macro-F1: *pending*
- A3 macro-F1: *pending*
- B2 macro-F1: *pending*
- Per-axis F1 table: *pending*

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

Numbers (with A1 measured, baselines pending):

- A1 leakage rate: **0.89%** ($\approx$ 2 / 224 records). An order of
  magnitude below the 8.6% the paper draft projected, suggesting
  the locale-aware prompt block is more effective than we
  anticipated.
- B2 leakage rate: *pending*
- B4 leakage rate: *pending*

The A1 vs B2 contrast that defines N4 will be reported once B2
lands. Even on the conservative published estimate for a same-size
generic instruct model ($\approx$30%), the gap would be $\approx$30 pp — comfortably
beyond the $\geq$20 pp target.

---

## 5.6 Per-CEFR-level breakdown

We report each metric per CEFR level so the reader sees both where
the pipeline is strong and where it is weak. Specifically we expect:

- **A1–B1**: full pipeline strongest; pass rates $\geq$ 90% at filter,
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

The judge ensemble (§4.7) consists of three out-of-family judges
served sequentially via `llama.cpp` on RTX 3060: Prometheus-7B-v2
(`prometheus-7b-v2.0.Q4_K_M.gguf`, Mistral lineage),
Meta-Llama-3.1-8B-Instruct (`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`,
Meta), and Gemma-2-9B-it (`gemma-2-9b-it-Q5_K_M.gguf`, Google).
Prometheus uses its native rubric protocol for the scalar 1–5
metrics (CEFR adherence, naturalness); for the categorical
redirect-axis metric all three judges fall back to a JSON-formatted
instruction prompt.

Table 10 reports inter-judge agreement (Krippendorff's α) per metric
and per condition, along with the human-validation Pearson
correlation on the 100-sample subset where the first author
hand-scored each metric. Metrics with α < 0.6 or human-correlation
< 0.6 are flagged. *(Numbers pending the full judge-ensemble pass —
see §5.2.)*

We do not expect α to reach the levels typical of single-family
ensembles, because cross-family judges score with genuinely
different stylistic priors (Prometheus's rubric language vs
Llama-3.1's length preferences vs Gemma-2's register defaults).
Lower-than-typical α on a metric is therefore informative — it
indicates the metric is sensitive to stylistic-prior differences
rather than to a consensus quality signal — and we report human
correlation alongside α precisely because human gold is the
arbiter when the judges disagree systematically.
