# Paper Outline (Working Draft)

**Working title:** "A Multi-Stream Yield-Aware Pipeline for Generating
Robust, Locale-Aware English-Tutor Dialogue Data"

**Target venue (primary):** BEA Workshop (Innovative Use of NLP for
Building Educational Applications). EI-indexed via ACL Anthology.
**Backup:** AIED / NLPCC / LREC-COLING.

**Length:** 8 pages + references (BEA format).

---

## Headline contributions

1. **12-stream SFT taxonomy** decomposing tutor-side behavior across
   `normal` + 7 single-shot redirect axes (locale, pedagogy, language,
   persona, topic, role_swap, generic) + 4 persistent 3-strike axes.
2. **4-variant structural design for persistent abuse handling** that
   defeats positional shortcuts by varying the sentinel-firing turn
   among 5/7/9/11. Critically, the choice of variant is
   hash-deterministic on the seed id.
3. **Yield-aware top-up loop with declarative ratio targets** that
   composes absolute floors and percentage-of-mix knobs via
   `T_per_level = sum_abs / (1 − sum_ratios)`.
4. **Locale-aware prompt engineering** with strict-Latin vs allow-L1
   variants for streams where L1 characters legitimately occur in the
   learner's turn (language_redirect).
5. **Six-filter mechanical-and-LLM cascade** with an SQLite verdict cache;
   includes the locale_judge with per-locale allowlists. (We discuss an
   engineering caveat about extractor-driven FPs in §6.5; not framed as
   a research contribution.)
6. Empirical demonstration on a **small student trained on RTX 3060**
   showing that taxonomy + persistent + locale-aware design produce a
   deployable small tutor model. (0.8B pilot; planned 4B follow-up.)

## Section structure

1. **Introduction** (1 page) — motivation, contributions, headline numbers
2. **Related Work** (1 page) — synthetic instruction data; LLM judges;
   tutor LLMs; persona/safety datasets
3. **Method** (~3 pages) — pipeline overview + each contribution
4. **Experimental Setup** (~1 page) — base model, training recipe,
   test sets, judges
5. **Results** (~1.5 pages) — main results table; ablations;
   sentinel precision/recall (mechanical); per-level breakdown
6. **Discussion** (~0.5 page) — locale_judge FP audit as methodological
   caution; C1/C2 naturalness gap; threats to validity
7. **Conclusion** (~0.25 page) — summary; future work

## Headline numbers to target

(Plan backwards from these — fill in as results land.)

- N1: **Sentinel-firing F1**: full system X% vs fixed-position-only ablation
  Y% (gap ≥ 25pp). Mechanical, not judged.
- N2: **Redirect-axis F1**: full system X% vs same-size off-the-shelf
  instruct Y% (gap ≥ 15pp).
- N3: **Naturalness on Tutor-Scenario**: full system within 0.3 of the
  9B teacher upper bound.
- N4: **Locale-leakage rate**: full system X% vs same-size off-the-shelf
  Y% (gap ≥ 20pp).
- (Engineering note, not a headline: 70.1% → 88.4% pass-rate from
  locale_judge allowlist patch. Reported in §6.5 as a caveat.)

## Baseline matrix

Trained ablations (same base, same training recipe, varying data):
- A1 — full system (SFT + DPO on all 12 streams)
- A2 — SFT only (no DPO) — measures DPO contribution
- A3 — no specialized redirects (drop 6 specialized streams)
- A4 — no persistent (drop 4 persistent streams)

Zero-shot baselines (no training):
- B1 — Qwen3.5-0.8B-Base raw
- B2 — Qwen3.5-0.8B post-trained (off-the-shelf instruct, same size)
- B3 — Qwen3.5-4B post-trained (larger same-family)
- B4 — Qwen3.5-9B-UD-Q4_K_XL via llama-server (distillation upper bound + judge)

## Held-out test sets (frozen 2026-06-15)

- Tutor-Scenario-224 — cold-start dialogue, all 6 levels
- Redirect-Probe-143 — 7 axes × varying levels, partial dialogue ending
  in user abuse turn
- Persistent-Probe — pilot (will be ~120 after persistent_topup completes)
- Locale-Leakage-224 — same scenarios as Tutor-Scenario, locale-leakage metric

## Hardware footnote

All experiments use a single consumer GPU (RTX 3060 12GB). This
constrains us to QLoRA over a 0.8B base; we argue that the
0.8B-on-3060 result is a paper feature (deployable on consumer
hardware), not a limitation. The LAN-served 9B teacher contributes
both as a baseline and as one of three multi-judge ensemble members.
