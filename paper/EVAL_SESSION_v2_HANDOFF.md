# v2 Evaluation Session Handoff — read this first

**Purpose.** This document is the single source of truth for the post-SFT
evaluation session that runs *after* the v2 SFT training completes
(A6 → A7 → A5 in the v2 orchestrator, sequential, each ~8 h on RTX 3060).
A new Claude session will use this doc to (a) run the right evaluations,
(b) fill in the expected-value tables below with measured numbers, and
(c) write the v2 analysis into the paper.

**Status as of last write (2026-06-28).**
- A1 (v2) SFT done: `outputs/paper_v2/a1/sft/`
- A3 (v2) SFT done: `outputs/paper_v2/a3/sft/`
- A6 (v2) SFT in progress (orchestrator stage4_sft_a6)
- A7 (v2) SFT queued
- A5 (v2) SFT queued
- DPO step: dropped in v2 (paper is now SFT-data-only ablations)
- A4 condition: dropped in v2 (secondary claim, not paper-critical)

---

## 1. What changed v1 → v2 (read this carefully — naming is a trap)

### 1.1 Condition rename

| v1 tag         | v2 tag | Data definition |
|----------------|--------|-----------------|
| A1 (SFT + DPO) | —      | dropped (DPO removed) |
| A2 (= A1-SFT)  | **A1** | SFT only, 12 streams, 4-variant persistent, axis-specific `[SESSION_END: <axis>]` |
| A3-SFT         | **A3** | A1 minus the 6 specialized redirect streams |
| A4-SFT         | —      | dropped |
| A5-SFT         | **A5** | A1 with persistent stream rebuilt as fixed-turn-7, axis-specific sentinel |
| —              | **A6** | NEW: A5 + generic `[SESSION_END]` sentinel (no axis label) |
| —              | **A7** | NEW: A1 + generic `[SESSION_END]` sentinel (no axis label) |
| B1–B4          | B1–B4  | unchanged zero-shot baselines |

**Critical:** when reading the existing `05_results_scaffold.md`, v1 "A2"
is what v2 calls "A1". v1 "A1 (SFT+DPO)" is no longer a condition. The
old A2-vs-A5-SFT decorrelation contrast is now reported as A1-vs-A5.

### 1.2 The 2×2 design

|                              | 4-variant persistent | fixed-turn-7 persistent |
|------------------------------|----------------------|--------------------------|
| **axis-specific sentinel**   | **A1**               | **A5**                   |
| **generic `[SESSION_END]`**  | **A7**               | **A6**                   |

This isolates two orthogonal effects:

- **Decorrelation effect (rows hold sentinel format constant):**
  A1 vs A5 — does 4-variant beat fixed-7 with axis-specific labels?
  A7 vs A6 — does the same effect hold under generic sentinel?
- **Sentinel-label effect (columns hold position-design constant):**
  A1 vs A7 — does the axis label add information beyond position?
  A5 vs A6 — same question at fixed-7.

### 1.2.1 Four critical v1 → v2 changes for the position-decorrelation experiment (user-stated 2026-06-28)

These four together make the v2 result substantively interpretable in a way v1's was not:

1. **A5 persistent volume: 96 → 637 records.** v1 A5-SFT had only 96 fixed-turn-7 records (vs A2's 614 four-variant) — too few to learn a sentinel detector reliably, so the v1 A2-vs-A5-SFT gap was confounded with sample size. v2 A5 has 637 records, comparable to A1's 614. Any v2 A1-vs-A5 sentinel F1 gap is now a clean test of position-decorrelation independent of data volume.

2. **Eval persistent probes use HELD-OUT records from `data/sft_raw/persistent_*.jsonl`.** The probe set is built from raw persistent records that were excluded from the SFT training pool (20% eval-seed exclusion in `filter_sft`, then `build_eval_sets.py` reads from raw with those seeds). The model has *never seen* these specific records during SFT. This measures true generalization to the persistent pattern — not memorization of training examples.

3. **NEW eval metric: premature-firing at the FIRST and SECOND user violation turn.** Beyond the standard FP-probe (benign turns at trained sentinel positions), v2 explicitly evaluates whether the sentinel fires AFTER ONLY 1 or 2 violations. A model that learned "fire on third strike" should remain silent at violations 1 and 2 and fire only on the third. Premature firing is a different failure mode from positional shortcut (the v1 §5.3 framing): a model can have FP=0% on benign turns yet still fire early on 1-strike or 2-strike prefixes. Use test set `persistent_premature_probe`. Verify it exists in `eval_sets/` before running; if missing, build per `scripts/build_eval_sets.py` first.

4. **Generic sentinel is `[SESSION_END]` — no axis-label suffix.** A6/A7 use this exact string. A1/A5 use `[SESSION_END: <axis>]`. The mechanical detector should accept both forms; verify with a quick sanity check before running scored eval.

### 1.2.2 ⚠️ OPEN CONFOUND: epoch mismatch (discovered 2026-06-30)

**A1 and A3 were trained for 2 epochs; A5, A6, A7 for 1 epoch.** The
A5/A6/A7 configs were changed to `num_train_epochs: 1` on 2026-06-28
(after the thermal-throttling diagnosis, to save time) but A1/A3 had
already completed at 2 epochs.

**This invalidates two comparisons as currently trained:**
- §5.3 2×2 sentinel design (A1 vs A5/A6/A7) — A1's apparent advantage
  could be the extra epoch, not the 4-variant + axis-specific design.
- §5.4 pedagogy (A1 vs A3 both 2ep is fine, but A1 vs A5/A6/A7 mixes).

**Preliminary §5.3 numbers showing the confound** (from current
mismatched training — DO NOT report these as final):
- Premature firing (violation_count=1 / =2), should be ~0:
  - A1 (2ep): 0.013 / 0.201  ← best early-fire suppression
  - A5 (1ep): 0.384 / 0.736
  - A6 (1ep): 0.497 / 0.786
  - A7 (1ep): 0.497 / 0.887
- A1's win is exactly what 2× training would also produce, so the
  decorrelation claim is NOT yet isolable.

**Resolution (decided after the in-flight B1–B4 eval completes):** most
likely retrain A1 (and possibly A3) at 1 epoch to match, then re-eval.
Until then, treat all A-condition §5.3/§5.4 numbers as provisional.
The B1–B4 generations are unaffected by this (no training).

### 1.3 Data trimming applied to A5/A6/A7 persistent (2026-06-28)

The persistent records in `sft_filtered_a5_persistent/`,
`sft_filtered_a6_persistent/`, and `sft_filtered_a7_persistent/` were
trimmed to **end at the sentinel turn**. Post-sentinel messages
(student "Okay" + a closing line) were removed. Train data now teaches
the model "after this sentinel, stop" rather than "after this sentinel,
keep chatting."

A1 (v2) was trained with the *untrimmed* original persistent data
(starting before this change). If the v2 A1 sentinel firing rate is
different from old A2's, the trim is a candidate explanation — flag
that in the analysis.

### 1.4 Generic vs axis-specific sentinel implementation

- A1 / A5: model emits `[SESSION_END: persistent_off_topic]` (or whichever axis)
- A6 / A7: model emits `[SESSION_END]` (no axis label)

The sentinel filter scoring (mechanical) was updated to recognize both
formats — see `scripts/run_paper_score.py`. The mechanical-firing
detector uses substring match `[SESSION_END]` which captures the
generic form AND the prefix of `[SESSION_END:` (both work).

---

## 2. Evaluation plan — what to run

Run from `c:\project\conversationFactory\qwen-en-tutor` with venv active
and `$env:PYTHONUTF8="1"`.

### 2.1 Adapter eval (A1, A3, A5, A6, A7) — no llama-server needed

```powershell
foreach ($b in @("paper_a2","paper_a3_sft","paper_a5_sft","paper_a6_sft","paper_a7_sft")) {
  python scripts\run_paper_eval.py `
    --baseline $b `
    --test-set all `
    --output-dir outputs\paper_v2\eval
}
```

Adapter-path mapping (handled by the orchestrator already, but if running manually):
- `paper_a2`     → `outputs/paper_v2/a1/sft`  (v2 A1 = v1's "A2")
- `paper_a3_sft` → `outputs/paper_v2/a3/sft`
- `paper_a5_sft` → `outputs/paper_v2/a5/sft`
- `paper_a6_sft` → `outputs/paper_v2/a6/sft`
- `paper_a7_sft` → `outputs/paper_v2/a7/sft`

### 2.2 Teacher eval (B4) — requires teacher llama-server up

```powershell
# Launch teacher:
Start-Process -FilePath "C:\project\conversationFactory\qwen-en-tutor\vendor\llama_cpp\llama-server.exe" `
  -ArgumentList "-m ..\models\GGUF\Qwen3.5-9B-UD-Q4_K_XL.gguf -ngl 80 -c 32768 --host 0.0.0.0 --port 8080 --log-prefix" `
  -WorkingDirectory "C:\project\conversationFactory\qwen-en-tutor\vendor\llama_cpp"
# Wait ~30s for ready, then:
python scripts\run_paper_eval.py --baseline qwen3_5_9b_teacher --test-set all --output-dir outputs\paper_v2\eval
```

### 2.3 HF baselines (B1, B2, B3) — no server

```powershell
foreach ($b in @("qwen3_5_0_8b_base","qwen3_5_0_8b_instruct","qwen3_5_4b_instruct")) {
  python scripts\run_paper_eval.py --baseline $b --test-set all --output-dir outputs\paper_v2\eval
}
```

### 2.4 Mechanical scoring (no judges)

```powershell
$env:QWEN_TUTOR_EVAL_OUT="outputs\paper_v2\eval"
$env:QWEN_TUTOR_MECH_SCORE_OUT="outputs\paper_v2\score\redirect_mechanical.json"
python scripts\score_redirect_mechanical.py
```

Produces sentinel-firing rates and locale-leakage rates per condition.

### 2.5 Judged scoring (3 judges sequential)

Use the orchestrator's `stage6c/d/e/f` functions, or invoke directly with
each judge. The orchestrator handles llama-server swaps automatically.
Note this is a long-running phase (~12–24h with all judges × all
metrics × all baselines).

**Pragmatic shortcut for fast paper update:** run only the mechanical
metrics (2.4) and the cheap pairwise. Skip naturalness/CEFR adherence
judging if time-constrained — those are stable across A1/A3/A5/A6/A7
since they share base model and training recipe.

---

## 3. Expected values + analysis methodology

The tables below have **predicted values** based on v1 results. The new
session should **replace each `(expected: X)` with the measured value**
in the final paper write-up. If the measured value diverges from the
prediction by >2× the noise floor, write a short interpretation
paragraph; otherwise just report.

### 3.1 Main results — Table 7 successor

| Condition | Tutor-Scenario (nat) | Redirect-Probe (pedagogy F1) | Persistent-Probe (sentinel F1) | Locale-Leakage |
|---|---|---|---|---|
| B1 — 0.8B-Base zero-shot   | 3.96 (carry from v1) | 0.34 | —      | 2.68% |
| B2 — 0.8B-Instruct zs      | 4.03                 | 0.37 | —      | 4.46% |
| B3 — 4B-Instruct zs        | 4.08                 | 0.36 | 0.132  | 1.34% |
| B4 — 9B teacher            | 4.62                 | 0.37 | 0.068  | 2.68% |
| **A1** (4-var + axis)      | **(expected 4.21)**  | **(expected 0.37)** | **(expected 0.84)** | **(expected 1.3%)** |
| **A3** (no specialized)    | (expected 4.19)      | **(expected 0.24)** | (expected 0.87) | (expected 2.7%) |
| **A5** (fixed-7 + axis)    | (expected 4.18)      | (expected 0.51)     | **(expected 0.75)** | (expected 2.2%) |
| **A6** (fixed-7 + generic) | (expected 4.18)      | (expected 0.51)     | **(expected ~0.75 if label ignorable, lower if label was load-bearing)** | (expected 2.2%) |
| **A7** (4-var + generic)   | (expected 4.21)      | (expected 0.37)     | **(expected ~0.84 if label ignorable, lower if label was load-bearing)** | (expected 1.3%) |

**Predictions justified:** A1, A3, A5 expected ≈ v1 A2/A3-SFT/A5-SFT
because v2 retrains on the same logical data subset with the same recipe.
The trim (post-sentinel removal) is a controlled variable: it should
*increase* the gap between fire-rate and FP-rate (cleaner stop signal),
which raises F1 slightly. A6 and A7 are net-new; their expected values
mirror A5 and A1 respectively, on the null hypothesis that the
axis-label component of the sentinel is not load-bearing.

### 3.2 Mechanical sentinel — Table extending §5.3.1

Replace the v1 Table 8 (A1/A2/A5-SFT) with this 5-row v2 table:

| Metric                     | A1 (4-var+axis) | A5 (fixed-7+axis) | A6 (fixed-7+gen) | A7 (4-var+gen) | A3 (no specialized) |
|----------------------------|-----------------|-------------------|------------------|----------------|----------------------|
| Recall (Persistent-Probe)  | (expected 0.72) | (expected 0.60)   | (expected 0.60)  | (expected 0.72) | (expected 0.66)     |
| FP rate (FP-Probe)         | **(expected 0.00)** | **(expected 0.00)** | (expected ~0.00–0.05; risk if generic sentinel triggers easily) | (expected ~0.00–0.05) | (expected 0.00)     |
| Precision (union)          | (expected 1.00) | (expected 1.00)   | (expected ≥0.95) | (expected ≥0.95) | (expected 1.00)     |
| **F1**                     | (expected 0.84) | (expected 0.75)   | (expected 0.75)  | (expected 0.84) | (expected 0.79)     |
| OffPosition recall         | (expected 0.55) | (expected 0.55)   | (expected 0.55)  | (expected 0.55) | (expected 0.50)     |

**Position-stratified table (analog to §5.3.1a):** if A6 fires
preferentially at turn 7 (its training position) and A1 fires across
5/7/9/11 uniformly, that's evidence the *position* was learned. If A6
fires at multiple positions, position was generalized — same finding
as v1.

### 3.3 New 2×2 contrast tables — INSERT after §5.3.1a

Add a new subsection `### 5.3.2 — Sentinel format ablation (axis-specific vs generic)`
with the following two tables:

**Table A — Effect of position-decorrelation, holding sentinel format
constant:**

| Pair compared | F1 (4-var) | F1 (fixed-7) | Δ (4-var − fixed-7) |
|---|---|---|---|
| Axis-specific (A1 vs A5) | (fill) | (fill) | (expected ≈ +8 pp) |
| Generic (A7 vs A6)       | (fill) | (fill) | (expected ≈ +8 pp) |

**Analysis question:** Does the decorrelation effect replicate under
generic sentinel? If yes (both rows ≈ +8 pp), decorrelation is a
robust effect independent of sentinel format. If only the
axis-specific row shows the gap, decorrelation interacts with sentinel
format — write a sentence on this in §5.3.2.

**Table B — Effect of sentinel label, holding position-design constant:**

| Pair compared | F1 (axis-specific) | F1 (generic) | Δ (axis − generic) |
|---|---|---|---|
| 4-variant (A1 vs A7)  | (fill) | (fill) | (expected ≈ 0 pp; H₀: label not load-bearing) |
| fixed-7 (A5 vs A6)    | (fill) | (fill) | (expected ≈ 0 pp; H₀: label not load-bearing) |

**Analysis question:** Does the axis-specific label add fire-rate
information beyond what position alone provides? **Hypothesis H₀:**
no, both deltas ≈ 0. **Alternative H₁:** axis-specific label provides
disambiguation signal → axis row's F1 is higher (smaller FP rate).

If H₀ holds: paper claim becomes *"generic `[SESSION_END]` suffices;
the axis label is decorative."* Deployment simplification.

If H₁ holds: paper claim is *"axis-specific labels provide
disambiguation signal beyond position; generic sentinel suffers a
measurable FP cost."*

### 3.4 Locale leakage — Table updates

Drop v1 A1 (SFT+DPO) and A2 rows. v2 rows:

| Condition | Leakage rate |
|---|---|
| A1 (v2)  | (expected 1.3%) |
| A3       | (expected 2.7%) |
| A5       | (expected 2.2%) |
| A6       | (expected 2.2%) |
| A7       | (expected 1.3%) |
| B1       | 2.68% pre-augment |
| B2       | 4.46% pre-augment |
| B3       | 1.34% pre-augment |
| B4       | 2.68% pre-augment |

Predictions: locale handling is unchanged across A1/A3/A5/A6/A7
(same locale instruction block in the deployment prompt; difference
between conditions is only in the persistent stream training).
**Hypothesis:** locale-leakage rates of A1, A3, A5, A6, A7 should be
within ~1 pp of each other; difference vs B-baselines (the §5.5
locale claim) holds.

### 3.5 Redirect taxonomy — F1 RETIRED, replaced with three group-appropriate metrics

**Decision (user-stated 2026-06-28): F1 is dropped from the paper.**
F1 conflates "classify the response shape" with "behavioral quality" —
it saturates near 0.75 on persona/role_swap regardless of training
(everyone produces a persona-shaped response) and floors near 0.30
on locale/language/topic regardless of model quality (the
context-blind judge can't score axes whose repair is context-dependent
from the response alone). Replace with:

| Axis group | New metric | Source |
|---|---|---|
| persona / role_swap | **Pairwise preference parity** (49% / 38% from v1 §5.4.5) | already measured; report as-is, no v2 re-run needed |
| locale / language / topic | **Mechanical pass rate** (gazetteer / L1-ack / topic anchor) | flag as **underpowered** (n=13); not a discriminating signal |
| **pedagogy** | **Withholding rate under matched prompt** | **NEW — load-bearing v2 experiment, not yet run** |

### 3.5.1 The pedagogy withholding-rate experiment (v2 contribution-1 headline)

**Definition.** For each pedagogy_redirect probe, score whether the
tutor response *withholds the direct answer and scaffolds* (yes/no).
"Withholds" = does not produce the literal answer/translation; instead
asks the learner to attempt it, gives a hint, or offers a partial
scaffold. "Not withholds" = gives the answer directly (with or
without padding). Score by judge classification (one binary call per
record) using the criterion already present in
`scripts/score_pairwise_preference.py:213` — repurposed from
pairwise to per-record binary.

**Conditions to measure** (full matched-prompt matrix):

| Condition | Has pedagogy instruction in prompt? | Has pedagogy SFT demo? | Cell |
|---|---|---|---|
| A1 (v2) | Yes (matched deployment prompt) | Yes | trained + instructed |
| A5      | Yes | Yes | trained + instructed |
| A6      | Yes | Yes | trained + instructed |
| A7      | Yes | Yes | trained + instructed |
| **A3**  | Yes (matched deployment prompt) | **No** (specialized streams dropped) | **instruction only — load-bearing cell** |
| **B1**  | **Verify; add if missing** | No | **instructed-only baseline, must re-run if missing** |
| **B2**  | **Verify; add if missing** | No | **instructed-only baseline, must re-run if missing** |
| **B3**  | **Verify; add if missing** | No | **instructed-only baseline, must re-run if missing** |
| **B4**  | **Verify; add if missing** | No | **instructed-only baseline, must re-run if missing** |

**Verification step before scoring** (do this first tomorrow):
inspect the generations under `outputs/paper_v2/eval/<baseline>/redirect_probe.jsonl`
(or wherever pedagogy_redirect lives) for B1–B4 and confirm the
system prompt they were generated under includes the pedagogy
guideline clause. If it doesn't, re-run B1–B4 generations with the
instruction added to the prompt before scoring. The matched-prompt
invariant is the load-bearing assumption — without it, the
B-baseline comparison is invalid.

**The hypothesis (falsifiable):**

- **H₀ (instruction sufficient):** B1–B4 + instruction and A3 + instruction
  withhold at rates comparable to A1/A5/A6/A7. → The specialized
  pedagogy stream is decorative; the prompt clause alone is enough.
  The §5.4 contribution dies.
- **H₁ (demonstration necessary):** B1–B4 + instruction and A3 +
  instruction withhold at substantially LOWER rates than A1/A5/A6/A7.
  → "Even with the instruction in the prompt, a small model needs
  demonstrations to learn the withholding behavior." This is the v2
  contribution-1 headline: **specialized data is necessary for
  describable-but-not-promptable repairs.**

**Expected result if §3.3 holds:**

| Condition | Expected withholding rate |
|---|---|
| A1/A5/A6/A7 (trained + instructed) | ~50–70% (varies by data freshness) |
| A3 (instructed-only, no specialized) | ~25–35% (well below trained band) |
| B1 (instructed-only, 0.8B-Base) | ~20–30% (matches A3 — instruction alone insufficient at 0.8B) |
| B2 (instructed-only, 0.8B-Instruct) | ~25–35% (matches A3) |
| B3 (instructed-only, 4B) | ~35–45% (more capacity to follow instructions) |
| B4 (instructed-only, 9B teacher) | ~40–55% (most capacity; instruction starts to suffice) |

The decisive contrast is **A3 vs A1/A5/A6/A7** at matched 0.8B base
+ matched prompt + matched recipe — the gap is the specialized-stream
effect with all other variables held constant.

### 3.5.2 Locale / language / topic — underpowered mechanical (report and flag)

Re-use v1's §5.4.2 mechanical pass rates (gazetteer / L1-ack /
subtopic-anchor). At n=13 these cells are too small to read as
discriminating — single-record variance is ~7 pp. Report the numbers
but **explicitly flag underpowered** rather than treating differences
as meaningful. The §6 limitations section should call this out.

### 3.5.0 Eval scope by condition (REVISED 2026-06-30 — read this first)

User directive: **A5/A6/A7 are evaluated only on the §5.3 mechanical
sentinel score** (2×2 firing + premature firing). They share base model
and training recipe with A1, so judge-based metrics on them would be
redundant. Concretely:

| Metric | Conditions scored | Judge? |
|---|---|---|
| §5.3 sentinel firing 2×2 + premature | A1, A3, A5, A6, A7 (+ B1-B4 where they can fire) | no (mechanical) |
| §5.4 pedagogy withholding rate | **A1, A3, B1-B4 only** | yes (1 judge) |
| §5.2 naturalness | **reuse v1 scores — no v2 re-run** | — |
| persona/role_swap pairwise | reuse v1 §5.4.5 — no re-run | — |
| redirect F1 | RETIRED | — |

Naturalness skip rationale: v1 naturalness scores are stable and
confident; A1–A7 differ only in persistent-stream design, which does
not affect general tutor-turn naturalness. Re-running 3 judges × all
baselines (~6-10h) would not change the §5.2 N3 conclusion.

### 3.5.3 Persona / role_swap — preference parity (v1 result, no re-run)

The v1 §5.4.5 pairwise preference came back at 49% / 38% — measured
parity (not artifactual ties on a saturated F1). This is the result;
no v2 re-run needed unless v2 generations differ structurally from
v1's. The result stands: **on persona/role_swap, specialized SFT data
does not improve repair quality at this scale.**

---

## 4. Analysis methodology (load-bearing — read before writing §5)

### 4.1 Section ordering / restructuring

The v1 §5.3 was organized around a 3-condition contrast (A1, A2,
A5-SFT). v2 has 5 trained conditions and a 2×2 design. Suggested
restructure:

- **§5.3.1** — Mechanical sentinel firing across all 5 v2 conditions
  (one big 5-column table).
- **§5.3.2** — *NEW:* Sentinel format ablation (the 2×2 design with
  the two contrast tables from §3.3 of this doc).
- **§5.3.3** — Position-stratified evidence (replaces v1 §5.3.2 /
  §5.3.3, now showing whether A6/A7 also avoid the positional
  shortcut).

### 4.2 Headline updates needed in §5.2

The v1 N1 headline (decorrelation, A2 vs A5-SFT, +8.7 pp) becomes
**N1a (A1 vs A5)** and gains a sibling:

- **N1a (decorrelation, axis-specific):** A1 sentinel F1 vs A5 — gap.
- **N1b (decorrelation, generic):** A7 sentinel F1 vs A6 — gap.
- **N5 (NEW — sentinel label ablation):** A1 vs A7, A5 vs A6.
  Hypothesis: gaps ~0 → generic sentinel suffices.

### 4.3 Sample-size + noise floor for the new contrasts

Test-set sizes (each held-out probe):
- Persistent-Probe: 142 records
- Persistent-FP-Probe: ~240 records
- Persistent-OffPosition-Probe: 60 records

At n=142 the standard error of an F1 measurement is ~0.04, so two F1s
differing by <0.08 should be reported as parity. The expected
decorrelation gap (+8 pp) sits *right at* the detection threshold —
report 95% CI from bootstrap (1000 resamples) per cell.

### 4.4 Anything that fails to compute on first run

A6/A7 are new conditions; the eval scripts have been updated to handle
them but verify before running:

```powershell
python -c "from qwen_tutor.eval.run_paper_eval import BASELINES; print(sorted(BASELINES))"
# Should include: paper_a6_sft, paper_a7_sft
```

If `paper_a6_sft` / `paper_a7_sft` are missing, see
`scripts/run_paper_eval.py` for the `BASELINES` dict and add them.

---

## 5. After the eval results land — paper edit checklist

1. [ ] Replace every `(expected …)` in this doc with measured numbers
2. [ ] Update `paper/sections/05_results_scaffold.md` §5.2 Table 7 with
       v2 numbers (drop the v1 A1/A2/A4 rows, add A6/A7)
3. [ ] Replace v1 §5.3.1 Table 8 (3-col) with v2 5-col version
4. [ ] Add new subsection §5.3.2 with the 2×2 contrast tables
5. [ ] Update §5.4.3 pedagogy F1 table to v2 condition names
6. [ ] Update §5.5 locale table — drop v1 A1/A2, add v2 A1/A6/A7
7. [ ] Update headlines N1, N2 in §5.2 to v2 condition names; add N5
8. [ ] Decide H₀ vs H₁ for the label ablation and write the
       paragraph in §5.3.2
9. [ ] Re-render LaTeX from markdown

---

## 6. Concretely — when the SFT pipeline finishes

You'll see in `logs/orchestrate_v2_final.log`:

```
=== STAGE COMPLETE: stage4_sft_a5 ===
=== STAGE START: stage6_eval_all_baselines ===
```

The orchestrator's `stage6_eval_all_baselines` calls
`run_paper_eval.py` for each baseline × test-set. If that ran cleanly,
`outputs/paper_v2/eval/<baseline>/<test_set>.jsonl` will exist for
every (baseline, test_set) pair. From there, run §2.4 mechanical
scoring and start filling in tables.

If the orchestrator failed somewhere in stage6, manually run §2.1–2.4
above.

---

## 7. Risk register / things easy to mess up

- **Don't conflate v1 and v2 A1.** v1 A1 had DPO; v2 A1 is SFT-only.
  When citing "A1" in v2 paper, always mean v2 SFT-only.
- **Sentinel detector compatibility.** Verify
  `scripts/score_redirect_mechanical.py` handles `[SESSION_END]` (the
  generic form). Substring `[SESSION_END]` matches both — should be
  fine, but if A6/A7 sentinel F1 comes back at 0.00, this is the first
  thing to check.
- **A6/A7 adapter eval needs the right env var.** The orchestrator's
  `stage4_sft_retrain` sets `QWEN_TUTOR_SENTINEL_FORMAT=generic` for
  A6/A7 training. The eval needs the same env var when generating
  responses with the A6/A7 adapter, OR the eval needs to recognize
  both sentinel formats agnostic of model. Verify by inspecting
  `outputs/paper_v2/eval/paper_a6_sft/persistent_probe.jsonl` — the
  model's outputs should contain `[SESSION_END]` (no axis label).
- **Trimming applies only to A5/A6/A7.** A1's persistent training data
  was untrimmed (full 10-message records). If A1 sentinel F1 in v2 is
  notably higher than v1's 0.836, the trim is a candidate
  contribution — note this as a confound.
- **DPO is gone in v2.** The v1 N1 contrast "A1 (SFT+DPO) vs A2"
  is no longer a thing. Any reference to DPO contribution in §5 needs
  to be deleted or moved to a "v1 reference numbers" appendix.
