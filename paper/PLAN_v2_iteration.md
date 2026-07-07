# PLAN_v2_iteration.md

Updated 2026-06-25 after the pairwise eval cleanup uncovered two
issues that the v1 PLAN_v2_iteration did not address:

1. Specialized-redirect SFT streams are too small (~7 records/level)
   to reliably train niche tutor behaviors. Three streams memorized
   the prompt's exact example phrasing (formulaicness) rather than
   learning the underlying behavior.
2. Two teacher-prompt templates teach the **wrong behavior** for
   their axis: `language_redirect` trains "respond in English" (which
   the deployment prompt already commits to — redundant), and
   `pedagogy_redirect` trains "give a brief casual answer" (the
   opposite of the paper's "withholding" claim).
3. Deployment system prompts diverge between A1-A5 (trained, raw
   deployment prompt) and B1-B4 (off-the-shelf, raw prompt plus an
   added response-shape addendum) — breaking matched-prompt parity.

The v2 cycle below fixes all three, regenerates the affected data,
retrains the SFT/DPO conditions, and re-runs the full evaluation.

---

## Scope

- **Specialized redirect streams (all 6 axes)**: regenerate at
  **25 records / CEFR level × 6 levels = 150 / axis**.
  6 axes × 150 = **900 specialized records** total.
  Axes: locale_redirect, language_redirect, pedagogy_redirect,
  persona_redirect, role_swap_redirect, topic_redirect.
- **Persistent streams**: regenerate at **×1.5 of current counts**.
  (currently ~250/level → ~375/level, all 4 persistent streams.)
- **Generic redirect** (`redirect`): **keep current**. Already at
  reasonable size; behavior is deployment-prompt-aligned.
- **Base tutor_scenario data**: **keep current**.
- **Locale_leakage probe data**: **keep current**.

## What gets deleted

Old specialized SFT records become stale once the prompts are
rewritten (Stage 0). The cleanest path is to delete and rebuild
rather than mix old and new examples:

- `data/sft_raw/locale_redirect_*.jsonl`
- `data/sft_raw/language_redirect_*.jsonl`
- `data/sft_raw/pedagogy_redirect_*.jsonl`
- `data/sft_raw/persona_redirect_*.jsonl`
- `data/sft_raw/role_swap_redirect_*.jsonl`
- `data/sft_raw/topic_redirect_*.jsonl`
- `data/sft_filtered/*_redirect_*.jsonl` (re-derived from raw)
- `data/dpo_pairs/*` (DPO chosen/rejected pairs are derived from the
  current SFT model; they become stale after the SFT retrain anyway)

Persistent streams: regenerate ADDITIONAL records (×0.5 of current)
and merge with existing — no need to delete current persistent data
since the persistent prompts have not been identified as buggy.

Back everything up under `data_backup_pre_v2_<date>/` first.

---

## Stage 0 — Prompt fixes (text-only, ~1-2h)

### 0.1 — `language_redirect` prompt

Current prompt teaches "paraphrase the L1 turn meaning in English"
(redundant with the deployment prompt's "respond in English"). Fix
the handling rules to teach **L1-acknowledgment as a separate
behavior**:

```text
HANDLING RULES (speaks_l1):
- Tutor MUST do all three, in order:
  1. ACKNOWLEDGE the L1 switch in one short clause ("I see you
     switched to Chinese", "You used Chinese there", "I noticed
     you wrote that in Chinese", etc.) — VARY the wording across
     examples; do NOT reuse the same opener.
  2. INVITE English with a brief practice-rationale ("Can you try
     in English?", "Let's try that in English — it's how you
     grow.", etc.) — VARY the wording.
  3. OPTIONALLY offer a scaffold ("If you need a word, ask me",
     "I can help if you get stuck on a phrase").
  Continue the scenario in English. NEVER paraphrase the L1 meaning
  as if it were the user's English turn — that defeats the example.

HANDLING RULES (requests_l1):
- Tutor declines with three parts:
  1. EMPATHIZE briefly ("I hear you", "I understand it can feel
     tiring") — VARY wording.
  2. EXPLAIN why English is best here, in one short clause.
  3. OFFER fallback support ("if a word is missing, ask me").
  NEVER reuse exact phrasings like "Practicing in English will help
  you more"; vary each example.

HARD FAIL (auto-rejected):
- speaks_l1: tutor response does NOT mention the L1 switch
  (no mention of language, switching, or L1 — auto-rejected).
- speaks_l1: tutor paraphrases L1 meaning into English as if it were
  the user's English turn.
- requests_l1: tutor response does NOT mention English practice or
  the value of staying in English.
- Any tutor turn contains L1 / native script.
- The "L1 turn" is in English, pinyin, romaji, or other romanization.
```

### 0.2 — `pedagogy_redirect` prompt

Current prompt teaches "give a brief casual answer in one sentence"
— the opposite of the paper's "pedagogical withholding" claim. Fix
the handling rules to teach **scaffolding without giving the answer**:

```text
CONVERSATIONAL HANDLING RULES:
- Tutor briefly acknowledges the ask in ONE short clause — VARY
  wording across examples; do not reuse "Good question!" verbatim.
- Tutor then does ONE of the following (pick whichever fits the
  scenario; vary across examples):
    A) COUNTER-QUESTION: "What do YOU think it might be?" /
       "Try saying it your way and we'll see together."
    B) HINT: "Think about what 'have' usually means..." /
       "Listen for the verb — what's the action?"
    C) PARTIAL SCAFFOLD: model the form in a USE-context that hides
       the rule rather than stating it ("Hmm, I HAVE a brother. How
       about you?") — never label the form as 'the rule', never
       follow with grammar metalanguage.
- Tutor then returns to the scenario topic via a follow-up question.
- NEVER state the rule directly ("the rule is X", "you say it like
  this: Y").
- NEVER provide the conjugation table, vocabulary list, or
  explanation the learner asked for.
- NO bullet lists. NO grammar metalanguage at A1-A2. At B1+ a brief
  observation is okay if it's framed as a question, not a rule.

HARD FAIL (auto-rejected):
- Tutor states the rule directly ("the rule for X is Y").
- Tutor provides the requested list, table, or explicit translation.
- Tutor uses bullets, numbered rules, or "Rule 1..." structure.
- Tutor response opens with "Good question!" verbatim (formulaic).
```

### 0.3 — Other prompts (sanity check)

Audit:

- `locale_redirect` prompt: already audited in §X of this doc; check
  for canned example phrasings to vary.
- `persona_redirect` prompt: confirmed good; multiple example tones,
  clear NO-rules.
- `role_swap_redirect` prompt: confirmed good; 2-sentence structure
  documented; covers direct_swap + incremental_swap.
- `topic_redirect` prompt: read and confirm no formulaic seeds.

### 0.4 — Deployment system prompt: targeted changes

Two changes needed beyond the SFT-prompt fixes:

**0.4a — Fold the response-shape addendum INTO the deployment prompt.**

Currently A1-A5 use the bare deployment scenario prompt; B1-B4 use
the bare prompt PLUS `_TUTOR_RESPONSE_SHAPE_ADDENDUM`. Simply
stripping the addendum would let off-the-shelf B1-B4 produce
runaway-length responses, and judges' verbosity bias would
unfairly penalize the trained A1-A5 (which follow the prompt's
length clauses correctly).

The fix: **fold the addendum INTO the deployment prompt itself** so
all 9 baselines see the same explicit length constraints.

The addendum content (~line 418 of `scripts/run_paper_eval.py`)
to fold in:

```text
[response shape]
This is a 1-on-1 tutoring session. Produce ONE short turn -- 1-3
sentences at A1/A2, 2-4 at B1/B2, 3-4 at C1/C2 -- then WAIT for the
learner to reply. Do NOT:
- introduce yourself as an AI or assistant
- write a multi-paragraph welcome message
- use markdown bullets, numbered lists, or section headers
- enumerate 3+ questions or options in one turn
- summarize what you will help with -- just help
```

This becomes part of the deployment system prompt template in
`src/qwen_tutor/generation/prompts.py` for all baselines (A1-A5 use
it at both train and eval time, B1-B4 use it at eval time).

After folding, `_AUGMENT_FOR_BASELINES` becomes unused — set it to
`frozenset()` in `scripts/run_paper_eval.py` so the dead code is
inert.

**Why this matters**: the deployment prompt now explicitly caps
length per CEFR level for everyone. Off-the-shelf B1-B4 that ignore
this explicit cap fail the deployment prompt's contract. Judges'
residual verbosity bias is bounded because the length differential
shrinks (B1-B4 won't be 5× longer than A1-A5; more like 1.5×). The
comparison becomes honest: *who follows the deployment prompt's
explicit contract?*

**0.4b — Modify the language clause to match the new SFT behavior.**

Current clause (in `src/qwen_tutor/generation/prompts.py` deployment
prompt template, ~line 2384):

```text
Respond ONLY in English, even if the learner switches to another
language. Do not code-switch or quote long non-English passages. If
the learner addresses you in their L1, respond in English while
staying in character.
```

Proposed replacement:

```text
If the learner switches to their L1 mid-lesson, briefly acknowledge
the switch in one short clause, then invite them back to English. If
they get stuck on a word, offer to help in English. Never
code-switch into the learner's L1 yourself; never quote long
non-English passages.
```

This makes the deployment prompt and the new language_redirect SFT
prompt **consistent** — both describe ACK + INVITE + (optional
SCAFFOLD). Avoids the train/inference contradiction.

**0.4c — Pedagogy stays silent.** Deliberately do NOT add a pedagogy
clause to the deployment prompt. Pedagogy remains the canonical
"unpromptable" axis whose behavior the SFT data alone teaches. This
preserves the §3.3 framework's strongest theoretical claim.

**0.4d — Persona stays implicit.** "Stay in character as
{model_role_name}" is kept; no explicit "don't disclose being an AI"
clause is added. Preserves persona as the "implicitly-covered"
middle case between fully-prompt-covered and pedagogy's fully-
uncovered.

This gives the cleanest theoretical test in v2:

- 4 fully-prompt-covered axes (locale, role_swap, topic, language):
  test "does specialized data improve execution reliability of the
  stated behavior?"
- 1 implicitly-covered axis (persona): test "does specialized data
  fill a partial gap?"
- 1 fully-uncovered axis (pedagogy): test "does specialized data
  fill a complete gap?"

---

## Stage 1 — Backup + delete stale data (~5 min)

```bash
# Backup
mkdir data_backup_pre_v2_20260625
cp -r data/sft_raw data_backup_pre_v2_20260625/
cp -r data/sft_filtered data_backup_pre_v2_20260625/
cp -r data/dpo_pairs data_backup_pre_v2_20260625/ 2>/dev/null
cp config/generation.yaml data_backup_pre_v2_20260625/

# Delete the 6 specialized streams (all CEFR levels)
rm data/sft_raw/locale_redirect_*.jsonl
rm data/sft_raw/language_redirect_*.jsonl
rm data/sft_raw/pedagogy_redirect_*.jsonl
rm data/sft_raw/persona_redirect_*.jsonl
rm data/sft_raw/role_swap_redirect_*.jsonl
rm data/sft_raw/topic_redirect_*.jsonl

# Delete derived filtered + DPO (rebuild from new raw)
rm data/sft_filtered/{locale,language,pedagogy,persona,role_swap,topic}_redirect_*.jsonl
rm -rf data/dpo_pairs/<specialized> 2>/dev/null
```

`generation.yaml` target counts to set to 25/level for specialized,
×1.5 of current for persistent.

---

## Stage 2 — Smoke regen (~30 min teacher inference + 30 min manual review)

Regenerate **5 records per (axis, CEFR level)** = 6 axes × 6 levels
× 5 = 180 smoke records. Then **manually inspect** the smoke output:

- For each axis: read 5-10 tutor responses at random levels.
- Verify the tutor response performs the intended behavior:
  - language: explicit L1-ack present, not paraphrase
  - pedagogy: counter-question / hint / partial scaffold present,
    not direct rule
  - locale/persona/role_swap/topic: per their prompt's behavior spec
- If a stream looks wrong (e.g. teacher reverts to formulaic phrasing
  despite the VARY instruction), tighten the prompt and re-smoke that
  axis only.

This step costs ~1 h but catches prompt bugs before committing to
full regen (~6-8 h). Skip at your peril.

---

## Stage 3 — Full data regen (~6-10 h teacher inference)

- Specialized: 6 axes × 25/level × 6 levels = **900 specialized records**.
- Persistent: ×1.5 of current → produce ~375 additional records per
  persistent stream (~1500 additional total).
- Filter pipeline runs as normal (`scripts/run_filter_pipeline.py`):
  pass rates expected to be similar (~80%).

Estimated wall-time on local Qwen3.5 9B Q4_K_XL teacher:

- 900 + 1500 = 2400 generations at ~10-30 sec each with parallelism
  → roughly 5-10 h.

---

## Stage 4 — SFT retrain (~30-40 h)

Retrain on the new SFT-filtered data. The conditions to retrain:

| Condition | SFT data composition | Why retrain |
| --- | --- | --- |
| **A1 / A2** | full SFT (all 12 streams, new specialized) | core "with-specialized" adapter |
| **A3** | drop 6 specialized streams (generic + persistent only) | the control: tests specialized data's contribution |
| **A4** | drop fixed-turn-7 anchor only | sentinel-firing ablation |
| **A5** | fixed-turn-7 anchor variant | sentinel-firing ablation |

Each ~10-13 h on RTX 3060 12 GB. Run sequentially:

- A1 (= A2 for SFT) → ~12 h
- A3 → ~10 h
- A4 → ~12 h
- A5 → ~12 h
- Total: ~46 h sequential

Output paths should be **fresh** (`outputs/paper_v2/{a1,a3,a4,a5}/sft/`)
to preserve existing v1 adapters as a baseline.

---

## Stage 5 — DPO data regen + DPO retrain (~10 h)

Current DPO pairs are stale (rejected responses came from the v1 SFT
student trained on bad data). Regenerate after Stage 4 SFT completes:

1. New SFT student (A1) generates "rejected" responses on tutor_scenario.
2. Teacher (or judge) selects "chosen" responses.
3. DPO retrain on new pairs (~5 h per condition; A1 only — A3-A5 are
   SFT-only ablations).

---

## Stage 6 — Evaluation (~6-8 h)

After A1/A3/A4/A5 are trained, EVERYTHING re-evals from scratch because
the deployment system prompt changes in two ways (language clause
modified, B1-B4 addendum stripped). No v1 generations are reusable.

### 6.1 — Rebuild eval_sets

`scripts/build_eval_sets.py` rebuilds against the new SFT data. The
axis validator (added 2026-06-25) stays in place.

### 6.2 — Regenerate ALL adapter responses

Both the trained adapters (A1-A5) AND the off-the-shelf baselines
(B1-B4) need new generations because:

- A1-A5: trained on new SFT data with the new deployment prompt clause
- B1-B4: deployment prompt now has the new language clause AND the
  response-shape addendum has been stripped (matched-prompt parity)

```bash
# all 9 baselines, all 5 test sets, ~9000 generations total
for b in paper_a1 paper_a2 paper_a3 paper_a4 paper_a5_sft \
         qwen3_5_0_8b_base qwen3_5_0_8b_instruct \
         qwen3_5_4b_instruct qwen3_5_9b_teacher; do
  python scripts/run_paper_eval.py --baseline "$b" --test-set all
done
```

Estimated 30-60 min per baseline (slower for the 9B teacher). Total
~5-7 h with sequential llama-server swaps for the GGUF baselines.

### 6.3 — Mechanical scoring

`scripts/score_redirect_mechanical.py`, locale-leakage gazetteer,
persistent metrics. Fast (~30 min).

### 6.4 — Per-axis F1

`scripts/run_paper_score.py --metric redirect_axis` on the cleaned
eval set, with the Llama-3.1 + Gemma-2 ensemble (Prometheus dropped
from this metric — see v1 notes). ~1 h.

### 6.5 — Pairwise on all 6 axes

`scripts/score_pairwise_preference.py` on A2 (keep) vs A3 (drop),
3-judge ensemble. Locale-aware prompts, 4 criteria, no naturalness —
same prompt design as the cleaned 2026-06-25 run. ~12 min for ~150
specialized pairs × 3 judges.

### 6.6 — Naturalness + locale-leakage judge runs

Same as v1.

---

## Stage 7 — Paper update (~half day)

Rewrite the affected sections with the new numbers:

- §1 contribution claims (broadened — see "narrative shift" below)
- §3.3 framework prediction (reframe around deployment-prompt
  coverage; see narrative shift below)
- §4 (data composition table updated)
- §5.4.2 (mechanical with new numbers)
- §5.4.3 (per-axis F1 with new numbers)
- §5.4.5 (pairwise table with all 6 axes, cleaned probe, new data)
- §6 (caveats: explain pedagogy mismatch if it persists, language
  ack acquisition at the new scale)

---

## Narrative shift (anticipated, from the pairwise-cleanup analysis)

The current paper says "specialized data necessary for pedagogy
only; other axes promptable, null expected, null observed." The
cleaner v2 story is:

- **The deployment system prompt covers** locale, language,
  role_swap, topic (explicit clauses).
- **It does NOT cover** persona (only implicit "stay in character")
  or pedagogy (no clause at all).
- At v1 SFT scale (~7/level specialized), the prompt-covered axes
  showed measured parity (specialized data redundant); persona
  showed clear keep-advantage (specialized data necessary).
- At v2 scale (25/level), the prediction is:
  - prompt-covered axes (locale/language/role_swap/topic): may
    show modest keep advantage as the specialized stream gains the
    headroom to reinforce reliable execution of the prompt's rule.
  - persona/pedagogy: should show clearer keep-advantage at scale.

If v2 confirms this, the contribution generalizes: specialized SFT
data adds *execution reliability* on top of *the prompt's stated
behavior*. Pedagogy specifically is the strongest test because the
prompt commits to nothing on this axis.

---

## Critical: matched-prompt parity (NEW v2 invariant)

**Single deployment system prompt across A1-A5 + B1-B4.** No
response-shape addendum for off-the-shelf baselines. The deployment
prompt's existing length/format clauses (CEFR register, "one or two
sentences", no markdown) are the matched-prompt floor. Any baseline
that produces long off-shape responses despite the prompt is a
legitimate baseline-vs-trained signal.

Files to edit:

- `scripts/run_paper_eval.py` — remove `augment_system_for_offshelf()`
  call site OR make `_AUGMENT_FOR_BASELINES = frozenset()`.

---

## Output directory hygiene (must-do before launch)

The v1 PLAN had a known bug where v2 SFT configs pointed to
`outputs/paper/aN/sft/` — same path as v1 adapters. v2 launch wrote
to that path and could have overwritten v1 (caught and killed before
any writes).

**v2 retrain configs MUST write to** `outputs/paper_v2/aN/sft/`:

- `config/paper/training_a1_full.yaml` → set `output_dir:
  outputs/paper_v2/a1/sft`
- same for a3, a4, a5
- DPO configs (`config/paper/training_a1_full_dpo.yaml`) → output
  to `outputs/paper_v2/a1/dpo`

---

## Estimated total wall-time

| Stage | Description | Time |
| --- | --- | --- |
| 0 | Prompt fixes (text editing) | 1-2 h |
| 1 | Backup + delete | 0.1 h |
| 2 | Smoke regen + manual inspect | 1 h |
| 3 | Full data regen | 6-10 h |
| 4 | SFT retrain (A1, A3, A4, A5) | 40-50 h |
| 5 | DPO data + retrain | 8-10 h |
| 6 | Evaluation | 3-4 h |
| 7 | Paper update | 4-6 h |
| **TOTAL** | end-to-end | **~70-80 h** |

Feasible across 4-5 days of compute + 1 day of paper editing.

---

## Orchestrator hygiene (lesson learned from v1)

The v1 orchestrator chain (`orchestrate_pairwise_followup.py`,
`orchestrate_v2_iteration.py`) used marker-based polling and fired
prematurely on stale markers. For v2:

- **Each orchestrator records a startup-time checksum of its
  upstream log**. Only treat markers written AFTER that checksum as
  triggers.
- **OR** truncate the upstream log before launching a downstream
  orchestrator.

This applies to: SFT-retrain → eval-gen → scoring chain in Stage 6.

---

## What does NOT change

- 0.8B student base model
- LoRA r=16 + QLoRA + bfloat16 recipe
- Teacher model (Qwen3.5 9B Q4_K_XL)
- 3-judge ensemble for pairwise (Prometheus + Llama-3.1 + Gemma-2)
- 2-judge ensemble for redirect_axis F1 (Llama-3.1 + Gemma-2;
  Prometheus dropped due to format incompatibility)
- Evaluation methodology (5 test sets, mechanical + judge metrics)
- Single locale (china) — multi-locale is future work
