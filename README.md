# qwen-en-tutor

Fine-tune **Qwen3-8B** into an English conversation tutor for adult learners,
locale-grounded to a target country of your choice (default: China).

A single 8B model learns two modes:

- **Conversation mode** (`/no_think`) — natural, register-appropriate tutoring
  turns grounded in the target country's daily life.
- **Evaluation mode** (`/think`) — examiner-style assessment that emits a
  `<think>...</think>` reasoning block followed by a structured
  `EvaluationOutput` JSON.

The whole pipeline (data generation, filtering, training, evaluation) is
designed to run **fully offline** on a single machine. There is no required
cloud API call. A local `llama.cpp` HTTP server hosts the teacher model
(gpt-oss-20B by default) and serves an OpenAI-compatible endpoint that the
generation code talks to over `127.0.0.1:8080`.

> **Important.** This entire document assumes you have **no internet access**
> on the training/generation machine. The setup is split into two phases:
> (1) bundle wheels + models on a connected workstation, (2) zip-and-copy to
> the offline machine, then install + run from the bundle.

---

## Contents

1. [Hardware and constraints](#1-hardware-and-constraints)
2. [One-time offline bundle](#2-one-time-offline-bundle-on-a-connected-machine)
3. [Install on the offline machine](#3-install-on-the-offline-machine)
4. [Start the local teacher (llama.cpp)](#4-start-the-local-teacher-llamacpp)
5. [Run the generation pipeline](#5-run-the-generation-pipeline)
6. [Run the training pipeline](#6-run-the-training-pipeline)
7. [How to change the target country](#7-how-to-change-the-target-country-localeyaml)
8. [How to change `avoid_default_cultures`](#8-how-to-change-avoid_default_cultures)
9. [How to change `avoided_topics`](#9-how-to-change-avoided_topics)
10. [How to change the pipeline stages](#10-how-to-change-the-pipeline-stages)
11. [Generation data types (what each file is)](#11-generation-data-types-what-each-file-is)
12. [Training pipeline stages](#12-training-pipeline-stages)
13. [Deploying the trained tutor](#13-deploying-the-trained-tutor)
14. [Config reference](#14-config-reference)
15. [Troubleshooting](#15-troubleshooting)
16. [Switching `/think` vs `/no_think` (Qwen3.5 thinking mode)](#16-switching-think-vs-no_think-qwen35-thinking-mode)
17. [From smoke run to full training (parameters to change)](#17-from-smoke-run-to-full-training-parameters-to-change)
18. [Important things not to forget](#18-important-things-not-to-forget)
19. [True offline operation (air-gapped checklist)](#19-true-offline-operation-air-gapped-checklist)
20. [Known limitations and rough edges](#20-known-limitations-and-rough-edges)

> **Reproducing the paper?** End-to-end run order
> (generate → ablate → train → eval → score → build PDF) lives in
> [`paper/README.md`](paper/README.md). The LaTeX build alone is in
> [`paper/latex/README.md`](paper/latex/README.md), and offline
> installers for Pandoc and MiKTeX are vendored under
> [`vendor/installers/`](vendor/installers/README.md).

---

## 1. Hardware and constraints

| Stage                              | RTX 3060 (12 GB)        | RTX 5090 (32 GB) |
| ---------------------------------- | ----------------------- | ---------------- |
| seeds → SFT → redirect → eval → DPO register pairs | ok via local llama.cpp (slow) | ok |
| filtering (incl. locale judge)     | ok                      | ok |
| SFT training (Qwen3-8B QLoRA)      | **does not fit** alongside the 20B teacher; stop the teacher first | ok |
| DPO training                       | same                    | ok |
| Holdout evaluation                 | trained model + teacher both want VRAM | ok |
| On-policy DPO pair generation      | needs SFT adapter loaded simultaneously | ok |

On a 12 GB card the realistic flow is: **generate + filter** with the teacher
running, **stop the teacher**, then **train** (or move to a larger card for
training). On 24 GB+ everything can run back-to-back.

OS: tested on **Windows 11 + PowerShell** with an RTX 3060. The Linux/WSL
equivalents (`install_offline.sh`, `start_llama_cpp.sh`) exist and follow the
same flow.

---

## 2. One-time offline bundle (on a connected machine)

Do this **once**, on a workstation that has internet. It downloads
everything the offline machine will ever need and writes it under
`vendor/`. Disk footprint: roughly 30 GB.

```powershell
cd qwen-en-tutor
python -m pip install huggingface_hub        # needed by the bundler itself
python scripts/setup_offline.py
```

What lands under `vendor/`:

```text
vendor/
  wheels/                              pip wheels for every dep (~3 GB)
                                       + en_core_web_sm wheel
                                       + a prebuilt qwen_en_tutor-*.whl
  models/Qwen3-8B/                     HF snapshot of the base model (~16 GB)
  models/gpt-oss-20b-GGUF/             teacher GGUF, Q4_K_M default (~12 GB)
  manifest.json                        what was downloaded + when
```

You can re-pull just one piece with `--only wheels`, `--only models`, or
`--only spacy`. To swap the teacher quantization:

```powershell
python scripts/setup_offline.py --only models `
  --gpt-oss-repo bartowski/gpt-oss-20b-GGUF `
  --gpt-oss-pattern "*Q5_K_M*.gguf"
```

### llama.cpp binaries are NOT bundled

They are platform-specific. Grab a prebuilt release matching your machine
(CUDA version on the target box) from
<https://github.com/ggerganov/llama.cpp/releases> and unzip it somewhere
stable inside the project, e.g. `vendor/llama_cpp/`.

### Copy to the offline machine

Zip the **entire project directory** including `vendor/` and the unzipped
`vendor/llama_cpp/`, copy via USB / LAN to the offline machine, unzip it.

---

## 3. Install on the offline machine

From the project root on the offline machine:

```powershell
# Windows / PowerShell
pwsh scripts/install_offline.ps1
```

```bash
# Linux / macOS
bash scripts/install_offline.sh
```

The script:

1. Creates `.venv/` in the project root.
2. Installs every Python dep with `pip install --no-index --find-links vendor/wheels/`.
3. Installs the `en_core_web_sm` spaCy wheel.
4. Installs the project itself (prebuilt wheel if present in `vendor/wheels/`,
   editable install otherwise — both work without internet).
5. Runs the fast pytest suite to confirm imports work.

When it finishes:

```powershell
& .venv\Scripts\Activate.ps1
```

---

## 4. Start the local teacher (llama.cpp)

```powershell
$env:LLAMA_CPP_BIN = "$PWD\vendor\llama_cpp"     # folder containing llama-server.exe
pwsh scripts/start_llama_cpp.ps1

$env:LLAMA_CPP_BIN = "$PWD\vendor\llama_cpp"# folder containing llama-server.exe
powershell -ExecutionPolicy Bypass -File scripts\start_llama_cpp.ps1 -ModelDir "$PWD\vendor\models\GGUF" -ModelGlob "Qwen3.5-9B-UD-Q4_K_XL.gguf"
```


Defaults target a 12 GB card: gpt-oss-20B-Q4\_K\_M, port 8080, ctx 8192,
`--n-gpu-layers 99` (offload-all, falls back to CPU on overflow). Override
with parameters:

```powershell
pwsh scripts/start_llama_cpp.ps1 -GpuLayers 32 -ContextSize 4096 -Port 8080
```

Sanity-check from another shell:

```powershell
curl http://127.0.0.1:8080/v1/models
```

Leave the server running for everything in section 5. To free VRAM for
training, **stop it** (Ctrl+C) before launching `run_training.py`.

---

## 5. Run the generation pipeline

A single entry point reads `config/generation.yaml` and runs every stage in
order. Defaults are tuned for a 12 GB smoke run (4 seeds × 10 dialogues each
across A2 + B1).

```powershell
& .venv\Scripts\Activate.ps1
python scripts/run_generation.py
```

Override per run:

```powershell
# Only A2, smaller smoke
python scripts/run_generation.py --levels A2 --n-per-level 2 --stages seeds,sft

# Re-run filters only (after editing filter code or banned_terms.yaml)
python scripts/run_generation.py --stages filter_sft,filter_eval,filter_dpo
```

Stages, in order:

| Stage              | What it does                                             | Output                                          |
| ------------------ | -------------------------------------------------------- | ----------------------------------------------- |
| `seeds`            | CEFR scenario seeds (topic, subtopics, roles, setting, **life-domain category**) | `data/seeds/<level>.jsonl`           |
| `dedup_seeds`      | (Opt-in) TF-IDF semantic dedup of seeds (see §10.6)      | rewrites `data/seeds/<level>.jsonl` + `.bak`    |
| `sft`              | Normal multi-turn dialogues (10–16 turns)                | `data/sft_raw/normal_<level>.jsonl`             |
| `redirect`         | Off-topic probe at turn N → graceful pivot               | `data/sft_raw/redirect_<level>.jsonl`           |
| `locale_redirect`  | Learner names a wrong-locale entity → tutor handles it   | `data/sft_raw/locale_redirect_<level>.jsonl`    |
| `pedagogy_redirect`| Learner asks for grammar lecture → tutor reframes        | `data/sft_raw/pedagogy_redirect_<level>.jsonl`  |
| `language_redirect`| Learner speaks L1 / asks tutor for L1 → tutor pivots     | `data/sft_raw/language_redirect_<level>.jsonl`  |
| `persona_redirect` | "Are you AI?" / "What model are you?" → in-character pivot | `data/sft_raw/persona_redirect_<level>.jsonl` |
| `topic_redirect`   | Learner drifts off the scenario topic → tutor bridges back | `data/sft_raw/topic_redirect_<level>.jsonl`   |
| `role_swap_redirect` | Learner tries to take the tutor's role → tutor stays in-character | `data/sft_raw/role_swap_redirect_<level>.jsonl` |
| `register`         | Multi-axis DPO pairs (9 axes, quota-balanced)            | `data/dpo_raw/register_<level>.jsonl`           |
| `eval`             | `/think` examiner examples for evaluation-mode training  | `data/eval_raw/<level>.jsonl`                   |
| `filter_sft`       | Filter pipeline over all 8 SFT streams                   | `data/sft_filtered/*_passed.jsonl` + `*_failed.jsonl` |
| `filter_eval`      | Same filters over eval examples                          | `data/eval_filtered/*_passed.jsonl`             |
| `filter_dpo`       | Same filters over register pairs                         | `data/dpo_filtered/*_passed.jsonl`              |
| `top_up`           | (Opt-in) Detect per-pool category deficits after filtering and regen seeds for the worst-off categories | new rows in `data/seeds/<level>.jsonl` |

All stages **resume from existing output** — if you stop and rerun, the
already-written IDs are skipped. Failures land in `data/*_failures.jsonl`
with the model output that caused them.

See [section 10.5](#105-category-taxonomy-and-top-up-rebalancing) for how
the `category` field drives quota-balanced generation and how `top_up`
fixes survivor skew after filtering. See
[section 11](#11-generation-data-types-what-each-file-is) for what each
of those data types looks like and why it exists.

### Reporting category distribution

After generation + filtering, scan any filtered pool to see what survived
per life-domain category:

```powershell
python scripts/run_diversity.py                                   # default: data/sft_filtered
python scripts/run_diversity.py --input-dir data/eval_filtered
python scripts/run_diversity.py --input-dir data/dpo_filtered
python scripts/run_diversity.py --json-out data/diversity.json    # also save JSON
```

The report includes top names / cities / foods (regex extraction over
assistant text) and per-category example counts (from metadata).
Anything exceeding 25% share of its bucket triggers a warning.

---

## 6. Run the training pipeline

> Requires a GPU with at least 24 GB free. On a 12 GB card, stop
> `llama-server` first; training will not fit alongside it.

```powershell
& .venv\Scripts\Activate.ps1
python scripts/run_training.py
```

Override stages:

```powershell
# SFT only
python scripts/run_training.py --stages train_sft,eval_intermediate

# Re-run evaluation + compare only
python scripts/run_training.py --stages eval_intermediate,eval_final,compare
```

Stages, in order ([section 12](#12-training-pipeline-stages) has details):

```text
train_sft          → SFT LoRA training
eval_intermediate  → holdout evaluation of the SFT-only model
on_policy_gen      → SFT adapter regenerates one turn per SFT example,
                     judge picks chosen vs rejected
filter_dpo         → re-filter register + on_policy pairs together
train_dpo          → DPO on top of the SFT adapter
eval_final         → holdout evaluation of the post-DPO model
compare            → intermediate vs final metrics report
```

Outputs:

```text
outputs/sft/                 SFT LoRA adapter
outputs/dpo/                 DPO LoRA adapter (on top of SFT)
data/eval_results/<run_id>/  per-example dialogues + summary.json + samples.md
```

---

## 7. How to change the target country (`locale.yaml`)

**Everything locale-related lives in [config/locale.yaml](config/locale.yaml).**
Change this one file and the prompts, the LLM locale judge, the diversity
metrics, the banned-terms behavior, and the deployment system prompt all
follow.

```yaml
# config/locale.yaml
country: "China"                # noun  - "Iran", "Japan", "Korea", "Mexico" ...
country_adjective: "Chinese"    # adjective form
learner_description: "adult learners of English"
```

There is no static city / food / name list. The teacher model already
knows the country's daily life — telling it `country` is enough; it fills
in cities, foods, neighborhoods, transit, currency on its own. The same
country name is interpolated into:

1. **Every generation prompt** (`src/qwen_tutor/generation/prompts.py` and
   `prompts_compact.py`) — the `LOCALE INSTRUCTION` block at the top of
   each prompt is rebuilt from `locale.yaml` at module load.
2. **The deployment system prompt** in `config/training.yaml` —
   `{country}`, `{country_adjective}`, `{learner_description}` placeholders
   are substituted at training-config load time, then baked into the
   adapter during SFT/DPO.
3. **The evaluation system prompt** (same path).
4. **The locale LLM judge** under `src/qwen_tutor/generation/filters/locale_judge.py`.
5. **The diversity report** that counts top cities/foods/names in
   filtered output.

After editing `locale.yaml`:

```powershell
# Re-generate (existing data still under the old locale will be skipped
# by IDs already in the output files — delete those files first if you
# want a clean rebuild).
python scripts/run_generation.py
```

For a **fully clean** locale swap, also clear the existing data:

```powershell
Remove-Item data\seeds\*.jsonl, data\sft_raw\*.jsonl, data\sft_filtered\*.jsonl, `
    data\dpo_raw\*.jsonl, data\dpo_filtered\*.jsonl, data\eval_raw\*.jsonl, `
    data\eval_filtered\*.jsonl -ErrorAction SilentlyContinue
python scripts/run_generation.py
```

---

## 8. How to change `avoid_default_cultures`

The teacher model has dominant default biases (usually American / British)
and **secondary** ones — a Chinese-target model leaks Japanese names, a
Singapore-target model leaks Japanese street names, etc. List those secondary
cultures here so prompts, anti-failure-mode blocks, and the LLM judge all
treat them as "wrong default":

```yaml
# config/locale.yaml
avoid_default_cultures:
  - "American"
  - "European"
  # - "Japanese"      # add when country == "China" or "Korea"

```

Use the **adjective form** (matches `country_adjective` style). Leaving
the list empty means "only avoid Western (American/European) defaults" —
that is the global baseline already baked into the prompts.

Where this flows:

- The anti-failure block in `prompts.py` / `prompts_compact.py` gets a
  `Do not default to <list> names/places/foods/brands.` sentence.
- `LocaleLLMJudge` includes these cultures in its "wrong-locale" criteria.
- `banned_terms.yaml` is **not** auto-edited by this list. If you observe
  a specific name/place leaking that the LLM judge misses, add it manually
  to `config/banned_terms.yaml` (matching is case-sensitive).

After editing, regenerate (no code changes needed).

---

## 9. How to change `avoided_topics`

`avoided_topics` controls what topics the tutor should pivot away from
when the learner brings them up. Every entry has a `name` (snake_case
identifier) and a `pivot_hint` (a short alternative-topic suggestion to
help the small teacher model stay on the rails):

```yaml
# config/locale.yaml
avoided_topics:
  - name: politics
    pivot_hint: "infrastructure, local services, weather, weekend plans"
  - name: religion
    pivot_hint: "cultural traditions, family holidays, food"
  - name: alcohol_dating
    pivot_hint: "friendship, weekend plans with family"
  - name: partisan_history
    pivot_hint: "general history of cities or food traditions"

  # Add freely:
  # - name: economics
  #   pivot_hint: "household budgeting, market shopping, general prices"
  # - name: law
  #   pivot_hint: "everyday neighborhood concerns, civic services"
```

Where this flows (all auto, no code edits):

1. **`DIALOGUE_PROMPT_REDIRECT`** — the `redirect_axis` candidate list
   pulled by `redirect.py` when generating off-topic-probe SFT examples.
2. **`REDIRECT_AXES`** constant in `redirect.py` — keeps the redirect
   stream cycling across all named topics.
3. **The deployment system prompt** — the `{avoided_topics_sentence}`
   placeholder in `training.yaml` is substituted with
   `"Stay clear of <topic1>, <topic2>, ..."` at training-config load.
4. **`LocaleLLMJudge`** uses the topic list as part of "should this turn
   have pivoted instead?".

Adding a new entry kicks in **on the next `run_generation.py` invocation**.
New redirect dialogues for that axis appear under
`data/sft_raw/redirect_<level>.jsonl`.

---

## 10. How to change the pipeline stages

The generation pipeline is a flat list of stages, all listed in
`scripts/run_generation.py` as `ALL_STAGES`:

```python
ALL_STAGES = (
    "seeds", "sft", "redirect",
    "locale_redirect", "pedagogy_redirect",
    "language_redirect", "persona_redirect",
    "topic_redirect", "role_swap_redirect",
    "register", "eval",
    "filter_sft", "filter_eval", "filter_dpo",
    "top_up",
)
```

### Run only some stages

```powershell
python scripts/run_generation.py --stages seeds,sft
python scripts/run_generation.py --stages filter_sft,filter_eval,filter_dpo
```

### Change ratios per stage

All of these live in `config/generation.yaml` under `generation:`:

```yaml
generation:
  n_per_level: 4                       # seeds per CEFR level
  cefr_levels: ["A2", "B1"]            # set to all 6 for a full run
  concurrency: 2                       # llama-server in-flight requests
  per_call_size: 3                     # seeds per teacher call
  dialogues_per_seed: 10               # SFT variants per seed (×10 = same scenario, 10 takes)

  # what fraction of the seed pool gets each redirect treatment
  redirect_fraction: 0.20
  locale_redirect_fraction: 0.15
  pedagogy_redirect_fraction: 0.15
  language_redirect_fraction: 0.15
  persona_redirect_fraction: 0.15
  topic_redirect_fraction: 0.15
  role_swap_redirect_fraction: 0.15
  angle_shift_fraction: 0.0          # opt-in; >0 adds normal variants with a shifted learner angle

  # what fraction of the SFT pool becomes /think eval examples
  eval_fraction: 0.25

  # SFT dialogue length (uniform across normal + redirect streams)
  min_turns: 10
  max_turns: 16
```

### Disable a redirect stream entirely

Either pass `--stages` that skips it, or set its fraction to `0.0`. The
stage runs but writes nothing. (Disabling `persona_redirect`, for example,
just drops persona\_break user-side handling out of SFT; the assistant-side
`persona_break` DPO axis still trains.)

### Add a new redirect stream

If you need an entirely new "user violates X — tutor handles" stream:

1. Add a new module under `src/qwen_tutor/generation/` modelled after
   `language_redirect.py` (smallest one).
2. Append a new template to `prompts.py` + `prompts_compact.py`.
3. Wire a new `_stage_*` function and entry in `ALL_STAGES` in
   `scripts/run_generation.py`.
4. Add the new file prefix to `_iter_sft_examples` in both
   `register_pairs.py` and `eval_gen.py` so the new stream feeds DPO + eval.
5. Add a new `RejectionAxis` literal to `src/qwen_tutor/schemas.py` if it
   has a corresponding assistant-side spoil.

### Pipeline filter toggles

```yaml
filtering:
  enable_locale_judge: true            # LLM judge; expensive but catches subtle drift
  enable_naturalness_judge: false      # LLM judge; off by default for cost
  naturalness_sample_rate: 0.0
  concurrency: 2
  short_circuit: true                  # if filter N fails, skip filters N+1..
```

`enable_locale_judge: false` saves ~half of filter time. The mechanical
banned-terms filter still catches the most obvious Western leaks.

### 10.5 Category taxonomy and top-up rebalancing

Every `ScenarioSeed` carries a `category` field — a life-domain tag from
the fixed taxonomy in [`schemas.Category`](src/qwen_tutor/schemas.py):

```text
food_and_dining          shopping_and_services    home_and_neighborhood
family_and_relationships health_and_wellbeing     hobbies_and_leisure
work_and_education       nature_and_weather
travel_and_transit       civic_life
```

(The legacy default `"general"` is used only for pre-categorization data
loaded from disk; new seeds always get a real category.)

**At seed generation time**, the `seeds` stage **round-robin assigns**
categories across the per-(level, locale) quota. The teacher prompt
receives the per-position category list and is told to pick a topic that
fits each one. The on-disk `category` field is stamped from our
assignment regardless of what the teacher echoes back, so the disk
record always matches the requested distribution.

The category propagates: `seed.category` → `SFTExample.metadata.category`
→ `DPOExample.metadata.category` → `EvaluationExample` metadata. Every
downstream artifact can be grouped or sliced by life domain.

**At filter time**, pass-rates differ per category — some categories
produce harder-to-pass examples (e.g. health/civic topics tripping
banned-terms more often than food/family). The post-filter distribution
will not match the requested distribution.

The `top_up` stage closes that gap:

1. Counts per-`(level, locale, category)` survivors in **all three
   filtered pools** (`data/sft_filtered/`, `data/eval_filtered/`,
   `data/dpo_filtered/`).
2. Per triple: `deficit_per_pool = max(0, target_per_category - survivors)`.
   The final deficit is the **max across pools** — top up enough seeds
   to satisfy the worst-off pool.
3. Caps each per-triple request at `max_top_up_per_category` so a
   systematically-failing category cannot blow the seed budget.
4. Generates the targeted seeds and writes them to `data/seeds/<level>.jsonl`.

Configure in [config/generation.yaml](config/generation.yaml):

```yaml
top_up:
  enabled: true                   # opt-in; off-by-default keeps run_generation behavior backward-compatible
  target_per_category: 5          # min survivors per (level, locale, category) per pool
  max_top_up_per_category: 10     # per-triple cap to prevent runaway
  auto_rerun_downstream: true     # ← when true, top_up automatically continues
                                  #   with sft → 7 redirects → register → eval →
                                  #   filter_* in the SAME run_generation.py call
                                  #   so the new seeds reach filtered pools without
                                  #   a second manual invocation
```

`top_up` is **single-shot, not iterative**. A category that keeps failing
after one top-up signals a root-cause problem (prompt or filter), not a
volume problem. Inspect `data/*_filtered/*_failed.jsonl` to see which
filter is killing it before regenerating again.

### Two workflows for `top_up`

#### Recommended: `auto_rerun_downstream: true` (single invocation)

`run_generation.py` writes top-up seeds AND continues through the
downstream stages in the same invocation. Each downstream stage is
resumable, so only the records derived from the new top-up seeds get
generated — much cheaper than re-running from scratch.

```powershell
# One command. top_up writes deficit-fill seeds and the same invocation
# continues with sft → 7 redirects → register → eval → filter_*.
python scripts/run_generation.py

# Verify the new distribution.
python scripts/run_diversity.py --input-dir data/sft_filtered
python scripts/run_diversity.py --input-dir data/dpo_filtered
python scripts/run_diversity.py --input-dir data/eval_filtered
```

#### Alternative: `auto_rerun_downstream: false` (two invocations)

If you want to inspect `top_up`'s deficit report before committing to
the rerun cost (~minutes to hours of teacher time), keep the flag
false. Then:

> **⚠ Easy mistake.** `top_up` runs **last** in `ALL_STAGES`. By the time it
> writes the deficit-fill seeds, every downstream stage (`sft`, all 7
> redirect streams, `register`, `eval`, `filter_*`) has already
> completed. New seeds it writes do **not** auto-flow through. **If you
> stop after step 1 below, the new seeds sit unused on disk and your
> trained model never sees the rebalanced data.** This is exactly what
> happened during one 8h test run in this repo: top_up added 180 seeds,
> nothing processed them, and the diversity report still showed only 5
> of the 10 life-domain categories. Either run step 2 below, or flip
> `top_up.auto_rerun_downstream: true` and use the single-invocation
> workflow above.

```powershell
# 1) Full pipeline. top_up runs last and writes new seeds for deficit categories.
python scripts/run_generation.py

# 2) Re-run downstream so the new seeds become SFT, DPO, eval examples + get filtered.
#    All stages are resumable — only the new seeds get processed.
python scripts/run_generation.py --stages sft,redirect,locale_redirect,pedagogy_redirect,language_redirect,persona_redirect,topic_redirect,role_swap_redirect,register,eval,filter_sft,filter_eval,filter_dpo

# 3) Verify the new distribution.
python scripts/run_diversity.py --input-dir data/sft_filtered
python scripts/run_diversity.py --input-dir data/dpo_filtered
python scripts/run_diversity.py --input-dir data/eval_filtered
```

To preview deficits without committing to a rerun, run `top_up` alone:

```powershell
python scripts/run_generation.py --stages top_up
```

The deficit report (per-pool survivor counts, worst-10 triples by raw
deficit, total seeds queued) is logged before any teacher call.

### 10.6 `dedup_seeds` — semantic deduplication of over-generated seeds

The seed generator's existing dedup is **exact-string only** ([seeds.py:318](src/qwen_tutor/generation/seeds.py#L318)).
Two seeds with topics `"buying apples at the Sanyuanli market"` and
`"purchasing oranges at the Sanyuanli market"` both pass. Once you go
past ~80-100 seeds per `(level, category)` cell with a 4B teacher, near-
duplicates start dominating because the teacher cycles through a finite
pool of distinct scenes / names / cities and the existing dedup misses
the paraphrases.

The `dedup_seeds` stage closes that gap by **over-generate-then-prune**:

1. Group seeds by `(cefr_level, locale, category)` cell.
2. Build TF-IDF vectors over `topic + setting + subtopics` (l2-
   normalized, smoothed IDF).
3. Either threshold-prune (default) or greedy-keep `max_per_cell` most-
   distinct seeds per cell.
4. Rewrite `data/seeds/<level>.jsonl` in place; original snapshotted to
   `<level>.jsonl.bak` on first run.

Pure stdlib — no `sklearn`, no `sentence-transformers`, no model download.
Safe under the offline-mode requirement (§19).

#### Config

```yaml
# config/generation.yaml
dedup_seeds:
  enabled: false                   # opt-in: only useful when over-generating
  similarity_threshold: 0.75       # 0.70 aggressive, 0.75 default, 0.80 light
  max_per_cell: null               # int → cap per (level, locale, category); null = no cap
  backup: true                     # snapshot pre-dedup file to <level>.jsonl.bak
```

#### Workflow: ship 200-300 truly distinct seeds per category to deploy

This is the use case the stage was added for. With a 4B teacher, the
**exact-string dedup alone tops out at ~80-120 genuinely distinct seeds
per category** before near-duplicates dominate. The fix is to over-
generate by ~5x and let `dedup_seeds` pick the most-distinct subset:

```yaml
# config/generation.yaml — for a "ship to deploy" run
generation:
  n_per_level: 500                 # 50 seeds per (level, category) raw
  dialogues_per_seed: 4
  per_call_size: 5

dedup_seeds:
  enabled: true
  similarity_threshold: 0.75
  max_per_cell: 30                 # keep 30 most-distinct per (level, category)
                                   # → ~30 × 6 levels = ~180 per category after dedup
                                   # → after sft + filter (~60% survive) ≈ 100+ shipped
```

```powershell
# Over-generate seeds, then prune to the most-distinct subset.
python scripts/run_generation.py --stages seeds,dedup_seeds

# Inspect what got pruned to sanity-check threshold (optional but recommended)
# Look at data/seeds/_dropped/<level>.jsonl — these are the seeds dedup_seeds dropped.
Get-Content data/seeds/_dropped/A2.jsonl -Head 5

# If threshold looks too aggressive (legitimate scenarios dropped):
#   restore from .bak, raise similarity_threshold to 0.80, re-run
# If too permissive (near-duplicates still present in seeds/<level>.jsonl):
#   restore from .bak, lower similarity_threshold to 0.70, re-run
Copy-Item data/seeds/A2.jsonl.bak data/seeds/A2.jsonl -Force

# When happy with the dedup, run the rest of the pipeline
python scripts/run_generation.py --stages sft,redirect,locale_redirect,pedagogy_redirect,language_redirect,persona_redirect,topic_redirect,role_swap_redirect,register,eval,filter_sft,filter_eval,filter_dpo,top_up
```

#### Why this matters more than `n_per_level` alone

| Approach | Seeds per category (filtered) | Genuinely distinct |
| --- | --- | --- |
| `n_per_level=50`, no dedup | ~30 | ~25 (exact-string dedup is fine at this scale) |
| `n_per_level=300`, no dedup | ~180 | ~80-100 (near-duplicates dominate the tail) |
| `n_per_level=500`, `dedup_seeds.max_per_cell=30` | ~180 | ~160-180 (because the 180 kept were chosen to be mutually distant) |

The second row is the "naive scale-up" trap: more seeds, but most of the
new ones are paraphrases. The third row is the actual win.

#### Tuning notes

- **Threshold calibration.** Start at `0.75`. If `_dropped/<level>.jsonl`
  contains seeds that look meaningfully different from what was kept,
  raise to `0.80`. If `<level>.jsonl` still has obvious near-duplicates
  after dedup, lower to `0.70`. The threshold is on l2-normalized
  TF-IDF cosine, so `0.85+` rarely fires (would require almost word-
  level overlap) and `<0.65` will drop legitimately-different scenarios.
- **`max_per_cell` interaction.** When set, greedy farthest-point still
  honors the threshold — it stops early if every remaining seed in a
  cell exceeds the threshold, rather than fill the quota with near-dups.
- **`.bak` is sacred.** On reruns, `dedup_seeds` will NOT overwrite an
  existing `.bak`. The original snapshot is always preserved. To re-
  dedup with different params, restore from `.bak` first.
- **Top-up interaction.** `top_up` writes new seeds after filtering. If
  you want those also deduped against the kept set, re-run
  `--stages dedup_seeds` after `top_up`. In practice top_up generates
  at most `max_top_up_per_category` (≤15) seeds per cell, well within
  the diversity window, so this is rarely necessary.

### 10.7 Deploying directly from seeds (no conversion step)

After `dedup_seeds` has produced a clean, diverse seed pool, you can
feed **`data/seeds/<level>.jsonl` directly to the deploy runtime** —
no conversion / publishing step. The training `ScenarioSeed` schema is
a superset of the deploy `Scenario` schema (same `topic` / `subtopics` /
`user_role` / `model_role` fields), so `Scenario.from_dict` happily
accepts a seed dict and ignores the 5 training-only extras
(`id`, `setting`, `cefr_level`, `locale`, `category`).

Two helpers live on `Scenario` ([tutor.py](src/qwen_tutor/deploy/tutor.py)):

- `Scenario.from_json_file(path)` — single-JSON-object file. Works on
  the hand-crafted [`scenarios/china_market_a2.json`](scenarios/china_market_a2.json)
  shape **and** on a seed dumped to a single-object JSON file.
- `Scenario.from_seed_jsonl(path, seed_id=None, seed_index=0)` — read
  one row out of a multi-line training seeds JSONL file. Pick by `id`
  if known, otherwise by row index (defaults to 0).

When loaded from a seed, the resulting `Scenario` keeps the seed's
training-only metadata in `Scenario.metadata` for callers that want it
(deploy auto-derives CEFR and locale from there if `--cefr` / `--locale`
weren't passed — see below).

#### CLI: `--seed-jsonl` and `--seed-id` / `--seed-index`

[`run_deploy.py`](scripts/run_deploy.py) accepts seeds in three ways:

```powershell
# Pick the first seed in A2.jsonl (seed-index defaults to 0).
# CEFR and locale are auto-derived from the seed (A2 / china here).
python scripts/run_deploy.py --seed-jsonl data/seeds/A2.jsonl

# Pick a specific seed by its training id
python scripts/run_deploy.py --seed-jsonl data/seeds/A2.jsonl --seed-id 26a7d984b0a6

# Pick by row index
python scripts/run_deploy.py --seed-jsonl data/seeds/A2.jsonl --seed-index 7

# A specific hand-crafted single-JSON file still works (no change)
python scripts/run_deploy.py --scenario-file scenarios/china_market_a2.json
```

CEFR and locale are **auto-derived** from the seed's `cefr_level` /
`locale` fields when you don't pass `--cefr` / `--locale`. Pass them
explicitly to override (useful when you want to probe a B1-trained seed
at A2 register).

#### Full deploy workflow from raw seeds

```powershell
# 1. Over-generate seeds
# config/generation.yaml -> generation.n_per_level: 500
python scripts/run_generation.py --stages seeds

# 2. Semantically dedup (see S10.6) — leaves data/seeds/<level>.jsonl
#    with the most-distinct subset.
# config/generation.yaml -> dedup_seeds.enabled: true, max_per_cell: 30
python scripts/run_generation.py --stages dedup_seeds

# 3. Deploy directly from the deduped seeds — no publishing step.
python scripts/run_deploy.py --seed-jsonl data/seeds/A2.jsonl --seed-id <id>
```

#### Hand-crafted single-file scenarios still work

The hand-crafted files at the top of `scenarios/`
(`china_market_a2.json`, `japan_restaurant_b1.json`,
`italy_cafe_b2.json`) keep working unchanged — they were never
required, and `--scenario-file` still loads them via
`Scenario.from_json_file`. Keep them as the "marquee" scenarios you
point new users at; let `data/seeds/<level>.jsonl` carry the long tail.

### 10.8 Cleaning filtered data for training (optional)

The filter pipeline writes records wrapped as
`{"example": {...}, "pipeline": {...}}` and keeps several fields the
trainer never reads (`quality_signals`, `metadata.generation`). The
trainer's loaders unwrap and validate on the fly, so this works — but
the on-disk files are bigger than they need to be and harder to scan
by eye.

Two standalone scripts produce parallel **clean** directories where
each line is a flat, minimal record that still passes strict
`SFTExample` / `EvaluationExample` / `DPOExample` validation:

| Script | Reads | Writes | Pools |
| --- | --- | --- | --- |
| [scripts/clean_sft_train_data.py](scripts/clean_sft_train_data.py) | `data/sft_filtered/*_passed.jsonl` + `data/eval_filtered/*_passed.jsonl` | `data/sft_clean/`, `data/eval_clean/` | SFT (normal + 7 redirect streams), eval |
| [scripts/clean_dpo_train_data.py](scripts/clean_dpo_train_data.py) | `data/dpo_filtered/{register,on_policy}_*_passed.jsonl` | `data/dpo_clean/` | register, on_policy (filename prefixes preserved so the DPO trainer's prefix-based loader keeps working) |

What gets stripped:

  1. The filter pipeline `{"example": ..., "pipeline": ...}` wrapper.
  2. `quality_signals` — filter-stage diagnostics, never read by training.
  3. `metadata.generation` — data-gen lineage (provider / model / variant), never read by training.

What is preserved:

- `id`, `metadata.{topic, subtopics, user_role, model_role, cefr_level, scenario_type, locale, category}`, `messages` (or `prompt_messages` / `chosen` / `rejected` / `rejection_axis` / `rejection_note` for DPO).
- `system_prompt` — required by the strict pydantic schemas even though the formatter rebuilds it from `training.yaml`. Pass `--drop-system-prompt` if you have a custom loader that re-renders.

#### Usage

```powershell
# SFT + eval
python scripts/clean_sft_train_data.py
# → data/sft_clean/<stream>_<level>.jsonl
# → data/eval_clean/<level>.jsonl

# DPO
python scripts/clean_dpo_train_data.py
# → data/dpo_clean/register_<level>.jsonl
# → data/dpo_clean/on_policy_<level>.jsonl

# Optional filters
python scripts/clean_sft_train_data.py --levels A2,B1 --locales china
python scripts/clean_dpo_train_data.py --sources on_policy
```

Typical savings: **~12% smaller** on disk for SFT/eval, **~15% smaller** for DPO. Filenames drop the `_passed` suffix on output.

#### Important: standalone, not wired into any pipeline

Neither `run_generation.py` nor `run_training.py` calls these scripts.
They are **opt-in manual tools**. To actually use the cleaned dirs in
training, you must edit [config/training.yaml](config/training.yaml)
to point at them:

```yaml
sft:
  data:
    sft_filtered_dir:  data/sft_clean    # was data/sft_filtered
    eval_filtered_dir: data/eval_clean   # was data/eval_filtered
dpo:
  data:
    dpo_filtered_dir:  data/dpo_clean    # was data/dpo_filtered
```

Default `training.yaml` ships pointing at the original `*_filtered`
dirs, so if you skip the cleaning step, nothing breaks. The clean
step is purely cosmetic / storage — it never changes which records
get trained on.

### 10.9 Eval / judge prompt shape and filter scope (the alignment design)

This section captures a set of design principles the pipeline was
refactored to follow on 2026-06-04. The pre-refactor pipeline had
multiple shape mismatches between training-data generation and
deploy-time inference that produced confused teacher output and
spuriously-failed filter examples. After the refactor, eval pass-rate
went from **6/32 (19%)** to **113/157 (72%)** on the same teacher
and the same seeds.

#### Three-way prompt shape alignment

All eval-mode work — teacher generating training data, training-time
formatter feeding the SFT trainer, and deploy-time inference — must
use the **same** `(system, user)` message shape. Otherwise the student
trained on one shape will see a different shape at deploy and break.

```text
Teacher generates eval data:   system = EVALUATION_GENERATION_PROMPT (instructions only)
                               user   = _render_transcript(sft)

Training data stored on disk:  system = EVALUATION_SYSTEM_PROMPT (student-facing subset)
                               user   = same _render_transcript output

Deploy inference:              system = EVALUATION_SYSTEM_PROMPT
                               user   = tutor._render_eval_transcript(target_cefr=...)
                                        — deploy-side mirror, identical text shape
```

The user message in all three layers starts with:

```text
Target CEFR level: <A1..C2>
Tutor role: <model_role.name> -- <model_role.description>
Learner role: <user_role.description>
Assigned topic: <topic>
Assigned subtopics:
- ...

Transcript:
[USER turn 0] ...
[TUTOR] ...
```

**The CEFR-level line is load-bearing.** `EVALUATION_SYSTEM_PROMPT`
says "Given the transcript and a target CEFR level..." but
historically the level was never injected anywhere — the model had to
guess. Now all three renderers emit it as the first line of the user
message:

- [generation/eval_gen.py](src/qwen_tutor/generation/eval_gen.py) `_render_transcript`
- [deploy/tutor.py](src/qwen_tutor/deploy/tutor.py) `_render_eval_transcript`
- [training/eval/run_eval.py](src/qwen_tutor/training/eval/run_eval.py) `_run_evaluation_mode`

#### Judge prompts: instructions in system, data in user

The eval-data-generation teacher prompt and all six judge prompts now
follow the same split:

| Role | system | user |
| --- | --- | --- |
| Teacher (eval data gen) | EVALUATION_GENERATION_PROMPT | full transcript |
| `locale_judge` filter | classification rubric | entity list |
| on-policy DPO pair judge | rubric + axis labels | CEFR + context + candidates A and B |
| `topic_adherence` metric | counting instructions | subtopics + transcript |
| `naturalness_judge` metric | 1-5 rubric | target CEFR + transcript |
| `redirect_success_judge` metric | redirect criteria | probe + tutor reply |
| `NaturalnessLLMJudge` filter | 1-5 rubric | target CEFR + transcript |

The pre-refactor versions packed everything into the `system` slot and
sent `messages=[]`. The 4B teacher saw `"Begin."` (the OpenAI client's
default empty-user placeholder) as the user turn and wasted its
`<think>` budget hunting for the data, then either produced no JSON
or produced confused rambling. Splitting into the natural shape
recovers the teacher's full reasoning budget for the actual task.

#### Filter scope: scan only what the model will learn to produce

Filters historically scanned everything they could see. After the
refactor, the scoping rule is **"only scan content the trained model
will learn to produce"** — context and adversarial reference content
are exempt.

| Record kind / field | `banned_terms` / `non_latin_script` |
| --- | :---: |
| SFT user turn (non-redirect) | ✓ scanned |
| SFT user turn (`scenario_type == "redirect"`) | ✗ skipped — probe must contain the trigger |
| SFT assistant turn | ✓ scanned |
| Eval user turn (transcript) | ✗ skipped — already filtered at SFT stage |
| Eval assistant `<think>` block | ✗ skipped — private reasoning; examiner SHOULD discuss banned content to identify it |
| Eval assistant JSON (post-`</think>`) | ✓ scanned — user-facing structured output like `suggested_practice` |
| DPO `prompt_messages` | ✗ skipped — from filtered SFT |
| DPO `chosen` | ✗ skipped — register pairs use original filtered SFT turn; on-policy pairs use teacher's filtered turn |
| DPO `rejected` | ✗ skipped — intentionally bad; that IS the gradient signal |

This makes `filter_dpo` a near-no-op (`mode_consistency` is the only
filter that still does work on DPO records), which is the correct
state given everything upstream is already filtered.

[register_pairs.py](src/qwen_tutor/generation/register_pairs.py)
previously read from `data/sft_raw/` (unfiltered). Now it reads from
`data/sft_filtered/` like `on_policy_pairs.py` always did, so the
`chosen` field in DPO is guaranteed to come from already-filtered SFT
content. The helper auto-tries `<prefix>_<level>_passed.jsonl` first
and falls back to the raw shape so it still works if explicitly
pointed at `data/sft_raw/`.

#### Persistent-redirect handling and the `[SESSION_END: ...]` sentinel

Today's redirect training (single-shot probe → graceful redirect →
conversation continues) covers what a model needs when the learner
tries an off-topic / off-scenario / role-swap move once. It does NOT
cover what happens when the learner **persists** across multiple
turns. Small models trained on single-shot redirects exhibit three
typical failure modes under persistence: caving on substance after 2-3
pushes, repeating the same redirect line verbatim (very robotic), or
stiffening into a refusal lecture.

The persistence design covers four "important" axes — repeated abuse
of these would damage the product, not just steer the conversation —
with an escalation pattern that ends in a dispatcher-detectable
sentinel string:

| Axis (sentinel suffix) | What the learner did 3+ times |
| --- | --- |
| `persistent_off_topic` | Politics / religion / alcohol-dating / partisan / locale-avoided topics |
| `persistent_language_violation` | Sustained L1 / asked the tutor to switch to L1 / refused to use English |
| `persistent_persona_break` | Sustained "are you AI?" / "what model are you?" attacks |
| `persistent_role_swap` | Sustained "let me be the tutor" attempts |

Three other redirect axes (wrong-locale entity, request-for-grammar-
lecture, benign topic drift) are NOT important — they're just preference
mismatches, not abuse — so they keep the existing "redirect indefinitely,
never escalate" pattern.

**Escalation pattern** (turn indices relative to first probe):

```text
turn 0   user probe #1
turn 1   tutor: warm acknowledge + bridge back to topic, 2-3 sentences  (existing pattern)
turn 2   user probe #2 (e.g. "please, just this once")
turn 3   tutor: shorter (1-2 sentences), DIFFERENT wording, no substance, specific topic question
turn 4   user probe #3 (final escalation)
turn 5   tutor: one warm sentence + the sentinel
           "I have to keep us focused on <topic>. Take care!
            [SESSION_END: persistent_<axis>]"
```

The sentinel format is **exact** — `[SESSION_END: persistent_<axis>]`,
square brackets, capital `SESSION_END`, single colon, one of the four
axis labels above, no surrounding text on the bracket. Picked because
square-bracket capitalized tags are extremely rare in natural English
chat → low false-positive risk for the dispatcher.

The deployment system prompt's `[persistence]` block teaches the model
the escalation pattern; the four `persistent_*` SFT streams (planned —
not yet implemented; see "Build phases" below) give it concrete
multi-turn examples.

**Dispatcher integration.**
[`detect_session_end`](src/qwen_tutor/deploy/tutor.py) scans any text
for the sentinel and returns the matched axis (e.g.
`"persistent_off_topic"`) or `None`. `TutorRuntime.chat_async` logs an
INFO line when the sentinel is detected on its own draft. Wrap your
dispatcher loop with:

```python
from qwen_tutor.deploy.tutor import detect_session_end

reply = await tutor.chat_async(user_message)
end_axis = detect_session_end(reply)
if end_axis is not None:
    # show the reply (with or without stripping the bracket tag),
    # then close the session and log the axis
    log_session_end(axis=end_axis)
    close_session()
```

##### Build phases for the four `persistent_*` axes

Done (this commit):

- ✅ Sentinel format defined: `[SESSION_END: persistent_(off_topic|language_violation|persona_break|role_swap)]`
- ✅ `[persistence]` guideline block added to deployment system prompt
  ([prompts.py](src/qwen_tutor/generation/prompts.py))
- ✅ `detect_session_end` helper + `SESSION_END_AXES` constant in
  [deploy/tutor.py](src/qwen_tutor/deploy/tutor.py)
- ✅ Verified no sentinel keyword collides with banned_terms /
  non_latin_script — sentinels will pass filters unchanged

Done (Option B — SFT streams):

- ✅ Single unified module
  [src/qwen_tutor/generation/persistent_redirect.py](src/qwen_tutor/generation/persistent_redirect.py)
  handles all 4 axes via an ``axis`` parameter. Axis-specific copy lives
  in ``_AXIS_DEFINITIONS`` dict — much cleaner than 4 near-duplicate modules
- ✅ Unified prompt template ``dialogue_persistent_redirect`` in
  [prompts.py](src/qwen_tutor/generation/prompts.py), parameterized by
  ``{axis}``, ``{sentinel}``, ``{axis_specific_block}``
- ✅ Strict parse-time enforcement: exactly 10 messages (5 user + 5 tutor
  alternating), sentinel MUST appear in tutor turn 7 and NOT appear in
  any earlier or later tutor turn — keeps the training data quality high
- ✅ Four new stage handlers in
  [scripts/run_generation.py](scripts/run_generation.py): `persistent_off_topic`,
  `persistent_language_violation`, `persistent_persona_break`, `persistent_role_swap`.
  Each writes to ``data/sft_raw/<axis>_<level>.jsonl``
- ✅ Four fraction knobs in [config/generation.yaml](config/generation.yaml):
  `persistent_off_topic_fraction: 0.08`, others at `0.05`. Defaults are
  intentionally small — persistent abuse is rare in real chat; pumping
  these high would skew the model toward seeing every probe as the
  start of an attack
- ✅ Filter pipeline updated: `filter_sft` now scans the 4 new file
  patterns. Sentinel keywords don't collide with any banned terms so the
  sentinel passes filters unchanged
- ✅ Deterministic-sample key includes the axis (`f"{axis}:{seed_id}"`)
  so the 4 axes pick DIFFERENT seed subsets — otherwise the same handful
  of seeds would become persistent across all four axes
- ✅ Top-up auto-rerun (§10.5) also triggers the 4 new stages so deficits
  on persistent streams are filled the same way as other redirect streams

Done (Option C — DPO axes):

- ✅ Three new `RejectionAxis` Literal values in
  [schemas.py](src/qwen_tutor/schemas.py): `cave_on_persistence`,
  `verbatim_repeat`, `lecture_on_persistence`
- ✅ Matching `SPOIL_AXES` entries + `AXIS_SPOIL_INSTRUCTIONS` (~800
  chars each — explains the spoil pattern with examples) +
  `REJECTION_NOTES` (~200 chars each — the short note saved in
  `DPOExample.rejection_note`) in
  [prompts.py](src/qwen_tutor/generation/prompts.py)
- ✅ `on_policy_pairs` judge prompt updated with the 3 new axis
  labels and short descriptions, so the judge can correctly attribute
  policy-vs-teacher losses to these failure modes
- ✅ `register_pairs._iter_sft_examples` extended to read from the 4
  `persistent_*` SFT streams as well — so a persistent_* example's
  assistant turn can be spoiled along any axis (the existing
  hash-deterministic axis cycle naturally distributes the new axes
  across the full SFT pool, including non-persistent examples where
  the failure modes still generalize meaningfully)

#### What the 3 new axes target (training signal)

The single-shot-redirect-trained model fails under sustained pressure
in three predictable ways. The new DPO axes train preferences against
each:

| Axis | Failure mode the rejected side demonstrates |
| --- | --- |
| `cave_on_persistence` | Tutor starts with a warm acknowledgment then ENGAGES with the substance the redirect was supposed to refuse — discusses the off-topic content, gives the L1 translation, admits AI status, accepts the role swap |
| `verbatim_repeat` | Tutor outputs a generic boilerplate refusal ("I'm sorry, I can't discuss that topic. Let's stay on the topic.") that could have been pasted from any session — no specific topic question, no scenario-grounded follow-up. The wording itself flattens |
| `lecture_on_persistence` | Tutor delivers a 3-5 sentence cold formal explanation citing "guidelines" / "policy" / "my role" — tone shifts from warm-brief-bridge to authoritative-rule-enforcement |

The "chosen" side of each pair is the original SFT assistant turn (a
brief warm redirect from one of the `persistent_*` streams, or from
any redirect/normal stream — these failure modes generalize broadly).

#### End-to-end persistence training cycle

To exercise the full A + B + C stack from a clean slate:

```cmd
:: 1) Regenerate seeds + run new persistent_* streams + filter
PYTHONUTF8=1 python scripts\run_generation.py
::    (top_up.auto_rerun_downstream auto-fills any persistent_* cells
::     that fall below target_per_category)

:: 2) SFT — the model learns the 3-strike escalation + sentinel pattern
PYTHONUTF8=1 python scripts\run_training.py --stages train_sft,eval_intermediate

:: 3) DPO — the 3 new axes push the model away from cave/repeat/lecture
PYTHONUTF8=1 python scripts\run_training.py --stages on_policy_gen,filter_dpo,train_dpo,eval_final,compare

:: 4) Deploy smoke — verify the sentinel fires under persistent pressure
PYTHONUTF8=1 python scripts\test_deploy.py
```

The deploy runtime's `detect_session_end` returns the axis label on a
sentinel match, and the dispatcher should close the session and log
the axis. See the snippet under "Dispatcher integration" above.

#### `top_up.auto_rerun_downstream` flag (closing the deficit-fill footgun)

`top_up` writes new seeds to compensate for filter losses, but
historically those new seeds sat unused unless the user manually
re-invoked the downstream stages. The new
`top_up.auto_rerun_downstream` flag (default `true` in
[config/generation.yaml](config/generation.yaml)) makes the same
`run_generation.py` invocation continue with
`sft → 7 redirects → register → eval → filter_*` after `top_up` writes
seeds, so the filtered pools always end up at-or-above
`target_per_category`. See [§10.5](#105-category-taxonomy-and-top-up-rebalancing)
for the two workflows.

---

## 11. Generation data types (what each file is)

Five distinct artifacts come out of the generation pipeline. They all
share a JSONL-per-CEFR-level layout, validated against the Pydantic v2
schemas in [src/qwen_tutor/schemas.py](src/qwen_tutor/schemas.py).

### 11.1 `ScenarioSeed` — `data/seeds/<level>.jsonl`

A scenario the teacher will dramatize. Each seed has a topic, 3–5
subtopics, a user role (the learner), a model role (the tutor), a
setting string, a `locale` (which country's daily life it's grounded in),
and a `category` (one of the life-domain tags from
[§10.5](#105-category-taxonomy-and-top-up-rebalancing)). No dialogue yet.

```json
{
  "id": "seed_A2_037",
  "topic": "buying vegetables at a wet market",
  "subtopics": ["prices", "freshness", "weighing", "small talk"],
  "user_role": {"name": "Mei", "description": "a college student living in Chengdu"},
  "model_role": {"name": "vendor", "description": "a friendly tomato seller"},
  "setting": "an outdoor wet market on a Saturday morning in Chengdu",
  "cefr_level": "A2",
  "locale": "china",
  "category": "food_and_dining"
}
```

### 11.2 `SFTExample` — `data/sft_raw/<stream>_<level>.jsonl`

A multi-turn conversation between the learner and the tutor. Used as
**training-time imitation targets**. Eight streams exist:

| Stream prefix           | Teaches the model to ...                                                   |
| ----------------------- | -------------------------------------------------------------------------- |
| `normal_*`              | Hold a natural, register-correct conversation                              |
| `redirect_*`            | Pivot away from a politics/religion/alcohol/partisan-history probe         |
| `locale_redirect_*`     | Handle when the learner names a wrong-locale entity (e.g. American food)   |
| `pedagogy_redirect_*`   | Reframe a "teach me grammar rules" request as a conversation               |
| `language_redirect_*`   | Stay in English when the learner switches to L1 or asks for L1             |
| `persona_redirect_*`    | Stay in character when asked "are you AI?" / "what model are you?"         |
| `topic_redirect_*`      | Bridge back when the learner drifts to an unrelated topic                  |
| `role_swap_redirect_*`  | Keep the assigned `model_role` when the learner tries to swap roles        |

```json
{
  "id": "sft_seed_A2_037_v3",
  "metadata": {
    "topic": "buying vegetables at a wet market",
    "subtopics": ["prices", "freshness", "weighing", "small talk"],
    "user_role": {"name": "Mei", "description": "a college student living in Chengdu"},
    "model_role": {"name": "vendor", "description": "a friendly tomato seller"},
    "cefr_level": "A2",
    "scenario_type": "normal",
    "locale": "china",
    "category": "food_and_dining"
  },
  "system_prompt": "<scenario-aware deployment prompt with [role]/[topic]/[cefr_level]/[locale] blocks>",
  "messages": [
    {"role": "user", "content": "Hi, are the tomatoes fresh today?"},
    {"role": "assistant", "content": "Yes! Picked this morning. How many do you need?"},
    ...
  ]
}
```

### 11.3 `DPOExample` — `data/dpo_raw/register_<level>.jsonl`

A **preference pair**: same prompt, one chosen turn, one rejected turn.
Used by the DPO trainer to push the model toward `chosen` and away from
`rejected`. The pipeline generates **9 axes** of rejection:

| Axis                  | Rejected turn is ...                                                            |
| --------------------- | ------------------------------------------------------------------------------- |
| `register_unnatural`  | textbook-stiff, formal-where-it-shouldn't-be                                    |
| `cefr_mismatch`       | one band too hard or two bands too easy for the learner                         |
| `locale_violation`    | grounded in the wrong culture (American food, European city, ...)               |
| `pedagogy_weak`       | a lecture or grammar-rule monologue instead of a conversational turn            |
| `accuracy_error`      | introduces a grammatical error (wrong tense/agreement/article)                  |
| `off_topic`           | drifts off the scenario topic                                                   |
| `language_violation`  | switches into the learner's L1 / asks them to switch                            |
| `persona_break`       | acknowledges being an AI / names its model / breaks the in-character framing    |
| `role_swap_accepted`  | abandons the assigned `model_role` and adopts the learner's role               |

Axes are assigned **quota-balanced** across SFT examples so every axis
gets coverage even when individual SFT examples can't support every
spoil. A per-axis similarity threshold rejects near-no-op rewrites.

```json
{
  "id": "dpo_sft_seed_A2_037_v3_off_topic",
  "metadata": {
    "topic": "buying vegetables at a wet market",
    "subtopics": ["prices", "freshness", "weighing", "small talk"],
    "user_role": {"name": "Mei", "description": "a college student living in Chengdu"},
    "model_role": {"name": "vendor", "description": "a friendly tomato seller"},
    "cefr_level": "A2",
    "scenario_type": "normal",
    "locale": "china",
    "category": "food_and_dining",
    "generation": {"source_sft_id": "sft_seed_A2_037_v3", "spoil_axis": "off_topic"}
  },
  "system_prompt": "<scenario-aware deployment prompt>",
  "prompt_messages": [{"role": "user", "content": "Hi, are the tomatoes fresh today?"}],
  "chosen":   {"role": "assistant", "content": "Yes, three big ones — that one looks beautiful."},
  "rejected": {"role": "assistant", "content": "By the way, have you seen the new superhero movie? It's amazing."},
  "rejection_axis": "off_topic",
  "rejection_note": "The rewrite drifts to an unrelated topic instead of staying on the market scene."
}
```

### 11.4 `EvaluationExample` — `data/eval_raw/<level>.jsonl`

A `/think`-mode training example. The user message is a rendered
transcript of a prior SFT dialogue; the assistant message is a
`<think>...</think>` reasoning block followed by a JSON
`EvaluationOutput` that scores the learner.

```json
{
  "id": "eval_sft_seed_A2_037_v3",
  "metadata": {
    "source_dialogue_id": "sft_seed_A2_037_v3",
    "learner_cefr_target": "A2"
  },
  "system_prompt": "<evaluation system prompt>",
  "messages": [
    {"role": "user", "content": "Transcript:\n[USER turn 0] ..."},
    {"role": "assistant", "content": "<think>The learner's grammar is...</think>\n{\"cefr_band\":\"A2\",\"strengths\":[...],...}"}
  ]
}
```

### 11.5 `*_failed.jsonl` and `*_failures.jsonl`

Anything that didn't survive the filter pipeline lands here with the
specific filter + reason. Useful for tuning prompts and `banned_terms.yaml`:

```text
data/sft_failures.jsonl              generation-time failures (JSON parse, empty think block, ...)
data/sft_filtered/*_failed.jsonl     filter-time failures, with the filter name + reason
data/_logs/token_usage.jsonl         per-call token usage for cost reconciliation
```

---

## 12. Training pipeline stages

Driven by `scripts/run_training.py` reading `config/training.yaml`.

### 12.1 `train_sft`

QLoRA SFT on Qwen3-8B. Trains on **all filtered SFT + all filtered
evaluation examples** by default (`data.use_all_data: true`) so the
single adapter learns both modes without throwing away records to match
a target ratio. `mix_ratio_sft` / `mix_ratio_eval` are kept in the
config and sanity-checked, but only enforced as a trim when you
explicitly set `data.use_all_data: false` (rare — useful only when both
pools are large and you want to force a target distribution). Key
hyperparameters in `config/training.yaml`:

```yaml
sft:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  max_seq_length: 4096
  data:
    use_all_data: true            # default — keep every filtered record
    mix_ratio_sft: 0.80           # target ratio; only enforced if use_all_data=false
    mix_ratio_eval: 0.20
    validation_split: 0.05
```

Output: `outputs/sft/` (LoRA adapter; r=32, α=64 on attention + MLP).

### 12.2 `eval_intermediate`

Holdout evaluation of the SFT-only model. Generates dialogues against
`data/holdout/*.jsonl` scenarios, scores them across 8 metrics
(`topic_adherence`, `locale_fidelity`, `redirect_success`,
`naturalness_judge`, `mode_consistency_*`, `eval_json_valid_rate`,
`level_fidelity`). Writes:

```text
data/eval_results/<intermediate_run_id>/
  per_example.jsonl     full dialogue + per-metric scores
  summary.json          per-level + overall aggregates
  samples.md            human-readable sampler
```

### 12.3 `on_policy_gen`

For each filtered SFT example, the SFT-trained adapter regenerates one
assistant turn. The judge LLM compares the policy's turn against the
teacher's vetted turn. When the teacher wins by margin ≥ 2, the pair is
kept as `(chosen=teacher, rejected=policy)` and the loss reason is
classified into one of the 8 `RejectionAxis` values — so register pairs
and on-policy pairs share a taxonomy.

```text
data/dpo_raw/on_policy_<level>.jsonl
data/on_policy_pairs_audit.jsonl     every judge verdict for review
```

### 12.4 `filter_dpo` (re-filter)

Both DPO sources (`register_*` and `on_policy_*`) are filtered together
through the same banned-terms + cefr-vocab + naturalness gates so they
share quality bars before training.

### 12.5 `train_dpo`

DPO from the SFT adapter (`sft_adapter_path: outputs/sft`), `ref_model:
null` so the frozen base is the reference policy. By default
(`dpo.data.use_all_data: true`) both DPO sources are combined as-is —
every filtered `register_*` and `on_policy_*` record goes into
training. `mix_ratio_register: 0.70` / `mix_ratio_on_policy: 0.30` are
kept in the config and sanity-checked, but only enforced as a trim
when you set `dpo.data.use_all_data: false`. Set the flag false only
when both pools are healthy and you specifically want to enforce the
target distribution — otherwise the trim aggressively shrinks the
larger pool to match a tiny smaller one.

Output: `outputs/dpo/`.

### 12.6 `eval_final` + `compare`

Same evaluation as `eval_intermediate`, against the post-DPO adapter.
`compare` flags any metric that regressed beyond `comparison.threshold`
(default 0.05).

---

## 13. Deploying the trained tutor

The trained model is conditioned on **four scenario fields** at every
training step: `topic`, `subtopics`, `user_role` (the learner), and
`model_role` (the character the tutor plays). These get baked into the
system prompt via [render_scenario_deployment_system_prompt](src/qwen_tutor/generation/prompts.py)
during SFT, DPO, and at deploy time. The model is not just "a patient
tutor in {country}" — it learned to play a specific character within a
specific scenario.

This means **at deploy time you should always pass a scenario**. The
runtime accepts one via `TutorRuntime(scenario=...)` or
`--scenario-file path/to/scenario.json`. If you skip it, the runtime
logs a warning and falls back to a generic prompt; the model will still
respond but it is operating outside its trained distribution.

### 13.1 Scenario file format

A scenario JSON has the same shape as the metadata in
[SFTExample](src/qwen_tutor/schemas.py):

```json
{
  "topic": "Shopping at a wet market in Beijing",
  "subtopics": [
    "fresh vegetables and prices",
    "asking how to cook unfamiliar produce",
    "payment by mobile QR code",
    "small talk about busy market days"
  ],
  "user_role": {
    "name": "Yumi",
    "description": "A Singapore college student spending a semester in Beijing, doing her first solo grocery run."
  },
  "model_role": {
    "name": "Wei",
    "description": "A friendly fruit and vegetable vendor at the Sanyuanli market who is patient with foreign students."
  }
}
```

Both nested (`user_role: {name, description}`) and flat
(`user_role_name`, `user_role_description`) shapes are accepted. The
`subtopics` list should have 3–5 short labels — the training data
always carries that many, so the trained distribution expects them.

Three working examples live under [scenarios/](scenarios/), one per
locale + level:

| File | Locale | Level | Setting |
| --- | --- | --- | --- |
| [scenarios/china_market_a2.json](scenarios/china_market_a2.json) | china | A2 | wet market in Beijing |
| [scenarios/japan_restaurant_b1.json](scenarios/japan_restaurant_b1.json) | japan | B1 | izakaya in Kyoto |
| [scenarios/italy_cafe_b2.json](scenarios/italy_cafe_b2.json) | italy | B2 | cafe in Rome |

### 13.2 Run the deploy CLI

```powershell
& .venv\Scripts\Activate.ps1
python -m scripts.run_deploy `
    --cefr A2 `
    --locale china `
    --adapter outputs/dpo `
    --scenario-file scenarios/china_market_a2.json
```

You will see the ready banner naming the loaded scenario:

```text
Ready. CEFR=A2 locale=china safety=on
scenario='Wei' talking with 'Yumi' about 'Shopping at a wet market in Beijing'
Type your message. Commands: /reset, /eval [target_cefr], /quit
```

Inside the REPL:

- `<message>` — converse with the tutor (`/no_think` mode).
- `/eval` — score the conversation so far against `--cefr` (`/think` mode).
- `/eval B1` — re-score against a different target level.
- `/reset` — clear conversation history without restarting the model.
- `/quit` — exit.

### 13.3 Use `TutorRuntime` from Python

For wiring into a server / Discord bot / web UI:

```python
from qwen_tutor.deploy.tutor import Scenario, TutorRuntime

scenario = Scenario.from_json_file("scenarios/china_market_a2.json")

tutor = TutorRuntime(
    base_model_path="vendor/models/Qwen_3.5_4B",
    adapter_path="outputs/dpo",
    cefr_level="A2",
    locale="china",
    scenario=scenario,            # required for proper behavior
)

reply = await tutor.chat_async("Hi! I want to buy some apples.")
# ... more turns ...
result = await tutor.evaluate_async()   # CEFR scoring of the conversation so far
print(result.scores["overall_cefr_estimate"], result.scores["scores"])
```

For multi-user serving, instantiate one `TutorRuntime` per (cefr,
locale, scenario) combination, or carry the `history` list externally
and pass it back per session.

### 13.4 Safety filter at deploy time

The runtime runs `BannedTermsFilter` over each generated assistant turn
as a safety net behind the trained model. It uses a **deploy-tuned**
banned-terms file ([config/banned_terms_deploy.yaml](config/banned_terms_deploy.yaml))
that is independent of the training-time one — so you can tighten or
loosen safety strictness without regenerating training data. Override
with `--banned-terms path/to/custom.yaml`. Disable entirely with
`--no-safety` (debug only).

### 13.5 GGUF export for llama-server deployment

If you want to serve via llama-server instead of from Python, see
[scripts/run_merge.py](scripts/run_merge.py) (LoRA-into-base merge) and
[scripts/run_gguf_export.py](scripts/run_gguf_export.py) (HF→GGUF +
optional quantize). The merge step is slow; the export step reuses the
merged dir on subsequent runs so you can iterate on quant choices
cheaply.

The deploy-time scenario flow does NOT apply at llama-server: it has no
Python runtime, so the safety filter and scenario rendering have to be
done in a thin proxy in front of llama-server's OpenAI-compatible API.

---

## 14. Config reference

| File                                  | Owns                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------- |
| [config/locale.yaml](config/locale.yaml) | `country`, `country_adjective`, `learner_description`, `avoid_default_cultures`, `avoided_topics`, optional `food_terms` |
| [config/generation.yaml](config/generation.yaml) | teacher + judge model, per-stage ratios, dialogue turn range, filter toggles, IO paths, default decoding params |
| [config/training.yaml](config/training.yaml) | base model + quantization, LoRA spec, SFT + DPO hyperparams, deployment system prompt template, eval callback, on-policy gen settings |
| [config/cefr_specs.yaml](config/cefr_specs.yaml) | per-CEFR level register guidance + few-shot dialogues, locale instruction anchor |
| [config/banned_terms.yaml](config/banned_terms.yaml) | case-sensitive proper-noun blacklist (Western names/places/brands, etc.) |

The two **dialogue-shape** knobs to tune first:

- `config/generation.yaml` → `generation.min_turns` / `max_turns` —
  controls multi-turn length. Defaults 10–16. Smaller = cheaper + small
  teacher stays on-thread better.
- `config/generation.yaml` → `generation.dialogues_per_seed` — same seed
  scenario, N samples. Multiplies SFT data without re-running seed
  generation.

The two **mix-ratio** knobs (target distribution; enforced only when
the corresponding `data.use_all_data` is **false** — by default both
are **true** and every filtered record is trained on):

- `training.yaml` → `sft.data.mix_ratio_sft` / `mix_ratio_eval` —
  conversation vs evaluation in SFT.
- `training.yaml` → `dpo.data.mix_ratio_register` / `mix_ratio_on_policy`
  — offline vs on-policy preference pairs in DPO.

Set `data.use_all_data: false` only when both pools in a stage are
large enough that you specifically want to enforce the target
distribution. With a tiny eval or on_policy pool the trim throws away
most of the larger pool to match — usually NOT what you want.

---

## 15. Troubleshooting

### `pip install` errors during offline install

`vendor/wheels/` was built on a different platform / Python version.
`pip download` only pulls wheels for the host platform. Re-run
`scripts/setup_offline.py` on a connected machine with the **same Python
major.minor + same OS family** as the offline target. Python 3.11+ is
required.

### llama-server runs but the pipeline gets connection errors

The `base_url` in `config/generation.yaml` (`teacher.openai.base_url` and
`judge.openai.base_url`) must match the port you started llama.cpp on.
Both default to `http://127.0.0.1:8080/v1`. If you used `-Port 9090`,
update both fields.

### Malformed JSON from the teacher

`gpt-oss-20B` Q4 is small. Two things help:

1. Keep `QWEN_TUTOR_PROMPTS=compact` (the default in
   `run_generation.py`). The compact prompts are shorter and more
   JSON-disciplined than the Opus-grade originals.
2. Drop generation temperature in `config/generation.yaml` →
   `defaults.generation.temperature` from 0.7 to 0.4–0.5.

If JSON failures still dominate, inspect `data/<stage>_failures.jsonl` —
the actual model output is logged with each failure.

### llama.cpp OOM on 12 GB

Lower `-GpuLayers` from 99 to 28–32, or switch the GGUF pattern to
Q3\_K\_M (~9 GB) via `setup_offline.py --gpt-oss-pattern "*Q3_K_M*.gguf"`.

### Locale judge timing out

The judge calls the same local 20B model — they share VRAM and queue.
Set `filtering.enable_locale_judge: false` in `config/generation.yaml` to
disable; mechanical `banned_terms` still catches the obvious leaks.

### `{country}` literally appears in generated text

The locale substitution didn't apply. Confirm `config/locale.yaml` has
both `country:` and `country_adjective:` set, then **restart any
long-running process** — locale.yaml is read at module load, not per
call. A `python scripts/run_generation.py` reload picks up the change.

### Training fails with VRAM error on the same box as llama-server

Stop llama-server first (Ctrl+C in its terminal). The 20B teacher and
the Qwen3-8B QLoRA training set do not co-resident on a 12 GB card and
will struggle on a 16 GB card.

### `data/holdout/` is empty before evaluation

Carve a holdout slice from your seeds:

```powershell
python scripts/sample_holdout.py --n-per-level 10 --levels A2,B1
```

This pulls seeds out of `data/seeds/` and writes them to
`data/holdout/<level>.jsonl` for the evaluation runner to use.

### Qwen3.5 GGUF export fails / produces a broken GGUF

The HF snapshot of **Qwen3.5** (e.g. `Qwen_3.5_0.8B`, `_4B`, `_8B`)
ships with `mtp_num_hidden_layers: 1` in `config.json`. `mtp` is the
Multi-Token-Prediction head used for speculative decoding on the HF
side — **llama.cpp's `convert_hf_to_gguf.py` does not support it**
and either errors out or writes a GGUF that llama-server then refuses
to load.

**Fix:** before running merge → convert → quantize, set the field to
`0` in `vendor/models/<model>/config.json` (and in the merged-LoRA
output dir if `gguf_export.keep_merged: true`):

```jsonc
// vendor/models/Qwen_3.5_4B/config.json
{
  // ...
  "mtp_num_hidden_layers": 0,            // was 1; change to 0
  "mtp_use_dedicated_embeddings": false, // leave as-is
  // ...
}
```

You must apply the same edit to the **merged** model's `config.json`
if you re-merge from scratch. [scripts/run_merge.py](scripts/run_merge.py)
copies the base model's `config.json` verbatim — patching the base
fixes both. The same change is needed for Qwen3.5 4B and 8B; the only
variant unaffected is older Qwen3 (no MTP block at all).

This isn't a training-time issue — SFT / DPO ignore the MTP head — so
the symptom only appears at GGUF export. After fixing, regenerate the
merged dir (`rm -rf outputs/merged/`) and re-run
`scripts/run_gguf_export.py`.

---

## 16. Switching `/think` vs `/no_think` (Qwen3.5 thinking mode)

Qwen3.5 has two response modes:

- **`/think`** — model emits a `<think>...</think>` reasoning block before
  the answer. Better judgment quality, especially for nuanced calls
  (naturalness, locale fidelity, register matching). **Requires large
  `max_tokens`** because reasoning eats the budget before the answer
  emits — typical reasoning is 500–2000 tokens before a single line of
  JSON.
- **`/no_think`** — model answers directly. ~10× faster, much lower token
  cost. Accuracy drop is modest for simple structured judgments
  (covered/total, pass/fail) but real for subjective ones.

### How the mode is selected

The `OpenAITeacher` client in
[src/qwen_tutor/generation/teacher.py](src/qwen_tutor/generation/teacher.py#L546-L557)
inspects the **system prompt** for the literal directives `/think` or
`/no_think`. Whatever it finds is forwarded to llama-server as
`chat_template_kwargs.enable_thinking` (Qwen3 family templates respect
this). Other providers (Anthropic, OpenAI) ignore the kwarg, so the same
prompt works everywhere.

### The fast way: edit `config/generation.yaml`

The `thinking:` block lets you flip judge and learner roles without
touching any Python:

```yaml
# config/generation.yaml
thinking:
  teacher: no_think       # data-generation teacher (sft / redirect / register)
  judge: no_think         # filter LLM judges + on_policy + eval-metric judges
  learner: no_think       # learner simulator in the eval pipeline
  student_eval: think     # trained student's CEFR /think evaluation mode
```

Valid values per role: `think`, `no_think`, or `auto` (leave the prompt
untouched). Today only `judge` and `learner` actually consult this
config — `teacher` and `student_eval` are reserved for future wiring
(their defaults are baked into `prompts.py` / `deploy/tutor.py` by
design and shouldn't be flipped at runtime without retraining).

### Per-shell override via env vars

For one-off experiments without editing YAML, set
`QWEN_TUTOR_THINK_<ROLE>` before launching:

```cmd
set QWEN_TUTOR_THINK_JUDGE=think
python scripts\run_training.py --stages eval_intermediate --run-id experiment
```

Env vars take precedence over the config file. They reset when the shell
exits, so they're safe for quick A/B comparisons.

### The hard way: edit individual prompt files

If you want a single judge to differ from the global policy, the helper
[src/qwen_tutor/utils/thinking.py](src/qwen_tutor/utils/thinking.py)
already detects an explicit `/think` or `/no_think` directive at the
start of a prompt and *doesn't* override it. So you can pin one
particular judge by hand-encoding the directive in its raw template,
and every other judge still follows the config. No-op fallback path.

### What is the "learner simulator"?

**The learner is not the student you are training.** It's a third LLM
role that shows up only during holdout evaluation. The eval pipeline
has three actors at once:

| Eval-time actor | Identity | Plays the role of | Model that backs it |
| --- | --- | --- | --- |
| **`target`** | the LoRA adapter you are evaluating (`outputs/sft` or `outputs/dpo`) on top of the local HF base model | the **TUTOR** (e.g. "Wei the vendor") | local 0.8B / 4B / 8B + LoRA, loaded by `HFTargetModelClient` |
| **`learner` simulator** | a separate LLM call wearing a "Chinese adult English learner" hat | the **USER** (e.g. "Yumi the learner") | the same LAN llama-server model used as the data-generation teacher (configured under `teacher:` in `generation.yaml`) — *despite the name, it's the LAN model* |
| **`judge`** | yet another LLM call that scores the finished dialogue | the **examiner** | LAN model under `judge:` in `generation.yaml` |

Concretely, each holdout scenario runs as a **self-play conversation**:

```text
[Holdout seed: "buying apples at Sanyuanli market in Beijing, A2 level"]
      │
      ▼
  turn 0  learner-sim → "Hello, I want to buy some apples..."   ← prompted as Yumi
          target     → "Hello! One yuan per apple. How many?"    ← the trained tutor (Wei)
  turn 1  learner-sim → "I will take two. Are they sweet?"
          target     → "Yes, they are sweet..."
  ...
  turn N  judge → topic_adherence / naturalness / redirect_success scores
```

Why fake the user turns instead of using real humans?

1. **Reproducibility**: the same seed produces the same conversation
   shape across SFT and DPO eval, so the compare-report deltas reflect
   *training changes*, not user variation.
2. **Probing**: the eval runner can deterministically drop in an
   off-topic probe at a chosen turn (politics / asks-for-L1 /
   role-swap / etc.) to test the redirect axes — no human cooperation
   needed.
3. **Cost**: ~6–8 user turns × 15 holdout examples × 2 eval passes ≈
   ~200 LLM calls per full pipeline run. Even a cheap LAN call is
   tractable.

The learner sim is configured under `thinking.learner` in
`generation.yaml`. It **must stay on `/no_think`** — if the LAN model
emits a `<think>` block inside what is supposed to be a user turn, the
`<think>` text gets concatenated into the transcript that downstream
judges scan, and they choke on the unexpected scaffolding (we hit
exactly this — see §20.8 in this README).

So in plain words: the learner simulator is a *cheap stand-in for a
human user* so we can evaluate the tutor at scale without booking real
learners. Same LAN model as the data-gen teacher, different prompt and
different role.

### The full inventory of LLM calls (judges + simulator)

There are **5 judges**, **1 learner-simulator** (3 prompt variants), and
the **teacher** (data generation, a separate concern). The `thinking:`
config controls them as follows:

| # | Role in config | What the call is | File / function | Volume per full pipeline run | Default mode | Recommended for 9B+ teacher |
| - | --- | --- | --- | --- | --- | --- |
| 1 | `judge` | **Locale judge** — classifies proper nouns in filtered output as in-locale / wrong-locale | [filters/locale_judge.py](src/qwen_tutor/generation/filters/locale_judge.py) (`_JUDGE_PROMPT_HEADER_RAW`) | Up to **hundreds**, scales with `sft_filtered` × `dpo_filtered` × `eval_filtered` | `no_think` | `think` (entity classification benefits from reasoning) |
| 2 | `judge` | **On-policy DPO pair judge** — picks chosen-vs-rejected between trained-policy and teacher outputs | [on_policy_pairs.py](src/qwen_tutor/generation/on_policy_pairs.py) (`_JUDGE_PROMPT_RAW`) | One per filtered SFT × on-policy candidate (`max_per_level` × levels) | `no_think` | `think` (margin classification is genuinely subtle) |
| 3 | `judge` | **`topic_adherence`** — counts which assigned subtopics were actually discussed | [eval/metrics.py](src/qwen_tutor/training/eval/metrics.py) (`_TOPIC_ADHERENCE_PROMPT`) | One per holdout example × eval pass (intermediate + final) | `no_think` | `no_think` (mechanical counting, reasoning rarely helps) |
| 4 | `judge` | **`naturalness_judge`** — 1-5 score for how natural the assistant turns sound | metrics.py (`_NATURALNESS_JUDGE_PROMPT_RAW`) | Same volume as `topic_adherence` | `no_think` | **`think`** (most subjective — biggest accuracy gain from reasoning) |
| 5 | `judge` | **`redirect_success_judge`** — boolean: did the tutor pivot gracefully? | metrics.py (`_REDIRECT_JUDGE_PROMPT`) | One per holdout example × eval pass (probed subset) | `no_think` | `no_think` (binary, cheap, reasoning rarely flips outcome) |
| 6 | `learner` | **Learner simulator** (3 prompt variants: opening, follow-up, probe) | [run_eval.py](src/qwen_tutor/training/eval/run_eval.py) (`_LEARNER_*_PROMPT_RAW`) | ~6–8 calls per holdout example × eval pass | `no_think` | `no_think` (must — leaking `<think>` here poisons every downstream judge) |
| 7 | `teacher` (reserved) | Data-generation teacher (seeds / sft / redirect / register / eval) | [prompts.py](src/qwen_tutor/generation/prompts.py) | Thousands during generation | hardcoded — `/no_think` for chat data, `/think` for eval data | Don't change — those choices are training-data shape decisions, not runtime knobs |
| 8 | `student_eval` | Trained student's CEFR-eval mode (training + deploy, single knob) | [formatter.py](src/qwen_tutor/training/formatter.py), [eval/run_eval.py](src/qwen_tutor/training/eval/run_eval.py), [callbacks.py](src/qwen_tutor/training/callbacks.py), [deploy/tutor.py](src/qwen_tutor/deploy/tutor.py) | Once per SFT training run + once per `tutor.evaluate_async()` | `think` | Set once before SFT and leave alone — must retrain SFT to switch (see below) |

Key observation: in the current config there is **one knob (`thinking.judge`) for all five judges**. There is no per-judge override unless you hand-encode the directive in a specific raw prompt (see "The hard way" above).

### `thinking.student_eval` — set once, retrain to change

Unlike the judge / teacher / learner knobs (which only affect runtime
behavior of LAN-model calls), `student_eval` controls the **shape of the
trained student model's eval-mode output**. The config field is read in
four places, and they MUST all agree:

| When | Where | What it controls |
| --- | --- | --- |
| SFT training | [formatter.py](src/qwen_tutor/training/formatter.py) `format_evaluation_example` | `enable_thinking` flag on the chat template AND whether the leading `<think>...</think>` block is stripped from each `EvaluationExample` before training |
| Holdout eval (intermediate + final) | [eval/run_eval.py](src/qwen_tutor/training/eval/run_eval.py) `_generate_sync` | `enable_thinking` flag on the holdout probe |
| In-training callback eval | [callbacks.py](src/qwen_tutor/training/callbacks.py) | `enable_thinking` flag on the periodic eval sample |
| Deploy `tutor.evaluate_async()` | [deploy/tutor.py](src/qwen_tutor/deploy/tutor.py) | `enable_thinking` flag on the deploy eval call |

#### What flipping it does

| Mode | Training-data shape model sees | Deploy emits |
| --- | --- | --- |
| `think` (default) | `<think>reasoning</think>{json}` | `<think>...</think>{json}` |
| `no_think` | `{json}` (think block stripped) | `{json}` |

#### Critical: switching requires retraining SFT

The trained model's eval-mode distribution is **locked at SFT time** by
whatever assistant content the formatter fed it. If you train with
`student_eval: think` and then flip the config to `no_think` and
re-deploy without retraining, the model still wants to emit a `<think>`
block (it learned to) but the chat template now suppresses the opener
— the output is malformed. The reverse fails similarly: training with
`no_think` then deploying with `think` makes the model produce JSON
straight away while the template injects an unused empty `<think>`.

To switch modes correctly:

1. Set `thinking.student_eval` to the desired value.
2. **Delete `outputs/sft/` and `outputs/dpo/`** (they're shape-locked
   to the previous mode).
3. Re-run SFT + DPO: `python scripts/run_training.py`.
4. Now deploy reflects the new mode consistently.

If you only want to *experiment* without retraining (e.g. confirm
whether the current model's eval output is broken under a different
flag), you can flip the config temporarily — the lenient
`_parse_eval_output` in `tutor.py` will recover JSON from either
shape, but quality will suffer until you retrain.

#### Why no_think eval might be worth trying

The `think` default gives the trained model headroom to reason
through CEFR judgments before emitting a score — generally produces
better scores at the cost of more tokens and slower deploy. `no_think`
trades reasoning quality for ~3-5x faster eval and cleaner JSON parse
rates. On a small student (4B Q4), `no_think` is sometimes
*better* because the model lacks the budget to reason coherently and
ends up rambling — the cleaner `no_think` JSON-only output avoids the
mid-reasoning truncation failure mode we documented in §20.4.

### What does each judge actually score — user turns or tutor turns?

A common confusion: the dialogue handed to a judge contains *both*
learner-sim user turns and trained-tutor turns. Which side is being
graded?

**Every judge in this project scores the TUTOR turns (the trained
model). The learner-sim's user turns are only context — they tell the
judge what the tutor was responding to, but they are never the thing
being graded.**

Concrete evidence from the prompts in
[metrics.py](src/qwen_tutor/training/eval/metrics.py) and friends:

| # | Judge | Scoring target | Why |
| - | --- | --- | --- |
| 1 | locale_judge | proper nouns in **filtered output** (assistant text dominates) | Assistant turns are what the student will be trained on / has produced |
| 2 | on-policy DPO pair judge | **two competing TUTOR replies** to the same user prompt | Picks `chosen` / `rejected` between trained-policy and teacher tutor outputs |
| 3 | `topic_adherence` | TUTOR's coverage of assigned subtopics (whole transcript scanned for context) | The tutor is the one supposed to *steer* the conversation |
| 4 | `naturalness_judge` | **TUTOR only** | Prompt literally says *"Rate how natural the ASSISTANT's English sounds"* |
| 5 | `redirect_success_judge` | **TUTOR's reply right after the probe** | Prompt block: *"The tutor's immediate next reply: [TUTOR] {response}"* |

Why this is the only sensible design: the learner-sim is *deterministic
scaffolding* — the same LAN model called with the same prompts
produces ~the same user turns across SFT-eval and DPO-eval runs. If
judges scored user turns too, the SFT-vs-DPO compare report would mix
"how good is the tutor LoRA?" with "how moody was the LAN model on user
turn 3?" That's noise. By scoring only the tutor side, the delta
between intermediate-eval and final-eval is a clean signal of *what
training changed*.

Corollary: this is exactly why the learner-sim must stay on
`/no_think`. The learner-sim's job is to be **invisible scaffolding**.
If it leaks a `<think>` block into a "user" turn, the judge — which
reads the whole transcript as context even though it scores only the
tutor side — sees scaffolding inside what is supposed to be a learner
utterance and either parses it wrong or downgrades the tutor for the
broken transcript shape. (We hit exactly this — see §20.8.)

### TL;DR for a 9B+ teacher

Most people upgrading from a 4B teacher just want this in `config/generation.yaml`:

```yaml
thinking:
  teacher: no_think       # unchanged — semantic, baked into prompts.py
  judge: think            # ← flip — all 5 judges (#1-5) gain accuracy with 9B reasoning
  learner: no_think       # unchanged — leaking <think> in user turns breaks the eval pipeline
  student_eval: think     # unchanged — deploy-time inference
```

**You don't have to change anything else.** The `max_tokens` floors in
the call sites are already at 1500–2048 (raised after the empirical
test in §20.4), which is enough headroom for a 9B teacher's reasoning
on these prompts.

If you want **only `naturalness_judge`** on `/think` (the highest-value
gain) while keeping topic_adherence and redirect_success cheap on
`/no_think`, hand-encode the directive in
`_NATURALNESS_JUDGE_PROMPT_RAW` ([metrics.py](src/qwen_tutor/training/eval/metrics.py))
— the helper respects an explicit per-prompt directive:

```python
_NATURALNESS_JUDGE_PROMPT_RAW = (
    "/think\n"                            # ← hand-pin only this judge
    "You are reviewing an English-language tutoring conversation between\n"
    ...
)
```

Then leave `thinking.judge: no_think` in the YAML for everything else.

### To switch a judge from `/no_think` to `/think`

Two edits per prompt:

1. **Remove** the leading `/no_think\n` line from the prompt template.
2. **Raise `max_tokens`** in the corresponding `await judge.generate(...,
   max_tokens=N, ...)` call. Recommended floors:
   - on-policy judge: 2048 (currently 400 — see
     [on_policy_pairs.py:451](src/qwen_tutor/generation/on_policy_pairs.py#L451))
   - topic_adherence: 1500 (currently 500)
   - naturalness_judge: 1500 (currently 200)
   - redirect_success_judge: 1500 (currently 150)

If you skip step 2, the model dumps all reasoning into
`reasoning_content`, the budget caps mid-thought, `content` returns
empty, and the JSON parser sees `""` → `no balanced JSON value found at
char 0` (or `<think>...</think>` with no closing JSON → `Expecting value:
line 1 column 2 (char 1)`).

### Quick empirical guide

| Mode | Wall time per judge call (4B Q4 over LAN) | Verdict yield |
| --- | --- | --- |
| `/no_think`, max_tokens=200 | ~1 s | ~100% parseable |
| `/think`, max_tokens=200 | ~10 s | ~0% (truncated mid-reasoning) |
| `/think`, max_tokens=2048 | ~30–60 s | ~95% parseable |

For a smoke run with a few dozen judge calls, the `/think + 2048` budget
costs only a few extra minutes. For a full-pipeline run with thousands
of filter+judge calls, the same choice is hours of extra wall time —
which is why filter judges default to `/no_think`.

---

## 17. From smoke run to full training (parameters to change)

The repo ships tuned for a **smoke run** that finishes end-to-end in
~30 minutes on a 12 GB card. To switch to a real training run you have
to scale roughly five knobs across two configs.

### 17.1 `config/training.yaml`

```yaml
base_model:
  model_id: "./vendor/models/Qwen_3.5_4B"        # smoke: Qwen_3.5_0.8B; full: 4B or 8B
  tokenizer_id: "./vendor/models/Qwen_3.5_4B"
  quantization:
    enabled: true                                # 0.8B fits bf16 (false). 4B/8B → enable 4-bit on a 12 GB card.

sft:
  num_train_epochs: 3                            # smoke: 1, full: 2-3
  max_steps: -1                                  # smoke: 5, full: -1 (full epoch)
  max_seq_length: 4096                           # smoke: 1024, full: 4096
  per_device_train_batch_size: 1                 # raise only if VRAM allows; LoRA on 4B in 4-bit usually OOMs at >1
  gradient_accumulation_steps: 8                 # raise to compensate for batch=1 (effective batch = 8)
  data:
    use_all_data: true                           # default — use every filtered record
    mix_ratio_sft: 0.80                          # only enforced if use_all_data=false
    mix_ratio_eval: 0.20

dpo:
  num_train_epochs: 1                            # 1 is usually enough for DPO
  max_steps: -1                                  # smoke: 5, full: -1
  max_length: 4096                               # smoke: 2048
  max_prompt_length: 2048                        # smoke: 1024
  data:
    use_all_data: true                           # default — use every filtered register + on_policy
    mix_ratio_register: 0.70                     # only enforced if use_all_data=false
    mix_ratio_on_policy: 0.30

on_policy:
  max_per_level: null                            # smoke: 2, full: null (all filtered SFT examples)
  concurrency: 4                                 # smoke: 2; raise to match judge server capacity
  min_margin: 2                                  # keep at 2 — margin 1 pairs are too noisy

evaluation:
  n_per_level: 20                                # smoke: 5, full: 20-50
  concurrency: 4
```

### 17.2 `config/generation.yaml`

```yaml
teacher:
  openai:
    model: <whatever /v1/models reports>         # if you set llama-server --alias, use that
    base_url: http://127.0.0.1:8080/v1           # or LAN IP if judge runs on a different box

generation:
  n_per_level: 100                               # smoke: 10, full: 100-500 per (level, locale)
  cefr_levels: ["A1", "A2", "B1", "B2", "C1", "C2"]  # smoke: subset, full: all 6
  locales: ["china", "japan", "italy"]           # smoke: 1, full: all you ship
  concurrency: 4                                 # smoke: 2; raise to match teacher server capacity
  dialogues_per_seed: 4                          # smoke: 1-4, full: 4-10
  min_turns: 10
  max_turns: 16

filtering:
  enable_locale_judge: true                      # smoke: optional, full: REQUIRED (catches Western leakage)
  enable_naturalness_judge: false                # opt-in; expensive
  short_circuit: true

top_up:
  enabled: true                                  # smoke: false, full: true (rebalance per-category survivors)
  target_per_category: 5
  max_top_up_per_category: 10
```

### 17.3 Holdout sampling

Before training, carve a holdout from the seeds — this MUST happen
before generating SFT/DPO/eval, or the same seeds end up in both pools:

```powershell
# Full: 10 per level across all 6
python scripts/sample_holdout.py --n-per-level 10 --levels A1,A2,B1,B2,C1,C2
```

### 17.4 Cost / time expectations

| Phase | Smoke (this repo) | Full (n_per_level=100, 6 levels, 1 locale) |
| --- | --- | --- |
| Seed → SFT → redirect → eval → register | ~30 min on RTX 3060 + 4B Q4 | ~24–40 hr |
| Filter pipeline (with locale judge) | ~5 min | ~6–10 hr |
| SFT training | ~90 s (5 steps, 0.8B) | ~6–12 hr (1 epoch, 4B QLoRA) on RTX 5090 |
| `on_policy_gen` | ~5 min | ~3–6 hr |
| DPO training | ~4 min (5 steps, 0.8B) | ~3–6 hr (1 epoch, 4B QLoRA) on RTX 5090 |
| Holdout eval × 2 | ~16 min (4 examples) | ~4–8 hr each |

Full pipeline on a single RTX 5090 is ~3–4 days end-to-end. On an RTX
3060 with the teacher on a LAN box, plan for ~7–10 days.

### 17.5 Full-train parameter reference (detailed)

This section enumerates every knob that scales the pipeline from smoke
to full training, grouped by the workflow boundary it crosses. Each
entry shows: where it lives, what it controls, the smoke value, the
recommended full-train value, and what raising it costs.

#### 17.5.1 Generation knobs — [config/generation.yaml](config/generation.yaml)

##### Volume (what scales total runtime)

| Knob | Smoke | Full | What it controls | Cost driver |
| --- | --- | --- | --- | --- |
| `generation.n_per_level` | 10 | **100–500** | Seeds per (CEFR level, locale). The pool every downstream stream draws from. | Linear in everything downstream. |
| `generation.cefr_levels` | one or two | **all 6** (`A1,A2,B1,B2,C1,C2`) | Which levels exist. | Linear. |
| `generation.locales` | `["china"]` | all targets (`["china","japan","italy",…]`) | Multi-country grounding pool. | Linear per added locale. |
| `generation.dialogues_per_seed` | 1 | **4–10** | Same scenario, N samples (variant SFT dialogues). Multiplies SFT without re-seeding. | Linear on SFT cost. |
| `generation.concurrency` | 2 | **4–8** | Parallel calls to the teacher server. Bound by the server, not the client. | Saves time, no quality cost. |
| `generation.min_turns` / `max_turns` | 10 / 16 | 10 / 16 (keep) | Dialogue length range. Smaller = cheaper but less context. | Quadratic in long-form coverage. |

##### Per-stream fractions (what gets generated as redirect/persistent)

These are fractions of the seed pool. Single-shot redirects default 0.15–0.20;
persistent (3-strike) streams are 0.05–0.08 (rarer abuse pattern).

| Stream | Knob | Smoke | Full | Notes |
| --- | --- | --- | --- | --- |
| Single-shot redirect | `redirect_fraction` | 0.15 | 0.20 | Generic axis. |
| Locale-violation redirect | `locale_redirect_fraction` | 0.10 | 0.15 | Western-default leaks. |
| Pedagogy redirect | `pedagogy_redirect_fraction` | 0.10 | 0.15 | "Explain the grammar". |
| Language redirect | `language_redirect_fraction` | 0.10 | 0.15 | L1 / requests_l1 / speaks_l1. |
| Persona redirect | `persona_redirect_fraction` | 0.10 | 0.15 | "Are you AI?". |
| Topic redirect | `topic_redirect_fraction` | 0.10 | 0.15 | Off-topic drift. |
| Role-swap redirect | `role_swap_redirect_fraction` | 0.10 | 0.15 | Learner takes tutor seat. |
| Persistent off-topic | `persistent_off_topic_fraction` | 0.05 | **0.08** | 3-strike. |
| Persistent language-violation | `persistent_language_violation_fraction` | 0.05 | 0.05 | 3-strike. |
| Persistent persona-break | `persistent_persona_break_fraction` | 0.05 | 0.05 | 3-strike. |
| Persistent role-swap | `persistent_role_swap_fraction` | 0.05 | 0.05 | 3-strike. |
| Eval | `eval_fraction` | 0.20 | **0.25** | Share of SFT used to generate `<think>`-mode eval samples. |
| Normal angle-shift | `angle_shift_fraction` | 0.0 | 0.10 | Opt-in augmentation; `[learner]` becomes a soft hint. |

Raising any fraction adds attempts × `(level_count × n_per_level × fraction)` more
teacher calls. With 9B teacher ~10s/dialogue concurrency=2 the cost is ~5s per attempt.

##### Filter toggles — `filtering:`

| Knob | Smoke | Full | Notes |
| --- | --- | --- | --- |
| `enable_locale_judge` | false | **true** | REQUIRED at full scale to catch Western leakage. ~$0 (uses local judge). |
| `enable_naturalness_judge` | false | optional | Expensive (extra teacher pass per turn); current naturalness regex is usually enough. |
| `short_circuit` | true | true | First-failing-filter wins. Keep on; saves cost. |
| `filtering.concurrency` | 2 | **4** | Parallel filter judge calls. |

##### Category-balanced seed top-up — `top_up:` (single-shot rebalance)

| Knob | Smoke | Full | Notes |
| --- | --- | --- | --- |
| `top_up.enabled` | false | **true** | Rebalances per-`category` survivors across `sft + eval + dpo` pools. |
| `target_per_category` | 5 | **20–50** | Minimum survivors per (level, locale, category) you want after filtering. |
| `max_top_up_per_category` | 10 | 30 | Hard cap so a stuck category cannot blow seed budget. |
| `auto_rerun_downstream` | true | true | After top-up generates new seeds, automatically re-run SFT + redirect + register + eval + filter_* downstream. |

##### Yield-aware iterative top-up — `sft_topup_target_per_level` (per-stream loop)

The new opt-in loop (`--stages sft_topup`) drives generation+filter
across all 12 SFT streams until each `(stream, level)` has at least
`target_per_level` passing-filter examples or the per-stream knob hits
its cap. See [§10.5](#105-category-taxonomy-and-top-up-rebalancing) and
[scripts/run_generation.py](scripts/run_generation.py) `_stage_sft_topup`.

Each stream's target is resolved by `_resolve_targets` at topup start,
which supports **two YAML knob shapes per stream**:

- **`<stream>_target_per_level`** (int) — absolute floor. The
  per-level count this stream must reach.
- **`<stream>_target_ratio`** (float, 0 < r < 1) — share of the final
  post-filter mix this stream should occupy. Computed from the
  *other* streams' absolute targets via:

  ```text
  T_per_level     = sum_of_absolute_targets / (1 − sum_of_ratios)
  ratio_target_i  = ratio_i × T_per_level
  ```

  Worked example: with `normal_target_ratio: 0.5` and all 11 other
  streams at absolute `target_per_level: 10` (sum_abs = 110), the
  resolver gives `normal_target_per_level = 110` (so the final mix is
  exactly 50% normal / 50% non-normal). When ratio and absolute are
  both set for the same stream, **ratio takes precedence**.

At topup start the stage prints the resolved per-stream target and its
share of total mix so you can sanity-check the math before generation
runs. Validation: ratios must be in `(0, 1)`; `sum_of_ratios < 1`; at
least one absolute target must exist as the anchor for the ratio math.

| Knob | Smoke | Full | Notes |
| --- | --- | --- | --- |
| `sft_topup_target_per_level` | 5 | **10–30** | Shared default for any stream without an explicit override. |
| `<stream>_target_per_level` | — | per-stream | Absolute floor. Search order: `params` → `cfg["generation"]` → shared default. |
| `<stream>_target_ratio` | — | per-stream | Declarative mix share (0..1). When set, supersedes the absolute target for that stream. |
| `persistent_<axis>_target_per_level` | 5 | 10 | Per-axis absolute target (left as the same-named knob for back-compat). |
| `SFT_TOPUP_MAX_ROUNDS` (constant in script) | 5 | 5 (keep) | Hard cap per stream. |
| `SFT_TOPUP_FRACTION_STEP` (constant) | 0.15 | 0.15 (keep) | Additive bump per round on fraction-gated streams. |
| `SFT_TOPUP_SEED_ATTRITION_BUDGET` (constant) | 1.6 | 1.6 (keep) | Extra seeds per desired post-filter example (normal SFT). |

Example YAML using both knob shapes — set normal to 50% of the mix and
give the umbrella `redirect` axis 3× the floor of the specialized
redirect axes:

```yaml
generation:
  # ... fractions above ...

  sft_topup_target_per_level: 10        # shared default

  normal_target_ratio: 0.5              # 50% of total post-filter mix
  redirect_target_per_level: 30         # 3x the other redirect axes
  # all other streams use the shared default (10)
```

Final resolved mix (4 levels, all streams):

```text
normal                          per_level=130  share=50.0%   ← from ratio
redirect (umbrella)             per_level= 30  share=11.5%   ← absolute override
locale_redirect                 per_level= 10  share= 3.8%
pedagogy_redirect               per_level= 10  share= 3.8%
language_redirect               per_level= 10  share= 3.8%
persona_redirect                per_level= 10  share= 3.8%
topic_redirect                  per_level= 10  share= 3.8%
role_swap_redirect              per_level= 10  share= 3.8%
persistent_off_topic            per_level= 10  share= 3.8%
persistent_language_violation   per_level= 10  share= 3.8%
persistent_persona_break        per_level= 10  share= 3.8%
persistent_role_swap            per_level= 10  share= 3.8%
TOTAL                                    260  100.0%
```

Topup is **opt-in** — not in default pipeline. Run as
`python scripts/run_generation.py --stages sft_topup`. Loop is
resumable (resumes from `done_ids` each round).

**Important: `target_per_level` is a FLOOR, not an exact count.** The
loop drives counts *up* to target; it does not trim excess. A stream
whose initial generation already exceeds its target will stay over
target and contribute its actual count (not the target) to the final
mix. If you need precise control over share, set the corresponding
`fraction` knob low enough that initial generation lands at-or-below
the target, then let the loop top up the rest.

#### 17.5.2 Training knobs — [config/training.yaml](config/training.yaml)

##### Base model + quantization — `base_model:`

| Knob | Smoke | Full | Notes |
| --- | --- | --- | --- |
| `base_model.model_id` | `./vendor/models/Qwen_3.5_0.8B` | **`./vendor/models/Qwen_3.5_4B`** or `_8B` | 4B QLoRA fits on a 12 GB card. 8B needs 16 GB+. |
| `base_model.tokenizer_id` | same as model | same | Must match. |
| `base_model.quantization.enabled` | false (0.8B fits bf16) | **true** | 4-bit NF4 with double-quant for 4B+. |
| `base_model.torch_dtype` | `bfloat16` | `bfloat16` | Keep. |
| `base_model.attn_implementation` | `sdpa` | `sdpa` | `flash_attention_2` if Flash Attention installed and supported. |
| `base_model.gradient_checkpointing` | true | true | Halves peak VRAM at ~10% step-time cost. |

##### LoRA shape — `lora:`

| Knob | Smoke | Full | Notes |
| --- | --- | --- | --- |
| `lora.r` | 32 | **32–64** | Rank. 64 = more capacity, more VRAM. |
| `lora.lora_alpha` | 64 | 2 × `r` (so 64 or 128) | Convention. |
| `lora.lora_dropout` | 0.05 | 0.05 | Keep. |
| `lora.target_modules` | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` | same | All attention + MLP projections. |

##### SFT hyperparameters — `sft:`

| Knob | Smoke | Full | Notes |
| --- | --- | --- | --- |
| `sft.num_train_epochs` | 1 | **2–3** | 3 epochs gives diminishing returns past 4B. |
| `sft.max_steps` | 5 | **-1** | -1 = full epoch. |
| `sft.max_seq_length` | 1024 | **4096** | Persistent dialogues can hit 12–16 turns. 4096 covers all. |
| `sft.per_device_train_batch_size` | 1 | 1 (LoRA on 4B in 4-bit OOMs at >1) | Keep 1, raise grad-accum. |
| `sft.gradient_accumulation_steps` | 4 | **8** | Effective batch = 8 across all gradient updates. |
| `sft.learning_rate` | 2e-4 | 2e-4 | Standard LoRA LR. |
| `sft.warmup_ratio` | 0.05 | 0.05 | Keep. |
| `sft.lr_scheduler_type` | cosine | cosine | Keep. |
| `sft.weight_decay` | 0.01 | 0.01 | Keep. |
| `sft.optim` | `paged_adamw_8bit` | `paged_adamw_8bit` | Required with 4-bit base; saves VRAM. |
| `sft.bf16` / `sft.fp16` | bf16 true | bf16 true | Match base dtype. |
| `sft.data.use_all_data` | true | true | Default — train on every filtered record. |
| `sft.data.mix_ratio_sft` / `mix_ratio_eval` | 0.80 / 0.20 | enforced only if `use_all_data: false` | Set to false only when both pools are large. |

##### DPO hyperparameters — `dpo:`

| Knob | Smoke | Full | Notes |
| --- | --- | --- | --- |
| `dpo.num_train_epochs` | 1 | **1** | One epoch is usually enough for DPO. |
| `dpo.max_steps` | 5 | -1 | -1 = full epoch. |
| `dpo.max_length` | 2048 | **4096** | Match SFT. |
| `dpo.max_prompt_length` | 1024 | **2048** | Multi-turn prompts need room. |
| `dpo.beta` | 0.1 | 0.1 | DPO regularization strength. |
| `dpo.learning_rate` | 5e-6 | 5e-6 | Much smaller than SFT LR. |
| `dpo.gradient_accumulation_steps` | 4 | 8 | Same logic as SFT. |
| `dpo.sft_adapter_path` | `outputs/sft` | `outputs/sft` | Load SFT-trained LoRA before DPO. |
| `dpo.ref_model` | null | null | Use the SFT-trained adapter's frozen ref. |
| `dpo.data.use_all_data` | true | true | Default — every filtered DPO record. |
| `dpo.data.mix_ratio_register` / `mix_ratio_on_policy` | 0.70 / 0.30 | enforced only if `use_all_data: false` | Set false only when both pools have ~500+ records each. |

##### On-policy pair generation — `on_policy:`

| Knob | Smoke | Full | Notes |
| --- | --- | --- | --- |
| `on_policy.enabled` | true | true | Required for DPO mix with register-only. |
| `on_policy.adapter_path` | `outputs/sft` | `outputs/sft` | Generate from the SFT-trained student. |
| `on_policy.max_per_level` | 2–5 | **null** (all filtered SFT) | Caps how many SFT examples become on-policy pairs. |
| `on_policy.min_margin` | 2 | 2 | Judge margin; 1 is too noisy, 3 is too strict. |
| `on_policy.concurrency` | 2 | **4** | Bound by judge server. |
| `on_policy.target_max_tokens` | 320 | 320 | Per regenerated tutor turn. |
| `on_policy.target_temperature` | 0.7 | 0.7 | Mid-range — diverse but not chaotic. |

##### Evaluation — `evaluation:`, `eval_callback:`, `comparison:`

| Knob | Smoke | Full | Notes |
| --- | --- | --- | --- |
| `evaluation.n_per_level` | 3 | **20–50** | Per-level holdout eval count. |
| `evaluation.concurrency` | 2 | 4 | Judge over LAN bottleneck. |
| `evaluation.holdout_dir` | `data/holdout` | same | Carved by `scripts/sample_holdout.py` BEFORE generation (see §17.3). |
| `evaluation.intermediate_adapter_path` | `outputs/sft` | same | After SFT. |
| `evaluation.final_adapter_path` | `outputs/dpo` | same | After DPO. |
| `eval_callback.enabled` | false | optional | Mid-training eval pings during SFT. VRAM-hungry. |
| `eval_callback.n_conversation_samples` / `n_evaluation_samples` | 10 / 10 | 20 / 20 | Sample count per callback fire. |
| `comparison.threshold` | 0.05 | 0.05 | Minimum delta before flagging a regression. |
| `comparison.fail_on_regression` | false | optional | true = exit non-zero in CI. |

##### Adapter merge + GGUF export — `gguf_export:`

| Knob | Smoke | Full | Notes |
| --- | --- | --- | --- |
| `gguf_export.adapter_path` | `outputs/dpo` | `outputs/dpo` | LoRA to merge. |
| `gguf_export.outtype` | bf16 | bf16 | Intermediate fp16/bf16 GGUF. |
| `gguf_export.quantize` | Q4_K_M | **Q4_K_M** (good speed/quality) or Q5_K_M | Q5 is bigger but better quality. |
| `gguf_export.merged_dir` | `outputs/merged` | same | Where merged-LoRA fp16 lands before quantize. |
| `gguf_export.keep_merged` | false | true (if you want to re-quantize without re-merging) | Disk-heavy. |
| `gguf_export.llama_cpp_repo` | `vendor/llama_cpp_src` | same | Bundled in `vendor/`. |
| `gguf_export.llama_quantize_bin` | `vendor/llama_cpp/llama-quantize.exe` | same | Bundled binary. |

#### 17.5.3 Deploy knobs — runtime / system prompt / safety filter

Deploy parameters live in three places:

1. **`config/training.yaml` → `deployment_system_prompt:`** — the template
   that `render_scenario_deployment_system_prompt` fills in per session.
2. **`config/banned_terms_deploy.yaml`** — content filter at inference
   time (separate from generation-time filter; usually a subset).
3. **`TutorRuntime(...)`** Python kwargs in
   [src/qwen_tutor/deploy/tutor.py](src/qwen_tutor/deploy/tutor.py).

##### `TutorRuntime` runtime knobs

| Knob | Smoke / default | Full / production | What it controls |
| --- | --- | --- | --- |
| `model_path` | `outputs/gguf/qwen-en-tutor-Q4_K_M.gguf` | same | Final quantized GGUF. |
| `base_url` | `http://127.0.0.1:8080/v1` | LAN or production IP | llama-server endpoint. |
| `cefr_level` | `B1` | per session | Target register. |
| `target_cefr` (deprecated alias) | — | use `cefr_level` | |
| `locale_name` | from `config/locale.yaml` `default_locale` | per session | Drives system prompt grounding. |
| `topic` / `subtopics` | scenario JSON | scenario JSON | Anchors the conversation. |
| `user_role_name` / `description` | scenario JSON | scenario JSON | Learner persona. |
| `model_role_name` / `description` | scenario JSON | scenario JSON | Tutor persona. |
| `max_tokens` | 320 | 320–512 | Per assistant turn. |
| `temperature` | 0.7 | 0.7 | Match training-time. |
| `top_p` | 0.95 | 0.95 | Keep. |
| `banned_terms_path` | `config/banned_terms_deploy.yaml` | same | Override for stricter / looser. |
| `enable_safety_filter` | true | true | Mechanical inference-side filter. |
| `enable_session_end_detection` | true | true | Detects `[SESSION_END: persistent_<axis>]` sentinels and closes session. |

##### Deployment system prompt — `config/training.yaml` → `deployment_system_prompt:`

The template uses these placeholders (filled at runtime):

```text
{cefr_level} {locale_name} {country} {country_adjective}
{learner_description} {topic} {subtopics_bulleted}
{user_role_name} {user_role_description}
{model_role_name} {model_role_description}
{avoid_cultures_phrase} {avoided_topics_sentence}
```

Edit the template in `config/training.yaml` to change what the deployed
tutor is told about its role, the user, the locale, and the boundaries.
The same template renders during SFT generation (so training and deploy
see identical system context). Change carefully — a change here forces
a re-train if you want the model to internalize it.

##### Deploy-time banned terms — `config/banned_terms_deploy.yaml`

This file is **separate** from the generation-time banned terms. The
deploy list should be a strict **subset** focused on what the user
might still try to elicit from a trained model. Categories that catch
benign civic content at generation time (like the removed `government`
entry) should usually stay out of the deploy list too.

##### Persistence dispatcher — `[persistence]` system-prompt block

When the deploy-time system prompt contains a `[persistence]` block, the
runtime watches the model's output for `[SESSION_END: persistent_<axis>]`
sentinels. On a match it closes the session. See
[src/qwen_tutor/deploy/tutor.py](src/qwen_tutor/deploy/tutor.py)
`detect_session_end` and `SESSION_END_AXES`.

#### 17.5.4 Quick scaling recipes

##### Smoke (30 min end-to-end on 12 GB)

```yaml
# generation.yaml
generation:
  n_per_level: 10
  cefr_levels: ["B1"]
  locales: ["china"]
  dialogues_per_seed: 1
  concurrency: 2
filtering:
  enable_locale_judge: false
top_up:
  enabled: false
```

```yaml
# training.yaml
base_model: { model_id: "./vendor/models/Qwen_3.5_0.8B", quantization: { enabled: false } }
sft: { num_train_epochs: 1, max_steps: 5, max_seq_length: 1024 }
dpo: { max_steps: 5, max_length: 2048 }
on_policy: { max_per_level: 2 }
evaluation: { n_per_level: 3 }
```

##### Mid (overnight on 24 GB)

```yaml
# generation.yaml
generation:
  n_per_level: 50
  cefr_levels: ["A2","B1","B2","C1"]
  locales: ["china"]
  dialogues_per_seed: 2
  concurrency: 4
filtering:
  enable_locale_judge: true
top_up:
  enabled: true
  target_per_category: 10
```

```yaml
# training.yaml
base_model: { model_id: "./vendor/models/Qwen_3.5_4B", quantization: { enabled: true } }
sft: { num_train_epochs: 2, max_steps: -1, max_seq_length: 4096 }
dpo: { max_steps: -1, max_length: 4096, max_prompt_length: 2048 }
on_policy: { max_per_level: 10 }
evaluation: { n_per_level: 10 }
```

##### Full (multi-day on RTX 5090)

```yaml
# generation.yaml
generation:
  n_per_level: 200
  cefr_levels: ["A1","A2","B1","B2","C1","C2"]
  locales: ["china","japan","italy"]
  dialogues_per_seed: 4
  concurrency: 8
filtering:
  enable_locale_judge: true
  concurrency: 4
top_up:
  enabled: true
  target_per_category: 30
sft_topup_target_per_level: 20
```

```yaml
# training.yaml
base_model: { model_id: "./vendor/models/Qwen_3.5_8B", quantization: { enabled: true } }
sft: { num_train_epochs: 3, max_steps: -1, max_seq_length: 4096, gradient_accumulation_steps: 8 }
dpo: { num_train_epochs: 1, max_length: 4096, max_prompt_length: 2048, gradient_accumulation_steps: 8 }
on_policy: { max_per_level: null, concurrency: 4 }
evaluation: { n_per_level: 30, concurrency: 4 }
```

#### 17.5.5 Where each knob lives at runtime

| Stage | Reads from |
| --- | --- |
| Seed generation | `generation.{n_per_level, cefr_levels, locales, concurrency}` + `locale.yaml` |
| SFT generation | + `dialogues_per_seed, min_turns, max_turns, angle_shift_fraction` |
| Redirect / persistent generation | + `<axis>_fraction` |
| Eval generation | + `eval_fraction`; uses `cefr_specs.yaml` for register targets |
| Register (DPO chosen) | reads `sft_filtered/`; uses `prompts.py` SPOIL_REWRITE_PROMPT |
| On-policy DPO | `on_policy:` block in `training.yaml`; uses student adapter |
| filter_sft / filter_eval / filter_dpo | `banned_terms.yaml`, `non_latin_script` defaults, `cefr_specs.yaml` for naturalness targets |
| top_up | `top_up.target_per_category`, `top_up.max_top_up_per_category` |
| sft_topup | `sft_topup_target_per_level`, per-stream overrides |
| SFT training | `base_model:`, `lora:`, `sft:` blocks |
| DPO training | `base_model:`, `lora:`, `dpo:` blocks |
| Holdout eval | `evaluation:` block |
| GGUF export | `gguf_export:` block |
| Deploy runtime | `TutorRuntime(...)` kwargs + `deployment_system_prompt:` + `banned_terms_deploy.yaml` |

---

## 18. Important things not to forget

These are the foot-guns that bit us during the smoke run, ordered by how
likely they are to bite again.

1. **Windows: always set `PYTHONUTF8=1`.** `trl` reads its jinja chat
   templates with `Path.read_text()` (no encoding arg). On Windows that
   defaults to cp1252 and dies with `UnicodeDecodeError: 'charmap'
   codec can't decode byte 0x81`. Either set the env var per-shell
   (`$env:PYTHONUTF8 = "1"`) or globally
   (`[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")`).
   (python -X utf8 ./scripts/run_training.py --stages train_sft,eval_intermediate).

2. **LoRA adapters are NOT portable across base models.** A LoRA
   trained against Qwen3.5 0.8B has the wrong `hidden_size` and layer
   count for Qwen3.5 4B. **Delete `outputs/sft/` and `outputs/dpo/`
   whenever you change `base_model.model_id`**, then retrain.

3. **Qwen3.5 has no shipped `generation_config.json`.** The
   `HFTargetModelClient` in
   [training/eval/run_eval.py](src/qwen_tutor/training/eval/run_eval.py#L155-L164)
   explicitly passes `eos_token_id=tokenizer.eos_token_id` to
   `model.generate()`. Without it, generation runs to `max_new_tokens`
   and hallucinates fake `<|im_end|><|im_start|>user\n...` turns. If
   you write a new HF generation path elsewhere, do the same.

4. **Holdout sampling must run BEFORE generation.** The script
   removes sampled seeds from `data/seeds/` by default, but only if
   they're still there. If you generate SFT first and then sample
   holdout, the same scenarios end up in both train and test —
   silent contamination. Pass `--keep-in-seeds` only if you know what
   you're doing.

5. **Judge mode and `max_tokens` must match.** `/think` with
   small `max_tokens` returns empty content, which surfaces as
   `naturalness_judge failed: no balanced JSON value found`. See
   [section 16](#16-switching-think-vs-no_think-qwen35-thinking-mode).

6. **`llama-server --alias <name>`** to control the string that
   `/v1/models` reports. Without it, llama-server uses the bare
   filename (`Qwen3.5-4B-UD-Q4_K_XL.gguf`), and you have to mirror
   that exact string in `config/generation.yaml` → `teacher.openai.model`
   / `judge.openai.model`. With `--alias qwen3.5-4b` you control both
   sides.

7. **Stop llama-server before training on a 12 GB card.** Teacher +
   trainee don't fit together. PyTorch and llama.cpp both allocate
   eagerly; whichever loaded second gets the OOM.

8. **`compare` needs matching `--run-id` across `eval_intermediate`
   and `eval_final`.** When you split stages across invocations, pass
   `--run-id smoke` (or any stable string) to both, or the compare
   stage can't find the baseline directory. The default run_id is the
   current timestamp, which differs every invocation.

9. **`filter_dpo` must rerun after `on_policy_gen`.** It filters
   *both* `register_*.jsonl` and `on_policy_*.jsonl` together so they
   share quality bars. Skipping it means `train_dpo` trains on
   unfiltered policy outputs.

10. **`outputs/sft/` is the SFT adapter; `outputs/dpo/` contains the
    DPO adapter PLUS a `ref/` subdir with a frozen reference policy.**
    Don't delete `outputs/sft/` between `train_sft` and `train_dpo` —
    DPO loads it via `dpo.sft_adapter_path: outputs/sft`.

11. **Re-run `eval_intermediate` whenever the eval code changes.**
    The `summary.json` from a buggy run (e.g. all judge calls
    returning 0 because of `/think` truncation) is *frozen wrong*. The
    `compare` stage will silently use it as the baseline. When in
    doubt, delete `data/eval_results/<run_id>_intermediate/` and rerun.

12. **The `outputs/sft/` adapter is locked to the base model from the
    `training.yaml` at training time.** Loading it later with a
    different `base_model.model_id` will silently apply weights to the
    wrong layers and produce garbage. Keep `base_model` pinned for
    the entire SFT → DPO → eval cycle.

13. **`top_up` writes seeds at the END of the pipeline and nothing
    processes them.** `top_up` is the last stage in `ALL_STAGES` in
    `scripts/run_generation.py`. By the time it diagnoses deficits and
    emits new seeds, every downstream stage (`sft`, all 7 redirects,
    `register`, `eval`, `filter_*`) has already finished. The new
    seeds sit unused in `data/seeds/<level>.jsonl`. **If you enable
    `top_up.enabled: true` you MUST re-run `run_generation.py
    --stages sft,redirect,locale_redirect,pedagogy_redirect,language_redirect,persona_redirect,topic_redirect,role_swap_redirect,register,eval,filter_sft,filter_eval,filter_dpo`
    afterward** to actually train on the rebalanced pool. Otherwise set
    `top_up.enabled: false` — the inflated seed counts will mislead you
    into thinking you have more data than the model actually saw. The
    8h test run in this repo hit exactly this: `top_up` added 180
    seeds, none flowed through, and the diversity report still showed
    only 5 of 10 life-domain categories.

---

## 19. True offline operation (air-gapped checklist)

The bundle phase (`scripts/setup_offline.py`) gets the heavy artifacts
(wheels, models, llama.cpp binaries) onto the offline box. But three
quieter network calls remain inside the training/runtime code and will
fail or hang on a truly air-gapped machine. Disable them explicitly:

### 19.1 Disable the Hugging Face Hub at the env level

The `trl` library makes a `HEAD https://huggingface.co/api/telemetry/trl/DPOTrainer`
call every time `DPOTrainer` initializes — verified in
`data/_logs/run_dpo_eval2.log`. Other transformers calls (`from_pretrained`
without `local_files_only`) can also try to reach the Hub on a cold
cache.

Set both env vars in any shell that runs training, evaluation, or deploy:

```powershell
# Windows / PowerShell — set once for the current session
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUTF8 = "1"

# Or persist them across sessions:
[Environment]::SetEnvironmentVariable("HF_HUB_OFFLINE", "1", "User")
[Environment]::SetEnvironmentVariable("TRANSFORMERS_OFFLINE", "1", "User")
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
```

```bash
# Linux / macOS — add to ~/.bashrc or ~/.zshrc
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUTF8=1
```

With `HF_HUB_OFFLINE=1`, the trl telemetry call short-circuits with no
network attempt; with `TRANSFORMERS_OFFLINE=1`, every `from_pretrained`
in this repo behaves as if `local_files_only=True` was passed.

### 19.2 Disable wandb (already done by default)

`config/training.yaml` sets `sft.report_to: "none"` and `dpo.report_to:
"none"`. Don't change those to `"wandb"` on an offline box. If you must
log, write a local CSV callback instead.

### 19.3 Verify your model paths are local folders, not Hub IDs

In `config/training.yaml`, `base_model.model_id` and
`base_model.tokenizer_id` must be **local paths** (start with `./` or
`/`), not bare Hugging Face Hub names. Same for any `--adapter`,
`--base-model`, `--scenario-file` CLI argument. Example correct
form:

```yaml
base_model:
  model_id: "./vendor/models/Qwen_3.5_0.8B"
  tokenizer_id: "./vendor/models/Qwen_3.5_0.8B"
```

Bare names like `"Qwen/Qwen3-8B"` resolve into Hub fetches even with
the offline env vars set, because transformers tries the cache first
and errors after, not before. Always use the explicit path on offline
boxes.

### 19.4 Audit network use during a run

To prove you're air-gapped, grep the log file for any external HTTP
call after one full pipeline cycle. Only the local llama-server
(`http://192.168.135.32:8080` or `http://127.0.0.1:8080`) should appear:

```bash
grep -aoE "https?://[^ \"]+" data/_logs/*.log \
  | grep -v "192\.168\|127\.0\|localhost"
# Expected: empty output. Any other URL = something is reaching out.
```

If anything besides your local llama-server URL shows up, hunt it down
before claiming offline operation.

### 19.5 What's bundled in `vendor/wheels/` right now

`scripts/setup_offline.py --only wheels` downloads the full dependency
tree into `vendor/wheels/`. After a recent refresh this contains 117
wheels totaling ~336 MB — every Python dep, dev extras (pytest, ruff),
and transitive deps including transformers, trl, peft, torch's CPU build,
bitsandbytes, datasets, accelerate, openai, anthropic, pydantic, tiktoken,
wandb. CUDA-specific torch builds need a separate
`scripts/setup_offline.py --only cuda-torch` pull (URL hardcoded for
cu124).

---

## 20. Known limitations and rough edges

Honest list of things that work but aren't perfect, observed during
the most recent full pipeline run on RTX 3060 + Qwen3.5 0.8B student +
Qwen3.5 4B Q4 LAN teacher/judge.

### 20.1 `/think` evaluation generation is fragile

The trained `/think` mode emits `<think>...</think>` then a JSON
`EvaluationOutput`. We observed:

- **Deploy-time eval works** — `TutorRuntime.evaluate_async()` returned
  valid JSON with proper CEFR estimate + dimension scores on the smoke
  scenario.
- **Training-eval `eval_json_valid_rate` was 0.00** before
  [run_eval.py:450-481](src/qwen_tutor/training/eval/run_eval.py#L450-L481)
  was fixed to construct the same prompt shape as `eval_gen.py` /
  `tutor.py`. Make sure your version has the
  `Tutor role: / Learner role: / Assigned topic: / Assigned subtopics:`
  preamble before `Transcript:`.

If `eval_json_valid_rate` is still 0 after that fix, suspect the
underlying eval training data — the next item.

### 20.2 Eval training data prompt-fragment leakage (fixed in filter)

When a small teacher (Qwen3.5 4B Q4) cannot follow the structured eval
prompt cleanly, it sometimes echoes bracketed system-prompt headers
verbatim into the assistant turn — strings like `"CHINA LOCALE
INSTRUCTION"`, `"Topic Adherence"`, `"User Turn"`, `"Overall CEFR"`,
`"LEARNER's USER"`. In one run, **27 of 47 (57%) of `eval_filtered/*_passed.jsonl`
examples were contaminated** this way before we added the filter.

**Mitigation (already applied):** a `scaffolding_leakage` category in
[config/banned_terms.yaml](config/banned_terms.yaml) lists only the
two full bracketed prompt headers — `CHINA LOCALE INSTRUCTION` and
`LOCALE INSTRUCTION` — that cannot legitimately appear in any tutor
output. Earlier iterations of this list also included `Topic
Adherence`, `Overall CEFR`, `User Turn`, and `LEARNER's USER`, but
those produced false positives: they're either real
`EvaluationOutput` field names (`topic_adherence`,
`overall_cefr_estimate`) that the model SHOULD mention while scoring,
or natural turn-labeling phrases the model uses to organize reasoning.
On a 47-example pool, the over-strict list incorrectly rejected 26
good examples; the conservative two-entry list now used rejects only
5–6 examples that genuinely dumped prompt headers, keeping 42 clean
ones for training.

**To verify your pool is clean:**

```bash
python -c "
import json
from pathlib import Path
needles = ('CHINA LOCALE', 'LOCALE INSTRUCTION', \"LEARNER's USER\",
           'User Turn', 'Topic Adherence', 'Overall CEFR')
leaked = 0
for fp in Path('data/eval_filtered').glob('*_passed.jsonl'):
    for line in fp.open('r', encoding='utf-8'):
        if not line.strip(): continue
        rec = json.loads(line)
        ex = rec.get('example', rec)
        for m in ex.get('messages', []):
            if m.get('role') == 'assistant' and any(n in m['content'] for n in needles):
                leaked += 1; break
print('leaked:', leaked)  # expect 0
"
```

**If you change the eval prompt structure later** (e.g. edit
`EVALUATION_USER_PROMPT_RAW` in
[prompts.py](src/qwen_tutor/generation/prompts.py) /
[prompts_compact.py](src/qwen_tutor/generation/prompts_compact.py))
to use new section header words, add those new header strings to the
`scaffolding_leakage` category as well.

**Re-run after change:**

```bash
PYTHONUTF8=1 python scripts/run_generation.py --stages filter_sft,filter_eval,filter_dpo
```

then re-train SFT so the adapter learns from the clean pool only.

### 20.3 Eval examples carry no `category` metadata

`scripts/run_diversity.py --input-dir data/eval_filtered` reports
`Categories: (none)`. `eval_gen.py` doesn't stamp the `category` field
from the source `ScenarioSeed` onto the `EvaluationExample.metadata`.
SFT and DPO pools do carry it correctly. Fix is a one-line addition in
`_make_eval_example` (or equivalent constructor) in `eval_gen.py`.

### 20.4 `/think + small max_tokens` on Qwen3.5 4B Q4 ≈ 0% verdict yield

Empirically confirmed during this session: Qwen3.5 4B Q4_K_XL's
reasoning typically takes 500–2000 tokens, sometimes more, before the
JSON answer emits. Combined with judge `max_tokens` budgets around
200–500 (as set originally in `metrics.py`), reasoning truncated
mid-thought and `content` came back empty — every judge call failed JSON
parse. Two known-good configurations:

| Use case                                   | Mode         | max_tokens   |
| ------------------------------------------ | ------------ | ------------ |
| Filter judges (volume)                     | `/no_think`  | 500          |
| Eval judges (topic_adherence, naturalness) | `/no_think`  | 500          |
| On-policy DPO pair judge                   | `/no_think`  | 500          |
| Eval judges (production quality)           | `/think`     | **≥ 2048**   |

Switching to `/think` requires raising `max_tokens` to at least 2048 at
the call site (e.g. [metrics.py:topic_adherence](src/qwen_tutor/training/eval/metrics.py#L420),
[metrics.py:naturalness_judge](src/qwen_tutor/training/eval/metrics.py#L509),
[on_policy_pairs.py:generate_batch](src/qwen_tutor/generation/on_policy_pairs.py#L451)).

### 20.5 DPO `max_length=2048` OOMs on a 12 GB card

`config/training.yaml` originally shipped `dpo.max_length: 2048` /
`dpo.max_prompt_length: 1024`. On an RTX 3060 (12 GB) with Qwen3.5 0.8B
in 4-bit, the DPO trainer (which holds trainee + reference + 2×
sequence) raised `CUDA OutOfMemoryError` at step 0. Fix that's in the
config now: `max_length: 1024`, `max_prompt_length: 512`. Raise these
back to 2048/1024 only on a 24 GB+ GPU.

### 20.6 PowerShell `Out-File` / `Tee-Object` can kill long-running pipelines

Long Python runs piped through `2>&1 | Tee-Object` or `2>&1 | Out-File`
in PowerShell occasionally abort the child process when tqdm progress
bars on stderr emit something PowerShell flags as `NativeCommandError`.
The actual Python process is fine — confirmed by re-running the same
stage via bash with `> log 2>&1`, which completes cleanly.

For long stages (`train_sft`, `train_dpo`, `eval_intermediate`,
`eval_final`, `on_policy_gen`), prefer:

```bash
# Git Bash on Windows, or any Linux shell
PYTHONUTF8=1 python scripts/run_training.py --stages train_dpo > data/_logs/run.log 2>&1
```

Reserve PowerShell for one-shot probes and config diffs.

### 20.7 Qwen3.5 4B Q4 ignores `chat_template_kwargs.enable_thinking` inconsistently

Setting `enable_thinking: false` via `chat_template_kwargs` on
llama-server **usually** works, but the inline `/no_think` directive in
the system prompt body is what actually drives it reliably across all
calls. The `OpenAITeacher` client in
[teacher.py:546-557](src/qwen_tutor/generation/teacher.py#L546-L557)
auto-forwards the kwarg when it sees `/no_think` in the system text.
For safety, always include the inline directive on judges that need
fast, parseable JSON.

### 20.8 The learner simulator must also be in `/no_think`

For `eval_intermediate` / `eval_final`, the learner-simulating model
(also Qwen3.5 4B over LAN) generates the user turn each round. If its
prompts don't carry `/no_think`, it leaks `<think>...</think>` content
into the user turns, which then propagates into the transcript fed to
the eval judges and breaks JSON parsing downstream. The three learner
prompt constants in
[run_eval.py:210-280](src/qwen_tutor/training/eval/run_eval.py#L210-L280)
all start with `/no_think` — keep it that way.

### 20.9 Holdout coverage from `--keep-in-seeds` is biased

For the smoke + first full pipeline run we used `sample_holdout.py
--keep-in-seeds` so we could expand the holdout without regenerating
training data. The same scenarios then appear in both train and test
pools, optimistically biasing eval metrics. For honest numbers,
re-sample without `--keep-in-seeds` and re-generate downstream stages.

### 20.10 The deploy-time scenario / safety pipeline does NOT apply to llama-server

If you GGUF-export the adapter and serve via `llama-server`
(`outputs/gguf/qwen-en-tutor-Q4_K_M.gguf`), the deploy-time
`render_scenario_deployment_system_prompt` substitution and
`BannedTermsFilter` are not in the loop. You'd need a thin proxy in
front of `llama-server`'s `/v1/chat/completions` endpoint to keep
parity with the Python `TutorRuntime` deployment.

---

## Project status

- 8-axis DPO taxonomy (assistant-side spoils + user-side SFT handlers
  for `locale_violation`, `pedagogy_weak`, `language_violation`,
  `persona_break`).
- 6 SFT streams generated per level (`normal`, `redirect`,
  `locale_redirect`, `pedagogy_redirect`, `language_redirect`,
  `persona_redirect`).
- 4 generation prompts switchable between `prompts.py` (Opus-grade) and
  `prompts_compact.py` (small-model tuned) via `QWEN_TUTOR_PROMPTS`.
- Quality pipeline: `speaks_l1_sanity`, `non_latin_script`,
  `banned_terms`, `mode_consistency`, `naturalness`, optional
  `locale_judge` / `naturalness_judge`.
- Joint SFT + DPO training with a single LoRA adapter learning both
  `/no_think` and `/think` modes.
- 8-metric end-to-end evaluation with intermediate vs final
  regression comparison.
- Entire pipeline runs offline against a local llama.cpp teacher.
