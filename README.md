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
13. [Config reference](#13-config-reference)
14. [Troubleshooting](#14-troubleshooting)

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
| `seeds`            | CEFR scenario seeds (topic, subtopics, roles, setting)   | `data/seeds/<level>.jsonl`                      |
| `sft`              | Normal multi-turn dialogues (10–16 turns)                | `data/sft_raw/normal_<level>.jsonl`             |
| `redirect`         | Off-topic probe at turn N → graceful pivot               | `data/sft_raw/redirect_<level>.jsonl`           |
| `locale_redirect`  | Learner names a wrong-locale entity → tutor handles it   | `data/sft_raw/locale_redirect_<level>.jsonl`    |
| `pedagogy_redirect`| Learner asks for grammar lecture → tutor reframes        | `data/sft_raw/pedagogy_redirect_<level>.jsonl`  |
| `language_redirect`| Learner speaks L1 / asks tutor for L1 → tutor pivots     | `data/sft_raw/language_redirect_<level>.jsonl`  |
| `persona_redirect` | "Are you AI?" / "What model are you?" → in-character pivot | `data/sft_raw/persona_redirect_<level>.jsonl` |
| `register`         | Multi-axis DPO pairs (8 axes, quota-balanced)            | `data/dpo_raw/register_<level>.jsonl`           |
| `eval`             | `/think` examiner examples for evaluation-mode training  | `data/eval_raw/<level>.jsonl`                   |
| `filter_sft`       | 5-filter pipeline over all 6 SFT streams                 | `data/sft_filtered/*_passed.jsonl` + `*_failed.jsonl` |
| `filter_eval`      | Same filters over eval examples                          | `data/eval_filtered/*_passed.jsonl`             |
| `filter_dpo`       | Same filters over register pairs                         | `data/dpo_filtered/*_passed.jsonl`              |

All stages **resume from existing output** — if you stop and rerun, the
already-written IDs are skipped. Failures land in `data/*_failures.jsonl`
with the model output that caused them.

See [section 11](#11-generation-data-types-what-each-file-is) for what each
of those data types looks like and why it exists.

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
Korean-target model leaks Japanese street names, etc. List those secondary
cultures here so prompts, anti-failure-mode blocks, and the LLM judge all
treat them as "wrong default":

```yaml
# config/locale.yaml
avoid_default_cultures:
  - "American"
  - "European"
  # - "Japanese"      # add when country == "China" or "Korea"
  # - "Korean"        # add when country == "Japan"
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
    "register", "eval",
    "filter_sft", "filter_eval", "filter_dpo",
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

---

## 11. Generation data types (what each file is)

Five distinct artifacts come out of the generation pipeline. They all
share a JSONL-per-CEFR-level layout, validated against the Pydantic v2
schemas in [src/qwen_tutor/schemas.py](src/qwen_tutor/schemas.py).

### 11.1 `ScenarioSeed` — `data/seeds/<level>.jsonl`

A scenario the teacher will dramatize. Each seed has a topic, 3–5
subtopics, a user role (the learner), a model role (the tutor), and a
setting string. No dialogue yet.

```json
{
  "id": "seed_A2_037",
  "topic": "buying vegetables at a wet market",
  "subtopics": ["prices", "freshness", "weighing", "small talk"],
  "user_role": {"name": "Mei", "description": "a college student living in Chengdu"},
  "model_role": {"name": "vendor", "description": "a friendly tomato seller"},
  "setting": "an outdoor wet market on a Saturday morning in Chengdu",
  "cefr_level": "A2"
}
```

### 11.2 `SFTExample` — `data/sft_raw/<stream>_<level>.jsonl`

A multi-turn conversation between the learner and the tutor. Used as
**training-time imitation targets**. Six streams exist:

| Stream prefix         | Teaches the model to ...                                                   |
| --------------------- | -------------------------------------------------------------------------- |
| `normal_*`            | Hold a natural, register-correct conversation                              |
| `redirect_*`          | Pivot away from a politics/religion/alcohol/partisan-history probe         |
| `locale_redirect_*`   | Handle when the learner names a wrong-locale entity (e.g. American food)   |
| `pedagogy_redirect_*` | Reframe a "teach me grammar rules" request as a conversation               |
| `language_redirect_*` | Stay in English when the learner switches to L1 or asks for L1            |
| `persona_redirect_*`  | Stay in character when asked "are you AI?" / "what model are you?"        |

```json
{
  "id": "sft_seed_A2_037_v3",
  "metadata": {"cefr_level": "A2", "scenario_type": "normal", "source_seed_id": "seed_A2_037"},
  "system_prompt": "<deployment system prompt with country baked in>",
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
`rejected`. The pipeline generates **8 axes** of rejection:

| Axis                 | Rejected turn is ...                                                 |
| -------------------- | -------------------------------------------------------------------- |
| `register_unnatural` | textbook-stiff, formal-where-it-shouldn't-be                         |
| `cefr_mismatch`      | one band too hard or two bands too easy for the learner              |
| `locale_violation`   | grounded in the wrong culture (American food, European city, ...)    |
| `pedagogy_weak`      | a lecture or grammar-rule monologue instead of a conversational turn |
| `accuracy_error`     | introduces a grammatical error (wrong tense/agreement/article)       |
| `off_topic`          | drifts off the scenario topic                                        |
| `language_violation` | switches into the learner's L1 / asks them to switch                 |
| `persona_break`      | acknowledges being an AI / names its model / breaks the in-character framing |

Axes are assigned **quota-balanced** across SFT examples so every axis
gets coverage even when individual SFT examples can't support every
spoil. A per-axis similarity threshold rejects near-no-op rewrites.

```json
{
  "id": "dpo_sft_seed_A2_037_v3_off_topic",
  "metadata": {
    "source_sft_id": "sft_seed_A2_037_v3",
    "axis": "off_topic",
    "cefr_level": "A2"
  },
  "system_prompt": "<deployment system prompt>",
  "prompt_messages": [...prior turns...],
  "chosen": "Yes, three big ones — that one looks beautiful.",
  "rejected": "By the way, have you seen the new superhero movie? It's amazing."
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

QLoRA SFT on Qwen3-8B. Mixes filtered SFT examples (80%) with filtered
evaluation examples (20%) so the single adapter learns both modes. Key
hyperparameters in `config/training.yaml`:

```yaml
sft:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  max_seq_length: 4096
  data:
    mix_ratio_sft: 0.80
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
null` so the frozen base is the reference policy. The two DPO sources
are mixed by `mix_ratio_register: 0.70` / `mix_ratio_on_policy: 0.30` —
the larger pool is trimmed to that ratio so it doesn't dominate batches.

Output: `outputs/dpo/`.

### 12.6 `eval_final` + `compare`

Same evaluation as `eval_intermediate`, against the post-DPO adapter.
`compare` flags any metric that regressed beyond `comparison.threshold`
(default 0.05).

---

## 13. Config reference

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

The two **mix-ratio** knobs to tune for training balance:

- `training.yaml` → `sft.data.mix_ratio_sft` / `mix_ratio_eval` —
  conversation vs evaluation in SFT.
- `training.yaml` → `dpo.data.mix_ratio_register` / `mix_ratio_on_policy`
  — offline vs on-policy preference pairs in DPO.

---

## 14. Troubleshooting

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
