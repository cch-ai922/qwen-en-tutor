# Offline + local-model setup

How to run qwen-en-tutor without an internet connection, with a local
`gpt-oss-20b` teacher served by `llama.cpp` instead of the Claude /
OpenAI APIs. Tested on a Windows machine with an RTX 3060 (12 GB).

## Constraints + what runs where

| Stage | RTX 3060 (12 GB) | RTX 5090 (32 GB) |
|---|---|---|
| seeds → SFT → redirect → eval-gen → DPO register pairs | ✅ via local llama.cpp, slow | ✅ |
| filtering (incl. locale judge against local model) | ✅ | ✅ |
| training SFT (Qwen3 8B QLoRA) | ❌ won't fit alongside the 20B teacher; close it first | ✅ |
| training DPO | ❌ same | ✅ |
| holdout evaluation | ❌ trained model + teacher both want VRAM | ✅ |
| on-policy DPO data gen | ❌ needs trained SFT adapter loaded simultaneously | ✅ |

The 3060 profile (`config/pipeline_local.yaml`) only runs the
generation + filtering chain. Switch to `config/pipeline.yaml` (and the
non-local `generation.yaml`) on the 5090.

## End-to-end workflow

### 1. On a workstation WITH internet — bundle everything

```powershell
# Windows / PowerShell (Linux/macOS: bash equivalent)
cd qwen-en-tutor

# Ensure huggingface_hub is available for the model download step
python -m pip install huggingface_hub

# Pull every Python wheel, the spaCy model wheel, the Qwen3-8B base,
# and the gpt-oss-20b Q4_K_M GGUF into vendor/
python scripts/setup_offline.py
```

Expect ~30 GB on disk:
- `vendor/wheels/` — pip wheels (~3 GB once torch/bitsandbytes/etc are pulled)
- `vendor/models/Qwen3-8B/` — base model (~16 GB)
- `vendor/models/gpt-oss-20b-GGUF/` — Q4_K_M weight (~12 GB)
- `vendor/manifest.json` — what was downloaded + when

If you only need to refresh part of it: `--only wheels`, `--only models`,
or `--only spacy`. Override the gpt-oss source with
`--gpt-oss-repo other/repo --gpt-oss-pattern "*Q5_K_M*.gguf"`.

### 2. Get the prebuilt llama.cpp server

`scripts/setup_offline.py` does NOT bundle llama.cpp binaries (they're
platform-specific). On the connected machine:

1. Download a prebuilt release from
   https://github.com/ggerganov/llama.cpp/releases (look for the
   CUDA-12 build matching your driver).
2. Unzip it somewhere stable like `vendor/llama_cpp/`.
3. The launcher script will use `$env:LLAMA_CPP_BIN` (Windows) or
   `LLAMA_CPP_BIN` (Linux) to find `llama-server`, or fall back to
   `PATH`.

### 3. Zip + copy to the offline machine

Zip the whole project directory (including `vendor/` and the unzipped
`llama_cpp/` build) and copy it to the offline target via USB or LAN.

### 4. On the offline machine — install + verify

```powershell
# Windows / PowerShell
cd qwen-en-tutor
pwsh scripts/install_offline.ps1
```

Or on Linux/macOS:

```bash
bash scripts/install_offline.sh
```

The install script:

1. Creates `.venv/` in the project root.
2. `pip install --no-index --find-links vendor/wheels/` for every dep.
3. Installs `en_core_web_sm` from the vendored wheel.
4. Installs the project itself in editable mode.
5. Runs the fast test suite (95 tests) to confirm everything imports.

### 5. Start the local teacher

```powershell
# Point at the unzipped llama.cpp build, then launch:
$env:LLAMA_CPP_BIN = "$PWD\vendor\llama_cpp"
pwsh scripts/start_llama_cpp.ps1
```

You should see llama.cpp load the GGUF, allocate VRAM, and print

```text
HTTP server listening | hostname="127.0.0.1" port="8080"
```

Sanity-check from another shell:

```powershell
curl http://127.0.0.1:8080/v1/models
```

### 6. Run the local pipeline (gen + filter only)

```powershell
& .venv\Scripts\Activate.ps1
$env:QWEN_TUTOR_PROMPTS = "compact"
python scripts/run_full_pipeline.py --config config/pipeline_local.yaml
```

Expected:

- ~3 scenarios per level × 3 levels = 9 base scenarios.
- The 8 generation + filter stages should complete in 15-30 minutes
  total on a 3060 (most of it spent waiting on the 20B teacher).
- Outputs land under `data/seeds/`, `data/sft_raw/`,
  `data/sft_filtered/`, `data/eval_raw/`, `data/eval_filtered/`,
  `data/dpo_raw/`, `data/dpo_filtered/`.
- A markdown report is written to
  `data/_pipeline_state/<run_id>/report.md`.

### 7. (Future, on RTX 5090) — full pipeline

When you migrate to the 5090:

```powershell
# Same install + llama.cpp setup, except point QWEN_TUTOR_PROMPTS=full
# back to the Opus-grade prompts (gpt-oss-20b can handle them on a 5090).
$env:QWEN_TUTOR_PROMPTS = "full"
python scripts/run_full_pipeline.py --config config/pipeline.yaml
```

Training will now run; the orchestrator's `train_sft`,
`eval_intermediate`, `on_policy_gen`, `train_dpo`, `eval_final`, and
`compare` stages all activate. The 32 GB on a 5090 easily fits the
Qwen3-8B QLoRA training + the 20B teacher running simultaneously for
on-policy DPO regeneration.

## Troubleshooting

### `pip install` fails with "Could not find a version" during offline install

The vendor/wheels/ bundle was built on a different platform / Python
version. `pip download` only pulls wheels for the host platform. Re-run
`scripts/setup_offline.py` ON the offline machine's identical Python
build (same major.minor + same OS family), or use a build container.

### `llama-server` runs but the pipeline gets connection errors

Check that the base_url in `config/generation.local.yaml` matches the
port you started llama.cpp on (default `8080`). If you changed
`--port`, edit both `base_url` fields in the YAML.

### gpt-oss-20b output is malformed JSON

Two things tend to help on a smaller teacher:

1. Keep `QWEN_TUTOR_PROMPTS=compact` set — the compact prompts in
   `prompts_compact.py` are tuned for shorter context + stronger JSON
   discipline. The default Opus-grade prompts are too verbose for a
   20B model.
2. Drop `temperature` in `config/generation.local.yaml` to 0.4-0.5 —
   the compact prompts default to 0.7 which is on the high end for a
   20B model.

If JSON failures still dominate, inspect
`data/<stage>_failures.jsonl` for the actual model output and tighten
the corresponding prompt in `prompts_compact.py`.

### Locale-judge is timing out

The local judge calls the same 20B model — they're not free. For the
RTX 3060 smoke profile we leave `enable_locale_judge: true` because it
catches the most common drift (Western names slipping through), but if
your run is taking too long you can flip it to `false` in
`config/pipeline_local.yaml`. The mechanical banned_terms filter still
catches the obvious cases.

### llama.cpp OOM on a 12 GB card

Drop `--n-gpu-layers` from 99 down to 28-32. The remaining transformer
blocks will run on CPU. Generation slows from ~10 tok/s to ~3 tok/s,
but you avoid the crash. Or switch the GGUF to Q3_K_M (~9 GB).
