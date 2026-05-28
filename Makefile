# qwen-en-tutor: shortcut targets for the common workflows.
#
# Works under POSIX make (Linux/macOS) and under git-bash / WSL on Windows.
# For native PowerShell, run the python commands directly — every target is
# a one-liner.

PYTHON ?= python
UV     ?= uv
PIPELINE_CONFIG       ?= config/pipeline.yaml
PIPELINE_SMOKE_CONFIG ?= config/pipeline_smoke.yaml
GENERATION_CONFIG     ?= config/generation.yaml
TRAINING_CONFIG       ?= config/training.yaml

# Comma-separated default — overridden per target where it makes sense.
LEVELS ?= A1,A2,B1,B2,C1,C2

.PHONY: help install spacy test smoke-test dry-run \
        seeds sft redirect eval-gen dpo-gen \
        filter-sft filter-eval filter-dpo \
        generate filter \
        train-sft train-dpo merge \
        eval compare full-pipeline \
        sample-holdout cost \
        clean clean-data clean-outputs

help:
	@echo "qwen-en-tutor make targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install         uv sync (creates .venv, installs deps)"
	@echo "    spacy           download the spaCy en_core_web_sm model"
	@echo "    test            run pytest (excludes slow tests by default)"
	@echo ""
	@echo "  Smoke / dev:"
	@echo "    dry-run         end-to-end pipeline with synthetic data (no API/GPU)"
	@echo "    smoke-test      30-minute real smoke run via config/pipeline_smoke.yaml"
	@echo ""
	@echo "  Generation:"
	@echo "    seeds | sft | redirect | eval-gen | dpo-gen"
	@echo "    generate        all five generation stages in order"
	@echo ""
	@echo "  Filtering:"
	@echo "    filter-sft | filter-eval | filter-dpo"
	@echo "    filter          all three filter stages"
	@echo ""
	@echo "  Training + eval:"
	@echo "    train-sft       SFT LoRA training"
	@echo "    train-dpo       DPO from the SFT adapter"
	@echo "    merge           merge LoRA into base model for export"
	@echo "    eval            full holdout evaluation"
	@echo "    compare BASELINE=... CANDIDATE=...  compare two eval runs"
	@echo ""
	@echo "  Utility:"
	@echo "    sample-holdout  carve a holdout set out of data/seeds"
	@echo "    cost            print forward token + USD cost estimate"
	@echo "    clean-data      remove generated data/ files (preserves .gitkeep)"
	@echo "    clean-outputs   remove training output dirs"
	@echo "    full-pipeline   orchestrator over config/pipeline.yaml"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:
	$(UV) sync

spacy:
	$(PYTHON) -m spacy download en_core_web_sm

test:
	$(PYTHON) -m pytest tests/ -q -m "not slow"

# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

dry-run:
	$(PYTHON) scripts/run_full_pipeline.py --config $(PIPELINE_SMOKE_CONFIG) --dry-run

smoke-test:
	$(PYTHON) scripts/run_full_pipeline.py --config $(PIPELINE_SMOKE_CONFIG)

full-pipeline:
	$(PYTHON) scripts/run_full_pipeline.py --config $(PIPELINE_CONFIG)

# ---------------------------------------------------------------------------
# Per-stage generation
# ---------------------------------------------------------------------------

seeds:
	$(PYTHON) scripts/generate_all.py --stages seeds --levels $(LEVELS)

sft:
	$(PYTHON) scripts/generate_all.py --stages sft --levels $(LEVELS)

redirect:
	$(PYTHON) scripts/generate_all.py --stages redirect --levels $(LEVELS)

eval-gen:
	$(PYTHON) scripts/generate_all.py --stages evaluation --levels $(LEVELS)

dpo-gen:
	$(PYTHON) scripts/generate_all.py --stages register --levels $(LEVELS)

generate:
	$(PYTHON) scripts/generate_all.py --levels $(LEVELS)

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

filter-sft:
	$(PYTHON) scripts/filter_data.py sft --levels $(LEVELS)

filter-eval:
	$(PYTHON) scripts/filter_data.py eval --levels $(LEVELS)

filter-dpo:
	$(PYTHON) scripts/filter_data.py dpo --levels $(LEVELS)

filter: filter-sft filter-eval filter-dpo

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

train-sft:
	$(PYTHON) scripts/train_sft.py --config $(TRAINING_CONFIG)

train-dpo:
	$(PYTHON) scripts/train_dpo.py --config $(TRAINING_CONFIG)

merge:
	@echo "Usage: PYTHON='python -c \"from qwen_tutor.training.merge import merge_lora; merge_lora(...)\"' make merge"
	@echo "(merge takes paths — see scripts/run_full_pipeline.py for orchestration)"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

ADAPTER ?= outputs/dpo
N_PER_LEVEL ?= 10

eval:
	$(PYTHON) scripts/run_eval.py --adapter-path $(ADAPTER) --n-per-level $(N_PER_LEVEL) --levels $(LEVELS)

compare:
	@if [ -z "$(BASELINE)" ] || [ -z "$(CANDIDATE)" ]; then \
	    echo "usage: make compare BASELINE=data/eval_results/X CANDIDATE=data/eval_results/Y"; \
	    exit 2; \
	fi
	$(PYTHON) scripts/compare_runs.py $(BASELINE) $(CANDIDATE) --threshold 0.05

# ---------------------------------------------------------------------------
# Holdout sampling
# ---------------------------------------------------------------------------

HOLDOUT_PER_LEVEL ?= 10

sample-holdout:
	$(PYTHON) scripts/sample_holdout.py --n-per-level $(HOLDOUT_PER_LEVEL) --levels $(LEVELS)

# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

N_FOR_COST ?= 100

cost:
	$(PYTHON) scripts/cost_estimate.py --n-per-level $(N_FOR_COST)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean-data:
	@echo "removing generated data files (keeping .gitkeep)..."
	@find data -type f ! -name '.gitkeep' -delete
	@echo "done."

clean-outputs:
	rm -rf outputs/sft outputs/dpo outputs/smoke_sft outputs/smoke_dpo

clean: clean-data clean-outputs
	rm -rf data/_pipeline_state
