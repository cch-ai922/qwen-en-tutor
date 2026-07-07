# A. Reproducibility and Hyperparameters

**Training.** QLoRA [@dettmers2023qlora] (NF4 4-bit, double-quant,
`bfloat16`) with LoRA [@hu2022lora] rank 16, alpha 32, dropout 0.05 over the
seven linear projections of the language tower; 2 epochs, peak LR
$2\times10^{-4}$ cosine, effective batch 8, max sequence length 1792,
`paged_adamw_8bit`, SDPA attention. No DPO enters any condition. Teacher
served via llama.cpp (context 32 768, 80 GPU layers); teacher and trainer are
swapped since they cannot co-reside on 12 GB.

**Persistent-variant codomain.** Sentinel positions {5,7,9,11} are forced:
dialogues open on a learner turn (only odd positions valid), with $\geq 5$ to
fit three strikes and two probes and $\leq 11$ to stay under the 1792-token
cap. Variant assignment `int(sha256(seed_id)[:8]) % 4` is seeded (not random)
so re-generation is stable and the train/eval split stays coherent.

**Pipeline filters.** Six filters run cheap-to-expensive: five mechanical
(script, banned terms, schema, naturalness) then one LLM-judge (`locale_judge`).
An audit of the `locale_judge` found ~85% false positives on its rejections
(common English sentence-initial words, locally-canonical landmarks),
remediable with static allowlists (global pass rate 70.1% $\to$ 88.4%) — a
reusable caution for any capitalization-based entity filter.

**Artifacts.** One Python codebase; generation driven by
`config/generation.yaml`, training by per-condition YAMLs, the six held-out
sets frozen by `scripts/build_eval_sets.py`. The full-length version of this
paper reports the per-capability statistics, the F1-bimodality negative
result, and the complete confound analysis.

**Code and Data Availability.** All code, training configurations, seeds, frozen
evaluation sets, judge prompts, synthetic datasets, and per-condition score
outputs are released at <https://github.com/cch-ai922/tutor-train>, with a
`reproducibility/` guide mapping each result to the script and config that
produce it. Trained adapters are deltas over the public base models
(Qwen3.5-0.8B-Base; Llama-3.2-1B-Instruct) and are available on request.
