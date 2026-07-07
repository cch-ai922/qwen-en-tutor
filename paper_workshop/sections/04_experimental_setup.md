# 4. Experimental Setup

**Models and recipe.** Student base `Qwen3.5-0.8B-Base`; teacher
`Qwen3.5-9B-UD-Q4_K_XL` (4-bit, llama.cpp), which also serves as the
strongest prompt-only baseline (B4). All experiments on one RTX 3060 12GB.
Every trained condition uses the *same* recipe — QLoRA [@dettmers2023qlora]
(NF4 4-bit) with LoRA [@hu2022lora] rank 16 / alpha 32, 2 epochs, no DPO —
so each ablation is single-variable (hyperparameters in the appendix).

**Conditions.** **A1** — the student, full 12-stream mix. **A3** — A1 minus
the six specialized single-shot redirect streams (the *generic-SFT*
baseline). **A5** — A1 with persistence rebuilt fixed-turn-7 (the
decorrelation contrast). Prompt-only: **B1** 0.8B base, **B2** 0.8B
instruct, **B3** 4B instruct, **B4** 9B teacher. All see the identical
deployment prompt (Figure~\ref{fig:design}).

**Held-out sets** (frozen, `locale=china`): Tutor-Scenario (224),
Redirect-Probe (143), and four persistence probes isolating recall
(Persistent-Probe, 159 positives) from two false-positive channels
(Persistent-FP-Probe benign negatives at trained positions;
Persistent-Premature-Probe under-threshold negatives) and off-grid firing
(Persistent-OffPosition-Probe).

**Metrics.** Persistence and locale leakage are **mechanical** (judge-free):
sentinel-firing recall against literal `[SESSION_END: <axis>]` strings, and
Western-default leakage via a word-boundary gazetteer. Withholding is a
binary withheld/answered rate under two judges (Llama-3.1-8B, Gemma-2-9B).
Repair *quality* on the promptable axes is a pairwise preference under a
cross-family three-judge ensemble (Prometheus / Llama-3.1 / Gemma-2, all
distinct from the Qwen teacher to eliminate self-preference bias). The
load-bearing A1/A3 conditions are reported over **three seeds** (42/123/7);
others single-seed.
