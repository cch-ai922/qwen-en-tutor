# Appendix A. Reproducibility

<!-- paper_v3 — "Don't Trim the Tail" -->

Every result in the paper is produced by a named script over generations already
on disk (no result depends on unseen inference). Base model: Qwen3.5-0.8B-Base,
QLoRA SFT, 1-epoch matched budget unless noted; the off-family replication
(§5.5b) uses Llama-3.2-1B-Base. The table below maps each reported result to the
script that computes it and the JSON it writes.

| Result (section) | Script | Output |
|------------------|--------|--------|
| Trim × (position×marker) premature rates (§5.1) | `scripts/score_phase0_attribution.py` | `outputs/paper_v2/score/phase0_attribution.json` |
| A1 positive-probe recall gate, 1-epoch (§5.1) | `scripts/run_paper_eval.py --baseline v3_a1_{untrim,trim} --test-set persistent_probe` | `outputs/paper_v3/eval_recall_check/v3_a1_{untrim,trim}/persistent_probe.jsonl` |
| Significance: Fisher / McNemar / Wilson CIs + effect sizes (odds ratio, Cohen's *h*) (§5.1) | `scripts/score_phase0_significance.py` (stdlib only) | `outputs/paper_v2/score/phase0_significance.json` |
| Seed-7 replication (§5.1) | `scripts/run_multiseed_s7.py` → `scripts/score_multiseed_s7.py` | `outputs/paper_v2/score/phase0_multiseed_s7.json` |
| vc-stratified premature (§5.2) | `scripts/score_phase0_attribution.py` | `outputs/paper_v2/score/phase0_attribution.json` |
| Logit-level P(sentinel) vs depth (§6.1) | `scripts/score_sentinel_logprob_vs_depth.py` | `outputs/paper_v3/score/sentinel_logprob_vs_depth.json` |
| Single-axis typed attribution / semantic gate (§5.3) | `scripts/score_phase0_attribution.py` | `outputs/paper_v2/score/phase0_attribution.json` |
| Attribution under distraction / mixed-violation probe (§5.4) | `scripts/score_mixed_violation_probe.py` | `outputs/paper_v3/score/mixed_violation.json` |
| Count-marker (Design B) construction (§5.5) | `scripts/convert_typed_to_count.py`, `scripts/assemble_a8_count.py` | `data/sft_filtered_a8_count*` |
| Off-family Llama replication (§5.5b) | `scripts/phase3_synthetic_trim.py` | `outputs/paper_v3/phase3/phase3_result.json` |
| Untrimmed A5/A6/A7 reconstruction (§4.4, §6.6) | `scripts/build_untrim_a5a6a7.py` | `outputs/paper_v2/phase0_trim_study/` |

The firing detector used throughout is the axis-capturing regex
`\[SESSION_END(?:\s*:\s*(?:STRIKE=\d+\s*:\s*)?(persistent_[a-zA-Z_]+))?\s*\]`,
which matches generic, typed, and count-annotated marker forms and captures the
axis label for attribution. Code, configs, and the synthetic datasets are in the
released repository (see Code and Data Availability); model weights and large
training outputs are excluded.

**Code and Data Availability.** All code, per-condition training configurations,
seeds, frozen evaluation sets, judge prompts, the synthetic datasets, and the
score outputs underlying every table are released at
<https://github.com/cch-ai922/tutor-train>. A `reproducibility/` guide maps each
result to the script, config, and expected number that produce it. The off-family
replication (§5.5b) uses Llama-3.2-1B-Base; trained LoRA adapters are low-rank
deltas over the public base models and are available from the authors on request.

# Appendix B. Supporting tables

**Table B.1 — Premature firing stratified by violation count (§5.2).** vc = number
of prior same-axis sub-threshold strikes. In every trimmed cell firing jumps
sharply from vc=1 to vc=2; untrimmed cells stay low at both — the threshold-laxity
signature.

| Cell | variant | vc=1 | vc=2 |
|------|---------|------|------|
| A1 | untrimmed | 0.063 | 0.352 |
| A1 | trimmed   | 0.654 | 0.912 |
| A5 | untrimmed | 0.031 | 0.207 |
| A5 | trimmed   | 0.384 | 0.736 |
| A6 | untrimmed | 0.050 | 0.264 |
| A6 | trimmed   | 0.497 | 0.786 |
| A7 | untrimmed | 0.069 | 0.358 |
| A7 | trimmed   | 0.497 | 0.887 |

**Table B.2 — Attribution under distraction (§5.4).** `fire_correct` records
(primary axis X at the third strike, distractor axis Y sub-threshold), typed cells
only (n=141 each). The gate names the threshold axis X on 0.96–0.98 of fires and is
pulled to the distractor under 2%.

| Cell | marker | fire_rate (X at 3) | correct-axis (X) | wrong-axis (Y) | no-axis fires |
|------|--------|--------------------|------------------|----------------|---------------|
| A1 | typed | 0.837 | **0.975** | 0.009 | 0 |
| A5 | typed | 0.766 | **0.963** | 0.018 | 0 |
