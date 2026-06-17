# 4. Experimental Setup

## 4.1 Hardware

All training is performed on a single consumer GPU: NVIDIA RTX 3060
with 12 GB VRAM. The 9B teacher is served separately by llama.cpp's
`llama-server` from a Q4_K_XL-quantised GGUF file on the same
machine; training and teacher inference cannot run concurrently
because their VRAM footprints (≈7.7 GB and ≈4 GB respectively) sum
to more than the device capacity. The pipeline orchestrator stops
the teacher server during training phases and restarts it for
on-policy DPO generation and the evaluation phase.

## 4.2 Base model

The student base model is **Qwen3.5-0.8B-Base**, a 0.8B-parameter
multimodal foundation model from the Qwen3.5 family with
architecture `Qwen3_5ForConditionalGeneration` and a hybrid
linear-attention / full-attention layer interleave. Training uses
only the language tower; the vision tower is frozen and never sees
the text-only training data.

The teacher model used for data generation is
**Qwen3.5-9B-UD-Q4_K_XL**, a 9B-parameter Qwen3.5 checkpoint
quantised to 4-bit and served via llama.cpp. The teacher is
served at `http://192.168.135.32:8080/v1` with chat completion
context length 32 768 and 80 GPU layers offloaded.

For reference and as zero-shot baselines, we use two additional
off-the-shelf Qwen3.5 checkpoints in their post-trained form:
**Qwen3.5-0.8B** and **Qwen3.5-4B**.

## 4.3 Training recipe

**SFT.** QLoRA with NF4 4-bit quantisation, double-quantisation
enabled, `bfloat16` compute dtype. LoRA rank 16, alpha 32, dropout
0.05; target modules cover all seven linear projections in the
attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP
(`gate_proj`, `up_proj`, `down_proj`) blocks of the language tower.
SFT trains for 2 epochs at peak learning rate $2 \times 10^{-4}$
with cosine decay, batch size 1, gradient accumulation 8 (effective
batch 8), max sequence length 1024 tokens. Optimizer:
`paged_adamw_8bit`. Gradient checkpointing is enabled. We use the
SDPA attention implementation (Flash-Attention-2 is not safe under
the Qwen3.5 linear/full layer interleave).

**DPO.** DPO trains on top of the SFT adapter for 1 epoch at peak
learning rate $5 \times 10^{-6}$, cosine decay, $\beta = 0.1$,
sigmoid loss, max length 1024, max prompt length 512, identical
batch and quantisation settings to SFT. The reference policy is the
frozen base model (`ref_model: null`).

**Mix.** SFT training data combines the 12-stream filtered SFT
corpus with the `<think>`-mode evaluator examples; the loader takes
all available filtered records (`use_all_data: true`). DPO training
data combines three pools — register-DPO pairs (~65%), on-policy
DPO pairs (~25%), and sentinel-DPO pairs (~10%) — with the
identically-named `mix_ratio_*` knobs and `use_all_data: true` by
default (see §3.8 for the sentinel pool's offline and on-policy
construction modes).

## 4.4 Training data composition

Table 1 (TODO) summarises the 12-stream SFT corpus produced by the
pipeline. After filter cascade and yield-aware top-up, the corpus
contains approximately 3 100 dialogues split across the streams and
levels. Per-stream and per-level counts and the
`<stream>_target_ratio` / `<stream>_target_per_level` knobs that
shaped them are reported in §5.

## 4.5 Held-out evaluation sets

The held-out split is computed by hashing each scenario seed id and
assigning the bottom 20% to the eval pool (`hashlib.sha256(seed_id)
[:8]` interpreted as integer, modulo 100). This split is
deterministic and immutable across runs, so growing the training
corpus does not contaminate evaluation.

We evaluate on four held-out sets (Table 2, TODO):

- **Tutor-Scenario (N=224)**: cold-start dialogues. Each baseline
  receives the system prompt rendered from the held-out seed and a
  short synthetic learner-opener turn (deterministically derived
  from the seed's user_role.name and topic) and generates an
  assistant turn. Tests pedagogical quality and CEFR adherence.

- **Redirect-Probe (N=143)**: partial dialogues from held-out
  redirect-stream records, ending in the user's violation turn. The
  baseline must produce the redirect response. Each record is
  tagged with its violation axis (one of seven). Tests redirect-
  axis classification accuracy on the produced response.

- **Persistent-Probe (N≥11, target ~120 after persistent_topup
  completes)**: partial dialogues ending one turn before the
  sentinel-firing turn. The baseline must produce the sentinel-
  firing response. Each record is tagged with the expected
  sentinel-firing turn (one of {5, 7, 9, 11}). Tests sentinel
  precision and recall.

- **Locale-Leakage (N=224)**: same cold-start scenarios as
  Tutor-Scenario, but the metric measures Western-default leakage in
  the produced response.

All four test sets are filtered to `locale=china`. Construction
script: `scripts/build_eval_sets.py`. Manifest:
`eval_sets/_split_manifest.json`.

## 4.6 Baseline matrix

We compare nine conditions. **Trained ablations** train the same
base model (`Qwen3.5-0.8B-Base`) with the same training recipe but
on different data:

| Tag | Data | Method | Purpose |
| --- | --- | --- | --- |
| A1 | All 12 streams + DPO | SFT + DPO | Full system |
| A2 | All 12 streams | SFT only | Isolates DPO contribution |
| A3 | 12 streams minus 6 specialized redirects | SFT + DPO | Isolates the redirect-taxonomy contribution |
| A4 | 12 streams minus 4 persistent streams | SFT + DPO | Isolates the persistent-stream contribution |

**Zero-shot baselines** apply a tutor-style system prompt to an
off-the-shelf checkpoint:

| Tag | Checkpoint | Purpose |
| --- | --- | --- |
| B1 | Qwen3.5-0.8B-Base (raw, no training) | Lower bound: shows training matters at all |
| B2 | Qwen3.5-0.8B post-trained | Same-size off-the-shelf comparison |
| B3 | Qwen3.5-4B post-trained | Larger same-family comparison |
| B4 | Qwen3.5-9B (4-bit, via llama-server) | Distillation upper bound + judge ensemble member |

A2 is implemented by training the A1 SFT adapter and using it
directly at inference time without the subsequent DPO step. The
remaining ablations train a separate SFT and DPO adapter pair per
condition.

## 4.7 Multi-judge evaluation protocol

Quality metrics that require a judge (CEFR-adherence, redirect-axis
F1 on the produced response, naturalness) are scored by an ensemble
of three judges:

- Qwen3.5-9B (the same checkpoint used as the teacher)
- Qwen3.5-4B post-trained
- Qwen3.5-0.8B post-trained

Per metric, each judge produces a 1–5 score; we report the median
across judges. To control for the **self-preference bias**
[@panickssery2024selfpreference] of using the teacher to judge its
own student, we
additionally report metric-by-metric inter-judge agreement
(Krippendorff's $\alpha$); judge ensembles whose $\alpha < 0.6$ on a
metric are flagged in the results table. Sentinel firing and
locale-leakage are **mechanical** metrics that do not use a judge
(see §4.8 and §4.9).

We validate the judge ensemble against 100 randomly-sampled
generations that the first author hand-judges on a 1–5 scale for
each metric. We report Pearson correlation between ensemble median
and human gold; judge ensembles whose human-correlation falls below
$\rho = 0.6$ on any metric are flagged. (Numbers in §5.)

## 4.8 Sentinel-firing metric (Persistent-Probe)

On Persistent-Probe, the baseline produces the assistant turn that
follows the third strike. We score sentinel firing mechanically:

- **Precision**: of baseline-produced turns that fire the sentinel,
  what fraction match the expected sentinel-firing turn position?
- **Recall**: of test records whose expected sentinel turn position
  matches the produced turn position, what fraction fire the
  sentinel?
- **F1**: harmonic mean.

Sentinel firing is detected by matching against a small fixed set of
sentinel markers (`[SESSION_END]`, `[ENDED_BY_TUTOR]`,
`ending this session`, etc.).

## 4.9 Locale-leakage rate (Locale-Leakage)

On Locale-Leakage, the baseline produces a tutor turn given a
cold-start china-locale scenario. The metric is the rate at which
the produced response contains a Western-default entity from a
fixed gazetteer (`config/western_entities.yaml`, ~200 entries
covering brands like Costco/Walmart, holidays like Thanksgiving,
US/UK place names, US/UK food items, and so on). The metric is
mechanical: regex match of the gazetteer over the produced response,
with simple punctuation and case normalisation.

## 4.10 Statistical reporting

We train one seed of the main condition (A1) due to compute
constraints. Ablations A3 and A4 likewise use one seed. We report
exact mechanical metrics (sentinel firing, locale leakage) as point
estimates; judged metrics (CEFR adherence, redirect F1,
naturalness) are reported with bootstrap 95% confidence intervals
over 1000 resamples of the test set. This is a deliberate
limitation: a more rigorous reporting would use three training
seeds per condition. We discuss this in §6 and treat it as a
limitation rather than a flaw in the methodology.

## 4.11 Reproducibility

The complete pipeline is one Python codebase. Generation is driven
by `config/generation.yaml`; training by per-condition YAMLs under
`config/paper/`. Held-out sets are produced by
`scripts/build_eval_sets.py` and frozen in
`eval_sets/`. The teacher model name embedded in record metadata is
auto-detected from the teacher's `/v1/models` endpoint at startup so
that records carry the actual served-model name and not a stale
config value. The entire run from seeds to evaluation is resumable.
