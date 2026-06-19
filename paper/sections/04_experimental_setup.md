# 4. Experimental Setup

## 4.1 Hardware

All training is performed on a single consumer GPU: NVIDIA RTX 3060
with 12 GB VRAM. The 9B teacher is served separately by llama.cpp's
`llama-server` from a Q4_K_XL-quantised GGUF file on the same
machine; training and teacher inference cannot run concurrently
because their VRAM footprints ($\approx$7.7 GB and $\approx$4 GB respectively) sum
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
data combines three pools and uses `use_all_data: true`, so the
realised mix is the natural ratio of the on-disk pools rather than
any nominal target:

- **register pool**: ~3 700 pairs (~76% of total). Offline
  preference pairs from a single teacher call per filtered SFT
  record.
- **on-policy pool**: ~400 pairs (~8% of total). Pairs where the
  rejected response is produced by the SFT-trained student under
  the same prompt as the teacher's preferred response, gated by an
  LLM-judge margin of $\geq 2$ on a 1--5 scale. We use
  `max_per_level=100` (six CEFR levels) as the per-level attempt
  cap, then filter to the kept set.
- **sentinel pool**: ~760 pairs (~16% of total). The offline-mode
  subset (~88%) is produced by deterministic strip of the
  `[SESSION_END: <axis>]` marker from the teacher's third-strike
  response; the on-policy-mode subset (~12%) is the SFT-trained
  student's regeneration of the same turn. See §3.8.

The `mix_ratio_register`, `mix_ratio_on_policy`, and
`mix_ratio_sentinel` knobs in the training config are honoured only
when `use_all_data: false`. Our on-policy share (8%) and sentinel
share (16%) place the mix in the *hybrid* regime used by Tulu-3
[@lambert2024tulu3] and Llama-3 [@touvron2024llama3]: a dominant
offline pool plus targeted on-policy / specialised pools that
account for ~10--30% of the gradient signal.

## 4.4 Training data composition

Table 4 (TODO) summarises the 12-stream SFT corpus produced by the
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

We evaluate on six held-out sets (Table 5, TODO):

- **Tutor-Scenario (N=224)**: cold-start dialogues. Each baseline
  receives the system prompt rendered from the held-out seed and a
  short synthetic learner-opener turn (deterministically derived
  from the seed's user_role.name and topic) and generates an
  assistant turn. Tests pedagogical quality and CEFR adherence.

- **Redirect-Probe (N=143)**: partial dialogues from held-out
  redirect-stream records, ending in the user's violation turn. The
  baseline must produce the redirect response. Each record is tagged
  with its violation axis (one of seven). Under the invariant framing
  of §3.3, the quantity of interest is not merely *which* axis the
  model recognises but whether the produced response carries the
  axis-appropriate **repair shape** (e.g. scaffolding for a pedagogy
  violation, acknowledge-and-steer for a code-switch). We therefore
  classify the *produced response* by the repair shape it exhibits
  and score against the expected axis; response-axis classification
  is thus a proxy for repair-shape correctness rather than for
  intent detection. We note this proxy explicitly because a model
  could in principle name the right axis while producing the wrong
  repair; §5.4 reports per-axis results so this can be inspected.

- **Persistent-Probe (N$\geq$11, target ~120 after persistent_topup
  completes)**: partial dialogues ending one turn before the
  sentinel-firing turn, drawn from the 4-variant persistent records.
  The baseline must produce the sentinel-firing response. Every
  record here is a *positive* — three same-axis strikes have
  occurred, so the sentinel should fire — and each is tagged with its
  expected sentinel turn (one of {5, 7, 9, 11}). This set measures
  **recall** (and the position-stratified fire-rate of §4.8).

- **Persistent-FP-Probe (target ~120)**: *negative* controls —
  benign dialogues that reach a trained sentinel position (5/7/9/11)
  *without* accumulating three same-axis strikes (e.g. normal
  scaffolding, or one or two isolated strikes that the learner then
  abandons). The sentinel should **not** fire. This set supplies the
  negatives needed to measure false-positive rate and precision, and
  it is the probe on which a turn-position shortcut and a
  trigger-detector visibly diverge: the shortcut fires on the benign
  trained position, the trigger-detector stays silent.

- **Persistent-OffPosition-Probe (target ~80)**: positives whose
  third strike lands at a turn *outside* the trained set {5, 7, 9,
  11} (e.g. 6 by inserting a single extra scaffolding turn, or 13 by
  extending lead-in). The sentinel should fire. A trigger-detector
  fires here; a model that memorised the four trained positions does
  not. Together with Persistent-FP-Probe this is what makes the
  decorrelation claim falsifiable rather than merely consistent with
  the data (§4.8).

- **Locale-Leakage (N=224)**: same cold-start scenarios as
  Tutor-Scenario, but the metric measures Western-default leakage in
  the produced response.

All six test sets are filtered to `locale=china`. Construction
script: `scripts/build_eval_sets.py`. Manifest:
`eval_sets/_split_manifest.json`.

## 4.6 Baseline matrix

We compare ten conditions. **Trained ablations** train the same
base model (`Qwen3.5-0.8B-Base`) with the same training recipe but
on different data:

| Tag | Data | Method | Purpose |
| --- | --- | --- | --- |
| A1 | All 12 streams + DPO | SFT + DPO | Full system |
| A2 | All 12 streams | SFT only | Isolates DPO contribution |
| A3 | 12 streams minus 6 specialized redirects | SFT + DPO | Isolates the redirect-taxonomy contribution |
| A4 | 12 streams minus 4 persistent streams | SFT + DPO | Isolates *presence* of persistent streams |
| A5 | 12 streams, persistent streams present but all sentinels fixed at turn 7 (no 4-variant) | SFT + DPO | Isolates the trigger-position *decorrelation* contribution |

A4 and A5 isolate two different things and are easy to conflate. A4
removes the persistent streams entirely, so A1-vs-A4 answers "does
training on persistence help at all?" A5 *keeps* the persistent
streams but reverts them to the naive fixed-turn-7 design, so
A1-vs-A5 answers the question contribution&nbsp;2 actually claims:
"does decorrelating sentinel position from the third-strike trigger
help, holding persistence training fixed?" Without A5 the
decorrelation claim is supported only indirectly; A5 is the condition
expected to *exhibit* the positional shortcut (high fire-rate at the
benign turn-7 case on Persistent-FP-Probe, low firing on
Persistent-OffPosition-Probe).

**Zero-shot baselines** apply a tutor-style system prompt to an
off-the-shelf checkpoint:

| Tag | Checkpoint | Purpose |
| --- | --- | --- |
| B1 | Qwen3.5-0.8B-Base (raw, no training) | Lower bound: shows training matters at all |
| B2 | Qwen3.5-0.8B post-trained | Same-size off-the-shelf comparison |
| B3 | Qwen3.5-4B post-trained | Larger same-family comparison |
| B4 | Qwen3.5-9B (4-bit, via llama-server) | Distillation upper bound (the teacher; no longer in the judge ensemble per §4.7) |

A2 is implemented by training the A1 SFT adapter and using it
directly at inference time without the subsequent DPO step. The
remaining ablations train a separate SFT and DPO adapter pair per
condition.

## 4.7 Multi-judge evaluation protocol

Quality metrics that require a judge (CEFR-adherence, redirect-axis
F1 on the produced response — i.e. whether the response exhibits the
axis-appropriate repair shape per §3.3, not merely intent
recognition — and naturalness) are scored by a **cross-family**
ensemble of three judges, each drawn from a model family **distinct
from the teacher's**:

- **Prometheus-7B-v2** (Mistral lineage) — purpose-built rubric
  evaluator [@kim2024prometheus]. Scalar 1–5 metrics use Prometheus's
  native rubric protocol (task description + response + score rubric
  $\to$ `Feedback: ... [RESULT] N`).
- **Llama-3.1-8B-Instruct** (Meta) [@touvron2024llama3].
- **Gemma-2-9B-it** (Google) [@gemmateam2024gemma2].

We deliberately exclude any Qwen-family judge from this ensemble to
**eliminate the self-preference bias**
[@panickssery2024selfpreference] inherent in using the teacher's own
family to score its student. Per metric, each judge produces a 1–5
score (or a categorical label, for redirect-axis); we report the
median across judges. We additionally report metric-by-metric
inter-judge agreement (Krippendorff's $\alpha$); judge ensembles
whose $\alpha < 0.6$ on a metric are flagged in the results table.
On 12 GB VRAM the three judges cannot coexist in memory, so judging
is run sequentially — load Prometheus, score every record, swap to
Llama-3.1, repeat, then Gemma-2 — adding ~3h to the eval pass.
Sentinel firing and locale-leakage are **mechanical** metrics that
do not use a judge (see §4.8 and §4.9).

We validate the judge ensemble against 100 randomly-sampled
generations that the first author hand-judges on a 1–5 scale for
each metric. We report Pearson correlation between ensemble median
and human gold; judge ensembles whose human-correlation falls below
$\rho = 0.6$ on any metric are flagged. (Numbers in §5.)

## 4.8 Sentinel-firing metric (Persistent / FP / OffPosition probes)

Sentinel firing is detected mechanically by matching the produced
assistant turn against a small fixed set of sentinel markers
(`[SESSION_END]`, `[ENDED_BY_TUTOR]`, `ending this session`, etc.).
The scored quantity is binary: did the produced turn fire a sentinel
or not? We compute precision and recall over the union of the
positive and negative probe sets defined in §4.5, which is what makes
"precision" meaningful — a metric scored on positives alone cannot
have a false-positive denominator.

- **Recall** (Persistent-Probe, all positives): of records where the
  sentinel *should* fire (three same-axis strikes have occurred),
  the fraction on which the model fires it. This is the true-positive
  rate.
- **Precision** (Persistent-Probe $\cup$ Persistent-FP-Probe): of all
  records on which the model fires the sentinel, the fraction where
  it *should* have fired. The Persistent-FP-Probe negatives are the
  only source of false positives, so precision is undefined without
  them.
- **False-positive rate** (Persistent-FP-Probe, all negatives): of
  records where the sentinel should *not* fire, the fraction on which
  the model fires it anyway. Reported separately because it is the
  sharpest single discriminator between a trigger-detector and a
  turn-position shortcut.
- **F1**: harmonic mean of precision and recall.
- **Fire-rate by expected sentinel position** (diagnostic): the
  positive fire-rate stratified by the record's sentinel position
  $\in$ {5, 7, 9, 11}, giving four per-position rates. A model that
  learned "third strike on the same axis" should fire approximately
  *uniformly* across the four positions; a model that collapsed onto
  one or two trained positions would fire non-uniformly. **This
  uniformity is necessary but not sufficient**: a model that
  memorised all four trained positions also fires uniformly on these
  in-distribution positives. We therefore do not rest the
  decorrelation claim on uniformity alone — the discriminating
  evidence is the Persistent-FP-Probe false-positive rate (a shortcut
  fires on a benign turn-5/7/9/11 utterance; a trigger-detector does
  not) and firing on Persistent-OffPosition-Probe (a trigger-detector
  fires when the third strike lands off the trained grid; a
  position-memoriser does not). §5.3 reports all three together, with
  A5 (fixed-turn-7) as the condition expected to exhibit the
  shortcut.

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
constraints. Ablations A3, A4, and A5 likewise use one seed. We report
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
