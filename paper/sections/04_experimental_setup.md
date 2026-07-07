# 4. Experimental Setup

## 4.1 Hardware, models, and training recipe

All experiments run on a single NVIDIA RTX 3060 (12 GB VRAM). The student
base is **Qwen3.5-0.8B-Base**; the teacher is **Qwen3.5-9B-UD-Q4_K_XL**
(4-bit, served via llama.cpp), which also serves as the strongest
prompt-only baseline (B4). Two off-the-shelf post-trained checkpoints,
**Qwen3.5-0.8B** and **Qwen3.5-4B**, are the B2/B3 baselines. Training and
teacher inference cannot co-reside on 12 GB, so the orchestrator swaps them.

Every trained condition uses the **same** recipe: QLoRA
[@dettmers2023qlora] (NF4 4-bit, `bfloat16` compute) with LoRA
[@hu2022lora] rank 16 / alpha 32 over the seven linear projections of the
language tower, 2 epochs, max sequence length 1792. **No DPO is applied to
any condition** — all of A1/A3/A5 are SFT-only (§4.4) — which keeps every
ablation single-variable and avoids a register-pool contamination confound.
Full hyperparameters and the (unused) DPO stage are in Appendix C.

## 4.2 Training data composition

Table~\ref{tab:sft-composition} reports per-stream, per-CEFR-level
record counts in the filtered SFT corpus. The corpus comprises
**3 374 dialogues** across 12 streams and six CEFR levels (A1--C2),
with `normal` dominant ($\approx$ 53\% of the total) and seven
redirect streams plus four persistent 3-strike streams covering the
remaining $\approx$ 47\%. The 12-stream layout instantiates the
invariant decomposition of \S3.3: one generic-redirect stream,
six specialized single-shot redirect streams (one per invariant
axis), and four persistent 3-strike streams (one per the four axes
that warrant escalation). C1/C2 counts in the six specialized
redirect streams are intentionally small ($\approx$\,7--10 records
per axis per level) because per-axis generation cost scales linearly
with axis count.

```{=latex}
\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}l r r r r r r r@{}}
\toprule
\textbf{Stream} & \textbf{A1} & \textbf{A2} & \textbf{B1} & \textbf{B2} & \textbf{C1} & \textbf{C2} & \textbf{Total} \\
\midrule
normal                          & 204 & 335 & 355 & 351 & 315 & 235 & 1\,795 \\
redirect (generic)              & 102 & 169 & 166 & 168 &  21 &  20 &   646 \\
\midrule
language\_redirect              &   5 &   8 &   7 &   9 &   2 &   5 &    36 \\
locale\_redirect                &   6 &   8 &   7 &  10 &   8 &   9 &    48 \\
pedagogy\_redirect              &   6 &   8 &   8 &   9 &   7 &   8 &    46 \\
persona\_redirect               &   6 &   8 &   7 &   9 &   7 &   9 &    46 \\
role\_swap\_redirect            &   6 &   8 &   7 &   9 &   7 &   7 &    44 \\
topic\_redirect                 &   6 &   6 &   7 &   9 &   6 &   7 &    41 \\
\midrule
persistent\_language\_violation &  29 &  38 &  26 &  30 &  20 &  22 &   165 \\
persistent\_off\_topic          &  22 &  34 &  30 &  24 &  10 &  17 &   137 \\
persistent\_persona\_break      &  28 &  42 &  27 &  31 &  30 &  48 &   206 \\
persistent\_role\_swap          &  17 &  44 &  36 &  25 &  20 &  22 &   164 \\
\midrule
\textbf{TOTAL}                  & \textbf{437} & \textbf{708} & \textbf{683} & \textbf{684} & \textbf{453} & \textbf{409} & \textbf{3\,374} \\
\bottomrule
\end{tabular}
\caption{\textbf{Filtered SFT corpus composition by stream and CEFR level.} Counts are post-filter passing records. The six specialized redirect streams (language, locale, pedagogy, persona, role\_swap, topic) carry $\approx$\,7--10 records per axis per CEFR level --- intentionally thinner than the generic streams, a known limitation we account for by leaning only on the large, seed-stable per-axis effects (\S5.6).}
\label{tab:sft-composition}
\end{table*}
```

## 4.3 Held-out evaluation sets

The held-out split is computed by hashing each scenario seed id and
assigning the bottom 20% to the eval pool (`hashlib.sha256(seed_id)
[:8]` interpreted as integer, modulo 100). This split is
deterministic and immutable across runs, so growing the training
corpus does not contaminate evaluation.

We evaluate on six held-out sets, summarized in
Table~\ref{tab:eval-sets} and detailed below (the table shows a seventh
row, Locale-Leakage, which reuses the Tutor-Scenario scenarios and so is
not counted as a distinct set):

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l r l@{}}
\toprule
\textbf{Eval set} & \textbf{N} & \textbf{Composition} \\
\midrule
Tutor-Scenario              & 224 & 6 CEFR levels (19--50 per level), no axes \\
Redirect-Probe              & 143 & 7 redirect axes, 6 CEFR levels \\
Persistent-Probe            & 159 & 4 persistence axes, 6 CEFR levels (positives) \\
Persistent-Premature-Probe  & 318 & under-threshold (vc=1,2) $\times$ turn-depth \\
Persistent-FP-Probe         & 240 & 4 trained positions \(\times\) 60 (40 per level) \\
Persistent-OffPosition-Probe &  60 & off-grid positions \{13, 15\}, 4 axes \\
Locale-Leakage              & 224 & same scenarios as Tutor-Scenario \\
\midrule
\textbf{TOTAL}              & \textbf{1\,144}$^{\ast}$ & \\
\bottomrule
\end{tabular}
\caption{\textbf{Held-out evaluation sets.} All sets are filtered to \texttt{locale=china} and constructed by \texttt{scripts/build\_eval\_sets.py} from the bottom-20\% hash-modulo split of scenario seed ids (\S4.3). $^{\ast}$TOTAL excludes the Locale-Leakage row, which reuses the same 224 cold-start scenarios as Tutor-Scenario (counting it would double-count); the remaining six rows sum to 1\,144. Persistent-Probe reports positives only (n=159, the recall denominator in \S5.3).}
\label{tab:eval-sets}
\end{table}
```

- **Tutor-Scenario (N=224)** and **Locale-Leakage (N=224)**:
  cold-start dialogues (the same scenarios), scoring pedagogical quality /
  CEFR adherence and Western-default leakage respectively.

- **Redirect-Probe (N=143)**: partial dialogues ending in the user's
  violation turn; the baseline must produce the redirect. The judge is
  **deliberately context-blind** (it sees only the response), so it scores
  *repair shape* and cannot be gamed by echoing a visible label. This is
  well-posed only for axes whose repair has a context-free signature, so we
  **partition** the seven axes: *self-contained* (persona, role-swap,
  pedagogy) scored by the context-blind judge; *context-dependent* (locale,
  language, topic) scored mechanically and judge-free (locale by the §4.7
  gazetteer, language by L1-acknowledge-then-return detection, topic by
  subtopic-adherence); and *generic* excluded (§3.3 catch-all). The two
  groups are reported separately, not macro-averaged (§5.4).

- **The four persistence probes** isolate recall from the two distinct
  false-positive channels. **Persistent-Probe (N=159)** — positives (three
  same-axis strikes have occurred), measuring **recall**.
  **Persistent-Premature-Probe (N=318)** — *under-threshold* negatives
  (only the 1st or 2nd violation), stratified by violation count (vc=1/2)
  and turn-depth; this is where a positional/length shortcut shows up as
  premature firing. **Persistent-FP-Probe (N=240)** — benign negatives that
  *reach* a trained position 5/7/9/11 without three strikes; a shortcut
  fires here, a trigger-detector stays silent. **Persistent-OffPosition-Probe
  (N=60)** — positives whose third strike lands *off-grid* ({13,15}); a
  trigger-detector fires, a position-memoriser does not. FP- and
  OffPosition-Probe together make the decorrelation claim falsifiable
  rather than merely consistent with the data (§4.6).

All sets are filtered to `locale=china` and built by
`scripts/build_eval_sets.py`.

## 4.4 Baseline matrix

We compare seven conditions — three trained ablations (A1, A3, A5) and
four prompt-only baselines (B1–B4). **Trained ablations** train the same
base model (`Qwen3.5-0.8B-Base`) with the same training recipe but
on different data:

```{=latex}
\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}c p{0.30\linewidth} l p{0.42\linewidth}@{}}
\toprule
\textbf{Tag} & \textbf{Data} & \textbf{Method} & \textbf{Purpose} \\
\midrule
A1     & All 12 streams, 4-variant persistent ($\{5,7,9,11\}$), axis-specific sentinel \texttt{[SESSION\_END: <axis>]} & SFT only & Full system (headline condition). \\
A3     & A1 minus the 6 specialized redirect streams (locale, pedagogy, language, persona, topic, role\_swap)  & SFT only   & \emph{Generic-SFT baseline}; §5.4 taxonomy contrast (A1 vs A3). \\
A5     & A1 with persistent rebuilt as fixed-turn-7, axis-specific sentinel \texttt{[SESSION\_END: <axis>]}      & SFT only   & §5.3.1 decorrelation contrast (A1 vs A5): the naive fixed-turn design. \\
\bottomrule
\end{tabular}
\caption{\textbf{Trained-ablation matrix.} All three share the same base (\texttt{Qwen3.5-0.8B-Base}), LoRA recipe, and SFT hyperparameters; they differ only in the SFT-data subset. None use DPO. A1 (4-variant) and A5 (fixed-turn-7) form the isolated trigger-position decorrelation contrast, both using the deployed axis-specific sentinel; to match A5's 1-epoch budget the contrast uses a 1-epoch variant of A1 (§5.3.1). A3 is the no-specialized-redirect baseline for the §5.4 taxonomy claim.}
\label{tab:trained-ablations}
\end{table*}
```

**A note on tags.** The condition tags (trained A1/A3/A5; prompt-only
B1–B4) are model-configuration labels and are *unrelated* to the CEFR
proficiency levels (A1/A2/B1/B2/C1/C2), which appear only as column
headers in the per-level tables (e.g. the training-data composition of
§4.2). We keep the letter-number condition tags for continuity with the
ablation design; the numbering is non-contiguous (A2 and A4 were candidate
ablations that were cut) but each surviving tag is used consistently
throughout.

Together with the prompt-only checkpoints below, these conditions span a
graded ladder of task-adaptation strength, all under the identical
deployment prompt (§3.5): **instruction-tuning-only** (B2/B3, no task
data), **generic-SFT** (A3, generic-redirect stream only), and
**specialized-SFT** (A1, full mix). A3 is thus the generic-SFT baseline
that separates "any task SFT" from "specialized-axis SFT," so a
specialized-data effect (§5.4, §5.6) is measured against a trained control,
not only against prompting. The two load-bearing contrasts are both
single-variable: **A1 vs A3** (taxonomy — do the specialized single-shot
streams add anything beyond the generic redirect?), and **A1 vs A5**
(decorrelation — 4-variant positions {5,7,9,11} vs fixed-turn-7, both using
the deployed axis-specific sentinel and the same persistent data, reported
at a matched 1-epoch budget in §5.3.1).

**Zero-shot baselines** (Table~\ref{tab:zeroshot-baselines}) apply a
tutor-style system prompt to an off-the-shelf checkpoint:

```{=latex}
\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}c l p{0.50\linewidth}@{}}
\toprule
\textbf{Tag} & \textbf{Checkpoint} & \textbf{Purpose} \\
\midrule
B1 & Qwen3.5-0.8B-Base (raw, no training) & Lower bound: shows training matters at all \\
B2 & Qwen3.5-0.8B post-trained             & Same-size off-the-shelf comparison \\
B3 & Qwen3.5-4B post-trained               & Larger same-family comparison \\
B4 & Qwen3.5-9B (4-bit, via llama-server)  & Distillation upper bound (the teacher; no longer in the judge ensemble per \S4.5) \\
\bottomrule
\end{tabular}
\caption{\textbf{Zero-shot baselines.} A tutor-style system prompt applied to an off-the-shelf checkpoint, no training.}
\label{tab:zeroshot-baselines}
\end{table*}
```

## 4.5 Multi-judge evaluation protocol

Judged metrics use a **cross-family** ensemble of three judges, each from
a family distinct from the teacher's — **Prometheus-7B-v2** (Mistral
lineage) [@kim2024prometheus], **Llama-3.1-8B-Instruct** (Meta)
[@grattafiori2024llama3], **Gemma-2-9B-it** (Google)
[@gemmateam2024gemma2] — deliberately excluding any Qwen-family judge to
eliminate self-preference bias [@panickssery2024selfpreference]. We report
the median across judges. One exception: the binary withholding rate
(§5.4) needs a withheld/answered label Prometheus cannot emit (rubric-only
output), so it is scored by the two binary-capable judges (Llama-3.1,
Gemma-2) and reported per judge. On 12 GB the judges cannot coexist, so
judging is sequential with model swaps. Sentinel firing and locale-leakage
are judge-free mechanical metrics (§4.6–§4.7).

## 4.6 Sentinel-firing metric (Persistent / FP / OffPosition probes)

Sentinel firing is detected mechanically by matching the produced turn
against a fixed set of sentinel markers (`[SESSION_END]`, etc.); the
scored quantity is binary. We report four rates over the §4.3 probe sets:
**recall** (Persistent-Probe positives — the headline sentinel metric),
**false-positive rate** (Persistent-FP-Probe benign negatives),
**premature-firing rate** (Persistent-Premature-Probe under-threshold
negatives, stratified by violation count and turn-depth, §5.3.1), and a
diagnostic **fire-rate by sentinel position** {5,7,9,11}. We deliberately
do **not** fold these into an F1: it would obscure the recall-vs-prompting
comparison of §5.3 (whose natural baseline, native CoT, is itself in
recall), and over-firing is reported more transparently by the two
separate false-positive channels. Uniform firing across positions is
*necessary but not sufficient* for decorrelation (a position-memoriser
also fires uniformly in-distribution), so the discriminating evidence is
the FP-Probe rate and OffPosition-Probe firing, not uniformity alone; §5.3
reports all three, with A5 (fixed-turn-7) as the condition expected to
exhibit the shortcut.

### 4.6.1 Persistence prompting ladder (steelman baseline)

To test whether the persistence gap is an artifact of weak (zero-shot)
prompting, we run a ladder of increasingly powerful prompting conditions on
the strongest prompt-only model — the 9B teacher (B4) — each scored by
recall on the Persistent-Probe positives exactly as for the trained
conditions: (1) **zero-shot** the full deployment prompt (§3.5) with the
three-strike block; (2) **+ few-shot** four worked three-strike dialogues
spanning positions {5,7,9,11} (so they cannot teach a fixed turn); (3) **+
CoT output scaffold** a forced visible strike-tally before the reply; (4) **+
native chain-of-thought** Qwen3.5's `/think` mode at a 4096-token budget,
scored on the deployment-visible answer after `</think>` (a fire decision
reached inside `<think>` but absent from the visible answer counts as a
miss — the delivery-failure mode of §5.3). Results in §5.3,
Table~\ref{tab:persistence-prompting-ladder}.

## 4.7 Locale-leakage rate (Locale-Leakage)

On Locale-Leakage, the baseline produces a tutor turn given a
cold-start china-locale scenario. The metric is the rate at which
the produced response contains a Western-default entity from a
fixed gazetteer (`config/western_entities.yaml`, ~200 entries
covering brands like Costco/Walmart, holidays like Thanksgiving,
US/UK place names, US/UK food items, and so on). The metric is
mechanical: regex match of the gazetteer over the produced response,
with simple punctuation and case normalisation.

## 4.8 Statistical reporting

**Three seeds on the load-bearing conditions.** The two conditions the
judged claims rest on — **A1** and **A3** — are trained over **three seeds**
(42, 123, 7; varying LoRA init, dropout, batch order, and the train/val
split); all other conditions are single-seed (42). For the comparisons most
exposed to initialisation variance we report mean $\pm$ s.d. over the three
seeds (withholding §5.4, persistence recall §5.3, locale leakage §5.7, and
the per-axis pairwise win-rates §5.6; values in Table~\ref{tab:stat-summary}
and each section). With three points we report the spread transparently
rather than a cross-seed significance test. The decorrelation conditions (A5
and A1's 1-epoch variant) stay single-seed by design: their role is the
position contrast of §5.3.1, whose verdict does not turn on initialisation
variance. Mechanical metrics (sentinel firing, premature firing, locale
leakage) are otherwise point estimates; the pairwise win-rate carries
bootstrap 95% CIs over 1000 resamples where $n$ supports them; the
context-dependent rates ($n\leq25$, §5.7) carry a small-$n$ caveat; and the
withholding rate ($n=63$) carries per-judge two-proportion tests on the
load-bearing contrasts (§5.4). The retired
redirect-axis F1 (Appendix&nbsp;A) is not used for any claim.

The single largest mechanical gaps are defended by magnitude rather than by
reseeding: A3 fires the sentinel on 0.000 of positive probes versus the
trained $\geq 0.83$, a separation no plausible initialisation variance can
close (§6.1). Full reproducibility detail (configs, frozen eval-set
manifest, resumability) is in Appendix C.
