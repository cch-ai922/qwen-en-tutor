# 5. Results: the train-versus-prompt boundary

## 5.1 Protocol and conditions

Every condition is evaluated under the **same** fully-specified deployment prompt
(§3.5; Figure~\ref{fig:design}) — every redirect-axis clause and the complete
three-strike persistence block — so a prompt-only failure isolates *promptability*
rather than under-specification. Trained conditions see the prompt at train and
eval; prompt-only baselines see it at eval.

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=0.99\columnwidth]{fig_design.png}
\caption{\textbf{Matched-prompt protocol.} The identical fully-specified deployment prompt is supplied to a fine-tuned 0.8B student, a generic-SFT ablation, and prompt-only baselines up to the 9B teacher; each is scored by the instrument matched to the capability under test. Because the instruction is present for every condition, a prompt-only failure isolates promptability rather than under-specification.}
\label{fig:design}
\end{figure}
```

Conditions (defined in §4.4, Table~\ref{tab:trained-ablations}): the
trained student **A1** (full 12-stream SFT: normal + generic redirect + 6
specialized redirect + 4 persistent streams; 3342 records), the generic-SFT
ablation **A3** (**normal + generic-redirect streams only** — it drops *both* the
six specialized redirect streams *and* the four persistent streams; 1895 records),
and the prompt-only ladder **B1**–**B4** (0.8B-base, 0.8B-instruct, 4B-instruct,
9B teacher). Because A3 contains no persistent training data, its 0.000 persistence
recall reflects the *absence* of persistence demonstration, not a failure despite
it; and because A3 also drops the specialized streams, an A1-vs-A3 contrast removes
*several* streams at once (we bound what this contrast can and cannot causally
attribute in §6, and specify budget-matched leave-one-out ablations as future
work). A1 has more records than A3, so A1 also receives more optimizer steps at
matched epochs; we note this budget asymmetry as a limitation of the A1-vs-A3
contrast (§6). The fixed-turn ablation A5 and the 1-epoch A1 variant are introduced
for the position contrast in §5.3. The load-bearing conditions
A1 and A3 are reported over three seeds (42/123/7); all others are
single-seed, with statistical caveats stated per metric and in §6.

## 5.2 The boundary at a glance

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=0.99\columnwidth]{fig_boundary.png}
\caption{\textbf{Three regimes of the train-versus-prompt boundary.} Each capability is scored by its own validated, capability-specific metric (no heterogeneous metrics are combined on one axis, and the retired context-blind axis-F1 is not used here; §5.5). \emph{Regime A (prompt-sufficient)}: locale fidelity --- prompt-only reaches parity. \emph{Regime B (prompt-elicitable but data-refinable)}: role-swap and topic re-anchoring --- the behavior type appears prompt-only, but specialized data wins the pairwise quality comparison (§5.6). \emph{Regime C (training-dependent under the tested regimes)}: multi-turn persistence and pedagogical withholding --- prompt-only falls far short despite the identical instruction. Persistence prompt-only is the 9B teacher's zero/few-shot recall ($\leq 0.06$); withholding prompt-only is the 9B teacher (0.45). Interval estimates and full statistics in Table~\ref{tab:stat-summary}.}
\label{fig:boundary}
\end{figure}
```

**Three regimes, not a binary** (Figure~\ref{fig:boundary}): **(A)
prompt-sufficient** — a clause elicits the behavior and data adds little (locale
fidelity); **(B) prompt-elicitable but data-refinable** — the behavior *type*
appears prompt-only, yet specialized data materially improves its *quality*
(role-swap and topic, §5.6); **(C) training-dependent under the tested regimes** —
the prompt names the behavior but does not reliably elicit it, and SFT installs it
(persistence, withholding). Regime B is what a binary framing hides. Each row of
the summary figure uses its own validated capability-specific metric (no
heterogeneous metrics on one axis; the retired context-blind axis-F1 is not used,
§5.5). The remainder of §5 establishes each row; Table~\ref{tab:stat-summary}
collects every headline number with its sample size and dispersion (three-seed
s.d., bootstrap CI, or paired test as applicable; §4.8, §6.1).

```{=latex}
\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}l l l l l@{}}
\toprule
\textbf{Headline result} & \textbf{Metric} & \textbf{$n$} & \textbf{Value} & \textbf{Dispersion / test} \\
\midrule
Persistence, A1 (trained)          & sentinel recall      & 159 & 0.83 & 3-seed mean $0.85\pm0.04$ (42/123/7) \\
Persistence, A3 (generic-SFT)      & sentinel recall      & 159 & 0.000 & point estimate; gap to $\geq 0.83$ magnitude-defended (§6.1) \\
Persistence, 9B teacher            & sentinel recall      & 159 & $\leq 0.06$ & single-run (off-the-shelf); native CoT 0.63 \\
Withholding, A1 (trained)          & withholding rate     & 63  & 0.611 & 3-seed $0.63\pm0.08$; A1-vs-A3 paired McNemar $p<10^{-6}$ (both judges) \\
Withholding, A3 (generic-SFT)      & withholding rate     & 63  & 0.119 & 3-seed $0.13\pm0.01$; non-overlapping every seed \\
Withholding, 9B teacher            & withholding rate     & 63  & 0.452 & A1-vs-teacher $z=1.96$/$1.62$ (edge; directional only) \\
Locale leakage, A1 / A3            & Western-default rate & 224 & 1.34\% / 0.89\% & 3-seed $0.022\pm0.008$ / $0.016\pm0.007$; indistinguishable \\
Pairwise, role\_swap (A1 vs A3)    & A1 win-rate          & --- & 0.87 & 3-seed $0.82\pm0.06$; bootstrap 95\% CI (§5.6) \\
Pairwise, language (A1 vs A3)      & A1 win-rate          & --- & 0.75 & 3-seed $0.76\pm0.06$ \\
Pairwise, generic (control)        & A1 win-rate          & 60  & 0.55 & 3-seed $0.53\pm0.05$; near parity \\
Context-dependent (locale/lang/topic) & mechanical rate  & $\leq 25$ & --- & point estimates; underpowered, suggestive only (§5.7) \\
\bottomrule
\end{tabular}
\caption{\textbf{Statistical summary of the headline results.} Every load-bearing number with its sample size and dispersion. Three-seed s.d. is reported for the reseeded A1/A3 conditions (§4.8); off-the-shelf baselines (B1--B4) carry no training seed and are single-run. The two mechanical results the boundary rests on — A3 persistence 0.000 vs trained $\geq 0.83$, and the A1-vs-A3 withholding gap significant under each judge — are the ones that do not turn on seed variance. Comparisons flagged "directional" or "underpowered" are reported as such throughout §5 and are not the basis of any boundary claim (§6.1).}
\label{tab:stat-summary}
\end{table*}
```

## 5.3 Regime C, I — multi-turn persistence (mechanical)

The sentinel-firing metrics are the only fully-mechanical evaluation in the paper
and consult no judge. The behavior: on the third same-axis violation, emit the
exact literal sentinel string for that axis. The deployment prompt specifies the
entire protocol (§3.5, `[persistence]`) — escalating brevity across the three
strikes, the four governed axes, the four literal `[SESSION_END: …]` strings, and
anti-jailbreak clauses.

**Prompting does not elicit it.** Under this fully-specified prompt, no
prompt-only model fires the sentinel reliably: on the strongest prompt-only
model (the 9B teacher) zero-shot and few-shot prompting reach $\leq 0.06$
recall (the full prompting ladder is below), and the ablation **A3**, which
sees the same prompt but is not trained on the specialized/persistent
streams, fires on **0.000** of positive probes — never. Every model trained
on the persistent streams fires the sentinel — A1 at recall **0.83** (3-seed
mean $0.85\pm0.04$ over 42/123/7), the sentinel-position ablations at 0.56--0.82
(§5.3.1) — whereas the strongest prompt-only
model tops out at 0.06 and A3 at 0.000. Under the tested prompting regimes,
counting same-axis violations across turns and emitting a rare literal marker on
the third is a behavior the prompt names but does not reliably elicit; it is
installed by demonstration. This is the single cleanest mechanical result in the
paper and it is design- and epoch-independent.

**Recall alone would overstate correctness — the full error taxonomy does not.**
High sentinel recall can in principle be achieved by firing too often, so we
report the trained student's persistence as a full operational error taxonomy
(Table~\ref{tab:persistence-taxonomy}), not a single recall number
(`scripts/score_paper_persistence_taxonomy.py`).

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l c c c c c c@{}}
\toprule
\textbf{Condition} & \textbf{Recall} & \textbf{Prem.@2} & \textbf{Benign FPR} & \textbf{Precision} & \textbf{Bal.\,acc} & \textbf{MCC} \\
\midrule
A1 student (SFT)     & 0.830 & 0.201 & 0.000 & 0.795 & 0.885 & 0.758 \\
A3 (generic-SFT)     & 0.000 & 0.000 & 0.000 & ---   & 0.500 & 0.000 \\
0.8B base (prompt)   & 0.000 & 0.000 & 0.008 & 0.000 & 0.497 & $-0.035$ \\
4B instruct (prompt) & 0.063 & 0.044 & 0.000 & 0.526 & 0.523 & 0.121 \\
9B teacher (prompt)  & 0.025 & 0.025 & 0.000 & 0.500 & 0.509 & 0.071 \\
\bottomrule
\end{tabular}
\caption{\textbf{Persistence as a full error taxonomy, not recall alone.} Recall on the positive probe (n=159); Prem.@2 = premature firing with only two prior strikes (sub-threshold); Benign FPR = firing in a fully benign context; Precision, balanced accuracy, and Matthews correlation (MCC) pool the positive probe with the premature and benign negatives into one confusion matrix, so recall inflated by over-firing is penalized. The trained student's 0.83 recall is \emph{not} an over-firing artifact: benign FPR is 0.000, malformed-marker and duplicate-fire rates are 0.000 (not shown), and MCC is 0.758 --- genuine discrimination. Every prompt-only condition and the generic-SFT ablation collapse to MCC $\approx 0$. Scored by \texttt{score\_paper\_persistence\_taxonomy.py}.}
\label{tab:persistence-taxonomy}
\end{table}
```

The trained student's headline 0.83 survives the harder metrics: benign
false-positive rate 0.000, precision 0.795, balanced accuracy 0.885, and
MCC 0.758 — it fires on genuine third strikes and stays silent in benign
contexts, rather than achieving recall by over-firing. It does fire prematurely on
20% of two-strike (sub-threshold) contexts, which we report openly and analyze in
§5.3.1. Every prompt-only condition and the generic-SFT ablation sit at
MCC $\approx 0$: they do not fire at all, so the gap is not a threshold-placement
difference but the presence or absence of the behavior.

**The gap is not an artifact of zero-shot prompting.** Counting same-axis
violations across turns is exactly the regime where in-context exemplars and
chain-of-thought are expected to help, so we tested the steelman on the
strongest prompt-only model (the 9B teacher, B4) under a ladder of
increasingly powerful prompting, scored mechanically on the same
Persistent-Probe positives. Exemplars are four complete worked three-strike
dialogues drawn from the *training* split, spanning sentinel positions
{5,7,9,11} so they cannot themselves teach a fixed firing turn.

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l l c@{}}
\toprule
\textbf{B4 (9B) prompting condition} & \textbf{mechanism} & \textbf{recall} \\
\midrule
zero-shot instruction          & instruction only                & 0.025 \\
+ few-shot exemplars           & 4 worked 3-strike dialogues      & 0.013 \\
+ CoT output scaffold (no-think) & forced strike-tally in output  & 0.057 \\
+ native chain-of-thought      & Qwen3.5 \texttt{/think} reasoning & \textbf{0.63} \\
\midrule
A1 trained 0.8B (reference)    & SFT                              & \textbf{0.83} \\
\bottomrule
\end{tabular}
\caption{\textbf{Persistence resists prompting up to, but not including, native chain-of-thought — and even that does not reach the trained student.} Recall = fraction of true third-strike positives on which the deployment-visible answer emits the sentinel. Few-shot exemplars and an output-forced counting scratchpad leave the 9B teacher at $\leq 0.06$ (no better than zero-shot). Only Qwen3.5's \emph{native} reasoning (\texttt{/think}) moves the needle, to 0.63 — substantially closing but not closing the gap to the trained 0.8B student (0.83). The A1 reference recall is 0.83 (seed 42; three-seed mean $0.85\pm0.04$ over 42/123/7, §4.8); the prompt-only-vs-trained gap ($\leq 0.06$ / 0.63 vs 0.83) is far too large for seed variance to close. The B4 prompting-ladder rows are single-run (off-the-shelf model, no training seed).}
\label{tab:persistence-prompting-ladder}
\end{table}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=0.99\columnwidth]{fig_failure.png}
\caption{\textbf{The two not-promptable behaviors.} \emph{Left}: on persistence, the 9B teacher stays at $\leq 0.06$ recall through few-shot and an output CoT scaffold; only native chain-of-thought moves it, to 0.63 --- still below the trained student's 0.83 (dashed). \emph{Right}: on withholding, every prompt-only condition (including the 9B teacher, 0.45) falls below the trained student's 0.61, and the no-specialized-data ablation (A3) collapses to near the untrained-base rate.}
\label{fig:failure}
\end{figure}
```

Two costs make the native-CoT result a *relocation* of the boundary, not a
refutation (Figure~\ref{fig:failure}). First, **accuracy**: even with unrestricted
reasoning the 9B teacher reaches only 0.63 (100/159), below the trained student's
0.83. Second, **cost and reliability**: the reasoning block runs $\approx$1.6–3.2k
tokens/turn, and at a practical 4096-token budget roughly a third of responses
either exhaust the budget inside `<think>` or conclude "fire" within the reasoning
while the deployment-visible answer omits the sentinel — the decision is reached,
but *delivering* it at deployment budget is what fails. So the claim is:
**persistence resists zero/few-shot prompting outright, and is only partially
recovered by native chain-of-thought — at a cost and reliability deficit SFT
removes** (§6.2–§6.3).

### 5.3.1 Trigger-position decorrelation result (bounded side-result)

In an isolated matched contrast (4-variant positions {5,7,9,11} vs the naive
fixed-turn-7 design A5, both with the deployed axis-specific sentinel and a
1-epoch budget; §3.4), decorrelation has a *two-sided* effect. It **removes
the positional recall bias** — the fixed-turn design fires best at its trained
turn 7 and degrades off-position (recall 0.56), while the 4-variant fires
uniformly across positions (0.82) — but it **does not reduce premature firing**
(0.208 vs 0.119). The reason is that at 0.8B the premature over-firing is
*threshold-laxity*: it tracks accumulated violation count rather than a fixed
turn position, so there is no positional shortcut for decorrelation to suppress.

**Strike count, not turn depth, drives the trained student's firing.** Because
violation count and conversation depth are correlated in natural dialogue, we
tested whether firing tracks the semantic strike count or merely conversation
depth, using the existing premature probe, in which three depths (turns 3, 5, 7)
each carry *both* one- and two-strike contexts
(`scripts/score_paper_depth_count.py`). Holding depth fixed, adding the second
strike raises the A1 student's firing at every shared depth (+0.12 at turn 3,
+0.18 at turn 5, +0.16 at turn 7). An item-level logistic model of firing on
standardized depth and strike count (cluster-robust SE, n=318) finds the
strike-count term strongly positive after adjusting for depth (count: $\beta=+2.55$,
OR 12.8, $p=0.001$; depth: $\beta=+0.68$, OR 2.0, $p=0.0015$). So the trained
student's firing tracks accumulated strike count over and above turn depth —
evidence it learned three-strike tracking rather than a depth or position
heuristic. The probe is not a fully balanced depth$\times$count grid, so we report
this as an observational stratification on existing data and flag a purpose-built
balanced grid (violation count $\{0,1,2,3\}\times$ depth $\{5,7,9,11,13,15\}$) as
future work (§6). The full contrast and the recall-by-turn stratification are in
Appendix B.6 (Table~\ref{tab:decorrelation} and Table~\ref{tab:recall-by-turn}). We
report the decorrelation construction as *partially validated* and do not lean on
it for the boundary claim; the robust, budget-independent result of this section
is the persistence row of Figure~\ref{fig:boundary}: **persistence is installed by
SFT under the tested regimes.**

## 5.4 Regime C, II — pedagogical withholding

The pedagogy axis tests whether the tutor *withholds* the answer and scaffolds
instead of supplying it — instructed explicitly in the deployment prompt ("give ONE
short sentence or example, then continue — no structured lesson", §3.5). We report
a **withholding rate**: the fraction of probes on which the model declined to supply
the requested answer and scaffolded instead. We score **63 held-out pedagogy
probes** under the matched prompt with the two binary-capable judges (Llama-3.1-8B,
Gemma-2-9B; Prometheus emits rubric-only output, §4.5).
Table~\ref{tab:withholding} reports each judge's rate and their mean.

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l l c c c@{}}
\toprule
\textbf{Condition} & \textbf{demonstrated?} & \textbf{Llama-3.1} & \textbf{Gemma-2} & \textbf{mean} \\
\midrule
A1 (full SFT)        & yes & 0.571 & 0.651 & \textbf{0.611} \\
A3 (no specialized)  & no  & 0.079 & 0.159 & 0.119 \\
B1 0.8B-base         & no  & 0.127 & 0.048 & 0.087 \\
B2 0.8B-instruct     & no  & 0.302 & 0.317 & 0.310 \\
B3 4B-instruct       & no  & 0.286 & 0.302 & 0.294 \\
B4 9B-teacher        & no  & 0.397 & 0.508 & 0.452 \\
\bottomrule
\end{tabular}
\caption{\textbf{Pedagogical withholding under a matched prompt (n=63, two judges).} The instruction to withhold is present for every condition. The trained student withholds at 0.61 (mean over the two judges); the no-specialized-data ablation A3 collapses to 0.12, near the untrained-base rate (B1, 0.09). Prompt-only models, including the 9B teacher (0.45), withhold far less than the trained 0.8B student despite the identical instruction. Per-judge rates are each over the full 63 probes (21 original + 42 fresh held-out, pooled). Per-judge rates shown are seed 42; the trained conditions are confirmed across three seeds (42/123/7): withholding A1 $0.63\pm0.08$ vs A3 $0.13\pm0.01$, non-overlapping at every seed (§4.8). B1--B4 are off-the-shelf and carry no training seed.}
\label{tab:withholding}
\end{table}
```

Two readings, separated by statistical weight at n=63:

**Load-bearing (robustly significant): the A3 ablation.** A1 (0.571 Llama /
0.651 Gemma) vs A3 (0.079 / 0.159) is a large gap between two models that
differ in *only* the pedagogy/specialized streams, under the *same* prompt.
Because A1 and A3 are judged on the **same 63 prompts**, the outcomes are paired
by item; we therefore use the **McNemar exact test** on the identical prompts
(rather than an unpaired two-proportion test) plus a **paired bootstrap** over
prompts (`scripts/score_paper_withholding_paired.py`). The paired test clears
significance under **each judge separately** (Llama: 31 prompts A1-withholds /
A3-does-not vs 0 the other way, exact $p=9.3\times10^{-10}$; Gemma: 35 vs 4,
$p=3.4\times10^{-7}$), with a paired-bootstrap 95% CI on the rate difference of
$[0.37, 0.62]$ (Llama) / $[0.33, 0.64]$ (Gemma) — both excluding zero. A3 falls to
roughly the untrained-base rate (B1): removing the demonstration data does not
merely fail to help, it leaves the model at baseline. This is the clean evidence
that the withholding behavior is *installed by demonstration*, not by the prompt
clause that describes it — and it is the result the boundary claim rests on. It is
robust to initialisation: across the three seeds the A1 and A3 withholding
distributions do not overlap at any seed (§4.8, Table~\ref{tab:withholding}).

**Directionally consistent, at the edge of significance: the teacher
comparison.** The strongest statement here is *absolute*: the 9B teacher, given
the identical explicit instruction to withhold, complies on under half of probes
(mean 0.452; Llama 0.397, Gemma 0.508) — a model an order of magnitude larger than
the student, told plainly to scaffold rather than answer, doing so less than half
the time is itself direct evidence withholding is not prompt-elicited here. The
trained student withholds more (0.611), but this *comparative* gap only approaches
per-judge significance (Llama $z=1.96$; Gemma $z=1.62$). We therefore rest the
claim on the teacher's absolute non-compliance and the A3 ablation, and report the
student-beats-teacher comparison as directional only; the boundary claim does not
require it. Both regime-C behaviors share a cause we develop in §6.5:
scaffolding-instead-of-answering requires suppressing the model's strong
helpfulness prior, which a clause can describe but demonstration supplies.

## 5.5 Why we do not report redirect-axis F1 as a primary metric

The conventional metric for multi-axis redirects is a per-axis F1 from a
judge that labels the produced response by axis. Because that judge is
context-blind, the metric is a *type* classifier, not a *quality* measure:
it floors on axes whose repair has no context-free surface form (locale,
language, topic) and ceilings on axes every model satisfies (persona,
role-swap), so its macro-average ties conditions of very different quality
(A1 0.409, B2 0.408, 9B teacher 0.409) and conceals the *largest*
specialized-data effect in the study (§5.6). We therefore retire it to
Appendix A as an evidenced negative result and score each capability with a
matched instrument: sentinel firing (§5.3), withholding rate (§5.4),
pairwise preference (§5.6), and mechanical leakage rates (§5.7).

## 5.6 Regime B — type is prompt-elicitable, quality is data-refined

The behavior *type* appears prompt-only on these axes (the type-F1 ceiling,
Appendix A); the remaining question is whether specialized data improves *quality*.
We test it with a **quality-aware pairwise preference**: for each held-out probe,
the repairs from A1 and A3 are shown to a three-judge cross-family ensemble in
randomized order and we report A1 win-rate. Both are SFT-only and differ in *only*
the specialized streams, so a win is a clean specialized-data quality effect (no
DPO, no prompt confound).

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l c c c c@{}}
\toprule
\textbf{Axis} & \textbf{Llama} & \textbf{Prometheus} & \textbf{Gemma} & \textbf{mean} \\
\midrule
role\_swap & 0.83 & 0.78 & 1.00 & \textbf{0.87} \\
language   & 0.57 & 0.86 & 0.81 & 0.75 \\
locale     & 0.64 & 0.72 & 0.68 & 0.68 \\
pedagogy   & 0.71 & 0.67 & 0.62 & 0.67 \\
persona    & 0.55 & 0.61 & 0.68 & 0.61 \\
topic      & 0.45 & 0.68 & 0.58 & 0.57 \\
\midrule
overall    & 0.61 & 0.71 & 0.71 & --- \\
\bottomrule
\end{tabular}
\caption{\textbf{Pairwise quality, A1 vs A3 (specialized stream vs none, both SFT-only).} A1 win-rate; 0.5 is parity. Every axis favors A1. role\_swap is the largest effect (0.87, Gemma preferring A1 on every pair) — on an axis the context-blind F1 reported as \emph{saturated parity}. The F1 ceiling concealed a real, large quality effect. Per-axis values shown are seed 42. Across three seeds (42/123/7) the two largest effects reproduce --- role\_swap $0.82\pm0.06$ and language $0.76\pm0.06$ --- while the remaining axes cluster at 0.54--0.67 with larger seed variance (locale in particular regresses toward parity, $0.54\pm0.12$); we therefore lean only on role\_swap and language. The generic-redirect negative control stays near parity across seeds ($0.53\pm0.05$, §4.8).}
\label{tab:pairwise}
\end{table}
```

The headline is the reconciliation, and it is exactly regime **B**: **role-swap is
prompt-elicitable as *type* (every model deflects) yet shows the largest
specialized-data *quality* win (0.87)** — on an axis the context-blind F1 called
saturated parity. This is why a binary promptable/not-promptable framing is
inadequate for role-swap and topic: the behavior *type* is prompt-elicitable, but
its deployment *quality* is data-refined. Promptability of the *type* determines
whether the behavior appears at all; specialized data determines its quality — a
distinct axis, not a contradiction. We lean only on the large, seed-stable effects
(role-swap, language) and read the smaller wins as near-parity.

**Negative control.** On the **generic** redirect stream, which *both* A1
and A3 train on, the pairwise win-rate is 0.55 (33 win / 22 lose / 5 tie,
n=60) — near parity. Where the data is shared, A1 does not win; the
specialized-axis wins above are therefore not a global "A1 is simply
better" artifact but axis-specific data effects.

## 5.7 Locale and language (mechanical)

**Locale fidelity is prompt-driven — on the leakage dimension we measure.**
Western-default *leakage*, measured by a word-boundary gazetteer over 224
cold-start china-locale generations: A1 1.34% (3/224), A3 0.89% (2/224), B1 1.34%
— statistically indistinguishable. The ablation that *removes* the specialized
locale stream (A3) does not leak more; low-leakage locale adherence is carried by
the one-line "ground cultural items in {country}" clause, not by the specialized
data. We are explicit about what this metric does and does not capture: it
measures *absence of out-of-locale (Western) entities*, which is not the same as
*positive in-locale grounding* — a response can avoid all Western entities yet
remain culturally generic. Our claim is therefore scoped to out-of-locale leakage;
positive in-locale grounding, and the relevance/forcing trade-off, are a separate
dimension we do not fully measure here (a two-dimension locale metric, validated on
a human-labeled subset, is specified as future work, §6). Within that scope, this
is a clean regime-A result that **corrects** any claim that the locale stream
drives low leakage. (The parity holds across three seeds: A1 leakage
$0.022\pm0.008$, A3 $0.016\pm0.007$ over 42/123/7 — both low and statistically
indistinguishable, §4.8.)

On the context-dependent mechanical scores (all n$\leq 25$ and so underpowered),
only language shows a sizable gap (A1 0.71 vs A3 0.33), which we report as
suggestive only, consistent with its pairwise win (0.75); locale and topic are
near-parity. We omit a naturalness rating (the only figures available were on an
SFT+DPO checkpoint, and every condition here is SFT-only); the boundary results do
not depend on it.
