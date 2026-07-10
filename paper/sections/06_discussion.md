# 6. Discussion

## 6.1 Statistical rigor: seeds and small probe counts

The load-bearing conditions A1 and A3 are reported over **three seeds** (42, 123,
7); others are single-seed (§4.8). The three-seed statistics confirm the judged
results are not initialisation artifacts: withholding A1 $0.63\pm0.08$ vs A3
$0.13\pm0.01$ (non-overlapping at every seed), A1 persistence recall
$0.85\pm0.04$; and the mechanical persistence result (prompt-only $\leq 0.06$ / A3
0.000 vs trained $\geq 0.83$) is far too large to be a seed artifact. Where the
judged metrics are more fragile we flag it: the withholding A1-vs-A3 contrast is
significant under each judge (paired McNemar $p<10^{-6}$), so the *necessity* claim
is robust, but the A1-vs-9B-teacher contrast is only at the edge and reported as
directional (§5.4); the context-dependent scores are n$\leq 25$ (§5.7); and the
pairwise eval leans only on the large effects. No *boundary* conclusion rests on an
underpowered comparison.

## 6.2 Threats to validity

**What "prompting" includes, and the few-shot/CoT steelman.** "Prompting" is
the complete zero-shot deployment instruction (§3.5). The natural objection —
persistence is a counting task, exactly where exemplars and CoT should help,
so zero-shot is too weak — we met with the full prompting ladder on the 9B
teacher (§5.3, Table~\ref{tab:persistence-prompting-ladder}): few-shot and an
output scaffold do not help, and only native CoT partially recovers recall,
still short of the trained student and at heavy inference cost. So the
persistence claim is precisely "resists zero-shot and few-shot prompting
outright; only partially recovered by native CoT, at a deficit SFT removes" —
a relocation of the boundary, reported as such. The pedagogy claim is less
exposed: withholding is a single-turn decision, so a zero-shot clause is a
fair test, and the result is anchored on the A3 ablation, not prompt-only
failure alone.

**Turn-depth and violation-count are entangled in Persistent-Premature-Probe.**
The probe varies both the premature turn and the number of prior violations
(vc$\in\{1,2\}$). They are correlated in natural dialogue, so the aggregate
by-turn premature curve could conflate a violation-count effect with a
turn-position effect. We partially separate them using the overlap structure the
probe *does* have — three depths (turns 3, 5, 7) each carry both vc=1 and vc=2 —
and find that, holding depth fixed, adding the second strike raises the trained
student's firing at every shared depth, and an item-level logistic model gives a
strongly positive strike-count coefficient after adjusting for depth (count OR
12.8, $p=0.001$; depth OR 2.0, $p=0.0015$; §5.3.1). So the effect tracks
accumulated strike count over and above depth, consistent with learned
three-strike tracking rather than a pure turn-position shortcut. This remains an
observational stratification, not a matched factorial: a residual depth component
is present, and a definitive separation needs a purpose-built probe crossing turn
depth $\{5,7,9,11,13,15\}$ with violation count $\{0,1,2,3\}$ orthogonally, which
we specify as future work. We report the depth-adjusted result and flag the
remaining entanglement rather than over-read the by-turn axis.

**Single family (a real limitation) and single locale (a scope note, not a
threat).** All experiments use a Qwen-family base and teacher at
`locale=china`. These are not equal limitations. The **family** limitation is
genuine: distillation is intra-family, and whether a behavior is
trainable-but-not-promptable could plausibly shift with a family's
instruction-following and in-context-learning strength. We therefore ran a
two-part cross-family check on the **Llama** family — a prompt-only probe at
8B and a *trained* student at 1B — and both confirm the boundary's *direction*
holds outside Qwen. Table~\ref{tab:crossfamily} collects the persistence
recall.

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l l c c@{}}
\toprule
\textbf{Model} & \textbf{Condition} & \textbf{Persist. recall} & \textbf{Withhold rate} \\
\midrule
Qwen-0.8B      & prompt-only (9B teacher)   & $\leq$0.06 & 0.09--0.45 \\
(in-family)    & trained (A1)               & 0.83--0.85 & 0.61 \\
\midrule
Llama-3.1-8B   & prompt-only, zero-shot     & 0.27       & 0.22--0.32 \\
               & prompt-only, prompted CoT  & 0.55       & --- \\
Llama-3.2-1B   & prompt-only (untrained)    & 0.25       & 0.11 \\
               & \textbf{trained, full SFT (A1)} & \textbf{0.91} & \textbf{0.50} \\
\bottomrule
\end{tabular}
\caption{\textbf{The boundary replicates in a second trained family.} On \emph{both} not-promptable behaviors, training the \emph{same} Llama-3.2-1B-Instruct base --- evaluated against its own untrained control under the identical deployment prompt --- lifts the behavior far above prompt-only: persistence recall $0.25\!\to\!0.91$, withholding $0.11\!\to\!0.50$ (two judges). Because the trained student and the prompt-only control share one base, this isolates \emph{training} from scale. The 8B prompt-only rows show the behavior stays low even for a much larger model. Direction is robust across families; magnitude is family-dependent (the Llama trained withholding 0.50 is below Qwen's 0.61).}
\label{tab:crossfamily}
\end{table}
```

This is the load-bearing generalization result (Table~\ref{tab:crossfamily}):
because the trained Llama-1B student and its prompt-only control share one base,
the contrast separates *training from scale within that base*, and it holds on both
regime-C behaviors; the 8B prompt-only rows confirm the gap is not closed by scale
alone (persistence 0.27→0.55 even with CoT), so the effect is not a Qwen artifact.

Two honest qualifications, neither of which touches the direction. First,
**magnitude is family-dependent**: the Llama trained withholding (0.50) sits
below the Qwen student's (0.61), and Llama attains markedly more *prompt-only*
persistence than the Qwen teacher ($\leq 0.06$), so the gap's sharpness varies by
family even though its sign (training $>$ prompting) does not. Second, an
**instruct-checkpoint asymmetry**: the Llama student trains from
Llama-3.2-1B-*Instruct* (the Qwen student trains from a base checkpoint), because
Llama-3.2-1B-*Base* could not learn the rare turn-end token under LoRA-SFT — itself
an echo of the paper's rare-token theme. We flag this asymmetry; broader
replication (a third family, a base-checkpoint student, multi-locale) is future
work (§6.3).

**Locale is a scope limitation we hedge rather than dismiss.** The regime-C claims
are *structural* behaviors (cross-turn violation counting; suppression of an answer
prior) whose formal triggers contain no locale-specific variables, so we
*hypothesize* they are less locale-dependent than the redirect axes. But empirical
behavior can still vary with cultural context and learner profile, so single-locale
evaluation limits external validity — we do not claim locale-independence as
established. It most directly bounds the locale-fidelity axis result and the
locale-specific `locale_judge` gazetteer (§6.4); a limited multi-locale check is
future work (§6.3).

**Judging.** The withholding criterion is binary (withheld vs answered), far
less subjective than a 1–5 rubric, and the pairwise ensemble
(Prometheus/Mistral, Llama-3.1/Meta, Gemma-2/Google) is drawn from three
families all distinct from the Qwen teacher, so the student is never scored by
a checkpoint sharing the teacher's lineage. The generic-redirect negative
control (0.55, §5.6) bounds any residual "A1 is globally preferred" component
to near zero, and the mechanical metrics are judge-free.

## 6.3 What we would do with more compute, in priority order

These are the experiments that would most strengthen the causal and
external-validity story; the first three directly address the strongest
open threats and would be run before broadening scope.

1. **Budget-matched leave-one-stream-out ablations.** The A1-vs-A3 contrast
   removes *several* streams at once (A3 = normal + generic-redirect only), and
   A1 has more records than A3, so it cannot cleanly attribute a capability
   effect to a *single* stream. The clean design is a leave-one-out per headline
   capability — A1-minus-pedagogy, A1-minus-persistence, A1-minus-role-swap —
   holding the training budget constant (matched total tokens / optimizer steps,
   or replacing the removed examples with an equal number of generic examples of
   similar length). The defensible claim would then be: *removing only capability
   X, at fixed budget, causes the capability-X metric to fall.*
2. **Human validation of the judged metrics.** The withholding and pairwise
   results currently rely on LLM judges. A blinded human study on 100--200 items
   — a four-class withholding rubric (direct answer / partial answer /
   hint-scaffold / other) and the strongest pairwise effects (role-swap 0.87) —
   reporting human--human and judge--human agreement (Cohen's $\kappa$ /
   Krippendorff's $\alpha$) and A/B-order/position-bias controls, would calibrate
   the judges. We provide the rubric and protocol (Appendix); the calibration
   itself is future work.
3. **Orthogonal depth$\times$violation-count probe.** A balanced grid crossing
   turn depth $\{5,7,9,11,13,15\}$ with violation count $\{0,1,2,3\}$ across all
   four axes would settle the depth-vs-count question directly; §5.3.1 shows count
   survives depth on the existing (unbalanced) probe, but a matched grid is the
   clean test.
4. **Out-of-distribution robustness set.** An independent OOD probe set
   (human-authored scenarios, a different generator family, paraphrases, unseen
   topics, additional locales; even 100--200 probes) would separate in-distribution
   held-out performance from genuine OOD robustness — the synthetic train/eval
   distributions are close despite the seed-level split (§4).
5. **Higher-precision teacher control.** The strongest prompt-only baseline is a
   4-bit-quantized 9B teacher; a Q8/BF16 control on a persistence subset would
   confirm the prompt-only gap is not an artifact of aggressive quantization.
6. **Broader cross-family replication and larger-scale decorrelation.** A third
   family (e.g.\ Gemma), a base-checkpoint Llama student, and 4B/7B decorrelation
   (§5.3.1), plus a larger reasoning-token budget to separate "cannot count" from
   "truncated before the sentinel rendered" in the native-CoT persistence number
   (§5.3).

We name these so a reviewer's "what about X" is met with a concrete plan.

## 6.4 Engineering caveat: locale_judge false positives

The pipeline's `locale_judge` uses a capitalization-based proper-noun
extractor, which in our initial run false-positived on common English
sentence-initial words (`Plus`, `Will`), locally-canonical landmarks
(`West Lake`, `Drum Tower`), and universal tools (`Python`). Two static
allowlists raised the pass rate from 70.1% to 88.4%. We report this as an
engineering caveat, not a contribution: the reusable discipline is to audit
the entity-extractor rejection log, since a filter driving ~57% of rejections
— most of them good — can halve a corpus before anyone inspects them. This
generalises to any capitalization-based entity filter.

## 6.5 Mechanistic hypotheses suggested by the results

The pattern of which behaviors fall in which regime suggests *why*. We frame these
as *hypotheses* the data are consistent with, not proven mechanisms — a transformer
can infer a count without an explicit external counter, so these are the most
direct reading of the evidence, not the only one. A behavior is **prompt-elicitable**
when a clause both *describes and elicits* it — the capability already lives in the
prior and the clause merely *selects* it (locale, role-swap, topic: the pretrained
model can produce these unprompted, so prompt-only reaches parity and there is no
gap for demonstration to close).

A behavior is **training-dependent (under the tested regimes)** when the clause
names something the prior does not supply on demand, via two failure modes. First,
**missing cross-turn state**: persistence requires counting same-axis violations
and firing on the third, but a single forward pass maintains no such counter — and
the diagnostic is that *native* chain-of-thought, which externalizes the count,
partially recovers persistence (0.06→0.63, §5.3) where few-shot and an output
scaffold do not. Second, **overriding a competing prior**: withholding requires
suppressing the strong helpfulness reflex, and the diagnostic is that ablating
the pedagogy demonstrations (A3) collapses withholding to the untrained-base
rate (§5.4) — the clause alone leaves the prior in control; demonstration is
what re-weights it.

So the boundary is not a list of arbitrary hard cases. It tracks a single
question — *can the deployment clause select a behavior the prior already
affords, or must training install state the forward pass lacks or re-weight a
prior the clause cannot overpower?* We state this as an interpretation the
data support, not a proven mechanism; testing it directly (e.g. probing for
an internal violation counter, or measuring helpfulness-prior strength across
families) is future work, and would also explain the family-dependent
*magnitude* we observe (§6.2).
