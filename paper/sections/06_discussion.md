# 6. Discussion

## 6.1 Statistical rigor: seeds and small probe counts

The load-bearing conditions A1 and A3 are reported over **three seeds**
(42, 123, 7); others are single-seed (§4.8). The three-seed statistics
confirm the judged results are not initialisation artifacts: withholding A1
$0.63\pm0.08$ vs A3 $0.13\pm0.01$ (non-overlapping at every seed), A1
persistence recall $0.85\pm0.04$. The headline mechanical persistence result —
prompt-only $\leq 0.06$ and A3 exactly 0.000 vs trained recall $\geq 0.83$ —
is far too large to be a seed artifact. Where the judged metrics are more
fragile we flag it: the withholding A1-vs-A3 contrast clears significance under
each judge (Llama $z=5.9$, Gemma $z=5.6$), so the *necessity* claim is robust,
but the A1-vs-9B-teacher contrast is only at the edge (Llama $z=1.96$, Gemma
$z=1.62$) and is reported as directional (§5.4); the context-dependent
mechanical scores are n$\leq 25$ (suggestive, §5.7); and the pairwise eval
leans only on the large effects (role-swap 0.87, language 0.75). No *boundary*
conclusion rests on an underpowered comparison.

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
(vc$\in\{1,2\}$), but not orthogonally: a shallow turn can only carry vc=1 and
only deep turns reach vc=2, so the aggregate by-turn premature curve conflates
a violation-count effect with any turn-position effect. Re-slicing within each
vc stratum (Appendix B.6) shows the premature rise is driven by accumulated
violation count and conversation depth — peaking at turn 9, not the trained
turn 7 — rather than by a turn-position shortcut; but the within-vc curves are
not flat either, so a residual depth component remains that this probe cannot
cleanly separate from position. A definitive separation needs a future probe
that crosses turn-depth with violation count orthogonally. We flag the
entanglement rather than over-read the by-turn axis.

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
because the trained Llama-1B student and its prompt-only control share one
base, the contrast isolates *training* from scale rather than the size
comparison an 8B-vs-0.8B probe would be, and it holds on *both*
not-promptable behaviors. The 8B prompt-only rows confirm the gap is not
closed by scale alone (persistence 0.27→0.55 even with CoT). The central
claim is therefore not a Qwen artifact.

Two honest qualifications, neither of which touches the direction. First,
**magnitude is family-dependent**: the Llama trained withholding (0.50) sits
below the Qwen student's (0.61), and Llama attains markedly more *prompt-only*
persistence than the Qwen teacher ($\leq 0.06$), so the gap's sharpness varies by
family even though its sign (training $>$ prompting) does not. Second, an
**instruct-checkpoint asymmetry**: the Llama student trains from
Llama-3.2-1B-*Instruct*, whereas the Qwen student trains from a base checkpoint,
because Llama-3.2-1B-*Base* could not learn to emit the rare turn-end token under
LoRA-SFT (its post-turn distribution stays near-uniform, producing
non-terminating generations) — a finding that itself echoes the paper's
rare-token theme. We report the instruct-based student as the working
cross-family analogue and flag the asymmetry. Broader replication (a third
family, a base-checkpoint student, multi-locale) remains future work (§6.3).

**Locale, by contrast, does not threaten the central boundary.** The
load-bearing claims —
persistence (firing on the third same-axis violation) and withholding
(scaffolding instead of answering) — are *structural* behaviors: cross-turn
violation counting and the suppression of a strong answer prior, respectively.
Neither mechanism depends on the locale backdrop of the dialogues, so there is
no route by which "which behaviors are promptable" would change across
locales. Single-locale bounds only two secondary things: (i) the generality of
the *locale-fidelity axis* result — one of the already-promptable axes — and
(ii) the `locale_judge` gazetteer, which is locale-specific and treated as
pipeline engineering (§6.4). Multi-locale repeats would therefore broaden the
promptable-axis surface, not shore up the persistence/withholding claim, which
is locale-independent by construction.

**Judging.** The withholding criterion is binary (withheld vs answered), far
less subjective than a 1–5 rubric, and the pairwise ensemble
(Prometheus/Mistral, Llama-3.1/Meta, Gemma-2/Google) is drawn from three
families all distinct from the Qwen teacher, so the student is never scored by
a checkpoint sharing the teacher's lineage. The generic-redirect negative
control (0.55, §5.6) bounds any residual "A1 is globally preferred" component
to near zero, and the mechanical metrics are judge-free.

## 6.3 What we would do with more compute, in priority order

1. **Broader cross-family replication.** The trained non-Qwen student is now
   done — a Llama-3.2-1B student on the existing teacher-distilled corpus
   confirms the persistence boundary holds outside Qwen (§6.2, full-SFT recall
   0.91 vs matched prompt-only 0.25). What remains is *breadth*: a third family
   (e.g.\ Gemma), a base-checkpoint Llama student (the current one trains from
   the instruct checkpoint, since Llama-3.2-1B-Base could not learn the turn-end
   token under LoRA-SFT), and multi-locale repeats.
2. **Larger-scale decorrelation.** Test at 4B/7B, where the positional route
   is cheaper relative to the semantic one, so a genuine positional component
   of premature firing — and thus a decorrelation benefit on it — may emerge
   (§5.3.1).
3. **Further power the pedagogy teacher comparison.** A larger probe set plus
   a third *binary-capable* judge (Prometheus's rubric-only output
   disqualifies it) would let the trained-vs-teacher withholding gap be
   claimed as robustly significant rather than directional (§5.4). The
   boundary claim does not depend on it.
4. **Tighten the native-CoT persistence number.** A larger reasoning-token
   budget would separate "cannot count" from "truncated before the sentinel
   rendered" — our data suggest the latter dominates (§5.3), which would
   sharpen the claim that the failure is *delivery at deployment budget*, not
   counting capability.

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

## 6.5 Why some capabilities are promptable and others are not

Our results do more than report *that* the boundary exists; the pattern of
which behaviors fall on which side suggests *why*. A behavior is
**promptable** when a single clause both *describes and elicits* it — the
capability already lives in the model's prior, and the clause merely
*selects* it. Locale fidelity, role-swap deflection, and topic re-anchoring
are all of this kind: the pretrained model can produce an in-locale
reference or an in-character deflection unprompted, and the deployment
clause only has to point at the behavior it already has. Consistent with
this, prompt-only models reach parity on these axes and A3 (which drops the
specialized streams) does not leak more locale entities than A1 (§5.7) —
there is no gap for demonstration to close.

A behavior is **not promptable** when the clause names something the prior
cannot supply on demand, and we see two distinct failure modes. The first is
**missing cross-turn state**: persistence requires counting same-axis
violations across turns and firing on the third, but a single forward pass
maintains no such counter, so the clause describes a state machine the model
does not run. The diagnostic evidence is that *native* chain-of-thought —
which externalizes the count into tokens — partially recovers persistence
(0.06→0.63, §5.3) where few-shot and an output scaffold do not: give the
model a scratchpad to hold the state and it can count; leave the counting
implicit and it cannot. The second is **overriding a competing prior**:
withholding requires suppressing the strong general-assistant helpfulness
reflex, and a clause that says "scaffold, don't answer" competes with a prior
the model weights toward heavily. The diagnostic evidence is that ablating
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
