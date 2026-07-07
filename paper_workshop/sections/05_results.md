# 5. Results: The Train-versus-Prompt Boundary

Every condition is evaluated under the **same** fully-specified deployment
prompt (§3), so a prompt-only failure isolates *promptability* rather than
under-specification (Figure~\ref{fig:design}). The behaviors separate sharply
(Figure~\ref{fig:boundary}, Table~\ref{tab:boundary}).

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=0.99\columnwidth]{fig_design.png}
\caption{\textbf{Matched-prompt protocol.} The identical deployment prompt is given to a fine-tuned 0.8B student, a generic-SFT ablation, and prompt-only baselines up to the 9B teacher; each is scored by the instrument matched to the capability.}
\label{fig:design}
\end{figure}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=0.99\columnwidth]{fig_boundary.png}
\caption{\textbf{The boundary.} \emph{Top}: behaviors one clause elicits --- prompt-only reaches parity. \emph{Bottom}: behaviors the prompt describes but cannot install --- prompt-only falls far short of the trained 0.8B student. Values from Table~\ref{tab:boundary}.}
\label{fig:boundary}
\end{figure}
```

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}p{0.30\linewidth} r r@{}}
\toprule
\textbf{Capability (metric)} & \textbf{Prompt-only} & \textbf{Trained} \\
\midrule
Persistence (recall)       & \(\leq\)0.06 / A3 0.000 & \textbf{0.83} \\
Withholding (rate)         & 0.09--0.45            & \textbf{0.61} \\
\midrule
Locale (1\(-\)leakage)     & 0.987                 & 0.987 \\
Role-swap (type F1)        & 0.71--0.86            & 0.75--0.86 \\
Topic (type F1)            & saturated             & saturated \\
\bottomrule
\end{tabular}
\caption{\textbf{The boundary.} \emph{Top}: not-promptable (the 9B teacher and, for persistence, the no-data ablation A3 fail despite the explicit instruction). \emph{Bottom}: promptable (prompt-only parity). Persistence prompt-only is the 9B teacher's zero/few-shot recall; native CoT reaches 0.63 (§5.1). Withholding prompt-only spans B1--B4 (9B teacher 0.45).}
\label{tab:boundary}
\end{table}
```

## 5.1 Not promptable I — multi-turn persistence

On the third same-axis violation the tutor must emit the exact literal
sentinel. Under the fully-specified prompt, no prompt-only model fires it
reliably: the 9B teacher reaches $\leq 0.06$ recall zero/few-shot, and the
no-data ablation A3 fires on **0.000** of positives, while every model
trained on the persistent streams fires (A1 recall **0.83**, 3-seed mean
$0.85 \pm 0.04$).

**The gap is not an artifact of zero-shot prompting.** We ran a prompting
ladder on the strongest prompt-only model (the 9B teacher; few-shot exemplars
span positions {5,7,9,11} so they cannot teach a fixed turn).
Table~\ref{tab:ladder} and Figure~\ref{fig:failure} (left) show few-shot and
an output counting-scaffold do *not* help ($\leq 0.06$); only Qwen3.5's
*native* chain-of-thought moves recall, to 0.63 — still below the trained
0.83, and at 1.6–3.2k reasoning tokens/turn with frequent delivery failures
(the fire decision is reached inside `<think>` but omitted from the visible
answer). Persistence is thus a *relocated* boundary: it resists zero/few-shot
outright and is only partially recovered by native CoT, at an accuracy,
reliability, and inference-cost deficit SFT removes.

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l c@{}}
\toprule
\textbf{9B teacher prompting condition} & \textbf{recall} \\
\midrule
zero-shot instruction            & 0.025 \\
+ few-shot exemplars             & 0.013 \\
+ CoT output scaffold (no-think) & 0.057 \\
+ native chain-of-thought        & \textbf{0.63} \\
\midrule
A1 trained 0.8B (reference)      & \textbf{0.83} \\
\bottomrule
\end{tabular}
\caption{\textbf{Persistence resists prompting up to, but not including, native CoT --- and even that stays below the trained student.} Recall on Persistent-Probe positives (n=159).}
\label{tab:ladder}
\end{table}
```

**Decorrelation, partially validated.** In an isolated matched contrast (A1
4-variant vs A5 fixed-turn-7, 1-epoch budget), decorrelation *removes the
positional recall bias* — the fixed-turn design recalls best at its trained
turn 7 and degrades off-position (overall 0.56), while the 4-variant fires
uniformly across positions (0.82) — but does *not* reduce premature firing
(0.208 vs 0.119). At 0.8B the premature over-firing tracks accumulated
violation count and conversation depth (peaking at turn 9, not the trained
7), so it is threshold-laxity, not a positional shortcut for decorrelation to
suppress.

## 5.2 Not promptable II — pedagogical withholding

The tutor must *withhold* the requested answer and scaffold instead, an
instruction the prompt states explicitly. We score 63 held-out probes under
two judges (Table in Figure~\ref{fig:failure}, right). The trained student
withholds at **0.61**; the no-specialized-data ablation A3 collapses to 0.12,
near the untrained-base rate — clean evidence the behavior is *installed by
demonstration* (A1-vs-A3 clears significance under each judge, $z=5.9$ /
$5.6$, $p \ll 0.001$). Every prompt-only model, including the 9B teacher (0.45),
withholds less than the trained 0.8B student despite the identical
instruction: a model an order of magnitude larger, told plainly to scaffold,
complies on under half of probes. (The student-beats-teacher *comparison* is
directional only at n=63; the boundary rests on the teacher's absolute
non-compliance and the A3 ablation.) Scaffolding requires suppressing the
model's strong helpfulness prior — a clause can describe that suppression but
not produce it.

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=0.99\columnwidth]{fig_failure.png}
\caption{\textbf{The two not-promptable behaviors.} \emph{Left}: on persistence, the 9B teacher stays at \(\leq\)0.06 through few-shot and an output CoT scaffold; only native CoT moves it, to 0.63 --- still below the trained 0.83 (dashed). \emph{Right}: on withholding, every prompt-only condition (incl.\ the 9B teacher, 0.45) falls below the trained 0.61, and the no-data ablation A3 collapses to baseline.}
\label{fig:failure}
\end{figure}
```

## 5.3 Promptable axes — type is prompted, quality is refined

Redirect-axis F1 from a context-blind judge is **uninformative**: it is a
*type* classifier that floors on axes with no context-free surface form
(locale, language, topic) and ceilings on axes every model satisfies
(persona, role-swap), tying A1, B2, and the 9B teacher at 0.409. We therefore
score *quality* on the promptable axes with a pairwise preference (A1 vs A3,
both SFT-only, differing only in the specialized streams). Every axis favors
A1; the largest effect is **role-swap (win-rate 0.87)** — on an axis F1
called saturated parity, so the F1 ceiling concealed a real, large
specialized-data effect. A generic-redirect negative control (both conditions
train on it) sits at 0.55, so the wins are axis-specific, not a global "A1 is
better" artifact. **Locale fidelity is prompt-driven**: Western-default
leakage is statistically indistinguishable across trained and prompt-only
models (A1 1.34%, A3 0.89%, B1 1.34%), and ablating the locale stream does
not increase leakage — it is carried by one prompt clause.
