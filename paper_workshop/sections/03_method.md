# 3. Method

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{fig_pipeline.png}
\caption{\textbf{The locale-aware, yield-aware generation pipeline.} A single locally-served 9B teacher drives twelve SFT streams; a six-filter cascade and a yield-aware top-up loop shape the corpus, which trains a 0.8B QLoRA student scored on six frozen held-out sets --- all on one RTX 3060 12GB.}
\label{fig:pipeline}
\end{figure*}
```

**Pipeline.** A single locally-served 9B teacher generates all training
data from a pool of CEFR-stratified scenario seeds through twelve parallel
SFT streams, a six-filter cascade, and a yield-aware top-up loop
(Figure~\ref{fig:pipeline}). The teacher receives a prompt of exactly the
shape the student sees at deployment, so there is no distribution shift
between demonstration and inference. Full pipeline mechanics (top-up
equations, filter cascade, hash-deterministic split) are in the appendix and
the released code.

**An invariant-based taxonomy.** We read the redirect axes off the
deployment prompt rather than intuiting them. A tutor is the keeper of a set
of **interaction invariants**, each corresponding to one commitment the
prompt makes; a learner *violation* breaks exactly one, and the correct
redirect is the **minimal repair** that restores it
(Table~\ref{tab:invariant-triples}). This makes the axis list auditable
(one clause ↔ one axis) and converts "a generic redirect stream is
insufficient" into a per-axis falsifiable prediction: specialized data
should help exactly on axes whose repair is not already supplied by
assistant priors or a prompt clause (§5).

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}p{0.30\linewidth} p{0.62\linewidth}@{}}
\toprule
\textbf{Invariant / violation} & \textbf{Minimal repair} \\
\midrule
Language (code-switch to L1)     & Acknowledge, steer back to target language \\
Topic (off-topic drift)          & Re-anchor to the subject \\
Role (``you be the learner'')    & Decline, reassert tutoring structure \\
Persona (``are you a chatbot?'') & Reassert the frame, continue in persona \\
Pedagogy (``just give the answer'') & Scaffold rather than supply the answer \\
Locale (out-of-locale reference) & Brief in-locale redirect \\
Appropriateness (politics/etc.)  & Generic safe redirect (catch-all) \\
\bottomrule
\end{tabular}
\caption{\textbf{The seven single-shot redirect axes} as invariant/violation/minimal-repair triples, read off the deployment prompt.}
\label{tab:invariant-triples}
\end{table}
```

The pipeline realises this through twelve streams: *normal* scaffolded
dialogue (1), *single-shot redirects* (7, one per axis), and *persistent
3-strike* streams (4, for the axes where repeated violation warrants
escalation to a session-end sentinel).

**Deployment prompt.** Every axis appears as an explicit `[guidelines]`
clause, and a `[persistence]` block specifies the full three-strike protocol:
first warm redirect, second shorter redirect, third a warm sentence plus the
exact literal sentinel `[SESSION_END: <axis>]`. This is what makes the
boundary interpretable — when a prompt-only model fails on persistence or
withholding (§5), it fails *despite* the behavior being spelled out.

**Trigger-position decorrelation.** A fixed-turn sentinel makes turn
position a perfect proxy for "third strike," inviting a positional shortcut.
We resample the sentinel turn over a parity-constrained set {5, 7, 9, 11}
(hash-deterministic per seed; every variant holds the trigger at exactly
three strikes, varying only lead-in scaffolding). We report this as
*partially validated* (§5).
