# 2. Related Work

Our study draws on four lines of prior work — synthetic instruction data,
LLM-as-judge filtering, tutor/educational LLMs, and persona/safety
adversarial data — plus the shortcut-learning literature that motivates our
persistence construction. We discuss each, then position the central
contribution: a matched-prompt map of which tutor behaviors require
fine-tuning versus prompting.

## 2.1 Synthetic instruction-tuning data

Using a strong "teacher" to bootstrap instruction data for a smaller
"student" was popularised by Self-Instruct [@wang2023selfinstruct] and
pushed further by Evol-Instruct / WizardLM [@xu2023wizardlm], which
iteratively rewrite instructions to broaden complexity. UltraChat
[@ding2023ultrachat] extends to multi-turn dialogue at scale;
OpenAssistant [@kopf2023openassistant] produced a human-annotated
instruction dataset whose preference labels enabled subsequent RLHF / DPO.
These pipelines target *general* instruction-following with a single
uniform recipe and a generic length/format or safety filter; they do not
decompose the behavior space by axis, nor do they ask which of the
behaviors they teach could have been obtained by prompting alone. The
closest prior work in spirit is Vicuna's filtered-ShareGPT approach
[@chiang2023vicuna] and the Tulu-3 [@lambert2024tulu3] mix-blending recipe,
which compose datasets from sub-pools. Our **declarative ratio-target**
mechanism (§3) generalises mix blending to a yield-aware iterative top-up,
but we use it as apparatus; our contribution is what the resulting data
does — and does not — buy over a fully-specified prompt.

## 2.2 LLM-as-judge filtering

Filtering generated data with another LLM is standard practice. Prometheus
[@kim2024prometheus], JudgeLM [@zhu2023judgelm], and PandaLM
[@wang2023pandalm] propose dedicated judge models; AlpacaEval / MT-Bench
[@zheng2023judging] use frontier models as judges. A growing strand
documents judge limits — position, length, judge-style, and self-preference
bias [@wang2023pandalm; @saito2023verbosity; @panickssery2024selfpreference];
the standard mitigation is multi-judge consensus. Our quality-aware pairwise
eval (§5.6) follows this with a deliberate cross-family constraint —
Prometheus-7B-v2 (Mistral lineage), Llama-3.1-8B-Instruct (Meta), and
Gemma-2-9B-it (Google), all distinct from the Qwen-family teacher — so no
judge shares lineage with the model that produced the student's supervision.
Beyond using judges, we contribute a *negative* result about a judged
metric: §5.5 and Appendix A show that a context-blind per-axis redirect
F1 is a type classifier that cannot measure repair quality, and we document
its bimodal floor/ceiling behavior. Separately, our **`locale_judge`** — a
single-axis in/out-of-locale entity classifier — exhibits a failure mode we
have not seen catalogued: systematic false positives on common-English
sentence-initial words and locally-canonical landmarks, ~85% of its
rejections in our setting (§6.4). The reusable lesson — that
capitalization-based entity filters need aggressive allowlists and periodic
rejection-log audits — generalises beyond locale.

## 2.3 Tutor and educational LLMs

Tutor systems include EduChat [@dan2023educhat] for general educational
dialogue, MathDial [@macina2023mathdial] for math-tutor scaffolding moves,
and language-learning systems [@caines2023chatbots; @tyen2022opendomain].
CEFR-aligned datasets for English learners — EFCAMDAT
[@geertzen2014efcamdat], TLE [@berzak2016tle] — are *learner-produced*
corpora, not tutor-side training data; LearningQ [@chen2018learningq]
provides difficulty-varied QA but is not multi-turn. To our knowledge no
publicly described tutor dataset both stratifies systematically across CEFR
A1–C2 and covers multi-axis redirect / persistent-abuse handling. More
pointedly for this paper: none asks, per behavior, whether the deployment
prompt already suffices. MathDial's explicit scaffolding moves are the
closest analogue to our pedagogical-withholding axis, and our matched-prompt
result — that even a 9B model told to scaffold mostly does not — gives a
data-side reason such moves are worth teaching rather than merely specifying.

## 2.4 Persona, safety, and adversarial dialogue data

Adversarial conversational data is well studied for safety: BBQ
[@parrish2022bbq] targets bias-eliciting forms, HarmBench
[@mazeika2024harmbench] catalogues unsafe inputs, AdvBench [@zou2023advbench]
focuses on jailbreaks. Persona consistency has been studied as a training
objective [@zhang2018personas] and as an attack surface that amplifies
toxicity [@deshpande2023toxicity]. Most of this work treats safety as a
*single-turn* problem — one unsafe prompt, one refusal. Two aspects of the
tutor setting are not addressed by it. First, **multi-axis decomposition
under a matched prompt**: we separate redirect behavior into prompt-derived
axes and then test, axis by axis, whether the prompt clause that names the
axis is sufficient — finding that role-swap and topic deflection are
promptable as *type* while pedagogical withholding is not. Second,
**multi-turn persistence**: our three-strike streams test whether a model
fires a session-end sentinel on the third same-axis violation, a behavior
requiring cross-turn counting that — we show — no amount of prompt
specification elicits prompt-only.

**Shortcut learning and the positional construction.** The persistence
sentinel is a setting where a rare structured marker must fire on a semantic
trigger but is positionally regular in fixed-turn training data — a
dialogue-level instance of *shortcut learning* [@geirhos2020shortcut], well
documented in NLI where annotation artifacts and shallow heuristics let
models succeed without the intended reasoning [@gururangan2018artifacts;
@mccoy2019hans]. The standard remedy is to break the spurious correlation in
the data; our 4-variant construction (§3.4) is the multi-turn
dialogue-sentinel form, with deterministic, parity-constrained resampling.
We are not aware of a directly comparable multi-turn construction. We
report, however, that at 0.8B and matched training budget the construction's
predicted benefit does not materialise as a position-resampling effect
(§5.3); we therefore position it as a tested, bounded construction and a
characterization of *when* the positional shortcut governs behavior, not as
a validated defense.

## 2.5 Positioning

The prompting-versus-fine-tuning trade-off has been examined at the level of
*general* alignment — LIMA [@zhou2023lima] shows a small demonstration set
suffices for broad instruction-following, InstructGPT [@ouyang2022instructgpt]
established instruction tuning over prompting alone, and
[@min2022rethinking] finds in-context demonstrations often supply format
rather than new capability. We localize this question to the *per-behavior*
level for a deployed task model: which specific commitments a fully-specified
prompt already elicits, and which demand demonstration. Our contribution is
thus orthogonal to all four lines above: a per-capability map of the
train-versus-prompt boundary for tutor redirect behavior, established under a
protocol in which the same fully-specified deployment prompt is given to
fine-tuned and prompt-only models alike. Table~\ref{tab:positioning}
positions that one contribution against each prior-work axis; the remaining
rows record the reusable apparatus and the single bounded side-result the
study also produced, which we do not advance as separate contributions.

```{=latex}
\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}p{0.34\linewidth} p{0.58\linewidth}@{}}
\toprule
\textbf{Prior work axis} & \textbf{Relation to this paper} \\
\midrule
\textit{Contribution --- the boundary} & \\
Self-Instruct / Evol-Instruct / WizardLM & Per-capability train-vs-prompt boundary: which taught behaviors a fully-specified prompt already elicits, and which require demonstration \\
Tutor / educational LLMs                  & Matched-prompt evidence that scaffolding/withholding resists prompting even at 9B; prompt-derived CEFR$\times$axis taxonomy anchoring the map \\
Single-turn safety / persona data         & Multi-turn persistence shown un-promptable under a full three-strike prompt spec; per-axis promptability map for redirects \\
\midrule
\textit{Secondary --- reusable apparatus and one bounded side-result} & \\
LLM-judge filtering                       & Negative result on context-blind redirect-axis F1 (type-not-quality, bimodal); quality-aware pairwise recovery; locale-judge allowlist + FP-audit methodology \\
Shortcut learning / spurious cues (NLI)   & Multi-turn dialogue-sentinel instantiation; matched-budget characterization of when the positional shortcut governs (bounded, not a validated defense) \\
\bottomrule
\end{tabular}
\caption{\textbf{Positioning relative to four prior-work axes plus the NLI shortcut-learning literature.} The single contribution is the boundary (top block); the judge-filtering and shortcut-learning rows record reusable apparatus and a bounded side-result (bottom block), not separate contributions. The decorrelation construction is tested-and-bounded (§5.3), not a headline principle.}
\label{tab:positioning}
\end{table*}
```

We demonstrate the boundary at the smallest practical scale (0.8B student on
a consumer GPU), the regime where the question "must this be trained, or
will the prompt do?" is most consequential.
