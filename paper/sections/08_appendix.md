# Appendix

## A. Macro-F1 over redirect axes: computed and shown to be uninformative

We retire redirect-axis macro-F1 as a primary metric (§5.5) and report
group-appropriate metrics instead (pedagogy withholding rate, §5.4;
quality-aware pairwise preference, §5.6; mechanical rates for the
context-dependent axes, §5.7). Because macro-F1 is the conventional number a
reader may expect, we report it here explicitly and show *why* it is
uninformative, so its retirement is an evidenced decision rather than an
omission.

The redirect-axis judge is context-blind by design (§4.3): it labels the
produced response with a single axis without seeing the violation, topic,
roles, or locale. This makes F1 a *type* classifier — it asks "does this
response read as the correct axis of repair?" — not a *quality* measure. The
consequence is a bimodal per-axis F1 (Table~\ref{tab:macro-f1-bimodal})
that floors on axes whose correct repair has no context-free surface form
and ceilings on axes where every model produces a classifiable response.

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l c l@{}}
\toprule
\textbf{Axis} & \textbf{Per-axis F1 range} & \textbf{Behaviour} \\
\midrule
language   & $\approx 0.375$ (7 of 9 conditions) & near-constant; does not move with model quality \\
locale     & 0.19--0.39 & compressed near floor \\
generic    & 0.087--0.275 & compressed; 9B teacher \emph{tied for lowest} \\
\midrule
persona    & 0.68--0.82 & high, with real spread \\
role\_swap & 0.71--0.86 & high, with real spread \\
\midrule
pedagogy   & mid-range & the only axis both judge-scorable and discriminating \\
\bottomrule
\end{tabular}
\caption{\textbf{Per-axis redirect F1 is bimodal.} The context-dependent axes (language, locale, generic) floor near a structural minimum for \emph{every} condition because the context-blind judge cannot score them from the response alone; the self-contained axes (persona, role\_swap) ceiling because every model produces a classifiable repair of the correct type. Only pedagogy sits in a discriminating mid-range. A macro-average mixes a floored instrument with a ceilinged one.}
\label{tab:macro-f1-bimodal}
\end{table}
```

The macro-average over these axes is consequently flat across conditions of
very different quality: A1 $=$ 0.409, B2 $=$ 0.408, and the 9B teacher B4 $=$
0.409 are **all tied**, despite A1 and the 9B teacher differing by an order of
magnitude in size and despite the pairwise quality eval (§5.6) showing A1
produces better repairs than A3 on all six axes (win-rate 0.57--0.87). A
number that cannot distinguish a 0.8B trained student from a 9B teacher, and
that reports "no effect" where a quality instrument finds a clear one, is not
measuring what the §3.3 claim is about.

Two specific failure modes make this concrete:

- **Ceiling hides quality differences.** Repair-shape F1 calls persona and
  role\_swap *saturated parity*: every condition, including the untrained base
  model, scores 0.68--0.86 because every condition emits a response
  classifiable as the right type. But the quality-aware pairwise eval (§5.6)
  finds role\_swap is in fact the *largest* specialized-data win (A1-win 0.87,
  with one judge preferring A1 on every pair). The F1 ceiling concealed a
  real, large effect.

- **Floor hides everything.** Repair-shape F1 returns a near-constant
  $\approx 0.375$ on language and 0.19--0.39 on locale for conditions spanning
  a 0.8B base model to a 9B teacher. These cells carry no signal; averaging
  them in only dilutes the one axis (pedagogy) that does — and even there, a
  withholding rate (§5.4) is the better instrument.

We therefore report macro-F1 only in this appendix, as a negative result
about the instrument, and base the §5 claims on the quality-aware and
mechanical metrics that measure the right thing for each capability.

## B. Pipeline implementation detail

This appendix collects the generation-pipeline apparatus that §3 summarizes.
None of it is load-bearing for the train-versus-prompt boundary; it is retained
for reproducibility. The corresponding source lives in
`scripts/run_generation.py` and `config/generation.yaml`.

### B.1 Yield-aware top-up with declarative ratio targets

Generation yield is below 100% on every stream because (i) the teacher sometimes
refuses, returns malformed output, or violates the schema; and (ii) the filter
cascade rejects records that fail any of the six filters (net pass-rate 70–90%
depending on stream and locale_judge setting). Rather than over-generate to a
fixed multiple, an **iterative top-up loop** keeps generating until each
(stream, level) cell reaches a target floor:

```text
for each round in 1..MAX_ROUNDS:
    for each cell (stream, level):
        if count(passed_filter(cell)) >= target_per_level(cell):
            mark cell DONE
            continue
        bump generation fraction or n_per_level
        generate(cell)
        run_filter(cell)
    if all cells DONE:
        break
```

The loop is per-cell and bounded by `MAX_ROUNDS=5`; after it, cells still below
target are logged as shortfalls for the operator to act on. Two knob types are
bumped per round: **fraction-gated streams** (the 7 single-shot redirects + 4
persistent streams) pick a hash-deterministic fraction of the seed pool, so no
seed is re-attempted and resume is automatic; **seed-count-gated streams** (the
`normal` stream) instead bump `n_per_level` and run the resumable seed generator.

Targets are declarative. `<stream>_target_per_level` (int) is an absolute floor;
`<stream>_target_ratio` (float in (0,1)) is a share of the final post-filter mix,
computed from the other streams' absolute floors:

$$T_{\text{per\_level}} = \frac{\sum_{s \in \text{absolute}} t_s}{1 - \sum_{r \in \text{ratio}} r_r}, \qquad t^*_r = r_r \cdot T_{\text{per\_level}} \;\; \forall r \in \text{ratio}.$$

If `normal_target_ratio=0.5` and the other streams' absolute targets sum to 270
per level, then $T_{\text{per\_level}} = 270 / (1 - 0.5) = 540$ and normal floors
to 270. The sum of ratio targets must be strictly less than 1 and at least one
stream must use an absolute target, or the equation has no solution.

### B.2 Six-filter cascade

After each generation pass, dialogues flow through six filters
(Table~\ref{tab:filter-cascade}) in a short-circuit cascade ordered
cheap-and-high-catch first, so most rejections occur before the expensive
LLM-judge filter is consulted.

```{=latex}
\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}c l l p{0.50\linewidth}@{}}
\toprule
\textbf{\#} & \textbf{Filter} & \textbf{Cost} & \textbf{Catches} \\
\midrule
1 & \texttt{speaks\_l1\_sanity}   & mechanical           & Degenerate speaks\_l1 records lacking the L1 code-switch turn \\
2 & \texttt{non\_latin\_script}   & mechanical           & Non-Latin characters in assistant or non-language\_redirect user turns \\
3 & \texttt{banned\_terms}        & mechanical           & Politics, religion, self-harm, and locale-sensitive vocabulary \\
4 & \texttt{mode\_consistency}    & mechanical           & \texttt{EvaluationExample} records whose JSON body fails schema validation \\
5 & \texttt{naturalness}          & mechanical heuristic & Stilted, low-perplexity, repetitive prose \\
6 & \texttt{locale\_judge}        & LLM call             & Out-of-locale entities (NYC, Thanksgiving, Costco, etc.) \\
\bottomrule
\end{tabular}
\caption{\textbf{Six-filter cascade.} Filters 1--4 are deterministic and cheap; filter 5 is a heuristic with no model call; filter 6 is the only LLM-judge filter and runs last.}
\label{tab:filter-cascade}
\end{table*}
```

Filter 6, the `locale_judge`, extracts proper-noun entities from each record and
asks the teacher to classify each as `in_locale` or `out_of_locale`; verdicts are
cached in SQLite keyed by `(entity, locale)` so each pair is judged at most once.
Two allowlists guard it: a **per-locale in-locale allowlist** (e.g. `WeChat`,
`Alipay`, `Yunnan`, `Mid-Autumn Festival`) that bypasses the judge call, and a
**sentence-initial common-word allowlist** (e.g. `Absolutely`, `Wi-Fi`, `Will`,
`Plus`, `Line`) that stops the extractor from reading sentence-initial
capitalization as proper-noun status. The false-positive audit motivating these
(57.5% of rejections, $\geq 85$% false-positive, remediated to a 70.1% $\to$ 88.4%
pass rate) is the reusable lesson of §6.4.

### B.3 `<think>`-mode evaluation examples

The pipeline produces evaluation examples for a self-judge mode on the student
(not exercised by the boundary experiments). Each record carries a transcript (a
held-out SFT dialogue with CEFR level, roles, topic, and subtopics prepended to
the user turn) and a teacher response of the form `<think>...</think>{json}`,
where the JSON follows an `EvaluationOutput` schema of per-criterion scores and a
verdict. Three design decisions: **three-way prompt alignment** — the teacher
sees the same prompt shape (system prompt, user turn carrying the transcript,
expected `<think>...</think>{json}` response) the student sees at deployment,
fixing an earlier failure where an empty user turn made small teachers waste
their `<think>` budget hunting for the transcript; a **mandatory non-empty
`<think>` block** (bodies under 30 characters are rejected, else the Qwen teacher
sometimes answers in `/no_think` mode with a valid-but-useless record); and
**`max_tokens=3072`**, since the default 2048 truncates the trailing JSON of ~9B
teacher responses and trips the `mode_consistency` filter.

### B.4 Auto-detected served model

For reproducibility, every record carries a `metadata.generation.model` field
populated by querying the teacher endpoint's `/v1/models` route at startup,
overriding the static `config/generation.yaml` value. This prevents a provenance
bug where the YAML names one model but the endpoint serves another, mis-tagging
records with the wrong teacher.

### B.5 Persistent 4-variant design, codomain, and hash-determinism

Each persistent dialogue uses one of four structural variants
(Table~\ref{tab:variants}); every variant holds the trigger at exactly three
strikes and varies only the lead-in scaffolding, shifting the sentinel to a
different absolute turn without changing what the model must detect.

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}c c l l l@{}}
\toprule
\textbf{Variant} & \textbf{Sentinel turn} & \textbf{Lead-in} & \textbf{Strike turns} & \textbf{Probe turns} \\
\midrule
V1 &  5 & 0 turns          & 0, 2, 4   & 1, 3 \\
V2 &  7 & 2 (turns 0--1)   & 2, 4, 6   & 3, 5 \\
V3 &  9 & 4 (turns 0--3)   & 4, 6, 8   & 5, 7 \\
V4 & 11 & 6 (turns 0--5)   & 6, 8, 10  & 7, 9 \\
\bottomrule
\end{tabular}
\caption{\textbf{The four structural variants of the persistent 3-strike streams.} Sentinel turn and lead-in length vary; variant is hash-deterministic per record. Strike turns are user turns; probe and sentinel turns are assistant turns.}
\label{tab:variants}
\end{table}
```

The four sentinel positions {5, 7, 9, 11} and the seeded (rather than
`random.choice`) variant assignment of §3.4 are both forced, not chosen for
convenience. **Codomain.** Dialogues open on a learner turn, so only odd
positions are valid; the practical range is $\geq 5$ (to fit three strikes and
two probes) and $\leq 11$ (to stay under the 1792-token SFT cap — the longest
variant, V4 at C2, medians 1685 tokens and overflows in ~1% of records, so the
cap binds only beyond turn 11). That leaves exactly {5, 7, 9, 11}.
**Hash-determinism.** Three pipeline invariants require the seeded form.
(i) *Resumability*: generators skip ids already in their output, so a
re-generated record must receive the *same* variant or the distribution drifts
across restarts. (ii) *Train/eval coherence*: the held-out split is itself
hash-deterministic on `seed_id`, so a per-run random assignment would reshuffle
the eval set's variant mix across ablations and make the A1-vs-A5 comparison
(§4.4) ill-defined. (iii) *Uniformity*: `int(sha256(seed_id)[:8]) % 4` is a
seeded pseudo-random function over the four-element codomain, giving the
25/25/25/25 split by construction.

### B.6 Isolated decorrelation contrast (full data for §5.3.1)

These two tables are the evidentiary backing for the §5.3.1 verdict that
decorrelation removes the positional recall bias but not premature firing. We
compare the 4-variant design against the naive fixed-turn-7 design (A5) as an
*isolated, matched* contrast under the deployed axis-specific sentinel
(`[SESSION_END: <axis>]`), holding sentinel format, training budget (1 epoch),
and persistent data fixed, so position design is the only variable (to match
A5's 1-epoch budget we use the 1-epoch variant of A1; the deployed A1 trains 2
epochs, §4.4).

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l l c c@{}}
\toprule
\textbf{Cond} & \textbf{position design} & \textbf{recall} & \textbf{premature} \\
\midrule
A1 (1 epoch) & 4-variant \{5,7,9,11\} & 0.818 & 0.208 \\
A5           & fixed-turn-7           & 0.560 & 0.119 \\
\bottomrule
\end{tabular}
\caption{\textbf{Trigger-position decorrelation, isolated.} \emph{recall} = correct firing on true third-strike positives (n=159); \emph{premature} = firing before the third strike (Persistent-Premature-Probe, n=318). Decorrelation raises recall (0.82 vs 0.56, by removing the fixed-turn's positional bias --- Table~\ref{tab:recall-by-turn}) but does \emph{not} reduce premature firing (0.208 vs 0.119). The deployed 2-epoch A1 has premature 0.107.}
\label{tab:decorrelation}
\end{table}
```

Stratifying recall by the turn at which the third strike lands shows *why*
recall improves: the fixed-turn design fires reliably only near its trained
position, while the 4-variant fires wherever the third strike lands.

```{=latex}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l c c c c c@{}}
\toprule
\textbf{Cond} & \textbf{t=5} & \textbf{t=7} & \textbf{t=9} & \textbf{t=11} & \textbf{overall} \\
\midrule
A1 (1 epoch), 4-variant & 0.836 & 0.818 & 0.826 & 0.778 & 0.818 \\
A5, fixed-turn-7        & 0.478 & \textbf{0.727} & 0.565 & 0.556 & 0.560 \\
\bottomrule
\end{tabular}
\caption{\textbf{Recall by the turn at which the third strike lands.} The fixed-turn design (A5) recalls best at its trained turn 7 (0.727) and degrades off-position (0.48--0.57); the 4-variant recalls uniformly across positions (0.78--0.84), so firing is conditioned on the semantic trigger rather than the turn.}
\label{tab:recall-by-turn}
\end{table}
```

On premature firing, by contrast, decorrelation does not help because the
pathology is not positional: for both conditions the premature rate
5--7$\times$es from one prior violation to two (A5 $0.031\rightarrow0.208$;
A1(1ep) $0.063\rightarrow0.352$), and a by-turn slice peaks at turn **9**, not
the trained turn 7. A genuine turn-7 shortcut would peak at 7 and fall off; the
rise past the trained position is the signature of conversation-depth/count
sensitivity — threshold-laxity that position resampling cannot suppress.

### B.7 Full deployment system prompt template

The complete template §3.5 excerpts. It is rendered per scenario from the
seed fields (§3.2) and the locale block; `{persistence_block}` is the
three-strike specification reproduced in §3.5.

```text
[role]
You are {model_role_name}: {model_role_description}.

[learner]
{user_role_description}.

[topic]
{topic}

[subtopics]
The conversation may naturally start from any of these and can move freely
between them or extend into adjacent practical content the learner might
want to practice:
{subtopics_block}

[cefr_level]
{cefr_level}

[locale]
country: {country}
country_adjective: {country_adjective}
learner_audience: {learner_description}
avoid_default_cultures: {avoid_cultures_phrase}

[avoided_topics]
{avoided_topics_sentence}

[guidelines]
- Sound like a real person, not a textbook. Stay in character as {model_role_name}.
- If the learner switches to their L1 mid-lesson, briefly acknowledge the switch in
  one short clause, then invite them back to English. NEVER code-switch into the
  learner's L1 yourself.                                        # language invariant
- Keep vocabulary, grammar, and sentence length at CEFR {cefr_level} unless the
  learner reaches higher and sustains it.
- If the user tries to swap roles, gently keep your own [role] in one in-character
  sentence and continue on topic.                              # role-swap invariant
- Redirect only on HARD drift; briefly acknowledge and guide back to the topic.
  One or two sentences -- do not lecture.                      # topic invariant
- If the learner brings up an avoided topic, briefly acknowledge and pivot to a
  safe adjacent topic without lecturing.                       # appropriateness
- Ground cultural items in {country}. Do not default to {avoid_cultures_phrase}
  names, places, foods, or brands.                             # locale invariant
- If the learner asks for a grammar rule, conjugation table, vocabulary list, or
  explanation, briefly acknowledge and give ONE short sentence or example, then
  continue -- no bullet lists, no structured lesson.           # pedagogy invariant
- One short turn per reply: 1-3 sentences at A1/A2, 2-4 at B1/B2, 3-4 at C1/C2 --
  then WAIT. Do NOT introduce yourself as an AI or assistant.  # persona invariant

{persistence_block}
```

## C. Training hyperparameters and reproducibility

**Models.** Student base `Qwen3.5-0.8B-Base` (`Qwen3_5ForConditionalGeneration`,
hybrid linear/full-attention interleave; only the language tower is trained,
the vision tower is frozen). Teacher `Qwen3.5-9B-UD-Q4_K_XL` served via
llama.cpp `llama-server` (context 32 768, 80 GPU layers). Teacher (~4 GB) and
trainer (~7.7 GB) cannot co-reside on 12 GB, so the orchestrator swaps them.

**SFT recipe (all trained conditions).** QLoRA [@dettmers2023qlora]: NF4 4-bit,
double-quant, `bfloat16` compute. LoRA [@hu2022lora] rank 16, alpha 32, dropout
0.05, over all seven linear projections (`q/k/v/o_proj`, `gate/up/down_proj`)
of the language tower. 2 epochs, peak LR $2\times10^{-4}$ cosine decay, batch
size 1, gradient accumulation 8 (effective 8), max sequence length 1792,
optimizer `paged_adamw_8bit`, gradient checkpointing on, SDPA attention
(Flash-Attention-2 is unsafe under the linear/full interleave). SFT data
combines the 12-stream filtered corpus with the `<think>`-mode evaluator
examples (`use_all_data: true`). No DPO enters any condition.

**DPO stage (pipeline capability, unused here).** The pipeline can emit DPO
[@rafailov2023dpo] preference pairs from three pools — *register* (a teacher
rewrite in an inappropriate register as `rejected`), *on-policy* (the SFT
student's own response as `rejected`, kept only above a judge margin), and
*sentinel* (the marker-bearing turn as `chosen` against a marker-stripped
`rejected`) — mixed at a default 65/25/10 ratio, with sentinel turns excluded
from the judge-mediated on-policy pool. No paper result depends on it.

**Reproducibility.** One Python codebase. Generation is driven by
`config/generation.yaml`, training by per-condition YAMLs under
`config/paper/`, and the six held-out sets are frozen by
`scripts/build_eval_sets.py` into `eval_sets/` (manifest
`_split_manifest.json`). The teacher name in record metadata is auto-detected
from the `/v1/models` endpoint at startup. The entire run from seeds to
evaluation is resumable.
