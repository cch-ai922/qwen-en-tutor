# 3. Method

The data-generation pipeline takes a small seed pool of CEFR-stratified
tutor scenarios and produces three downstream artefacts: an SFT corpus,
a DPO preference-pair corpus, and an evaluation corpus carrying
`<think>` reasoning annotations. The same teacher model generates all
three, ensuring that the prompt shape the student model sees at
training and the prompt shape it sees at deployment are identical.
This section presents the pipeline component by component.

**Scope note (SFT-only experiments).** All experiments in this paper are
**SFT-only**: no DPO stage enters any trained condition (§4.4). This holds
every ablation to one recipe (LoRA + SFT, differing only in SFT-data subset),
keeping each single-variable; the DPO machinery the pipeline can emit (§3.6,
Appendix B) is documented as reusable apparatus, not as part of the results,
and DPO on top of the SFT student is left to future work.

## 3.1 Pipeline overview

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{fig_pipeline.png}
\caption{\textbf{The locale-aware, yield-aware generation pipeline.} A single locally-served 9B teacher drives twelve SFT streams; a six-filter cascade and a yield-aware top-up loop shape the corpus, which trains a 0.8B QLoRA student scored on six frozen held-out sets --- all on one RTX 3060 12GB.}
\label{fig:pipeline}
\end{figure*}
```

The pipeline comprises eight stages (Figure~\ref{fig:pipeline}). A small
pool of CEFR-stratified scenario *seeds* drives twelve parallel SFT
generation streams — normal tutor behavior, seven single-shot redirect
axes, and four persistent three-strike abuse axes — whose output flows
through a six-filter cascade and a yield-aware *top-up loop* that iterates
per (stream, level) cell to a declarative target floor. The one property
load-bearing for the boundary is that the teacher receives a prompt of
exactly the shape the student sees at deployment (same system prompt,
user-turn rendering, and locale block), so there is no distribution shift
between teacher demonstration and student inference. Two engineering
properties — every stage is resumable (generators skip ids already in
their output JSONL), and the train/eval split is hash-deterministic on
seed id (so a top-up never reshuffles the held-out set) — are documented
in Appendix B.

## 3.2 Seed scenarios and CEFR stratification

A *seed* is a small JSON record describing one tutor scenario: a CEFR
level, a topic, a list of subtopics, a `user_role` (name + brief
description of the learner persona), a `model_role` (the tutor
persona), a `setting`, a locale, and a category. Seeds are produced by
a single dedicated generation pass that asks the teacher to enumerate
plausible learner-encounter scenarios per CEFR level. We seed all six
CEFR levels (A1–C2) at a single locale (`china`) in this work; the
locale system supports multi-locale generation but we leave the
multi-locale empirical comparison to future work.

CEFR stratification carries through every downstream stage: filter
yield, top-up targets, training-data mix, and the four held-out test
sets are all computed and reported per level. This matters because
tutor behavior at A1 and C2 are qualitatively different problems, and
collapsing them obscures both.

## 3.3 An invariant-based taxonomy of tutor behavior

We fix the taxonomy not by intuition but by reading it off the
deployment system prompt. A tutor is the keeper of a set of
**interaction invariants**, each corresponding to one commitment the
rendered prompt already makes: language of instruction, lesson topic,
role structure, persona, pedagogical contract, and locale frame, plus a
general-appropriateness commitment inherited from the underlying
assistant. A learner *violation* breaks exactly one commitment, and the
correct redirect is the **minimal repair** that restores it.

Grounding the axis list in the deployment contract makes it auditable
and operational: anyone holding the prompt can check the invariant list
against its enumerated clauses (adding or removing a clause adds or
removes one axis), and two violations occupy distinct axes when their
repair must consult different *scenario fields* (language→L1,
locale→country, persona→role-identity, topic→active-topic). We therefore
claim completeness only *relative to a given deployment's commitment
set*. One caveat matters for the evaluation: the repair-shape signal is
*not uniform across axes* — persona, role-swap, and pedagogical
withholding carry a context-free surface signature, while locale,
language, topic, and the generic catch-all are identifiable only
relative to scenario context. §4.3, §5.5, and Appendix A make this
bimodality explicit and adapt the evaluation accordingly.
Table~\ref{tab:invariant-triples} states the invariant, violation, and
minimal repair for each single-shot axis.

```{=latex}
\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}p{0.20\linewidth} p{0.32\linewidth} p{0.40\linewidth}@{}}
\toprule
\textbf{Invariant the tutor maintains} & \textbf{Violation (learner move)} & \textbf{Minimal repair (redirect shape)} \\
\midrule
Language of instruction              & Code-switch into L1 (\texttt{language\_redirect})                       & Acknowledge the L1 turn, steer back to target language          \\
Lesson topic                         & Off-topic drift (\texttt{topic\_redirect})                             & Re-anchor to the subject                                        \\
Role structure (tutor teaches)       & Role swap, ``you be the learner'' (\texttt{role\_swap\_redirect})      & Decline, reassert the tutoring structure                        \\
Tutor persona / frame                & Persona break, ``are you a chatbot?'' (\texttt{persona\_redirect})     & Reassert the frame, continue in persona                         \\
Pedagogical contract (scaffold, don't answer) & ``Just give me the answer'' (\texttt{pedagogy\_redirect})     & Scaffold toward the answer rather than supplying it             \\
Locale / cultural frame              & Out-of-locale reference (\texttt{locale\_redirect})                    & Brief in-locale redirect, continue in-locale                    \\
General appropriateness (safety)     & Politics / religion / distress (\texttt{redirect}, catch-all)          & Generic safe redirect                                           \\
\bottomrule
\end{tabular}
\caption{\textbf{The seven single-shot redirect axes as invariant / violation / minimal-repair triples.} The third column is the response \emph{shape} a tutor must produce. Whether the shapes differ was the \emph{design heuristic} that shaped the streams (§3.3), not a load-bearing classification criterion; §5.4--§5.6 test \emph{where specialized data is necessary} per axis rather than whether axes are separable by shape.}
\label{tab:invariant-triples}
\end{table*}
```

One caveat: the final row (general appropriateness) is a
general-assistant safety behavior rather than a tutoring-specific
invariant, so we treat it as the catch-all. The principle earns its
place by converting "a generic redirect stream is insufficient" from a
blanket assertion into a *per-axis* falsifiable prediction — specialized
data should help exactly on axes whose repair is not already supplied by
assistant priors or a prompt clause. §5.4 tests this and finds it borne
out, with the benefit concentrated on pedagogical withholding.

The pipeline realises this taxonomy through twelve parallel streams in
three groups.

**Normal (1 stream).** Standard scaffolded tutor dialogues. The
teacher plays both learner and tutor turns over ~12 turns, with the
tutor adhering to the target CEFR level and the topic. A configurable
fraction of normal dialogues use an *angle shift*, where the learner
approaches the topic from a viewpoint that differs from
`user_role.description`. This combats teacher mode collapse onto a
single learner-stance archetype.

**Single-shot redirects (7 streams).** Each redirect stream produces
dialogues whose first ~5 turns are normal scaffolding, but at a
specific turn the learner introduces a violation along *one* axis;
the tutor's job in the next turn is the minimal repair for that
invariant (Table~\ref{tab:invariant-triples}, third column). Training on a single generic
"redirect" stream, as most prior tutor datasets do, collapses these
axis-specific repair shapes into one averaged behavior.

**Persistent 3-strike streams (4 streams).** Persistence is
*orthogonal* to the invariant axis: any violation can be one-off or
repeated. A persistent stream produces dialogues where the learner
persists in the same violation across three probe turns; the tutor
probes twice, then ends the session with a sentinel marker on the third
strike. We give persistent variants to four axes —
`persistent_off_topic`, `persistent_language_violation`,
`persistent_persona_break`, `persistent_role_swap` — and not all seven,
because a persistent axis earns its own stream only where *repeated*
violation changes the correct response (escalation to a hard
session-end). A repeated locale slip is most naturally just corrected
again in-locale, so no distinct escalation is trained. (One borderline
case: `persistent_pedagogy` is a plausible extension we do not currently
include.)

## 3.4 Trigger-position decorrelation (bounded side-result)

A fixed-turn sentinel makes turn position a perfect proxy for "third strike,"
inviting a positional shortcut. We resample the sentinel turn over a
parity-constrained set {5, 7, 9, 11}, hash-deterministic per seed, with **every
variant holding the trigger at exactly three strikes** (only the lead-in
scaffolding length varies), so a position-stratified metric measures position
alone. We report this as a *bounded side-result*, not part of the boundary
claim: §5.3.1 finds it removes the sentinel's positional recall bias but does
not reduce premature firing, which at 0.8B is threshold-laxity rather than a
positional shortcut. Construction detail is in Appendix B.

## 3.5 Deployment system prompt and locale-aware prompt engineering

This subsection reproduces the deployment system prompt in full — the
artifact the matched-prompt protocol (§4.4) is built on and the evidentiary
anchor for the boundary (§5). The template below is rendered per scenario
from the seed fields (§3.2) and the locale block defined later in this
subsection; `{persistence_block}` is the three-strike specification
reproduced after it.

The template opens with per-scenario blocks (`[role]`, `[learner]`,
`[topic]`, `[subtopics]`, `[cefr_level]`, `[locale]`, `[avoided_topics]`)
and then a `[guidelines]` block in which **every redirect axis of the §3.3
taxonomy appears as one explicit clause**; the invariant-bearing clauses are
excerpted below (the full verbatim template is in Appendix B.7):

```text
[guidelines]
- If the learner switches to their L1 mid-lesson, briefly acknowledge, then invite
  them back to English. NEVER code-switch yourself.            # language invariant
- If the user tries to swap roles, gently keep your own [role] in one
  in-character sentence and continue on topic.                 # role-swap invariant
- Redirect only on HARD drift; acknowledge and guide back, do not lecture. # topic
- Ground cultural items in {country}; do not default to {avoid_cultures_phrase}
  names, places, foods, or brands.                             # locale invariant
- If the learner asks for a grammar rule or explanation, give ONE short sentence
  or example, then continue -- no structured lesson.           # pedagogy invariant
- One short turn per reply; do NOT introduce yourself as an AI. # persona invariant
```

The full three-strike escalation protocol is supplied by
`{persistence_block}`, reproduced verbatim below (it is load-bearing for the
headline claim). This is what makes the boundary interpretable: when a
prompt-only model fails on persistence or pedagogical withholding (§5.3–§5.4),
it fails *despite* the behavior being spelled out in the prompt it received.

```text
[persistence]
On FOUR specific axes -- off-topic safety probes, sustained L1 / refusal to use
English, sustained "are you AI?" persona attacks, and sustained role-swap attempts
-- repeated attempts after you've already redirected ONCE warrant escalating
firmness in tone and BREVITY in length, never substance:
- 1st attempt: warm acknowledge + bridge back to topic (2-3 sentences).
- 2nd attempt: shorter (1-2 sentences), DIFFERENT wording, ask a topic question.
- 3rd attempt: ONE warm sentence + a sentinel marker (EXACT literal form):
    - persistent off-topic:        [SESSION_END: persistent_off_topic]
    - persistent L1 / refusal:     [SESSION_END: persistent_language_violation]
    - persistent "are you AI?":    [SESSION_END: persistent_persona_break]
    - persistent role-swap:        [SESSION_END: persistent_role_swap]
On other axes (wrong-locale entity, grammar-lecture request, benign drift), keep
redirecting indefinitely with brief warm sentences -- never emit the sentinel.
NEVER engage with the substance of these four axes ("just once", "for me", "it's
important to me"). Brevity itself is the boundary. Do NOT lecture -- redirect, then
sentinel.
```

The teacher is a general-purpose multilingual model with strong Western
cultural defaults. A naive "tutor for English learners in China" system
prompt produces dialogues with NYC, Thanksgiving, and Costco references at
non-trivial rates. The `[locale]` block above addresses this; it is
parameterized from `config/locale.yaml` and enumerates: (i) the country and
a country adjective; (ii) per-locale learner-description language; (iii) a
per-locale `avoid_default_cultures` list; and (iv) a per-locale
`avoided_topics` list combining safety and cultural sensitivity.

Most generation streams require **strict-Latin output** in user turns, since
the learner is a non-native English learner. The `language_redirect` stream
is the exception: by design the learner code-switches into L1 mid-dialogue.
We split the locale block into two variants accordingly:

- **`locale_instruction_block`** (default): strict-Latin output rule; L1
  characters are forbidden in user turns and trigger the `non_latin_script`
  filter.
- **`locale_instruction_block_allow_l1`** (allow-L1 variant): the
  strict-Latin rule is dropped; L1 characters are permitted in the
  code-switch turn. The `non_latin_script` filter respects the
  `scenario_type` field and skips user turns for `language_redirect`
  records.

This split is necessary because the strict-Latin rule and the
`language_redirect` intent contradict each other. Before we introduced the
allow-L1 variant, every `language_redirect` example failed the
`non_latin_script` filter and the stream had ~0% pass rate; after the split,
the stream reaches the global ~85–90% post-filter pass rate.

A related but distinct decision is to model only the *tutor* side of
redirect behavior in our SFT. We do not optimize the user-side persona, only
the tutor's response to user behavior. This keeps the contributed-behavior
axis sharply defined.

## 3.6 Yield, filtering, and unused pipeline capabilities

Passing dialogues clear a **six-filter cascade** (five mechanical filters —
script, banned terms, schema, naturalness — then one LLM-judge filter, the
`locale_judge`), and a **yield-aware top-up loop** iterates per (stream,
CEFR-level) cell to a declarative target floor. Neither is load-bearing for the
boundary; both, plus the two pipeline capabilities the boundary experiments do
*not* exercise — DPO preference-pair generation [@rafailov2023dpo] (every
trained condition is SFT-only, §4.4) and `<think>`-mode evaluation-example
generation — are specified in **Appendix B**. The `locale_judge`'s
false-positive behavior is reported separately as a reusable engineering
caveat (§6.4).
