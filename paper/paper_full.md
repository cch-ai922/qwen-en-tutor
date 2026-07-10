---
title: "What Must Be Trained and What Can Be Prompted: A Per-Capability Study of Tutor Redirect Behavior"
author: "Miles Yung — Independent Research (milesyung2026@gmail.com)"
abstract: |
  Deploying a small language model as an English tutor forces a practical
  decision: for each behavior a deployment prompt already *describes*,
  does an explicit prompt clause *elicit* it, or must it be demonstrated
  through fine-tuning? We answer this per-capability under a
  **matched-prompt** protocol --- the identical fully-specified prompt
  given to a fine-tuned 0.8B student and to prompt-only baselines up to a
  9B teacher --- so a prompt-only failure isolates promptability rather
  than under-specification.
  
  Under the tested small-model and prompting regimes, the behaviors
  separate into *three* operational regimes rather than a binary. **(A)
  Prompt-sufficient**: locale-leakage fidelity reaches prompt-only parity.
  **(B) Prompt-elicitable but data-refinable**: role-swap and topic
  re-anchoring appear prompt-only as a *type*, yet specialized data wins a
  pairwise quality comparison decisively (role-swap win-rate 0.87). **(C)
  Training-dependent (under the tested regimes)**: multi-turn persistence
  (firing a session-end sentinel on the third same-axis violation) and
  pedagogical withholding are named by the prompt but not reliably
  elicited --- the 9B teacher reaches $\leq$ 0.06
  persistence recall zero/few-shot and 0.63 with native chain-of-thought
  (below the student's 0.83, at 1.6--3.2k reasoning tokens/turn), and
  withholds on under half of probes (0.45) versus the student's 0.61
  (paired tests, two judges). The regime-C effects replicate *in
  direction* in a trained Llama-3.2-1B student evaluated against its own
  untrained base (persistence 0.25$\to$ 0.91, withholding
  0.11$\to$ 0.50), separating training from scale within a
  shared base; magnitude is family-dependent.
  
  Reading the regime-A/B axes honestly required retiring the conventional
  context-blind redirect-axis macro-F1 (a *type* classifier that ties
  student and teacher) in favor of capability-specific metrics. All
  experiments run on one RTX 3060 12GB GPU; we release the apparatus and
  analysis scripts.
---



# 1. Introduction

**Problem.** We ask, for each behavior a deployed English tutor needs,
whether a sufficiently explicit system prompt is enough or the behavior
must be demonstrated in fine-tuning. A deployed tutor is governed by a
long system prompt that already *states* most of what it should do —
stay in character, ground culture in the learner's locale, refuse role
swaps, give one short example rather than a grammar lecture, escalate to
a session-end signal under sustained abuse. If stating a behavior were
sufficient, the data-side problem would be trivial.

**Why it matters.** Knowing which behaviors a prompt clause elicits and which it
only describes is exactly the decision a practitioner faces when building a
task-specific model on consumer hardware — spend the data-collection and training
budget only where a prompt will not do. The tutor setting makes this question both
unavoidable and answerable, because each target behavior corresponds to a clause
already present in a realistic deployment prompt (§3.5): redirect behavior spans
several *interaction invariants* (language, topic, role, persona, pedagogical
contract, locale), each a distinct clause we can test in isolation.

**Gap in prior work.** The prompting-versus-fine-tuning question has been
studied at the level of general alignment — a small demonstration set can install
broad instruction-following [@zhou2023lima; @ouyang2022instructgpt], and in-context
demonstrations often convey format more than new capability [@min2022rethinking] —
but not resolved *per behavior* for a deployed task model. Synthetic-instruction
pipelines (§2.1), tutor-LLM datasets (§2.3), and single-turn safety/persona work
(§2.4) target general instruction-following or one-prompt-one-refusal redirection;
none asks, per behavior, whether the deployment prompt already elicits what the
data teaches, nor addresses multi-turn persistence.

We fill this gap with a **matched-prompt per-capability evaluation**: the
same fully-specified deployment prompt — every redirect-axis instruction
and the full three-strike persistence protocol — is supplied at evaluation
to a fine-tuned 0.8B student and to a ladder of prompt-only baselines (0.8B
base, 0.8B instruct, 4B instruct, and the 9B teacher that generated the
training data). Holding the instruction fixed across conditions is what
makes a prompt-only failure interpretable: it isolates whether the behavior
is *promptable* at all (the full logic is in §5.1).

**Contribution.** Our contribution is a single one: **a per-capability map
of the train-versus-prompt boundary** for tutor redirect behavior,
established under a matched-prompt protocol that makes prompt-only failures
interpretable. Under the tested small-model and prompting regimes, the behaviors
separate into **three operational regimes** rather than a binary: **(A)
prompt-sufficient** — a single clause both describes and elicits the behavior and
data adds little (locale-leakage fidelity); **(B) prompt-elicitable but
data-refinable** — the behavior *type* is elicited by the clause (role-swap and
topic re-anchoring reach prompt-only type-parity), yet specialized data materially
improves its *quality*; and **(C) training-dependent under the tested regimes** —
the behavior requires cross-turn state-tracking (multi-turn persistence) or the
suppression of a strong competing prior (pedagogical withholding), which a clause
names but the tested prompts do not reliably elicit. Regime B is what a binary
promptable/not-promptable framing hides.

In support of that map — not as separate contributions — we report an evidenced
negative result on the conventional metric (context-blind redirect-axis F1 ties
student and 9B teacher at 0.409, while a quality-aware pairwise eval prefers the
student on every axis; §5.5, Appendix A), and release the reusable generation
pipeline the study is built on (§3, with a `locale_judge` FP audit, §6.4) plus a
*partially validated* trigger-position decorrelation construction (§5.3.1).

**Results.** Under the matched prompt, the two regime-C behaviors fail
prompt-only in the tested regimes and are installed by SFT. Persistence resists
zero-shot and few-shot prompting on the 9B teacher (recall $\leq 0.06$) and is only
partially recovered by native chain-of-thought (0.63, still below the trained
student's 0.83 and at 1.6–3.2k reasoning tokens per turn), while the
no-persistence-data ablation A3 fires 0% of the time (A3 trains on neither the
specialized nor the persistent streams, §5.1). Withholding stays at 0.09–0.45
prompt-only (9B teacher 0.45) against the trained student's 0.61 (two judges,
n=63), collapsing to near the untrained-base rate when the specialized streams are
ablated. The trained student's persistence survives a full error taxonomy
(precision 0.795, balanced accuracy 0.885, MCC 0.758, benign FPR 0.000; §5.3), so
its recall is genuine discrimination, not over-firing. The regime-A/B axes reach
prompt-only type-parity.

**The regime-C effects replicate in a second trained family.** A Llama-3.2-1B
student, evaluated against its *own* untrained base, lifts *both* regime-C
behaviors far above prompt-only — persistence 0.25$\rightarrow$0.91 and withholding 0.11$\rightarrow$0.50.
Because the student and its control share one base and differ only in training,
this separates *training from scale within that base* (it is not a claim about
scale independence across model sizes, which we do not test). A larger
Llama-3.1-8B prompt-only probe stays low even with chain-of-thought; the direction
is robust across families, the magnitude family-dependent (§6.2). The load-bearing
conditions (A1, A3) are reported over three seeds with mean$\pm$s.d., the
withholding contrasts carry per-judge paired McNemar tests and paired bootstrap
CIs, and the pairwise win-rates carry bootstrap CIs (§4.8,
Table 8, §6.1).

**Scope and non-goals.** We focus on the *data side* and treat the
training recipe as fixed (QLoRA SFT). We do not contribute to
CEFR-leveling itself. We evaluate with a single base/teacher family
(Qwen) — a genuine limitation (§6.2) — and at a single locale
(`china`). We *hypothesize* the persistence and withholding boundaries are less
locale-dependent than the redirect axes, because their formal trigger definitions
(count same-axis strikes; withhold the requested answer) contain no locale-specific
variables; but single-locale evaluation still limits external validity, and we do
not claim locale-independence as established (§6.2).
The 0.8B student on RTX 3060 12GB is a deliberate choice: the boundary is
most consequential precisely where a large general-purpose model is not
deployable.

**Paper organization.** §2 surveys related work. §3 presents the
pipeline and reproduces the deployment prompt (§3.5). §4 specifies the
matched-prompt experimental setup. §5 reports the per-capability boundary
results. §6 discusses limitations, confounds, and methodological lessons.
§7 concludes.



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

## 2.5 Multi-turn instruction following and judge reliability

Two fast-moving literatures bear directly on our setup. First, **multi-turn and
stateful instruction following**: a growing line of benchmarks probes whether
models maintain instructions and structural constraints across turns, where
single-turn success does not predict multi-turn robustness. Our persistence task
is an instance of this — firing on the *third* same-axis strike is a cross-turn
state commitment that a single-turn instruction-following score would miss. Second,
**LLM-as-judge reliability**: judges exhibit position bias, verbosity and
self-preference effects, and correlated failures, and synthetic preference data can
leak stylistic artifacts [@zheng2023judging; @panickssery2024selfpreference]. We
mitigate but do not eliminate these (a three-family judge ensemble distinct from
the teacher, a randomized-order pairwise protocol, a mechanical judge-free
persistence metric, and a generic-redirect negative control), and we flag human
calibration of the judged metrics as future work (§6.3). We avoid an
absolute-novelty claim pending a fuller survey of these quickly-developing areas.

## 2.5.1 Positioning

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
fine-tuned and prompt-only models alike. Table 1
positions that one contribution against each prior-work axis; the remaining
rows record the reusable apparatus and the single bounded side-result the
study also produced, which we do not advance as separate contributions.

| **Prior work axis** | **Relation to this paper** |
| :----------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| *Contribution --- the boundary* |  |
| Self-Instruct / Evol-Instruct / WizardLM | Per-capability train-vs-prompt boundary: which taught behaviors a fully-specified prompt already elicits, and which require demonstration |
| Tutor / educational LLMs | Matched-prompt evidence that scaffolding/withholding resists prompting even at 9B; prompt-derived CEFR$\times$axis taxonomy anchoring the map |
| Single-turn safety / persona data | Multi-turn persistence shown to resist prompting (under the tested regimes) despite a full three-strike prompt spec; per-axis promptability map for redirects |
| *Secondary --- reusable apparatus and one bounded side-result* |  |
| LLM-judge filtering | Negative result on context-blind redirect-axis F1 (type-not-quality, bimodal); quality-aware pairwise recovery; locale-judge allowlist + FP-audit methodology |
| Shortcut learning / spurious cues (NLI) | Multi-turn dialogue-sentinel instantiation; matched-budget characterization of when the positional shortcut governs (bounded, not a validated defense) |

: **Table 1. Positioning relative to four prior-work axes plus the NLI
shortcut-learning literature.** The single contribution is the boundary
(top block); the judge-filtering and shortcut-learning rows record
reusable apparatus and a bounded side-result (bottom block), not
separate contributions. The decorrelation construction is
tested-and-bounded (§5.3), not a headline principle. {#tab:positioning}

We demonstrate the boundary at the smallest practical scale (0.8B student on
a consumer GPU), the regime where the question "must this be trained, or
will the prompt do?" is most consequential.



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

![**The locale-aware, yield-aware generation pipeline.** A single locally-served 9B teacher drives twelve SFT streams; a six-filter cascade and a yield-aware top-up loop shape the corpus, which trains a 0.8B QLoRA student scored on six frozen held-out sets — all on one RTX 3060 12GB.](paper/figures/fig_pipeline.png)

The pipeline comprises eight stages (Figure 1). A small
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

We fix the taxonomy by reading it off the deployment system prompt rather than by
intuition. A tutor keeps a set of **interaction invariants**, each corresponding
to one commitment the prompt already makes: language, topic, role, persona,
pedagogical contract, and locale, plus a general-appropriateness commitment
inherited from the assistant. A learner *violation* breaks exactly one commitment;
the correct redirect is the **minimal repair** that restores it. Grounding the axis
list in the deployment contract makes it auditable — adding or removing a clause
adds or removes one axis — so we claim completeness only *relative to a given
deployment's commitment set*. One bimodality matters for evaluation: persona,
role-swap, and pedagogical withholding carry a context-free surface signature,
while locale, language, topic, and the catch-all are identifiable only relative to
scenario context (§4.3, §5.5, Appendix A adapt the evaluation accordingly).
Table 2 states each single-shot axis.

| **Invariant the tutor maintains** | **Violation (learner move)** | **Minimal repair (redirect shape)** |
| :-------------------------------------------- | :----------------------------------------------------- | :----------------------------------------------------- |
| Language of instruction | Code-switch into L1 (`language_redirect`) | Acknowledge the L1 turn, steer back to target language |
| Lesson topic | Off-topic drift (`topic_redirect`) | Re-anchor to the subject |
| Role structure (tutor teaches) | Role swap, "you be the learner" (`role_swap_redirect`) | Decline, reassert the tutoring structure |
| Tutor persona / frame | Persona break, "are you a chatbot?" (`persona_redirect`) | Reassert the frame, continue in persona |
| Pedagogical contract (scaffold, don't answer) | "Just give me the answer" (`pedagogy_redirect`) | Scaffold toward the answer rather than supplying it |
| Locale / cultural frame | Out-of-locale reference (`locale_redirect`) | Brief in-locale redirect, continue in-locale |
| General appropriateness (safety) | Politics / religion / distress (`redirect`, catch-all) | Generic safe redirect |

: **Table 2. The seven single-shot redirect axes as invariant /
violation / minimal-repair triples.** The third column is the response
*shape* a tutor must produce. Whether the shapes differ was the *design
heuristic* that shaped the streams (§3.3), not a load-bearing
classification criterion; §5.4--§5.6 test *where specialized data is
necessary* per axis rather than whether axes are separable by shape.
{#tab:invariant-triples}

The final row (general appropriateness) is a general-assistant safety behavior, so
we treat it as the catch-all. The taxonomy earns its place by converting "a generic
redirect stream is insufficient" into a *per-axis* falsifiable prediction —
specialized data should help exactly on axes whose repair is not already supplied
by assistant priors or a prompt clause — which §5.4 finds borne out, concentrated
on pedagogical withholding.

The pipeline realises this taxonomy through twelve parallel streams in three
groups.

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
invariant (Table 2, third column). Training on a single generic
"redirect" stream, as most prior tutor datasets do, collapses these
axis-specific repair shapes into one averaged behavior.

**Persistent 3-strike streams (4 streams).** Persistence is *orthogonal* to the
invariant axis: any violation can be one-off or repeated. A persistent stream has
the learner persist in the same violation across three probe turns; the tutor
probes twice, then ends the session with a sentinel on the third strike. We give
persistent variants to four axes (`persistent_off_topic`,
`persistent_language_violation`, `persistent_persona_break`,
`persistent_role_swap`) and not all seven, because a persistent stream earns its
place only where *repeated* violation changes the correct response (escalation to a
hard session-end); a repeated locale slip is just corrected again in-locale.

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

The teacher is a general-purpose multilingual model with strong Western defaults
(a naive "tutor in China" prompt yields NYC/Thanksgiving/Costco references at
non-trivial rates); the `[locale]` block above, parameterized from
`config/locale.yaml`, addresses this by enumerating the country, per-locale learner
language, an `avoid_default_cultures` list, and an `avoided_topics` list. One
locale-engineering detail (full description in Appendix B): most streams require
strict-Latin user turns, but `language_redirect` by design code-switches into L1,
so we split the locale block into default (strict-Latin) and allow-L1 variants — a
split that lifts the `language_redirect` pass rate from ~0% (every record failing
the `non_latin_script` filter) to the global ~85–90%. We model only the *tutor*
side of redirect behavior in SFT, keeping the contributed-behavior axis sharply
defined.

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

**Decoding and exact-generation provenance.** Because two of our metrics turn on
*exact-string* emission (the sentinel) and on prompt-only baselines, we fix and
report the generation settings. Mechanical sentinel/premature/FP evaluations use
**greedy decoding** (temperature 0), so those results are deterministic given the
checkpoint; the judged withholding/pairwise generations use temperature 0.7 with a
fixed generation seed. The teacher and prompt-only baselines are served via
llama.cpp with the model's native chat template and stop tokens. Appendix C pins
the exact library versions (`transformers`, `bitsandbytes`, PyTorch, CUDA,
llama.cpp commit), the quantized-checkpoint source and revision, chat templates,
and per-stage decoding parameters (temperature, top-$p$/top-$k$, max new tokens,
seed) so every reported number is reproducible from the released configs.

## 4.2 Training data composition

Table 3 reports per-stream, per-CEFR-level
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

| **Stream** | **A1** | **A2** | **B1** | **B2** | **C1** | **C2** | **Total** |
| :-------------------------------- | :-- | :-- | :-- | :-- | :-- | :-- | :----- |
| normal | 204 | 335 | 355 | 351 | 315 | 235 | 1 795 |
| redirect (generic) | 102 | 169 | 166 | 168 | 21 | 20 | 646 |
| language_redirect | 5 | 8 | 7 | 9 | 2 | 5 | 36 |
| locale_redirect | 6 | 8 | 7 | 10 | 8 | 9 | 48 |
| pedagogy_redirect | 6 | 8 | 8 | 9 | 7 | 8 | 46 |
| persona_redirect | 6 | 8 | 7 | 9 | 7 | 9 | 46 |
| role_swap_redirect | 6 | 8 | 7 | 9 | 7 | 7 | 44 |
| topic_redirect | 6 | 6 | 7 | 9 | 6 | 7 | 41 |
| persistent_language_violation | 29 | 38 | 26 | 30 | 20 | 22 | 165 |
| persistent_off_topic | 22 | 34 | 30 | 24 | 10 | 17 | 137 |
| persistent_persona_break | 28 | 42 | 27 | 31 | 30 | 48 | 206 |
| persistent_role_swap | 17 | 44 | 36 | 25 | 20 | 22 | 164 |
| **TOTAL** | **437** | **708** | **683** | **684** | **453** | **409** | **3 374** |

: **Table 3. Filtered SFT corpus composition by stream and CEFR level.**
Counts are post-filter passing records. The six specialized redirect
streams (language, locale, pedagogy, persona, role_swap, topic) carry
$\approx$ 7--10 records per axis per CEFR level --- intentionally
thinner than the generic streams, a known limitation we account for by
leaning only on the large, seed-stable per-axis effects (§5.6).
{#tab:sft-composition}

## 4.3 Held-out evaluation sets

The held-out split is computed by hashing each scenario seed id and
assigning the bottom 20% to the eval pool (`hashlib.sha256(seed_id)
[:8]` interpreted as integer, modulo 100). This split is
deterministic and immutable across runs, so growing the training
corpus does not contaminate evaluation.

We evaluate on six held-out sets, summarized in
Table 4 and detailed below (the table shows a seventh
row, Locale-Leakage, which reuses the Tutor-Scenario scenarios and so is
not counted as a distinct set):

| **Eval set** | **N** | **Composition** |
| :--------------------------- | :------------- | :--------------------------------------------- |
| Tutor-Scenario | 224 | 6 CEFR levels (19--50 per level), no axes |
| Redirect-Probe | 143 | 7 redirect axes, 6 CEFR levels |
| Persistent-Probe | 159 | 4 persistence axes, 6 CEFR levels (positives) |
| Persistent-Premature-Probe | 318 | under-threshold (vc=1,2) $\times$ turn-depth |
| Persistent-FP-Probe | 240 | 4 trained positions $\times$ 60 (40 per level) |
| Persistent-OffPosition-Probe | 60 | off-grid positions {13, 15}, 4 axes |
| Locale-Leakage | 224 | same scenarios as Tutor-Scenario |
| **TOTAL** | **1 144**$^{\ast}$ |  |

: **Table 4. Held-out evaluation sets.** All sets are filtered to
`locale=china` and constructed by `scripts/build_eval_sets.py` from the
bottom-20% hash-modulo split of scenario seed ids (§4.3). $^{\ast}$TOTAL
excludes the Locale-Leakage row, which reuses the same 224 cold-start
scenarios as Tutor-Scenario (counting it would double-count); the
remaining six rows sum to 1 144. Persistent-Probe reports positives only
(n=159, the recall denominator in §5.3). {#tab:eval-sets}

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

| **Tag** | **Data** | **Method** | **Purpose** |
| :-- | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :------- | :--------------------------------------------------------------------- |
| A1 | All 12 streams, 4-variant persistent ($\{5,7,9,11\}$), axis-specific sentinel `[SESSION_END: <axis>]` | SFT only | Full system (headline condition). |
| A3 | **Normal + generic-redirect streams only** --- drops *both* the 6 specialized redirect streams *and* the 4 persistent streams (1 895 records vs A1's 3 342) | SFT only | *Generic-SFT baseline*; §5.4 taxonomy contrast (A1 vs A3). |
| A5 | A1 with persistent rebuilt as fixed-turn-7, axis-specific sentinel `[SESSION_END: <axis>]` | SFT only | §5.3.1 decorrelation contrast (A1 vs A5): the naive fixed-turn design. |

: **Table 5. Trained-ablation matrix.** All three share the same base
(`Qwen3.5-0.8B-Base`), LoRA recipe, and SFT hyperparameters; they differ
only in the SFT-data subset. None use DPO. **A3 is trained on the
normal + generic-redirect streams only** --- it contains *no*
specialized-redirect and *no* persistent data (materialized by
`setup_paper_ablation_data.py`; verified against the on-disk
`data/sft_filtered_a3/` dir). Its 0.000 persistence recall therefore
reflects the *absence* of persistence demonstration, not a failure
despite it. A1 (4-variant) and A5 (fixed-turn-7) form the isolated
trigger-position decorrelation contrast, both using the deployed
axis-specific sentinel; to match A5's 1-epoch budget the contrast uses a
1-epoch variant of A1 (§5.3.1). {#tab:trained-ablations}

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
not only against prompting. The two contrasts are **A1 vs A3** (taxonomy — do
the specialized/persistent streams add anything beyond the generic redirect?) and
**A1 vs A5** (decorrelation — 4-variant positions {5,7,9,11} vs fixed-turn-7, both
using the deployed axis-specific sentinel and the same persistent data, reported at
a matched 1-epoch budget in §5.3.1). We caution that A1-vs-A3 is *not* a
single-stream contrast: A3 removes several streams at once (all specialized and all
persistent) and trains on fewer records (1,895 vs 3,342), so it receives fewer
optimizer steps at matched epochs. A1-vs-A3 therefore establishes that *some*
combination of the removed streams is necessary for a capability, not that any one
stream is; budget-matched leave-one-stream-out ablations (§6.3) are needed to
attribute an effect to a single stream.

**Authoritative condition composition and data-lineage audit.**
Table 6 is the single authoritative source for what each
reported condition trained on and for train/eval separation
(`scripts/score_paper_composition_audit.py`). The split is at the level of the
source **scenario seed id** (a content hash; all variants of a scenario share it,
§4.3), so no variant of an eval scenario can appear in training. We verify this
holds for every evaluated condition: **0 of the 224 held-out eval scenario ids
appear in A1's or A3's training data**. This audit also subsumes the
`<think>`-mode evaluator-example lineage question: any such example that entered an
SFT dir contributes its scenario id to the training-id set, so an eval-seed leak
would surface here — none does.

| **Cond.** | **Normal** | **Gen. redir.** | **Spec. (of 6)** | **Persist. (of 4)** | **Records** | **Eval-id leaks** |
| :---- | :----- | :---------- | :----------- | :-------------- | :------ | :------------ |
| A1 | yes | yes | 6/6 | 4/4 | 3 342 | 0 / 224 |
| A3 | yes | yes | 0/6 | 0/4 | 1 895 | 0 / 224 |

: **Table 6. Authoritative condition composition and data-separation
audit.** Stream membership and record counts from the materialized SFT
dirs; "Eval-id leaks" = held-out eval scenario ids appearing in the
condition's training data (of 224). Both reported conditions are clean.
`score_paper_composition_audit.py`. {#tab:composition-audit}

**Zero-shot baselines** (Table 7) apply a
tutor-style system prompt to an off-the-shelf checkpoint:

| **Tag** | **Checkpoint** | **Purpose** |
| :-- | :----------------------------------- | :------------------------------------------------------------------------------- |
| B1 | Qwen3.5-0.8B-Base (raw, no training) | Lower bound: shows training matters at all |
| B2 | Qwen3.5-0.8B post-trained | Same-size off-the-shelf comparison |
| B3 | Qwen3.5-4B post-trained | Larger same-family comparison |
| B4 | Qwen3.5-9B (4-bit, via llama-server) | Distillation upper bound (the teacher; no longer in the judge ensemble per §4.5) |

: **Table 7. Zero-shot baselines.** A tutor-style system prompt applied
to an off-the-shelf checkpoint, no training. {#tab:zeroshot-baselines}

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
Table 10.

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
the per-axis pairwise win-rates §5.6; values in Table 8
and each section). With three points we report the spread transparently
rather than a cross-seed significance test. The decorrelation conditions (A5
and A1's 1-epoch variant) stay single-seed by design: their role is the
position contrast of §5.3.1, whose verdict does not turn on initialisation
variance. Mechanical metrics (sentinel firing, premature firing, locale
leakage) are otherwise point estimates; the pairwise win-rate carries
bootstrap 95% CIs over 1000 resamples where $n$ supports them; the
context-dependent rates ($n\leq25$, §5.7) carry a small-$n$ caveat; and the
withholding rate ($n=63$) carries, on the load-bearing A1-vs-A3 contrast, a
**paired McNemar exact test** (the conditions are judged on the identical prompts,
so the comparison is paired by item) together with a **paired bootstrap 95% CI**
over prompts (§5.4). Where we do not detect a difference (e.g.\ locale-leakage
parity, §5.7) we treat it as absence of evidence, not evidence of equivalence, and
report the interval rather than asserting parity. The retired redirect-axis F1
(Appendix&nbsp;A) is not used for any claim.

The single largest mechanical gaps are defended by magnitude rather than by
reseeding: A3 fires the sentinel on 0.000 of positive probes versus the
trained $\geq 0.83$, a separation no plausible initialisation variance can
close (§6.1). Full reproducibility detail (configs, frozen eval-set
manifest, resumability) is in Appendix C.



# 5. Results: the train-versus-prompt boundary

## 5.1 Protocol and conditions

Every condition is evaluated under the **same** fully-specified deployment prompt
(§3.5; Figure 2) — every redirect-axis clause and the complete
three-strike persistence block — so a prompt-only failure isolates *promptability*
rather than under-specification. Trained conditions see the prompt at train and
eval; prompt-only baselines see it at eval.

![**Matched-prompt protocol.** The identical fully-specified deployment prompt is supplied to a fine-tuned 0.8B student, a generic-SFT ablation, and prompt-only baselines up to the 9B teacher; each is scored by the instrument matched to the capability under test. Because the instruction is present for every condition, a prompt-only failure isolates promptability rather than under-specification.](paper/figures/fig_design.png)

Conditions (defined in §4.4, Table 5): the
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

![**Three regimes of the train-versus-prompt boundary.** Each capability is scored by its own validated, capability-specific metric (no heterogeneous metrics are combined on one axis, and the retired context-blind axis-F1 is not used here; §5.5). *Regime A (prompt-sufficient)*: locale fidelity — prompt-only reaches parity. *Regime B (prompt-elicitable but data-refinable)*: role-swap and topic re-anchoring — the behavior type appears prompt-only, but specialized data wins the pairwise quality comparison (§5.6). *Regime C (training-dependent under the tested regimes)*: multi-turn persistence and pedagogical withholding — prompt-only falls far short despite the identical instruction. Persistence prompt-only is the 9B teacher’s zero/few-shot recall ( $\le$ 0.06); withholding prompt-only is the 9B teacher (0.45). Interval estimates and full statistics in Table 8.](paper/figures/fig_boundary.png)

**Three regimes, not a binary** (Figure 3): **(A)
prompt-sufficient** — a clause elicits the behavior and data adds little (locale
fidelity); **(B) prompt-elicitable but data-refinable** — the behavior *type*
appears prompt-only, yet specialized data materially improves its *quality*
(role-swap and topic, §5.6); **(C) training-dependent under the tested regimes** —
the prompt names the behavior but does not reliably elicit it, and SFT installs it
(persistence, withholding). Regime B is what a binary framing hides. Each row of
the summary figure uses its own validated capability-specific metric (no
heterogeneous metrics on one axis; the retired context-blind axis-F1 is not used,
§5.5). The remainder of §5 establishes each row; Table 8
collects every headline number with its sample size and dispersion (three-seed
s.d., bootstrap CI, or paired test as applicable; §4.8, §6.1).

| **Headline result** | **Metric** | **$n$** | **Value** | **Dispersion / test** |
| :------------------------------------ | :------------------- | :-------- | :------------ | :---------------------------------------------------------------------- |
| Persistence, A1 (trained) | sentinel recall | 159 | 0.83 | 3-seed mean $0.85\pm0.04$ (42/123/7) |
| Persistence, A3 (generic-SFT) | sentinel recall | 159 | 0.000 | point estimate; gap to $\geq 0.83$ magnitude-defended (§6.1) |
| Persistence, 9B teacher | sentinel recall | 159 | $\leq 0.06$ | single-run (off-the-shelf); native CoT 0.63 |
| Withholding, A1 (trained) | withholding rate | 63 | 0.611 | 3-seed $0.63\pm0.08$; A1-vs-A3 paired McNemar $p<10^{-6}$ (both judges) |
| Withholding, A3 (generic-SFT) | withholding rate | 63 | 0.119 | 3-seed $0.13\pm0.01$; non-overlapping every seed |
| Withholding, 9B teacher | withholding rate | 63 | 0.452 | A1-vs-teacher $z=1.96$/$1.62$ (edge; directional only) |
| Locale leakage, A1 / A3 | Western-default rate | 224 | 1.34% / 0.89% | 3-seed $0.022\pm0.008$ / $0.016\pm0.007$; indistinguishable |
| Pairwise, role_swap (A1 vs A3) | A1 win-rate | --- | 0.87 | 3-seed $0.82\pm0.06$; bootstrap 95% CI (§5.6) |
| Pairwise, language (A1 vs A3) | A1 win-rate | --- | 0.75 | 3-seed $0.76\pm0.06$ |
| Pairwise, generic (control) | A1 win-rate | 60 | 0.55 | 3-seed $0.53\pm0.05$; near parity |
| Context-dependent (locale/lang/topic) | mechanical rate | $\leq 25$ | --- | point estimates; underpowered, suggestive only (§5.7) |

: **Table 8. Statistical summary of the headline results.** Every
load-bearing number with its sample size and dispersion. Three-seed s.d.
is reported for the reseeded A1/A3 conditions (§4.8); off-the-shelf
baselines (B1--B4) carry no training seed and are single-run. The two
mechanical results the boundary rests on --- A3 persistence 0.000 vs
trained $\geq 0.83$, and the A1-vs-A3 withholding gap significant under
each judge --- are the ones that do not turn on seed variance.
Comparisons flagged \"directional\" or \"underpowered\" are reported as
such throughout §5 and are not the basis of any boundary claim (§6.1).
{#tab:stat-summary}

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
(Table 9), not a single recall number
(`scripts/score_paper_persistence_taxonomy.py`).

| **Condition** | **Recall** | **Prem.@2** | **Benign FPR** | **Precision** | **Bal. acc** | **MCC** |
| :------------------- | :----- | :------ | :--------- | :-------- | :------- | :------- |
| A1 student (SFT) | 0.830 | 0.201 | 0.000 | 0.795 | 0.885 | 0.758 |
| A3 (generic-SFT) | 0.000 | 0.000 | 0.000 | --- | 0.500 | 0.000 |
| 0.8B base (prompt) | 0.000 | 0.000 | 0.008 | 0.000 | 0.497 | $-0.035$ |
| 4B instruct (prompt) | 0.063 | 0.044 | 0.000 | 0.526 | 0.523 | 0.121 |
| 9B teacher (prompt) | 0.025 | 0.025 | 0.000 | 0.500 | 0.509 | 0.071 |

: **Table 9. Persistence as a full error taxonomy, not recall alone.**
Recall on the positive probe (n=159); Prem.@2 = premature firing with
only two prior strikes (sub-threshold); Benign FPR = firing in a fully
benign context; Precision, balanced accuracy, and Matthews correlation
(MCC) pool the positive probe with the premature and benign negatives
into one confusion matrix, so recall inflated by over-firing is
penalized. The trained student's 0.83 recall is *not* an over-firing
artifact: benign FPR is 0.000, malformed-marker and duplicate-fire rates
are 0.000 (not shown), and MCC is 0.758 --- genuine discrimination.
Every prompt-only condition and the generic-SFT ablation collapse to MCC
$\approx 0$. Scored by `score_paper_persistence_taxonomy.py`.
{#tab:persistence-taxonomy}

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

| **B4 (9B) prompting condition**   | **mechanism**                 | **recall** |
| :-------------------------------- | :---------------------------- | :----- |
| zero-shot instruction             | instruction only              | 0.025      |
| \+ few-shot exemplars             | 4 worked 3-strike dialogues   | 0.013      |
| \+ CoT output scaffold (no-think) | forced strike-tally in output | 0.057      |
| \+ native chain-of-thought        | Qwen3.5 `/think` reasoning    | **0.63**   |
| A1 trained 0.8B (reference)       | SFT                           | **0.83**   |

: **Table 10. Persistence resists prompting up to, but not including,
native chain-of-thought --- and even that does not reach the trained
student.** Recall = fraction of true third-strike positives on which the
deployment-visible answer emits the sentinel. Few-shot exemplars and an
output-forced counting scratchpad leave the 9B teacher at $\leq 0.06$
(no better than zero-shot). Only Qwen3.5's *native* reasoning (`/think`)
moves the needle, to 0.63 --- substantially closing but not closing the
gap to the trained 0.8B student (0.83). The A1 reference recall is 0.83
(seed 42; three-seed mean $0.85\pm0.04$ over 42/123/7, §4.8); the
prompt-only-vs-trained gap ($\leq 0.06$ / 0.63 vs 0.83) is far too large
for seed variance to close. The B4 prompting-ladder rows are single-run
(off-the-shelf model, no training seed).
{#tab:persistence-prompting-ladder}

![**The two not-promptable behaviors.** *Left*: on persistence, the 9B teacher stays at $\le$ 0.06 recall through few-shot and an output CoT scaffold; only native chain-of-thought moves it, to 0.63 — still below the trained student’s 0.83 (dashed). *Right*: on withholding, every prompt-only condition (including the 9B teacher, 0.45) falls below the trained student’s 0.61, and the no-specialized-data ablation (A3) collapses to near the untrained-base rate.](paper/figures/fig_failure.png)

Two costs make the native-CoT result a *relocation* of the boundary, not a
refutation (Figure 4). First, **accuracy**: even with unrestricted
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
Appendix B.6 (Table 17 and Table 18). We
report the decorrelation construction as *partially validated* and do not lean on
it for the boundary claim; the robust, budget-independent result of this section
is the persistence row of Figure 3: **persistence is installed by
SFT under the tested regimes.**

## 5.4 Regime C, II — pedagogical withholding

The pedagogy axis tests whether the tutor *withholds* the answer and scaffolds
instead of supplying it — instructed explicitly in the deployment prompt ("give ONE
short sentence or example, then continue — no structured lesson", §3.5). We report
a **withholding rate**: the fraction of probes on which the model declined to supply
the requested answer and scaffolded instead. We score **63 held-out pedagogy
probes** under the matched prompt with the two binary-capable judges (Llama-3.1-8B,
Gemma-2-9B; Prometheus emits rubric-only output, §4.5).
Table 11 reports each judge's rate and their mean.

| **Condition**       | **demonstrated?** | **Llama-3.1** | **Gemma-2** | **mean**  |
| :--------------------- | :-------------- | :--------- | :------- | :----- |
| A1 (full SFT)       | yes               | 0.571         | 0.651       | **0.611** |
| A3 (no specialized) | no                | 0.079         | 0.159       | 0.119     |
| B1 0.8B-base        | no                | 0.127         | 0.048       | 0.087     |
| B2 0.8B-instruct    | no                | 0.302         | 0.317       | 0.310     |
| B3 4B-instruct      | no                | 0.286         | 0.302       | 0.294     |
| B4 9B-teacher       | no                | 0.397         | 0.508       | 0.452     |

: **Table 11. Pedagogical withholding under a matched prompt (n=63, two
judges).** The instruction to withhold is present for every condition.
The trained student withholds at 0.61 (mean over the two judges); the
no-specialized-data ablation A3 collapses to 0.12, near the
untrained-base rate (B1, 0.09). Prompt-only models, including the 9B
teacher (0.45), withhold far less than the trained 0.8B student despite
the identical instruction. Per-judge rates are each over the full 63
probes (21 original + 42 fresh held-out, pooled). Per-judge rates shown
are seed 42; the trained conditions are confirmed across three seeds
(42/123/7): withholding A1 $0.63\pm0.08$ vs A3 $0.13\pm0.01$,
non-overlapping at every seed (§4.8). B1--B4 are off-the-shelf and carry
no training seed. {#tab:withholding}

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
distributions do not overlap at any seed (§4.8, Table 11).

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

| **Axis**  | **Llama** | **Prometheus** | **Gemma** | **mean** |
| :--------------- | :-------- | :----------------- | :-------- | :------ |
| role_swap | 0.83      | 0.78           | 1.00      | **0.87** |
| language  | 0.57      | 0.86           | 0.81      | 0.75     |
| locale    | 0.64      | 0.72           | 0.68      | 0.68     |
| pedagogy  | 0.71      | 0.67           | 0.62      | 0.67     |
| persona   | 0.55      | 0.61           | 0.68      | 0.61     |
| topic     | 0.45      | 0.68           | 0.58      | 0.57     |
| overall   | 0.61      | 0.71           | 0.71      | ---      |

: **Table 12. Pairwise quality, A1 vs A3 (specialized stream vs none,
both SFT-only).** A1 win-rate; 0.5 is parity. Every axis favors A1.
role_swap is the largest effect (0.87, Gemma preferring A1 on every
pair) --- on an axis the context-blind F1 reported as *saturated
parity*. The F1 ceiling concealed a real, large quality effect. Per-axis
values shown are seed 42. Across three seeds (42/123/7) the two largest
effects reproduce --- role_swap $0.82\pm0.06$ and language $0.76\pm0.06$
--- while the remaining axes cluster at 0.54--0.67 with larger seed
variance (locale in particular regresses toward parity, $0.54\pm0.12$);
we therefore lean only on role_swap and language. The generic-redirect
negative control stays near parity across seeds ($0.53\pm0.05$, §4.8).
{#tab:pairwise}

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
teacher (§5.3, Table 10): few-shot and an
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
holds outside Qwen. Table 13 collects the persistence
recall.

| **Model** | **Condition** | **Persist. recall** | **Withhold rate** |
| :----------- | :------------------------ | :------------------------ | :------------ |
| Qwen-0.8B | prompt-only (9B teacher) | $\leq$ 0.06 | 0.09--0.45 |
| (in-family) | trained (A1) | 0.83--0.85 | 0.61 |
| Llama-3.1-8B | prompt-only, zero-shot | 0.27 | 0.22--0.32 |
|  | prompt-only, prompted CoT | 0.55 | --- |
| Llama-3.2-1B | prompt-only (untrained) | 0.25 | 0.11 |
|  | **trained, full SFT (A1)** | **0.91** | **0.50** |

: **Table 13. The boundary replicates in a second trained family.** On
*both* not-promptable behaviors, training the *same*
Llama-3.2-1B-Instruct base --- evaluated against its own untrained
control under the identical deployment prompt --- lifts the behavior far
above prompt-only: persistence recall $0.25\!\to\!0.91$, withholding
$0.11\!\to\!0.50$ (two judges). Because the trained student and the
prompt-only control share one base, this isolates *training* from scale.
The 8B prompt-only rows show the behavior stays low even for a much
larger model. Direction is robust across families; magnitude is
family-dependent (the Llama trained withholding 0.50 is below Qwen's
0.61). {#tab:crossfamily}

This is the load-bearing generalization result (Table 13):
because the trained Llama-1B student and its prompt-only control share one base,
the contrast separates *training from scale within that base*, and it holds on both
regime-C behaviors; the 8B prompt-only rows confirm the gap is not closed by scale
alone (persistence 0.27$\rightarrow$0.55 even with CoT), so the effect is not a Qwen artifact.

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
partially recovers persistence (0.06$\rightarrow$0.63, §5.3) where few-shot and an output
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



# 7. Conclusion

We asked, per capability, which behaviors a deployed tutor needs are elicited by
an explicit system prompt and which are installed by fine-tuning. Evaluating a
fine-tuned 0.8B student and a ladder of prompt-only baselines up to a 9B teacher
**under the same fully-specified deployment prompt**, and under the tested
small-model and prompting regimes, we find the behaviors separate into **three
operational regimes** rather than a binary. *(A) Prompt-sufficient*: locale-leakage
fidelity reaches prompt-only parity. *(B) Prompt-elicitable but data-refinable*:
role-swap and topic re-anchoring appear prompt-only as a type, yet specialized data
wins the pairwise quality comparison (role-swap 0.87) on an axis the context-blind
F1 called saturated parity. *(C) Training-dependent under the tested regimes*:
persistence resists zero-shot and few-shot prompting (recall $\leq 0.06$), recovers
only partially under native chain-of-thought (0.63, below the trained 0.83 and at
heavy inference cost), and fires 0.000 for the no-persistence-data ablation, while
withholding stays at 0.09–0.45 prompt-only against the trained student's 0.61
(paired tests), collapsing to baseline when the specialized streams are removed.
Concretely: across the tested small-model regimes, explicit prompt clauses were
sufficient to elicit several redirect types, whereas reliable three-strike
persistence and pedagogical withholding required specialized SFT; native reasoning
partially closed the persistence gap but remained less reliable and substantially
more expensive at inference time. Mapping these regimes under a matched-prompt
protocol is the paper's contribution.

Reaching it cleanly required retiring the conventional context-blind
redirect-axis F1 — a *type* classifier that ties a 0.8B student with a 9B
teacher — for a quality-aware pairwise eval that recovers what it hides
(role-swap repair quality, win-rate 0.87, on an axis F1 called saturated
parity). We package these findings with the locale-aware generation pipeline
as reusable apparatus (including the `locale_judge` FP audit, pass rate
70.1% $\to$ 88.4%), and report the trigger-position decorrelation construction
as *partially validated*: it removes the positional recall bias but does not
reduce premature firing, which at 0.8B is threshold-laxity, not a positional
shortcut.

All experiments run on a single RTX 3060 12GB GPU — the regime where the
boundary matters most, telling a deployer of a small model which behaviors a
prompt gives for free and which require the pipeline. The regime-C effects already
replicate in direction in a second trained family (a Llama-3.2-1B student, §6.2);
the open edges, in priority order (§6.3): **budget-matched leave-one-stream-out
ablations** to attribute each capability effect to a single stream; **human
validation** of the judged withholding and pairwise metrics; an **orthogonal
depth$\times$violation-count probe**; an **out-of-distribution robustness set**; a
**higher-precision teacher control**; and broader cross-family/multi-locale
replication — plus transfer of the matched-prompt methodology to other rare,
semantically-triggered markers (refusal-token and tool-call emission), where the
same "describable but not reliably promptable" question applies.

## Limitations

We surface the limitations detailed in §6.1–§6.2 here for visibility.

- **Primarily one model family** (Qwen). A cross-family probe corroborates the
  boundary outside Qwen — prompt-only at 8B and, more decisively, a *trained*
  Llama-3.2-1B student (persistence recall 0.91 vs 0.25 for the same untrained
  base, §6.2). Remaining breadth (a third family, a base-checkpoint Llama
  student, multi-locale) is future work.
- **Single locale** (`china`) — persistence and withholding are *hypothesized*
  to be less locale-dependent (their triggers contain no locale variables), but
  single-locale evaluation still limits external validity and we do not claim
  locale-independence as established (§6.2); the locale-fidelity axis result and
  the `locale_judge` gazetteer are locale-specific.
- **Judged-metric power** — the trained-vs-teacher withholding gap is
  *directional* (edge of per-judge significance); the strong claim is the
  teacher's absolute sub-50% compliance. Context-dependent mechanical scores
  are $n\leq25$ (suggestive); the pairwise eval leans only on the large,
  seed-stable effects (role-swap, language).
- **Seeds and side-results** — the load-bearing A1/A3 conditions use three
  seeds (42/123/7), all others single-seed; the decorrelation construction is
  partially validated (§5.3.1), and the pipeline is apparatus, not a
  contribution.

## Code and Data Availability

All code (data generation, training, evaluation, and scoring), the per-condition
training configurations, random seeds, frozen evaluation sets, and judge prompts
are released at <https://github.com/cch-ai922/tutor-train>, together with the
synthetic training data and the per-condition score outputs underlying every
reported table. A `reproducibility/` guide maps each claim to the script, config,
and expected number that produce it. Trained LoRA adapters are low-rank deltas
over the public base models (Qwen3.5-0.8B-Base; Llama-3.2-1B-Instruct for the
cross-family replication) and are available from the authors on request.

## Ethics and data statement

All training and evaluation data are **model-generated** (distilled from a
locally-served teacher) and contain **no personal data or PII**; no human
subjects were involved and no human-authored text was collected. The released
artifacts (code, scripts, configuration, and the synthetic datasets;
Code and Data Availability) inherit this property. The intended use is
research on data curation and the train-versus-prompt boundary for small
task-specific models; we are not aware of heightened dual-use risk beyond that
of the underlying open base model.



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
consequence is a bimodal per-axis F1 (Table 14)
that floors on axes whose correct repair has no context-free surface form
and ceilings on axes where every model produces a classifiable response.

| **Axis** | **Per-axis F1 range** | **Behaviour** |
| :-------- | :---------------------------------- | :--------------------------------------------------- |
| language | $\approx 0.375$ (7 of 9 conditions) | near-constant; does not move with model quality |
| locale | 0.19--0.39 | compressed near floor |
| generic | 0.087--0.275 | compressed; 9B teacher *tied for lowest* |
| persona | 0.68--0.82 | high, with real spread |
| role_swap | 0.71--0.86 | high, with real spread |
| pedagogy | mid-range | the only axis both judge-scorable and discriminating |

: **Table 14. Per-axis redirect F1 is bimodal.** The context-dependent
axes (language, locale, generic) floor near a structural minimum for
*every* condition because the context-blind judge cannot score them from
the response alone; the self-contained axes (persona, role_swap) ceiling
because every model produces a classifiable repair of the correct type.
Only pedagogy sits in a discriminating mid-range. A macro-average mixes
a floored instrument with a ceilinged one. {#tab:macro-f1-bimodal}

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
(Table 15) in a short-circuit cascade ordered
cheap-and-high-catch first, so most rejections occur before the expensive
LLM-judge filter is consulted.

| **\#** | **Filter** | **Cost** | **Catches** |
| :-- | :--------------- | :------------------- | :-------------------------------------------------------------------- |
| 1 | `speaks_l1_sanity` | mechanical | Degenerate speaks_l1 records lacking the L1 code-switch turn |
| 2 | `non_latin_script` | mechanical | Non-Latin characters in assistant or non-language_redirect user turns |
| 3 | `banned_terms` | mechanical | Politics, religion, self-harm, and locale-sensitive vocabulary |
| 4 | `mode_consistency` | mechanical | `EvaluationExample` records whose JSON body fails schema validation |
| 5 | `naturalness` | mechanical heuristic | Stilted, low-perplexity, repetitive prose |
| 6 | `locale_judge` | LLM call | Out-of-locale entities (NYC, Thanksgiving, Costco, etc.) |

: **Table 15. Six-filter cascade.** Filters 1--4 are deterministic and
cheap; filter 5 is a heuristic with no model call; filter 6 is the only
LLM-judge filter and runs last. {#tab:filter-cascade}

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
(Table 16); every variant holds the trigger at exactly three
strikes and varies only the lead-in scaffolding, shifting the sentinel to a
different absolute turn without changing what the model must detect.

| **Variant** | **Sentinel turn** | **Lead-in** | **Strike turns** | **Probe turns** |
| :------ | :------------- | :-------------- | :------------ | :----------- |
| V1 | 5 | 0 turns | 0, 2, 4 | 1, 3 |
| V2 | 7 | 2 (turns 0--1) | 2, 4, 6 | 3, 5 |
| V3 | 9 | 4 (turns 0--3) | 4, 6, 8 | 5, 7 |
| V4 | 11 | 6 (turns 0--5) | 6, 8, 10 | 7, 9 |

: **Table 16. The four structural variants of the persistent 3-strike
streams.** Sentinel turn and lead-in length vary; variant is
hash-deterministic per record. Strike turns are user turns; probe and
sentinel turns are assistant turns. {#tab:variants}

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

| **Cond**     | **position design**  | **recall** | **premature** |
| :-------------- | :------------------------- | :------- | :---------- |
| A1 (1 epoch) | 4-variant {5,7,9,11} | 0.818      | 0.208         |
| A5           | fixed-turn-7         | 0.560      | 0.119         |

: **Table 17. Trigger-position decorrelation, isolated.** *recall* =
correct firing on true third-strike positives (n=159); *premature* =
firing before the third strike (Persistent-Premature-Probe, n=318).
Decorrelation raises recall (0.82 vs 0.56, by removing the fixed-turn's
positional bias ---
Table 18) but does *not* reduce premature firing
(0.208 vs 0.119). The deployed 2-epoch A1 has premature 0.107.
{#tab:decorrelation}

Stratifying recall by the turn at which the third strike lands shows *why*
recall improves: the fixed-turn design fires reliably only near its trained
position, while the 4-variant fires wherever the third strike lands.

| **Cond**                | **t=5** | **t=7**   | **t=9** | **t=11** | **overall** |
| :--------------------------- | :----- | :----- | :----- | :----- | :------- |
| A1 (1 epoch), 4-variant | 0.836   | 0.818     | 0.826   | 0.778    | 0.818       |
| A5, fixed-turn-7        | 0.478   | **0.727** | 0.565   | 0.556    | 0.560       |

: **Table 18. Recall by the turn at which the third strike lands.** The
fixed-turn design (A5) recalls best at its trained turn 7 (0.727) and
degrades off-position (0.48--0.57); the 4-variant recalls uniformly
across positions (0.78--0.84), so firing is conditioned on the semantic
trigger rather than the turn. {#tab:recall-by-turn}

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
