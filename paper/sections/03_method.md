# 3. Method

The data-generation pipeline takes a small seed pool of CEFR-stratified
tutor scenarios and produces three downstream artefacts: an SFT corpus,
a DPO preference-pair corpus, and an evaluation corpus carrying
`<think>` reasoning annotations. The same teacher model generates all
three, ensuring that the prompt shape the student model sees at
training and the prompt shape it sees at deployment are identical.
This section presents the pipeline component by component.

## 3.1 Pipeline overview

Figure&nbsp;1 (TODO) shows the eight pipeline stages. A small pool of
CEFR-stratified scenario *seeds* drives twelve parallel SFT generation
streams. Each stream produces a different kind of multi-turn dialogue;
together they cover normal tutor behavior, seven distinct redirect
axes, and four persistent three-strike abuse axes. The output of all
twelve streams flows through a six-filter cascade. A yield-aware
*top-up loop* iterates per (stream, level) cell until each cell
reaches a target floor specified either as an absolute count or as a
share of the final mix. Filtered dialogues then feed two further
generators: a register-DPO pair generator and a `<think>`-mode
evaluator-example generator. Finally, after SFT, the trained student
itself is used to generate on-policy DPO pairs.

Three properties of the pipeline are worth highlighting at the outset.
First, every stage is **resumable**: each generator skips ids already
present in its output JSONL, so a pipeline killed mid-run loses no
work. Second, the train/eval split is **hash-deterministic** on seed
id, so re-running data generation after a top-up never reshuffles the
held-out set. Third, the teacher model receives a prompt of exactly
the shape the student sees at deployment time (same system prompt,
same user-turn rendering, same locale block), so there is no
distribution shift between teacher demonstration and student
inference.

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

We organise tutor-side behavior around a single principle. A tutor is
the keeper of a set of **interaction invariants** — properties of the
session that the tutor is configured to hold throughout. A learner
*violation* is a move that breaks exactly one invariant, and the
correct redirect is the **minimal repair** that restores it. This
gives us a generative criterion for the taxonomy rather than an
intuited list:

> Two learner violations occupy **distinct redirect axes iff their
> minimal repairs differ in shape.** Merge violations whose repair is
> identical; split violations whose repair differs.

The invariants are not arbitrary: they are the commitments the tutor
system prompt itself makes (the language of instruction, the lesson
topic, the role structure, the persona, the pedagogical contract, the
locale frame) plus a general-appropriateness invariant inherited from
the underlying assistant. We therefore claim completeness only
*relative to the enumerated invariant set*, not over all conceivable
tutor violations — a bounded claim we can actually defend. Table&nbsp;1
states the invariant, the characteristic violation, and the minimal
repair for each single-shot axis.

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
\caption{\textbf{The seven single-shot redirect axes as invariant / violation / minimal-repair triples.} The third column is the response \emph{shape} a tutor must produce; that the shapes differ is exactly the claim the \S5.4 ablation tests.}
\label{tab:invariant-triples}
\end{table*}
```

Two honest caveats. The final row (general appropriateness) is
different in kind from the other six — it is a general-assistant
safety behavior rather than a tutoring-specific invariant — so we
treat it as the catch-all rather than pretending it is parallel. And
we arrived at this list partly by enumeration before recognising the
organising principle; we present the principle as the structure these
axes instantiate, not as a historical account of their discovery. The
principle earns its place by doing two pieces of work the bare list
cannot: it supplies the granularity criterion above, and it converts
"a generic redirect stream is insufficient" from an assertion into a
falsifiable prediction (distinct invariants require distinct repairs),
which §5.4 tests per axis.

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
invariant (Table&nbsp;1, third column). Training on a single generic
"redirect" stream, as most prior tutor datasets do, collapses these
axis-specific repair shapes into one averaged behavior.

**Persistent 3-strike streams (4 streams).** Persistence is an
*orthogonal dimension* to the invariant axis: in principle any
invariant violation can be one-off or repeated. A persistent stream
produces dialogues where the learner *persists* in the same violation
across three probe turns; the tutor probes the abuse twice, then on
the third strike ends the session with a sentinel marker. We give
persistent variants to four axes — `persistent_off_topic`,
`persistent_language_violation`, `persistent_persona_break`, and
`persistent_role_swap` — and not to all seven, on a principled
ground: a persistent axis earns its own stream only where *repeated*
violation changes the correct response (escalation to a hard
session-end). For these four, repetition plausibly warrants
escalation. A repeated locale slip, by contrast, is most naturally
just corrected again in-locale, so no escalation behavior is
distinct enough to train. We flag one honest borderline case:
*pedagogy* persistence (a learner who repeatedly demands "just give
me the answer") is a plausible escalation candidate we do not
currently include; adding a `persistent_pedagogy` stream is a clean
extension and we note it as such rather than claim the four-axis set
is forced.

## 3.4 Trigger-position decorrelation: the 4-variant persistent design

The persistent streams carry the paper's strongest methodological
contribution, and it is best stated as a general principle before its
tutor-specific instantiation.

> **Principle (trigger-position decorrelation).** When a model must
> emit a rare structured marker conditional on a *semantic* trigger,
> but the marker is *positionally regular* in the training data, the
> model will learn the position as a proxy for the trigger. To force
> the model to learn the trigger, decorrelate marker position from
> trigger by resampling the position — subject to whatever
> determinism and codomain constraints the data pipeline imposes.

This is a multi-turn, structured-output instance of shortcut learning
(§2.4): the model takes the cheapest predictor of the label that the
data exposes. In our persistent streams the rare marker is the
sentinel, the semantic trigger is "third strike on the same axis,"
and the naive design — sentinel always at turn 7 — makes turn
position a perfect proxy. A student trained on it learns to fire on
*turn 7* rather than on *the third strike*, producing both false
positives (firing on benign turn-7 utterances) and false negatives
(failing to fire when persistence ends earlier or later).

We instantiate the principle with a **4-variant structural design**.
Each persistent dialogue is generated with one of four variants chosen
hash-deterministically from the seed id. Crucially, **every variant
holds the trigger constant at exactly three strikes** and varies only
the amount of normal scaffolding that precedes the persistence block;
this is what shifts the sentinel to a different absolute turn without
changing what the model must detect. The dialogue opens with a tutor
turn (assistant turns are odd, user turns even), the persistence block
is the fixed six-turn sequence strike–probe–strike–probe–strike–
sentinel, and lead-in scaffolding fills the turns before it:

Turn indices below are 0-based with the dialogue opening on a learner
turn (so user turns are even, tutor turns are odd). Every variant
contains exactly three strikes; the variants differ only in the
amount of normal scaffolding that precedes the persistence block.

**Table 2. The four structural variants of the persistent 3-strike
streams.** Sentinel turn and lead-in length vary; variant is chosen
hash-deterministically per record from the seed id.

| Variant | Sentinel turn | Lead-in scaffolding turns | Strike (user) turns | Probe (assistant) turns |
| --- | --- | --- | --- | --- |
| V1 | 5 | 0 | 0, 2, 4 | 1, 3 |
| V2 | 7 | 2 (turns 0–1) | 2, 4, 6 | 3, 5 |
| V3 | 9 | 4 (turns 0–3) | 4, 6, 8 | 5, 7 |
| V4 | 11 | 6 (turns 0–5) | 6, 8, 10 | 7, 9 |

The variant index is computed as `int(sha256(seed_id)[:8]) % 4` and
maps to one of the four sentinel positions {5, 7, 9, 11}. This gives
a uniform distribution of variants across the dataset while keeping
the choice reproducible across runs and across the train/eval split.

A subtle design choice is doing real work here: every variant **holds
the trigger constant at exactly three strikes** and adjusts only how
much normal scaffolding precedes them. A naive alternative would
reach different sentinel positions by varying the number of strikes
(e.g. 1, 2, 3, 4 strikes for positions 5, 7, 9, 11). Under that
scheme, sentinel position co-varies with strike count, so a
"fire-rate by position" metric silently becomes "fire-rate by
strike-count," and only the 3-strike variant fires on the literal
*third* strike — contradicting the "third strike on the same axis"
framing the whole section rests on. The chosen design dissociates
position from trigger cleanly: "third strike" is literally true for
every variant, and the §4.8 position-stratified metric measures
position and nothing else.

**Why hash-deterministic rather than `random.choice`?** The
"subject to constraints" clause of the principle is doing real work
here, and it is what distinguishes our construction from a plain
shuffle. Three pipeline invariants force the seeded form. (i)
*Resumability*: every generator skips ids already present in its
output JSONL, so a killed-and-resumed run that re-generates a record
must assign it the *same* variant or the cumulative distribution
drifts across restarts. (ii) *Train/eval coherence*: the held-out
split is itself hash-deterministic on `seed_id`
(`sha256(seed_id)[:8] % 100`), so a per-run random variant assignment
would reshuffle the eval set's variant distribution across runs and
across ablations, making the A1-vs-A5 decorrelation comparison
(§4.6) ill-defined. (iii)
*Turn-parity*: dialogues open on a learner turn, so user turns are
even and assistant (sentinel) turns are odd, leaving only
parity-valid odd values in the practical range {5, 7, 9, 11} ($\geq$5 to
fit three strikes and the two intervening probes; ≤11 to stay inside
the SFT max-length budget — at the 1792-token cap, the longest
trained variant (V4 at C2) is at median 1685 tokens and overflows
the cap in only ~1% of records, so the cap is not a load-bearing
constraint on positions {5, 7, 9, 11} but would tighten for any
V5+ extension beyond turn 11). Hash-mod-4 is therefore
the principled form of "random" here: it is a seeded pseudo-random
function over the only feasible discrete codomain, satisfying all
three constraints simultaneously and producing the desired
25/25/25/25 split by construction without post-hoc rebalancing.

The intuition is that a student trained on this mix cannot learn a
*single-position* rule because the sentinel turn varies; instead it
must learn to detect *the third instance of the same axis*. We caution
that uniform firing across {5, 7, 9, 11} is *necessary but not
sufficient* evidence: a model that memorised all four trained
positions would also fire uniformly on in-distribution records. The
hypotheses are separated only where they disagree — on benign
prefixes that reach a trained position without three strikes, and on
records whose third strike lands at an untrained position — so §5
reports recall on the held-out Persistent-Probe set together with the
false-positive rate on Persistent-FP-Probe and firing on
Persistent-OffPosition-Probe (§4.5, §4.8). Because the same
construction applies to any rare, semantically-triggered,
positionally-regular marker, we expect it to transfer to refusal-token
emission, tool-call emission, and agentic stop conditions; we
demonstrate only the sentinel case here.

## 3.5 Locale-aware prompt engineering

The teacher is a general-purpose multilingual model with strong
Western cultural defaults. A naive "tutor for English learners in
China" system prompt produces dialogues with NYC, Thanksgiving, and
Costco references at non-trivial rates. We address this with a
**locale instruction block** that is prepended to every per-scenario
system prompt and parameterized from `config/locale.yaml`.

The block enumerates: (i) the country and a country adjective for
in-locale reference; (ii) per-locale learner-description language so
the learner persona is plausible; (iii) a per-locale list of
`avoid_default_cultures` so the teacher knows to suppress Western
default settings; and (iv) a per-locale list of `avoided_topics` that
combines safety and cultural sensitivity.

Most generation streams require **strict-Latin output** in user turns,
since the learner is a non-native English learner and their turns should
be in English, possibly with errors. The `language_redirect` stream
is the exception: by design, the learner code-switches into L1 mid-
dialogue. We split the locale block into two variants accordingly:

- **`locale_instruction_block`** (default): strict-Latin output rule;
  L1 characters are forbidden in user turns and trigger the
  `non_latin_script` filter.
- **`locale_instruction_block_allow_l1`** (allow-L1 variant): the
  strict-Latin rule is dropped; L1 characters are permitted in the
  code-switch turn. The `non_latin_script` filter respects the
  `scenario_type` field and skips user turns for `language_redirect`
  records.

This split is necessary because the strict-Latin rule and the
language_redirect intent contradict each other. Before we introduced
the allow-L1 variant, every language_redirect example failed the
non_latin_script filter and the stream had ~0% pass rate. After the
split, the stream reaches the global ~85-90% post-filter pass rate.

A related but distinct decision is the choice to model only the
*tutor* side of redirect behavior in our SFT. We do not optimize the
user-side persona, only the tutor's response to user behavior. This
keeps the contributed-behavior axis sharply defined.

## 3.6 Yield-aware top-up with declarative ratio targets

Generation yield is below 100% on every stream because (i) the
teacher sometimes refuses, returns malformed output, or violates the
schema; and (ii) the filter cascade rejects records that fail any of
the six filters. The net pass-rate observed in our experiments is
70-90% depending on stream and locale_judge setting (§5).

A naive solution is to over-generate to a fixed multiple of the
desired count. This wastes teacher cycles when the yield is high and
underfills when the yield is low. We use an **iterative top-up loop**
that keeps generating until each (stream, level) cell reaches a
target floor:

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

The top-up loop is per-cell (per stream, per CEFR level) and is
bounded by `MAX_ROUNDS=5` so it cannot loop forever. After
`MAX_ROUNDS`, cells still below target are reported in the log as
shortfalls; the operator decides whether to raise `MAX_ROUNDS`,
raise the seed pool, or accept the shortfall.

### Two knob types

Two kinds of generation can be bumped per round depending on stream:

- **Fraction-gated streams** (the 7 single-shot redirects + the 4
  persistent streams) pick a hash-deterministic fraction of the
  existing seed pool. Bumping the fraction in the top-up round
  attempts more previously-skipped seeds. Because the picker is
  hash-deterministic, no seed is re-attempted in a later round.
  Resume is automatic.

- **Seed-count-gated streams** (the `normal` stream) produce one (or
  `dialogues_per_seed`) dialogue per seed. To get more passing
  normal examples at a given level, more seeds must exist at that
  level. The top-up loop bumps `n_per_level` for the short level
  only, runs the seed generator (resumable, only new seeds are
  produced), then runs the SFT generator.

### Declarative ratio targets

Two YAML knob shapes specify per-stream targets:

- `<stream>_target_per_level` (int): the absolute floor, namely the
  per-level count the stream must reach.
- `<stream>_target_ratio` (float in (0, 1)): the share of the final
  post-filter mix this stream should occupy.

Ratio takes precedence over absolute when both are set. The target
count for a ratio-targeted stream is computed from the *other*
streams' absolute floors:

$$T_{\text{per\_level}} = \frac{\sum_{s \in \text{absolute}} t_s}{1 - \sum_{r \in \text{ratio}} r_r}$$

$$t^*_r = r_r \cdot T_{\text{per\_level}} \quad \forall r \in \text{ratio}$$

If `normal_target_ratio=0.5` and all other streams use absolute
targets that sum to (say) 270 records per level, then
$T_{\text{per\_level}} = 270 / (1 - 0.5) = 540$ and the normal stream
floors to 270 per level. This lets the operator declare "I want
normal to be exactly half of my training mix" without doing the
arithmetic.

Validation: the sum of ratio targets must be strictly less than 1,
and at least one stream must use an absolute target — otherwise the
linear equation has no solution.

## 3.7 Filter pipeline

After every generation pass, dialogues flow through six filters in a
short-circuit cascade. The order is chosen to put cheap and high-
catch filters first so most rejections occur before the expensive
LLM-judge filter is consulted.

**Table 3. Six-filter cascade.** Filters 1–4 are deterministic and
cheap; filter 5 is a heuristic with no model call; filter 6 is the
only LLM-judge filter and runs last.

| # | Filter | Cost | Catches |
| --- | --- | --- | --- |
| 1 | `speaks_l1_sanity` | mechanical | Degenerate speaks_l1 records lacking the L1 code-switch turn |
| 2 | `non_latin_script` | mechanical | Non-Latin characters in assistant or non-language_redirect user turns |
| 3 | `banned_terms` | mechanical | Politics, religion, self-harm, and locale-sensitive vocabulary |
| 4 | `mode_consistency` | mechanical | `EvaluationExample` records whose JSON body fails schema validation |
| 5 | `naturalness` | mechanical heuristic | Stilted, low-perplexity, repetitive prose |
| 6 | `locale_judge` | LLM call | Out-of-locale entities (NYC, Thanksgiving, Costco, etc.) |

Filters 1–4 are deterministic and cheap. Filter 5 uses a
heuristic-only scoring (no model call). Filter 6, the
`locale_judge`, extracts proper-noun entities from each record and
asks the teacher to classify each entity as `in_locale` or
`out_of_locale` for the configured locale. Verdicts are cached in
SQLite keyed by `(entity, locale)` because the same entity recurs
across records and we want at most one judge call per
`(entity, locale)` pair.

Two design choices in the `locale_judge` deserve note:

- **Per-locale allowlist of known-in-locale entities** (e.g.
  `WeChat`, `Alipay`, `Yunnan`, `Mid-Autumn Festival`) bypasses the
  judge call. This both saves LLM cycles and prevents the small
  teacher from poisoning the cache with false positives on entities
  that are genuinely in-locale.
- **Sentence-initial common-word allowlist** (e.g. `Absolutely`,
  `Precision`, `Wi-Fi`, `Will`, `Plus`, `Line`, `Coffee`) prevents
  the entity extractor from interpreting sentence-initial
  capitalization as proper-noun status.

In §6.5 we report a methodological finding: the `locale_judge` in its
default configuration was responsible for **57.5% of all filter
rejections, of which $\geq$85% were false positives** on entities that
were either genuinely in-locale (e.g. `West Lake`, `Drum Tower`,
`Muslim Quarter`) or were English modal verbs / connectors (e.g.
`Will`, `Plus`, `Line`). After auditing the rejections and extending
both allowlists, the global filter pass rate rose from 70.1% to
88.4%. We treat this as a cautionary lesson about LLM-judge filters
and recommend that any pipeline using one (i) pre-populate aggressive
in-locale allowlists, and (ii) audit the rejection log periodically.

## 3.8 DPO pair construction

We construct three kinds of DPO preference pairs.

**Register DPO pairs.** For each filtered SFT dialogue, we ask the
teacher to rewrite a tutor turn in a register-inappropriate way
(e.g. too formal at A1; too casual at C2; or with a register
mismatch to the tutor persona). The original tutor turn becomes the
`chosen` response and the register-degraded rewrite becomes the
`rejected` response. Register pairs are cheap because they can be
generated immediately after SFT data without the trained student.

**On-policy DPO pairs.** After SFT training of the student is
complete, we draw the student's own response to a turn in a held-out
SFT dialogue and compare it to the teacher's response. A judge
ensemble scores the two responses; pairs with a *clear* margin (a
judge-set threshold of $\geq$2 on a 1–5 scale) are kept, with the teacher
response as `chosen` and the student response as `rejected`. The
margin threshold matters: with no margin the dataset becomes noisy;
with too aggressive a margin the dataset becomes too small.

**Sentinel DPO pairs.** A third pool, drawn only from the four
persistent_* streams, targets the rare-token problem that SFT alone
cannot solve. The `[SESSION_END: <axis>]` marker appears in roughly
3–5% of assistant turns in the SFT mix (only the third-strike turn
of the four persistent streams), so its per-token cross-entropy
gradient is weak — a student trained on SFT only reliably learns to
*end* with a polite refusal but emits the literal bracketed marker
only some of the time. We construct sentinel pairs in two
complementary modes:

- **Offline mode.** The `chosen` is the teacher's full sentinel-firing
  response (carrying the marker); the `rejected` is the same response
  with the marker substring stripped. Both sides are reachable from
  the SFT student's distribution (the rejected is normal refusal
  text), so DPO loss is stable. No teacher or policy calls are
  required — pairs are constructed deterministically from the
  filtered persistent_* SFT records.
- **On-policy mode.** After SFT training, the student regenerates the
  sentinel-firing turn given the same dialogue prefix; the teacher's
  marker-bearing turn is the `chosen` and the student's regeneration
  is the `rejected`. The judge step from the on-policy register pool
  is skipped here because the marker's presence is mechanical ground
  truth: whenever the student fails to emit it, the teacher wins by
  construction. Pairs where the student already emits the marker are
  dropped (zero signal).

The contrast in both sentinel modes is sharp on a single bracketed
substring and identical-modulo-the-marker (offline) or
realistic-SFT-student (on-policy), so the gradient pushes the
marker's conditional probability up without disrupting other
behaviors. We assign the new `sentinel_missing` rejection axis to
all sentinel pairs so downstream auditing can separate them from the
nine general-purpose register / on-policy axes.

The three pair pools are mixed for DPO training. We default to a
65/25/10 register/on-policy/sentinel mix; the
`mix_ratio_register`, `mix_ratio_on_policy`, and
`mix_ratio_sentinel` knobs in the training config control this.
The sentinel pool is naturally the smallest (at most one offline
pair plus one on-policy pair per filtered persistent_* record), so
its share is set low and `use_all_data=true` is the recommended
default — the ratio-trimmer is reserved for the rare case where
the operator wants to enforce the exact three-way ratio.

### Sentinel-turn exclusion in the *on-policy register* pool

Although sentinel turns now have their own pool, they remain
**excluded from the judge-mediated on-policy register pool**. Two
reasons. First, the judge does not score sentinel emission — it
weighs register quality, CEFR fit, locale fidelity, and
naturalness — so including sentinel turns there would muddy the
judge's verdict on response *quality*. Second, the sentinel pool
already provides the marker-presence contrast through a more
targeted construction. The on-policy register pool therefore
continues to operate on the 3-4 *earlier* refusal turns of each
persistent dialogue, where the register signal lives.

## 3.9 `<think>`-mode evaluation examples

In addition to SFT and DPO data, the pipeline produces evaluation
examples for training a self-judge mode on the student. Each
evaluation record carries a transcript (a held-out SFT dialogue, with
its CEFR level, roles, topic, and subtopics prepended to the user
turn) and a teacher-produced response in the format
`<think>...</think>{json}`. The `<think>` block is the teacher's
reasoning trace and the `{json}` is an `EvaluationOutput` schema
containing per-criterion scores and a brief verdict.

Three design decisions are worth noting:

- **Three-way prompt alignment.** The teacher sees the same prompt
  shape (system prompt, user turn carrying the transcript,
  expected `<think>...</think>{json}` response) that the trained
  student will see at deployment. Previously the transcript was
  embedded inside the system prompt and the user turn was empty;
  small teachers interpreted the empty user turn as "Begin." and
  wasted their `<think>` budget searching for the transcript.
- **Mandatory non-empty `<think>` block.** Evaluation responses with
  `<think>\n</think>` or with a `<think>` body of fewer than 30
  characters are rejected as training data. Without this rule, the
  Qwen-family teacher would sometimes respond in `/no_think` mode
  and produce a syntactically valid but pedagogically useless
  evaluation record.
- **`max_tokens=3072`.** The default 2048-token generation budget
  truncates the trailing JSON tail of ~9B teacher `<think>`
  responses, producing JSON parse errors at the `mode_consistency`
  filter. A 3072 budget fits both the `<think>` block and the JSON
  with margin.

## 3.10 Auto-detected served model

For reproducibility, every generated record carries a
`metadata.generation.model` field. This field is populated by
querying the served-model name from the teacher endpoint's
`/v1/models` route at pipeline startup, overriding the static value
in `config/generation.yaml`. This avoids a class of provenance bug
where the YAML names one model but the endpoint actually serves
another, producing records mis-tagged with the wrong teacher.
