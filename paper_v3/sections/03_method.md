# 3. Method

<!-- paper_v3 — "Don't Trim the Tail" -->
<!-- All reported numbers are from Phase-0 existing generations. -->

## 3.1 Task: a rare structured control marker

The study task is a four-axis **session-ending sentinel**. A tutor model
converses in character; on four governed abuse axes — off-topic safety probes,
sustained L1/refusal-to-use-English, sustained persona attacks, and sustained
role-swap — the third *same-axis* violation must trigger a literal marker that
an external dispatcher consumes to close the session:

    persistent_off_topic          -> [SESSION_END: persistent_off_topic]
    persistent_language_violation -> [SESSION_END: persistent_language_violation]
    persistent_persona_break      -> [SESSION_END: persistent_persona_break]
    persistent_role_swap          -> [SESSION_END: persistent_role_swap]

The marker is *rare* (one firing per governed dialogue, absent from ~4/5 of the
corpus), has a *count-based trigger* (exactly the third same-axis strike), and
is *machine-consumed* (its axis label routes downstream handling). These three
properties make it a clean testbed for control-token curation.

## 3.2 Factors under study

We manipulate two data-shape factors, fully crossed, at matched training budget
(1 epoch, QLoRA [@dettmers2023qlora] SFT, 0.8B base):

- **Position design** — the trained firing turn is either **fixed** (always
  turn 7) or **4-variant decorrelated** (turns {5,7,9,11}).
- **Marker format** — **typed** (`[SESSION_END: <axis>]`, carries the axis) or
  **generic** (bare `[SESSION_END]`, no axis).

This yields four cells: **A1** (4var+typed), **A5** (fix7+typed),
**A6** (fix7+generic), **A7** (4var+generic).

Crossed orthogonally with a third factor:

- **Post-marker trim** — each persistent training record is either **trimmed**
  to end at the marker, or **untrimmed** (the benign post-marker continuation is
  retained). All four cells exist in both variants, giving a 2×2×2.

Every other factor is held constant across all cells and variants — base model,
tokenizer, optimizer, learning-rate schedule, batch size, epoch budget, seed
policy, and the shared evaluation prompts — so any behavioral difference is
attributable to the three data-shape factors above, and trim in particular is
isolated by construction (only the post-marker continuation differs; §4.4
verifies the trimmed and untrimmed records share an exact prefix).

## 3.3 The trim manipulation (the primary lever)

![Untrimmed vs. trimmed training record. Both share an identical prefix through the marker; trimming removes only the benign post-marker continuation — the turns that show an escalated context *not* followed by a further fire.](paper_v3/figures/fig1_sequences.png){width=85%}

A trimmed record ends at the marker; every escalated conversation in trimmed
training therefore *terminates* at a fire. An untrimmed record keeps the turns
after the marker (the tutor continues benignly), supplying examples in which an
escalated context is present but *not* followed by a fire. Our hypothesis (§3.6)
is that trimming deletes exactly the counterexamples the model needs to learn
that escalation is necessary but not *sufficient* — only the third strike is.

## 3.4 Metrics

- **Recall** — on true third-strike positives, did the marker fire?
- **Premature-firing rate** — on sub-threshold contexts (1–2 strikes), did it
  fire anyway? This is the failure mode trim is hypothesized to induce.
  Stratified by violation count (vc=1 vs vc=2) and by turn.
- **Axis-attribution accuracy** — of typed fires, the fraction naming the
  *correct* axis. Undefined for generic markers by construction; that asymmetry
  is itself the semantic-gate result.

## 3.5 Probes

- **persistent_probe** — true third-strike positives (recall).
- **persistent_premature_probe** — single-axis sub-threshold contexts
  (premature firing), n=318, stratified by (vc, turn).
- **mixed_violation_probe** (Phase 1, this paper) — the decisive attribution
  test: primary axis X escalates to threshold while a distractor axis Y appears
  sub-threshold. `fire_correct` records (X at 3rd strike) test correct
  attribution under distraction; `distractor_sub` records (X at 2nd strike, Y
  present) test that mere escalation of *some* axis does not trigger firing.
  n≈282 (141 fire_correct / 141 distractor_sub), all 12 X×Y axis pairs covered.

## 3.6 Hypotheses

- **H1 (trim → premature).** Trimming raises premature-firing rate in every
  (position×marker) cell, by an amount larger than either design factor.
- **H2 (typed = semantic gate).** Typed models attribute the correct axis at
  high accuracy and never emit a contentless marker; generic models cannot
  attribute at all. Attribution is robust to trim even where timing is not.
- **H3 (count remedy).** Annotating the marker with the strike count
  (`[SESSION_END: <axis>, strike=3]`) supervises the latent counter and
  neutralizes the trim-induced premature firing. *(Phase 2.)*
