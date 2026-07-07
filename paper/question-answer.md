# Reviewer Question–Answer Bank

Answers grounded in the current manuscript (`sections/01`–`08`). Section
references (§) point to the paper. This is a defense/rebuttal preparation
document, not part of the paper itself.

---

## 1. Novelty Questions (Highest Priority)

### Q1. What is the actual scientific contribution of this paper? (pipeline / dataset / benchmark / evaluation / prompt engineering / train-vs-prompt boundary — which is the real one?)

The single contribution is the **train-versus-prompt boundary**: a
per-capability map of which deployed-tutor behaviors a fully-specified system
prompt can *elicit* versus which must be *demonstrated* in fine-tuning (§1
"Contribution"; §2.5). Everything else is explicitly demoted to secondary
status in the paper itself: the pipeline, dataset, and locale-judge audit are
labeled "reusable apparatus," and the retired-F1 result plus the decorrelation
construction are labeled "bounded side-results" (§1, §2.5 Table on
positioning). We do not advance the pipeline or dataset as contributions; they
exist to make the boundary question answerable. The one-line answer to a
reviewer: *the boundary is the contribution; the rest is the apparatus that
produced it.*

### Q2. How is this different from Self-Instruct / Evol-Instruct / LIMA / InstructGPT? What has never been done before?

Those papers answer the prompting-vs-tuning question at the level of *general
alignment*: LIMA and InstructGPT show a small demonstration set installs broad
instruction-following, and Self-Instruct/Evol-Instruct/WizardLM bootstrap
general instruction data with a uniform recipe (§2.1, §2.5). None of them asks,
**per behavior**, whether the deployment prompt clause that *names* a behavior
already *elicits* it. Our novelty is the localization: we decompose a deployed
task (tutoring) into per-clause behaviors and test each one under a
matched-prompt protocol, finding the behaviors split cleanly into promptable
(locale, role-swap, topic) and not-promptable (multi-turn persistence,
pedagogical withholding). That per-behavior resolution — "which specific
commitments does a fully-specified prompt already buy?" — is what is new (§1
"Gap in prior work"; §2.5).

### Q3. Why is the "train-versus-prompt boundary" scientifically important rather than just an engineering observation?

Because it is *predictive and mechanistic*, not a one-off measurement. §6.5
gives the interpretation the data support: a behavior is promptable exactly
when a clause can *select* a behavior already in the model's prior, and
not-promptable when the clause names something the prior cannot supply — either
missing cross-turn state (persistence) or a competing prior that must be
re-weighted (withholding). This turns "generic redirect data is insufficient"
into a *falsifiable per-axis prediction* (§3.3) — specialized data should help
exactly where the prior does not already afford the behavior — which §5.4
confirms. A finding that (a) generalizes across a second model family (§6.2)
and (b) comes with a mechanism that predicts *which side* a new behavior lands
on is a scientific claim, not an engineering anecdote.

### Q4. Could your conclusions have been predicted from previous scaling-law papers? If not, why?

No. Scaling laws predict aggregate loss/accuracy improvements with size; they
do not predict that a *specific* behavior is categorically unreachable by
prompting at any tested scale while a tiny trained model installs it. Our
strongest evidence against a pure-scale account is direct: the 9B teacher —
more than 10× the student — stays at ≤0.06 persistence recall and 0.45
withholding under the identical prompt, and a Llama-3.1-**8B** prompt-only
probe stays at 0.27→0.55 even with CoT (§5.3, §6.2 Table crossfamily). Meanwhile
the *0.8B* trained student reaches 0.83/0.61. The boundary is about
*training-vs-prompting*, and the size comparison (8B-vs-0.8B) is deliberately
isolated from the training comparison by the same-base Llama-1B student-vs-control
contrast (§6.2). Scale does not close the gap; demonstration does.

### Q5. Why should readers outside educational dialogue care?

Because the mechanism is task-agnostic (§6.5): "missing cross-turn state" and
"overriding a competing prior" are general failure modes of prompting, not
tutor-specific ones. The conclusion tells any deployer of a small,
prompt-governed model *where to spend a data-collection budget* — a decision
faced whenever you build a task model on consumer hardware (§1 "Why it
matters"). §7 names the direct transfer targets: other rare,
semantically-triggered markers such as refusal-token and tool-call emission,
where the same "describable-but-not-promptable" question applies.

---

## 2. Methodology Questions

### Q6. Why tutor dialogue? Would another task give the same conclusion?

Tutoring was chosen because it makes the question *both unavoidable and
answerable*: a deployed tutor is already governed by a long system prompt that
literally states each target behavior, and each behavior maps to one auditable
clause (§1, §3.3). That lets us test, clause by clause, whether stating it
suffices — a property most tasks lack. Whether the same *conclusion* holds
elsewhere is an empirical generalization question we address by mechanism
(§6.5, task-agnostic) and by naming concrete transfer targets (refusal-token,
tool-call emission; §7). We do not claim the specific *magnitudes* transfer;
we claim the *mechanism* predicts the side of the boundary.

### Q7. Why only English tutoring?

English-learner tutoring gives a clean, widely-understood CEFR proficiency
axis and a rich, realistic set of interaction invariants (language, locale,
persona, pedagogy, role, topic) to decompose (§3.3). The load-bearing
behaviors — persistence and withholding — are *structural*, not
English-specific: cross-turn counting and answer-prior suppression are
language-independent mechanisms (§6.2 "Locale... does not threaten the central
boundary"). English is the substrate, not the claim.

### Q8. Why only CEFR? Would the boundary change without CEFR?

CEFR is used only as a *stratification* variable so tutor difficulty (A1 vs C2)
is not collapsed (§3.2). We explicitly state we do *not* contribute to
CEFR-leveling (§1 "Scope and non-goals"). The boundary claim is about
persistence/withholding, which are orthogonal to proficiency level — CEFR is a
covariate carried through every table, not a load-bearing factor. Removing CEFR
would remove a reporting dimension, not change which behaviors are promptable.

### Q9. Why only China locale? Could another locale change the conclusions?

The locale (`china`) is a single scope choice, argued in §6.2 to *not* threaten
the central boundary: persistence and withholding are structural behaviors
whose mechanisms (violation-counting, prior-suppression) do not depend on the
cultural backdrop, so "which behaviors are promptable" is locale-independent by
construction. Single-locale bounds only two *secondary* things: the generality
of the locale-fidelity axis (itself an already-promptable axis) and the
locale-specific gazetteer/`locale_judge` (§6.2, §6.4). Multi-locale repeats
would broaden the promptable-axis surface, not overturn persistence/withholding.

### Q10. Why only one deployment prompt? Would another prompt produce another boundary?

One prompt is essential to the *matched-prompt* design: the same
fully-specified deployment prompt is given to every condition, so a prompt-only
failure isolates promptability rather than under-specification (§5.1). The
prompt is not weak — it contains an explicit clause for every redirect axis and
the *complete* three-strike persistence protocol (§3.5). We further steelman
against "your prompt was too weak" with a prompting ladder (zero-shot →
few-shot → output-CoT → native CoT) on the 9B teacher (§4.6.1, §5.3). A
*different* prompt could shift promptable-axis *quality*, but the not-promptable
behaviors fail because of structural limits (§6.5), not wording — see Q73–Q75.

---

## 3. Dataset Questions

### Q11. How much human-written data exists?

None. All training and evaluation data are model-generated (distilled from a
locally-served teacher); no human-authored text was collected (§7 "Ethics and
data statement"). This is stated as a property, not a gap — it also means zero
PII and full reproducibility of the generation process.

### Q12. How much synthetic data exists?

The entire corpus: **3,374 filtered SFT dialogues** across 12 streams and six
CEFR levels (§4.2, Table sft-composition), plus the six held-out evaluation
sets totaling 1,144 probes (§4.3, Table eval-sets). All synthetic.

### Q13. Who verified the generated data?

A **six-filter cascade**: five mechanical filters (script, banned terms,
schema, naturalness) plus one LLM-judge filter (the `locale_judge`), followed
by a yield-aware top-up loop (§3.6). The `locale_judge` itself was audited —
its false-positive behavior on capitalized common words and local landmarks
drove ~57% of rejections and was corrected with two allowlists, raising the
pass rate 70.1%→88.4% (§6.4). Evaluation is by a cross-family three-judge
ensemble plus judge-free mechanical metrics (§4.5–§4.7). Honest limitation:
there is no independent human annotation pass — see Q14–Q15.

### Q14. What is the inter-annotator agreement?

We do not report classical human IAA because there are no human annotators;
verification is automated (Q13). The evaluation-side analogue is *inter-judge*
consistency: we use a three-judge cross-family ensemble and report the median,
and for the withholding metric report each judge separately (Llama 0.571 /
Gemma 0.651 for A1) so the reader sees cross-judge spread directly (§4.5,
§5.4). A genuine limitation to state plainly: a human IAA study is not part of
this work.

### Q15. How many examples were manually inspected?

The paper does not claim a specific manual-inspection count as evidence — this
is an honest gap. What we *can* point to is that the pipeline's failure modes
were found by manual audit of rejection logs (the `locale_judge` FP audit,
§6.4) and that the mechanical metrics (sentinel firing, locale gazetteer) are
deterministic and re-runnable by any reader against the released data (§7 Code
and Data). If pressed, commit to adding a stratified manual-inspection sample
with reported agreement in revision.

### Q16. Could teacher hallucinations be learned by the student?

Possible in principle, mitigated in three ways. (1) The evaluated behaviors are
*structural* (fire a literal sentinel on the 3rd strike; withhold vs answer),
not fact-heavy, so factual hallucination is a smaller risk surface. (2) Locale
leakage — the most hallucination-like failure — is measured mechanically and is
low (A1 1.34%, §5.7). (3) Judges are *cross-family* (Mistral/Meta/Google),
sharing no lineage with the Qwen teacher, so a teacher-inherited artifact is
not rubber-stamped by a same-family judge (§4.5). The deeper "student just
copies the teacher" objection is answered in Q70–Q71.

### Q17. How much diversity exists? Are conversations just template variations?

Diversity is engineered against template collapse: an *angle-shift* mechanism
makes a configurable fraction of normal dialogues approach the topic from a
viewpoint differing from the learner persona, explicitly "to combat teacher
mode collapse onto a single learner-stance archetype" (§3.3 Normal stream). The
corpus spans 12 behaviorally distinct streams × 6 CEFR levels, and the
persistence streams use 4 positional variants ({5,7,9,11}) so lead-in length
varies (§3.4). Honest caveat: the six specialized redirect streams are thin at
C1/C2 (~7–10 records/axis/level, §4.2), which we account for by leaning only on
the large, seed-stable effects (§5.6).

### Q18. How much duplicate data exists?

Every generation stage is resumable by skipping ids already present in output
JSONL, and generation is per-(stream, level) cell to a declarative floor, so
exact-id duplication is prevented by construction (§3.1, Appendix B). The
schema/naturalness filters remove degenerate outputs. We do not report an
explicit near-duplicate embedding-dedup number — a fair item to add in
revision; the released dataset (§7) lets any reviewer compute it.

### Q19. How do you guarantee no train–test leakage?

The split is **hash-deterministic on seed id**: `sha256(seed_id)[:8] mod 100`,
bottom 20% to eval, immutable across runs, so growing the training corpus never
reshuffles or contaminates the held-out set (§3.1, §4.3). The top-up loop
cannot pull an eval seed into training. All eval sets are frozen and released
as a manifest (§4.3, §7).

### Q20. Can another researcher reproduce exactly the same dataset?

The code, configs, seeds, and the *generated* dataset itself are all released
(§7 Code and Data), with a `reproducibility/` guide mapping each claim to its
script/config/expected number. Exact byte-identical *regeneration* depends on
teacher sampling determinism (a local llama.cpp server), but the released
artifacts make the study reproducible without regeneration, and the split logic
is deterministic (Q19). We release the data so reproduction does not hinge on
re-running the teacher.

---

## 4. Experimental Design

### Q21. Why QLoRA rather than full fine-tuning?

Hardware: the entire study runs on a single RTX 3060 12GB, the deliberately
chosen regime where the boundary is most consequential (a large general model
is not deployable there; §1 Scope, §4.1). QLoRA (NF4 4-bit, LoRA r16/α32) is
what fits, and holding it *fixed* across all conditions makes every ablation
single-variable (§4.1). The recipe is treated as fixed apparatus, not a claim.

### Q22. Would full fine-tuning change the boundary?

The boundary is a claim about *prompting-vs-any-training*, and the mechanism
(§6.5) predicts full FT would only *strengthen* the training side (more
capacity to install cross-turn state / re-weight the prior), not create
promptability where the prompt already fails. So the direction is safe; the
*magnitude* of the trained-side numbers could rise. We flag training-recipe
variation as fixed scope (§3, §4.1) and note full-FT as reasonable future work.

### Q23. Would DPO change the conclusions?

All experiments are deliberately **SFT-only** to keep each ablation
single-variable and avoid a preference-pool contamination confound (§3 Scope
note, §4.1). The pipeline *can* emit DPO pairs, documented as unused apparatus
(§3.6, Appendix). DPO on top of the SFT student is named as future work. It
would most plausibly refine *quality* on promptable axes (the §5.6 layer
"beneath" the boundary), not move a not-promptable behavior across the line,
since the SFT student already installs those behaviors.

### Q24. Why only two epochs?

Two epochs is the fixed recipe for all trained conditions (§4.1). Crucially,
the headline persistence result is stated to be **epoch-independent**: A3 fires
0.000 and the trained conditions ≥0.83 regardless of epoch, a gap no epoch
count closes (§5.3). Where epoch budget *did* matter — the A1-vs-A5
decorrelation contrast — we matched both to a **1-epoch** budget to keep it
single-variable (§4.4, §5.3.1). So epoch count is controlled where it could
confound and irrelevant where it cannot.

### Q25. Did you tune hyperparameters?

Hyperparameters are held **fixed and identical** across every trained
condition (same LoRA rank/alpha, epochs, max-seq-len; §4.1, full set in
Appendix C). This is intentional: the design isolates the *data* variable, so
tuning per-condition would reintroduce a confound. We are not claiming an
optimally-tuned model — we are claiming a *data* effect under a fixed recipe.

### Q26. Did you repeat every experiment?

Not every one — by design. The two **load-bearing** conditions (A1, A3) are run
over **three seeds** (42/123/7, varying LoRA init, dropout, batch order,
train/val split); all others are single-seed (42) (§4.8). The single-seed
conditions are either off-the-shelf baselines (no training seed exists) or the
decorrelation contrast, whose verdict is a position effect that does not turn
on init variance (§4.8).

### Q27. How large is variance?

Reported transparently in Table stat-summary and §6.1: withholding A1
0.63±0.08 vs A3 0.13±0.01 (non-overlapping at *every* seed); persistence recall
A1 0.85±0.04; locale leakage A1 0.022±0.008; pairwise role-swap 0.82±0.06,
language 0.76±0.06. The largest mechanical gap (A3 0.000 vs trained ≥0.83) is
magnitude-defended: no plausible init variance closes it (§6.1).

### Q28. Would another random seed change the conclusions?

For the boundary claims, no. The three-seed spreads are non-overlapping on the
load-bearing contrasts, and the biggest gaps are orders of magnitude larger
than any seed spread (§6.1). We explicitly restrict which effects we lean on to
the seed-stable ones: on the pairwise eval we rely only on role-swap and
language, noting locale regresses toward parity across seeds (0.54±0.12, §5.6).

### Q29. Why only a 0.8B student?

Deliberate, not incidental: the boundary is "most consequential precisely where
a large general-purpose model is not deployable" (§1 Scope; §4.1). 0.8B on an
RTX 3060 is the regime where a practitioner genuinely must decide "train or
prompt?" A frontier model would make the question moot.

### Q30. Would a 3B student show the same boundary?

Direction: yes, per the mechanism (§6.5) and the cross-family evidence — a
Llama-3.2-1B trained student lifts both behaviors far above prompt-only, and a
Llama-3.1-8B prompt-only probe stays low even with CoT (§6.2). We also name
larger-scale testing (4B/7B) as priority future work, specifically for the
*decorrelation* side-result where a positional route may emerge at larger scale
(§6.3 item 2). The boundary direction is not expected to flip; magnitudes are
family/scale-dependent.

---

## 5. Generalization Questions

### Q31. Does the boundary hold for coding / translation / summarization / medical QA / legal QA?

Not tested here — stated honestly. What we offer instead of an untested claim:
(1) a *mechanism* (§6.5) that is task-agnostic — "missing cross-turn state" and
"overriding a competing prior" are general prompting failure modes; and (2)
named transfer targets where the same structure recurs (refusal-token,
tool-call emission; §7). We do **not** assert the boundary for those tasks; we
assert the mechanism predicts which behaviors within them would be
not-promptable. Overclaiming here is exactly what §1 Scope and §6.2 guard
against.

### Q32. Is this a tutor boundary or an LLM boundary?

The load-bearing behaviors are framed as **structural LLM behaviors**, not
tutor-specific ones: persistence = cross-turn violation counting + rare-marker
emission; withholding = suppressing the general-assistant helpfulness prior
(§5.3, §5.4, §6.5). The tutor setting is the *measurement apparatus* that makes
them cleanly isolable, not the source of the effect. The cross-family
replication (§6.2) treats them as base-model properties, supporting the
"LLM boundary, measured in a tutor" reading.

### Q33. Would GPT-5 show the same behavior?

Untested and out of scope — the whole study is small-model, consumer-hardware
by design (§4.1). Our data speak to the *direction* at the scales tested: a 9B
teacher and an 8B Llama both fail prompt-only despite huge size advantages over
the trained students (§5.3, §6.2), which argues the effect is not a
small-model-only artifact. We would not claim a specific GPT-5 number; the
honest statement is "the gap persists at every scale we could test, and the
mechanism predicts why."

### Q34. Would Claude show the same behavior?

Same answer as Q33: untested (no such experiment is in the paper), but the
cross-family result (Qwen *and* Llama both show the boundary; §6.2) makes a
same-family-artifact explanation unlikely. We report family-dependence of
*magnitude*, not of direction, and would not overclaim a specific frontier
model's number.

### Q35. Would DeepSeek show the same behavior?

Not tested. The generalization evidence we *do* have is two families (Qwen,
Llama) sharing the boundary direction (§6.2). A third family (Gemma is
suggested) is explicitly the top future-work item (§6.3 item 1). We present
two-family replication as the current evidence and name breadth as the open
edge, rather than speculate on specific untested models.

### Q36. How many model families are enough before making this claim?

We currently have **two trained families** (Qwen 0.8B, Llama 1B) plus a larger
prompt-only probe (Llama 8B) all showing the same direction (§6.2). We do not
claim a universal law from two; we claim the boundary "replicates in a second
trained family" and that "the direction is robust, the magnitude
family-dependent" — a calibrated statement (§1, §6.2). A third family is named
as the first future-work priority (§6.3). The claim's strength is deliberately
scoped to the evidence.

---

## 6. Evaluation Questions

### Q37. Why trust LLM judges?

We minimize reliance on them and de-bias where used. The two *headline*
not-promptable results lean hardest on **judge-free mechanical metrics**:
sentinel firing (persistence, §5.3) is a deterministic string match, and
locale leakage is a gazetteer regex (§4.7). Where judges are needed
(withholding, pairwise), we use a **cross-family** ensemble (Mistral/Meta/Google,
none sharing the Qwen teacher's lineage) to eliminate self-preference bias
(§4.5), report the median (or per-judge), and reduce withholding to a *binary*
withheld/answered label — far less subjective than a 1–5 rubric (§6.2 Judging).

### Q38. Why not human judges?

Cost/scale on a solo-consumer-hardware study, and the design was built to
*minimize* the subjectivity that would make human judging necessary: the
load-bearing metrics are mechanical, and the judged one is binary (§6.2). We
state the absence of a human study as a limitation candidate (Q14–Q15). Human
evaluation on a stratified sample is a reasonable revision addition.

### Q39. What happens if judges disagree?

For the median-reported metrics, disagreement is absorbed by taking the median
of three (§4.5). For withholding (only two binary-capable judges), we report
**each judge separately** rather than hiding disagreement — e.g. A1 0.571
(Llama) vs 0.651 (Gemma) — and the significance test clears under *each judge
independently* (Llama z=5.9, Gemma z=5.6), so the conclusion does not depend on
resolving a disagreement (§5.4).

### Q40. How often did judges disagree?

The paper reports per-judge numbers that let the reader read off disagreement
directly (e.g. §5.4 withholding table; §5.6 pairwise per-judge columns). The
key robustness fact: on the load-bearing withholding contrast the two judges
*agree in direction and significance* despite differing in absolute rate, and
on pairwise, the effects we lean on (role-swap, language) are favored by all
three judges (§5.6). A single aggregate disagreement-rate statistic is not
tabulated — a fair addition, computable from released per-judge outputs (§7).

### Q41. Why these judges?

A deliberate **cross-family constraint**: Prometheus-7B-v2 (Mistral lineage),
Llama-3.1-8B-Instruct (Meta), Gemma-2-9B-it (Google) — all distinct from the
Qwen-family teacher, so no judge shares lineage with the model that produced
the student's supervision, eliminating self-preference bias (§2.2, §4.5). The
9B Qwen teacher was *removed* from the judge ensemble for exactly this reason
(§4.4 B4 note).

### Q42. Could judge bias affect the results?

The design bounds it. Self-preference bias is removed by the cross-family
constraint (Q41). A residual "A1 is globally preferred" bias is bounded to near
zero by the **generic-redirect negative control**: on the stream *both* A1 and
A3 train on, the pairwise win-rate is 0.55 — near parity (§5.6). If judges
simply liked A1, the control would not sit at parity. Position bias in pairwise
is mitigated by randomized presentation order (§5.6).

### Q43. Would GPT-5 judge differently?

Untested. The safeguards are structural rather than judge-identity-specific:
median-of-three, cross-family lineage, binary criteria, and a negative control
that bounds global preference bias (§4.5, §5.6, §6.2). A different judge could
shift absolute rates, but the load-bearing contrasts survive per-judge (Q39),
and the headline persistence result uses *no judge at all* (§5.3). So a
different judge cannot overturn the mechanical spine of the boundary.

### Q44. How stable are pairwise evaluations?

Reported with dispersion: the two effects we lean on reproduce across three
seeds (role-swap 0.82±0.06, language 0.76±0.06) and the negative control stays
at parity (0.53±0.05); larger-variance axes (locale 0.54±0.12) are explicitly
*not* leaned on (§5.6, §4.8). Win-rates carry bootstrap 95% CIs over 1000
resamples where n supports (§4.8). Stability is reported, not assumed.

### Q45. Can humans reproduce the rankings?

The pairwise protocol (show both repairs in randomized order, pick the better)
is exactly a human-reproducible task, and all repairs + judge prompts are
released (§7), so a human replication is directly runnable. We do not *report*
a human replication (an honest gap, Q38), but the large, seed-stable effects
(role-swap 0.87) are the kind expected to survive it, and the negative control
guards against a spurious global preference.

---

## 7. Statistical Questions

### Q46. Why only three seeds?

Three seeds on the two load-bearing conditions (A1, A3) is a compute-bounded
choice on a single 12GB GPU (§4.1, §4.8), targeted where it matters most. With
three points we report the **spread transparently** (mean±s.d.) rather than a
cross-seed significance test we could not power, and we restrict claims to
effects whose three-seed spreads are non-overlapping or magnitude-dominant
(§4.8, §6.1).

### Q47. Why not ten seeds?

Compute: every seed is a full QLoRA train + multi-set, multi-judge eval with
model swaps on one GPU. Ten seeds × the load-bearing conditions was not
feasible. The mitigation is that the conclusions rest on gaps far larger than
the observed 3-seed s.d. (e.g. 0.000 vs ≥0.83; withholding non-overlapping at
every seed), so additional seeds would tighten CIs, not change signs (§6.1).

### Q48. Did you compute confidence intervals?

Yes, where sample size supports them: pairwise win-rates carry **bootstrap 95%
CIs** over 1000 resamples (§4.8). For the reseeded conditions we report
mean±s.d. over three seeds. Underpowered cells (context-dependent mechanical
scores, n≤25) carry an explicit small-n caveat rather than a false-precision CI
(§4.8, §5.7).

### Q49. Did you test significance correctly?

The load-bearing withholding contrast uses a **two-proportion z-test computed
per judge** on n=63 (Llama z=5.9, Gemma z=5.6, both p≪0.001), and the
edge-case teacher comparison is reported *as* edge (z=1.96/1.62, labeled
directional only) rather than overstated (§5.4, §4.8, §6.1). The largest
mechanical gap is defended by magnitude, not a test, because 0.000 vs ≥0.83 is
not a case a test is needed for (§6.1). We are explicit about which claims are
significance-backed vs directional vs underpowered.

### Q50. Why report mean instead of median?

Both are used, matched to the metric. Across-judge aggregation uses the
**median** (robust to one outlier judge; §4.5). Across-*seed* aggregation uses
**mean±s.d.** because with three seeds the mean plus spread is more informative
than a median that discards two of three points (§4.8). The choice is
deliberate per axis of aggregation, not uniform.

---

## 8. Boundary Questions (core scientific)

### Q51. How do you define a capability?

Operationally, off the deployment prompt: a capability is one **interaction
invariant** the tutor must maintain, each corresponding to exactly one clause
the rendered prompt already makes (language, topic, role, persona, pedagogy,
locale, + a general-appropriateness catch-all) (§3.3). This makes the capability
list *auditable* — anyone holding the prompt can check the invariant list
against its clauses — and we claim completeness only *relative to a given
deployment's commitment set* (§3.3).

### Q52. Why is persistence considered a capability?

Because it is a distinct behavior requiring machinery the single-shot axes do
not: counting same-axis violations *across turns* and emitting a rare literal
sentinel on the third strike (§3.3 persistent streams, §5.3). It is
*orthogonal* to the invariant axis (any violation can be one-off or repeated),
and it is fully specified in the deployment prompt (§3.5 persistence block), so
it is a legitimate per-clause capability to test.

### Q53. Could persistence simply require a longer context window?

No — the failure is not context-length. The 9B teacher sees the full dialogue
and the full persistence spec and still fires ≤0.06 zero/few-shot (§5.3). The
diagnostic is that *native chain-of-thought* — which externalizes the count
into generated tokens — partially recovers it (0.06→0.63) where few-shot and an
output scaffold do not (§5.3, §6.5). That isolates the deficit as **missing
cross-turn state maintained by the forward pass**, not missing context: give it
a scratchpad to hold the count and it improves; the tokens were always in
context.

### Q54. Is withholding really a capability or merely stronger instruction following?

It is a capability, evidenced by the **A3 ablation**, not by instruction
strength. A1 and A3 receive the *identical* prompt and differ only in the
pedagogy/specialized streams; A3 collapses to the untrained-base rate (0.12 ≈
B1 0.09) while A1 reaches 0.61 (§5.4). If it were just instruction-following,
the shared instruction would produce shared behavior. It does not: demonstration
re-weights the competing helpfulness prior; the clause alone leaves the prior in
control (§6.5).

### Q55. Could prompting improve with a better prompt?

We tested a *ladder* of stronger prompting, not one prompt: zero-shot →
few-shot (4 worked 3-strike dialogues spanning positions {5,7,9,11}) →
output-CoT scaffold → native CoT, on the strongest prompt-only model (§4.6.1,
§5.3). Few-shot and the output scaffold do **not** help (≤0.06); only native
CoT partially recovers (0.63), and even that stays below the trained student at
heavy inference cost. The withholding claim is additionally anchored on the *A3
ablation* (§5.4), which no prompt wording touches. So "a better prompt"
was steelmanned, and the structural behaviors still fail — see Q56.

### Q56. How do you know prompting has reached its limit?

We do not claim an absolute limit — we claim a *relocated boundary*, stated
precisely: "persistence resists zero-shot and few-shot prompting outright, and
is only partially recovered by native chain-of-thought — at an inference cost,
reliability, and accuracy deficit that SFT removes" (§5.3, §6.2). The honest
framing is: within a large, well-specified prompt plus few-shot plus output-CoT,
the behavior does not appear; only a fundamentally different regime (native
reasoning tokens) partially recovers it, and even then not to the trained
level. That is a characterized relocation, not an unfalsifiable "prompting can
never."

### Q57. Would larger models remove the boundary?

The evidence says no at every scale tested: 9B teacher ≤0.06 persistence /
0.45 withholding, and Llama-8B 0.27→0.55 even with CoT — both far above the
0.8B student in size yet below the *trained* small students (§5.3, §6.2). The
same-base Llama-1B student-vs-control contrast isolates training from scale
(§6.2). So scale does not close it; training does. Larger-scale
*decorrelation* testing is separately named as future work (§6.3).

### Q58. Is the boundary continuous rather than discrete?

There are two levels, and the paper is careful about this. The *acquisition*
boundary (does the behavior appear at all?) is sharp — promptable axes reach
parity, not-promptable ones fail prompt-only (§5.2). *Beneath* it there is a
continuous *quality* dimension on the promptable axes, where specialized data
still polishes an already-present behavior (role-swap pairwise 0.87 despite type
parity; §5.6). "Prompt determines acquisition, data determines quality" (§5.6)
— a discrete boundary with a continuous refinement layer under it.

### Q59. How should future researchers decide which side a new capability is on?

§6.5 gives the operational test: ask *can the deployment clause select a
behavior the prior already affords, or must training install state the forward
pass lacks or re-weight a prior the clause cannot overpower?* Promptable →
selectable from prior (locale, role-swap, topic). Not-promptable → needs
cross-turn state (persistence) or prior-override (withholding). The diagnostic
signatures are given too: native-CoT recovery signals a missing-state case;
ablation-to-baseline signals a competing-prior case (§6.5).

### Q60. Can the boundary predict training requirements before collecting data?

That is the intended payoff (§3.3, §6.5): classify each deployment clause by
the §6.5 test, and collect specialized demonstration data *only* for the
not-promptable behaviors (persistence, withholding), spending the budget where a
prompt will not do (§1 "Why it matters"). §5.4 confirms the prediction is
falsifiable and borne out — specialized data helps exactly where the prior does
not already afford the behavior. The map is meant to be used *before* data
collection, not only explained after.

---

## 9. Ablation Questions

### Q61. Which stream contributes the most?

The **pedagogy/specialized streams**, measured by the A1-vs-A3 contrast: their
removal collapses withholding from 0.61 to 0.12 (near untrained baseline) and
sentinel recall — via the persistent streams inside A1 — from ≥0.83 to 0.000
(§5.3, §5.4). On the quality layer, the largest single-axis effect is role-swap
(pairwise 0.87; §5.6). The specialized-redirect + persistent block is the
load-bearing contribution.

### Q62. What happens if you remove locale?

Locale fidelity is **prompt-driven, not data-driven**: A3 (which removes the
specialized locale stream) does *not* leak more Western-default entities than
A1 (A1 1.34%, A3 0.89%, statistically indistinguishable across three seeds;
§5.7). This is reported as a *correction* to any assumption that the locale
stream drives fidelity — the one-line "ground cultural items in {country}"
clause carries it. A clean promptable-axis result.

### Q63. Remove persistence (streams)?

That *is* the A3 vs A1 mechanical contrast on sentinel firing: A3, lacking the
persistent streams, fires on **0.000** of positive probes while trained
conditions fire ≥0.83 (§5.3). This is the single cleanest, epoch- and
seed-independent result in the paper — removing the demonstration removes the
behavior entirely.

### Q64. Remove pedagogy?

Covered by A3 (Q54, Q61): withholding drops to 0.119, essentially the
untrained-base rate (B1 0.087), with the A1-vs-A3 gap significant under each
judge (z=5.9/5.6) and non-overlapping across all three seeds (§5.4). Pedagogy
withholding is the behavior most dependent on demonstration.

### Q65. Remove role swap?

Role-swap is **promptable as type** (every model deflects — the type-F1 ceiling,
Appendix A) but shows the **largest specialized-data quality win** (pairwise
0.87, Gemma preferring A1 on every pair; §5.6). So removing the role-swap
stream does not remove the behavior (prompt covers acquisition) but degrades
repair *quality*. This is the headline reconciliation of §5.6 — the F1 ceiling
concealed a real, large effect.

### Q66. Remove topic redirect?

Topic is near-parity on quality (pairwise 0.57, the smallest effect) and
promptable as type (§5.6). Removing it is expected to have little effect — it
sits firmly on the promptable side, consistent with the mechanism (topic
re-anchoring is a behavior the prior already affords; §6.5). We do not lean on
topic for any data-necessity claim.

### Q67. Remove language redirect?

Language is the **one context-dependent mechanical signal that may favor data**:
A1 0.71 vs A3 0.33 on L1-acknowledge-and-return, consistent with the language
pairwise win (0.75) (§5.7, §5.6). Reported as suggestive-and-underpowered
(n≤25) but one of the two effects (with role-swap) stable enough across seeds to
lean on (§4.8). Removing it plausibly costs measurable quality.

### Q68. Remove persona redirect?

Persona is promptable as type and near-parity on quality (pairwise 0.61; §5.6).
Like role-swap it has a context-free surface signature (§3.3), so acquisition is
prompt-covered; the data quality effect is modest. Removing it is low-impact on
the boundary claim.

### Q69. Could fewer streams achieve the same performance?

Partly — and the paper says so honestly. The *promptable* axes (locale, topic,
persona, and role-swap/topic at the acquisition level) reach parity prompt-only,
so their specialized streams are not required for the behavior to *appear* (§5.7,
§6.5) — they buy only quality (§5.6). The streams that are **necessary** are the
persistent and pedagogy streams, without which persistence and withholding
collapse (§5.3, §5.4). So a leaner corpus could focus specialized data on the
not-promptable behaviors — which is precisely the practical recommendation the
boundary yields (Q60).

---

## 10. Reviewer Attack Questions (hardest)

### Q70. Your dataset was generated by the same teacher used for evaluation — is this circular?

The teacher generates *training* data; it is **not** in the judge ensemble (it
was explicitly removed, §4.4 B4 note). Judged metrics use a **cross-family**
ensemble (Mistral/Meta/Google) sharing no lineage with the Qwen teacher (§4.5),
so the student is never scored by a checkpoint sharing the teacher's lineage.
The headline persistence metric is **judge-free and mechanical** (string match;
§5.3), so it has no judge circularity at all. Where the teacher *does* appear
is as a prompt-only *baseline* (B4) — and there it is the strongest baseline the
student must beat, which strengthens rather than weakens the boundary claim.

### Q71. Your student simply memorized the teacher — how do you rule this out?

Several ways. (1) The teacher *itself* fails prompt-only on the not-promptable
behaviors (≤0.06 persistence, 0.45 withholding; §5.3, §5.4) — you cannot
memorize a behavior the source does not exhibit under the prompt. The student at
0.83/0.61 *exceeds* the teacher, which is the opposite of copying. (2)
Evaluation is on a **hash-deterministic held-out split** the student never
trained on (§4.3, §6.19). (3) The behavior generalizes: sentinel firing holds
on **off-grid positions {13,15}** the training never used (Persistent-OffPosition-Probe,
§4.3) — a memorizer would fail there, a trigger-detector fires. (4) The boundary
replicates in a *different* base family (Llama; §6.2). Memorization explains
none of these.

### Q72. The deployment prompt is extremely detailed — would a realistic industrial prompt be this long?

Yes — a real deployed tutor *is* governed by exactly this kind of long,
clause-by-clause system prompt; that realism is the premise, not an artifact
(§1). And the detail *helps the opposing side*: a maximally-specified prompt is
the **strongest possible case for prompting**. That the not-promptable
behaviors fail *despite* every behavior being spelled out is what makes the
failure interpretable — under-specification is ruled out by construction (§5.1).
A shorter, weaker prompt would only make prompting look worse.

### Q73. Your boundary may only exist because of your prompt wording.

The not-promptable failures are **structural, not lexical** (§6.5): persistence
fails because the forward pass maintains no cross-turn counter (diagnosed by the
native-CoT recovery, §5.3), and withholding fails because a competing prior must
be re-weighted (diagnosed by the A3 ablation collapsing to baseline, §5.4).
Neither diagnosis depends on wording. We also varied the *prompting regime*
(ladder up to native CoT, §4.6.1) — a far stronger perturbation than rewording —
and the structural behaviors still failed. The A3 ablation in particular holds
prompt wording *fixed* and varies only data, isolating the effect from wording
entirely.

### Q74. You claim prompting cannot solve persistence — how many prompts did you actually try?

We ran a **four-rung prompting ladder** on the strongest prompt-only model, each
a genuine strengthening: (1) zero-shot full spec, (2) + few-shot (4 worked
3-strike dialogues spanning {5,7,9,11}), (3) + output-CoT strike-tally scaffold,
(4) + native chain-of-thought (§4.6.1, §5.3). The claim is stated precisely to
match what we tested: resists zero-shot *and few-shot* outright; only
*partially* recovered by native CoT, still below trained (§5.3, §6.2). We do not
claim "no prompt could ever" — we claim a characterized relocation across a
tested ladder (Q56).

### Q75. How do you know another prompting strategy would not work?

We do not claim omniscience over all strategies — we claim the ones on the
ladder (few-shot, output-CoT) fail and native CoT only partially succeeds, and
crucially that *even native CoT's* recovery is bounded by a **delivery** failure
at deployment budget (the model reaches "fire" inside `<think>` but the visible
answer omits the sentinel in ~1/3 of cases; §5.3). The withholding half of the
claim rests on the *A3 ablation*, which is strategy-independent (§5.4). So the
claim is empirically bounded and mechanism-backed, not a blanket impossibility.

### Q76. Your conclusions depend heavily on one task — isn't this overclaiming?

The paper is deliberately scoped to avoid this: §1 "Scope and non-goals" and
§6.2 restrict the claim to what is shown, the contribution is stated as *one*
thing (§1), and generalization beyond tutoring is offered as *mechanism* (§6.5)
plus *named transfer targets* (§7), not as tested fact. The cross-*family*
replication (§6.2) shows the effect is not even Qwen-specific, let alone a
single-run artifact. We claim a boundary demonstrated in one task with a
task-agnostic mechanism — not a proven universal.

### Q77. Why should reviewers believe this is a general principle?

Two independent supports beyond the single task: (1) **replication in a second
trained family** (Llama-1B: persistence 0.25→0.91, withholding 0.11→0.50, with a
same-base control isolating training from scale; §6.2); and (2) a **mechanism**
(§6.5) that predicts the side of the boundary from a clause's relation to the
model's prior, testable in future work (probing for an internal counter,
measuring prior strength). Generality is argued as "direction robust across
families, magnitude family-dependent" — a calibrated claim (§6.2), not a leap.

### Q78. Could reinforcement learning eliminate the boundary?

RL is not tested (SFT-only design, §3, §4.1). Conceptually, RL/DPO would most
plausibly act on the *quality* layer beneath the boundary (§5.6) or further
re-weight the competing prior for withholding — i.e. operate on the *training*
side. It would not make a behavior *promptable* (that is a property of the
prompt-only condition, which RL on the student does not touch). We note DPO as
future work (§3.6, §6.3) and would frame RL as another training method, still on
the "must be trained" side.

### Q79. Could tool use eliminate the boundary?

Not tested. But note the mechanism predicts the outcome: an external
violation-counter tool would supply exactly the "missing cross-turn state" that
§6.5 identifies as persistence's deficit — consistent with, not contrary to, our
account (an external counter is a way to *install the state the forward pass
lacks*). That would move persistence across the line by adding machinery, which
is precisely the point: the behavior needs *something beyond the prompt*, whether
demonstration or a tool. It supports the boundary's framing rather than
refuting it.

### Q80. Could memory modules eliminate persistence failures?

Same logic as Q79: a memory module that persists the violation count is another
way to supply the missing cross-turn state (§6.5). Our claim is that the
behavior is *not promptable* — it needs installed state — and a memory module is
one mechanism for installing that state, alongside SFT and native-CoT
scratchpads (the latter already shown to partially help, §5.3). It corroborates
the diagnosis rather than undermining the boundary.

---

## 11. Writing Questions

### Q81. The Introduction is very long — can it be shortened?

Yes. The intro is structured as labeled blocks (Problem / Why it matters / Gap /
Contribution / Results / Boundary-replicates / Scope / Organization; §1). The
compressible parts are the Results and boundary-replication paragraphs, which
duplicate numbers stated again in §5 and §6.2 — these can be trimmed to
one-line pointers. The Contribution and Scope blocks should stay (they are what
reviewers check first). A tightened intro keeps Problem/Gap/Contribution and
defers detailed numbers.

### Q82. Some contributions overlap — can they be merged?

The paper already asserts a **single** contribution (the boundary) and demotes
everything else to "reusable apparatus / bounded side-result" explicitly (§1,
§2.5 positioning table). If a reviewer still sees overlap, the fix is to merge
the apparatus bullets (pipeline + locale-judge audit) into one "released
apparatus" sentence and the two side-results (retired-F1 + decorrelation) into
one "bounded side-results" sentence, so the single contribution stands even more
alone.

### Q83. The paper repeats the matched-prompt idea many times.

Intentional but reducible. The matched-prompt protocol is the interpretive
linchpin (it is *what makes a prompt-only failure mean "not promptable"*), so it
is restated in §1, §4.4, §5.1, and figure captions. For revision, keep the full
statement once (§5.1) and replace the others with a short back-reference. Some
repetition in the intro and results is defensible because reviewers read
non-linearly, but the caption-level repeats can go.

### Q84. Some tables belong in the appendix.

Agreed for several. The per-axis F1 bimodality table is *already* in Appendix A.
Candidates to move: the full decorrelation/recall-by-turn tables (already in
Appendix B.6), and possibly the zero-shot-baseline definition table (§4.4) could
compress into text. The load-bearing tables to keep in the body: stat-summary
(§5.2), persistence ladder (§5.3), withholding (§5.4), pairwise (§5.6), and
cross-family (§6.2).

### Q85. Some implementation details interrupt the scientific narrative.

Valid. The pipeline mechanics (six-filter cascade, yield-aware top-up,
resumability, locale-block split) are apparatus and mostly already relegated to
§3.6 pointers + Appendix B. The remaining in-body engineering detail (e.g. the
allow-L1 locale-block split, §3.5) could move to the appendix, keeping §3 focused
on the taxonomy and the deployment prompt (the two things the boundary claim
actually needs).

---

## 12. Accept-or-Reject Questions

### Q86. What would be lost if this paper were not published?

The practitioner-facing decision procedure would stay folklore: teams building
small task models would keep discovering per-behavior, by trial and error, that
some prompt clauses "just work" and others silently do not — with no map and no
mechanism to predict which. The paper converts that into a testable,
data-budget-guiding boundary with a mechanism (§6.5) and a concrete
before-you-collect-data test (Q60). Also lost: the evidenced *negative* result
that the conventional redirect-axis F1 is a type-not-quality classifier that
ties a 0.8B student with a 9B teacher (§5.5, Appendix A) — a trap others would
keep falling into.

### Q87. After reading the paper, what new scientific knowledge have I gained?

(1) That deployed-tutor behaviors split *sharply* into promptable vs
not-promptable under a maximally-specified prompt, and *which* fall where (§5.2).
(2) *Why* they split — the §6.5 mechanism (select-from-prior vs
install-missing-state vs override-competing-prior), with diagnostic signatures
(native-CoT recovery, ablation-to-baseline). (3) That this is
training-not-scale (a 9B/8B fail prompt-only; a same-base 1B trained student
succeeds; §6.2). (4) That a widely-used redirect F1 is measuring type, not
quality (§5.5). Concretely: a rule for deciding, before collecting data, what a
prompt will and will not buy.

### Q88. Can another researcher build upon this work?

Yes, and the paper is built for it: all code, configs, seeds, frozen eval sets,
judge prompts, the synthetic dataset, and per-condition score outputs are
released with a claim→script→number reproducibility guide (§7). The mechanism
(§6.5) proposes directly-testable follow-ups (probe for an internal violation
counter; measure helpfulness-prior strength across families). The methodology
(matched-prompt per-capability evaluation) transfers to any deployed prompt-governed
task, and §7 names refusal-token and tool-call emission as immediate next
applications.

### Q89. Will this paper still matter in five years?

The *specific numbers* (0.8B, Qwen3.5, an RTX 3060) will age. The *question* —
"which behaviors must be trained vs prompted, and how do I tell in advance?" —
does not, as long as practitioners deploy prompt-governed models under budget
constraints. The mechanism (missing state / competing prior) is a durable lens.
The matched-prompt methodology and the type-vs-quality F1 caution are reusable
regardless of model generation. What dates fastest is the scale regime; what
persists is the boundary framing.

### Q90. Is this a new scientific phenomenon or merely an engineering result?

A phenomenon, on three grounds. (1) It is *categorical and interpretable*, not a
tuning curve: behaviors split cleanly, and the split tracks a single mechanistic
question (§6.5). (2) It is *predictive* — the §3.3 taxonomy turns "generic data
is insufficient" into a falsifiable per-axis prediction that §5.4 confirms. (3)
It *replicates across model families* with a stated direction-robust /
magnitude-dependent law (§6.2). The engineering (pipeline, judges) is the
*apparatus* that isolated the phenomenon; the phenomenon is the boundary and its
mechanism. An engineering result would be "here is a good tutor model"; this is
"here is a reproducible law about what prompting can and cannot install, and
why."

---

## Appendix: The five highest-risk questions (per author's own assessment)

- **Q1 (single main contribution)** — answer: the train-vs-prompt boundary;
  everything else is explicitly demoted apparatus/side-result (§1, §2.5).
- **Q31/Q32 (generality beyond tutoring)** — answer: mechanism is task-agnostic
  (§6.5) + cross-family replication (§6.2); we claim mechanism + named transfer
  targets, *not* untested tasks (§7). Scoped to avoid overclaiming.
- **Q55/Q56 (prompting's practical limit)** — answer: a four-rung prompting
  ladder (§4.6.1, §5.3); claim is a *characterized relocation* ("resists
  zero/few-shot, partial native-CoT recovery at a deficit SFT removes"), not an
  absolute impossibility.
- **Q70/Q71 (teacher / evaluation circularity)** — answer: teacher excluded from
  judges (cross-family ensemble); headline metric judge-free; student *exceeds*
  the teacher it supposedly copied; off-grid + cross-family generalization
  rules out memorization (§4.5, §5.3, §6.2).
- **Q90 (science vs engineering)** — answer: categorical + predictive +
  cross-family-replicated phenomenon with a mechanism (§6.5), distinguished from
  its enabling apparatus.
