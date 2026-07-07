# 7. Conclusion

We asked, per capability, which behaviors a deployed tutor needs can be
elicited by an explicit system prompt and which must be demonstrated through
fine-tuning. Evaluating a fine-tuned 0.8B student and a ladder of prompt-only
baselines up to a 9B teacher **under the same fully-specified deployment
prompt**, we find a sharp and interpretable boundary. Behaviors a single
clause elicits reach prompt-only parity (locale fidelity, role-swap and topic
deflection). Behaviors the prompt *describes but cannot install* do not:
persistence resists zero-shot and few-shot prompting (recall $\leq 0.06$),
recovers only partially under native chain-of-thought (0.63, below the trained
0.83 and at heavy inference cost), and fires 0.000 for the no-data ablation;
withholding stays at 0.09–0.45 prompt-only against the trained student's 0.61,
collapsing to baseline when the pedagogy stream is removed. The line is
interpretable — promptable when one clause both describes *and* elicits, not
promptable when the behavior needs cross-turn state-tracking or the
suppression of a strong competing prior. Mapping this boundary under a
matched-prompt protocol is the paper's contribution.

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
prompt gives for free and which require the pipeline. The boundary already
replicates in a second trained family (a Llama-3.2-1B student, §6.2); the open
edges, in priority order (§6.3): **broader cross-family replication** (a third
family, a base-checkpoint student, multi-locale); **larger-scale
decorrelation**; a **better-powered pedagogy teacher comparison**; and transfer
of the matched-prompt methodology to other rare, semantically-triggered markers
(refusal-token and tool-call emission), where the same "describable but not
promptable" question applies.

## Limitations

We surface the limitations detailed in §6.1–§6.2 here for visibility.

- **Primarily one model family** (Qwen). A cross-family probe corroborates the
  boundary outside Qwen — prompt-only at 8B and, more decisively, a *trained*
  Llama-3.2-1B student (persistence recall 0.91 vs 0.25 for the same untrained
  base, §6.2). Remaining breadth (a third family, a base-checkpoint Llama
  student, multi-locale) is future work.
- **Single locale** (`china`) — does not threaten the central boundary
  (persistence and withholding are locale-independent structural behaviors,
  §6.2), but the locale-fidelity axis result and the `locale_judge` gazetteer
  are locale-specific.
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
