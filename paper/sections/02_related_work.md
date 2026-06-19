# 2. Related Work

Our pipeline draws on four lines of prior work. We discuss each in
turn, then position our contributions against them.

## 2.1 Synthetic instruction-tuning data

The general technique of using a strong "teacher" model to bootstrap
instruction-tuning data for a smaller "student" was popularised by
Self-Instruct [@wang2023selfinstruct] and pushed further by
Evol-Instruct / WizardLM [@xu2023wizardlm] which iteratively rewrite
instructions to broaden complexity. UltraChat [@ding2023ultrachat]
extends to multi-turn dialogue at scale. OpenAssistant
[@kopf2023openassistant] produced a human-annotated instruction
dataset whose preference labels enabled subsequent RLHF / DPO work.

These pipelines share two assumptions our work departs from. First,
they target *general* instruction-following capability and apply a
single uniform generation recipe; they do not decompose the
instruction space by behavior axis. Second, the filter step is
typically a length / format check or an off-the-shelf safety filter;
there is no notion of per-locale or per-axis filtering beyond the
generic.

The closest prior work in spirit is Vicuna's "deduplicated and
filtered ShareGPT" approach [@chiang2023vicuna] and the Tulu-3
[@lambert2024tulu3] mix-blending recipe, both of which carefully
compose datasets from sub-pools. Our **declarative ratio-target**
mechanism is closely related: it generalises mix blending to a
yield-aware iterative top-up so the operator can declare per-stream
mix shares without recomputing arithmetic as yields shift.

## 2.2 LLM-as-judge filtering

Filtering generated data with another LLM is now standard practice.
Prometheus [@kim2024prometheus], JudgeLM [@zhu2023judgelm], and
PandaLM [@wang2023pandalm] propose dedicated judge models;
AlpacaEval / MT-Bench [@zheng2023judging] use frontier models as
judges. A growing strand of work documents the limits of LLM
judges: position bias, length bias, judge-style bias, and self-
preference bias [@wang2023pandalm; @saito2023verbosity;
@panickssery2024selfpreference]. The standard mitigation is
multi-judge consensus with paired-comparison protocols. Our own
judge ensemble (§4.7) follows the multi-judge mitigation but adds
a deliberate cross-family constraint: we use Prometheus-7B-v2
(Mistral lineage), Llama-3.1-8B-Instruct (Meta), and Gemma-2-9B-it
(Google) — three families all distinct from the Qwen-family
teacher — so that no judge shares pre-training or post-training
lineage with the model that produced the student's supervision
data. Prometheus is doubly relevant here: it functions both as
related work on judge-model methodology and as a working component
of our evaluation pipeline.

Our **`locale_judge`** is a specialised single-axis judge whose only
job is to classify entities as `in_locale` or `out_of_locale` for
the configured locale. We document a failure mode that to our
knowledge has not been catalogued in this literature: **systematic
false positives on common-English sentence-initial words and
locally-canonical landmarks**, with FP rates of ~85% on the judge's
rejections in our setting. The methodological lesson (that LLM-judge
filters benefit from aggressive in-locale and English-word
allowlists, and that the rejection log requires periodic auditing)
is, we believe, the part of this finding that generalises beyond
locale.

## 2.3 Tutor and educational LLMs

Recent tutor systems include EduChat [@dan2023educhat] for general
educational dialogue, MathDial [@macina2023mathdial] for math tutor
behavior with explicit scaffolding moves, and a small set of
language-learning specific systems [@caines2023chatbots;
@tyen2022opendomain]. CEFR-aligned datasets for English learners
include EFCAMDAT [@geertzen2014efcamdat] and TLE
[@yannakoudakis2018tle], but these are *learner-produced* corpora
rather than tutor-side training data.

LearningQ [@chen2018learningq] and similar resources provide
question-answer pairs at varying difficulty levels but are not
multi-turn. To our knowledge, no publicly described tutor dataset
both (i) stratifies systematically across CEFR levels A1–C2 and
(ii) covers multi-axis redirect / persistent abuse handling. Our
invariant-based taxonomy (§3.3) is our contribution to this gap: it
treats redirect behavior as the restoration of a small, enumerated
set of interaction invariants rather than as a single undifferentiated
"handle bad input" behavior.

## 2.4 Persona, safety, and adversarial dialogue data

Adversarial conversational data is well-studied for safety. BBQ
[@parrish2022bbq] targets bias-eliciting question forms;
HarmBench [@mazeika2024harmbench] systematically catalogues unsafe
inputs across categories; AdvBench [@zou2023advbench] focuses on
jailbreak prompts. Persona consistency has been studied as both a
training objective [@zhang2018personas] and an attack surface
through which assigned personas amplify toxicity
[@deshpande2023toxicity].

Most of this work treats safety as a *single-turn* problem: a single
unsafe prompt, a single refusal. Two contributions of our pipeline
are not addressed by this prior work. First, **invariant-based
multi-axis decomposition**: we separate redirect behavior into seven
single-shot axes because the correct *response shape* — the minimal
repair — differs by invariant (a locale violation is repaired by a
brief in-locale redirect; a pedagogy weakness by scaffolding rather
than answering; a code-switch by acknowledging L1 and steering
back). Second, **multi-turn persistence**: the persistent 3-strike
streams test whether a student trained on fixed-position sentinel
data learns *the third strike* or only *turn N*.

**Shortcut learning and the positional defense.** The failure our
persistent streams target is a dialogue-level instance of a general
phenomenon: neural models latch onto superficial cues that are
correlated with the label in training but spurious with respect to
the intended task — *shortcut learning* [@geirhos2020shortcut]. The
phenomenon is well documented in natural-language inference, where
annotation artifacts and shallow syntactic heuristics let models
succeed without the intended reasoning [@gururangan2018artifacts;
@mccoy2019hans]. In our setting the spurious cue is turn position:
under a fixed-turn sentinel, "turn 7" is perfectly correlated with
"third strike," so the student learns the cheaper positional rule.
The standard remedy in that literature is to break the spurious
correlation in the data itself. Our 4-variant construction (§3.4) is
the multi-turn dialogue-sentinel form of that remedy, with the
added engineering constraints — unaddressed by the NLI work — that
the position resampling be *deterministic* (for resumability and
train/eval-split coherence) and confined to a *parity-valid
codomain*. We are not aware of a directly comparable construction in
the multi-turn adversarial-dialogue literature.

## 2.5 Positioning

Against this backdrop, our contributions are positioned as follows:

| Prior work axis | Our contribution beyond |
| --- | --- |
| Self-Instruct / Evol-Instruct / WizardLM | Invariant-based multi-axis decomposition + yield-aware ratio targets |
| LLM-judge filtering | Locale-judge with allowlist defenses + FP audit methodology |
| Tutor / educational LLMs | Six-level CEFR × twelve-stream invariant taxonomy + locale-aware prompts |
| Single-turn safety / persona data | Multi-turn persistent 3-strike + trigger-position decorrelation defense |
| Shortcut learning / spurious cues (NLI) | Multi-turn dialogue-sentinel instantiation with deterministic, parity-constrained resampling |

We empirically demonstrate the pipeline at the smallest practical
scale (0.8B student on consumer GPU) so the result transfers to
deployment contexts where a much larger model is not feasible.
