# 2. Related Work

**Synthetic instruction data.** Teacher-to-student bootstrapping was
popularised by Self-Instruct [@wang2023selfinstruct] and Evol-Instruct /
WizardLM [@xu2023wizardlm]; UltraChat [@ding2023ultrachat] extends to
multi-turn dialogue, and Tulu-3 [@lambert2024tulu3] composes datasets from
sub-pools. These target *general* instruction-following with a uniform
recipe and do not ask which taught behaviors a prompt already elicits.

**LLM-as-judge filtering.** Prometheus [@kim2024prometheus], JudgeLM
[@zhu2023judgelm], and PandaLM [@wang2023pandalm] propose dedicated judges;
a strand documents position, length, and self-preference bias
[@saito2023verbosity; @panickssery2024selfpreference], mitigated by
multi-judge consensus. Our pairwise eval (§5) follows this with a
cross-family constraint so no judge shares the teacher's lineage; we add a
*negative* result about a judged metric (context-blind redirect F1).

**Tutor and educational LLMs.** EduChat [@dan2023educhat] and MathDial
[@macina2023mathdial] target educational dialogue; CEFR-aligned corpora
(EFCAMDAT [@geertzen2014efcamdat], TLE [@berzak2016tle]) are
*learner-produced*, not tutor-side. None asks, per behavior, whether the
deployment prompt already suffices. MathDial's scaffolding moves are the
closest analogue to our pedagogical-withholding axis.

**Safety, persona, and shortcut learning.** Adversarial safety data
[@parrish2022bbq; @mazeika2024harmbench; @zou2023advbench] and persona
consistency [@zhang2018personas; @deshpande2023toxicity] mostly treat
redirection as *single-turn*. Our three-strike persistence is multi-turn and
carries a *shortcut-learning* [@geirhos2020shortcut; @mccoy2019hans] risk: a
positionally-regular sentinel invites a position-as-trigger shortcut, which
our decorrelation construction (§3) targets.

**Positioning.** The prompting-vs-fine-tuning question has been examined for
*general* alignment [@zhou2023lima; @ouyang2022instructgpt; @min2022rethinking];
we localize it to the *per-behavior* level for a deployed task model, under a
matched-prompt protocol giving the same prompt to trained and prompt-only
models alike.
