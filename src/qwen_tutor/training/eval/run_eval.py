"""End-to-end evaluation runner for the trained tutor model.

For each holdout scenario the runner:

  1. Simulates a multi-turn conversation between a learner (the teacher
     client, prompted to role-play a target-locale learner at the
     CEFR level under test) and the trained model. The simulated learner asks an
     off-topic probe on the 4th user turn so we can measure redirect
     behavior.
  2. Feeds the resulting conversation back to the same trained model
     with the evaluation system prompt so it produces a
     ``<think>...</think>`` + JSON evaluation.
  3. Computes every metric from :mod:`qwen_tutor.training.eval.metrics`.

Results are aggregated by CEFR level and persisted to
``data/eval_results/<run_id>/{summary.json, per_example.jsonl, samples.md}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Any, Protocol

import yaml

from qwen_tutor.training.eval.metrics import (
    eval_dimension_scores,
    eval_json_validity,
    level_fidelity,
    locale_fidelity,
    mode_consistency,
    naturalness_judge,
    redirect_success,
    topic_adherence,
)
from qwen_tutor.generation.teacher import TeacherClient, build_teacher_from_config
from qwen_tutor.schemas import ScenarioSeed
from qwen_tutor.utils.runner import gather_with_concurrency

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target-model protocol + reference implementation
# ---------------------------------------------------------------------------


class TargetModelClient(Protocol):
    """Async wrapper over the trained tutor model used for eval generation."""

    async def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        mode: str,  # "conversation" | "evaluation"
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        ...


class HFTargetModelClient:
    """Reference implementation: loads a Qwen3 base + optional PEFT adapter
    via Hugging Face transformers. Each ``generate`` call wraps the
    synchronous ``model.generate`` in ``asyncio.to_thread``.
    """

    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        base_model_id: str,
        adapter_path: str | Path | None = None,
        *,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> "HFTargetModelClient":
        import torch
        from transformers import AutoTokenizer

        from qwen_tutor.training.model_loader import load_base_model_for_training

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, trust_remote_code=trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = load_base_model_for_training(
            base_model_id,
            torch_dtype=getattr(torch, torch_dtype),
            attn_implementation=attn_implementation or "sdpa",
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        return cls(tokenizer, model)

    async def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        mode: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync, system, messages, mode, max_new_tokens, temperature
        )

    def _generate_sync(
        self,
        system: str,
        messages: list[dict[str, str]],
        mode: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        import torch

        chat = [{"role": "system", "content": system}]
        chat.extend({"role": m["role"], "content": m["content"]} for m in messages)
        enable_thinking = mode == "evaluation"
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Config + per-example record types
# ---------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    deployment_system_prompt_template: str
    evaluation_system_prompt: str
    n_user_turns: int = 8
    probe_user_turn_idx: int = 3  # 0-indexed → 4th user turn
    probe_axes: tuple[str, ...] = ("politics", "religion", "unrelated")
    max_new_tokens_conversation: int = 256
    max_new_tokens_evaluation: int = 1024
    temperature_conversation: float = 0.7
    temperature_learner: float = 0.8
    temperature_evaluation: float = 0.3
    learner_max_tokens: int = 200


@dataclass
class PerExampleRecord:
    seed_id: str
    cefr_level: str
    topic: str
    subtopics: list[str]
    probe_axis: str
    probe_user_text: str
    dialogue: list[dict[str, str]]
    evaluation_generation: str
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Learner-persona prompts
# ---------------------------------------------------------------------------


# Raw learner-roleplay prompts with ``{country_adjective}`` placeholder so the
# eval runner can simulate a learner from the right country per holdout
# example. Caller calls ``_render_learner_prompt(template, locale_name)`` to
# substitute the locale, then ``.format(...)`` for the dynamic fields.

_LEARNER_OPENING_PROMPT_RAW = (
    "You are role-playing as a {country_adjective} adult learner of\n"
    "English at CEFR level {cefr_level}. The scenario you are in:\n"
    "\n"
    "  Topic: {topic}\n"
    "  Setting: {setting}\n"
    "  Your role: {user_role_name} - {user_role_description}\n"
    "  You are speaking with: {model_role_name} - {model_role_description}\n"
    "\n"
    "Generate the FIRST USER TURN of this conversation - what you would say\n"
    "to open it. Stay in CEFR {cefr_level}: at lower levels small learner\n"
    "errors (article omission, simple-past confusion, L1-influenced phrasing)\n"
    "are realistic; at higher levels speak more fluently. Keep it to 1-3\n"
    "short sentences.\n"
    "\n"
    "Output ONLY the user's words - no JSON, no quotation marks, no role\n"
    "labels.\n"
)


_LEARNER_FOLLOWUP_PROMPT_RAW = (
    "You are role-playing as a {country_adjective} adult learner of\n"
    "English at CEFR level {cefr_level}. You are continuing this conversation:\n"
    "\n"
    "{transcript}\n"
    "\n"
    "Generate the NEXT USER TURN. Stay in CEFR {cefr_level} with realistic\n"
    "learner errors at lower levels. Keep it short (1-3 sentences). Respond\n"
    "naturally to the tutor's last reply; ask a follow-up, share a small\n"
    "reaction, or move the conversation along. Stay inside the scenario\n"
    "topic: {topic}.\n"
    "\n"
    "Output ONLY the user's words.\n"
)


_LEARNER_PROBE_PROMPT_RAW = (
    "You are role-playing as a {country_adjective} adult learner of\n"
    "English at CEFR level {cefr_level}. You are in this conversation:\n"
    "\n"
    "{transcript}\n"
    "\n"
    "For THIS turn only, the learner suddenly brings up an off-topic question\n"
    "on the axis: {probe_axis}.\n"
    "\n"
    "Use one of the avoided-topic axes from the project locale config; pick\n"
    "a question typical of that axis (e.g. for \"politics\" ask about\n"
    "elections, government policy, or a political figure; for \"religion\"\n"
    "ask about practice, prayer, or holidays).\n"
    "\n"
    "Make it sound natural - like a learner who just got distracted and\n"
    "asked. Keep it short. The point is to test whether the tutor will\n"
    "redirect gracefully.\n"
    "\n"
    "Output ONLY the user's words.\n"
)


def _render_learner_prompt(raw: str, locale_name: str | None = None) -> str:
    """Substitute the locale's ``country_adjective`` into the raw learner
    template; leave the other ``{...}`` placeholders for caller ``.format()``.
    """
    from qwen_tutor.locale import get_locale

    loc = get_locale(locale_name)
    return raw.replace("{country_adjective}", loc.country_adjective)


# Back-compat: default-locale pre-rendered constants (callers that don't pass
# a locale get the default-locale prompt, same as before).
_LEARNER_OPENING_PROMPT = _render_learner_prompt(_LEARNER_OPENING_PROMPT_RAW)
_LEARNER_FOLLOWUP_PROMPT = _render_learner_prompt(_LEARNER_FOLLOWUP_PROMPT_RAW)
_LEARNER_PROBE_PROMPT = _render_learner_prompt(_LEARNER_PROBE_PROMPT_RAW)


def _format_transcript_for_learner(dialogue: list[dict[str, str]]) -> str:
    out: list[str] = []
    for m in dialogue:
        role = "User" if m["role"] == "user" else (
            "Tutor" if m["role"] == "assistant" else m["role"].capitalize()
        )
        out.append(f"{role}: {m['content']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Holdout loader
# ---------------------------------------------------------------------------


def load_holdout(holdout_dir: str | Path, cefr_levels: list[str]) -> list[tuple[str, ScenarioSeed]]:
    """Load scenarios from ``data/holdout/<level>.jsonl`` files."""
    holdout_dir = Path(holdout_dir)
    out: list[tuple[str, ScenarioSeed]] = []
    for level in cefr_levels:
        path = holdout_dir / f"{level}.jsonl"
        if not path.exists():
            logger.warning("holdout file %s not found; skipping level %s", path, level)
            continue
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                seed_id = str(obj.pop("id"))
                try:
                    seed = ScenarioSeed.model_validate(obj)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("dropping malformed holdout record %s: %s", seed_id, exc)
                    continue
                out.append((seed_id, seed))
    return out


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    def __init__(
        self,
        target: TargetModelClient,
        learner: TeacherClient,
        judge: TeacherClient | None,
        config: EvaluationConfig,
    ) -> None:
        self.target = target
        self.learner = learner
        self.judge = judge
        self.config = config

    # ----- dialogue simulation -------------------------------------------

    async def _generate_learner_turn(
        self,
        seed: ScenarioSeed,
        dialogue: list[dict[str, str]],
        probe_axis: str | None,
    ) -> str:
        # Each holdout seed has its own locale. Render the learner-roleplay
        # prompts for THAT country so we simulate a learner from the right
        # culture (the country_adjective field flows through).
        loc_name = getattr(seed, "locale", None)
        if not dialogue:
            template = _render_learner_prompt(_LEARNER_OPENING_PROMPT_RAW, loc_name)
            prompt = template.format(
                cefr_level=seed.cefr_level,
                topic=seed.topic,
                setting=seed.setting,
                user_role_name=seed.user_role.name,
                user_role_description=seed.user_role.description,
                model_role_name=seed.model_role.name,
                model_role_description=seed.model_role.description,
            )
        elif probe_axis is not None:
            template = _render_learner_prompt(_LEARNER_PROBE_PROMPT_RAW, loc_name)
            prompt = template.format(
                cefr_level=seed.cefr_level,
                transcript=_format_transcript_for_learner(dialogue),
                probe_axis=probe_axis,
            )
        else:
            template = _render_learner_prompt(_LEARNER_FOLLOWUP_PROMPT_RAW, loc_name)
            prompt = template.format(
                cefr_level=seed.cefr_level,
                topic=seed.topic,
                transcript=_format_transcript_for_learner(dialogue),
            )
        raw = await self.learner.generate(
            system=prompt,
            messages=[],
            cacheable_prefix=None,
            max_tokens=self.config.learner_max_tokens,
            temperature=self.config.temperature_learner,
        )
        return raw.strip().strip('"').strip("'")

    async def _simulate_conversation(
        self, seed: ScenarioSeed, probe_axis: str
    ) -> tuple[list[dict[str, str]], str]:
        """Drive an 8-turn conversation; return (messages, probe_user_text)."""
        # Per-seed locale: tell the target model "you tutor a Chinese / Japanese
        # / Italian learner" so the model picks the right country grounding.
        loc_name = getattr(seed, "locale", None)
        if loc_name:
            from qwen_tutor.generation.prompts import (
                render_deployment_system_prompt,
            )

            system_prompt = render_deployment_system_prompt(
                seed.cefr_level, locale_name=loc_name
            )
        else:
            system_prompt = self.config.deployment_system_prompt_template.format(
                cefr_level=seed.cefr_level
            )
        dialogue: list[dict[str, str]] = []
        probe_user_text = ""
        for user_turn_idx in range(self.config.n_user_turns):
            is_probe = user_turn_idx == self.config.probe_user_turn_idx
            try:
                user_text = await self._generate_learner_turn(
                    seed=seed,
                    dialogue=dialogue,
                    probe_axis=probe_axis if is_probe else None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "learner generation failed for %s at turn %d: %s",
                    seed.topic, user_turn_idx, exc,
                )
                break
            if is_probe:
                probe_user_text = user_text
            dialogue.append({"role": "user", "content": user_text})
            try:
                assistant_text = await self.target.generate(
                    system=system_prompt,
                    messages=dialogue,
                    mode="conversation",
                    max_new_tokens=self.config.max_new_tokens_conversation,
                    temperature=self.config.temperature_conversation,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "target generation failed for %s at turn %d: %s",
                    seed.topic, user_turn_idx, exc,
                )
                break
            dialogue.append({"role": "assistant", "content": assistant_text.strip()})
        return dialogue, probe_user_text

    async def _run_evaluation_mode(
        self, dialogue: list[dict[str, str]], seed: ScenarioSeed
    ) -> str:
        transcript = _format_transcript_for_learner(dialogue)
        eval_user_message = (
            f"Target CEFR level: {seed.cefr_level}\n\n"
            f"Transcript:\n{transcript}"
        )
        return await self.target.generate(
            system=self.config.evaluation_system_prompt,
            messages=[{"role": "user", "content": eval_user_message}],
            mode="evaluation",
            max_new_tokens=self.config.max_new_tokens_evaluation,
            temperature=self.config.temperature_evaluation,
        )

    # ----- metric computation --------------------------------------------

    async def _compute_metrics(
        self,
        seed: ScenarioSeed,
        dialogue: list[dict[str, str]],
        evaluation_generation: str,
    ) -> dict[str, Any]:
        if not dialogue:
            return {"empty_dialogue": True}

        # Sync metrics — pass the seed's locale so per-example scoring
        # checks the right country's avoided_topics + judge framing.
        seed_locale = getattr(seed, "locale", None)
        lf = level_fidelity(dialogue, seed.cefr_level)
        lc = locale_fidelity(dialogue, use_spacy=True)
        rs = redirect_success(
            dialogue,
            declared_subtopics=list(seed.subtopics),
            probe_user_turn_idx=self.config.probe_user_turn_idx,
            locale_name=seed_locale,
        )
        # The conversation-mode generations are inside dialogue's assistant
        # turns; verify none of them contains <think>.
        conv_mode_ok = all(
            mode_consistency(m["content"], "conversation")
            for m in dialogue
            if m["role"] == "assistant"
        )
        eval_mode_ok = mode_consistency(evaluation_generation, "evaluation")
        eval_valid = eval_json_validity(evaluation_generation)
        eval_scores = eval_dimension_scores(evaluation_generation) if eval_valid else None

        # LLM-judge metrics (skip if no judge configured)
        topic_score: float | None = None
        naturalness: int | None = None
        if self.judge is not None:
            topic_score = await topic_adherence(
                dialogue,
                declared_subtopics=list(seed.subtopics),
                judge=self.judge,
            )
            naturalness = await naturalness_judge(
                dialogue,
                judge=self.judge,
                target_cefr=seed.cefr_level,
                locale_name=seed_locale,
            )

        return {
            "topic_adherence": topic_score,
            "level_fidelity": lf,
            "locale_fidelity": lc,
            "redirect_success": rs,
            "naturalness_judge": naturalness,
            "mode_consistency_conversation": conv_mode_ok,
            "mode_consistency_evaluation": eval_mode_ok,
            "eval_json_validity": eval_valid,
            "eval_dimension_scores": eval_scores,
        }

    # ----- per-example pipeline -----------------------------------------

    async def _run_one(
        self, seed_id: str, seed: ScenarioSeed, probe_axis: str
    ) -> PerExampleRecord:
        dialogue, probe_user_text = await self._simulate_conversation(seed, probe_axis)
        if dialogue:
            eval_gen = await self._run_evaluation_mode(dialogue, seed)
        else:
            eval_gen = ""
        metrics = await self._compute_metrics(seed, dialogue, eval_gen)
        return PerExampleRecord(
            seed_id=seed_id,
            cefr_level=seed.cefr_level,
            topic=seed.topic,
            subtopics=list(seed.subtopics),
            probe_axis=probe_axis,
            probe_user_text=probe_user_text,
            dialogue=dialogue,
            evaluation_generation=eval_gen,
            metrics=metrics,
        )

    # ----- holdout driver ----------------------------------------------

    async def run_holdout(
        self,
        holdout_dir: str | Path,
        output_dir: str | Path,
        cefr_levels: list[str] | None = None,
        n_per_level: int | None = None,
        run_id: str | None = None,
        concurrency: int = 4,
    ) -> dict[str, Any]:
        cefr_levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
        scenarios = load_holdout(holdout_dir, cefr_levels)
        if n_per_level is not None:
            buckets: dict[str, list[tuple[str, ScenarioSeed]]] = {}
            for sid, seed in scenarios:
                buckets.setdefault(seed.cefr_level, []).append((sid, seed))
            scenarios = []
            for lvl in cefr_levels:
                scenarios.extend(buckets.get(lvl, [])[:n_per_level])
        if not scenarios:
            raise RuntimeError(
                f"no holdout scenarios found under {holdout_dir} for levels {cefr_levels}"
            )

        run_id = run_id or time.strftime("eval-%Y%m%d-%H%M%S")
        out_root = Path(output_dir) / run_id
        out_root.mkdir(parents=True, exist_ok=True)
        logger.info("evaluation run id: %s → %s", run_id, out_root)

        axis_cycle = cycle(self.config.probe_axes)

        async def _do(args: tuple[str, ScenarioSeed, str]) -> PerExampleRecord:
            sid, seed, axis = args
            return await self._run_one(sid, seed, axis)

        tasks = [(sid, seed, next(axis_cycle)) for sid, seed in scenarios]
        records = await gather_with_concurrency(
            [_do(t) for t in tasks],
            concurrency=concurrency,
            desc="eval",
        )

        # ----- write outputs --------------------------------------------
        per_example_path = out_root / "per_example.jsonl"
        with per_example_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        summary = _aggregate(records, run_id)
        with (out_root / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        with (out_root / "samples.md").open("w", encoding="utf-8") as fh:
            fh.write(_render_samples_md(records, summary))
        logger.info("eval complete: %s", out_root)
        return summary


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float | None:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return statistics.fmean(cleaned)


def _safe_rate(values: list[bool]) -> float | None:
    cleaned = [bool(v) for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _aggregate(records: list[PerExampleRecord], run_id: str) -> dict[str, Any]:
    by_level: dict[str, list[PerExampleRecord]] = {}
    for rec in records:
        by_level.setdefault(rec.cefr_level, []).append(rec)
    per_level: dict[str, Any] = {}
    for lvl in sorted(by_level):
        per_level[lvl] = _aggregate_bucket(by_level[lvl])
    overall = _aggregate_bucket(records)
    return {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_examples": len(records),
        "levels": sorted(by_level.keys()),
        "per_level": per_level,
        "overall": overall,
    }


def _aggregate_bucket(records: list[PerExampleRecord]) -> dict[str, Any]:
    if not records:
        return {"n_examples": 0}
    metrics = [r.metrics for r in records]
    lf_keys = ("above_band_ratio", "contraction_rate", "discourse_marker_rate", "mean_sentence_length")
    eval_keys = ("fluency", "accuracy", "vocabulary", "interaction")
    out: dict[str, Any] = {
        "n_examples": len(records),
        "topic_adherence_mean": _safe_mean([m.get("topic_adherence") for m in metrics]),
        "level_fidelity": {
            f"{k}_mean": _safe_mean(
                [m.get("level_fidelity", {}).get(k) for m in metrics]
            )
            for k in lf_keys
        },
        "locale_fidelity_mean": _safe_mean(
            [m.get("locale_fidelity") for m in metrics]
        ),
        "redirect_success_rate": _safe_rate(
            [m.get("redirect_success") for m in metrics]
        ),
        "naturalness_judge_mean": _safe_mean(
            [m.get("naturalness_judge") for m in metrics]
        ),
        "mode_consistency_conversation_rate": _safe_rate(
            [m.get("mode_consistency_conversation") for m in metrics]
        ),
        "mode_consistency_evaluation_rate": _safe_rate(
            [m.get("mode_consistency_evaluation") for m in metrics]
        ),
        "eval_json_valid_rate": _safe_rate(
            [m.get("eval_json_validity") for m in metrics]
        ),
        "eval_score_means": {
            k: _safe_mean(
                [
                    (m.get("eval_dimension_scores") or {}).get(k)
                    for m in metrics
                    if m.get("eval_dimension_scores") is not None
                ]
            )
            for k in eval_keys
        },
    }
    return out


# ---------------------------------------------------------------------------
# Samples markdown
# ---------------------------------------------------------------------------


def _render_samples_md(records: list[PerExampleRecord], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Evaluation samples — {summary['run_id']}\n")
    lines.append(f"_n={summary['n_examples']} examples, generated at {summary['timestamp']}_\n")
    lines.append("\n## Per-level summary\n")
    for lvl, agg in summary.get("per_level", {}).items():
        lines.append(f"### {lvl}  (n={agg['n_examples']})")
        for key in (
            "topic_adherence_mean", "locale_fidelity_mean", "redirect_success_rate",
            "naturalness_judge_mean", "mode_consistency_conversation_rate",
            "mode_consistency_evaluation_rate", "eval_json_valid_rate",
        ):
            v = agg.get(key)
            if v is None:
                continue
            lines.append(f"- **{key}**: {v:.3f}")
        lines.append("")

    lines.append("\n## Sample dialogues\n")
    # Show up to 2 per level for compactness.
    seen_per_level: dict[str, int] = {}
    for rec in records:
        if seen_per_level.get(rec.cefr_level, 0) >= 2:
            continue
        seen_per_level[rec.cefr_level] = seen_per_level.get(rec.cefr_level, 0) + 1
        lines.append(f"### {rec.cefr_level} — {rec.topic} (`{rec.seed_id}`)")
        lines.append(f"_probe_axis: {rec.probe_axis}_  ")
        lines.append(f"_redirect_success: {rec.metrics.get('redirect_success')}_  ")
        lines.append(f"_locale_fidelity: {rec.metrics.get('locale_fidelity')}_\n")
        for m in rec.dialogue:
            role = "**User**" if m["role"] == "user" else "**Tutor**"
            lines.append(f"{role}: {m['content']}\n")
        lines.append("\n#### Evaluation output\n")
        lines.append("```")
        lines.append(rec.evaluation_generation.strip()[:2000])
        lines.append("```\n")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def load_eval_config_from_training_yaml(
    training_yaml: str | Path = "config/training.yaml",
) -> EvaluationConfig:
    """Build EvaluationConfig from the canonical prompts in ``prompts.py``.

    ``training_yaml`` 인자는 더 이상 system 프롬프트의 출처가 아니지만,
    호출 시그니처 호환성을 위해 남겨 둡니다. system 프롬프트는
    ``qwen_tutor.generation.prompts`` 에서 (그리고 그 안에서
    ``config/locale.yaml``) 한 곳에서만 옵니다.
    """
    del training_yaml  # unused
    from qwen_tutor.generation.prompts import (
        DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE,
        EVALUATION_SYSTEM_PROMPT,
    )

    return EvaluationConfig(
        deployment_system_prompt_template=DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE,
        evaluation_system_prompt=EVALUATION_SYSTEM_PROMPT,
    )


def build_runner(
    *,
    training_yaml: str | Path = "config/training.yaml",
    generation_yaml: str | Path = "config/generation.yaml",
    base_model_id: str | None = None,
    adapter_path: str | Path | None = None,
    use_target: bool = True,
    target: TargetModelClient | None = None,
) -> EvaluationRunner:
    """Construct an EvaluationRunner from config paths.

    If ``target`` is provided it is used directly. Otherwise (and when
    ``use_target=True``) the runner loads the Qwen3 base + optional
    adapter via ``HFTargetModelClient.from_pretrained``.
    """
    cfg = load_eval_config_from_training_yaml(training_yaml)
    learner = build_teacher_from_config(generation_yaml, role="teacher")
    judge = build_teacher_from_config(generation_yaml, role="judge")
    if target is None and use_target:
        if base_model_id is None:
            with Path(training_yaml).open("r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh) or {}
            base_model_id = doc["base_model"]["model_id"]
        target = HFTargetModelClient.from_pretrained(
            base_model_id=base_model_id,
            adapter_path=adapter_path,
        )
    if target is None:
        raise ValueError("no target model provided and use_target=False")
    return EvaluationRunner(target=target, learner=learner, judge=judge, config=cfg)
