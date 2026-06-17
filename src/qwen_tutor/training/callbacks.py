"""Eval-time generation + filter-based metrics for SFT and DPO training.

Hooks into ``transformers.TrainerCallback.on_evaluate``. At each eval
step the callback:

  1. Loads a fixed pool of held-out prompts (one half flagged for
     conversation mode, one half for evaluation mode).
  2. Generates 10 conversation samples (/no_think) and 10 evaluation
     samples (/think).
  3. On conversation samples, runs the mechanical filters
     (banned_terms, mode_consistency, naturalness) and reports
     per-filter pass rates to wandb.
  4. On evaluation samples, parses the JSON output and reports the
     valid-JSON rate and mean per-dimension scores.
  5. Logs a small ``wandb.Table`` of sample generations for human
     spot-check.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_tutor.generation.filters.banned_terms import BannedTermsFilter
from qwen_tutor.generation.filters.base import Filter, FilterResult
from qwen_tutor.generation.filters.mode_consistency import ModeConsistencyFilter
from qwen_tutor.generation.filters.naturalness import NaturalnessFilter
from qwen_tutor.schemas import (
    EvaluationExample,
    EvaluationMetadata,
    ExampleMetadata,
    Message,
    ModelRole,
    SFTExample,
    UserRole,
)
from qwen_tutor.utils.runner import extract_first_json

logger = logging.getLogger(__name__)

_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Held-out prompt schema
# ---------------------------------------------------------------------------


@dataclass
class HeldoutPrompt:
    """One held-out generation prompt the callback drives the model with."""

    id: str
    mode: str  # "conversation" or "evaluation"
    cefr_level: str
    user_message: str  # the user-turn content to seed generation
    # For evaluation prompts we sometimes carry the transcript-as-user-text
    # already; for conversation prompts it's just an opening line.


def _load_heldout_prompts(path: str | Path | None) -> list[HeldoutPrompt]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        logger.warning("held-out prompts file %s not found; callback will be a no-op", p)
        return []
    out: list[HeldoutPrompt] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                HeldoutPrompt(
                    id=str(obj["id"]),
                    mode=obj["mode"],
                    cefr_level=obj.get("cefr_level", "B1"),
                    user_message=obj["user_message"],
                )
            )
    return out


# ---------------------------------------------------------------------------
# Helpers to bridge filters and generated text
# ---------------------------------------------------------------------------


def _run_filter_sync(flt: Filter, example) -> FilterResult:
    """Synchronously run a single async ``Filter.check`` from a callback."""
    try:
        return asyncio.run(flt.check(example))
    except Exception as exc:  # noqa: BLE001
        return FilterResult(passed=False, reason=f"filter raised {type(exc).__name__}: {exc}")


def _build_conversation_sft_example(prompt: HeldoutPrompt, assistant_text: str) -> SFTExample:
    """Wrap a single generated turn in an SFTExample so filters can scan it."""
    return SFTExample(
        id=f"callback_conv_{prompt.id}",
        metadata=ExampleMetadata(
            topic="callback sample",
            subtopics=["sample", "callback", "evaluation"],
            user_role=UserRole(name="Learner", description="held-out learner"),
            model_role=ModelRole(name="Tutor", description="qwen-en-tutor"),
            cefr_level=prompt.cefr_level,
            scenario_type="normal",
        ),
        system_prompt="callback synthetic",
        messages=[
            Message(role="user", content=prompt.user_message),
            Message(role="assistant", content=assistant_text),
        ],
    )


def _build_evaluation_example_for_filter(prompt: HeldoutPrompt, assistant_text: str) -> EvaluationExample:
    return EvaluationExample(
        id=f"callback_eval_{prompt.id}",
        metadata=EvaluationMetadata(
            source_dialogue_id=prompt.id,
            learner_cefr_target=prompt.cefr_level,
        ),
        system_prompt="callback synthetic",
        messages=[
            Message(role="user", content=prompt.user_message),
            Message(role="assistant", content=assistant_text),
        ],
    )


def _parse_evaluation_output(assistant_text: str) -> dict[str, Any] | None:
    close = _THINK_CLOSE_RE.search(assistant_text)
    if not close:
        return None
    after = assistant_text[close.end() :].strip()
    if not after:
        return None
    try:
        parsed = extract_first_json(after)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover — transformers unavailable in unit tests
    TrainerCallback = object  # type: ignore[assignment,misc]


class TutorEvalCallback(TrainerCallback):  # type: ignore[misc]
    """Periodically generate held-out samples and report quality to wandb."""

    def __init__(
        self,
        *,
        formatter: Any,  # qwen_tutor.training.formatter.ChatFormatter
        prompts_path: str | Path | None,
        n_conversation: int = 10,
        n_evaluation: int = 10,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        log_samples_table: bool = True,
        conversation_filters: list[str] | None = None,
    ) -> None:
        self.formatter = formatter
        self.prompts = _load_heldout_prompts(prompts_path)
        self.n_conversation = n_conversation
        self.n_evaluation = n_evaluation
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.log_samples_table = log_samples_table
        self._conversation_filter_names = set(
            conversation_filters
            or ["banned_terms", "mode_consistency", "naturalness"]
        )
        # Lazy-built filter instances — they take a moment to load
        # (banned_terms parses YAML, naturalness compiles regexes).
        self._filters: list[Filter] | None = None

    # ----- filter assembly -----------------------------------------------

    def _ensure_filters(self) -> list[Filter]:
        if self._filters is not None:
            return self._filters
        registry: dict[str, Filter] = {}
        if "banned_terms" in self._conversation_filter_names:
            registry["banned_terms"] = BannedTermsFilter()
        if "mode_consistency" in self._conversation_filter_names:
            registry["mode_consistency"] = ModeConsistencyFilter()
        if "naturalness" in self._conversation_filter_names:
            registry["naturalness"] = NaturalnessFilter()
        self._filters = list(registry.values())
        return self._filters

    # ----- generation -----------------------------------------------------

    def _build_prompt_text(self, prompt: HeldoutPrompt) -> str:
        """Render a generation-ready prompt string for the model.

        Uses the formatter's chat template with ``add_generation_prompt=True``
        so the model sees ``<|im_start|>assistant\\n`` and is asked to complete.
        """
        tok = self.formatter.tokenizer
        if prompt.mode == "conversation":
            # Use the scenario-aware deployment prompt with a neutral default
            # scenario — same template the SFT/DPO formatter feeds the model
            # at training time, so callback samples stay inside the trained
            # distribution. (Held-out prompts carry no per-scenario fields.)
            from qwen_tutor.generation.prompts import (
                render_default_scenario_deployment_system_prompt,
            )

            system_content = render_default_scenario_deployment_system_prompt(
                cefr_level=prompt.cefr_level,
            )
            user_content = self.formatter._ensure_no_think(prompt.user_message)
            enable_thinking = False
        else:
            # Honor ``thinking.student_eval`` so the in-training eval
            # callback probe matches what the trained model will actually
            # emit at deploy. See formatter.py for the matching SFT-time
            # treatment that strips <think> blocks when no_think is set.
            from qwen_tutor.utils.thinking import get_thinking_mode

            system_content = self.formatter.evaluation_system_prompt
            user_content = prompt.user_message
            enable_thinking = get_thinking_mode("student_eval") != "no_think"
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _generate(self, model, tokenizer, prompt_text: str) -> str:
        import torch

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        # strip the prompt portion
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ----- metric computation --------------------------------------------

    def _score_conversation_samples(
        self, samples: list[tuple[HeldoutPrompt, str]]
    ) -> dict[str, Any]:
        filters = self._ensure_filters()
        per_filter_passes: dict[str, list[bool]] = {f.name: [] for f in filters}
        rows: list[dict[str, Any]] = []
        for prompt, text in samples:
            ex = _build_conversation_sft_example(prompt, text)
            row: dict[str, Any] = {
                "prompt_id": prompt.id,
                "cefr_level": prompt.cefr_level,
                "assistant_text": text[:500],
            }
            for flt in filters:
                result = _run_filter_sync(flt, ex)
                per_filter_passes[flt.name].append(result.passed)
                row[flt.name] = "PASS" if result.passed else f"FAIL ({result.reason})"
            rows.append(row)
        metrics = {
            f"callback/conversation/{name}_pass_rate": (
                sum(passes) / len(passes) if passes else 0.0
            )
            for name, passes in per_filter_passes.items()
        }
        metrics["callback/conversation/n_samples"] = len(samples)
        return {"metrics": metrics, "rows": rows}

    def _score_evaluation_samples(
        self, samples: list[tuple[HeldoutPrompt, str]]
    ) -> dict[str, Any]:
        valid = 0
        rows: list[dict[str, Any]] = []
        score_sums = {
            "fluency": 0, "accuracy": 0, "vocabulary": 0,
            "interaction": 0, "topic_adherence": 0,
        }
        score_counts = 0
        for prompt, text in samples:
            parsed = _parse_evaluation_output(text)
            row: dict[str, Any] = {
                "prompt_id": prompt.id,
                "cefr_level": prompt.cefr_level,
                "raw_text": text[:500],
                "json_valid": parsed is not None,
            }
            if parsed is not None:
                valid += 1
                scores = parsed.get("scores") or {}
                if all(k in scores for k in score_sums):
                    score_counts += 1
                    for k in score_sums:
                        try:
                            score_sums[k] += int(scores[k])
                        except (TypeError, ValueError):
                            score_counts -= 1
                            break
                row["overall_cefr_estimate"] = parsed.get("overall_cefr_estimate")
                row["scores"] = scores
            rows.append(row)
        metrics: dict[str, Any] = {
            "callback/evaluation/json_valid_rate": (valid / len(samples)) if samples else 0.0,
            "callback/evaluation/n_samples": len(samples),
        }
        if score_counts > 0:
            for k, total in score_sums.items():
                metrics[f"callback/evaluation/mean_{k}"] = total / score_counts
        return {"metrics": metrics, "rows": rows}

    # ----- callback entry points -----------------------------------------

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: D401
        if not self.prompts:
            return control
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
        if model is None or tokenizer is None:
            logger.warning("callback: model or tokenizer missing from on_evaluate kwargs")
            return control

        conv_prompts = [p for p in self.prompts if p.mode == "conversation"][: self.n_conversation]
        eval_prompts = [p for p in self.prompts if p.mode == "evaluation"][: self.n_evaluation]

        conv_samples: list[tuple[HeldoutPrompt, str]] = []
        for p in conv_prompts:
            try:
                gen = self._generate(model, tokenizer, self._build_prompt_text(p))
            except Exception as exc:  # noqa: BLE001
                logger.warning("callback conv generation failed for %s: %s", p.id, exc)
                gen = ""
            conv_samples.append((p, gen))

        eval_samples: list[tuple[HeldoutPrompt, str]] = []
        for p in eval_prompts:
            try:
                gen = self._generate(model, tokenizer, self._build_prompt_text(p))
            except Exception as exc:  # noqa: BLE001
                logger.warning("callback eval generation failed for %s: %s", p.id, exc)
                gen = ""
            eval_samples.append((p, gen))

        conv_out = self._score_conversation_samples(conv_samples)
        eval_out = self._score_evaluation_samples(eval_samples)
        all_metrics = {**conv_out["metrics"], **eval_out["metrics"]}

        # ----- wandb logging --------------------------------------------
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({**all_metrics, "step": state.global_step})
                if self.log_samples_table:
                    conv_cols = (
                        ["prompt_id", "cefr_level", "assistant_text"]
                        + [f.name for f in self._ensure_filters()]
                    )
                    conv_table = wandb.Table(
                        columns=conv_cols,
                        data=[[r.get(c, "") for c in conv_cols] for r in conv_out["rows"]],
                    )
                    eval_cols = [
                        "prompt_id", "cefr_level", "json_valid",
                        "overall_cefr_estimate", "raw_text",
                    ]
                    eval_table = wandb.Table(
                        columns=eval_cols,
                        data=[[r.get(c, "") for c in eval_cols] for r in eval_out["rows"]],
                    )
                    wandb.log(
                        {
                            "callback/conversation_samples": conv_table,
                            "callback/evaluation_samples": eval_table,
                            "step": state.global_step,
                        }
                    )
        except ImportError:  # pragma: no cover
            logger.warning("callback: wandb not installed; metrics only printed to log")

        logger.info("callback metrics @ step %d: %s", state.global_step, all_metrics)
        return control
