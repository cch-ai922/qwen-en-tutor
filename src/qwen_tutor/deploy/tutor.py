"""TutorRuntime  -  inference wrapper around the trained Qwen3.5 LoRA adapter.

Loads the base model + LoRA adapter once, holds a multi-turn chat history,
applies the same deployment system prompt the formatter uses at training
time (so the model sees the exact distribution it was trained on), and
runs ``BannedTermsFilter`` over each generated assistant turn as a safety
net. On a filter hit it resamples once at higher temperature; if the
second draft also trips the filter, it returns a canned redirect.

Threading model: synchronous. The safety filter is async but is awaited
internally inside ``chat()`` via ``asyncio.run`` — callers see a plain
``str`` return. If you wire this into FastAPI / Discord / etc., expose
``chat_async()`` instead and let the host event loop run it.

Typical usage::

    from qwen_tutor.deploy.tutor import TutorRuntime

    tutor = TutorRuntime(
        base_model_path="vendor/models/Qwen_3.5_4B",
        adapter_path="outputs/dpo",     # or outputs/sft if you skipped DPO
        cefr_level="A2",
        locale="china",
    )
    print(tutor.chat("Hi! Can you help me practice ordering food?"))
    print(tutor.chat("I want to eat dumplings."))
    tutor.reset()                       # clear history between sessions
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duck-typed example so BannedTermsFilter can scan a single assistant turn
# without us building a full SFTExample (which would require topic/subtopics/
# roles/etc. we don't have at inference time).
# ---------------------------------------------------------------------------


@dataclass
class _Msg:
    role: str
    content: str


@dataclass
class _Meta:
    scenario_type: str = "normal"  # "normal" so the filter scans the message


@dataclass
class _MiniExample:
    messages: list[_Msg]
    metadata: _Meta = field(default_factory=_Meta)
    system_prompt: str = ""


# ---------------------------------------------------------------------------
# Safety fallback message used when the model trips the banned-terms filter
# twice in a row. Locale-agnostic, deliberately bland.
# ---------------------------------------------------------------------------

_CANNED_FALLBACK = (
    "Let's talk about something else. How about your hobbies, or food you "
    "like to cook? I am happy to practice English with you on those topics."
)

# Deploy-time banned-terms list, independent of the training-time file so
# you can tune safety strictness without regenerating training data.
DEFAULT_DEPLOY_BANNED_TERMS = Path("config/banned_terms_deploy.yaml")


# ---------------------------------------------------------------------------
# EvaluationResult — what `TutorRuntime.evaluate()` returns
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Parsed output of a /think CEFR evaluation pass.

    ``raw`` is always the unmodified text the model produced. ``reasoning``
    and ``scores`` are best-effort extractions: if the model emits
    well-formed ``<think>...</think>{json}`` they're populated; if not,
    one or both are ``None`` and the caller can inspect ``raw``.
    """

    raw: str
    reasoning: str | None
    scores: dict[str, Any] | None
    target_cefr: str
    parse_error: str | None = None


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# Trailing-comma fixups: ",}" and ",]" with optional whitespace/newline between.
_TRAILING_COMMA_OBJ_RE = re.compile(r",(\s*[}\]])")


def _strip_code_fences(text: str) -> str:
    """If the text contains a ```json ... ``` block, return its inner body.

    Small models sometimes wrap JSON in markdown code fences even when the
    system prompt forbids it. Strip the fence so json.loads works.
    """
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1)
    return text


def _parse_eval_output(raw: str) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Pull (<think>body</think>, {json}) out of the model's raw text.

    Tolerates several common malformations from small / undertrained models:
      * markdown ```json ... ``` code fences
      * trailing commas inside objects/arrays
      * Python-style single-quoted dicts (last-resort ``ast.literal_eval``)

    Returns (reasoning, scores, parse_error). Each may be None if not found.
    """
    reasoning: str | None = None
    m = _THINK_RE.search(raw)
    if m:
        reasoning = m.group(1).strip()
        after = raw[m.end():]
    else:
        after = raw

    after = _strip_code_fences(after)

    json_start = after.find("{")
    if json_start < 0:
        return reasoning, None, "no JSON object found after </think>"
    snippet = after[json_start:]
    last_brace = snippet.rfind("}")
    if last_brace < 0:
        return reasoning, None, "JSON object never closed"
    candidate = snippet[: last_brace + 1]

    # 1) Strict JSON.
    try:
        return reasoning, json.loads(candidate), None
    except json.JSONDecodeError as exc:
        first_err = exc.msg

    # 2) Strip trailing commas and retry.
    fixed = _TRAILING_COMMA_OBJ_RE.sub(r"\1", candidate)
    if fixed != candidate:
        try:
            return reasoning, json.loads(fixed), None
        except json.JSONDecodeError:
            pass

    # 3) Last resort: ast.literal_eval for Python-style single-quoted dicts.
    #    Only accepted if the result is a dict; ast rejects anything dangerous.
    try:
        import ast

        obj = ast.literal_eval(candidate)
        if isinstance(obj, dict):
            return reasoning, obj, None
    except (ValueError, SyntaxError):
        pass

    return reasoning, None, f"json decode failed: {first_err}"


class TutorRuntime:
    """Single-process tutor inference runtime."""

    def __init__(
        self,
        base_model_path: str | Path,
        adapter_path: str | Path,
        cefr_level: str = "A2",
        locale: str = "china",
        *,
        use_4bit: bool = True,
        device_map: str | dict[str, Any] = "auto",
        trust_remote_code: bool = True,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        enable_safety_filter: bool = True,
        banned_terms_path: str | Path | None = None,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, BitsAndBytesConfig

        from qwen_tutor.training.formatter import ChatFormatter
        from qwen_tutor.training.model_loader import load_base_model_for_training

        self.base_model_path = str(base_model_path)
        self.adapter_path = str(adapter_path)
        self.cefr_level = cefr_level
        self.locale = locale
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # -- tokenizer --
        logger.info("loading tokenizer from %s", self.base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -- base model (optional 4-bit) --
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        logger.info("loading base model from %s (4bit=%s)", self.base_model_path, use_4bit)
        base = load_base_model_for_training(
            self.base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )

        # -- LoRA adapter --
        logger.info("attaching LoRA adapter from %s", self.adapter_path)
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

        # -- formatter (for the deployment system prompt) --
        # max_seq_length here only caps tokenization in format_for_training;
        # we don't call that path at deploy. Pick a generous value.
        self.formatter = ChatFormatter(tokenizer=self.tokenizer, max_seq_length=8192)
        self.system_prompt = self.formatter._render_deployment_system_prompt(
            self.cefr_level, locale_name=self.locale
        )
        # /think evaluation system prompt — same one the formatter uses at
        # training time, so the LoRA sees its trained distribution.
        self.evaluation_system_prompt = self.formatter._render_evaluation_system_prompt(
            locale_name=self.locale
        )

        # -- safety filter --
        # Defaults to config/banned_terms_deploy.yaml (deploy-tuned strictness),
        # NOT the training file. Pass banned_terms_path to override.
        self.enable_safety_filter = enable_safety_filter
        self._safety_filter = None
        if enable_safety_filter:
            from qwen_tutor.generation.filters.banned_terms import BannedTermsFilter

            terms_path = Path(banned_terms_path) if banned_terms_path else DEFAULT_DEPLOY_BANNED_TERMS
            if not terms_path.exists():
                # Fall back to the training file with a warning so deploy
                # doesn't silently run unguarded if the deploy YAML is missing.
                from qwen_tutor.generation.filters.banned_terms import DEFAULT_BANNED_TERMS_PATH

                logger.warning(
                    "deploy banned-terms file %s not found; falling back to training file %s",
                    terms_path, DEFAULT_BANNED_TERMS_PATH,
                )
                terms_path = DEFAULT_BANNED_TERMS_PATH
            self._safety_filter = BannedTermsFilter(banned_terms_path=terms_path)
            logger.info("safety filter loaded from %s", terms_path)

        # -- chat state --
        self.history: list[_Msg] = []
        self._no_think_anchored = False

    # ----- public API -----------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history (start a new session)."""
        self.history.clear()
        self._no_think_anchored = False

    def chat(self, user_message: str) -> str:
        """Run a single turn: append user msg → generate → safety-check → return."""
        return asyncio.run(self.chat_async(user_message))

    async def chat_async(self, user_message: str) -> str:
        # 1) inject /no_think on the FIRST user turn of the session, exactly
        #    like the training formatter does. Subsequent turns are plain text.
        content = user_message
        if not self._no_think_anchored:
            content = self.formatter._ensure_no_think(content)
            self._no_think_anchored = True
        self.history.append(_Msg(role="user", content=content))

        # 2) first draft.
        draft = self._generate(self.temperature)

        # 3) safety check; resample once at higher temperature on hit.
        if self._safety_filter is not None:
            ok, reason = await self._check(draft)
            if not ok:
                logger.warning("safety filter hit on first draft: %s", reason)
                draft2 = self._generate(min(1.0, self.temperature + 0.2))
                ok2, reason2 = await self._check(draft2)
                if ok2:
                    draft = draft2
                else:
                    logger.warning("safety filter hit on resample: %s", reason2)
                    draft = _CANNED_FALLBACK

        self.history.append(_Msg(role="assistant", content=draft))
        return draft

    def evaluate(
        self,
        target_cefr: str | None = None,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> EvaluationResult:
        """Run a /think CEFR evaluation over the current conversation.

        Mirrors the EvaluationExample training format: the system prompt is
        the evaluation prompt, the user turn is a rendered transcript of the
        session (user + tutor turns labeled), and the model is invoked with
        ``enable_thinking=True`` so it emits ``<think>...</think>{json}``.

        ``target_cefr`` defaults to the level the runtime was constructed
        with. Does NOT mutate ``self.history`` — eval is a shadow call.
        """
        return asyncio.run(self.evaluate_async(
            target_cefr=target_cefr,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ))

    async def evaluate_async(
        self,
        target_cefr: str | None = None,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> EvaluationResult:
        if not self.history:
            raise RuntimeError(
                "evaluate() called with empty history — have a conversation first"
            )
        if not any(m.role == "user" for m in self.history):
            raise RuntimeError("no user turns in history to evaluate")

        target = target_cefr or self.cefr_level
        transcript = self._render_eval_transcript(target)
        chat = [
            {"role": "system", "content": self.evaluation_system_prompt},
            {"role": "user", "content": transcript},
        ]
        raw = self._generate_chat(
            chat,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            enable_thinking=True,
        )
        reasoning, scores, parse_error = _parse_eval_output(raw)
        return EvaluationResult(
            raw=raw,
            reasoning=reasoning,
            scores=scores,
            target_cefr=target,
            parse_error=parse_error,
        )

    # ----- internals ------------------------------------------------------

    def _render_eval_transcript(self, target_cefr: str) -> str:
        """Build the transcript-style user turn the model was trained on.

        Format mirrors ``qwen_tutor.generation.eval_gen._render_transcript`` so
        the LoRA's eval mode receives the exact same text shape. ``/no_think``
        markers from training-time anchoring are stripped — they're a
        conversation-mode artifact and irrelevant to evaluation.
        """
        lines: list[str] = [f"Target CEFR level: {target_cefr}", "", "Transcript:"]
        user_idx = 0
        for m in self.history:
            content = m.content
            # Strip the /no_think anchor we injected on turn 0 — it's a
            # conversation-mode directive that shouldn't appear in the
            # transcript the examiner reads.
            if content.lstrip().startswith("/no_think"):
                content = content.lstrip()[len("/no_think"):].lstrip()
            if m.role == "user":
                lines.append(f"[USER turn {user_idx}] {content}")
                user_idx += 1
            elif m.role == "assistant":
                lines.append(f"[TUTOR] {content}")
            else:
                lines.append(f"[{m.role.upper()}] {content}")
        return "\n".join(lines)

    def _build_chat(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            *[{"role": m.role, "content": m.content} for m in self.history],
        ]

    def _generate(self, temperature: float) -> str:
        """Conversation-mode generate (enable_thinking=False, current history)."""
        return self._generate_chat(
            self._build_chat(),
            temperature=temperature,
            max_new_tokens=self.max_new_tokens,
            enable_thinking=False,
        )

    def _generate_chat(
        self,
        chat: list[dict[str, str]],
        *,
        temperature: float,
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> str:
        """Apply chat template + generate. Shared by conversation and eval."""
        import torch

        try:
            input_ids = self.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_tensors="pt",
            )
        except TypeError:
            input_ids = self.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        # apply_chat_template(return_tensors="pt") can return either a plain
        # tensor or a BatchEncoding (dict-like wrapper). BatchEncoding is NOT
        # an isinstance(dict), so unwrap via attribute / item access.
        if hasattr(input_ids, "input_ids"):
            attention_mask = getattr(input_ids, "attention_mask", None)
            input_ids = input_ids.input_ids
        elif isinstance(input_ids, dict) and "input_ids" in input_ids:
            attention_mask = input_ids.get("attention_mask")
            input_ids = input_ids["input_ids"]
        else:
            attention_mask = None
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        else:
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out[0, input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    async def _check(self, assistant_text: str) -> tuple[bool, str | None]:
        """Run BannedTermsFilter on JUST the latest assistant turn."""
        if self._safety_filter is None:
            return True, None
        ex = _MiniExample(messages=[_Msg(role="assistant", content=assistant_text)])
        result = await self._safety_filter.check(ex)  # type: ignore[arg-type]
        return result.passed, result.reason
