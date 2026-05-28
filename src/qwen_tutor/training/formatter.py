"""Chat formatting for joint conversation + evaluation training.

A SINGLE Qwen3-8B model must learn two modes:

  * conversation mode (/no_think) — used at inference for tutoring turns.
    Trained on ``SFTExample`` records whose ``scenario_type`` is
    ``"normal"``, ``"redirect"``, or ``"unleveled_natural"``.
  * evaluation mode  (/think) — used at inference for CEFR scoring.
    Trained on ``EvaluationExample`` records whose assistant turn already
    contains a ``<think>...</think>`` block followed by an
    ``EvaluationOutput`` JSON object.

The formatter:

  * ALWAYS uses the deployment system prompt template loaded from
    ``training.yaml`` (the ``system_prompt`` field stored on each
    example is intentionally ignored — see "Critical configuration" in
    the task spec).
  * Prepends ``/no_think`` to the first user message in conversation
    examples, and passes ``enable_thinking=False`` to the chat template.
  * For evaluation examples, passes ``enable_thinking=True`` but does
    NOT inject any extra ``<think>`` tag — the assistant content
    already contains one.
  * Returns formatted text plus an assistant-only loss mask
    (``labels`` with ``-100`` on every non-assistant token).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from qwen_tutor.schemas import EvaluationExample, Message, SFTExample

logger = logging.getLogger(__name__)

NO_THINK_TAG = "/no_think"
THINK_TAG = "/think"
LOSS_IGNORE_INDEX = -100

CONVERSATION_SCENARIO_TYPES = ("normal", "redirect", "unleveled_natural")


def _flatten_token_ids(result: Any) -> list[int]:
    """Normalize the many shapes ``apply_chat_template(tokenize=True)`` can
    return into a flat ``list[int]``.

    Observed shapes across tokenizer versions / models:
      * flat ``list[int]``                       (Qwen3-8B Qwen2Tokenizer fast)
      * ``BatchEncoding`` with ``input_ids``     (some HF versions)
      * single ``Encoding`` with ``.ids``        (raw tokenizers backend)
      * list of ``Encoding`` (batched)           (Qwen3-VL Qwen2Tokenizer fast)
      * 2-D nested list ``[[id, id, ...]]``      (batched output)
    """
    # Direct flat list[int] is the easy case.
    if isinstance(result, list) and result and isinstance(result[0], int):
        return list(result)
    # Single Encoding object exposing .ids.
    if hasattr(result, "ids") and not isinstance(result, (list, dict)):
        return list(result.ids)
    # BatchEncoding (dict-like) — try input_ids key.
    if hasattr(result, "keys") and "input_ids" in result:
        ids = result["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return list(ids[0])
        if isinstance(ids, list) and ids and isinstance(ids[0], int):
            return list(ids)
        if hasattr(ids, "tolist"):
            tl = ids.tolist()
            if isinstance(tl, list) and tl and isinstance(tl[0], list):
                return list(tl[0])
            return list(tl)
    # List wrapping Encoding objects (batched tokenizers backend).
    if isinstance(result, list) and result and hasattr(result[0], "ids"):
        return list(result[0].ids)
    # 2-D nested list.
    if isinstance(result, list) and result and isinstance(result[0], list):
        return list(result[0])
    raise TypeError(
        f"apply_chat_template returned an unsupported shape: "
        f"{type(result).__name__} (sample: {repr(result)[:200]})"
    )

# Special-token names we expect Qwen3's tokenizer to define. The
# formatter looks these up to validate that the chat template wired up
# correctly; missing tokens raise.
EXPECTED_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FormattedChat:
    """Output of the (non-tokenizing) formatter pass.

    Holds the chat-messages list ready to be fed into
    ``tokenizer.apply_chat_template`` along with the ``enable_thinking``
    flag the template should be called with.
    """

    messages: list[dict[str, str]]
    enable_thinking: bool
    cefr_level: str
    example_type: str  # "sft" | "evaluation"
    source_example_id: str


@dataclass
class TokenizedExample:
    """Output of the tokenizing pass.

    The ``labels`` list is the same length as ``input_ids``: positions
    that should NOT contribute to the loss carry ``LOSS_IGNORE_INDEX``
    (-100); positions inside an assistant turn carry their token id.
    """

    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    text: str
    chat: FormattedChat = field(repr=False)


# ---------------------------------------------------------------------------
# ChatFormatter
# ---------------------------------------------------------------------------


class ChatFormatter:
    def __init__(
        self,
        tokenizer: Any | None,
        deployment_system_prompt_template: str | None = None,
        evaluation_system_prompt: str | None = None,
        max_seq_length: int = 4096,
    ) -> None:
        # Single source of truth: when callers don't supply templates,
        # pull them from ``qwen_tutor.generation.prompts``. That module
        # already runs them through ``LOCALE.localize()`` which substitutes
        # {country}, {country_adjective}, {learner_description},
        # {avoided_topics_sentence}, and {avoid_cultures_phrase} from
        # config/locale.yaml. Editing locale.yaml is then the only knob
        # that affects what the trained model hears.
        if deployment_system_prompt_template is None:
            from qwen_tutor.generation.prompts import (
                DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE,
            )
            deployment_system_prompt_template = DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE
        if evaluation_system_prompt is None:
            from qwen_tutor.generation.prompts import EVALUATION_SYSTEM_PROMPT

            evaluation_system_prompt = EVALUATION_SYSTEM_PROMPT
        if "{cefr_level}" not in deployment_system_prompt_template:
            raise ValueError(
                "deployment_system_prompt_template must contain a "
                "{cefr_level} placeholder"
            )
        self.tokenizer = tokenizer
        self.deployment_template = deployment_system_prompt_template
        self.evaluation_system_prompt = evaluation_system_prompt
        self.max_seq_length = max_seq_length
        if tokenizer is not None:
            self._validate_tokenizer_specials()

    # ----- tokenizer validation ------------------------------------------

    def _validate_tokenizer_specials(self) -> None:
        """Check that Qwen3's special tokens are present in the tokenizer."""
        tok = self.tokenizer
        vocab = getattr(tok, "get_vocab", lambda: {})()
        added = set()
        for src in (
            getattr(tok, "additional_special_tokens", None) or [],
            getattr(tok, "all_special_tokens", None) or [],
        ):
            for t in src:
                added.add(str(t))
        # Some tokenizers expose special tokens via vocab keys directly.
        missing: list[str] = []
        for needed in EXPECTED_SPECIAL_TOKENS:
            if needed in vocab or needed in added:
                continue
            # Final attempt: see if the tokenizer encodes it back to a
            # single id without splitting.
            try:
                ids = tok.encode(needed, add_special_tokens=False)
                if len(ids) == 1:
                    continue
            except Exception:  # noqa: BLE001
                pass
            missing.append(needed)
        if missing:
            raise ValueError(
                f"tokenizer is missing expected special tokens: {missing}. "
                "Confirm it was loaded from the Qwen3 series."
            )

    # ----- conversation / evaluation prep --------------------------------

    def _render_deployment_system_prompt(
        self, cefr_level: str, locale_name: str | None = None
    ) -> str:
        """Render the deployment system prompt for a given CEFR level + locale.

        When ``locale_name`` is None or matches the default locale, the
        pre-rendered ``self.deployment_template`` is used (fast path, no
        re-substitution). Otherwise we delegate to the prompts module which
        re-renders for the requested locale.
        """
        if locale_name is None:
            return self.deployment_template.format(cefr_level=cefr_level)
        from qwen_tutor.generation.prompts import render_deployment_system_prompt
        from qwen_tutor.locale import DEFAULT_LOCALE_NAME

        if locale_name == DEFAULT_LOCALE_NAME:
            return self.deployment_template.format(cefr_level=cefr_level)
        return render_deployment_system_prompt(cefr_level, locale_name=locale_name)

    def _render_evaluation_system_prompt(
        self, locale_name: str | None = None
    ) -> str:
        if locale_name is None:
            return self.evaluation_system_prompt
        from qwen_tutor.generation.prompts import render_evaluation_system_prompt
        from qwen_tutor.locale import DEFAULT_LOCALE_NAME

        if locale_name == DEFAULT_LOCALE_NAME:
            return self.evaluation_system_prompt
        return render_evaluation_system_prompt(locale_name=locale_name)

    @staticmethod
    def _ensure_no_think(content: str) -> str:
        stripped = content.lstrip()
        if stripped.startswith(NO_THINK_TAG) or stripped.startswith(THINK_TAG):
            return content
        return f"{NO_THINK_TAG}\n{content}" if content else NO_THINK_TAG

    def format_sft_example(self, example: SFTExample) -> FormattedChat:
        if example.metadata.scenario_type not in CONVERSATION_SCENARIO_TYPES:
            raise ValueError(
                f"SFTExample {example.id!r} has scenario_type "
                f"{example.metadata.scenario_type!r}; expected one of "
                f"{CONVERSATION_SCENARIO_TYPES}"
            )
        system_content = self._render_deployment_system_prompt(
            example.metadata.cefr_level, locale_name=example.metadata.locale
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        first_user_seen = False
        for m in example.messages:
            if m.role == "system":
                # We've already injected our authoritative system prompt.
                continue
            content = m.content
            if m.role == "user" and not first_user_seen:
                content = self._ensure_no_think(content)
                first_user_seen = True
            messages.append({"role": m.role, "content": content})
        if not first_user_seen:
            raise ValueError(
                f"SFTExample {example.id!r} has no user turn to anchor /no_think on"
            )
        return FormattedChat(
            messages=messages,
            enable_thinking=False,
            cefr_level=example.metadata.cefr_level,
            example_type="sft",
            source_example_id=example.id,
        )

    def format_evaluation_example(self, example: EvaluationExample) -> FormattedChat:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._render_evaluation_system_prompt(
                    locale_name=example.metadata.locale
                ),
            }
        ]
        had_user = False
        for m in example.messages:
            if m.role == "system":
                continue
            if m.role == "assistant" and "<think>" not in m.content:
                logger.warning(
                    "EvaluationExample %s has an assistant turn without <think> "
                    "block; the model will still learn from it but won't see a "
                    "thinking exemplar",
                    example.id,
                )
            messages.append({"role": m.role, "content": m.content})
            had_user = had_user or m.role == "user"
        if not had_user:
            raise ValueError(
                f"EvaluationExample {example.id!r} has no user-side transcript turn"
            )
        return FormattedChat(
            messages=messages,
            enable_thinking=True,
            cefr_level=example.metadata.learner_cefr_target,
            example_type="evaluation",
            source_example_id=example.id,
        )

    # ----- tokenization + loss mask --------------------------------------

    def _apply_chat_template(
        self, messages: Sequence[dict[str, str]], enable_thinking: bool
    ) -> list[int]:
        """Wrap ``tokenizer.apply_chat_template`` with our standard args.

        Returns a flat ``list[int]`` of token ids. Different tokenizers
        accept ``enable_thinking`` differently, so we try the kw first
        and fall back to omitting it.
        """
        if self.tokenizer is None:
            raise RuntimeError("ChatFormatter has no tokenizer; cannot tokenize")
        # Different tokenizers return wildly different shapes for
        # ``tokenize=True``: a flat ``list[int]``, a ``BatchEncoding`` dict,
        # or a list of ``Encoding`` objects. Normalize to a flat ``list[int]``.
        try:
            result = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older tokenizers without the enable_thinking kwarg.
            result = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=False,
            )
        return _flatten_token_ids(result)

    def _assistant_marker_ids(self) -> tuple[list[int], int]:
        """Returns the (assistant-start marker token ids, im_end token id).

        Marker = the token sequence the chat template emits to OPEN an
        assistant turn. For chatml-style templates this is
        ``<|im_start|>assistant\\n`` → ``[<|im_start|>, "assistant", "\\n"]``.
        We use it to find assistant spans inside the full tokenized stream
        without relying on prefix-stability of the chat template (Qwen3.5's
        template, for instance, is NOT prefix-stable).
        """
        if self.tokenizer is None:
            raise RuntimeError("ChatFormatter has no tokenizer")
        start_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if not start_ids or not end_ids or len(end_ids) != 1:
            raise RuntimeError(
                "tokenizer does not encode the chatml role markers cleanly: "
                f"start={start_ids}, end={end_ids}"
            )
        return start_ids, end_ids[0]

    def tokenize(self, chat: FormattedChat) -> TokenizedExample:
        if self.tokenizer is None:
            raise RuntimeError("ChatFormatter has no tokenizer; cannot tokenize")
        full_ids = self._apply_chat_template(chat.messages, chat.enable_thinking)

        # Find every assistant span via marker scan. This sidesteps the chat
        # template's prefix instability — we trust the FULL tokenization and
        # locate spans by their delimiter tokens.
        labels = [LOSS_IGNORE_INDEX] * len(full_ids)
        start_marker, end_marker = self._assistant_marker_ids()
        m = len(start_marker)
        i = 0
        spans: list[tuple[int, int]] = []
        while i <= len(full_ids) - m:
            if full_ids[i : i + m] == start_marker:
                # Find the matching <|im_end|>.
                j = i + m
                while j < len(full_ids) and full_ids[j] != end_marker:
                    j += 1
                if j == len(full_ids):
                    # No closing marker — assistant turn at the tail.
                    spans.append((i, len(full_ids)))
                    break
                # Include the closing <|im_end|> in the labeled span so the
                # model learns to emit the end-of-turn token.
                spans.append((i, j + 1))
                i = j + 1
                continue
            i += 1
        if not spans:
            raise RuntimeError(
                "no assistant spans found in tokenized chat; the chat template "
                "may not be emitting <|im_start|>assistant\\n markers as expected"
            )
        for s, e in spans:
            for k in range(s, e):
                labels[k] = full_ids[k]

        # Truncate to max_seq_length, preserving alignment.
        if len(full_ids) > self.max_seq_length:
            full_ids = full_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

        attention_mask = [1] * len(full_ids)
        text = self.tokenizer.decode(full_ids, skip_special_tokens=False)
        return TokenizedExample(
            input_ids=full_ids,
            labels=labels,
            attention_mask=attention_mask,
            text=text,
            chat=chat,
        )

    # ----- top-level convenience -----------------------------------------

    def format_for_training(
        self, example: SFTExample | EvaluationExample
    ) -> TokenizedExample:
        if isinstance(example, SFTExample):
            chat = self.format_sft_example(example)
        elif isinstance(example, EvaluationExample):
            chat = self.format_evaluation_example(example)
        else:
            raise TypeError(
                f"unsupported example type: {type(example).__name__}"
            )
        return self.tokenize(chat)
