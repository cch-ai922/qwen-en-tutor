"""Prompt-style dispatcher.

Generation modules import the five template constants from this module
instead of from ``prompts`` directly. At import time we check the
``QWEN_TUTOR_PROMPTS`` environment variable and choose either the full
Opus-grade templates (``prompts.py``, default) or the compact variants
(``prompts_compact.py``) tuned for smaller local teachers like
gpt-oss-20b served by llama.cpp.

Set ``QWEN_TUTOR_PROMPTS=compact`` to opt in. Anything else (or unset)
keeps the original full prompts.

The helpers ``render_level_spec`` and
``validate_prompt_has_locale_instruction`` always come from
``prompts.py`` — they don't have full/compact variants.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_STYLE = (os.environ.get("QWEN_TUTOR_PROMPTS") or "").strip().lower()


def prompt_style() -> str:
    """Return the active style: ``"compact"`` or ``"full"``."""
    return "compact" if _STYLE == "compact" else "full"


if prompt_style() == "compact":
    logger.info("prompts: using COMPACT variants (QWEN_TUTOR_PROMPTS=compact)")
    from qwen_tutor.generation.prompts_compact import (  # noqa: F401
        DIALOGUE_PROMPT_LANGUAGE_REDIRECT,
        DIALOGUE_PROMPT_LOCALE_REDIRECT,
        DIALOGUE_PROMPT_NORMAL,
        DIALOGUE_PROMPT_PEDAGOGY_REDIRECT,
        DIALOGUE_PROMPT_PERSONA_REDIRECT,
        DIALOGUE_PROMPT_REDIRECT,
        EVALUATION_GENERATION_PROMPT,
        REGISTER_REWRITE_PROMPT,
        SPOIL_REWRITE_PROMPT,
        TEMPLATES,
        TOPIC_SEED_PROMPT,
        render_prompt,
    )
else:
    from qwen_tutor.generation.prompts import (  # noqa: F401
        DIALOGUE_PROMPT_LANGUAGE_REDIRECT,
        DIALOGUE_PROMPT_LOCALE_REDIRECT,
        DIALOGUE_PROMPT_NORMAL,
        DIALOGUE_PROMPT_PEDAGOGY_REDIRECT,
        DIALOGUE_PROMPT_PERSONA_REDIRECT,
        DIALOGUE_PROMPT_REDIRECT,
        EVALUATION_GENERATION_PROMPT,
        REGISTER_REWRITE_PROMPT,
        SPOIL_REWRITE_PROMPT,
        TEMPLATES,
        TOPIC_SEED_PROMPT,
        render_prompt,
    )

# These helpers are style-independent. AXIS_SPOIL_INSTRUCTIONS / SPOIL_AXES /
# REJECTION_NOTES are the same dict in both prompt modules, so we import once.
from qwen_tutor.generation.prompts import (  # noqa: F401, E402
    AXIS_SPOIL_INSTRUCTIONS,
    LOCALE_INSTRUCTION_BLOCK,
    LOCALE_INSTRUCTION_HEADER,
    REJECTION_NOTES,
    SPOIL_AXES,
    render_scenario_deployment_system_prompt,
    render_level_spec,
    validate_prompt_has_locale_instruction,
)
