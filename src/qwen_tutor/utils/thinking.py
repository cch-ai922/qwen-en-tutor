"""Central knob for Qwen3.5 ``<think>...</think>`` reasoning mode.

Every place in the codebase that used to hardcode ``"/no_think\\n" + PROMPT``
should call :func:`with_thinking_directive` instead. The directive is then
chosen by ``config/generation.yaml``'s ``thinking:`` block at module-load
time, so flipping a role between ``think`` and ``no_think`` is a one-line
config change with no code edits.

Roles
-----
``teacher``
    Data-generation teacher (SFT dialogues, redirect streams, register pairs,
    eval examples). Almost always ``no_think`` for cost / throughput.

``judge``
    All LLM judges: filter-pipeline judges (``locale_judge`` etc.), the
    on-policy DPO pair judge, and the eval-metric judges
    (``topic_adherence``, ``naturalness_judge``, ``redirect_success_judge``).
    ``no_think`` on small (<=4B Q4) teachers; ``think`` if you can afford
    2048+ output tokens per call and have an 8B+ teacher.

``learner``
    The learner-simulator that produces user turns during ``run_eval``.
    Must be ``no_think`` — leaving ``<think>`` content in user turns
    poisons every downstream judge call.

``student_eval``
    The trained student's ``/think`` CEFR evaluation mode. ``think`` for
    accuracy; ``no_think`` only if you need faster shallow scoring.

Resolution order
----------------
1. Env var ``QWEN_TUTOR_THINK_<ROLE>=think|no_think|auto`` if set (e.g.
   ``QWEN_TUTOR_THINK_JUDGE=think``). Useful for one-off experiments.
2. ``config/generation.yaml`` → ``thinking.<role>`` if present.
3. Built-in default per role.

The OpenAITeacher client in :mod:`qwen_tutor.generation.teacher` already
detects ``/think`` / ``/no_think`` substrings in the system prompt and
forwards them as ``chat_template_kwargs.enable_thinking`` to llama-server,
so prepending the directive is sufficient — no additional wiring needed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

ThinkingMode = Literal["think", "no_think", "auto"]
Role = Literal["teacher", "judge", "learner", "student_eval"]

# Built-in defaults if neither env vars nor config specify a role.
_DEFAULTS: dict[Role, ThinkingMode] = {
    "teacher": "no_think",
    "judge": "no_think",
    "learner": "no_think",
    "student_eval": "think",
}

_VALID_MODES = frozenset({"think", "no_think", "auto"})

_DEFAULT_CONFIG_PATH = Path("config/generation.yaml")

# In-process cache. ``load_thinking_config`` is called once per process at
# module import; explicit reloads can be done via :func:`reset_cache`.
_cached: dict[Role, ThinkingMode] | None = None


def _read_config(path: Path) -> dict[Role, ThinkingMode]:
    """Load the ``thinking:`` block from generation.yaml.

    Missing file / missing block / unknown values fall back to the
    per-role default and emit a warning.
    """
    out: dict[Role, ThinkingMode] = dict(_DEFAULTS)
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("could not read %s: %s; using thinking defaults", path, exc)
        return out
    block = doc.get("thinking") or {}
    if not isinstance(block, dict):
        logger.warning("config.thinking is not a mapping in %s; using defaults", path)
        return out
    for role in _DEFAULTS:
        if role not in block:
            continue
        value = block[role]
        if isinstance(value, str) and value in _VALID_MODES:
            out[role] = value  # type: ignore[assignment]
        else:
            logger.warning(
                "config.thinking.%s = %r is invalid (expected one of %s); "
                "falling back to default %r",
                role, value, sorted(_VALID_MODES), _DEFAULTS[role],
            )
    return out


def load_thinking_config(
    path: str | Path | None = None,
    *,
    force_reload: bool = False,
) -> dict[Role, ThinkingMode]:
    """Return the effective {role: mode} map, applying env-var overrides.

    Cached per process; pass ``force_reload=True`` after editing the
    config file to pick up changes without restarting Python.
    """
    global _cached
    if _cached is not None and not force_reload and path is None:
        return dict(_cached)

    resolved = _read_config(Path(path) if path else _DEFAULT_CONFIG_PATH)

    # Env-var overrides take precedence so an operator can change a role
    # for one shell session without editing YAML.
    for role in _DEFAULTS:
        env_key = f"QWEN_TUTOR_THINK_{role.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is None:
            continue
        if env_val in _VALID_MODES:
            resolved[role] = env_val  # type: ignore[assignment]
        else:
            logger.warning(
                "%s=%r is not one of %s; ignoring",
                env_key, env_val, sorted(_VALID_MODES),
            )

    if path is None:
        _cached = dict(resolved)
    return resolved


def reset_cache() -> None:
    """Drop the cached config so the next call reads YAML and env again."""
    global _cached
    _cached = None


def get_thinking_mode(role: Role) -> ThinkingMode:
    """Return the effective mode for a role (``think`` / ``no_think`` / ``auto``)."""
    return load_thinking_config()[role]


def with_thinking_directive(prompt: str, *, role: Role) -> str:
    """Prepend the configured ``/think`` or ``/no_think`` directive.

    Skips prepending when the mode is ``auto`` or when the prompt already
    contains an explicit directive at its very start, so callers that
    hand-encode the directive are not double-prepended.
    """
    mode = get_thinking_mode(role)
    if mode == "auto":
        return prompt
    stripped = prompt.lstrip()
    if stripped.startswith("/think") or stripped.startswith("/no_think"):
        # Caller already encoded a directive — respect it.
        return prompt
    directive = "/think" if mode == "think" else "/no_think"
    return f"{directive}\n{prompt}"


__all__ = [
    "Role",
    "ThinkingMode",
    "get_thinking_mode",
    "load_thinking_config",
    "reset_cache",
    "with_thinking_directive",
]
