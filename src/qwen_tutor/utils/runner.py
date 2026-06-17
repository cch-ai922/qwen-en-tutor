"""Shared async helpers for the generation pipeline.

Every generation stage (seeds, sft, redirect, register pairs, evaluation)
shares the same scaffolding:

  * resume from checkpoint by reading already-written IDs from a JSONL
    output file;
  * run a bounded-concurrency batch of async LLM calls with a tqdm
    progress bar;
  * append successes to the output JSONL and failures to a per-stage
    ``<stage>_failures.jsonl`` log.

This module centralizes those primitives so the per-stage modules stay
focused on prompt construction and parsing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

from tqdm.asyncio import tqdm as atqdm

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Deterministic hash-based subsampling
# ---------------------------------------------------------------------------


def deterministic_sample(
    items: list[T],
    fraction: float,
    key: Callable[[T], str],
) -> list[T]:
    """Return roughly ``fraction`` of ``items``, selected by hashing
    ``key(item)``.

    Unlike ``random.sample``, this returns the same subset across seed
    reruns and partial resumes (it keeps the ``ceil(N*fraction)`` items
    whose hash is smallest). Returns an empty list when ``fraction`` is
    0 and every item when ``fraction`` is >= 1.
    """
    if fraction >= 1.0:
        return list(items)
    if fraction <= 0.0 or not items:
        return []
    n_keep = max(1, int(round(len(items) * fraction)))

    def _h(item: T) -> int:
        return int(hashlib.sha256(key(item).encode("utf-8")).hexdigest()[:16], 16)

    ranked = sorted(items, key=_h)
    return ranked[:n_keep]


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def load_existing_ids(jsonl_path: str | Path, id_field: str = "id") -> set[str]:
    """Return the set of IDs already present in ``jsonl_path``.

    Used for resume-from-checkpoint logic: a generation stage looks up the
    set of finished IDs and skips any input item whose target ID is in it.
    Missing or empty files return an empty set.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and id_field in obj:
                ids.add(str(obj[id_field]))
    return ids


def append_jsonl(path: str | Path, obj: Any) -> None:
    """Append a single record to a JSONL file.

    ``obj`` may be a Pydantic ``BaseModel`` (in which case
    ``model_dump_json`` is used), or any JSON-serializable value.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        if hasattr(obj, "model_dump_json"):
            fh.write(obj.model_dump_json())
        else:
            fh.write(json.dumps(obj, ensure_ascii=False))
        fh.write("\n")


def append_failure(
    failures_path: str | Path,
    custom_id: str,
    error: str,
    **meta: Any,
) -> None:
    """Append a failure record to a stage-specific failures log."""
    append_jsonl(
        failures_path,
        {"custom_id": custom_id, "error": error, **meta},
    )


# ---------------------------------------------------------------------------
# Bounded concurrency
# ---------------------------------------------------------------------------


async def gather_with_concurrency(
    coros: Iterable[Awaitable[T]],
    concurrency: int = 20,
    desc: str | None = None,
) -> list[T]:
    """Run an iterable of coroutines with bounded concurrency + a tqdm bar.

    Results are returned in input order. Exceptions from individual
    coroutines propagate; wrap each coroutine in a try/except at the call
    site if you want to convert failures into a sentinel value.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(coro: Awaitable[T]) -> T:
        async with sem:
            return await coro

    wrapped = [_bounded(c) for c in coros]
    if not wrapped:
        return []
    return await atqdm.gather(*wrapped, desc=desc)


# ---------------------------------------------------------------------------
# Light JSON-parsing helpers used across stages
# ---------------------------------------------------------------------------


def strip_code_fence(text: str) -> str:
    """Drop a leading/trailing ``\`\`\`json ... \`\`\`\`` fence if present.

    Teacher models occasionally wrap a JSON-only response in a markdown
    fence despite explicit instructions not to. This helper keeps parsing
    robust without making the prompt lie about format.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if not lines:
        return s
    # drop the opening fence (e.g. ```json)
    body = lines[1:]
    # drop the closing fence if present
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body)


# Reasoning models (Qwen3, DeepSeek-R1, gpt-oss, ...) emit a private
# chain-of-thought block before the real answer. The most common shapes:
#
#   Qwen3 / DeepSeek-R1 / generic:  <think>...</think>
#   gpt-oss "harmony":              <|channel|>analysis<|message|>...<|end|>
#                                   <|channel|>final<|message|>{actual answer}
#
# llama.cpp usually strips harmony markers from the OpenAI `content` field
# itself, but some builds leak them through. Strip both shapes defensively
# before JSON extraction.
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_HARMONY_ANALYSIS_RE = re.compile(
    # consume the analysis channel body AND any trailing <|end|> so the
    # token doesn't leak into the parser's view of the payload.
    r"<\|channel\|>\s*analysis\s*<\|message\|>.*?(?:<\|end\|>|(?=<\|channel\|>)|$)",
    re.DOTALL | re.IGNORECASE,
)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|$)",
    re.DOTALL | re.IGNORECASE,
)
# Any stray harmony tokens that survived (e.g. an isolated <|end|>) -
# strip them too, since they can confuse JSON extraction downstream.
_HARMONY_TOKEN_RE = re.compile(r"<\|(?:end|start|channel|message)\|>", re.IGNORECASE)


def strip_think_block(text: str) -> str:
    """Remove reasoning-model "private thought" blocks before parsing.

    Handles both:
      * generic ``<think>...</think>`` (Qwen3, DeepSeek-R1, ...).
      * gpt-oss harmony channel markers (``<|channel|>analysis...``).
        For harmony, if a ``<|channel|>final<|message|>...`` block exists
        we KEEP only its body and discard everything else; otherwise we
        just strip the analysis block.
    """
    out = text
    # 1) gpt-oss "harmony": prefer the 'final' channel body if present.
    final_match = _HARMONY_FINAL_RE.search(out)
    if final_match:
        return _HARMONY_TOKEN_RE.sub("", final_match.group(1)).strip()
    # No final channel — drop any analysis channel blob and keep the rest.
    out = _HARMONY_ANALYSIS_RE.sub("", out)
    # Sweep up any stray harmony tokens that survived.
    out = _HARMONY_TOKEN_RE.sub("", out)

    # 2) generic <think>...</think> blocks.
    if "<think" in out.lower():
        out = _THINK_BLOCK_RE.sub("", out)
        lower = out.lower()
        # If a ``<think>`` tag remains unclosed, drop everything up to (and
        # including) it — we cannot recover well, but the JSON the caller
        # wants is usually after, not inside.
        if "<think" in lower:
            idx = lower.rfind("<think")
            out = out[:idx]
    return out.strip()


def extract_first_json(text: str) -> Any:
    """Strip fences then parse the first valid JSON value out of ``text``.

    Tries a direct ``json.loads`` first; if that fails, locates the first
    balanced ``{...}`` or ``[...]`` block at the top level and parses
    that. Raises ``json.JSONDecodeError`` if nothing parses.

    Also strips ``<think>...</think>`` chain-of-thought blocks emitted by
    reasoning models (Qwen3, DeepSeek-R1, etc.) before parsing, so a JSON
    fragment quoted inside the model's reasoning never gets returned.
    """
    s = strip_think_block(text)
    s = strip_code_fence(s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Find first balanced { or [ at depth 0 (string-aware)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(s[start : i + 1])
    raise json.JSONDecodeError("no balanced JSON value found", s, 0)
