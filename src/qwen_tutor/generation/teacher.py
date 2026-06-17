"""Async teacher-model client for SFT / DPO / eval data generation.

This module exposes a unified ``TeacherClient`` abstraction with two concrete
implementations:

  * ``AnthropicTeacher`` — uses the ``anthropic`` async SDK. Supports prompt
    caching via ``cache_control`` on the system block, and submits batches
    through the Anthropic Messages Batch API.
  * ``OpenAITeacher`` — uses the ``openai`` async SDK. OpenAI's prompt
    caching is automatic for prefixes ≥ 1024 tokens, so we simply place any
    ``cacheable_prefix`` at the front of the system message. Batch
    submission uses the OpenAI Batch API (files + ``/v1/chat/completions``).

Both implementations:

  * Retry with exponential backoff (via ``tenacity``) on rate limits, 5xx
    statuses, connection errors, and timeouts. 4xx errors fail fast.
  * Append a JSONL record per call to a usage log so cost can be tracked
    across the pipeline.
  * Read provider, model, API key, retry policy, and batch settings from
    ``config/generation.yaml`` via ``build_teacher_from_config``.

The ``batch_mode`` flag stored on the client signals intent (e.g. for the
caller's bookkeeping); ``generate``/``submit_batch``/``fetch_batch`` are
always available regardless of the flag.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import openai
import yaml
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from qwen_tutor.schemas import Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchSettings:
    enabled: bool = False
    poll_interval_seconds: float = 30.0
    max_wait_seconds: float = 86400.0


@dataclass(slots=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str
    max_retries: int = 6
    initial_retry_seconds: float = 1.0
    max_retry_seconds: float = 60.0
    # Custom base URL for OpenAI-compatible endpoints (llama.cpp server,
    # vLLM, LM Studio, etc.). Ignored by AnthropicTeacher.
    base_url: str | None = None
    batch: BatchSettings = field(default_factory=BatchSettings)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_provider_config(
    generation_yaml: str | Path, role: str = "teacher"
) -> ProviderConfig:
    """Load the provider config for ``role`` ("teacher" or "judge")."""

    doc = _read_yaml(generation_yaml)
    if role not in doc:
        raise KeyError(f"generation.yaml has no '{role}' section")
    role_doc = doc[role]
    provider = role_doc["provider"]
    if provider not in role_doc:
        raise KeyError(
            f"generation.yaml '{role}.{provider}' section is missing "
            f"(provider is set to '{provider}')"
        )
    section = role_doc[provider]

    api_key_env = section.get("api_key_env")
    api_key_static = section.get("api_key_static")
    # api_key_env takes precedence (production), api_key_static is a fallback
    # used for local OpenAI-compatible servers (llama.cpp etc.) that don't
    # require a real key but the SDK still wants a non-empty string.
    api_key = ""
    if api_key_env:
        api_key = os.environ.get(api_key_env, "")
    if not api_key and api_key_static:
        api_key = str(api_key_static)
    base_url = section.get("base_url")

    batch_doc = section.get("batch") or {}
    batch = BatchSettings(
        enabled=bool(batch_doc.get("enabled", False)),
        poll_interval_seconds=float(batch_doc.get("poll_interval_seconds", 30.0)),
        max_wait_seconds=float(batch_doc.get("max_wait_seconds", 86400.0)),
    )

    return ProviderConfig(
        provider=provider,
        model=section["model"],
        api_key=api_key,
        max_retries=int(section.get("max_retries", 6)),
        initial_retry_seconds=float(section.get("initial_retry_seconds", 1.0)),
        max_retry_seconds=float(section.get("max_retry_seconds", 60.0)),
        base_url=base_url,
        batch=batch,
    )


# ---------------------------------------------------------------------------
# Usage logging
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UsageRecord:
    timestamp: float = 0.0
    provider: str = ""
    model: str = ""
    call_kind: str = "generate"  # "generate" | "batch_submit" | "batch_fetch"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_ms: float = 0.0
    custom_id: str | None = None


class UsageLogger:
    """Append-only JSONL writer for token-usage records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def log(self, record: UsageRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")


# ---------------------------------------------------------------------------
# Batch request / result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchRequest:
    custom_id: str
    system: str
    messages: list[Message]
    cacheable_prefix: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.8


@dataclass(slots=True)
class BatchResult:
    custom_id: str
    content: str | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------


class TeacherClient(ABC):
    """Abstract async client over a teacher LLM."""

    def __init__(
        self,
        config: ProviderConfig,
        usage_logger: UsageLogger,
        batch_mode: bool = False,
    ) -> None:
        self.config = config
        self.usage_logger = usage_logger
        self.batch_mode = batch_mode

    @abstractmethod
    async def generate(
        self,
        system: str,
        messages: list[Message],
        cacheable_prefix: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> str:
        """Run a single generation, returning the assistant text."""

    @abstractmethod
    async def submit_batch(self, requests: list[BatchRequest]) -> str:
        """Submit a batch and return a provider-specific batch id."""

    @abstractmethod
    async def fetch_batch(self, batch_id: str) -> list[BatchResult]:
        """Wait for a batch to finish and return its results."""


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------


def _is_retriable_anthropic(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is not None and status >= 500
    return False


class AnthropicTeacher(TeacherClient):
    def __init__(
        self,
        config: ProviderConfig,
        usage_logger: UsageLogger,
        batch_mode: bool = False,
    ) -> None:
        super().__init__(config, usage_logger, batch_mode)
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key or None)

    def _build_system_blocks(
        self, system: str, cacheable_prefix: str | None
    ) -> list[dict[str, Any]]:
        if cacheable_prefix:
            return [
                {
                    "type": "text",
                    "text": cacheable_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": system},
            ]
        return [{"type": "text", "text": system}]

    @staticmethod
    def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                # Anthropic carries the system prompt at the top level, not in
                # the messages array. Skip any inline system turns.
                continue
            out.append({"role": m.role, "content": m.content})
        return out

    def _retryer(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_random_exponential(
                multiplier=self.config.initial_retry_seconds,
                max=self.config.max_retry_seconds,
            ),
            retry=retry_if_exception(_is_retriable_anthropic),
            reraise=True,
        )

    async def generate(
        self,
        system: str,
        messages: list[Message],
        cacheable_prefix: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> str:
        system_blocks = self._build_system_blocks(system, cacheable_prefix)
        anthropic_msgs = self._to_anthropic_messages(messages)

        start = time.monotonic()
        response = None
        async for attempt in self._retryer():
            with attempt:
                response = await self.client.messages.create(
                    model=self.config.model,
                    system=system_blocks,
                    messages=anthropic_msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
        latency_ms = (time.monotonic() - start) * 1000.0
        assert response is not None  # reraise=True guarantees this

        usage = response.usage
        self.usage_logger.log(
            UsageRecord(
                timestamp=time.time(),
                provider="anthropic",
                model=self.config.model,
                call_kind="generate",
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_creation_input_tokens=getattr(
                    usage, "cache_creation_input_tokens", 0
                )
                or 0,
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0)
                or 0,
                latency_ms=latency_ms,
            )
        )

        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    async def submit_batch(self, requests: list[BatchRequest]) -> str:
        batch_payload = []
        for req in requests:
            batch_payload.append(
                {
                    "custom_id": req.custom_id,
                    "params": {
                        "model": self.config.model,
                        "system": self._build_system_blocks(req.system, req.cacheable_prefix),
                        "messages": self._to_anthropic_messages(req.messages),
                        "max_tokens": req.max_tokens,
                        "temperature": req.temperature,
                    },
                }
            )

        batch = await self.client.messages.batches.create(requests=batch_payload)
        self.usage_logger.log(
            UsageRecord(
                timestamp=time.time(),
                provider="anthropic",
                model=self.config.model,
                call_kind="batch_submit",
            )
        )
        return batch.id

    async def fetch_batch(self, batch_id: str) -> list[BatchResult]:
        deadline = time.monotonic() + self.config.batch.max_wait_seconds
        while True:
            batch = await self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Anthropic batch {batch_id} did not complete within "
                    f"{self.config.batch.max_wait_seconds:.0f}s"
                )
            await asyncio.sleep(self.config.batch.poll_interval_seconds)

        results: list[BatchResult] = []
        async for entry in await self.client.messages.batches.results(batch_id):
            custom_id = entry.custom_id
            result = entry.result
            if result.type == "succeeded":
                msg = result.message
                content = "".join(
                    b.text for b in msg.content if getattr(b, "type", None) == "text"
                )
                usage = msg.usage
                self.usage_logger.log(
                    UsageRecord(
                        timestamp=time.time(),
                        provider="anthropic",
                        model=self.config.model,
                        call_kind="batch_fetch",
                        input_tokens=getattr(usage, "input_tokens", 0) or 0,
                        output_tokens=getattr(usage, "output_tokens", 0) or 0,
                        cache_creation_input_tokens=getattr(
                            usage, "cache_creation_input_tokens", 0
                        )
                        or 0,
                        cache_read_input_tokens=getattr(
                            usage, "cache_read_input_tokens", 0
                        )
                        or 0,
                        custom_id=custom_id,
                    )
                )
                results.append(BatchResult(custom_id=custom_id, content=content))
            else:
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        content=None,
                        error=f"{result.type}: {getattr(result, 'error', '')}",
                    )
                )
        return results


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


def _merge_reasoning_content(message: Any) -> str:
    """Merge ``reasoning_content`` (DeepSeek-R1 / gpt-oss via llama.cpp / vLLM)
    into the visible content as a ``<think>...</think>`` block.

    OpenAI-compatible backends (llama.cpp, vLLM) return gpt-oss harmony
    responses with reasoning in a separate ``message.reasoning_content``
    field. If the prompt does not already embed a ``<think>`` tag in the
    visible content, the eval stage may mistakenly consider reasoning to be
    empty, so we merge it here.

    Qwen3 behavior does not change when reasoning_content is absent or empty.
    """
    if isinstance(message, dict):
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
    else:
        content = getattr(message, "content", "") or ""
        reasoning = getattr(message, "reasoning_content", "") or ""
    reasoning = reasoning.strip()
    if not reasoning:
        return content
    if "<think>" in content and "</think>" in content:
        # If content already contains a <think> block, preserve it
        # (to avoid double wrapping).
        return content
    return f"<think>\n{reasoning}\n</think>\n{content}"


def _is_retriable_openai(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is not None and status >= 500
    return False


class OpenAITeacher(TeacherClient):
    def __init__(
        self,
        config: ProviderConfig,
        usage_logger: UsageLogger,
        batch_mode: bool = False,
    ) -> None:
        super().__init__(config, usage_logger, batch_mode)
        client_kwargs: dict[str, Any] = {"api_key": config.api_key or None}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self.client = openai.AsyncOpenAI(**client_kwargs)

    def _build_messages(
        self,
        system: str,
        cacheable_prefix: str | None,
        messages: list[Message],
    ) -> list[dict[str, str]]:
        # OpenAI prompt caching is automatic for ≥ 1024-token prefixes, keyed
        # on the exact leading bytes of the prompt. Placing cacheable_prefix
        # at the front of the system message maximizes cache hits.
        if cacheable_prefix:
            system_content = f"{cacheable_prefix}\n\n{system}"
        else:
            system_content = system
        out: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        has_user = False
        for m in messages:
            if m.role == "system":
                continue
            out.append({"role": m.role, "content": m.content})
            if m.role == "user":
                has_user = True
        # Some chat templates (Qwen3 chatml, Qwen3.5, ...) require at least one
        # user turn and raise a Jinja exception when given a system-only prompt.
        # gpt-oss harmony tolerates system-only, so this is invisible there.
        # Adding a trivial user turn at the end satisfies the strict templates
        # without disturbing the system payload that carries the real instruction.
        if not has_user:
            out.append({"role": "user", "content": "Begin."})
        return out

    def _retryer(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_random_exponential(
                multiplier=self.config.initial_retry_seconds,
                max=self.config.max_retry_seconds,
            ),
            retry=retry_if_exception(_is_retriable_openai),
            reraise=True,
        )

    async def generate(
        self,
        system: str,
        messages: list[Message],
        cacheable_prefix: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> str:
        oai_msgs = self._build_messages(system, cacheable_prefix, messages)

        # Decide whether to disable Qwen3-style "thinking" mode based on
        # the system prompt's directive. Qwen3 family templates respect
        # ``enable_thinking`` via chat_template_kwargs; the inline
        # ``/no_think`` / ``/think`` directive in the prompt body is NOT
        # honored by every model (Qwen3.5 ignores it and burns max_tokens
        # in reasoning_content). Detect the directive in the system text
        # and forward it as a chat-template kwarg so the model actually
        # complies. Other providers ignore unknown extra_body fields.
        enable_thinking: bool | None
        if "/no_think" in system:
            enable_thinking = False
        elif "/think" in system:
            enable_thinking = True
        else:
            enable_thinking = None
        extra_body: dict[str, Any] = {}
        if enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking
            }

        start = time.monotonic()
        response = None
        async for attempt in self._retryer():
            with attempt:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=oai_msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=extra_body or None,
                )
        latency_ms = (time.monotonic() - start) * 1000.0
        assert response is not None

        usage = response.usage
        cached_tokens = 0
        if usage is not None and getattr(usage, "prompt_tokens_details", None) is not None:
            cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        self.usage_logger.log(
            UsageRecord(
                timestamp=time.time(),
                provider="openai",
                model=self.config.model,
                call_kind="generate",
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cache_read_input_tokens=cached_tokens,
                latency_ms=latency_ms,
            )
        )

        return _merge_reasoning_content(response.choices[0].message)

    async def submit_batch(self, requests: list[BatchRequest]) -> str:
        lines: list[str] = []
        for req in requests:
            body = {
                "model": self.config.model,
                "messages": self._build_messages(
                    req.system, req.cacheable_prefix, req.messages
                ),
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
            }
            lines.append(
                json.dumps(
                    {
                        "custom_id": req.custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
            )
        payload_bytes = ("\n".join(lines) + "\n").encode("utf-8")

        file_obj = await self.client.files.create(
            file=("batch_input.jsonl", io.BytesIO(payload_bytes)),
            purpose="batch",
        )
        batch = await self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        self.usage_logger.log(
            UsageRecord(
                timestamp=time.time(),
                provider="openai",
                model=self.config.model,
                call_kind="batch_submit",
            )
        )
        return batch.id

    async def fetch_batch(self, batch_id: str) -> list[BatchResult]:
        deadline = time.monotonic() + self.config.batch.max_wait_seconds
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            batch = await self.client.batches.retrieve(batch_id)
            if batch.status in terminal:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"OpenAI batch {batch_id} did not complete within "
                    f"{self.config.batch.max_wait_seconds:.0f}s"
                )
            await asyncio.sleep(self.config.batch.poll_interval_seconds)

        results: list[BatchResult] = []
        if not batch.output_file_id:
            return results

        content_resp = await self.client.files.content(batch.output_file_id)
        raw = await content_resp.aread() if hasattr(content_resp, "aread") else content_resp.read()
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id", "")
            err = entry.get("error")
            if err:
                results.append(
                    BatchResult(custom_id=custom_id, content=None, error=json.dumps(err))
                )
                continue
            response_body = entry["response"]["body"]
            content = _merge_reasoning_content(
                response_body["choices"][0]["message"]
            )
            usage = response_body.get("usage") or {}
            cached_tokens = 0
            details = usage.get("prompt_tokens_details") or {}
            if isinstance(details, dict):
                cached_tokens = int(details.get("cached_tokens") or 0)
            self.usage_logger.log(
                UsageRecord(
                    timestamp=time.time(),
                    provider="openai",
                    model=self.config.model,
                    call_kind="batch_fetch",
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    cache_read_input_tokens=cached_tokens,
                    custom_id=custom_id,
                )
            )
            results.append(BatchResult(custom_id=custom_id, content=content))
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _detect_served_model(
    base_url: str, api_key: str, timeout: float = 3.0,
) -> str | None:
    """Query the OpenAI-compatible /models endpoint and return the model
    name actually being served. Returns ``None`` on any failure (timeout,
    connection error, malformed response).

    Used to auto-correct ``ProviderConfig.model`` when the YAML points at a
    self-hosted llama-server / vLLM / LM Studio that ignores the
    client-supplied model name and serves whatever was loaded. Without this,
    every generated record has stale ``metadata.generation.model`` that
    mismatches the actual teacher (e.g. the YAML says ``qwen3.5-4b`` but the
    server is serving ``Qwen3.5-9B-UD-Q4_K_XL.gguf``).
    """
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("_detect_served_model: failed to query %s: %s", url, exc)
        return None
    if not isinstance(data, dict):
        return None
    # llama-server response:  {"models": [{"name": "...", ...}], ...}
    # OpenAI / vLLM response: {"data":   [{"id": "...", ...}], ...}
    for arr_key, name_key in (("models", "name"), ("data", "id")):
        arr = data.get(arr_key)
        if isinstance(arr, list) and arr:
            first = arr[0]
            if isinstance(first, dict):
                name = first.get(name_key) or first.get("id") or first.get("model")
                if isinstance(name, str) and name:
                    return name
    return None


def build_teacher_from_config(
    generation_yaml: str | Path = "config/generation.yaml",
    role: str = "teacher",
    batch_mode: bool | None = None,
) -> TeacherClient:
    """Construct a TeacherClient for ``role`` ("teacher" or "judge")."""

    cfg = load_provider_config(generation_yaml, role=role)
    doc = _read_yaml(generation_yaml)
    usage_log_path = (doc.get("usage_log") or {}).get(
        "path", "data/_logs/token_usage.jsonl"
    )
    usage_logger = UsageLogger(usage_log_path)
    effective_batch_mode = cfg.batch.enabled if batch_mode is None else batch_mode

    # Auto-detect the actually-served model when talking to a local
    # OpenAI-compatible endpoint. llama-server/vLLM/LM Studio all ignore
    # the model name in client requests, so the YAML can lie. Override
    # cfg.model in-place so every downstream record records the truth.
    # (For real cloud OpenAI/Anthropic, base_url is None and we skip this.)
    if cfg.provider == "openai" and cfg.base_url:
        detected = _detect_served_model(cfg.base_url, cfg.api_key)
        if detected and detected != cfg.model:
            logger.info(
                "endpoint %s serves %r (config said %r); using actual served name in metadata",
                cfg.base_url, detected, cfg.model,
            )
            cfg.model = detected

    if cfg.provider == "anthropic":
        return AnthropicTeacher(cfg, usage_logger, batch_mode=effective_batch_mode)
    if cfg.provider == "openai":
        return OpenAITeacher(cfg, usage_logger, batch_mode=effective_batch_mode)
    raise ValueError(f"Unknown provider: {cfg.provider!r}")
