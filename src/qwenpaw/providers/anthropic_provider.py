# -*- coding: utf-8 -*-
"""An Anthropic provider implementation."""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from datetime import datetime  # noqa: F401  # kept for parity with main
from pathlib import Path
from typing import Any, Dict, List

import httpx
from agentscope.model import AnthropicChatModel, ChatModelBase
import anthropic

from qwenpaw.providers.multimodal_prober import (
    ProbeResult,
    _PROBE_IMAGE_B64,
    _IMAGE_PROBE_PROMPT,
    _is_media_keyword_error,
    evaluate_image_probe_answer,
)
from qwenpaw.providers.provider import ModelInfo, Provider

logger = logging.getLogger(__name__)

# Maximum number of body bytes to dump per request log line.
# Bodies larger than this are truncated to avoid flooding logs.
_MAX_LOGGED_BODY_BYTES = 16 * 1024
# Header names that must never be logged in clear text.
_SENSITIVE_HEADER_KEYS = {
    "authorization",
    "x-api-key",
    "auth_token",
    "proxy-authorization",
}
# Dedicated logger + log file for raw Anthropic HTTP requests.
# Lives next to the main qwenpaw.log so it's easy to find but stays
# isolated (propagate=False) to avoid polluting the main log/stderr.
_ANTHROPIC_REQUEST_LOG_BASENAME = "qwenpaw-anthropic.log"
_ANTHROPIC_REQUEST_LOGGER_NAME = "qwenpaw.anthropic_http"
_ANTHROPIC_REQUEST_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
_ANTHROPIC_REQUEST_LOG_BACKUP_COUNT = 5
_request_logger_initialized = False


def _get_request_logger() -> logging.Logger:
    """Return a dedicated logger that writes Anthropic HTTP request details
    to a separate rotating log file (``qwenpaw-anthropic.log``).

    Initialization is lazy and idempotent so importing this module never
    creates the log file as a side effect.
    """
    global _request_logger_initialized  # pylint: disable=global-statement
    req_logger = logging.getLogger(_ANTHROPIC_REQUEST_LOGGER_NAME)
    if _request_logger_initialized:
        return req_logger
    try:
        # Lazy imports keep this module decoupled from app startup order.
        from qwenpaw.constant import WORKING_DIR
        from qwenpaw.utils.logging import (
            PlainFormatter,
            _SafeRotatingFileHandler,
        )

        log_path = Path(WORKING_DIR) / _ANTHROPIC_REQUEST_LOG_BASENAME
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path_resolved = log_path.resolve()

        already_attached = False
        for handler in req_logger.handlers:
            base = getattr(handler, "baseFilename", None)
            if (
                base is not None
                and Path(base).resolve() == log_path_resolved
            ):
                already_attached = True
                break

        if not already_attached:
            file_handler = _SafeRotatingFileHandler(
                log_path,
                encoding="utf-8",
                maxBytes=_ANTHROPIC_REQUEST_LOG_MAX_BYTES,
                backupCount=_ANTHROPIC_REQUEST_LOG_BACKUP_COUNT,
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                PlainFormatter(
                    "%(asctime)s | %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                ),
            )
            req_logger.addHandler(file_handler)

        req_logger.setLevel(logging.INFO)
        # Keep these verbose request dumps out of stderr / qwenpaw.log.
        req_logger.propagate = False
    except Exception:
        # Never let logging setup break the actual request path; fall
        # back to the module logger if file handler creation fails.
        logger.exception("Failed to init Anthropic request log file")
    finally:
        _request_logger_initialized = True
    return req_logger


def _redact_headers(headers: Any) -> Dict[str, str]:
    """Return a dict copy of headers with sensitive values masked.

    Prefers ``headers.raw`` (the *exact* bytes that httpx will write to
    the wire, preserving the original case) so the log faithfully
    reflects what the server sees.  Falls back to ``.items()`` (which
    httpx normalizes to lowercase) for non-httpx header containers.
    """
    redacted: Dict[str, str] = {}

    raw = getattr(headers, "raw", None)
    if raw is not None:
        try:
            for k_bytes, v_bytes in raw:
                key = (
                    k_bytes.decode("ascii", "replace")
                    if isinstance(k_bytes, (bytes, bytearray))
                    else str(k_bytes)
                )
                value = (
                    v_bytes.decode("utf-8", "replace")
                    if isinstance(v_bytes, (bytes, bytearray))
                    else str(v_bytes)
                )
                if key.lower() in _SENSITIVE_HEADER_KEYS:
                    redacted[key] = "***REDACTED***"
                else:
                    redacted[key] = value
            return redacted
        except Exception:  # pragma: no cover - fall back to items()
            redacted.clear()

    try:
        items = headers.items()
    except AttributeError:
        items = list(headers or [])
    for key, value in items:
        if str(key).lower() in _SENSITIVE_HEADER_KEYS:
            redacted[str(key)] = "***REDACTED***"
        else:
            redacted[str(key)] = str(value)
    return redacted


def _format_request_body(content: Any) -> str:
    """Best-effort decode of an httpx request body for logging."""
    if not content:
        return ""
    if isinstance(content, (bytes, bytearray)):
        raw = bytes(content)
        if len(raw) > _MAX_LOGGED_BODY_BYTES:
            return (
                raw[:_MAX_LOGGED_BODY_BYTES].decode("utf-8", "replace")
                + f"...[truncated {len(raw) - _MAX_LOGGED_BODY_BYTES} bytes]"
            )
        return raw.decode("utf-8", "replace")
    text = str(content)
    if len(text) > _MAX_LOGGED_BODY_BYTES:
        return (
            text[:_MAX_LOGGED_BODY_BYTES]
            + f"...[truncated {len(text) - _MAX_LOGGED_BODY_BYTES} chars]"
        )
    return text


def _log_anthropic_request(
    request: httpx.Request,
    *,
    note: str = "",
) -> None:
    """Emit a single INFO log line containing method, URL, headers and body
    of an outgoing Anthropic HTTP request.

    Sensitive headers are redacted; the body is decoded and truncated to
    ``_MAX_LOGGED_BODY_BYTES``.  Output goes to a dedicated rotating file
    (``qwenpaw-anthropic.log``) under ``WORKING_DIR`` and is *not*
    propagated to the main qwenpaw logger.
    """
    try:
        body_text = _format_request_body(request.content)
        _get_request_logger().info(
            "Anthropic v2 request%s | method=%s url=%s headers=%s body=%s",
            f" [{note}]" if note else "",
            request.method,
            str(request.url),
            _redact_headers(request.headers),
            body_text,
        )
    except Exception:  # pragma: no cover - logging must never raise
        logger.exception("Failed to log Anthropic request")


DASHSCOPE_BASE_URLS = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
)
CODING_DASHSCOPE_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
TOKEN_PLAN_BASE_URL = (
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)


class _StripApiKeyTransport(httpx.AsyncHTTPTransport):
    """Async transport that removes the x-api-key header from every request.

    Used when auth_mode='auth_token' to avoid sending both x-api-key and
    Authorization headers simultaneously, which some proxies reject.

    The request is reconstructed with ``extensions`` preserved so that
    per-request configuration such as timeouts and SSE hints set by the
    Anthropic SDK are not lost.
    """

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        # IMPORTANT: iterate over ``headers.raw`` (list of byte tuples that
        # preserve the *original* casing) instead of ``headers.items()``
        # (which httpx normalizes to lowercase).  Re-building the Request
        # from lowercased names would force every header on the wire to be
        # lowercase, breaking proxies that are (incorrectly) case-sensitive
        # for headers such as ``Venus-Sticky-Routing``.
        filtered_raw = [
            (k_bytes, v_bytes)
            for k_bytes, v_bytes in request.headers.raw
            if k_bytes.lower() != b"x-api-key"
        ]
        new_request = httpx.Request(
            method=request.method,
            url=request.url,
            headers=filtered_raw,
            content=request.content,
            extensions=request.extensions,
        )
        # Log the *final* outgoing request (after stripping x-api-key) so
        # the entry reflects exactly what is sent over the wire.
        _log_anthropic_request(new_request, note="auth_token")
        return await super().handle_async_request(new_request)


class _LoggingTransport(httpx.AsyncHTTPTransport):
    """Async transport that logs every outgoing request.

    Used for ``auth_mode='api_key'`` (and other non-strip flows) so that
    headers/body of every Anthropic call are visible in the log just like
    they are for ``auth_token`` mode.
    """

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        _log_anthropic_request(request, note="api_key")
        return await super().handle_async_request(request)


# ---------------------------------------------------------------------------
# Prompt caching helpers
# ---------------------------------------------------------------------------

# Default ``cache_control`` block. Anthropic currently only supports
# ``ephemeral`` (5-minute TTL) for the public API.
_EPHEMERAL_CACHE_CONTROL: Dict[str, str] = {"type": "ephemeral"}

# Roll-out switch.  Defaults to ON; set ``QWENPAW_PROMPT_CACHE_ENABLED=0``
# (or ``false``/``no``/``off``) to disable cache_control injection at
# runtime without redeploying.
_PROMPT_CACHE_ENV_VAR = "QWENPAW_PROMPT_CACHE_ENABLED"


def _prompt_cache_enabled() -> bool:
    raw = os.environ.get(_PROMPT_CACHE_ENV_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _dump_anthropic_request_on_error(
    *,
    pre_messages: Any,
    post_messages: Any,
    tools: Any,
    tool_choice: Any,
    structured_model: Any,
    generate_kwargs: Dict[str, Any],
    exception: BaseException,
) -> None:
    """Write the full Anthropic request inputs to ``/tmp`` whenever the
    upstream call raises.  Each invocation gets its own filename so a
    later successful call cannot overwrite the failing snapshot.

    The file is intended for ad-hoc forensics after a 400/5xx failure;
    it contains the request as it was about to be sent (after cache
    injection) plus the original pre-injection messages, the exception
    type/message, and a minimal context block.
    """
    out_dir = os.environ.get(
        "QWENPAW_ANTHROPIC_DUMP_DIR",
        "/tmp",
    )
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        out_dir = "/tmp"

    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = f"qwenpaw-anthropic-error-{ts}-{os.getpid()}-{id(exception):x}.json"
    fpath = os.path.join(out_dir, fname)

    payload: Dict[str, Any] = {
        "exception_type": type(exception).__name__,
        "exception_repr": repr(exception),
        "tool_choice": tool_choice,
        "structured_model": (
            None if structured_model is None else repr(structured_model)
        ),
        "generate_kwargs": {
            k: v for k, v in (generate_kwargs or {}).items()
            if isinstance(v, (str, int, float, bool, type(None), list, dict))
        },
        "tools": tools,
        "pre_messages": pre_messages,
        "post_messages": post_messages,
    }
    body = getattr(exception, "body", None)
    if body is not None:
        payload["exception_body"] = body
    response = getattr(exception, "response", None)
    if response is not None:
        payload["exception_status_code"] = getattr(
            response, "status_code", None,
        )

    try:
        with open(fpath, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, default=str, indent=2)
    except Exception:  # pragma: no cover - forensics best-effort
        logger.exception("Could not write %s", fpath)
        return
    logger.error(
        "Anthropic request snapshot dumped to %s (exc=%s)",
        fpath, type(exception).__name__,
    )

class _CachingAnthropicChatModel(AnthropicChatModel):
    """An :class:`AnthropicChatModel` that automatically attaches
    ``cache_control: {"type": "ephemeral"}`` breakpoints to enable
    Anthropic prompt caching.

    Three breakpoints are placed (per Anthropic's "up to 4 cache
    breakpoints per request" rule) on the most-stable parts of the
    payload, in priority order:

    1. The **last tool** in ``tools`` — tool schemas rarely change, so
       caching them yields the highest hit rate across turns.
    2. The **system prompt** — the system message is constant for the
       lifetime of an agent.  We rewrite it from the plain-string form
       into Anthropic's structured ``[{"type": "text", "text": ...,
       "cache_control": {...}}]`` form so the breakpoint can be carried.
    3. The **last message block** — caching the conversation up to (and
       including) the latest user/assistant turn lets follow-up turns
       reuse all prior context.

    Injection is purely additive: existing ``cache_control`` values
    (whether attached by the caller upstream or already present on a
    structured content block) are respected and never overwritten.
    Empty / missing pieces are skipped silently so e.g. a tools-less
    request stays valid.

    The behaviour can be disabled at runtime via the environment
    variable ``QWENPAW_PROMPT_CACHE_ENABLED=0``.
    """

    async def __call__(  # type: ignore[override]
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: Any = None,
        structured_model: Any = None,
        **generate_kwargs: Any,
    ) -> Any:
        # Snapshot of pre-injection messages for crash forensics.
        pre_messages = messages
        if _prompt_cache_enabled():
            try:
                tools = self._inject_tools_cache(tools)
                messages = self._inject_messages_cache(messages)
            except Exception:  # pragma: no cover - never break the call
                logger.exception(
                    "Failed to inject Anthropic prompt cache breakpoints; "
                    "falling back to uncached request",
                )
        try:
            return await super().__call__(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                structured_model=structured_model,
                **generate_kwargs,
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            # On any 4xx/5xx from Anthropic, dump the exact request inputs
            # to /tmp so the offending payload can never be lost to log
            # rotation or overwritten by a subsequent successful call.
            try:
                _dump_anthropic_request_on_error(
                    pre_messages=pre_messages,
                    post_messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    structured_model=structured_model,
                    generate_kwargs=generate_kwargs,
                    exception=exc,
                )
            except Exception:  # pragma: no cover - dumping must never mask
                logger.exception("Failed to dump Anthropic request on error")
            raise

    @staticmethod
    def _inject_tools_cache(
        tools: List[Dict[str, Any]] | None,
    ) -> List[Dict[str, Any]] | None:
        """Attach ``cache_control`` to the last tool in ``tools``.

        The vendored AgentScope ``_format_tools_json_schemas`` already
        forwards a top-level ``cache_control`` key on each schema to the
        Anthropic API, so we only need to set it here.
        """
        if not tools:
            return tools
        # Defensive copy: never mutate the caller's list/elements.
        out = list(tools)
        last = copy.copy(out[-1])
        if "cache_control" not in last:
            last["cache_control"] = dict(_EPHEMERAL_CACHE_CONTROL)
            out[-1] = last
        return out

    @classmethod
    def _inject_messages_cache(
        cls,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rewrite ``messages`` so that:

        * the system prompt (if any, must be ``messages[0]`` per Anthropic
          convention) carries a ``cache_control`` breakpoint, and
        * the **last** content block of the **last** message carries a
          ``cache_control`` breakpoint.
        """
        if not messages:
            return messages

        out = [dict(m) for m in messages]

        # 1. System breakpoint — only when the first message is a system
        #    role.  We rewrite ``content`` to the structured array form
        #    so it can hold cache_control.
        if out and out[0].get("role") == "system":
            sys_msg = out[0]
            sys_msg["content"] = cls._content_with_cache(
                sys_msg.get("content"),
            )

        # 2. Last-message breakpoint.  Skip the system message if it is
        #    the only message (degenerate case; system already cached).
        last_idx = len(out) - 1
        if last_idx >= 0 and not (
            last_idx == 0 and out[0].get("role") == "system"
        ):
            last_msg = out[last_idx]
            last_msg["content"] = cls._content_with_cache(
                last_msg.get("content"),
            )

        return out

    @staticmethod
    def _content_with_cache(content: Any) -> Any:
        """Return a copy of ``content`` with a trailing
        ``cache_control: {"type": "ephemeral"}`` breakpoint applied to
        its last text block.

        Accepts both the plain-string form and the structured list form
        used by Anthropic's content-blocks API.  If the content is
        empty or unrecognized it is returned unchanged.
        """
        if content is None:
            return content

        # Plain string -> wrap into a single text block carrying the
        # cache_control breakpoint.  This is the canonical way to attach
        # cache_control to the system prompt.
        if isinstance(content, str):
            if not content:
                return content
            return [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": dict(_EPHEMERAL_CACHE_CONTROL),
                },
            ]

        # Structured list of blocks -> attach to the last block that
        # supports it (text/image/document/tool_result/tool_use all do).
        if isinstance(content, list):
            if not content:
                return content
            new_list = list(content)
            last_block = new_list[-1]
            if isinstance(last_block, dict):
                if "cache_control" not in last_block:
                    last_block = dict(last_block)
                    last_block["cache_control"] = dict(
                        _EPHEMERAL_CACHE_CONTROL,
                    )
                    new_list[-1] = last_block
            return new_list

        # Unknown shape — leave untouched.
        return content


class AnthropicProvider(Provider):
    """Provider implementation for Anthropic API."""

    # Cached AsyncClient for auth_token mode; re-created when auth_mode
    # changes so that the transport is always consistent with the current
    # provider config.
    _strip_http_client: httpx.AsyncClient | None = None
    # Cached AsyncClient for api_key mode that adds request logging without
    # stripping any header.
    _logging_http_client: httpx.AsyncClient | None = None

    def _build_default_headers(self) -> Dict[str, str]:
        return dict(self.custom_headers) if self.custom_headers else {}

    def _get_strip_http_client(self) -> httpx.AsyncClient:
        """Return a cached AsyncClient backed by _StripApiKeyTransport."""
        if self._strip_http_client is None:
            self._strip_http_client = httpx.AsyncClient(
                transport=_StripApiKeyTransport(),
            )
        return self._strip_http_client

    def _get_logging_http_client(self) -> httpx.AsyncClient:
        """Return a cached AsyncClient backed by _LoggingTransport."""
        if self._logging_http_client is None:
            self._logging_http_client = httpx.AsyncClient(
                transport=_LoggingTransport(),
            )
        return self._logging_http_client

    def _client(self, timeout: float = 5) -> anthropic.AsyncAnthropic:
        default_headers = self._build_default_headers()
        if self.auth_mode == "auth_token":
            return anthropic.AsyncAnthropic(
                auth_token=self.api_key,
                base_url=self.base_url,
                default_headers=default_headers,
                http_client=self._get_strip_http_client(),
                timeout=timeout,
            )
        return anthropic.AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=default_headers,
            http_client=self._get_logging_http_client(),
            timeout=timeout,
        )

    @staticmethod
    def _normalize_models_payload(payload: Any) -> List[ModelInfo]:
        if isinstance(payload, dict):
            rows = payload.get("data", [])
        else:
            rows = getattr(payload, "data", payload)

        models: List[ModelInfo] = []
        for row in rows or []:
            model_id = str(
                getattr(row, "id", "") or "",
            ).strip()
            model_name = str(
                getattr(row, "display_name", "") or model_id,
            ).strip()

            if not model_id:
                continue
            models.append(ModelInfo(id=model_id, name=model_name))

        deduped: List[ModelInfo] = []
        seen: set[str] = set()
        for model in models:
            if model.id in seen:
                continue
            seen.add(model.id)
            deduped.append(model)
        return deduped

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        """Check if Anthropic provider is reachable.

        First tries models.list(); if that endpoint is not supported by the
        proxy (e.g. returns 404/405) falls back to a minimal messages.create
        call so that custom proxies that only expose the messages API still
        pass the connection test.
        """
        client = self._client(timeout=timeout)
        try:
            await client.models.list()
            return True, ""
        except anthropic.APIStatusError as e:
            # Some proxies don't implement the models endpoint (404/405).
            # Fall back to a lightweight messages probe instead.
            if e.status_code in (404, 405):
                return await self._check_connection_via_messages(client)
            return False, f"Anthropic API error: {e}"
        except anthropic.APIError as e:
            # Network / auth errors from models.list – report directly
            return False, f"Anthropic API error: {e}"
        except Exception:
            return (
                False,
                f"Unknown exception when connecting to `{self.base_url}`",
            )

    async def _check_connection_via_messages(
        self,
        client: anthropic.AsyncAnthropic,
    ) -> tuple[bool, str]:
        """Fallback: check reachability via messages.create."""
        model = self.models[0].id if self.models else "claude-opus-4-5"
        try:
            await client.messages.create(
                model=model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True, ""
        except anthropic.APIStatusError as e:
            # 400/404/422: server is reachable and auth is accepted –
            # the model may simply not exist on this proxy, which is fine
            # for a connection check.
            if e.status_code in (400, 404, 422):
                return True, ""
            return False, f"Anthropic API error: {e}"
        except anthropic.APIError as e:
            return False, f"Anthropic API error: {e}"
        except Exception as e:
            return False, f"Unknown exception: {e}"

    async def fetch_models(self, timeout: float = 5) -> List[ModelInfo]:
        """Fetch available models."""
        client = self._client(timeout=timeout)
        payload = await client.models.list()
        models = self._normalize_models_payload(payload)
        return models

    async def check_model_connection(
        self,
        model_id: str,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        """Check if a specific model is reachable/usable."""
        target = (model_id or "").strip()
        if not target:
            return False, "Empty model ID"

        body = {
            "model": target,
            "max_tokens": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "ping",
                        },
                    ],
                },
            ],
            "stream": True,
        }
        try:
            client = self._client(timeout=timeout)
            resp = await client.messages.create(**body)
            # consume the stream to ensure the model is actually responsive
            async for _ in resp:
                break
            return True, ""
        except anthropic.APIError:
            return False, f"Model '{model_id}' is not reachable or usable"
        except Exception:
            return (
                False,
                f"Unknown exception when connecting to model '{model_id}'",
            )

    def get_chat_model_instance(self, model_id: str) -> ChatModelBase:
        client_kwargs: Dict[str, Any] = {"base_url": self.base_url}

        # Start with any user-defined custom headers
        merged_headers: Dict[str, str] = self._build_default_headers()

        if self.base_url in DASHSCOPE_BASE_URLS:
            merged_headers["x-dashscope-agentapp"] = json.dumps(
                {
                    "agentType": "QwenPaw",
                    "deployType": "UnKnown",
                    "moduleCode": "model",
                    "agentCode": "UnKnown",
                },
                ensure_ascii=False,
            )
        elif self.base_url in (CODING_DASHSCOPE_BASE_URL, TOKEN_PLAN_BASE_URL):
            merged_headers["X-DashScope-Cdpl"] = json.dumps(
                {
                    "agentType": "QwenPaw",
                    "deployType": "UnKnown",
                    "moduleCode": "model",
                    "agentCode": "UnKnown",
                },
                ensure_ascii=False,
            )

        if merged_headers:
            client_kwargs["default_headers"] = merged_headers

        if self.auth_mode == "auth_token":
            client_kwargs["http_client"] = httpx.AsyncClient(
                transport=_StripApiKeyTransport(),
            )
            client_kwargs["auth_token"] = self.api_key
            api_key_arg = None
        else:
            client_kwargs["http_client"] = httpx.AsyncClient(
                transport=_LoggingTransport(),
            )
            api_key_arg = self.api_key

        effective_generate_kwargs = self.get_effective_generate_kwargs(
            model_id,
        )
        max_tokens = effective_generate_kwargs.pop("max_tokens", 16384)

        return _CachingAnthropicChatModel(
            model_name=model_id,
            max_tokens=max_tokens,
            stream=True,
            api_key=api_key_arg,
            stream_tool_parsing=False,
            client_kwargs=client_kwargs,
            generate_kwargs=effective_generate_kwargs,
        )

    async def probe_model_multimodal(
        self,
        model_id: str,
        timeout: float = 60,
        image_only: bool = False,  # pylint: disable=unused-argument
    ) -> ProbeResult:
        """Probe multimodal support using Anthropic messages API format.

        Anthropic does not support video input, so supports_video is
        always False.  Image support is probed by sending a minimal 1x1
        PNG via the Anthropic base64 image source format.
        """
        img_ok, img_msg = await self._probe_image_support(
            model_id,
            timeout,
        )
        return ProbeResult(
            supports_image=img_ok,
            supports_video=False,
            image_message=img_msg,
            video_message="Video not supported by Anthropic",
        )

    async def _probe_image_support(
        self,
        model_id: str,
        timeout: float = 10,
    ) -> tuple[bool, str]:
        """Probe image support via Anthropic messages API.

        Uses a two-stage check (same strategy as OpenAIProvider):
        1. If the API rejects the request (400 / media-keyword error)
           -> not supported.
        2. If accepted, verify the model can *actually perceive* the
           image by asking for the dominant color of a solid-red PNG.
           Some providers silently accept image payloads without
           processing them, so a pure API-error check would produce
           false positives.
        """
        logger.info(
            "Image probe start: model=%s url=%s",
            model_id,
            self.base_url,
        )
        start_time = time.monotonic()
        client = self._client(timeout=timeout)
        try:
            resp = await client.messages.create(
                model=model_id,
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _PROBE_IMAGE_B64,
                                },
                            },
                            {
                                "type": "text",
                                "text": _IMAGE_PROBE_PROMPT,
                            },
                        ],
                    },
                ],
            )
            answer = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    answer += block.text
            return evaluate_image_probe_answer(
                answer,
                model_id,
                start_time,
            )
        except anthropic.APIError as e:
            elapsed = time.monotonic() - start_time
            logger.warning(
                "Image probe error: model=%s type=%s msg=%s %.2fs",
                model_id,
                type(e).__name__,
                e,
                elapsed,
            )
            status = getattr(e, "status_code", None)
            if status == 400 or _is_media_keyword_error(e):
                return False, f"Image not supported: {e}"
            return False, f"Probe inconclusive: {e}"
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.warning(
                "Image probe error: model=%s type=%s msg=%s %.2fs",
                model_id,
                type(e).__name__,
                e,
                elapsed,
            )
            return False, f"Probe failed: {e}"
