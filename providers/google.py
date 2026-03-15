"""Google Gemini provider using the google-genai SDK."""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable

from google import genai
from google.genai import types as genai_types

from bot.schemas import (
    HistoryMessage, InputAttachment, ProviderResponse, ImageResponse, UsageStats,
)
from .base import BaseProvider

logger = logging.getLogger("bot")


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [s for s in (x.strip() for x in items) if s and s not in seen and not seen.add(s)]


def _normalize(name: str) -> str:
    s = name.strip()
    return s.split("/", 1)[1] if s.startswith("models/") else s


def _extract_text(resp: Any) -> str:
    if not resp:
        return ""
    if t := getattr(resp, "text", None):
        if isinstance(t, str) and t.strip():
            return t.strip()
    for cand in getattr(resp, "candidates", None) or []:
        parts = getattr(getattr(cand, "content", None), "parts", None) or []
        texts = [p.text.strip() for p in parts if hasattr(p, "text") and p.text.strip()]
        if texts:
            return "\n".join(texts)
    return ""


def _extract_usage(resp: Any) -> dict[str, int]:
    meta = getattr(resp, "usage_metadata", None) or getattr(resp, "usageMetadata", None)
    if meta is None:
        return {}
    out: dict[str, int] = {}

    def _int(src: Any, *keys: str) -> int | None:
        for k in keys:
            v = getattr(src, k, None) if not isinstance(src, dict) else src.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        return None

    if (pt := _int(meta, "prompt_token_count", "input_token_count")) is not None:
        out["prompt_tokens"] = pt
    if (ct := _int(meta, "candidates_token_count", "output_token_count")) is not None:
        out["completion_tokens"] = ct
    if "prompt_tokens" in out and "completion_tokens" in out:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    elif (tt := _int(meta, "total_token_count")) is not None:
        out["total_tokens"] = tt
    return out


class GoogleProvider(BaseProvider):
    """Google Gemini provider using the official google-genai SDK."""

    def __init__(
        self,
        *,
        keys: list[str] | None = None,
        fallback_models: list[str] | None = None,
        label: str = "Google Gemini",
    ) -> None:
        super().__init__("google", label)
        self.keys = _unique(keys or [])
        self.fallback_models = _unique(fallback_models or [])
        self._cursor = 0
        self._lock = threading.Lock()
        self._clients: dict[str, Any] = {}

    def key_count(self) -> int:
        return len(self.keys)

    def supports_attachments(self) -> bool:
        return True

    # ── internals ───────────────────────────────────────────────────

    def _client(self, key: str):
        with self._lock:
            if key not in self._clients:
                self._clients[key] = genai.Client(api_key=key)
            return self._clients[key]

    def _rotated_keys(self) -> list[str]:
        if not self.keys:
            return []
        with self._lock:
            i = self._cursor % len(self.keys)
            self._cursor = (i + 1) % len(self.keys)
        return self.keys[i:] + self.keys[:i]

    def _supports_generate(self, model: Any) -> bool:
        methods: set[str] = set()
        for field in ("supported_actions", "supported_generation_methods"):
            raw = getattr(model, field, None)
            if isinstance(raw, (list, tuple, set)):
                methods.update(str(x).strip().lower() for x in raw if str(x).strip())
        if not methods:
            return True
        return any(
            m in methods
            for m in ("generatecontent", "models.generatecontent", "generate_content")
        )

    def _is_retryable(self, exc: Exception) -> bool:
        t = str(exc).lower()
        cls = exc.__class__.__name__.lower()
        markers = (
            "resourceexhausted", "toomanyrequests", "serviceunavailable",
            "deadlineexceeded", "timeout", "temporar", "quota", "rate limit",
            "429", "503", "502", "504", "500", "408",
            "401", "403", "permission denied", "unauthenticated", "invalid api key",
        )
        return any(m in cls or m in t for m in markers)

    # ── discover ────────────────────────────────────────────────────

    async def discover_models(self, timeout: int) -> list[str]:
        if not self.keys:
            return list(self.fallback_models)
        for key in self._rotated_keys():
            try:
                client = self._client(key)
                names: list[str] = []
                seen: set[str] = set()
                async for model in await asyncio.wait_for(
                    client.aio.models.list(), timeout=timeout,
                ):
                    name = _normalize(str(getattr(model, "name", "")))
                    if name and name not in seen and self._supports_generate(model):
                        seen.add(name)
                        names.append(name)
                names.sort()
                if names:
                    return names
            except Exception as exc:
                logger.warning("Gemini discovery failed: %s", exc)
        return list(self.fallback_models)

    # ── text generation ─────────────────────────────────────────────

    def _build_prompt(self, prompt: str, history: list[HistoryMessage]) -> str:
        if not history:
            return prompt
        lines = ["Conversation history:"]
        for msg in history:
            role = msg.role if isinstance(msg, HistoryMessage) else msg.get("role", "user")
            content = msg.content if isinstance(msg, HistoryMessage) else msg.get("content", "")
            if content and role in ("user", "assistant"):
                lines.append(f"{role}: {content}")
        lines.append(f"user: {prompt}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _build_content(self, prompt: str, attachments: list[InputAttachment]) -> Any:
        if not attachments:
            return prompt

        parts: list[Any] = []
        Part = getattr(genai_types, "Part", None)

        # text part
        text = prompt or "Analyze attached content."
        if Part and hasattr(Part, "from_text"):
            try:
                parts.append(Part.from_text(text=text))
            except Exception:
                parts.append(text)
        else:
            parts.append(text)

        # attachment parts
        usable = 0
        for att in attachments:
            a = att if isinstance(att, InputAttachment) else InputAttachment(**att)
            if Part and hasattr(Part, "from_bytes") and a.bytes:
                try:
                    parts.append(Part.from_bytes(data=a.bytes, mime_type=a.mime_type))
                    usable += 1
                    continue
                except Exception:
                    pass
            if a.text:
                t = f"Attached file ({a.file_name}, {a.mime_type}):\n{a.text}"
                if Part and hasattr(Part, "from_text"):
                    try:
                        parts.append(Part.from_text(text=t))
                    except Exception:
                        parts.append(t)
                else:
                    parts.append(t)
                usable += 1

        if usable == 0:
            raise RuntimeError("Google provider could not use the uploaded attachments.")

        Content = getattr(genai_types, "Content", None)
        if Content:
            try:
                return [Content(role="user", parts=parts)]
            except Exception:
                pass
        return parts

    async def generate_text(
        self,
        prompt: str,
        model: str,
        history: list[HistoryMessage],
        attachments: list[InputAttachment] | None = None,
        *,
        max_tokens: int,
        timeout: int,
        strip_reasoning: Callable[[str], str],
    ) -> ProviderResponse:
        keys = self._rotated_keys()
        if not keys:
            raise RuntimeError("Google provider: no API key configured.")

        full = self._build_prompt(prompt, history or [])
        content = self._build_content(full, attachments or [])
        last_err: Exception | None = None

        for key in keys:
            try:
                client = self._client(key)
                kwargs: dict[str, Any] = {"model": model, "contents": content}
                Config = getattr(genai_types, "GenerateContentConfig", None)
                if Config:
                    kwargs["config"] = Config(max_output_tokens=max_tokens)
                resp = await asyncio.wait_for(
                    client.aio.models.generate_content(**kwargs),
                    timeout=timeout,
                )
                text = _extract_text(resp)
                if not text:
                    raise RuntimeError(f"{model} returned an empty response.")
                return ProviderResponse(
                    text=strip_reasoning(text),
                    usage=UsageStats(**_extract_usage(resp)),
                )
            except Exception as exc:
                last_err = exc
                if not self._is_retryable(exc):
                    break

        raise RuntimeError(f"Provider 'google' API error: {last_err}")

    # ── image generation ────────────────────────────────────────────

    async def generate_image(
        self,
        prompt: str,
        model: str,
        *,
        size: str,
        timeout: int,
    ) -> ImageResponse:
        raise NotImplementedError("Image generation not supported by Google Gemini provider.")
