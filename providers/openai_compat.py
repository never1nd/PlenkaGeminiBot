"""OpenAI-compatible provider (handles nvidia, void, openrouter, sosikibot, etc.)."""
from __future__ import annotations

import base64
import logging
import threading
from typing import Callable

import httpx

from bot.schemas import (
    HistoryMessage, InputAttachment, ProviderResponse, ImageResponse, UsageStats,
)
from .base import BaseProvider

logger = logging.getLogger("bot")


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _auth_value(token: str, prefix: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if prefix and token.lower().startswith(prefix.lower()):
        return token
    return f"{prefix}{token}" if prefix else token


def _parse_text(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            (p if isinstance(p, str) else p.get("text", ""))
            for p in content
        ).strip()
    return ""


def _parse_usage(payload: dict) -> dict[str, int]:
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = usage.get(key)
        if isinstance(v, (int, float)):
            out[key] = int(v)
    if "total_tokens" not in out and "prompt_tokens" in out and "completion_tokens" in out:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


class OpenAICompatProvider(BaseProvider):
    """Generic OpenAI-compatible API provider.

    Handles: nvidia, void, sosikibot, aithing, and any custom provider.
    Only OpenRouter overrides discover_models for free-pricing filtering.
    """

    def __init__(
        self,
        *,
        provider_id: str,
        label: str,
        base_url: str,
        keys: list[str] | None = None,
        fallback_models: list[str] | None = None,
        discover: bool = True,
        models_path: str = "/models",
        chat_path: str = "/chat/completions",
        image_path: str = "/images/generations",
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
    ) -> None:
        super().__init__(provider_id, label)
        self.base_url = base_url.strip().rstrip("/")
        self.keys = _unique(keys or [])
        self.fallback_models = _unique(fallback_models or [])
        self._discover = discover
        self._models_path = models_path.strip() or "/models"
        self._chat_path = chat_path.strip() or "/chat/completions"
        self._image_path = image_path.strip() or "/images/generations"
        self._auth_header = auth_header.strip() or "Authorization"
        self._auth_prefix = auth_prefix if auth_prefix is not None else "Bearer "
        self._cursor = 0
        self._lock = threading.Lock()
        self._client: httpx.AsyncClient | None = None

    def key_count(self) -> int:
        return len(self.keys)

    def supports_attachments(self) -> bool:
        return True

    def supports_attachment_kind(self, attachment: InputAttachment) -> bool:
        if attachment.kind == "image":
            return bool(attachment.bytes)
        if attachment.text:
            return True
        return False

    # ── internals ───────────────────────────────────────────────────

    def _rotated_keys(self) -> list[str]:
        if not self.keys:
            return []
        with self._lock:
            i = self._cursor % len(self.keys)
            self._cursor = (i + 1) % len(self.keys)
        return self.keys[i:] + self.keys[:i]

    def _headers(self, api_key: str) -> dict[str, str]:
        h: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        val = _auth_value(api_key, self._auth_prefix)
        if self._auth_header and val:
            h[self._auth_header] = val
        return h

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                http2=True,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=64),
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
        return self._client

    # ── discover ────────────────────────────────────────────────────

    async def discover_models(self, timeout: int) -> list[str]:
        if not self._discover or not self.base_url or not self.keys:
            return list(self.fallback_models)
        try:
            url = f"{self.base_url}/{self._models_path.lstrip('/')}"
            resp = await self._http().get(
                url, headers=self._headers(self.keys[0]), timeout=timeout,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Model discovery failed for %s: %s %s",
                    self.provider_id, resp.status_code, resp.text[:200],
                )
                return list(self.fallback_models)
            data = resp.json()
            rows = data.get("data", []) if isinstance(data, dict) else data
            if not isinstance(rows, list):
                return list(self.fallback_models)
            names = [str(r.get("id", "")).strip() for r in rows if isinstance(r, dict)]
            return _unique(names) or list(self.fallback_models)
        except Exception as exc:
            logger.warning("Model discovery failed for %s: %s", self.provider_id, exc)
            return list(self.fallback_models)

    # ── text generation ─────────────────────────────────────────────

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
        if not self.base_url:
            raise RuntimeError(f"Provider '{self.provider_id}' has no base_url.")
        keys = self._rotated_keys()
        if not keys:
            raise RuntimeError(f"No API keys for provider '{self.provider_id}'.")

        # build messages
        messages: list[dict] = []
        for msg in (history or []):
            role = msg.role if isinstance(msg, HistoryMessage) else msg.get("role", "user")
            content = msg.content if isinstance(msg, HistoryMessage) else msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        items = attachments or []
        if not items:
            messages.append({"role": "user", "content": prompt})
        else:
            parts: list[dict] = []
            if prompt.strip():
                parts.append({"type": "text", "text": prompt})
            for att in items:
                a = att if isinstance(att, InputAttachment) else InputAttachment(**att)
                if a.kind == "image" and a.bytes:
                    b64 = base64.b64encode(a.bytes).decode("ascii")
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{a.mime_type};base64,{b64}"},
                    })
                elif a.text:
                    parts.append({
                        "type": "text",
                        "text": f"Attached file ({a.file_name}, {a.mime_type}):\n{a.text}",
                    })
            if not parts:
                parts.append({"type": "text", "text": prompt or "Analyze attached content."})
            messages.append({"role": "user", "content": parts})

        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
        url = f"{self.base_url}/{self._chat_path.lstrip('/')}"
        client = self._http()
        last_err = ""

        for key in keys:
            try:
                resp = await client.post(
                    url, headers=self._headers(key), json=body, timeout=timeout,
                )
            except httpx.RequestError as exc:
                last_err = f"request error: {exc}"
                break

            if resp.status_code == 200:
                data = resp.json()
                text = strip_reasoning(_parse_text(data))
                if not text:
                    raise RuntimeError(f"{model} returned an empty response.")
                return ProviderResponse(text=text, usage=UsageStats(**_parse_usage(data)))

            last_err = f"{resp.status_code} {resp.text[:300]}"
            if resp.status_code not in (401, 403, 429):
                break

        raise RuntimeError(f"Provider '{self.provider_id}' API error: {last_err}")

    # ── image generation ────────────────────────────────────────────

    async def generate_image(
        self,
        prompt: str,
        model: str,
        *,
        size: str,
        timeout: int,
    ) -> ImageResponse:
        if not self.base_url:
            raise RuntimeError(f"Provider '{self.provider_id}' has no base_url.")
        keys = self._rotated_keys()
        if not keys:
            raise RuntimeError(f"No API keys for provider '{self.provider_id}'.")

        body = {"model": model, "prompt": prompt, "size": size}
        url = f"{self.base_url}/{self._image_path.lstrip('/')}"
        client = self._http()
        last_err = ""

        for key in keys:
            try:
                resp = await client.post(
                    url, headers=self._headers(key), json=body, timeout=timeout,
                )
            except httpx.RequestError as exc:
                last_err = f"request error: {exc}"
                break

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                rows = data.get("data", []) if isinstance(data, dict) else []
                if isinstance(rows, list):
                    for item in rows:
                        if not isinstance(item, dict):
                            continue
                        if url_val := str(item.get("url", "")).strip():
                            return ImageResponse(url=url_val)
                        if b64 := str(item.get("b64_json", "") or item.get("b64", "")).strip():
                            return ImageResponse(b64_json=b64)
                if direct := str(data.get("url", "")).strip() if isinstance(data, dict) else "":
                    return ImageResponse(url=direct)
                raise RuntimeError(f"{model} returned no image data.")

            last_err = f"{resp.status_code} {resp.text[:300]}"
            if resp.status_code not in (401, 403, 429):
                break

        raise RuntimeError(f"Provider '{self.provider_id}' API error: {last_err}")


class OpenRouterProvider(OpenAICompatProvider):
    """OpenRouter with free-pricing model filtering."""

    async def discover_models(self, timeout: int) -> list[str]:
        if not self._discover or not self.base_url or not self.keys:
            return list(self.fallback_models)
        try:
            url = f"{self.base_url}/{self._models_path.lstrip('/')}"
            resp = await self._http().get(
                url, headers=self._headers(self.keys[0]), timeout=timeout,
            )
            if resp.status_code != 200:
                return list(self.fallback_models)
            rows = resp.json().get("data", [])
            if not isinstance(rows, list):
                return list(self.fallback_models)

            names: list[str] = []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                pricing = item.get("pricing", {})
                if isinstance(pricing, dict) and self._is_free(pricing):
                    name = str(item.get("id", "")).strip()
                    if name:
                        names.append(name)

            if "openrouter/free" not in names:
                names.insert(0, "openrouter/free")
            return _unique(names) or list(self.fallback_models)
        except Exception as exc:
            logger.warning("Model discovery failed for %s: %s", self.provider_id, exc)
            return list(self.fallback_models)

    @staticmethod
    def _is_free(pricing: dict) -> bool:
        try:
            return (
                float(str(pricing.get("prompt", "1"))) == 0.0
                and float(str(pricing.get("completion", "1"))) == 0.0
            )
        except (TypeError, ValueError):
            return False
