"""Cloudflare Workers AI provider — image generation only."""
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

def _auth_value(token: str, prefix: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if prefix and token.lower().startswith(prefix.lower()):
        return token
    return f"{prefix}{token}" if prefix else token


class CloudflareProvider(BaseProvider):
    """Cloudflare Workers AI provider.

    Supports image generation via models like SDXL.
    The API returns raw binary PNG — not OpenAI-compatible JSON.
    """

    def __init__(
        self,
        *,
        provider_id: str = "cloudflare",
        label: str = "Cloudflare AI",
        account_id: str,
        keys: list[str] | None = None,
        fallback_models: list[str] | None = None,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
        auth_email: str = "",
    ) -> None:
        super().__init__(provider_id, label)
        self.account_id = account_id.strip()
        self.keys = list(keys or [])
        self.fallback_models = list(fallback_models or [])
        self._auth_header = (auth_header or "Authorization").strip() or "Authorization"
        self._auth_prefix = auth_prefix if auth_prefix is not None else "Bearer "
        self._auth_email = str(auth_email or "").strip()
        self._cursor = 0
        self._lock = threading.Lock()
        self._client: httpx.AsyncClient | None = None

    def key_count(self) -> int:
        return len(self.keys)

    def supports_text(self) -> bool:
        return False

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                http2=True,
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                ),
            )
        return self._client

    def _rotated_keys(self) -> list[str]:
        if not self.keys:
            return []
        with self._lock:
            i = self._cursor % len(self.keys)
            self._cursor = (i + 1) % len(self.keys)
        return self.keys[i:] + self.keys[:i]

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        auth_value = _auth_value(api_key, self._auth_prefix)
        if self._auth_header and auth_value:
            headers[self._auth_header] = auth_value
        if self._auth_email:
            headers["X-Auth-Email"] = self._auth_email
        return headers

    def _run_url(self, model: str) -> str:
        """Build the Workers AI run URL for a given model."""
        return (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{model}"
        )

    # ── discovery ────────────────────────────────────────────────────

    async def discover_models(self, timeout: int) -> list[str]:
        """Return static fallback models (no dynamic discovery)."""
        return list(self.fallback_models)

    # ── text generation (not supported) ──────────────────────────────

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
        raise RuntimeError(
            f"Cloudflare provider does not support text generation (model={model})."
        )

    # ── image generation ─────────────────────────────────────────────

    async def generate_image(
        self,
        prompt: str,
        model: str,
        *,
        size: str,
        timeout: int,
    ) -> ImageResponse:
        keys = self._rotated_keys()
        if not keys:
            raise RuntimeError("No API keys for Cloudflare provider.")

        # parse size like "1024x1024" into width/height
        try:
            w, h = (int(x) for x in size.split("x", 1))
        except (ValueError, TypeError):
            w, h = 1024, 1024

        body = {"prompt": prompt, "width": w, "height": h, "num_steps": 20}
        url = self._run_url(model)
        client = self._http()
        last_err = ""

        for key in keys:
            headers = self._headers(key)
            try:
                resp = await client.post(
                    url, headers=headers, json=body, timeout=timeout,
                )
            except httpx.RequestError as exc:
                last_err = f"request error: {exc}"
                break

            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "image" in content_type:
                    # raw binary image → base64 encode it
                    b64 = base64.b64encode(resp.content).decode("ascii")
                    return ImageResponse(b64_json=b64)
                # some models may return JSON with result
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        result = data.get("result", {})
                        if isinstance(result, dict):
                            if img := result.get("image"):
                                return ImageResponse(b64_json=str(img))
                except Exception:
                    pass
                raise RuntimeError(f"Cloudflare {model}: unexpected response format.")

            last_err = f"{resp.status_code} {resp.text[:300]}"
            if resp.status_code not in (401, 403, 429):
                break

        raise RuntimeError(f"Cloudflare API error: {last_err}")

    def get_model_display_name(self, model: str) -> str:
        """Shorten @cf/stabilityai/... to a readable name."""
        if "/" in model:
            parts = model.rsplit("/", 1)
            return parts[-1]
        return model
