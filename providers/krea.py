"""Krea image generation provider (polling job-based API)."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable

import httpx

from bot.schemas import (
    HistoryMessage, InputAttachment, ProviderResponse, ImageResponse,
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


class KreaProvider(BaseProvider):
    """Krea image-only provider.

    Uses job-based API:
    POST /generate/image/{model} -> job_id
    GET /jobs/{job_id} -> result urls
    """

    def __init__(
        self,
        *,
        provider_id: str = "krea",
        label: str = "Krea",
        base_url: str,
        keys: list[str] | None = None,
        fallback_models: list[str] | None = None,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
    ) -> None:
        super().__init__(provider_id, label)
        self.base_url = base_url.strip().rstrip("/")
        self.keys = list(keys or [])
        self.fallback_models = list(fallback_models or [])
        self._auth_header = (auth_header or "Authorization").strip() or "Authorization"
        self._auth_prefix = auth_prefix if auth_prefix is not None else "Bearer "
        self._cursor = 0
        self._lock = threading.Lock()
        self._disabled_keys: set[str] = set()
        self._client: httpx.AsyncClient | None = None

    def key_count(self) -> int:
        return len(self.keys)

    def supports_text(self) -> bool:
        return False

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                http2=True,
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
            )
        return self._client

    def _rotated_keys(self) -> list[str]:
        if not self.keys:
            return []
        with self._lock:
            active = [k for k in self.keys if k not in self._disabled_keys]
            if not active:
                return []
            i = self._cursor % len(active)
            self._cursor = (i + 1) % len(active)
        return active[i:] + active[:i]

    def _disable_key(self, api_key: str) -> None:
        if not api_key:
            return
        with self._lock:
            self._disabled_keys.add(api_key)

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        auth_value = _auth_value(api_key, self._auth_prefix)
        if self._auth_header and auth_value:
            headers[self._auth_header] = auth_value
        return headers

    async def discover_models(self, timeout: int) -> list[str]:
        return list(self.fallback_models)

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
        raise RuntimeError(f"Krea provider does not support text generation (model={model}).")

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
            raise RuntimeError("No working API keys for Krea provider.")
        if not self.base_url:
            raise RuntimeError("Krea provider has no base_url.")

        # parse size like "1024x1024" into width/height
        try:
            w, h = (int(x) for x in size.split("x", 1))
        except (ValueError, TypeError):
            w, h = 1024, 1024

        body = {"prompt": prompt, "width": w, "height": h, "steps": 28}
        create_url = f"{self.base_url}/generate/image/{model}"
        client = self._http()
        last_err = ""

        for key in keys:
            retryable = False
            headers = self._headers(key)
            try:
                resp = await client.post(create_url, headers=headers, json=body, timeout=timeout)
            except httpx.RequestError as exc:
                last_err = f"request error: {exc}"
                break

            if resp.status_code // 100 != 2:
                last_err = f"{resp.status_code} {resp.text[:300]}"
                retryable = resp.status_code in (401, 403, 429)
                if resp.status_code in (401, 403):
                    logger.warning("Disabling Krea API key due to auth error (%s).", resp.status_code)
                    self._disable_key(key)
                if retryable:
                    continue
                break

            try:
                data = resp.json()
            except Exception:
                data = {}

            job_id = data.get("job_id") or data.get("id")
            if not isinstance(job_id, str) or not job_id.strip():
                last_err = "missing job_id in response"
                break

            poll_url = f"{self.base_url}/jobs/{job_id}"
            deadline = time.monotonic() + max(5, timeout)
            poll_headers: dict[str, str] = {}
            auth_val = headers.get(self._auth_header, "")
            if auth_val:
                poll_headers[self._auth_header] = auth_val
            while True:
                if time.monotonic() >= deadline:
                    last_err = "timeout waiting for job completion"
                    retryable = False
                    break
                try:
                    poll = await client.get(
                        poll_url,
                        headers=poll_headers,
                        timeout=min(10, max(1, deadline - time.monotonic())),
                    )
                except httpx.RequestError as exc:
                    last_err = f"request error: {exc}"
                    retryable = False
                    break

                if poll.status_code // 100 != 2:
                    last_err = f"{poll.status_code} {poll.text[:300]}"
                    retryable = poll.status_code in (401, 403, 429)
                    if poll.status_code in (401, 403):
                        logger.warning("Disabling Krea API key due to auth error (%s).", poll.status_code)
                        self._disable_key(key)
                    break

                try:
                    job = poll.json()
                except Exception:
                    job = {}

                status = str(job.get("status", "")).lower()
                completed_at = job.get("completed_at")

                if status in {"failed", "error", "canceled", "cancelled"}:
                    last_err = f"job failed: {status}"
                    retryable = False
                    break

                if status == "completed":
                    result = job.get("result", {}) if isinstance(job, dict) else {}
                    if isinstance(result, dict):
                        urls = result.get("urls")
                        if isinstance(urls, list) and urls:
                            return ImageResponse(url=str(urls[0]))
                        if url := result.get("url"):
                            return ImageResponse(url=str(url))
                    last_err = "completed but no image url returned"
                    retryable = False
                    break

                if completed_at:
                    last_err = f"job failed: {status or 'unknown'}"
                    retryable = False
                    break

                await asyncio.sleep(2)

            if retryable:
                continue
            break

        raise RuntimeError(f"Krea API error: {last_err}")
