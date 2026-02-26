from __future__ import annotations

import logging
import threading
from typing import Callable

import requests

from .base import BaseProviderHandler, HistoryMessage, UsageStats
from .utils import build_auth_value, parse_chat_completion_text, parse_chat_completion_usage, unique_keep_order

logger = logging.getLogger("bot")


class OpenAICompatProviderHandler(BaseProviderHandler):
    def __init__(
        self,
        *,
        provider_id: str,
        label: str,
        base_url: str,
        keys: list[str] | None,
        fallback_models: list[str] | None,
        discover_models: bool = True,
        models_path: str = "/models",
        chat_path: str = "/chat/completions",
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
    ) -> None:
        super().__init__(provider_id=provider_id, label=label)
        self.base_url = base_url.strip().rstrip("/")
        self.models_path = (models_path or "/models").strip() or "/models"
        self.chat_path = (chat_path or "/chat/completions").strip() or "/chat/completions"
        self.auth_header = (auth_header or "Authorization").strip() or "Authorization"
        self.auth_prefix = auth_prefix if auth_prefix is not None else "Bearer "
        self.discover_models_enabled = bool(discover_models)
        self.keys = unique_keep_order(keys or [])
        self.fallback_models = unique_keep_order(fallback_models or [])
        self._cursor = 0
        self._lock = threading.Lock()

    def key_count(self) -> int:
        return len(self.keys)

    def _rotated_keys(self) -> list[str]:
        if not self.keys:
            return []
        with self._lock:
            cursor = self._cursor % len(self.keys)
            self._cursor = (cursor + 1) % len(self.keys)
        return self.keys[cursor:] + self.keys[:cursor]

    def _build_headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        auth_value = build_auth_value(api_key, self.auth_prefix)
        if self.auth_header and auth_value:
            headers[self.auth_header] = auth_value
        return headers

    def discover_models(self, timeout_seconds: int) -> list[str]:
        if not self.discover_models_enabled:
            return list(self.fallback_models)
        if not self.base_url or not self.keys:
            return list(self.fallback_models)
        try:
            url = f"{self.base_url}/{self.models_path.lstrip('/')}"
            resp = requests.get(
                url,
                headers=self._build_headers(self.keys[0]),
                timeout=timeout_seconds,
            )
            if not resp.ok:
                logger.warning(
                    "Provider model discovery failed for %s: %s %s",
                    self.provider_id,
                    resp.status_code,
                    resp.text[:200],
                )
                return list(self.fallback_models)
            payload = resp.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                return list(self.fallback_models)
            model_names: list[str] = []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                model_name = str(item.get("id", "")).strip()
                if model_name:
                    model_names.append(model_name)
            return unique_keep_order(model_names) or list(self.fallback_models)
        except Exception as exc:
            logger.warning("Provider model discovery failed for %s: %s", self.provider_id, exc)
            return list(self.fallback_models)

    def generate_text(
        self,
        prompt: str,
        model_name: str,
        history: list[HistoryMessage],
        *,
        max_output_tokens: int,
        timeout_seconds: int,
        strip_reasoning: Callable[[str], str],
    ) -> tuple[str, UsageStats]:
        if not self.base_url:
            raise RuntimeError(f"Provider '{self.provider_id}' base_url is missing.")

        keys = self._rotated_keys()
        if not keys:
            raise RuntimeError(f"No API keys configured for provider: {self.provider_id}")

        messages: list[dict[str, str]] = []
        for msg in history:
            role = msg.get("role", "user")
            if role not in {"user", "assistant"}:
                continue
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        url = f"{self.base_url}/{self.chat_path.lstrip('/')}"
        key_retry_statuses = {401, 403, 429}
        last_error = ""
        for api_key in keys:
            try:
                resp = requests.post(
                    url,
                    headers=self._build_headers(api_key),
                    json=payload,
                    timeout=timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = f"request error: {exc}"
                break

            if resp.ok:
                payload_data = resp.json()
                text = parse_chat_completion_text(payload_data)
                usage = parse_chat_completion_usage(payload_data)
                text = strip_reasoning(text)
                if not text:
                    raise RuntimeError(f"{model_name} returned an empty response.")
                return text, usage

            last_error = f"{resp.status_code} {resp.text[:300]}"
            if resp.status_code not in key_retry_statuses:
                break

        raise RuntimeError(f"Provider '{self.provider_id}' API error: {last_error}")
