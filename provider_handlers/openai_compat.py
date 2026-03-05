from __future__ import annotations

import base64
import logging
import threading
from typing import Callable

import requests
from requests.adapters import HTTPAdapter

from .base import BaseProviderHandler, HistoryMessage, InputAttachment, UsageStats
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
        image_path: str = "/images/generations",
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
    ) -> None:
        super().__init__(provider_id=provider_id, label=label)
        self.base_url = base_url.strip().rstrip("/")
        self.models_path = (models_path or "/models").strip() or "/models"
        self.chat_path = (chat_path or "/chat/completions").strip() or "/chat/completions"
        self.image_path = (image_path or "/images/generations").strip() or "/images/generations"
        self.auth_header = (auth_header or "Authorization").strip() or "Authorization"
        self.auth_prefix = auth_prefix if auth_prefix is not None else "Bearer "
        self.discover_models_enabled = bool(discover_models)
        self.keys = unique_keep_order(keys or [])
        self.fallback_models = unique_keep_order(fallback_models or [])
        self._cursor = 0
        self._lock = threading.Lock()
        self._session_local = threading.local()

    def key_count(self) -> int:
        return len(self.keys)

    def supports_input_attachments(self) -> bool:
        return True

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

    def _get_http_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if isinstance(session, requests.Session):
            return session
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        self._session_local.session = session
        return session

    def get_http_session(self) -> requests.Session:
        return self._get_http_session()

    def discover_models(self, timeout_seconds: int) -> list[str]:
        if not self.discover_models_enabled:
            return list(self.fallback_models)
        if not self.base_url or not self.keys:
            return list(self.fallback_models)
        try:
            session = self._get_http_session()
            url = f"{self.base_url}/{self.models_path.lstrip('/')}"
            resp = session.get(
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
        attachments: list[InputAttachment] | None = None,
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

        messages: list[dict[str, object]] = []
        for msg in history:
            role = msg.get("role", "user")
            if role not in {"user", "assistant"}:
                continue
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        attachment_items = attachments or []
        if not attachment_items:
            messages.append({"role": "user", "content": prompt})
        else:
            content_parts: list[dict[str, object]] = []
            if prompt.strip():
                content_parts.append({"type": "text", "text": prompt})
            usable_attachments = 0
            for item in attachment_items:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind", "")).strip().lower()
                mime_type = str(item.get("mime_type", "")).strip() or "application/octet-stream"
                file_name = str(item.get("file_name", "")).strip() or "attachment"
                raw_bytes = item.get("bytes", b"")
                if isinstance(raw_bytes, bytearray):
                    raw_bytes = bytes(raw_bytes)
                if kind == "image" and isinstance(raw_bytes, bytes) and raw_bytes:
                    image_b64 = base64.b64encode(raw_bytes).decode("ascii")
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}",
                            },
                        }
                    )
                    usable_attachments += 1
                    continue
                text_payload = str(item.get("text", "")).strip()
                if text_payload:
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f"Attached file ({file_name}, {mime_type}):\n{text_payload}",
                        }
                    )
                    usable_attachments += 1

            if usable_attachments == 0:
                raise RuntimeError(
                    f"Provider '{self.provider_id}' does not support the uploaded attachment types for chat."
                )
            if not content_parts:
                content_parts.append({"type": "text", "text": prompt or "Analyze attached content."})
            messages.append({"role": "user", "content": content_parts})

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        url = f"{self.base_url}/{self.chat_path.lstrip('/')}"
        key_retry_statuses = {401, 403, 429}
        last_error = ""
        session = self._get_http_session()
        for api_key in keys:
            try:
                resp = session.post(
                    url,
                    headers=self._build_headers(api_key),
                    json=payload,
                    timeout=timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = f"request error: {exc}"
                logger.warning(
                    "Provider '%s' request failed with current key, trying next key: %s",
                    self.provider_id,
                    exc,
                )
                continue

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

    def generate_image(
        self,
        prompt: str,
        model_name: str,
        *,
        size: str,
        timeout_seconds: int,
    ) -> dict[str, str]:
        if not self.base_url:
            raise RuntimeError(f"Provider '{self.provider_id}' base_url is missing.")

        keys = self._rotated_keys()
        if not keys:
            raise RuntimeError(f"No API keys configured for provider: {self.provider_id}")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "size": size,
        }
        url = f"{self.base_url}/{self.image_path.lstrip('/')}"
        key_retry_statuses = {401, 403, 429}
        last_error = ""
        session = self._get_http_session()
        for api_key in keys:
            try:
                resp = session.post(
                    url,
                    headers=self._build_headers(api_key),
                    json=payload,
                    timeout=timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = f"request error: {exc}"
                logger.warning(
                    "Provider '%s' image request failed with current key, trying next key: %s",
                    self.provider_id,
                    exc,
                )
                continue

            if resp.ok:
                try:
                    payload_data = resp.json()
                except Exception:
                    payload_data = {}
                rows = payload_data.get("data", []) if isinstance(payload_data, dict) else []
                if isinstance(rows, list):
                    for item in rows:
                        if not isinstance(item, dict):
                            continue
                        image_url = str(item.get("url", "")).strip()
                        if image_url:
                            return {"url": image_url}
                        image_b64 = str(item.get("b64_json", "") or item.get("b64", "")).strip()
                        if image_b64:
                            return {"b64_json": image_b64}
                direct_url = str(payload_data.get("url", "")).strip() if isinstance(payload_data, dict) else ""
                if direct_url:
                    return {"url": direct_url}
                raise RuntimeError(f"{model_name} returned no image data.")

            last_error = f"{resp.status_code} {resp.text[:300]}"
            if resp.status_code not in key_retry_statuses:
                break

        raise RuntimeError(f"Provider '{self.provider_id}' API error: {last_error}")
