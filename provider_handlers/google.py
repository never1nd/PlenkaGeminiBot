from __future__ import annotations

import logging
import threading
from typing import Any, Callable
from google import genai
from google.genai import types as genai_types
from .base import BaseProviderHandler, HistoryMessage, InputAttachment, UsageStats
from .utils import unique_keep_order

logger = logging.getLogger("bot")


def normalize_gemini_model_name(model_name: str) -> str:
    stripped = model_name.strip()
    if stripped.startswith("models/"):
        return stripped.split("/", 1)[1]
    return stripped


def extract_text(response: Any) -> str:
    if not response:
        return ""
    direct = getattr(response, "text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    first_candidate = candidates[0]
    content = getattr(first_candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts).strip()


def extract_usage(response: Any) -> UsageStats:
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta is None:
        usage_meta = getattr(response, "usageMetadata", None)
    if usage_meta is None:
        return {}

    def read_int(source: Any, *keys: str) -> int | None:
        for key in keys:
            value = None
            if isinstance(source, dict):
                value = source.get(key)
            else:
                value = getattr(source, key, None)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
        return None

    prompt_tokens = read_int(usage_meta, "prompt_token_count", "input_token_count")
    completion_tokens = read_int(usage_meta, "candidates_token_count", "output_token_count")
    total_tokens = read_int(usage_meta, "total_token_count")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    usage: UsageStats = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    return usage


class GoogleGeminiProviderHandler(BaseProviderHandler):
    def __init__(self, *, api_keys: list[str] | None, fallback_models: list[str], label: str = "Google Gemini") -> None:
        super().__init__("google", label)
        self.keys = unique_keep_order(api_keys or [])
        self.fallback_models = unique_keep_order(fallback_models)
        self._cursor = 0
        self._lock = threading.Lock()
        self._client_cache: dict[str, Any] = {}
        self._missing_sdk_logged = False

    def key_count(self) -> int:
        return len(self.keys)

    def supports_input_attachments(self) -> bool:
        return True

    def _log_missing_sdk_once(self) -> None:
        if self._missing_sdk_logged:
            return
        self._missing_sdk_logged = True
        logger.warning(
            "google-genai package is not installed. Install dependency `google-genai` for provider 'google'."
        )

    def _get_client(self, api_key: str):
        if genai is None:
            raise RuntimeError("google-genai package is not installed.")
        with self._lock:
            cached = self._client_cache.get(api_key)
            if cached is not None:
                return cached
            client = genai.Client(api_key=api_key)
            self._client_cache[api_key] = client
            return client

    def _rotated_keys(self) -> list[str]:
        if not self.keys:
            return []
        with self._lock:
            cursor = self._cursor % len(self.keys)
            self._cursor = (cursor + 1) % len(self.keys)
        return self.keys[cursor:] + self.keys[:cursor]

    def primary_key(self) -> str:
        if not self.keys:
            return ""
        return self.keys[0]

    def _supports_generate_content_method(self, model: Any) -> bool:
        methods: set[str] = set()
        for field in ("supported_actions", "supported_generation_methods", "supportedGenerationMethods"):
            raw = getattr(model, field, None)
            if not isinstance(raw, (list, tuple, set)):
                continue
            methods.update(str(x).strip().lower() for x in raw if str(x).strip())
        if not methods:
            return True
        markers = (
            "generatecontent",
            "models.generatecontent",
            "generate_content",
        )
        return any(marker in methods for marker in markers)

    def _is_retryable_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        class_name = exc.__class__.__name__.lower()
        retryable_markers = (
            "resourceexhausted",
            "toomanyrequests",
            "serviceunavailable",
            "deadlineexceeded",
            "internalservererror",
            "unavailable",
            "aborted",
            "timeout",
            "temporar",
            "quota",
            "rate limit",
            "429",
            "503",
            "502",
            "504",
            "500",
            "408",
            "401",
            "403",
            "permission denied",
            "unauthenticated",
            "invalid api key",
        )
        return any(marker in class_name or marker in text for marker in retryable_markers)

    def _build_conversation_prompt(self, prompt: str, history: list[HistoryMessage]) -> str:
        if not history:
            return prompt
        lines = ["Conversation history:"]
        for msg in history:
            role = msg.get("role", "user")
            if role not in {"user", "assistant"}:
                role = "user"
            content = msg.get("content", "")
            if content:
                lines.append(f"{role}: {content}")
        lines.append(f"user: {prompt}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _build_text_part(self, text: str) -> Any:
        if genai_types is not None:
            part_class = getattr(genai_types, "Part", None)
            if part_class is not None and hasattr(part_class, "from_text"):
                try:
                    return part_class.from_text(text=text)
                except Exception:
                    pass
        return text

    def _build_bytes_part(self, raw_bytes: bytes, mime_type: str) -> Any | None:
        if genai_types is None:
            return None
        part_class = getattr(genai_types, "Part", None)
        if part_class is None or not hasattr(part_class, "from_bytes"):
            return None
        try:
            return part_class.from_bytes(data=raw_bytes, mime_type=mime_type)
        except Exception:
            return None

    def _build_request_content(self, prompt: str, attachments: list[InputAttachment]) -> Any:
        if not attachments:
            return prompt

        parts: list[Any] = [self._build_text_part(prompt or "Analyze attached content.")]
        usable_attachments = 0
        for item in attachments:
            if not isinstance(item, dict):
                continue
            mime_type = str(item.get("mime_type", "")).strip() or "application/octet-stream"
            file_name = str(item.get("file_name", "")).strip() or "attachment"
            raw_bytes = item.get("bytes", b"")
            if isinstance(raw_bytes, bytearray):
                raw_bytes = bytes(raw_bytes)
            if isinstance(raw_bytes, bytes) and raw_bytes:
                part = self._build_bytes_part(raw_bytes, mime_type)
                if part is not None:
                    parts.append(part)
                    usable_attachments += 1
                    continue
            text_payload = str(item.get("text", "")).strip()
            if text_payload:
                parts.append(self._build_text_part(f"Attached file ({file_name}, {mime_type}):\n{text_payload}"))
                usable_attachments += 1

        if usable_attachments == 0:
            raise RuntimeError("Google provider could not use the uploaded attachments.")

        if genai_types is not None:
            content_class = getattr(genai_types, "Content", None)
            if content_class is not None:
                try:
                    return [content_class(role="user", parts=parts)]
                except Exception:
                    pass
        return parts

    def discover_models(self, timeout_seconds: int) -> list[str]:
        _ = timeout_seconds
        if not self.keys:
            return list(self.fallback_models)
        if genai is None:
            self._log_missing_sdk_once()
            return list(self.fallback_models)
        for api_key in self._rotated_keys():
            try:
                client = self._get_client(api_key)
                names: list[str] = []
                seen: set[str] = set()
                for model in client.models.list():
                    model_name = normalize_gemini_model_name(str(getattr(model, "name", "")))
                    if not model_name:
                        continue
                    if not self._supports_generate_content_method(model):
                        continue
                    if model_name in seen:
                        continue
                    seen.add(model_name)
                    names.append(model_name)
                names.sort()
                if names:
                    return names
            except Exception as exc:
                logger.warning("Gemini model discovery failed: %s", exc)
                continue
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
        keys = self._rotated_keys()
        if not keys:
            raise RuntimeError("Google provider API key is not configured.")
        if genai is None:
            raise RuntimeError("google-genai package is not installed for provider 'google'.")

        full_prompt = self._build_conversation_prompt(prompt, history)
        request_content = self._build_request_content(full_prompt, attachments or [])

        last_error: Exception | None = None
        for api_key in keys:
            try:
                _ = timeout_seconds
                client = self._get_client(api_key)
                request_kwargs: dict[str, Any] = {
                    "model": model_name,
                    "contents": request_content,
                }
                if genai_types is not None:
                    config_class = getattr(genai_types, "GenerateContentConfig", None)
                    if config_class is not None:
                        request_kwargs["config"] = config_class(max_output_tokens=max_output_tokens)
                response = client.models.generate_content(**request_kwargs)
                text = extract_text(response)
                usage = extract_usage(response)
                if not text:
                    raise RuntimeError(f"{model_name} returned an empty response.")
                return strip_reasoning(text), usage
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_error(exc):
                    break
                continue

        raise RuntimeError(f"Provider 'google' API error: {last_error}")
