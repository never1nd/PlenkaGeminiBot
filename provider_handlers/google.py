from __future__ import annotations

import logging
import threading
from typing import Callable

import google.generativeai as genai

from .base import BaseProviderHandler, HistoryMessage, UsageStats
from .utils import unique_keep_order

logger = logging.getLogger("bot")


def normalize_gemini_model_name(model_name: str) -> str:
    stripped = model_name.strip()
    if stripped.startswith("models/"):
        return stripped.split("/", 1)[1]
    return stripped


def extract_text(response) -> str:
    if not response or not getattr(response, "candidates", None):
        return ""
    content = response.candidates[0].content
    if not content:
        return ""
    parts = getattr(content, "parts", []) or []
    texts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def extract_usage(response) -> UsageStats:
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta is None:
        return {}

    def read_int(source, *keys: str) -> int | None:
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
        if self.keys:
            genai.configure(api_key=self.keys[0])

    def key_count(self) -> int:
        return len(self.keys)

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
            "401",
            "403",
            "permission denied",
            "unauthenticated",
            "invalid api key",
        )
        return any(marker in class_name or marker in text for marker in retryable_markers)

    def discover_models(self, timeout_seconds: int) -> list[str]:
        if not self.keys:
            return list(self.fallback_models)
        for api_key in self._rotated_keys():
            try:
                genai.configure(api_key=api_key)
                names: list[str] = []
                seen: set[str] = set()
                for model in genai.list_models():
                    model_name = normalize_gemini_model_name(str(getattr(model, "name", "")))
                    if not model_name:
                        continue
                    methods = getattr(model, "supported_generation_methods", None) or []
                    method_names = {str(x).strip().lower() for x in methods if str(x).strip()}
                    if "generatecontent" not in method_names:
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
        *,
        max_output_tokens: int,
        timeout_seconds: int,
        strip_reasoning: Callable[[str], str],
    ) -> tuple[str, UsageStats]:
        keys = self._rotated_keys()
        if not keys:
            raise RuntimeError("Google provider API key is not configured.")

        if history:
            lines = ["Conversation history:"]
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    lines.append(f"{role}: {content}")
            lines.append(f"user: {prompt}")
            lines.append("assistant:")
            full_prompt = "\n".join(lines)
        else:
            full_prompt = prompt

        last_error: Exception | None = None
        for api_key in keys:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    full_prompt,
                    generation_config={
                        "max_output_tokens": max_output_tokens,
                    },
                    request_options={"timeout": timeout_seconds},
                )
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
