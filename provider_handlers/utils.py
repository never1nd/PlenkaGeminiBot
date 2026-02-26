from __future__ import annotations

from typing import Any


def unique_keep_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = str(value).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        result.append(stripped)
    return result


def build_auth_value(token: str, prefix: str) -> str:
    token = token.strip()
    if not token:
        return ""
    normalized_prefix = prefix or ""
    if normalized_prefix and token.lower().startswith(normalized_prefix.lower()):
        return token
    return f"{normalized_prefix}{token}" if normalized_prefix else token


def parse_chat_completion_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_part = item.get("text")
                if isinstance(text_part, str) and text_part.strip():
                    parts.append(text_part.strip())
        return "\n".join(parts).strip()
    return ""


def parse_chat_completion_usage(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        return {}

    def read_int(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
        return None

    prompt_tokens = read_int("prompt_tokens", "input_tokens")
    completion_tokens = read_int("completion_tokens", "output_tokens")
    total_tokens = read_int("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    result: dict[str, int] = {}
    if prompt_tokens is not None:
        result["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        result["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        result["total_tokens"] = total_tokens
    return result
