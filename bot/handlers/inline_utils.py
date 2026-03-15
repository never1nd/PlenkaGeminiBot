from __future__ import annotations

import time
from typing import Any

INLINE_CACHE_KEY = "inline_prompt_cache"
INLINE_CACHE_TTL_SECONDS = 15 * 60


def _purge(cache: dict[str, dict[str, Any]], now: float) -> None:
    expired = [
        k for k, v in cache.items()
        if now - float(v.get("ts", 0)) > INLINE_CACHE_TTL_SECONDS
    ]
    for k in expired:
        cache.pop(k, None)


def store_inline_prompt(context: Any, result_id: str, prompt: str, user_id: int | None) -> None:
    cache = context.application.bot_data.setdefault(INLINE_CACHE_KEY, {})
    now = time.time()
    cache[str(result_id)] = {
        "prompt": str(prompt or ""),
        "ts": now,
        "user_id": int(user_id) if user_id is not None else None,
    }
    _purge(cache, now)


def fetch_inline_prompt(context: Any, result_id: str, user_id: int | None) -> str:
    cache = context.application.bot_data.get(INLINE_CACHE_KEY, {})
    if not isinstance(cache, dict):
        return ""
    now = time.time()
    _purge(cache, now)
    entry = cache.get(str(result_id))
    if not isinstance(entry, dict):
        return ""
    cached_uid = entry.get("user_id")
    if cached_uid is not None and user_id is not None and int(cached_uid) != int(user_id):
        return ""
    return str(entry.get("prompt", "") or "").strip()
