"""Capability probing and caching.

Manages per-model availability status and per-provider health checks.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from bot.config import Settings
from bot.services.registry import ModelRegistry
from providers.errors import ErrorKind, classify, extract_retry_after

logger = logging.getLogger("bot")


# ── capability cache ────────────────────────────────────────────────

class CapabilityCache:
    """TTL-based cache of per-model availability status."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._cache: dict[str, dict[str, Any]] = {}
        self._latency: dict[str, float] = {}  # key -> latency_ms

    def _ttl(self, kind: ErrorKind, error_text: str) -> int:
        base = self._s.model_capability_ttl
        if kind == ErrorKind.AVAILABLE:
            return base
        if kind == ErrorKind.UNSUPPORTED:
            return max(base, 86400)
        if kind == ErrorKind.AUTH:
            return base
        if kind == ErrorKind.QUOTA:
            ra = extract_retry_after(error_text) or 0
            return max(120, min(base, ra + 30 if ra else 900))
        if kind == ErrorKind.TRANSIENT:
            return min(base, 300)
        return min(base, 900)

    def get(self, pid: str, model: str) -> dict[str, Any] | None:
        key = f"{pid}::{model}".lower()
        entry = self._cache.get(key)
        if not entry:
            return None
        if int(entry.get("expires_at", 0)) <= int(time.time()):
            self._cache.pop(key, None)
            return None
        return dict(entry)

    def set(
        self, pid: str, model: str, kind: ErrorKind,
        *, error_text: str = "", latency_ms: float = 0.0,
    ) -> None:
        now = int(time.time())
        ttl = self._ttl(kind, error_text)
        key = f"{pid}::{model}".lower()
        self._cache[key] = {
            "provider_id": pid.lower(),
            "model_name": model,
            "status": kind.value,
            "checked_at": now,
            "expires_at": now + max(60, ttl),
            "error": str(error_text)[:1200],
        }
        if latency_ms > 0 and kind == ErrorKind.AVAILABLE:
            self._latency[key] = latency_ms

    def should_skip(self, pid: str, model: str) -> tuple[bool, str]:
        entry = self.get(pid, model)
        if not entry:
            return False, ""
        status = entry.get("status", "")
        kind = ErrorKind(status) if status in {e.value for e in ErrorKind} else ErrorKind.UNKNOWN
        return kind.is_blocking, status

    def mark_transient(self, pid: str, model: str, reason: str) -> None:
        existing = self.get(pid, model)
        if existing and existing.get("status") == ErrorKind.TRANSIENT.value:
            return
        self.set(pid, model, ErrorKind.TRANSIENT, error_text=reason)

    def clear_transient(self, pid: str) -> int:
        removed = 0
        for key in list(self._cache):
            entry = self._cache[key]
            if entry.get("provider_id") == pid.lower() and entry.get("status") == ErrorKind.TRANSIENT.value:
                del self._cache[key]
                removed += 1
        return removed

    def blocking_count(self) -> int:
        now = int(time.time())
        return sum(
            1 for e in self._cache.values()
            if int(e.get("expires_at", 0)) > now
            and ErrorKind(e["status"]).is_blocking
        )

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        return dict(self._cache)

    def get_latency(self, pid: str, model: str) -> float | None:
        key = f"{pid}::{model}".lower()
        return self._latency.get(key)

    def get_all_latencies(self) -> dict[str, float]:
        return dict(self._latency)


# ── provider availability ──────────────────────────────────────────

class ProviderAvailability:
    """Cache of per-provider health status, probed via model discovery."""

    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self._s = settings
        self._reg = registry
        self._cache: dict[str, dict[str, Any]] = {}

    def get_cached(self, pid: str) -> tuple[bool, str] | None:
        entry = self._cache.get(pid.lower())
        if not entry:
            return None
        if int(entry.get("expires_at", 0)) <= int(time.time()):
            self._cache.pop(pid.lower(), None)
            return None
        return (entry["status"] == "available", str(entry.get("reason", "")))

    def set_cached(self, pid: str, available: bool, reason: str, ttl: int) -> None:
        self._cache[pid.lower()] = {
            "status": "available" if available else "unavailable",
            "reason": str(reason)[:400],
            "checked_at": int(time.time()),
            "expires_at": int(time.time()) + max(10, ttl),
        }

    async def check(self, pid: str, *, force: bool = False) -> tuple[bool, str]:
        pid = pid.lower()
        if not pid:
            return False, "invalid provider id"

        if pid in self._s.non_reprobe:
            self.set_cached(pid, True, "", self._s.provider_available_ttl)
            return True, ""

        if not force:
            cached = self.get_cached(pid)
            if cached is not None:
                return cached

        handler = self._reg.handlers.get(pid)
        if not handler:
            return False, "handler not loaded"

        try:
            models = await handler.discover_models(timeout=5)
            if any(str(x).strip() for x in models):
                self.set_cached(pid, True, "", self._s.provider_available_ttl)
                return True, ""
            self.set_cached(pid, False, "no models returned", self._s.provider_unavailable_ttl)
            return False, "no models returned"
        except Exception as exc:
            reason = f"discovery error: {exc}"
            kind = _probe_category(reason)
            if kind == "unavailable":
                self.set_cached(pid, False, reason, self._s.provider_unavailable_ttl)
                return False, reason
            self.set_cached(pid, True, reason, self._s.provider_available_ttl)
            return True, reason

    @property
    def cache_stats(self) -> tuple[int, int]:
        total = len(self._cache)
        down = sum(1 for v in self._cache.values() if v.get("status") == "unavailable")
        return total, down


# ── helpers ─────────────────────────────────────────────────────────

_RE_THINK = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_RE_REASONING = re.compile(r"<reasoning>[\s\S]*?</reasoning>", re.IGNORECASE)
_RE_REASON_PREFIX = re.compile(r"^\s*Reasoning:\s*", re.IGNORECASE)
_RE_COT_PREFIX = re.compile(r"^\s*Chain of thought:\s*", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    if not text:
        return text
    s = text
    s = _RE_THINK.sub("", s)
    s = _RE_REASONING.sub("", s)
    s = _RE_REASON_PREFIX.sub("", s)
    s = _RE_COT_PREFIX.sub("", s)
    return s.strip() or text


def _probe_category(text: str) -> str:
    t = str(text).lower()
    for m in ("timeout", "timed out", "connection", "service unavailable",
              "bad gateway", "gateway timeout", "internal error", "temporar",
              "500", "502", "503", "504"):
        if m in t:
            return "unavailable"
    return "other"
