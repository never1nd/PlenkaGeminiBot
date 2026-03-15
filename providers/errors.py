"""Unified error classification for provider errors.

Replaces the old triple of classify_error / classify_probe_error / is_retryable
with a single classifier and an ErrorKind enum.
"""
from __future__ import annotations

import enum
import math
import re


class ErrorKind(enum.Enum):
    AVAILABLE = "available"           # succeeded or empty response (model works)
    QUOTA = "quota_blocked"           # 429 / rate limit / quota exhausted
    UNSUPPORTED = "unsupported"       # model doesn't exist or doesn't support the operation
    AUTH = "auth_error"               # bad key / 401 / 403
    TRANSIENT = "transient"           # timeout / 5xx / connection error
    UNKNOWN = "unknown"

    @property
    def is_blocking(self) -> bool:
        return self in _BLOCKING

    @property
    def is_retryable(self) -> bool:
        """Should the retry loop continue to the next model?"""
        return self != ErrorKind.AVAILABLE


_BLOCKING = {ErrorKind.QUOTA, ErrorKind.UNSUPPORTED, ErrorKind.AUTH, ErrorKind.TRANSIENT}


def classify(error_text: str) -> ErrorKind:
    """Classify an error string into an ErrorKind."""
    t = str(error_text).lower()

    if "returned an empty response" in t:
        return ErrorKind.AVAILABLE

    for m in (
        "model not found", "unknown model", "unsupported model", "invalid model",
        "invalid_model", "model is not available", "does not support", "does not exist",
        "invalid_argument", "only supports", "404",
    ):
        if m in t:
            return ErrorKind.UNSUPPORTED

    for m in (
        "invalid api key", "unauthorized", "unauthenticated",
        "permission denied", "forbidden", "401", "403",
    ):
        if m in t:
            return ErrorKind.AUTH

    for m in (
        "quota", "rate limit", "resource exhausted", "resource_exhausted",
        "resourceexhausted", "too many requests", "429",
    ):
        if m in t:
            return ErrorKind.QUOTA

    for m in (
        "timeout", "timed out", "connection", "temporar", "internal error",
        "service unavailable", "bad gateway", "gateway timeout",
        "500", "502", "503", "504",
    ):
        if m in t:
            return ErrorKind.TRANSIENT

    return ErrorKind.UNKNOWN


def extract_retry_after(text: str) -> int | None:
    """Try to parse a retry-after value (in seconds) from error text."""
    for pattern in (
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"retry[_\s-]?after[:=\s]+([0-9]+(?:\.[0-9]+)?)",
        r"seconds:\s*([0-9]+)",
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                if v > 0:
                    return int(math.ceil(v))
            except ValueError:
                continue
    return None
