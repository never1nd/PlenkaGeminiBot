import asyncio
import base64
import concurrent.futures
import contextlib
import hashlib
import html
import io
import json
import logging
import math
import mimetypes
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import requests
from dotenv import load_dotenv
from provider_handlers import (
    BaseProviderHandler,
    load_external_provider_handlers,
)
from provider_handlers.base import InputAttachment
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputFile,
    InputTextMessageContent,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    ChosenInlineResultHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

load_dotenv()

ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ENV_TRUE_VALUES


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def parse_user_id_list(raw_values: list[str]) -> set[int]:
    combined = ",".join(value for value in raw_values if value)
    if not combined:
        return set()
    parts = re.split(r"[,\s;]+", combined)
    ids: set[int] = set()
    for part in parts:
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            ids.add(value)
    return ids


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
def parse_owner_user_ids(raw_values: list[str]) -> set[int]:
    combined = ",".join(value for value in raw_values if value)
    if not combined:
        return set()
    parts = re.split(r"[,\s;]+", combined)
    owner_ids: set[int] = set()
    for part in parts:
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            owner_ids.add(value)
    return owner_ids


def load_owner_user_ids() -> set[int]:
    raw_values: list[str] = []
    raw_primary = os.getenv("OWNER_USER_ID", "").strip()
    raw_list = os.getenv("OWNER_USER_IDS", "").strip()
    if raw_primary:
        raw_values.append(raw_primary)
    if raw_list:
        raw_values.append(raw_list)
    if not raw_values:
        raw_values.append("8082486311")
    return parse_owner_user_ids(raw_values)


OWNER_USER_IDS = load_owner_user_ids()
BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME", "").strip().lstrip("@")
GROUP_KEYWORD_TRIGGER = os.getenv("GROUP_KEYWORD_TRIGGER", "\u043a\u043e\u043a\u0441").strip()
GROUP_KEYWORD_REQUIRE_QUESTION = env_bool("GROUP_KEYWORD_REQUIRE_QUESTION", "0")
REPLY_INSTRUCTION_DEFAULT = (
    "Reply in Telegram Markdown. No HTML. Give a complete answer without filler. "
    "Ты - кокс. это твое имя. Отвечай как человек и похуист. пытайся быть не очень борзым"
)
REPLY_INSTRUCTION_SHORT = (
    "Reply in Telegram Markdown. No HTML. Be direct and to the point without filler. "
    "Ты - кокс. это твое имя. Отвечай как человек и похуист. пытайся быть не очень борзым"
)
GROUP_PROVIDER_ALLOWLIST = {
    item.strip().lower()
    for item in os.getenv("GROUP_PROVIDER_ALLOWLIST", "nvidia,sosikibot,openrouter").split(",")
    if item.strip()
}
GROUP_PHOTO_KEYWORD = os.getenv("GROUP_PHOTO_KEYWORD", "\u043a\u043e\u043a\u0441\u0444\u043e\u0442\u043e").strip()
GROUP_FAST_MODEL_NAME = os.getenv("GROUP_FAST_MODEL_NAME", "").strip()
GROUP_FAST_MODEL_PROVIDER = os.getenv("GROUP_FAST_MODEL_PROVIDER", "").strip().lower()
SPECIAL_DENIED_USER_IDS = parse_user_id_list(
    [
        os.getenv("SPECIAL_DENIED_USER_ID", "").strip(),
        os.getenv("SPECIAL_DENIED_USER_IDS", "").strip(),
    ]
)
SPECIAL_DENIED_MESSAGE = os.getenv("SPECIAL_DENIED_MESSAGE", "\u0443 \u043c\u0435\u043d\u044f 1 \u0442\u0440\u0438\u043b\u043b\u0438\u043e\u043d \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432").strip()

MEMORY_CONTEXT_MESSAGES = int(os.getenv("MEMORY_CONTEXT_MESSAGES", "20"))
CUSTOM_PROVIDER_CONFIG_FILE = os.getenv("CUSTOM_PROVIDER_CONFIG_FILE", "providers.json").strip()

REGULAR_MODEL_TIMEOUT_SECONDS = int(os.getenv("REGULAR_MODEL_TIMEOUT_SECONDS", "20"))
REASONING_MODEL_TIMEOUT_SECONDS = int(os.getenv("REASONING_MODEL_TIMEOUT_SECONDS", "60"))
IMAGE_MODEL_TIMEOUT_SECONDS = int(os.getenv("IMAGE_MODEL_TIMEOUT_SECONDS", "50"))
FALLBACK_ATTEMPT_TIMEOUT_SECONDS = int(os.getenv("FALLBACK_ATTEMPT_TIMEOUT_SECONDS", "10"))
MODEL_CAPABILITY_TTL_SECONDS = int(os.getenv("MODEL_CAPABILITY_TTL_SECONDS", "300"))
MODEL_PROBE_TIMEOUT_SECONDS = int(os.getenv("MODEL_PROBE_TIMEOUT_SECONDS", "10"))
MODEL_PROBE_WORKERS = int(os.getenv("MODEL_PROBE_WORKERS", "8"))
MODEL_PROBE_SCOPE = os.getenv("MODEL_PROBE_SCOPE", "smart").strip().lower()
MODEL_PROBE_MAX_MODELS = int(os.getenv("MODEL_PROBE_MAX_MODELS", "0"))
MODEL_CAPABILITY_PROBE_ENABLED = env_bool("MODEL_CAPABILITY_PROBE_ENABLED", "1")
MODEL_CAPABILITY_RECHECK_ENABLED = env_bool("MODEL_CAPABILITY_RECHECK_ENABLED", "1")
MODEL_CAPABILITY_RECHECK_INTERVAL_SECONDS = int(os.getenv("MODEL_CAPABILITY_RECHECK_INTERVAL_SECONDS", "300"))
MODEL_CAPABILITY_RECHECK_INITIAL_DELAY_SECONDS = int(os.getenv("MODEL_CAPABILITY_RECHECK_INITIAL_DELAY_SECONDS", "45"))
MODEL_CAPABILITY_RECHECK_FORCE_FULL = env_bool("MODEL_CAPABILITY_RECHECK_FORCE_FULL", "1")
MODEL_HIDE_UNAVAILABLE_MODELS = env_bool("MODEL_HIDE_UNAVAILABLE_MODELS", "1")
PROVIDER_MODEL_LIST_PROBE_TIMEOUT_SECONDS = int(os.getenv("PROVIDER_MODEL_LIST_PROBE_TIMEOUT_SECONDS", "8"))
PROVIDER_UNAVAILABLE_TTL_SECONDS = max(15, int(os.getenv("PROVIDER_UNAVAILABLE_TTL_SECONDS", "15")))
PROVIDER_AVAILABLE_TTL_SECONDS = max(10, int(os.getenv("PROVIDER_AVAILABLE_TTL_SECONDS", "10")))
PROVIDER_AVAILABILITY_RECHECK_ENABLED = env_bool("PROVIDER_AVAILABILITY_RECHECK_ENABLED", "1")
PROVIDER_AVAILABILITY_RECHECK_INTERVAL_SECONDS = int(os.getenv("PROVIDER_AVAILABILITY_RECHECK_INTERVAL_SECONDS", "90"))
PROVIDER_AVAILABILITY_RECHECK_INITIAL_DELAY_SECONDS = int(os.getenv("PROVIDER_AVAILABILITY_RECHECK_INITIAL_DELAY_SECONDS", "15"))
NON_REPROBE_PROVIDERS = {
    item.strip().lower()
    for item in os.getenv("NON_REPROBE_PROVIDERS", "sosikibot,openrouter").split(",")
    if item.strip()
}

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024") or "1024")
INLINE_MAX_OUTPUT_TOKENS = 250
INLINE_DEBOUNCE_SECONDS = 0.0
INLINE_ALLOWED_MODEL_BASES = (
    "gemini-3.1-flash-lite",
    "gemini-3.0-flash",
    "gemini-2.5-flash",
)
INLINE_PLACEHOLDER_TEXT = "Generating response..."
IMAGE_GENERATION_SIZE = os.getenv("IMAGE_GENERATION_SIZE", "1024x1024").strip() or "1024x1024"
DEFAULT_IMAGE_MODEL_NAME = os.getenv("DEFAULT_IMAGE_MODEL_NAME", "gpt image 1.5").strip()
DEFAULT_IMAGE_MODEL_PROVIDER = os.getenv("DEFAULT_IMAGE_MODEL_PROVIDER", "").strip().lower()
MAX_INPUT_ATTACHMENT_COUNT = max(1, int(os.getenv("MAX_INPUT_ATTACHMENT_COUNT", "3") or "3"))
MAX_INPUT_ATTACHMENT_BYTES = max(64 * 1024, int(os.getenv("MAX_INPUT_ATTACHMENT_BYTES", str(10 * 1024 * 1024)) or str(10 * 1024 * 1024)))
MAX_INPUT_TEXT_ATTACHMENT_BYTES = int(os.getenv("MAX_INPUT_TEXT_ATTACHMENT_BYTES", str(512 * 1024)))
MAX_INPUT_TEXT_ATTACHMENT_CHARS = int(os.getenv("MAX_INPUT_TEXT_ATTACHMENT_CHARS", "20000"))
TELEGRAM_REPLY_CHUNK_CHARS = 4000

PROVIDER_DISPLAY_NAMES: dict[str, str] = {"google": "Google Gemini", "nvidia": "NVIDIA"}
MODEL_CAPABILITIES_LOCK = threading.Lock()
PROVIDER_AVAILABILITY_CACHE: dict[str, dict[str, Any]] = {}
PROVIDER_AVAILABILITY_LOCK = threading.Lock()
PROVIDER_AVAILABILITY_RECHECK_TASK: asyncio.Task[Any] | None = None
MODEL_CAPABILITY_RECHECK_TASK: asyncio.Task[Any] | None = None
MODEL_CAPABILITY_RECHECK_LOCK = threading.Lock()
CHAT_MEMORY_DEFAULT_USER_ID = 0
CHAT_MEMORY_CACHE: dict[tuple[int, int], bool] = {}
CHAT_MODE_CACHE: dict[int, str] = {}
CHAT_SETTINGS_CACHE_LOCK = threading.Lock()
INLINE_PENDING_LOCK = threading.Lock()
INLINE_PENDING_REQUESTS: dict[int, dict[str, object]] = {}

MODEL_CAPABILITY_STATUS_AVAILABLE = "available"
MODEL_CAPABILITY_STATUS_QUOTA_BLOCKED = "quota_blocked"
MODEL_CAPABILITY_STATUS_UNSUPPORTED = "unsupported"
MODEL_CAPABILITY_STATUS_AUTH_ERROR = "auth_error"
MODEL_CAPABILITY_STATUS_TRANSIENT = "transient"
MODEL_CAPABILITY_STATUS_UNKNOWN = "unknown"
MODEL_CAPABILITY_BLOCKING_STATUSES = {
    MODEL_CAPABILITY_STATUS_QUOTA_BLOCKED,
    MODEL_CAPABILITY_STATUS_UNSUPPORTED,
    MODEL_CAPABILITY_STATUS_AUTH_ERROR,
    MODEL_CAPABILITY_STATUS_TRANSIENT,
}
MODEL_CAPABILITY_VALID_STATUSES = MODEL_CAPABILITY_BLOCKING_STATUSES | {
    MODEL_CAPABILITY_STATUS_AVAILABLE,
    MODEL_CAPABILITY_STATUS_UNKNOWN,
}
# Keep capability cache in memory (used by probing/filtering), but do not persist to disk.
MODEL_CAPABILITY_CACHE_ENABLED = True
MODEL_CAPABILITY_LOAD_FROM_FILE = False
MODEL_CAPABILITY_PERSIST_TO_FILE = False

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
)
logger = logging.getLogger("bot")

if not TELEGRAM_BOT_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN is required")

allowlist_path = Path(os.getenv("ALLOWLIST_FILE", "allowlist.json").strip())
users_db_path = Path(os.getenv("USERS_DB_FILE", "users.db").strip())
model_prefs_path = Path(os.getenv("MODEL_PREFS_FILE", "model_prefs.json").strip())
model_capabilities_path = Path(os.getenv("MODEL_CAPABILITIES_FILE", "model_capabilities.json").strip())
SQLITE_BUSY_TIMEOUT_MS = max(1000, int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "5000") or "5000"))
SQLITE_CONNECT_TIMEOUT_SECONDS = max(1.0, SQLITE_BUSY_TIMEOUT_MS / 1000.0)
_DB_LOCAL = threading.local()


def get_db_connection() -> sqlite3.Connection:
    conn = getattr(_DB_LOCAL, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    conn = sqlite3.connect(users_db_path, timeout=SQLITE_CONNECT_TIMEOUT_SECONDS)
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    _DB_LOCAL.conn = conn
    return conn


def is_reasoning_model(model_name: str) -> bool:
    lowered = model_name.lower()
    hints = (
        "thinking",
        "reason",
        "deepseek-r1",
        "qwq",
        "r1-distill",
    )
    return any(hint in lowered for hint in hints)


def get_timeout_for_model(model_name: str) -> int:
    if is_reasoning_model(model_name):
        return REASONING_MODEL_TIMEOUT_SECONDS
    return REGULAR_MODEL_TIMEOUT_SECONDS


def build_provider_model_key(prefix: str, model_name: str) -> str:
    digest = hashlib.sha1(f"{prefix}:{model_name}".encode("utf-8")).hexdigest()[:10]
    return f"{prefix}{digest}"


def build_model_label(model_name: str) -> str:
    max_len = 56
    if len(model_name) <= max_len:
        return model_name
    return f"{model_name[: max_len - 3]}..."


def get_model_key_prefix(provider_id: str) -> str:
    if provider_id == "google":
        return "gm_"
    if provider_id == "nvidia":
        return "nv_"
    return f"cp_{provider_id}_"


def model_capability_key(provider_id: str, model_name: str) -> str:
    provider = str(provider_id).strip().lower()
    model = str(model_name).strip().lower()
    return f"{provider}::{model}"


def load_model_capabilities() -> dict[str, dict[str, Any]]:
    if not MODEL_CAPABILITY_CACHE_ENABLED:
        return {}
    if not MODEL_CAPABILITY_LOAD_FROM_FILE:
        return {}
    if not model_capabilities_path.exists():
        return {}
    try:
        payload = json.loads(model_capabilities_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse model capabilities file %s: %s", model_capabilities_path, exc)
        return {}

    raw = payload.get("capabilities", payload) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        return {}

    now_ts = int(time.time())
    result: dict[str, dict[str, Any]] = {}
    for raw_key, raw_entry in raw.items():
        if not isinstance(raw_entry, dict):
            continue
        status = str(raw_entry.get("status", MODEL_CAPABILITY_STATUS_UNKNOWN)).strip().lower()
        if status not in MODEL_CAPABILITY_VALID_STATUSES:
            status = MODEL_CAPABILITY_STATUS_UNKNOWN
        expires_at = int(raw_entry.get("expires_at", 0) or 0)
        if expires_at and expires_at <= now_ts:
            continue
        provider_id = str(raw_entry.get("provider_id", "")).strip().lower()
        model_name = str(raw_entry.get("model_name", "")).strip()
        if not provider_id or not model_name:
            key_parts = str(raw_key).split("::", 1)
            if len(key_parts) == 2:
                provider_id = provider_id or key_parts[0].strip().lower()
                model_name = model_name or key_parts[1].strip()
        if not provider_id or not model_name:
            continue
        cache_key = model_capability_key(provider_id, model_name)
        result[cache_key] = {
            "provider_id": provider_id,
            "model_name": model_name,
            "status": status,
            "checked_at": int(raw_entry.get("checked_at", now_ts) or now_ts),
            "expires_at": expires_at,
            "error": str(raw_entry.get("error", ""))[:1200],
        }
    if result:
        logger.info("Loaded model capability cache entries: %d", len(result))
    return result


def save_model_capabilities() -> None:
    if not MODEL_CAPABILITY_CACHE_ENABLED:
        return
    if not MODEL_CAPABILITY_PERSIST_TO_FILE:
        return
    with MODEL_CAPABILITIES_LOCK:
        payload = {
            "version": 1,
            "updated_at": int(time.time()),
            "capabilities": MODEL_CAPABILITIES,
        }
        model_capabilities_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_retry_after_seconds(error_text: str) -> int | None:
    text = str(error_text)
    if not text:
        return None
    patterns = [
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"retry[_\s-]?after[:=\s]+([0-9]+(?:\.[0-9]+)?)",
        r"seconds:\s*([0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1))
            if value > 0:
                return int(math.ceil(value))
        except ValueError:
            continue
    return None


def is_empty_response_error(error_text: str) -> bool:
    text = str(error_text).strip().lower()
    if not text:
        return False
    return "returned an empty response" in text


def is_attachment_related_error(error_text: str) -> bool:
    text = str(error_text).strip().lower()
    if not text:
        return False
    markers = (
        "attachment",
        "image_url",
        "input_image",
        "input image",
        "multimodal",
        "vision",
        "file content",
        "does not support image input",
        "unsupported image",
        "unsupported file",
        "invalid image",
        "invalid file",
        "content parts",
    )
    return any(marker in text for marker in markers)


def classify_generation_error(error: str) -> str:
    if is_empty_response_error(error):
        # Empty output still means provider/model endpoint responded and is reachable.
        return MODEL_CAPABILITY_STATUS_AVAILABLE

    text = str(error).lower()
    unsupported_markers = (
        "model not found",
        "not found for account",
        "unknown model",
        "unsupported model",
        "unsupported",
        "invalid model",
        "invalid_model",
        "model is not available",
        "not available for account",
        "does not have access to model",
        "no access to model",
        "not in your current plan",
        "upgrade your plan",
        "does not support /v1/chat/completions",
        "does not support /v1/responses",
        "does not support",
        "does not exist",
        "404",
    )
    if any(marker in text for marker in unsupported_markers):
        return MODEL_CAPABILITY_STATUS_UNSUPPORTED

    auth_markers = (
        "invalid api key",
        "unauthorized",
        "unauthenticated",
        "permission denied",
        "forbidden",
        "401",
        "403",
    )
    if any(marker in text for marker in auth_markers):
        return MODEL_CAPABILITY_STATUS_AUTH_ERROR

    quota_markers = (
        "quota",
        "rate limit",
        "resource exhausted",
        "resource_exhausted",
        "resourceexhausted",
        "too many requests",
        "429",
    )
    if any(marker in text for marker in quota_markers):
        return MODEL_CAPABILITY_STATUS_QUOTA_BLOCKED

    transient_markers = (
        "timeout",
        "timed out",
        "connection",
        "temporar",
        "internal error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "408",
        "500",
        "502",
        "503",
        "504",
    )
    if any(marker in text for marker in transient_markers):
        return MODEL_CAPABILITY_STATUS_TRANSIENT

    return MODEL_CAPABILITY_STATUS_UNKNOWN


def classify_provider_probe_error(error_text: str) -> str:
    text = str(error_text).strip().lower()
    if not text:
        return "unknown"

    unavailable_markers = (
        "timeout",
        "timed out",
        "connection",
        "failed to establish a new connection",
        "max retries exceeded",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "internal error",
        "temporar",
        "request error",
        "read timed out",
        "connect timeout",
        "name or service not known",
        "connection reset",
        "connection aborted",
        "408",
        "500",
        "502",
        "503",
        "504",
    )
    if any(marker in text for marker in unavailable_markers):
        return "unavailable"

    auth_markers = (
        "invalid api key",
        "authentication_error",
        "unauthorized",
        "unauthenticated",
        "permission denied",
        "forbidden",
        "invalid_key",
        "401",
        "403",
    )
    if any(marker in text for marker in auth_markers):
        return "auth"

    quota_markers = ("quota", "rate limit", "too many requests", "429", "resourceexhausted")
    if any(marker in text for marker in quota_markers):
        return "quota"

    model_markers = (
        "model not found",
        "invalid model",
        "invalid_model",
        "unsupported",
        "does not have access to model",
        "not in your current plan",
        "upgrade your plan",
        "404",
    )
    if any(marker in text for marker in model_markers):
        return "model"

    return "unknown"


def build_auth_value(token: str, prefix: str) -> str:
    cleaned = str(token).strip()
    if not cleaned:
        return ""
    normalized_prefix = str(prefix or "")
    if normalized_prefix and cleaned.lower().startswith(normalized_prefix.lower()):
        return cleaned
    return f"{normalized_prefix}{cleaned}" if normalized_prefix else cleaned


def get_cached_provider_availability(provider_id: str) -> tuple[bool, str] | None:
    normalized = str(provider_id).strip().lower()
    if not normalized:
        return None
    now_ts = int(time.time())
    with PROVIDER_AVAILABILITY_LOCK:
        entry = PROVIDER_AVAILABILITY_CACHE.get(normalized)
        if not isinstance(entry, dict):
            return None
        expires_at = int(entry.get("expires_at", 0) or 0)
        if expires_at and expires_at <= now_ts:
            PROVIDER_AVAILABILITY_CACHE.pop(normalized, None)
            return None
        status = str(entry.get("status", "")).strip().lower()
        reason = str(entry.get("reason", ""))
        if status == "available":
            return True, reason
        if status == "unavailable":
            return False, reason
    return None


def set_cached_provider_availability(provider_id: str, available: bool, reason: str, ttl_seconds: int) -> None:
    normalized = str(provider_id).strip().lower()
    if not normalized:
        return
    now_ts = int(time.time())
    with PROVIDER_AVAILABILITY_LOCK:
        PROVIDER_AVAILABILITY_CACHE[normalized] = {
            "status": "available" if available else "unavailable",
            "reason": str(reason or "")[:400],
            "checked_at": now_ts,
            "expires_at": now_ts + max(10, int(ttl_seconds)),
        }


def probe_provider_model_list(provider_id: str, timeout_seconds: int) -> tuple[bool, str]:
    normalized = str(provider_id).strip().lower()
    handler = TEXT_PROVIDER_HANDLERS.get(normalized)
    if not normalized or not handler:
        return False, "provider handler is not loaded"

    if normalized in {"google"}:
        try:
            models = handler.discover_models(timeout_seconds=timeout_seconds)
            model_names = [str(x).strip() for x in models if str(x).strip()]
            if model_names:
                return True, ""
            return False, "model list probe returned no models"
        except Exception as exc:
            return False, f"model list discovery error: {exc}"

    base_url = str(getattr(handler, "base_url", "") or "").strip().rstrip("/")
    models_path = str(getattr(handler, "models_path", "/models") or "/models").strip() or "/models"
    auth_header = str(getattr(handler, "auth_header", "Authorization") or "Authorization").strip() or "Authorization"
    auth_prefix = str(getattr(handler, "auth_prefix", "Bearer ") or "")
    keys = list(getattr(handler, "keys", []) or [])

    if not base_url:
        return False, "missing base_url"
    if not keys:
        return False, "no API keys configured"

    token = str(keys[0]).strip()
    url = f"{base_url}/{models_path.lstrip('/')}"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    auth_value = build_auth_value(token, auth_prefix)
    if auth_header and auth_value:
        headers[auth_header] = auth_value

    try:
        response_getter = requests.get
        session_getter = getattr(handler, "get_http_session", None)
        if callable(session_getter):
            session = session_getter()
            if isinstance(session, requests.Session):
                response_getter = session.get
        response = response_getter(url, headers=headers, timeout=timeout_seconds)
    except requests.RequestException as exc:
        return False, f"model list request error: {exc}"
    except Exception as exc:
        return False, f"model list request setup error: {exc}"
    if not response.ok:
        return False, f"model list HTTP {response.status_code}: {response.text[:220]}"
    try:
        payload = response.json()
    except Exception as exc:
        return False, f"model list invalid JSON: {exc}"
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return False, "model list payload is not a list"
    models: list[str] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        model_name = str(item.get("id", "")).strip()
        if model_name:
            models.append(model_name)
    if models:
        return True, ""
    return False, "model list probe returned no models"


def check_provider_availability(provider_id: str, *, force: bool = False) -> tuple[bool, str]:
    normalized = str(provider_id).strip().lower()
    if not normalized:
        return False, "invalid provider id"

    if normalized in NON_REPROBE_PROVIDERS:
        set_cached_provider_availability(normalized, True, "", PROVIDER_AVAILABLE_TTL_SECONDS)
        return True, ""

    if not force:
        cached = get_cached_provider_availability(normalized)
        if cached is not None:
            return cached

    ok, reason = probe_provider_model_list(normalized, PROVIDER_MODEL_LIST_PROBE_TIMEOUT_SECONDS)
    if ok:
        set_cached_provider_availability(normalized, True, "", PROVIDER_AVAILABLE_TTL_SECONDS)
        return True, ""

    category = classify_provider_probe_error(reason)
    if category == "unavailable":
        set_cached_provider_availability(normalized, False, reason, PROVIDER_UNAVAILABLE_TTL_SECONDS)
        return False, reason

    # Non-availability failures (auth/quota/model/etc) should not globally disable provider retries.
    set_cached_provider_availability(normalized, True, reason, PROVIDER_AVAILABLE_TTL_SECONDS)
    return True, reason


def get_all_model_keys_for_provider(provider_id: str) -> list[str]:
    normalized = str(provider_id).strip().lower()
    if not normalized:
        return []
    return [key for key in ALL_MODEL_ORDER if MODEL_PROVIDER_BY_KEY.get(key, "") == normalized]


def mark_provider_models_transient(provider_id: str, reason: str) -> int:
    if not MODEL_CAPABILITY_CACHE_ENABLED:
        return 0
    normalized = str(provider_id).strip().lower()
    if not normalized:
        return 0
    updated = 0
    for model_key in get_all_model_keys_for_provider(normalized):
        model_name = MODEL_NAME_BY_KEY.get(model_key, "")
        if not model_name:
            continue
        existing = get_model_capability(normalized, model_name)
        current_status = str(existing.get("status", "")).strip().lower() if isinstance(existing, dict) else ""
        if current_status == MODEL_CAPABILITY_STATUS_TRANSIENT:
            continue
        set_model_capability(
            normalized,
            model_name,
            MODEL_CAPABILITY_STATUS_TRANSIENT,
            error_text=reason,
            persist=False,
        )
        updated += 1
    return updated


def clear_provider_transient_capabilities(provider_id: str) -> int:
    if not MODEL_CAPABILITY_CACHE_ENABLED:
        return 0
    normalized = str(provider_id).strip().lower()
    if not normalized:
        return 0
    removed = 0
    with MODEL_CAPABILITIES_LOCK:
        for cache_key, entry in list(MODEL_CAPABILITIES.items()):
            if not isinstance(entry, dict):
                continue
            entry_provider = str(entry.get("provider_id", "")).strip().lower()
            entry_status = str(entry.get("status", "")).strip().lower()
            if entry_provider != normalized:
                continue
            if entry_status != MODEL_CAPABILITY_STATUS_TRANSIENT:
                continue
            MODEL_CAPABILITIES.pop(cache_key, None)
            removed += 1
    return removed


def reconcile_provider_availability_states(*, force: bool = False) -> dict[str, int]:
    provider_ids = sorted(pid for pid in TEXT_PROVIDER_HANDLERS.keys() if pid not in NON_REPROBE_PROVIDERS)
    skipped_non_reprobe = len(TEXT_PROVIDER_HANDLERS) - len(provider_ids)
    marked_transient = 0
    cleared_transient = 0
    unavailable_providers = 0
    available_providers = 0
    changed = False

    for provider_id in provider_ids:
        previous = get_cached_provider_availability(provider_id)
        was_unavailable = previous is not None and not bool(previous[0])
        ok, reason = check_provider_availability(provider_id, force=force)
        if ok:
            available_providers += 1
            if was_unavailable:
                removed = clear_provider_transient_capabilities(provider_id)
                if removed > 0:
                    cleared_transient += removed
                    changed = True
        else:
            unavailable_providers += 1
            updated = mark_provider_models_transient(provider_id, reason)
            if updated > 0:
                marked_transient += updated
                changed = True

    if changed:
        save_model_capabilities()
        apply_model_capability_filter(force=True, strict=False)

    return {
        "providers_total": len(TEXT_PROVIDER_HANDLERS),
        "providers_checked": len(provider_ids),
        "providers_skipped_non_reprobe": skipped_non_reprobe,
        "providers_available": available_providers,
        "providers_unavailable": unavailable_providers,
        "marked_transient": marked_transient,
        "cleared_transient": cleared_transient,
        "changed": 1 if changed else 0,
    }


def get_capability_ttl_seconds(status: str, error_text: str) -> int:
    if status == MODEL_CAPABILITY_STATUS_AVAILABLE:
        return MODEL_CAPABILITY_TTL_SECONDS
    if status == MODEL_CAPABILITY_STATUS_UNSUPPORTED:
        return max(MODEL_CAPABILITY_TTL_SECONDS, 86400)
    if status == MODEL_CAPABILITY_STATUS_AUTH_ERROR:
        return MODEL_CAPABILITY_TTL_SECONDS
    if status == MODEL_CAPABILITY_STATUS_QUOTA_BLOCKED:
        retry_after = extract_retry_after_seconds(error_text) or 0
        return max(120, min(MODEL_CAPABILITY_TTL_SECONDS, retry_after + 30 if retry_after else 900))
    if status == MODEL_CAPABILITY_STATUS_TRANSIENT:
        return min(MODEL_CAPABILITY_TTL_SECONDS, 300)
    return min(MODEL_CAPABILITY_TTL_SECONDS, 900)


def get_model_capability(provider_id: str, model_name: str) -> dict[str, Any] | None:
    if not MODEL_CAPABILITY_CACHE_ENABLED:
        return None
    now_ts = int(time.time())
    cache_key = model_capability_key(provider_id, model_name)
    with MODEL_CAPABILITIES_LOCK:
        entry = MODEL_CAPABILITIES.get(cache_key)
        if not isinstance(entry, dict):
            return None
        expires_at = int(entry.get("expires_at", 0) or 0)
        if expires_at and expires_at <= now_ts:
            del MODEL_CAPABILITIES[cache_key]
            return None
        return dict(entry)


def set_model_capability(
    provider_id: str,
    model_name: str,
    status: str,
    *,
    error_text: str = "",
    persist: bool = True,
) -> None:
    if not MODEL_CAPABILITY_CACHE_ENABLED:
        return
    normalized_status = str(status).strip().lower()
    if normalized_status not in MODEL_CAPABILITY_VALID_STATUSES:
        normalized_status = MODEL_CAPABILITY_STATUS_UNKNOWN
    now_ts = int(time.time())
    ttl_seconds = get_capability_ttl_seconds(normalized_status, error_text)
    cache_key = model_capability_key(provider_id, model_name)
    updated = False
    with MODEL_CAPABILITIES_LOCK:
        existing = MODEL_CAPABILITIES.get(cache_key)
        expires_at = now_ts + max(60, ttl_seconds)
        normalized_error = str(error_text or "")[:1200]
        if isinstance(existing, dict):
            same_status = str(existing.get("status", "")).strip().lower() == normalized_status
            same_error = str(existing.get("error", "")) == normalized_error
            existing_expires_at = int(existing.get("expires_at", 0) or 0)
            refresh_margin = max(60, min(600, ttl_seconds // 3))
            if same_status and same_error and existing_expires_at > now_ts + refresh_margin:
                return
        new_entry = {
            "provider_id": str(provider_id).strip().lower(),
            "model_name": str(model_name).strip(),
            "status": normalized_status,
            "checked_at": now_ts,
            "expires_at": expires_at,
            "error": normalized_error,
        }
        if not isinstance(existing, dict) or any(existing.get(k) != v for k, v in new_entry.items()):
            MODEL_CAPABILITIES[cache_key] = new_entry
            updated = True
        if MODEL_CAPABILITY_PERSIST_TO_FILE and persist and updated:
            payload = {
                "version": 1,
                "updated_at": now_ts,
                "capabilities": MODEL_CAPABILITIES,
            }
            model_capabilities_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def should_skip_model_candidate(provider_id: str, model_name: str) -> tuple[bool, str]:
    if not MODEL_CAPABILITY_CACHE_ENABLED:
        return False, ""
    entry = get_model_capability(provider_id, model_name)
    if not entry:
        return False, ""
    status = str(entry.get("status", "")).strip().lower()
    if status in MODEL_CAPABILITY_BLOCKING_STATUSES:
        return True, status
    return False, status


def build_text_provider_handlers() -> dict[str, BaseProviderHandler]:
    return load_external_provider_handlers(CUSTOM_PROVIDER_CONFIG_FILE)


def build_model_catalog() -> tuple[
    dict[str, str],
    dict[str, BaseProviderHandler],
    dict[str, str],
    list[str],
    dict[str, str],
    dict[str, str],
]:
    labels: dict[str, str] = {}
    provider_handlers = build_text_provider_handlers()
    provider_labels: dict[str, str] = dict(PROVIDER_DISPLAY_NAMES)
    model_provider_by_key: dict[str, str] = {}
    model_name_by_key: dict[str, str] = {}
    order: list[str] = []

    provider_order = [pid for pid in ["google", "nvidia"] if pid in provider_handlers]
    provider_order.extend(sorted(pid for pid in provider_handlers if pid not in {"google", "nvidia"}))

    for provider_id in provider_order:
        handler = provider_handlers[provider_id]
        provider_labels[provider_id] = handler.label
        model_names = handler.discover_models(REGULAR_MODEL_TIMEOUT_SECONDS)
        if model_names:
            logger.info("Loaded %d text models for provider '%s' (%s).", len(model_names), provider_id, handler.label)
        else:
            logger.warning("No text models loaded for provider '%s' (%s).", provider_id, handler.label)

        for model_name in model_names:
            model_name = str(model_name).strip()
            if not model_name:
                continue
            key_base = build_provider_model_key(get_model_key_prefix(provider_id), model_name)
            model_key = key_base
            suffix = 2
            while model_key in labels:
                model_key = f"{key_base}_{suffix}"
                suffix += 1
            display_name = ""
            display_method = getattr(handler, "get_model_display_name", None)
            if callable(display_method):
                try:
                    display_name = str(display_method(model_name)).strip()
                except Exception:
                    display_name = ""
            labels[model_key] = build_model_label(display_name or model_name)
            model_provider_by_key[model_key] = provider_id
            model_name_by_key[model_key] = model_name
            order.append(model_key)

    return (
        labels,
        provider_handlers,
        provider_labels,
        order,
        model_provider_by_key,
        model_name_by_key,
    )


(
    MODEL_LABELS,
    TEXT_PROVIDER_HANDLERS,
    PROVIDER_DISPLAY_NAMES,
    MODEL_ORDER,
    MODEL_PROVIDER_BY_KEY,
    MODEL_NAME_BY_KEY,
) = build_model_catalog()
MODEL_ORDER_SET = set(MODEL_ORDER)


def get_model_label(model_key: str) -> str:
    return MODEL_LABELS.get(model_key, model_key)


def get_model_full_name(model_key: str) -> str:
    return MODEL_NAME_BY_KEY.get(model_key, get_model_label(model_key))


def get_model_provider_id(model_key: str) -> str:
    return MODEL_PROVIDER_BY_KEY.get(model_key, "unknown")


def get_provider_label(provider_id: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider_id, provider_id)


ALL_MODEL_ORDER = list(MODEL_ORDER)
DEFAULT_TEXT_PROVIDER = MODEL_ORDER[0] if MODEL_ORDER else ""


def build_provider_index() -> tuple[list[str], dict[str, list[str]]]:
    mapping: dict[str, list[str]] = {}
    for model_key in MODEL_ORDER:
        provider_id = get_model_provider_id(model_key)
        mapping.setdefault(provider_id, []).append(model_key)
    # Keep providers visible in selection menu even if all of their models are currently filtered/unavailable.
    for provider_id in TEXT_PROVIDER_HANDLERS.keys():
        mapping.setdefault(provider_id, [])
    priority = {"google": 0, "nvidia": 1}
    provider_ids = sorted(mapping.keys(), key=lambda x: (priority.get(x, 2), x))
    return provider_ids, mapping


PROVIDER_ORDER, PROVIDER_MODEL_KEYS = build_provider_index()
MODEL_CAPABILITIES = load_model_capabilities()


def init_users_db() -> None:
    conn = get_db_connection()
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS allowed_users (
                user_id INTEGER PRIMARY KEY,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content_enc TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_settings (
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(chat_id, user_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_mode_settings (
                chat_id INTEGER PRIMARY KEY,
                mode TEXT NOT NULL DEFAULT 'text'
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_history_chat_id_id
            ON chat_history(chat_id, id DESC)
            """
        )
        memory_columns = conn.execute("PRAGMA table_info(memory_settings)").fetchall()
        memory_column_names = {str(row[1]).strip().lower() for row in memory_columns}
        if memory_columns and "user_id" not in memory_column_names:
            conn.execute("ALTER TABLE memory_settings RENAME TO memory_settings_legacy")
            conn.execute(
                """
                CREATE TABLE memory_settings (
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(chat_id, user_id)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO memory_settings(chat_id, user_id, enabled)
                SELECT chat_id, ?, enabled FROM memory_settings_legacy
                """,
                (CHAT_MEMORY_DEFAULT_USER_ID,),
            )
            conn.execute("DROP TABLE memory_settings_legacy")


def add_allowed_user_to_db(user_id: int) -> None:
    conn = get_db_connection()
    with conn:
        conn.execute("INSERT OR IGNORE INTO allowed_users(user_id) VALUES (?)", (user_id,))


def remove_allowed_user_from_db(user_id: int) -> None:
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM allowed_users WHERE user_id = ?", (user_id,))


def load_allowed_user_ids_from_db() -> set[int]:
    conn = get_db_connection()
    rows = conn.execute("SELECT user_id FROM allowed_users").fetchall()
    return {int(row[0]) for row in rows}


def list_allowed_users_from_db() -> list[tuple[int, str]]:
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT user_id, added_at FROM allowed_users ORDER BY added_at DESC, user_id DESC"
    ).fetchall()
    return [(int(row[0]), str(row[1])) for row in rows]


def encrypt_text(plain: str) -> str:
    return plain


def decrypt_text(cipher: str) -> str:
    return cipher


def is_memory_enabled(chat_id: int, user_id: int) -> bool:
    cache_key = (chat_id, user_id)
    with CHAT_SETTINGS_CACHE_LOCK:
        cached = CHAT_MEMORY_CACHE.get(cache_key)
    if cached is not None:
        return bool(cached)

    conn = get_db_connection()
    row = conn.execute(
        "SELECT enabled FROM memory_settings WHERE chat_id = ? AND user_id = ?",
        (chat_id, user_id),
    ).fetchone()
    if row is not None:
        enabled = bool(int(row[0]))
    else:
        default_row = conn.execute(
            "SELECT enabled FROM memory_settings WHERE chat_id = ? AND user_id = ?",
            (chat_id, CHAT_MEMORY_DEFAULT_USER_ID),
        ).fetchone()
        enabled = bool(int(default_row[0])) if default_row is not None else True
    with CHAT_SETTINGS_CACHE_LOCK:
        CHAT_MEMORY_CACHE[cache_key] = enabled
    return enabled


def set_memory_enabled(chat_id: int, user_id: int, enabled: bool) -> None:
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO memory_settings(chat_id, user_id, enabled)
            VALUES(?, ?, ?)
            ON CONFLICT(chat_id, user_id) DO UPDATE SET enabled = excluded.enabled
            """,
            (chat_id, user_id, 1 if enabled else 0),
        )
    with CHAT_SETTINGS_CACHE_LOCK:
        CHAT_MEMORY_CACHE[(chat_id, user_id)] = bool(enabled)


def get_chat_mode(chat_id: int) -> str:
    with CHAT_SETTINGS_CACHE_LOCK:
        cached = CHAT_MODE_CACHE.get(chat_id)
    if cached in {"text", "image"}:
        return cached

    conn = get_db_connection()
    row = conn.execute("SELECT mode FROM chat_mode_settings WHERE chat_id = ?", (chat_id,)).fetchone()
    if not row:
        with CHAT_SETTINGS_CACHE_LOCK:
            CHAT_MODE_CACHE[chat_id] = "text"
        return "text"
    mode = str(row[0]).strip().lower()
    if mode not in {"text", "image"}:
        with CHAT_SETTINGS_CACHE_LOCK:
            CHAT_MODE_CACHE[chat_id] = "text"
        return "text"
    with CHAT_SETTINGS_CACHE_LOCK:
        CHAT_MODE_CACHE[chat_id] = mode
    return mode


def set_chat_mode(chat_id: int, mode: str) -> None:
    normalized = str(mode).strip().lower()
    if normalized not in {"text", "image"}:
        raise ValueError("Invalid chat mode")
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO chat_mode_settings(chat_id, mode)
            VALUES(?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET mode = excluded.mode
            """,
            (chat_id, normalized),
        )
    with CHAT_SETTINGS_CACHE_LOCK:
        CHAT_MODE_CACHE[chat_id] = normalized


def add_history_message(chat_id: int, user_id: int, role: str, content: str) -> None:
    add_history_messages(chat_id, user_id, [(role, content)])


def add_history_messages(chat_id: int, user_id: int, messages: list[tuple[str, str]]) -> None:
    rows: list[tuple[int, int, str, str]] = []
    for role, content in messages:
        if role not in {"user", "assistant"}:
            continue
        enc = encrypt_text(content)
        rows.append((chat_id, user_id, role, enc))
    if not rows:
        return
    conn = get_db_connection()
    with conn:
        conn.executemany(
            "INSERT INTO chat_history(chat_id, user_id, role, content_enc) VALUES (?, ?, ?, ?)",
            rows,
        )


def clear_chat_history(chat_id: int) -> None:
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))


def get_recent_history(chat_id: int, limit: int) -> list[dict[str, str]]:
    safe_limit = max(0, int(limit))
    if safe_limit <= 0:
        return []
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT role, content_enc
        FROM chat_history
        WHERE chat_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (chat_id, safe_limit),
    ).fetchall()
    history: list[dict[str, str]] = []
    for role, content_enc in reversed(rows):
        content = decrypt_text(content_enc)
        if content:
            history.append({"role": str(role), "content": content})
    return history


def migrate_legacy_allowlist_to_db() -> None:
    ids_to_add: list[int] = []
    if allowlist_path.exists():
        try:
            data = json.loads(allowlist_path.read_text(encoding="utf-8"))
            for raw in data.get("user_ids", []):
                try:
                    ids_to_add.append(int(raw))
                except ValueError:
                    logger.warning("Skipping invalid user id in allowlist file: %s", raw)
        except Exception as exc:
            logger.warning("Failed to read allowlist file: %s", exc)

    for uid in ids_to_add:
        add_allowed_user_to_db(uid)


def write_allowlist_backup(ids: set[int]) -> None:
    data = {"user_ids": sorted(ids)}
    allowlist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model_prefs() -> dict[str, str]:
    if not model_prefs_path.exists():
        return {}
    try:
        data = json.loads(model_prefs_path.read_text(encoding="utf-8"))
        prefs = data.get("prefs", {})
        if isinstance(prefs, dict):
            normalized = {}
            for k, v in prefs.items():
                model_key = str(v)
                if model_key in MODEL_ORDER_SET:
                    normalized[str(k)] = model_key
            return normalized
    except Exception as exc:
        logger.warning("Failed to read model prefs file: %s", exc)
    return {}


def save_model_prefs(prefs: dict[str, str]) -> None:
    data = {"prefs": prefs}
    model_prefs_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


init_users_db()
migrate_legacy_allowlist_to_db()
allowed_user_ids = load_allowed_user_ids_from_db()
# Owner is always allowed even if not present in DB.
if OWNER_USER_IDS:
    allowed_user_ids.update(OWNER_USER_IDS)
write_allowlist_backup(allowed_user_ids)
user_model_prefs = load_model_prefs()


def is_allowed(user) -> bool:
    if not user:
        return False
    if user.id in allowed_user_ids:
        return True
    return False


def is_special_denied_user(user) -> bool:
    if not user:
        return False
    if user.id in SPECIAL_DENIED_USER_IDS:
        return True
    username = str(getattr(user, "username", "") or "").strip()
    if username and "дима" in username.casefold():
        return True
    return False


def get_denied_text_for_user(user, default_text: str) -> str:
    if is_special_denied_user(user):
        return ""
    return default_text


def is_owner(user) -> bool:
    if not user:
        return False
    return user.id in OWNER_USER_IDS


def parse_user_id(arg: str) -> Optional[int]:
    arg = arg.strip()
    if not arg:
        return None
    try:
        return int(arg)
    except ValueError:
        return None


def get_user_model_key(user_id: Optional[int]) -> str:
    if user_id is None:
        return DEFAULT_TEXT_PROVIDER
    model_key = user_model_prefs.get(str(user_id), DEFAULT_TEXT_PROVIDER)
    if model_key not in MODEL_ORDER_SET:
        return DEFAULT_TEXT_PROVIDER
    return model_key


def has_user_model_preference(user_id: Optional[int]) -> bool:
    if user_id is None:
        return False
    model_key = user_model_prefs.get(str(user_id), "")
    if not model_key:
        return False
    return model_key in MODEL_ORDER_SET


def set_user_model_key(user_id: int, model_key: str) -> None:
    if model_key not in MODEL_ORDER_SET:
        raise ValueError("Unsupported model key")
    user_model_prefs[str(user_id)] = model_key
    save_model_prefs(user_model_prefs)


def mask_key(value: str) -> str:
    if not value:
        return "missing"
    if len(value) <= 10:
        return "set"
    return f"{value[:8]}...{value[-4:]}"


def strip_reasoning(text: str) -> str:
    if not text:
        return text

    cleaned = text
    # Remove explicit reasoning blocks used by thinking models.
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", cleaned, flags=re.IGNORECASE)

    # Remove common leading markers if model still outputs reasoning inline.
    cleaned = re.sub(r"^\s*Reasoning:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*Chain of thought:\s*", "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip()
    return cleaned or text


def build_model_prompt(prompt: str, *, short: bool = False) -> str:
    base = str(prompt or "").strip()
    if not base:
        base = "Help the user."
    instruction = REPLY_INSTRUCTION_SHORT if short else REPLY_INSTRUCTION_DEFAULT
    if not instruction:
        return base
    return f"{base}\n\n{instruction}"


def build_inline_prompt(prompt: str) -> str:
    base = str(prompt or "").strip()
    if not base:
        base = "Help the user."
    instruction = REPLY_INSTRUCTION_SHORT
    if not instruction:
        return base
    return f"{base}\n\n{instruction}"


def render_markdown_inline_to_html(text: str) -> str:
    raw_text = str(text or "")
    inline_code_tokens: list[str] = []

    def inline_code_replacer(match: re.Match[str]) -> str:
        token = f"@@INLINECODE{len(inline_code_tokens)}@@"
        inline_code_tokens.append(match.group(1))
        return token

    with_tokens = re.sub(r"`([^`\n]+)`", inline_code_replacer, raw_text)
    escaped = html.escape(with_tokens)

    escaped = re.sub(
        r"\[([^\]\n]{1,400})\]\((https?://[^\s)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        escaped,
    )

    style_patterns: list[tuple[str, str]] = [
        (r"\*\*([^\n*]+?)\*\*", "b"),
        (r"(?<!\w)\*([^\n*]+?)\*(?!\w)", "b"),
        (r"__([^\n_]+?)__", "u"),
        (r"(?<!\w)_([^\n_]+?)_(?!\w)", "i"),
        (r"~~([^\n~]+?)~~", "s"),
        (r"(?<!\w)~([^\n~]+?)~(?!\w)", "s"),
        (r"\|\|([^\n|]+?)\|\|", "tg-spoiler"),
    ]
    for pattern, tag in style_patterns:
        escaped = re.sub(pattern, lambda m, tag_name=tag: f"<{tag_name}>{m.group(1)}</{tag_name}>", escaped)

    for idx, code_text in enumerate(inline_code_tokens):
        escaped = escaped.replace(f"@@INLINECODE{idx}@@", f"<code>{html.escape(code_text)}</code>")
    return escaped


def markdown_code_to_telegram_html(text: str) -> str:
    if not text:
        return text

    parts: list[str] = []
    last = 0
    fence_re = re.compile(r"```([a-zA-Z0-9_+\-#.]*)\n([\s\S]*?)```")

    for match in fence_re.finditer(text):
        start, end = match.span()
        before = text[last:start]
        if before:
            parts.append(render_markdown_inline_to_html(before))

        lang = (match.group(1) or "").strip()
        code = match.group(2) or ""
        code_escaped = html.escape(code.rstrip("\n"))
        if lang:
            parts.append(f'<pre><code class="language-{html.escape(lang)}">{code_escaped}</code></pre>')
        else:
            parts.append(f"<pre>{code_escaped}</pre>")
        last = end

    tail = text[last:]
    if tail:
        parts.append(render_markdown_inline_to_html(tail))

    return "".join(parts)


def split_text_for_telegram(text: str, limit: int) -> list[str]:
    normalized = str(text or "")
    max_len = max(128, int(limit))
    if len(normalized) <= max_len:
        return [normalized]

    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > max_len:
        split_at = -1
        for separator in ("\n\n", "\n", " "):
            candidate = remaining.rfind(separator, 0, max_len + 1)
            if candidate > max_len // 3:
                split_at = candidate + len(separator)
                break
        if split_at <= 0:
            split_at = max_len

        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:max_len]
            split_at = len(chunk)
        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks or [normalized[:max_len]]


def trim_inline_text(text: str, limit: int = TELEGRAM_REPLY_CHUNK_CHARS) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    max_len = max(128, int(limit))
    if len(normalized) <= max_len:
        return normalized
    trimmed = normalized[: max_len - 3].rstrip()
    return f"{trimmed}..." if trimmed else normalized[:max_len]


def format_inline_answer(question: str, answer: str) -> str:
    prompt_text = str(question or "").strip()
    if not prompt_text:
        prompt_text = "Help the user."
    answer_text = str(answer or "").strip()
    if not answer_text:
        answer_text = "Empty response."
    return f"\"{prompt_text}\"\n\n---\n\n\"{answer_text}\""


async def edit_inline_message_text(bot, inline_message_id: str, text: str) -> None:
    trimmed = trim_inline_text(text)
    if not trimmed:
        trimmed = "Empty response."
    payload = markdown_code_to_telegram_html(trimmed) if trimmed else ""
    try:
        await bot.edit_message_text(payload, inline_message_id=inline_message_id, parse_mode=ParseMode.HTML)
        return
    except BadRequest as exc:
        reason = str(exc).strip().lower()
        if "message is too long" in reason:
            fallback_trim = trim_inline_text(trimmed, max(128, TELEGRAM_REPLY_CHUNK_CHARS // 2))
            payload = markdown_code_to_telegram_html(fallback_trim) if fallback_trim else ""
            try:
                await bot.edit_message_text(
                    payload,
                    inline_message_id=inline_message_id,
                    parse_mode=ParseMode.HTML,
                )
                return
            except BadRequest:
                trimmed = fallback_trim
        await bot.edit_message_text(trimmed, inline_message_id=inline_message_id)


async def send_reply_text_chunk(message, text: str, *, parse_html: bool = True) -> None:
    chunk_text = str(text or "")
    payload = markdown_code_to_telegram_html(chunk_text) if parse_html else chunk_text
    try:
        if parse_html:
            await message.reply_text(payload, parse_mode=ParseMode.HTML)
        else:
            await message.reply_text(payload)
        return
    except BadRequest as exc:
        reason = str(exc).strip().lower()
        if "message is too long" in reason:
            reduced_limit = max(256, min(len(chunk_text) // 2, TELEGRAM_REPLY_CHUNK_CHARS // 2))
            for part in split_text_for_telegram(chunk_text, reduced_limit):
                await send_reply_text_chunk(message, part, parse_html=parse_html)
            return
        if parse_html:
            await send_reply_text_chunk(message, chunk_text, parse_html=False)
            return
        raise


async def send_reply_text(message, text: str) -> None:
    full_text = str(text or "").strip()
    if not full_text:
        await message.reply_text("Empty response.")
        return
    for chunk in split_text_for_telegram(full_text, TELEGRAM_REPLY_CHUNK_CHARS):
        await send_reply_text_chunk(message, chunk, parse_html=True)


async def send_reply_image(message, image_data: dict[str, str], *, caption: str = "") -> None:
    image_url = str(image_data.get("url", "")).strip()
    if image_url:
        await message.reply_photo(photo=image_url, caption=caption or None)
        return

    image_b64 = str(image_data.get("b64_json", "")).strip()
    if image_b64:
        try:
            decoded = base64.b64decode(image_b64, validate=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode image payload: {exc}") from exc
        if not decoded:
            raise RuntimeError("Image payload is empty after base64 decoding.")
        buffer = io.BytesIO(decoded)
        buffer.name = "generated.png"
        await message.reply_photo(photo=InputFile(buffer, filename=buffer.name), caption=caption or None)
        return

    raise RuntimeError("No image data in provider response.")


def guess_mime_type(file_name: str) -> str:
    guessed, _encoding = mimetypes.guess_type(file_name)
    return str(guessed or "").strip().lower()


def is_text_like_mime_type(mime_type: str, file_name: str) -> bool:
    normalized = str(mime_type or "").strip().lower()
    if normalized.startswith("text/"):
        return True
    if normalized in {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/csv",
    }:
        return True
    extension = Path(file_name or "").suffix.strip().lower()
    return extension in {
        ".txt",
        ".md",
        ".markdown",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".csv",
        ".sql",
        ".log",
        ".ini",
        ".toml",
    }


def extract_text_from_attachment_bytes(raw_bytes: bytes) -> str:
    if not raw_bytes:
        return ""
    if b"\x00" in raw_bytes[:4096]:
        return ""
    decoded = ""
    for encoding in ("utf-8", "utf-16", "cp1251"):
        try:
            decoded = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not decoded:
        return ""
    normalized = decoded.strip()
    if not normalized:
        return ""
    return normalized[:MAX_INPUT_TEXT_ATTACHMENT_CHARS]


def describe_attachment(attachment: InputAttachment) -> str:
    file_name = str(attachment.get("file_name", "attachment")).strip() or "attachment"
    mime_type = str(attachment.get("mime_type", "")).strip() or "application/octet-stream"
    kind = str(attachment.get("kind", "file")).strip().lower() or "file"
    return f"{kind}:{file_name} ({mime_type})"


class AttachmentDecisionRequiredError(RuntimeError):
    def __init__(self, provider_id: str, model_name: str, reason: str) -> None:
        self.provider_id = str(provider_id).strip().lower()
        self.model_name = str(model_name).strip()
        self.reason = str(reason or "").strip()
        message = (
            f"Selected model '{self.provider_id}/{self.model_name}' does not support the given attachment. "
            f"{self.reason}"
        ).strip()
        super().__init__(message)


def build_attachment_decision_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("Proceed Without Attachment", callback_data="attachresolve:without")],
        [InlineKeyboardButton("Proceed With Auto-Model", callback_data="attachresolve:auto")],
        [InlineKeyboardButton("Choose Another Model", callback_data="attachresolve:choose")],
    ]
    return InlineKeyboardMarkup(rows)


def set_pending_attachment_request(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    prompt: str,
    attachments: list[InputAttachment],
    chat_id: int,
) -> None:
    context.user_data["pending_attachment_request"] = {
        "prompt": str(prompt or ""),
        "attachments": list(attachments or []),
        "chat_id": int(chat_id),
    }
    context.user_data["pending_attachment_choose_model"] = False


def get_pending_attachment_request(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any] | None:
    raw = context.user_data.get("pending_attachment_request")
    if not isinstance(raw, dict):
        return None
    prompt = str(raw.get("prompt", ""))
    attachments = raw.get("attachments", [])
    chat_id = raw.get("chat_id")
    if not isinstance(attachments, list):
        return None
    try:
        chat_id_value = int(chat_id)
    except Exception:
        return None
    return {
        "prompt": prompt,
        "attachments": attachments,
        "chat_id": chat_id_value,
    }


def clear_pending_attachment_request(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("pending_attachment_request", None)
    context.user_data.pop("pending_attachment_choose_model", None)


async def extract_message_attachments(message) -> tuple[list[InputAttachment], list[str]]:
    attachments: list[InputAttachment] = []
    notices: list[str] = []

    async def download_file_payload(file_ref, *, max_bytes: int) -> bytes:
        remote_file = await file_ref.get_file()
        payload = await remote_file.download_as_bytearray()
        raw = bytes(payload or b"")
        if len(raw) > max_bytes:
            raise RuntimeError(f"attachment exceeds {max_bytes} bytes")
        return raw

    if message.photo and len(attachments) < MAX_INPUT_ATTACHMENT_COUNT:
        photo = message.photo[-1]
        file_size = int(getattr(photo, "file_size", 0) or 0)
        if file_size > MAX_INPUT_ATTACHMENT_BYTES:
            notices.append(
                f"Photo skipped: size {file_size} bytes exceeds limit {MAX_INPUT_ATTACHMENT_BYTES}."
            )
        else:
            try:
                raw = await download_file_payload(photo, max_bytes=MAX_INPUT_ATTACHMENT_BYTES)
                if raw:
                    attachments.append(
                        {
                            "kind": "image",
                            "mime_type": "image/jpeg",
                            "file_name": "photo.jpg",
                            "bytes": raw,
                        }
                    )
            except Exception as exc:
                notices.append(f"Photo download failed: {exc}")

    document = getattr(message, "document", None)
    if document is not None and len(attachments) >= MAX_INPUT_ATTACHMENT_COUNT:
        notices.append(f"Only first {MAX_INPUT_ATTACHMENT_COUNT} attachments are processed.")
    elif document is not None:
        file_name = str(getattr(document, "file_name", "") or "document").strip() or "document"
        mime_type = str(getattr(document, "mime_type", "") or "").strip().lower() or guess_mime_type(file_name)
        file_size = int(getattr(document, "file_size", 0) or 0)
        if file_size > MAX_INPUT_ATTACHMENT_BYTES:
            notices.append(
                f"File '{file_name}' skipped: size {file_size} bytes exceeds limit {MAX_INPUT_ATTACHMENT_BYTES}."
            )
        else:
            try:
                raw = await download_file_payload(document, max_bytes=MAX_INPUT_ATTACHMENT_BYTES)
                if not raw:
                    notices.append(f"File '{file_name}' is empty.")
                else:
                    kind = "image" if mime_type.startswith("image/") else "file"
                    item: InputAttachment = {
                        "kind": kind,
                        "mime_type": mime_type or "application/octet-stream",
                        "file_name": file_name,
                        "bytes": raw,
                    }
                    if kind == "file" and len(raw) <= MAX_INPUT_TEXT_ATTACHMENT_BYTES and is_text_like_mime_type(
                        mime_type,
                        file_name,
                    ):
                        extracted_text = extract_text_from_attachment_bytes(raw)
                        if extracted_text:
                            item["text"] = extracted_text
                    attachments.append(item)
            except Exception as exc:
                notices.append(f"File '{file_name}' download failed: {exc}")

    return attachments, notices


def generate_text_with_handler(
    provider_id: str,
    prompt: str,
    model_name: str,
    history: list[dict[str, str]],
    attachments: list[InputAttachment] | None = None,
    *,
    timeout_seconds: int | None = None,
    max_output_tokens: int | None = None,
) -> Tuple[str, str, dict[str, int]]:
    handler = TEXT_PROVIDER_HANDLERS.get(provider_id)
    if not handler:
        raise RuntimeError(f"Provider handler is not loaded: {provider_id}")
    effective_timeout = timeout_seconds if timeout_seconds is not None else get_timeout_for_model(model_name)
    effective_max_output_tokens = MAX_OUTPUT_TOKENS if max_output_tokens is None else max(1, int(max_output_tokens))
    text, usage = handler.generate_text(
        prompt,
        model_name,
        history,
        attachments,
        max_output_tokens=effective_max_output_tokens,
        timeout_seconds=effective_timeout,
        strip_reasoning=strip_reasoning,
    )
    if not text:
        raise RuntimeError(f"{model_name} returned an empty response.")
    return text, model_name, usage


def generate_image_with_handler(
    provider_id: str,
    prompt: str,
    model_name: str,
    *,
    timeout_seconds: int | None = None,
) -> tuple[dict[str, str], str]:
    handler = TEXT_PROVIDER_HANDLERS.get(provider_id)
    if not handler:
        raise RuntimeError(f"Provider handler is not loaded: {provider_id}")
    effective_timeout = timeout_seconds if timeout_seconds is not None else get_timeout_for_model(model_name)
    image_data = handler.generate_image(
        prompt,
        model_name,
        size=IMAGE_GENERATION_SIZE,
        timeout_seconds=effective_timeout,
    )
    if not isinstance(image_data, dict):
        raise RuntimeError(f"{model_name} returned invalid image response.")
    if not str(image_data.get("url", "")).strip() and not str(image_data.get("b64_json", "")).strip():
        raise RuntimeError(f"{model_name} returned no image data.")
    return image_data, model_name


def is_retryable_generation_error(exc: Exception) -> bool:
    text = str(exc).lower()
    hard_stop_markers = (
        "safety",
        "blocked due to safety",
        "blocked prompt",
        "content policy",
    )
    if any(marker in text for marker in hard_stop_markers):
        return False

    retryable_markers = (
        "quota",
        "resourceexhausted",
        "resource_exhausted",
        "too many requests",
        "rate limit",
        "retry",
        "timeout",
        "timed out",
        "connection",
        "temporar",
        "service unavailable",
        "unavailable",
        "internal error",
        "401",
        "403",
        "408",
        "409",
        "423",
        "425",
        "429",
        "500",
        "502",
        "503",
        "504",
        "unauthorized",
        "permission denied",
        "invalid api key",
        "model not found",
        "not found",
        "404",
        "not configured",
        "unsupported",
        "invalid argument",
    )
    if any(marker in text for marker in retryable_markers):
        return True

    # Unknown provider errors should still continue to the next fallback candidate.
    return True


def find_provider_model_key(provider_id: str, model_name: str) -> str:
    provider_keys = PROVIDER_MODEL_KEYS.get(provider_id, [])
    if not provider_keys:
        return ""

    # Exact match first.
    for model_key in provider_keys:
        if MODEL_NAME_BY_KEY.get(model_key, "") == model_name:
            return model_key

    # Case-insensitive exact match.
    lowered = model_name.strip().lower()
    if lowered:
        for model_key in provider_keys:
            if MODEL_NAME_BY_KEY.get(model_key, "").strip().lower() == lowered:
                return model_key

    return ""


def normalize_model_name_for_match(model_name: str) -> str:
    lowered = str(model_name or "").strip().lower()
    if not lowered:
        return ""
    return re.sub(r"[\s_]+", "-", lowered)


def build_default_image_model_variants(model_name: str) -> list[str]:
    raw = str(model_name or "").strip()
    if not raw:
        return []
    variants = [raw]
    normalized = normalize_model_name_for_match(raw)
    if normalized and normalized != raw:
        variants.append(normalized)
    return variants


def find_model_key_by_name(model_name: str, *, provider_id: str = "") -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return ""
    normalized = normalize_model_name_for_match(raw)
    provider_id = str(provider_id or "").strip().lower()
    provider_ids = [provider_id] if provider_id else PROVIDER_ORDER

    # Exact match first (case-insensitive).
    lowered = raw.lower()
    for provider in provider_ids:
        for model_key in PROVIDER_MODEL_KEYS.get(provider, []):
            if MODEL_NAME_BY_KEY.get(model_key, "").strip().lower() == lowered:
                return model_key

    # Normalized match (treat spaces/underscores as hyphens).
    if normalized:
        for provider in provider_ids:
            for model_key in PROVIDER_MODEL_KEYS.get(provider, []):
                candidate = MODEL_NAME_BY_KEY.get(model_key, "")
                if normalize_model_name_for_match(candidate) == normalized:
                    return model_key

    return ""


def get_default_image_model_key() -> str:
    if not DEFAULT_IMAGE_MODEL_NAME:
        return ""
    if DEFAULT_IMAGE_MODEL_PROVIDER:
        key = find_model_key_by_name(DEFAULT_IMAGE_MODEL_NAME, provider_id=DEFAULT_IMAGE_MODEL_PROVIDER)
        if key:
            return key
    return find_model_key_by_name(DEFAULT_IMAGE_MODEL_NAME)


def score_fast_model_name(model_name: str) -> int:
    lowered = str(model_name or "").strip().lower()
    if not lowered:
        return -10_000
    score = 0
    if is_reasoning_model(model_name):
        score -= 80
    fast_hints = (
        ("flash", 80),
        ("lite", 60),
        ("mini", 50),
        ("nano", 40),
        ("haiku", 35),
        ("turbo", 30),
        ("fast", 25),
        ("insta", 20),
    )
    slow_hints = (
        ("thinking", -80),
        ("reason", -70),
        ("r1", -60),
        ("deepseek-r1", -90),
        ("qwq", -50),
        ("pro", -25),
        ("ultra", -40),
        ("max", -20),
        ("large", -20),
        ("xl", -15),
        ("mega", -20),
    )
    for token, weight in fast_hints:
        if token in lowered:
            score += weight
    for token, weight in slow_hints:
        if token in lowered:
            score += weight
    normalized_inline = normalize_inline_model_name(model_name)
    for base in INLINE_ALLOWED_MODEL_BASES:
        if normalized_inline == base or normalized_inline.startswith(f"{base}-"):
            score += 120
            break
    return score


def get_group_fast_model_key(provider_allowlist: set[str] | None = None) -> str:
    normalized_allowlist: set[str] | None = None
    if provider_allowlist is not None:
        normalized_allowlist = {item.strip().lower() for item in provider_allowlist if str(item).strip()}
        if not normalized_allowlist:
            return ""

    if GROUP_FAST_MODEL_NAME:
        preferred = find_model_key_by_name(GROUP_FAST_MODEL_NAME, provider_id=GROUP_FAST_MODEL_PROVIDER)
        if preferred:
            provider_id = MODEL_PROVIDER_BY_KEY.get(preferred, "")
            if not normalized_allowlist or provider_id in normalized_allowlist:
                return preferred
        logger.warning(
            "Group fast model '%s' not found or not allowed; falling back to heuristic selection.",
            GROUP_FAST_MODEL_NAME,
        )

    best_key = ""
    best_score = -10_000
    for model_key in MODEL_ORDER:
        provider_id = MODEL_PROVIDER_BY_KEY.get(model_key, "")
        model_name = MODEL_NAME_BY_KEY.get(model_key, "")
        if not provider_id or not model_name:
            continue
        if normalized_allowlist and provider_id not in normalized_allowlist:
            continue
        score = score_fast_model_name(model_name)
        if score > best_score:
            best_score = score
            best_key = model_key
    return best_key


def build_generation_model_keys(primary_key: str) -> list[str]:
    primary_provider = MODEL_PROVIDER_BY_KEY.get(primary_key, "")
    primary_model_name = MODEL_NAME_BY_KEY.get(primary_key, "")
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(model_key: str) -> None:
        if not model_key or model_key in seen:
            return
        if model_key not in MODEL_ORDER_SET:
            return
        seen.add(model_key)
        candidates.append(model_key)

    # Selected API/provider goes first.
    add_candidate(primary_key)

    # Then try all APIs from the selected provider.
    for model_key in PROVIDER_MODEL_KEYS.get(primary_provider, []):
        add_candidate(model_key)

    # Then try the same API/model name on other providers.
    if primary_model_name:
        for provider_id in PROVIDER_ORDER:
            if provider_id == primary_provider:
                continue
            add_candidate(find_provider_model_key(provider_id, primary_model_name))

    # Finally scroll through every other available API/model.
    for model_key in MODEL_ORDER:
        add_candidate(model_key)

    return candidates


def is_likely_image_model(model_name: str) -> bool:
    lowered = str(model_name).strip().lower()
    if not lowered:
        return False
    markers = (
        "image",
        "imagen",
        "imagine",
        "midjourney",
        "flux",
        "recraft",
        "dall",
        "stable-diffusion",
        "sdxl",
        "kontext",
    )
    return any(marker in lowered for marker in markers)


def build_image_generation_model_keys(primary_key: str) -> list[str]:
    primary_provider = MODEL_PROVIDER_BY_KEY.get(primary_key, "")
    primary_model_name = MODEL_NAME_BY_KEY.get(primary_key, "")
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(model_key: str) -> None:
        if not model_key or model_key in seen:
            return
        if model_key not in MODEL_ORDER_SET:
            return
        seen.add(model_key)
        candidates.append(model_key)

    # User-selected API/model goes first even if not image-capable.
    add_candidate(primary_key)

    # Then image-like models from selected provider.
    for model_key in PROVIDER_MODEL_KEYS.get(primary_provider, []):
        model_name = MODEL_NAME_BY_KEY.get(model_key, "")
        if is_likely_image_model(model_name):
            add_candidate(model_key)

    # Then same model name on other providers.
    if primary_model_name:
        for provider_id in PROVIDER_ORDER:
            if provider_id == primary_provider:
                continue
            add_candidate(find_provider_model_key(provider_id, primary_model_name))

    # Finally image-like models across all providers.
    for model_key in MODEL_ORDER:
        model_name = MODEL_NAME_BY_KEY.get(model_key, "")
        if is_likely_image_model(model_name):
            add_candidate(model_key)

    return candidates


def normalize_inline_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower()
    if normalized.startswith("models/"):
        normalized = normalized.split("/", 1)[1]
    return normalized


def build_inline_model_keys() -> list[str]:
    if not MODEL_ORDER:
        return []
    buckets: dict[str, list[str]] = {base: [] for base in INLINE_ALLOWED_MODEL_BASES}
    for model_key in MODEL_ORDER:
        model_name = MODEL_NAME_BY_KEY.get(model_key, "")
        if not model_name:
            continue
        normalized = normalize_inline_model_name(model_name)
        for base in INLINE_ALLOWED_MODEL_BASES:
            if normalized == base or normalized.startswith(f"{base}-"):
                buckets[base].append(model_key)
                break
    ordered: list[str] = []
    seen: set[str] = set()
    for base in INLINE_ALLOWED_MODEL_BASES:
        for model_key in buckets[base]:
            if model_key in seen or model_key not in MODEL_ORDER_SET:
                continue
            seen.add(model_key)
            ordered.append(model_key)
    return ordered


def build_model_probe_keys() -> list[str]:
    scope = MODEL_PROBE_SCOPE if MODEL_PROBE_SCOPE in {"all", "smart"} else "smart"
    if scope == "all":
        keys = [
            key
            for key in (ALL_MODEL_ORDER or MODEL_ORDER)
            if MODEL_PROVIDER_BY_KEY.get(key, "") not in NON_REPROBE_PROVIDERS
        ]
        if MODEL_PROBE_MAX_MODELS > 0:
            return keys[:MODEL_PROBE_MAX_MODELS]
        return keys

    result: list[str] = []
    seen: set[str] = set()

    # Probe each provider fallback model list as it is commonly used during retries.
    for provider_id, handler in TEXT_PROVIDER_HANDLERS.items():
        if provider_id in NON_REPROBE_PROVIDERS:
            continue
        fallback_models = getattr(handler, "fallback_models", []) or []
        if not isinstance(fallback_models, list):
            continue
        for model_name in fallback_models:
            model_key = find_provider_model_key(provider_id, str(model_name))
            if model_key and model_key not in seen:
                seen.add(model_key)
                result.append(model_key)

    if (
        DEFAULT_TEXT_PROVIDER
        and DEFAULT_TEXT_PROVIDER not in seen
        and MODEL_PROVIDER_BY_KEY.get(DEFAULT_TEXT_PROVIDER, "") not in NON_REPROBE_PROVIDERS
    ):
        seen.add(DEFAULT_TEXT_PROVIDER)
        result.append(DEFAULT_TEXT_PROVIDER)

    if not result:
        result = [
            key
            for key in (ALL_MODEL_ORDER or MODEL_ORDER)
            if MODEL_PROVIDER_BY_KEY.get(key, "") not in NON_REPROBE_PROVIDERS
        ]

    if MODEL_PROBE_MAX_MODELS > 0:
        result = result[:MODEL_PROBE_MAX_MODELS]
    return result


def probe_single_model_capability(model_key: str) -> tuple[str, str, str]:
    provider_id = MODEL_PROVIDER_BY_KEY.get(model_key, "")
    model_name = MODEL_NAME_BY_KEY.get(model_key, "")
    if provider_id in NON_REPROBE_PROVIDERS:
        return MODEL_CAPABILITY_STATUS_UNKNOWN, provider_id, model_name
    handler = TEXT_PROVIDER_HANDLERS.get(provider_id)
    if not provider_id or not model_name or not handler:
        return MODEL_CAPABILITY_STATUS_UNKNOWN, provider_id, model_name

    try:
        text, _usage = handler.generate_text(
            "Reply with exactly: ok",
            model_name,
            [],
            max_output_tokens=8,
            timeout_seconds=MODEL_PROBE_TIMEOUT_SECONDS,
            strip_reasoning=strip_reasoning,
        )
        if text.strip():
            set_model_capability(provider_id, model_name, MODEL_CAPABILITY_STATUS_AVAILABLE, persist=False)
            return MODEL_CAPABILITY_STATUS_AVAILABLE, provider_id, model_name
        set_model_capability(
            provider_id,
            model_name,
            MODEL_CAPABILITY_STATUS_UNKNOWN,
            error_text="empty probe response",
            persist=False,
        )
        return MODEL_CAPABILITY_STATUS_UNKNOWN, provider_id, model_name
    except Exception as exc:
        error_text = str(exc)
        status = classify_generation_error(error_text)
        cache_error = "" if status == MODEL_CAPABILITY_STATUS_AVAILABLE else error_text
        set_model_capability(provider_id, model_name, status, error_text=cache_error, persist=False)
        return status, provider_id, model_name


def run_startup_model_capability_probe(*, force_full: bool = False) -> None:
    if not MODEL_CAPABILITY_CACHE_ENABLED:
        logger.info("Model capability cache is disabled; skipping startup capability probe.")
        return
    if not MODEL_CAPABILITY_PROBE_ENABLED and not force_full:
        logger.info("Model capability probe is disabled (MODEL_CAPABILITY_PROBE_ENABLED=0).")
        return
    if force_full:
        probe_keys = list(ALL_MODEL_ORDER or MODEL_ORDER)
    else:
        probe_keys = build_model_probe_keys()
    probe_keys = [
        key
        for key in probe_keys
        if MODEL_PROVIDER_BY_KEY.get(key, "") not in NON_REPROBE_PROVIDERS
    ]
    if not probe_keys:
        return

    if force_full:
        stale_keys = list(dict.fromkeys(probe_keys))
        fresh_count = 0
    else:
        now_ts = int(time.time())
        stale_keys = []
        fresh_count = 0
        for model_key in probe_keys:
            provider_id = MODEL_PROVIDER_BY_KEY.get(model_key, "")
            model_name = MODEL_NAME_BY_KEY.get(model_key, "")
            if not provider_id or not model_name:
                continue
            cached = get_model_capability(provider_id, model_name)
            if cached and int(cached.get("expires_at", 0) or 0) > now_ts:
                fresh_count += 1
                continue
            stale_keys.append(model_key)

        if not stale_keys:
            logger.info("Model capability probe: all %d selected entries are fresh.", fresh_count)
            return

    provider_probe_cache: dict[str, tuple[bool, str]] = {}
    filtered_stale_keys: list[str] = []
    skipped_unavailable = 0
    for model_key in stale_keys:
        provider_id = MODEL_PROVIDER_BY_KEY.get(model_key, "")
        model_name = MODEL_NAME_BY_KEY.get(model_key, "")
        if not provider_id or not model_name:
            continue
        if provider_id not in provider_probe_cache:
            provider_probe_cache[provider_id] = check_provider_availability(provider_id, force=force_full)
        provider_ok, provider_reason = provider_probe_cache[provider_id]
        if not provider_ok:
            skipped_unavailable += 1
            set_model_capability(
                provider_id,
                model_name,
                MODEL_CAPABILITY_STATUS_TRANSIENT,
                error_text=provider_reason,
                persist=False,
            )
            continue
        filtered_stale_keys.append(model_key)

    stale_keys = filtered_stale_keys
    if skipped_unavailable:
        logger.warning(
            "Model capability probe: skipped %d entries because provider model-list endpoints are unavailable.",
            skipped_unavailable,
        )
    if not stale_keys:
        save_model_capabilities()
        logger.info("Model capability probe: no entries left after provider availability pre-check.")
        return

    logger.info(
        "Model capability probe: checking %d entries (fresh=%d, scope=%s, forced_full=%s, workers=%d, timeout=%ss).",
        len(stale_keys),
        fresh_count,
        "all" if force_full else MODEL_PROBE_SCOPE,
        force_full,
        MODEL_PROBE_WORKERS,
        MODEL_PROBE_TIMEOUT_SECONDS,
    )
    counters: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MODEL_PROBE_WORKERS) as executor:
        futures = [executor.submit(probe_single_model_capability, key) for key in stale_keys]
        for future in concurrent.futures.as_completed(futures):
            try:
                status, _provider_id, _model_name = future.result()
                counters[status] = counters.get(status, 0) + 1
            except Exception as exc:
                logger.warning("Model capability probe task failed: %s", exc)
                counters[MODEL_CAPABILITY_STATUS_UNKNOWN] = counters.get(MODEL_CAPABILITY_STATUS_UNKNOWN, 0) + 1

    save_model_capabilities()
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counters.items())) or "none"
    logger.info("Model capability probe complete: %s", summary)


def apply_model_capability_filter(*, force: bool = False, strict: bool = False) -> None:
    global MODEL_ORDER, MODEL_ORDER_SET, PROVIDER_ORDER, PROVIDER_MODEL_KEYS, DEFAULT_TEXT_PROVIDER
    if not MODEL_CAPABILITY_CACHE_ENABLED:
        MODEL_ORDER = list(ALL_MODEL_ORDER)
        MODEL_ORDER_SET = set(MODEL_ORDER)
        PROVIDER_ORDER, PROVIDER_MODEL_KEYS = build_provider_index()
        DEFAULT_TEXT_PROVIDER = MODEL_ORDER[0] if MODEL_ORDER else ""
        return
    if not MODEL_HIDE_UNAVAILABLE_MODELS and not force:
        return
    if not ALL_MODEL_ORDER:
        return

    filtered: list[str] = []
    hidden = 0
    for model_key in ALL_MODEL_ORDER:
        provider_id = MODEL_PROVIDER_BY_KEY.get(model_key, "")
        model_name = MODEL_NAME_BY_KEY.get(model_key, "")
        if not provider_id or not model_name:
            continue
        skip, _status = should_skip_model_candidate(provider_id, model_name)
        if skip:
            hidden += 1
            continue
        filtered.append(model_key)

    if filtered:
        MODEL_ORDER = filtered
    else:
        MODEL_ORDER = []
        logger.error("Capability filter removed all models.")
        if strict:
            raise RuntimeError("No available models after startup capability probe.")

    MODEL_ORDER_SET = set(MODEL_ORDER)
    PROVIDER_ORDER, PROVIDER_MODEL_KEYS = build_provider_index()
    DEFAULT_TEXT_PROVIDER = MODEL_ORDER[0] if MODEL_ORDER else ""
    if hidden:
        logger.info("Capability filter hidden %d unavailable model entries.", hidden)


def generate_text(
    prompt: str,
    model_key: str,
    history: list[dict[str, str]],
    attachments: list[InputAttachment] | None = None,
    *,
    allow_attachment_auto_fallback: bool = True,
    selected_only: bool = False,
    provider_allowlist: set[str] | None = None,
) -> Tuple[str, str, str, dict[str, int]]:
    chosen_key = model_key if model_key in MODEL_ORDER_SET else DEFAULT_TEXT_PROVIDER
    if not chosen_key:
        raise RuntimeError("No text models available.")

    normalized_attachments = attachments or []
    requires_attachment_support = bool(normalized_attachments)
    selected_provider = MODEL_PROVIDER_BY_KEY.get(chosen_key, "")
    selected_model_name = MODEL_NAME_BY_KEY.get(chosen_key, "")
    if selected_only:
        raw_attempt_keys = [chosen_key]
    else:
        raw_attempt_keys = build_generation_model_keys(chosen_key) or [chosen_key]
    if provider_allowlist is not None:
        normalized_allowlist = {item.strip().lower() for item in provider_allowlist if str(item).strip()}
        if not normalized_allowlist:
            raise RuntimeError("Provider allowlist is empty.")
        filtered_attempts = [
            key for key in raw_attempt_keys
            if MODEL_PROVIDER_BY_KEY.get(key, "").strip().lower() in normalized_allowlist
        ]
        if not filtered_attempts:
            filtered_attempts = [
                key for key in MODEL_ORDER
                if MODEL_PROVIDER_BY_KEY.get(key, "").strip().lower() in normalized_allowlist
            ]
        if not filtered_attempts:
            raise RuntimeError("No allowed providers are configured for this request.")
        raw_attempt_keys = filtered_attempts
    if requires_attachment_support and not allow_attachment_auto_fallback:
        selected_handler = TEXT_PROVIDER_HANDLERS.get(selected_provider)
        if selected_handler is None or not selected_handler.supports_input_attachments():
            raise AttachmentDecisionRequiredError(
                selected_provider,
                selected_model_name,
                "Provider-level attachment input is unavailable.",
            )
    logger.info(
        "Generation routing selection: provider=%s api=%s candidates=%d attachments=%d selected_only=%s",
        selected_provider or "unknown",
        selected_model_name or "unknown",
        len(raw_attempt_keys),
        len(normalized_attachments),
        selected_only,
    )
    attempt_keys: list[str] = []
    skipped_candidates: list[str] = []
    provider_availability: dict[str, tuple[bool, str]] = {}
    for candidate_key in raw_attempt_keys:
        provider_id = MODEL_PROVIDER_BY_KEY.get(candidate_key, "")
        model_name = MODEL_NAME_BY_KEY.get(candidate_key, "")
        if not provider_id or not model_name:
            continue
        handler = TEXT_PROVIDER_HANDLERS.get(provider_id)
        if requires_attachment_support and (handler is None or not handler.supports_input_attachments()):
            skipped_candidates.append(f"{provider_id}:{model_name}(attachments_unsupported)")
            continue
        just_checked = False
        if provider_id not in provider_availability:
            provider_availability[provider_id] = check_provider_availability(provider_id)
            just_checked = True
        provider_ok, provider_reason = provider_availability[provider_id]
        if not provider_ok:
            skipped_candidates.append(f"{provider_id}:{model_name}(provider_unavailable)")
            continue
        if just_checked and provider_reason:
            logger.info(
                "Provider model-list probe for %s returned non-blocking issue: %s",
                provider_id,
                provider_reason,
            )
        should_skip, status = should_skip_model_candidate(provider_id, model_name)
        if should_skip:
            skipped_candidates.append(f"{provider_id}:{model_name}({status})")
            continue
        attempt_keys.append(candidate_key)
    if skipped_candidates:
        logger.info("Skipping ineligible candidates: %s", ", ".join(skipped_candidates))
    if not attempt_keys:
        if requires_attachment_support:
            if not allow_attachment_auto_fallback:
                raise AttachmentDecisionRequiredError(
                    selected_provider,
                    selected_model_name,
                    "Current model is unavailable for this attachment request.",
                )
            raise RuntimeError(
                "No available models/providers currently support the uploaded attachment types."
            )
        attempt_keys = list(raw_attempt_keys)

    started_at = time.monotonic()
    attempted: list[str] = []
    last_error: Exception | None = None
    last_status = MODEL_CAPABILITY_STATUS_UNKNOWN
    for idx, candidate_key in enumerate(attempt_keys):
        provider_id = MODEL_PROVIDER_BY_KEY.get(candidate_key, "")
        model_name = MODEL_NAME_BY_KEY.get(candidate_key, "")
        if not provider_id or not model_name:
            continue
        base_timeout = get_timeout_for_model(model_name)
        elapsed = int(time.monotonic() - started_at)
        remaining_budget = max(5, base_timeout - elapsed)
        per_attempt_timeout = remaining_budget if idx == 0 else min(remaining_budget, FALLBACK_ATTEMPT_TIMEOUT_SECONDS)

        logger.info(
            "Generation attempt %d/%d provider=%s model=%s timeout=%ss",
            idx + 1,
            len(attempt_keys),
            provider_id,
            model_name,
            per_attempt_timeout,
        )
        attempted.append(f"{provider_id}:{model_name}")
        try:
            if idx > 0:
                logger.warning(
                    "Retrying generation via fallback provider/model: %s (%s)",
                    provider_id,
                    model_name,
                )
            text, used_model, usage = generate_text_with_handler(
                provider_id,
                prompt,
                model_name,
                history,
                normalized_attachments,
                timeout_seconds=per_attempt_timeout,
            )
            set_model_capability(provider_id, model_name, MODEL_CAPABILITY_STATUS_AVAILABLE)
            return text, used_model, provider_id, usage
        except Exception as exc:
            last_error = exc
            status = classify_generation_error(exc)
            error_text = str(exc)
            attachment_related_error = requires_attachment_support and is_attachment_related_error(error_text)
            if attachment_related_error:
                status = MODEL_CAPABILITY_STATUS_UNKNOWN
            if requires_attachment_support and not allow_attachment_auto_fallback:
                if attachment_related_error or status == MODEL_CAPABILITY_STATUS_UNSUPPORTED:
                    raise AttachmentDecisionRequiredError(provider_id, model_name, error_text) from exc
            last_status = status
            if classify_provider_probe_error(error_text) == "unavailable":
                set_cached_provider_availability(
                    provider_id,
                    False,
                    error_text,
                    PROVIDER_UNAVAILABLE_TTL_SECONDS,
                )
            if not attachment_related_error:
                cache_error = "" if status == MODEL_CAPABILITY_STATUS_AVAILABLE else error_text
                set_model_capability(provider_id, model_name, status, error_text=cache_error)
            if status == MODEL_CAPABILITY_STATUS_AVAILABLE and is_empty_response_error(exc):
                logger.info(
                    "Generation attempt returned empty response for provider=%s model=%s; "
                    "treating API as working and continuing fallback.",
                    provider_id,
                    model_name,
                )
            elif attachment_related_error:
                logger.warning(
                    "Generation attempt failed for provider=%s model=%s due to attachment support mismatch; "
                    "keeping model in normal text pool. error=%s",
                    provider_id,
                    model_name,
                    exc,
                )
            elif status == MODEL_CAPABILITY_STATUS_UNSUPPORTED:
                logger.warning(
                    "Generation attempt marked unsupported for provider=%s model=%s. error=%s",
                    provider_id,
                    model_name,
                    exc,
                )
            else:
                logger.warning(
                    "Generation attempt failed for provider=%s model=%s: %s",
                    provider_id,
                    model_name,
                    exc,
                )
            if not is_retryable_generation_error(exc):
                break
            continue

    attempts_str = ", ".join(attempted) if attempted else "none"
    if last_error is not None:
        if last_status == MODEL_CAPABILITY_STATUS_UNSUPPORTED:
            raise RuntimeError(
                f"Generation failed after trying: {attempts_str}. Last error: {last_error}. "
                "Detected model access/compatibility issue (unsupported or invalid_model)."
            ) from last_error
        raise RuntimeError(
            f"Generation failed after trying: {attempts_str}. Last error: {last_error}"
        ) from last_error
    raise RuntimeError("No text models available.")


def generate_inline_text(prompt: str) -> tuple[str, str, str, dict[str, int]]:
    candidate_keys = build_inline_model_keys()
    if not candidate_keys:
        raise RuntimeError("Inline mode is unavailable (no Gemini Flash models configured).")

    logger.info(
        "Inline generation selection: candidates=%d max_output_tokens=%d",
        len(candidate_keys),
        INLINE_MAX_OUTPUT_TOKENS,
    )

    provider_availability: dict[str, tuple[bool, str]] = {}
    attempt_keys: list[str] = []
    skipped_candidates: list[str] = []
    for candidate_key in candidate_keys:
        provider_id = MODEL_PROVIDER_BY_KEY.get(candidate_key, "")
        model_name = MODEL_NAME_BY_KEY.get(candidate_key, "")
        if not provider_id or not model_name:
            continue
        if provider_id not in provider_availability:
            provider_availability[provider_id] = check_provider_availability(provider_id)
        provider_ok, provider_reason = provider_availability[provider_id]
        if not provider_ok:
            skipped_candidates.append(f"{provider_id}:{model_name}(provider_unavailable)")
            continue
        if provider_reason:
            logger.info(
                "Inline provider model-list probe for %s returned non-blocking issue: %s",
                provider_id,
                provider_reason,
            )
        should_skip, status = should_skip_model_candidate(provider_id, model_name)
        if should_skip:
            skipped_candidates.append(f"{provider_id}:{model_name}({status})")
            continue
        attempt_keys.append(candidate_key)

    if skipped_candidates:
        logger.info("Inline skipping ineligible candidates: %s", ", ".join(skipped_candidates))
    if not attempt_keys:
        attempt_keys = list(candidate_keys)

    started_at = time.monotonic()
    attempted: list[str] = []
    last_error: Exception | None = None
    last_status = MODEL_CAPABILITY_STATUS_UNKNOWN
    for idx, candidate_key in enumerate(attempt_keys):
        provider_id = MODEL_PROVIDER_BY_KEY.get(candidate_key, "")
        model_name = MODEL_NAME_BY_KEY.get(candidate_key, "")
        if not provider_id or not model_name:
            continue
        base_timeout = get_timeout_for_model(model_name)
        elapsed = int(time.monotonic() - started_at)
        remaining_budget = max(5, base_timeout - elapsed)
        per_attempt_timeout = remaining_budget if idx == 0 else min(remaining_budget, FALLBACK_ATTEMPT_TIMEOUT_SECONDS)

        logger.info(
            "Inline generation attempt %d/%d provider=%s model=%s timeout=%ss",
            idx + 1,
            len(attempt_keys),
            provider_id,
            model_name,
            per_attempt_timeout,
        )
        attempted.append(f"{provider_id}:{model_name}")
        try:
            if idx > 0:
                logger.warning(
                    "Retrying inline generation via fallback provider/model: %s (%s)",
                    provider_id,
                    model_name,
                )
            text, used_model, usage = generate_text_with_handler(
                provider_id,
                prompt,
                model_name,
                [],
                [],
                timeout_seconds=per_attempt_timeout,
                max_output_tokens=INLINE_MAX_OUTPUT_TOKENS,
            )
            set_model_capability(provider_id, model_name, MODEL_CAPABILITY_STATUS_AVAILABLE)
            return text, used_model, provider_id, usage
        except Exception as exc:
            last_error = exc
            status = classify_generation_error(exc)
            error_text = str(exc)
            last_status = status
            if classify_provider_probe_error(error_text) == "unavailable":
                set_cached_provider_availability(
                    provider_id,
                    False,
                    error_text,
                    PROVIDER_UNAVAILABLE_TTL_SECONDS,
                )
            cache_error = "" if status == MODEL_CAPABILITY_STATUS_AVAILABLE else error_text
            set_model_capability(provider_id, model_name, status, error_text=cache_error)
            if status == MODEL_CAPABILITY_STATUS_AVAILABLE and is_empty_response_error(exc):
                logger.info(
                    "Inline generation returned empty response for provider=%s model=%s; continuing fallback.",
                    provider_id,
                    model_name,
                )
            elif status == MODEL_CAPABILITY_STATUS_UNSUPPORTED:
                logger.warning(
                    "Inline generation marked unsupported for provider=%s model=%s. error=%s",
                    provider_id,
                    model_name,
                    exc,
                )
            else:
                logger.warning(
                    "Inline generation failed for provider=%s model=%s: %s",
                    provider_id,
                    model_name,
                    exc,
                )
            if not is_retryable_generation_error(exc):
                break

    attempts_str = ", ".join(attempted) if attempted else "none"
    if last_error is not None:
        if last_status == MODEL_CAPABILITY_STATUS_UNSUPPORTED:
            raise RuntimeError(
                f"Inline generation failed after trying: {attempts_str}. Last error: {last_error}. "
                "Detected model access/compatibility issue (unsupported or invalid_model)."
            ) from last_error
        raise RuntimeError(
            f"Inline generation failed after trying: {attempts_str}. Last error: {last_error}"
        ) from last_error
    raise RuntimeError("No inline models available.")


def generate_image(
    prompt: str,
    model_key: str,
    *,
    provider_allowlist: set[str] | None = None,
    prefer_default_image_model: bool = False,
) -> tuple[dict[str, str], str, str]:
    chosen_key = model_key if model_key in MODEL_ORDER_SET else DEFAULT_TEXT_PROVIDER
    if not chosen_key:
        raise RuntimeError("No models available.")

    selected_provider = MODEL_PROVIDER_BY_KEY.get(chosen_key, "")
    selected_model_name = MODEL_NAME_BY_KEY.get(chosen_key, "")
    raw_attempt_keys: list[str] = build_image_generation_model_keys(chosen_key) or [chosen_key]
    direct_default_attempts: list[tuple[str, str]] = []
    if prefer_default_image_model and DEFAULT_IMAGE_MODEL_NAME:
        default_image_key = get_default_image_model_key()
        if default_image_key:
            raw_attempt_keys = [default_image_key] + [key for key in raw_attempt_keys if key != default_image_key]
        else:
            if DEFAULT_IMAGE_MODEL_PROVIDER:
                direct_default_attempts = [(DEFAULT_IMAGE_MODEL_PROVIDER, DEFAULT_IMAGE_MODEL_NAME)]
            else:
                direct_default_attempts = [(provider_id, DEFAULT_IMAGE_MODEL_NAME) for provider_id in PROVIDER_ORDER]
            logger.info(
                "Default image model '%s' not found in catalog; attempting direct provider calls.",
                DEFAULT_IMAGE_MODEL_NAME,
            )
    normalized_allowlist: set[str] | None = None
    if provider_allowlist is not None:
        normalized_allowlist = {item.strip().lower() for item in provider_allowlist if str(item).strip()}
        if not normalized_allowlist:
            raise RuntimeError("Provider allowlist is empty.")
        filtered_attempts = [
            key for key in raw_attempt_keys
            if MODEL_PROVIDER_BY_KEY.get(key, "").strip().lower() in normalized_allowlist
        ]
        if not filtered_attempts:
            filtered_attempts = [
                key for key in MODEL_ORDER
                if MODEL_PROVIDER_BY_KEY.get(key, "").strip().lower() in normalized_allowlist
            ]
        if not filtered_attempts:
            raise RuntimeError("No allowed providers are configured for this request.")
        raw_attempt_keys = filtered_attempts
    logger.info(
        "Image routing selection: provider=%s model=%s candidates=%d size=%s",
        selected_provider or "unknown",
        selected_model_name or "unknown",
        len(raw_attempt_keys),
        IMAGE_GENERATION_SIZE,
    )

    provider_availability: dict[str, tuple[bool, str]] = {}
    attempt_keys: list[str] = []
    skipped_candidates: list[str] = []
    for candidate_key in raw_attempt_keys:
        provider_id = MODEL_PROVIDER_BY_KEY.get(candidate_key, "")
        model_name = MODEL_NAME_BY_KEY.get(candidate_key, "")
        if not provider_id or not model_name:
            continue
        if provider_id not in provider_availability:
            provider_availability[provider_id] = check_provider_availability(provider_id)
        provider_ok, _provider_reason = provider_availability[provider_id]
        if not provider_ok:
            skipped_candidates.append(f"{provider_id}:{model_name}(provider_unavailable)")
            continue
        attempt_keys.append(candidate_key)

    if skipped_candidates:
        logger.info("Skipping unavailable image candidates: %s", ", ".join(skipped_candidates))
    if not attempt_keys:
        attempt_keys = list(raw_attempt_keys)

    attempted: list[str] = []
    last_error: Exception | None = None
    if direct_default_attempts:
        for provider_id, model_name in direct_default_attempts:
            provider_id = str(provider_id or "").strip().lower()
            if not provider_id:
                continue
            if normalized_allowlist is not None and provider_id not in normalized_allowlist:
                continue
            if provider_id not in TEXT_PROVIDER_HANDLERS:
                logger.warning("Default image model provider '%s' is not configured.", provider_id)
                continue
            if provider_id not in provider_availability:
                provider_availability[provider_id] = check_provider_availability(provider_id)
            provider_ok, _provider_reason = provider_availability[provider_id]
            if not provider_ok:
                skipped_candidates.append(f"{provider_id}:{model_name}(provider_unavailable)")
                continue
            for variant_name in build_default_image_model_variants(model_name):
                attempted.append(f"{provider_id}:{variant_name}")
                try:
                    image_data, used_model = generate_image_with_handler(
                        provider_id,
                        prompt,
                        variant_name,
                        timeout_seconds=IMAGE_MODEL_TIMEOUT_SECONDS,
                    )
                    return image_data, used_model, provider_id
                except Exception as exc:
                    last_error = exc
                    error_text = str(exc)
                    if classify_provider_probe_error(error_text) == "unavailable":
                        set_cached_provider_availability(
                            provider_id,
                            False,
                            error_text,
                            PROVIDER_UNAVAILABLE_TTL_SECONDS,
                        )
                    logger.warning(
                        "Default image generation attempt failed for provider=%s model=%s: %s",
                        provider_id,
                        variant_name,
                        exc,
                    )
                    continue
    for idx, candidate_key in enumerate(attempt_keys):
        provider_id = MODEL_PROVIDER_BY_KEY.get(candidate_key, "")
        model_name = MODEL_NAME_BY_KEY.get(candidate_key, "")
        if not provider_id or not model_name:
            continue
        per_attempt_timeout = IMAGE_MODEL_TIMEOUT_SECONDS

        logger.info(
            "Image generation attempt %d/%d provider=%s model=%s timeout=%ss",
            idx + 1,
            len(attempt_keys),
            provider_id,
            model_name,
            per_attempt_timeout,
        )
        attempted.append(f"{provider_id}:{model_name}")
        try:
            image_data, used_model = generate_image_with_handler(
                provider_id,
                prompt,
                model_name,
                timeout_seconds=per_attempt_timeout,
            )
            return image_data, used_model, provider_id
        except Exception as exc:
            last_error = exc
            error_text = str(exc)
            if classify_provider_probe_error(error_text) == "unavailable":
                set_cached_provider_availability(
                    provider_id,
                    False,
                    error_text,
                    PROVIDER_UNAVAILABLE_TTL_SECONDS,
                )
            logger.warning(
                "Image generation attempt failed for provider=%s model=%s: %s",
                provider_id,
                model_name,
                exc,
            )
            if not is_retryable_generation_error(exc):
                break
            continue

    attempts_str = ", ".join(attempted) if attempted else "none"
    if last_error is not None:
        raise RuntimeError(
            f"Image generation failed after trying: {attempts_str}. Last error: {last_error}"
        ) from last_error
    raise RuntimeError("No models available for image generation.")


def get_model_menu_page_size() -> int:
    return 12


def get_model_page_count(model_keys: Optional[list[str]] = None) -> int:
    keys = model_keys if model_keys is not None else MODEL_ORDER
    if not keys:
        return 1
    page_size = get_model_menu_page_size()
    return max(1, math.ceil(len(keys) / page_size))


def clamp_model_page(page: int, model_keys: Optional[list[str]] = None) -> int:
    return max(0, min(page, get_model_page_count(model_keys) - 1))


def get_model_page_slice(model_keys: list[str], page: int) -> tuple[int, int, list[str]]:
    page = clamp_model_page(page, model_keys)
    page_size = get_model_menu_page_size()
    start = page * page_size
    stop = start + page_size
    page_count = get_model_page_count(model_keys)
    return page, page_count, model_keys[start:stop]


def find_model_page(model_key: str) -> int:
    if model_key not in MODEL_ORDER_SET:
        return 0
    idx = MODEL_ORDER.index(model_key)
    return idx // get_model_menu_page_size()


def search_model_keys(query: str) -> list[str]:
    raw = query.strip().lower()
    if not raw:
        return list(MODEL_ORDER)

    terms = [x for x in re.split(r"\s+", raw) if x]
    ranked: list[tuple[int, int, str]] = []
    for idx, model_key in enumerate(MODEL_ORDER):
        label = get_model_label(model_key).lower()
        full_name = get_model_full_name(model_key).lower()
        provider = get_model_provider_id(model_key).lower()
        score = 0

        if raw in full_name:
            score += 120
        if raw in label:
            score += 90
        if raw in provider:
            score += 110

        for term in terms:
            if term in provider:
                score += 30
            if term in full_name:
                score += 25
            if term in label:
                score += 15

        if score > 0:
            ranked.append((score, idx, model_key))

    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in ranked]


def build_model_select_rows(selected_key: str, model_keys: list[str]) -> list[list[InlineKeyboardButton]]:
    rows: list[list[InlineKeyboardButton]] = []
    for model_key in model_keys:
        title = get_model_label(model_key)
        if selected_key == model_key:
            title = f"* {title}"
        rows.append([InlineKeyboardButton(title, callback_data=f"model:{model_key}")])
    return rows


def build_model_keyboard(selected_key: str, page: int) -> InlineKeyboardMarkup:
    page, page_count, keys = get_model_page_slice(MODEL_ORDER, page)
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton("Providers", callback_data="modelproviders:0"),
            InlineKeyboardButton("Search", callback_data="modelsearchprompt"),
        ]
    ]
    rows.extend(build_model_select_rows(selected_key, keys))

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("Prev", callback_data=f"modelpage:{page - 1}"))
    if page < page_count - 1:
        nav.append(InlineKeyboardButton("Next", callback_data=f"modelpage:{page + 1}"))
    if nav:
        rows.append(nav)

    return InlineKeyboardMarkup(rows)


def build_provider_keyboard(page: int) -> InlineKeyboardMarkup:
    provider_page_size = get_model_menu_page_size()
    if not PROVIDER_ORDER:
        return InlineKeyboardMarkup([[InlineKeyboardButton("Search", callback_data="modelsearchprompt")]])

    provider_page_count = max(1, math.ceil(len(PROVIDER_ORDER) / provider_page_size))
    page = max(0, min(page, provider_page_count - 1))
    start = page * provider_page_size
    stop = start + provider_page_size
    provider_slice = PROVIDER_ORDER[start:stop]

    rows: list[list[InlineKeyboardButton]] = []
    for idx, provider_id in enumerate(provider_slice, start=start):
        provider_models = PROVIDER_MODEL_KEYS.get(provider_id, [])
        title = f"{get_provider_label(provider_id)} ({len(provider_models)})"
        rows.append([InlineKeyboardButton(title, callback_data=f"modelprovider:{idx}:0")])

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("Prev", callback_data=f"modelproviders:{page - 1}"))
    if page < provider_page_count - 1:
        nav.append(InlineKeyboardButton("Next", callback_data=f"modelproviders:{page + 1}"))
    if nav:
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton("All Models", callback_data="modelpage:0"),
            InlineKeyboardButton("Search", callback_data="modelsearchprompt"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def build_provider_model_keyboard(selected_key: str, provider_index: int, page: int) -> InlineKeyboardMarkup:
    if provider_index < 0 or provider_index >= len(PROVIDER_ORDER):
        return build_provider_keyboard(0)
    provider_id = PROVIDER_ORDER[provider_index]
    provider_keys = PROVIDER_MODEL_KEYS.get(provider_id, [])
    page, page_count, keys = get_model_page_slice(provider_keys, page)
    rows = build_model_select_rows(selected_key, keys)

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("Prev", callback_data=f"modelprovider:{provider_index}:{page - 1}"))
    if page < page_count - 1:
        nav.append(InlineKeyboardButton("Next", callback_data=f"modelprovider:{provider_index}:{page + 1}"))
    if nav:
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton("Providers", callback_data="modelproviders:0"),
            InlineKeyboardButton("Search", callback_data="modelsearchprompt"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def build_search_model_keyboard(selected_key: str, query: str, page: int) -> InlineKeyboardMarkup:
    keys = search_model_keys(query)
    page, page_count, page_keys = get_model_page_slice(keys, page)
    rows = build_model_select_rows(selected_key, page_keys)

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("Prev", callback_data=f"modelsearchpage:{page - 1}"))
    if page < page_count - 1:
        nav.append(InlineKeyboardButton("Next", callback_data=f"modelsearchpage:{page + 1}"))
    if nav:
        rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton("New Search", callback_data="modelsearchprompt"),
            InlineKeyboardButton("All Models", callback_data="modelpage:0"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def get_effective_user_id(update: Update) -> int | None:
    user = update.effective_user
    return user.id if user is not None else None


def get_effective_chat_id(update: Update) -> int | None:
    chat = update.effective_chat
    return chat.id if chat is not None else None


def get_effective_message(update: Update):
    return update.effective_message


def first_context_arg(context: ContextTypes.DEFAULT_TYPE) -> str:
    if not context.args:
        return ""
    return str(context.args[0]).strip()


def parse_callback_suffix(data: str, prefix: str) -> str | None:
    normalized = str(data or "")
    if not normalized.startswith(prefix):
        return None
    return normalized[len(prefix):]


def parse_int_or_default(value: str, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def build_inline_result_id(user_id: int | None, prompt: str) -> str:
    seed = f"{user_id or 0}:{prompt}:{time.time_ns()}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return digest[:32]


def build_inline_result_title(prompt: str) -> str:
    cleaned = str(prompt or "").strip()
    if not cleaned:
        return "Generate response"
    if len(cleaned) <= 72:
        return cleaned
    return f"{cleaned[:69]}..."


def build_inline_keyboard() -> InlineKeyboardMarkup | None:
    if not BOT_USERNAME:
        return None
    url = f"https://t.me/{BOT_USERNAME}"
    return InlineKeyboardMarkup([[InlineKeyboardButton("Open Bot", url=url)]])


def build_model_menu_text(model_key: str, page: int) -> str:
    page_count = get_model_page_count()
    return (
        f"Current model: {get_model_label(model_key)}\n"
        f"Models: {len(MODEL_ORDER)} across {len(PROVIDER_ORDER)} providers.\n"
        f"Select model ({page + 1}/{page_count}):"
    )


def build_provider_menu_text(model_key: str, page: int, page_count: int) -> str:
    return (
        f"Current model: {get_model_label(model_key)}\n"
        f"Providers: {len(PROVIDER_ORDER)}\n"
        f"Select provider ({page + 1}/{page_count}):"
    )


async def reply_to_update(update: Update, text: str, **kwargs) -> bool:
    message = get_effective_message(update)
    if message is None:
        return False
    await message.reply_text(text, **kwargs)
    return True


async def ensure_allowed_command(update: Update, *, denied_text: str = "Access denied.") -> bool:
    if is_allowed(update.effective_user):
        return True
    if is_special_denied_user(update.effective_user):
        return False
    await reply_to_update(update, denied_text)
    return False


async def ensure_owner_command(update: Update) -> bool:
    if is_owner(update.effective_user):
        return True
    await reply_to_update(update, "Owner only command.")
    return False


async def ensure_allowed_chat(update: Update, *, denied_text: str = "Access denied.") -> int | None:
    if not await ensure_allowed_command(update, denied_text=denied_text):
        return None
    chat_id = get_effective_chat_id(update)
    if chat_id is None:
        await reply_to_update(update, "Chat not found.")
        return None
    return chat_id


async def ensure_owner_private_command(update: Update) -> bool:
    if not await ensure_owner_command(update):
        return False
    chat = update.effective_chat
    if chat and chat.type != "private":
        await reply_to_update(update, "Use this command in private chat with bot.")
        return False
    return True


async def get_allowed_callback_query(update: Update):
    query = update.callback_query
    if query is None:
        return None, None
    user = query.from_user
    if not is_allowed(user):
        if is_special_denied_user(user):
            return None, None
        await query.answer("Access denied.", show_alert=True)
        return None, None
    return query, user


async def get_allowed_callback_suffix(update: Update, prefix: str):
    query, user = await get_allowed_callback_query(update)
    if query is None or user is None:
        return None
    suffix = parse_callback_suffix(query.data or "", prefix)
    if suffix is None:
        await query.answer()
        return None
    return query, user, suffix


def resolve_allowlist_target_user_id(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    command_name: str,
    username_error: str,
) -> tuple[int | None, str | None]:
    message = get_effective_message(update)
    reply_to_message = getattr(message, "reply_to_message", None)
    target_user = getattr(reply_to_message, "from_user", None)
    if target_user is not None:
        return target_user.id, None

    raw_arg = first_context_arg(context)
    if not raw_arg:
        return None, f"Usage: /{command_name} <user_id> or reply to user message."
    if raw_arg.startswith("@"):
        return None, username_error
    user_id = parse_user_id(raw_arg)
    if user_id is None:
        return None, "Invalid user_id."
    return user_id, None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_allowed_command(update, denied_text="Access denied. Ask owner to add you."):
        return
    user_id = get_effective_user_id(update)
    model_key = get_user_model_key(user_id)
    chat_id = get_effective_chat_id(update)
    mode = get_chat_mode(chat_id) if chat_id is not None else "text"
    await reply_to_update(
        update,
        f"Ready. Current model: {get_model_label(model_key)}\n"
        f"Current mode: {mode}\n"
        "Send text/photo/file in text mode, use /api or /provider to browse, /mode text|image to switch, or /modelsearch <text>.",
    )


async def model_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_allowed_command(update):
        return
    user_id = get_effective_user_id(update)
    model_key = get_user_model_key(user_id)
    page = find_model_page(model_key)
    await reply_to_update(
        update,
        build_model_menu_text(model_key, page),
        reply_markup=build_model_keyboard(model_key, page),
    )


async def provider_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_allowed_command(update):
        return
    user_id = get_effective_user_id(update)
    model_key = get_user_model_key(user_id)
    provider_page_count = max(1, math.ceil(len(PROVIDER_ORDER) / get_model_menu_page_size()))
    await reply_to_update(
        update,
        build_provider_menu_text(model_key, 0, provider_page_count),
        reply_markup=build_provider_keyboard(0),
    )


def build_model_search_view(model_key: str, query: str, page: int) -> tuple[str, InlineKeyboardMarkup]:
    matches = search_model_keys(query)
    page = clamp_model_page(page, matches)
    page_count = get_model_page_count(matches)
    text = (
        f"Current model: {get_model_label(model_key)}\n"
        f"Search: {query}\n"
        f"Matches: {len(matches)}\n"
        f"Select model ({page + 1}/{page_count}):"
    )
    return text, build_search_model_keyboard(model_key, query, page)


async def model_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_allowed_command(update):
        return
    query = " ".join(context.args).strip() if context.args else ""
    if not query:
        context.user_data["model_search_waiting_for_query"] = True
        await reply_to_update(
            update,
            "Send text to search models by provider/name.\n"
            "Examples: `google flash`, `qwen thinking`, `minimax`.",
        )
        return
    context.user_data["model_search_waiting_for_query"] = False
    context.user_data["model_search_query"] = query
    user_id = get_effective_user_id(update)
    model_key = get_user_model_key(user_id)
    text, keyboard = build_model_search_view(model_key, query, 0)
    await reply_to_update(update, text, reply_markup=keyboard)


async def retry_pending_attachment_after_model_selection(
    context: ContextTypes.DEFAULT_TYPE,
    query_message,
    *,
    user_id: int,
) -> None:
    if not context.user_data.get("pending_attachment_choose_model"):
        return
    context.user_data["pending_attachment_choose_model"] = False
    pending = get_pending_attachment_request(context)
    if pending is None:
        return
    if int(pending["chat_id"]) != query_message.chat.id:
        clear_pending_attachment_request(context)
        return

    await query_message.reply_text("Trying pending request with selected model...")
    try:
        await run_text_generation_flow(
            query_message,
            user_id=user_id,
            chat_id=int(pending["chat_id"]),
            prompt=str(pending["prompt"]),
            attachments=list(pending["attachments"]),
            selected_only=True,
            allow_attachment_auto_fallback=False,
            short_response=is_group_chat_type(query_message.chat),
            provider_allowlist=GROUP_PROVIDER_ALLOWLIST if is_group_chat_type(query_message.chat) else None,
        )
        clear_pending_attachment_request(context)
    except AttachmentDecisionRequiredError as exc:
        await query_message.reply_text(
            "This selected model still cannot handle the attachment.\n"
            f"Reason: {exc.reason or 'unsupported attachment input'}\n\n"
            "Choose how to proceed:",
            reply_markup=build_attachment_decision_keyboard(),
        )
    except Exception as exc:
        logger.exception("Pending attachment retry after model selection failed")
        await query_message.reply_text(f"Generation error: {exc}")


async def on_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parsed = await get_allowed_callback_suffix(update, "model:")
    if parsed is None:
        return
    query, user, model_key = parsed
    if model_key not in MODEL_ORDER_SET:
        await query.answer("Unknown model.", show_alert=True)
        return
    set_user_model_key(user.id, model_key)
    page = find_model_page(model_key)
    await query.answer(f"Selected: {get_model_label(model_key)}")
    await query.edit_message_text(
        build_model_menu_text(model_key, page),
        reply_markup=build_model_keyboard(model_key, page),
    )
    query_message = query.message
    if query_message is None:
        return
    await retry_pending_attachment_after_model_selection(context, query_message, user_id=user.id)


async def on_model_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parsed = await get_allowed_callback_suffix(update, "modelpage:")
    if parsed is None:
        return
    query, user, page_raw = parsed
    page = clamp_model_page(parse_int_or_default(page_raw, 0), MODEL_ORDER)
    model_key = get_user_model_key(user.id)
    await query.answer()
    await query.edit_message_text(
        build_model_menu_text(model_key, page),
        reply_markup=build_model_keyboard(model_key, page),
    )


async def on_model_providers_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parsed = await get_allowed_callback_suffix(update, "modelproviders:")
    if parsed is None:
        return
    query, user, page_raw = parsed
    page = parse_int_or_default(page_raw, 0)
    provider_page_count = max(1, math.ceil(len(PROVIDER_ORDER) / get_model_menu_page_size()))
    page = max(0, min(page, provider_page_count - 1))
    model_key = get_user_model_key(user.id)
    await query.answer()
    await query.edit_message_text(
        build_provider_menu_text(model_key, page, provider_page_count),
        reply_markup=build_provider_keyboard(page),
    )


async def on_model_provider_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parsed = await get_allowed_callback_suffix(update, "modelprovider:")
    if parsed is None:
        return
    query, user, suffix = parsed
    parts = suffix.split(":")
    if len(parts) != 2:
        await query.answer("Invalid provider request.", show_alert=True)
        return
    try:
        provider_index = int(parts[0])
        page = int(parts[1])
    except ValueError:
        await query.answer("Invalid provider request.", show_alert=True)
        return
    if provider_index < 0 or provider_index >= len(PROVIDER_ORDER):
        await query.answer("Unknown provider.", show_alert=True)
        return
    provider_id = PROVIDER_ORDER[provider_index]
    provider_keys = PROVIDER_MODEL_KEYS.get(provider_id, [])
    page = clamp_model_page(page, provider_keys)
    page_count = get_model_page_count(provider_keys)
    model_key = get_user_model_key(user.id)
    await query.answer()
    await query.edit_message_text(
        f"Current model: {get_model_label(model_key)}\n"
        f"Provider: {get_provider_label(provider_id)} ({len(provider_keys)} models)\n"
        f"Select model ({page + 1}/{page_count}):",
        reply_markup=build_provider_model_keyboard(model_key, provider_index, page),
    )


async def on_model_search_prompt_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query, user = await get_allowed_callback_query(update)
    if query is None or user is None:
        return
    context.user_data["model_search_waiting_for_query"] = True
    await query.answer("Send search text in chat.")
    await query.edit_message_text(
        "Model search is ready.\n"
        "Send text to search by provider/name.\n"
        "Examples: `google flash`, `qwen thinking`, `deepseek`."
    )


async def on_model_search_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parsed = await get_allowed_callback_suffix(update, "modelsearchpage:")
    if parsed is None:
        return
    query, user, page_raw = parsed
    page = parse_int_or_default(page_raw, 0)
    search_query = str(context.user_data.get("model_search_query", "")).strip()
    if not search_query:
        await query.answer("No active search. Use /modelsearch.", show_alert=True)
        return
    model_key = get_user_model_key(user.id)
    text, keyboard = build_model_search_view(model_key, search_query, page)
    await query.answer()
    await query.edit_message_text(text, reply_markup=keyboard)


async def on_attachment_resolution_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parsed = await get_allowed_callback_suffix(update, "attachresolve:")
    if parsed is None:
        return
    query, user, suffix = parsed
    action = suffix.strip().lower()
    pending = get_pending_attachment_request(context)
    if pending is None:
        await query.answer("No pending attachment request.", show_alert=True)
        return

    query_message = query.message
    if query_message is None:
        await query.answer("This action is unavailable here.", show_alert=True)
        return
    if pending["chat_id"] != query_message.chat.id:
        clear_pending_attachment_request(context)
        await query.answer("Pending request expired for this chat.", show_alert=True)
        return
    short_response = is_group_chat_type(query_message.chat)
    provider_allowlist = GROUP_PROVIDER_ALLOWLIST if short_response else None

    if action == "choose":
        context.user_data["pending_attachment_choose_model"] = True
        model_key = get_user_model_key(user.id)
        page = find_model_page(model_key)
        await query.answer("Choose another model.")
        await query.edit_message_text(
            build_model_menu_text(model_key, page),
            reply_markup=build_model_keyboard(model_key, page),
        )
        return

    if action not in {"without", "auto"}:
        await query.answer("Unknown action.", show_alert=True)
        return

    await query.answer("Processing...")
    prompt = pending["prompt"]
    attachments = pending["attachments"]
    try:
        if action == "without":
            await run_text_generation_flow(
                query_message,
                user_id=user.id,
                chat_id=pending["chat_id"],
                prompt=prompt,
                attachments=[],
                selected_only=True,
                allow_attachment_auto_fallback=False,
                short_response=short_response,
                provider_allowlist=provider_allowlist,
            )
        else:
            has_file_attachment = any(
                isinstance(item, dict) and str(item.get("kind", "")).strip().lower() == "file"
                for item in attachments
            )
            await run_text_generation_flow(
                query_message,
                user_id=user.id,
                chat_id=pending["chat_id"],
                prompt=prompt,
                attachments=attachments,
                selected_only=False,
                allow_attachment_auto_fallback=True,
                short_response=short_response,
                provider_allowlist=provider_allowlist,
                append_used_model_footer=has_file_attachment and not short_response,
            )
        clear_pending_attachment_request(context)
        with contextlib.suppress(Exception):
            await query.edit_message_reply_markup(reply_markup=None)
    except AttachmentDecisionRequiredError as exc:
        await query_message.reply_text(
            "Selected model still cannot handle this attachment.\n"
            f"Reason: {exc.reason or 'unsupported attachment input'}\n\n"
            "Choose how to proceed:",
            reply_markup=build_attachment_decision_keyboard(),
        )
    except Exception as exc:
        logger.exception("Attachment resolution action failed")
        await query_message.reply_text(f"Generation error: {exc}")


async def run_provider_availability_recheck_once() -> None:
    try:
        stats = await asyncio.to_thread(reconcile_provider_availability_states, force=True)
        if int(stats.get("changed", 0)) > 0 or int(stats.get("providers_unavailable", 0)) > 0:
            logger.info(
                "Provider availability recheck: unavailable=%d/%d checked=%d skipped_non_reprobe=%d marked_transient=%d "
                "cleared_transient=%d changed=%s",
                int(stats.get("providers_unavailable", 0)),
                int(stats.get("providers_total", 0)),
                int(stats.get("providers_checked", 0)),
                int(stats.get("providers_skipped_non_reprobe", 0)),
                int(stats.get("marked_transient", 0)),
                int(stats.get("cleared_transient", 0)),
                bool(int(stats.get("changed", 0))),
            )
    except Exception:
        logger.exception("Provider availability recheck job failed")


def run_model_capability_recheck_sync() -> dict[str, int]:
    if not MODEL_CAPABILITY_CACHE_ENABLED:
        active_models = len(MODEL_ORDER)
        return {
            "busy": 0,
            "before_cache_entries": 0,
            "after_cache_entries": 0,
            "before_active_models": active_models,
            "after_active_models": active_models,
        }
    if not MODEL_CAPABILITY_RECHECK_LOCK.acquire(blocking=False):
        return {"busy": 1}
    try:
        with MODEL_CAPABILITIES_LOCK:
            before_cache_entries = len(MODEL_CAPABILITIES)
        before_active_models = len(MODEL_ORDER)
        run_startup_model_capability_probe(force_full=MODEL_CAPABILITY_RECHECK_FORCE_FULL)
        apply_model_capability_filter(force=True, strict=False)
        with MODEL_CAPABILITIES_LOCK:
            after_cache_entries = len(MODEL_CAPABILITIES)
        after_active_models = len(MODEL_ORDER)
        return {
            "busy": 0,
            "before_cache_entries": before_cache_entries,
            "after_cache_entries": after_cache_entries,
            "before_active_models": before_active_models,
            "after_active_models": after_active_models,
        }
    finally:
        MODEL_CAPABILITY_RECHECK_LOCK.release()


async def run_model_capability_recheck_once() -> None:
    try:
        stats = await asyncio.to_thread(run_model_capability_recheck_sync)
        if int(stats.get("busy", 0)) > 0:
            logger.info("Model capability recheck skipped: previous run is still in progress.")
            return
        logger.info(
            "Model capability recheck complete: active_models=%d->%d cache_entries=%d->%d forced_full=%s",
            int(stats.get("before_active_models", 0)),
            int(stats.get("after_active_models", 0)),
            int(stats.get("before_cache_entries", 0)),
            int(stats.get("after_cache_entries", 0)),
            MODEL_CAPABILITY_RECHECK_FORCE_FULL,
        )
    except Exception:
        logger.exception("Model capability recheck job failed")


async def provider_availability_recheck_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    await run_provider_availability_recheck_once()


async def model_capability_recheck_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    await run_model_capability_recheck_once()


async def provider_availability_recheck_loop() -> None:
    try:
        await asyncio.sleep(PROVIDER_AVAILABILITY_RECHECK_INITIAL_DELAY_SECONDS)
        while True:
            await run_provider_availability_recheck_once()
            await asyncio.sleep(PROVIDER_AVAILABILITY_RECHECK_INTERVAL_SECONDS)
    except asyncio.CancelledError:
        logger.info("Provider availability recheck loop cancelled.")
        raise
    except Exception:
        logger.exception("Provider availability recheck loop failed")


async def model_capability_recheck_loop() -> None:
    try:
        await asyncio.sleep(MODEL_CAPABILITY_RECHECK_INITIAL_DELAY_SECONDS)
        while True:
            await run_model_capability_recheck_once()
            await asyncio.sleep(MODEL_CAPABILITY_RECHECK_INTERVAL_SECONDS)
    except asyncio.CancelledError:
        logger.info("Model capability recheck loop cancelled.")
        raise
    except Exception:
        logger.exception("Model capability recheck loop failed")


async def stop_provider_availability_recheck_loop() -> None:
    global PROVIDER_AVAILABILITY_RECHECK_TASK
    task = PROVIDER_AVAILABILITY_RECHECK_TASK
    PROVIDER_AVAILABILITY_RECHECK_TASK = None
    if task is None:
        return
    if task.done():
        with contextlib.suppress(Exception):
            _ = task.result()
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def stop_model_capability_recheck_loop() -> None:
    global MODEL_CAPABILITY_RECHECK_TASK
    task = MODEL_CAPABILITY_RECHECK_TASK
    MODEL_CAPABILITY_RECHECK_TASK = None
    if task is None:
        return
    if task.done():
        with contextlib.suppress(Exception):
            _ = task.result()
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def on_app_post_init(application) -> None:
    global PROVIDER_AVAILABILITY_RECHECK_TASK, MODEL_CAPABILITY_RECHECK_TASK, BOT_USERNAME
    if not BOT_USERNAME:
        with contextlib.suppress(Exception):
            me = await application.bot.get_me()
            username = str(getattr(me, "username", "") or "").strip().lstrip("@")
            if username:
                BOT_USERNAME = username
    if not BOT_USERNAME:
        logger.warning("BOT_USERNAME is not set; inline Open Bot button will be hidden.")
    job_queue = getattr(application, "_job_queue", None)
    if PROVIDER_AVAILABILITY_RECHECK_ENABLED:
        if job_queue is not None:
            job_queue.run_repeating(
                provider_availability_recheck_job,
                interval=PROVIDER_AVAILABILITY_RECHECK_INTERVAL_SECONDS,
                first=PROVIDER_AVAILABILITY_RECHECK_INITIAL_DELAY_SECONDS,
                name="provider_availability_recheck",
            )
            logger.info(
                "Scheduled provider availability recheck via JobQueue: interval=%ss initial_delay=%ss",
                PROVIDER_AVAILABILITY_RECHECK_INTERVAL_SECONDS,
                PROVIDER_AVAILABILITY_RECHECK_INITIAL_DELAY_SECONDS,
            )
        else:
            PROVIDER_AVAILABILITY_RECHECK_TASK = asyncio.create_task(
                provider_availability_recheck_loop(),
                name="provider_availability_recheck_fallback",
            )
            logger.info(
                "Scheduled provider availability recheck via asyncio loop fallback: interval=%ss initial_delay=%ss",
                PROVIDER_AVAILABILITY_RECHECK_INTERVAL_SECONDS,
                PROVIDER_AVAILABILITY_RECHECK_INITIAL_DELAY_SECONDS,
            )
    if MODEL_CAPABILITY_RECHECK_ENABLED and MODEL_CAPABILITY_CACHE_ENABLED:
        if job_queue is not None:
            job_queue.run_repeating(
                model_capability_recheck_job,
                interval=MODEL_CAPABILITY_RECHECK_INTERVAL_SECONDS,
                first=MODEL_CAPABILITY_RECHECK_INITIAL_DELAY_SECONDS,
                name="model_capability_recheck",
            )
            logger.info(
                "Scheduled model capability recheck via JobQueue: interval=%ss initial_delay=%ss forced_full=%s",
                MODEL_CAPABILITY_RECHECK_INTERVAL_SECONDS,
                MODEL_CAPABILITY_RECHECK_INITIAL_DELAY_SECONDS,
                MODEL_CAPABILITY_RECHECK_FORCE_FULL,
            )
        else:
            MODEL_CAPABILITY_RECHECK_TASK = asyncio.create_task(
                model_capability_recheck_loop(),
                name="model_capability_recheck_fallback",
            )
            logger.info(
                "Scheduled model capability recheck via asyncio loop fallback: interval=%ss initial_delay=%ss forced_full=%s",
                MODEL_CAPABILITY_RECHECK_INTERVAL_SECONDS,
                MODEL_CAPABILITY_RECHECK_INITIAL_DELAY_SECONDS,
                MODEL_CAPABILITY_RECHECK_FORCE_FULL,
            )
    elif MODEL_CAPABILITY_RECHECK_ENABLED and not MODEL_CAPABILITY_CACHE_ENABLED:
        logger.info("Model capability recheck is disabled because capability cache is disabled.")


async def on_app_post_stop(application) -> None:
    await stop_provider_availability_recheck_loop()
    await stop_model_capability_recheck_loop()


async def on_app_post_shutdown(application) -> None:
    await stop_provider_availability_recheck_loop()
    await stop_model_capability_recheck_loop()


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        await reply_to_update(update, "Could not identify user.")
        return
    await reply_to_update(update, f"Your user_id: {user.id}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_owner_command(update):
        return
    lines = [
        "Owner commands:",
        "/start - start bot",
        "/whoami - show your user_id",
        "/api - choose text API/model",
        "/provider - browse providers",
        "/model - choose text model",
        "/modelsearch <text> - search text models",
        "/mode text|image|status - switch response mode",
        "Send photo/file with optional caption in text mode for attachment-aware models.",
        "/memory on|off|status - toggle memory context for you in this chat",
        "/clear - clear chat memory",
        "/allow <user_id> - grant access",
        "/deny <user_id> - revoke access",
        "/list - list allowed users",
        "/keys - show key status",
        "/help - show this help",
    ]
    await reply_to_update(update, "\n".join(lines))


async def keys_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_owner_command(update):
        return
    custom_provider_ids = [pid for pid in TEXT_PROVIDER_HANDLERS.keys() if pid not in {"google", "nvidia"}]
    gemini_model_count = len(PROVIDER_MODEL_KEYS.get("google", []))
    nvidia_model_count = len(PROVIDER_MODEL_KEYS.get("nvidia", []))
    blocked_capabilities = 0
    now_ts = int(time.time())
    for entry in MODEL_CAPABILITIES.values():
        if not isinstance(entry, dict):
            continue
        if int(entry.get("expires_at", 0) or 0) <= now_ts:
            continue
        if str(entry.get("status", "")).strip().lower() in MODEL_CAPABILITY_BLOCKING_STATUSES:
            blocked_capabilities += 1
    with PROVIDER_AVAILABILITY_LOCK:
        provider_cache_total = len(PROVIDER_AVAILABILITY_CACHE)
        provider_cache_unavailable = sum(
            1 for item in PROVIDER_AVAILABILITY_CACHE.values() if str(item.get("status", "")).strip().lower() == "unavailable"
        )
    lines = [
        f"USERS_DB_FILE: {users_db_path}",
        f"ALLOWLIST_FILE: {allowlist_path}",
        f"CUSTOM_PROVIDER_CONFIG_FILE: {CUSTOM_PROVIDER_CONFIG_FILE}",
        f"MODEL_CAPABILITIES_FILE: {model_capabilities_path}",
        f"Model capability cache (memory): {MODEL_CAPABILITY_CACHE_ENABLED}",
        f"Model capability file load: {MODEL_CAPABILITY_LOAD_FROM_FILE}",
        f"Model capability file save: {MODEL_CAPABILITY_PERSIST_TO_FILE}",
        f"MEMORY_CONTEXT_MESSAGES: {MEMORY_CONTEXT_MESSAGES}",
        f"Regular model timeout: {REGULAR_MODEL_TIMEOUT_SECONDS}s",
        f"Reasoning model timeout: {REASONING_MODEL_TIMEOUT_SECONDS}s",
        f"Fallback attempt timeout: {FALLBACK_ATTEMPT_TIMEOUT_SECONDS}s",
        (
            "Attachment limits: "
            f"count={MAX_INPUT_ATTACHMENT_COUNT}, "
            f"file_bytes={MAX_INPUT_ATTACHMENT_BYTES}, "
            f"text_file_bytes={MAX_INPUT_TEXT_ATTACHMENT_BYTES}, "
            f"text_chars={MAX_INPUT_TEXT_ATTACHMENT_CHARS}"
        ),
        f"Telegram reply chunk chars: {TELEGRAM_REPLY_CHUNK_CHARS}",
        f"Model probe enabled: {MODEL_CAPABILITY_PROBE_ENABLED}",
        f"Model probe scope: {MODEL_PROBE_SCOPE}",
        f"Model probe timeout/workers: {MODEL_PROBE_TIMEOUT_SECONDS}s/{MODEL_PROBE_WORKERS}",
        (
            "Model capability recheck: "
            f"{MODEL_CAPABILITY_RECHECK_ENABLED} "
            f"every {MODEL_CAPABILITY_RECHECK_INTERVAL_SECONDS}s "
            f"(initial {MODEL_CAPABILITY_RECHECK_INITIAL_DELAY_SECONDS}s, force_full={MODEL_CAPABILITY_RECHECK_FORCE_FULL})"
        ),
        f"Provider model-list probe timeout: {PROVIDER_MODEL_LIST_PROBE_TIMEOUT_SECONDS}s",
        "Non-reprobe providers: "
        + (", ".join(sorted(NON_REPROBE_PROVIDERS)) if NON_REPROBE_PROVIDERS else "(none)"),
        f"Provider availability TTL (down/up): {PROVIDER_UNAVAILABLE_TTL_SECONDS}s/{PROVIDER_AVAILABLE_TTL_SECONDS}s",
        (
            "Provider availability recheck: "
            f"{PROVIDER_AVAILABILITY_RECHECK_ENABLED} "
            f"every {PROVIDER_AVAILABILITY_RECHECK_INTERVAL_SECONDS}s "
            f"(initial {PROVIDER_AVAILABILITY_RECHECK_INITIAL_DELAY_SECONDS}s)"
        ),
        f"Provider availability cache: {provider_cache_unavailable}/{provider_cache_total} unavailable",
        f"Hide unavailable models: {MODEL_HIDE_UNAVAILABLE_MODELS}",
        f"Model capability cache: {len(MODEL_CAPABILITIES)} entries ({blocked_capabilities} blocking)",
        f"Models active/all: {len(MODEL_ORDER)}/{len(ALL_MODEL_ORDER)}",
        f"Gemini models loaded: {gemini_model_count}",
        f"NVIDIA models loaded: {nvidia_model_count}",
        f"Text providers loaded: {len(TEXT_PROVIDER_HANDLERS)}",
        f"Custom providers loaded: {len(custom_provider_ids)}",
    ]
    for provider_id in sorted(TEXT_PROVIDER_HANDLERS.keys()):
        handler = TEXT_PROVIDER_HANDLERS[provider_id]
        model_count = len(PROVIDER_MODEL_KEYS.get(provider_id, []))
        lines.append(f"{handler.label}: keys={handler.key_count()}, models={model_count}")
    await reply_to_update(update, "\n".join(lines))


async def list_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_owner_command(update):
        return
    rows = list_allowed_users_from_db()
    if not rows:
        await reply_to_update(update, "Allow list is empty.")
        return
    lines = ["Allowed users:"]
    for uid, added_at in rows:
        lines.append(f"- {uid} (added: {added_at})")
    await reply_to_update(update, "\n".join(lines))


async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = await ensure_allowed_chat(update)
    if chat_id is None:
        return
    clear_chat_history(chat_id)
    await reply_to_update(update, "Memory cleared for this chat.")


async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = await ensure_allowed_chat(update)
    if chat_id is None:
        return
    user_id = get_effective_user_id(update)
    if user_id is None:
        await reply_to_update(update, "Could not identify user.")
        return
    arg = first_context_arg(context).lower()
    if not arg:
        state = "on" if is_memory_enabled(chat_id, user_id) else "off"
        await reply_to_update(update, f"Memory is {state} for you in this chat. Usage: /memory on|off|status")
        return
    if arg == "status":
        state = "on" if is_memory_enabled(chat_id, user_id) else "off"
        await reply_to_update(update, f"Memory is {state} for you in this chat.")
        return
    if arg == "on":
        set_memory_enabled(chat_id, user_id, True)
        await reply_to_update(update, "Memory enabled for you in this chat.")
        return
    if arg == "off":
        set_memory_enabled(chat_id, user_id, False)
        await reply_to_update(update, "Memory disabled for you in this chat.")
        return
    await reply_to_update(update, "Usage: /memory on|off|status")


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = await ensure_allowed_chat(update)
    if chat_id is None:
        return
    current = get_chat_mode(chat_id)
    arg = first_context_arg(context).lower()
    if not arg:
        await reply_to_update(update, f"Mode is {current}. Usage: /mode text|image|status")
        return
    if arg == "status":
        await reply_to_update(update, f"Mode is {current}.")
        return
    if arg in {"text", "image"}:
        set_chat_mode(chat_id, arg)
        await reply_to_update(update, f"Mode set to {arg}.")
        return
    await reply_to_update(update, "Usage: /mode text|image|status")


async def allow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_owner_private_command(update):
        return

    uid, error_message = resolve_allowlist_target_user_id(
        update,
        context,
        command_name="allow",
        username_error="Cannot add by @username. Use user_id.",
    )
    if error_message:
        await reply_to_update(update, error_message)
        return
    if uid is None:
        await reply_to_update(update, "Invalid user_id.")
        return
    allowed_user_ids.add(uid)
    add_allowed_user_to_db(uid)
    write_allowlist_backup(allowed_user_ids)
    await reply_to_update(update, f"Added user_id {uid}.")


async def deny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_owner_private_command(update):
        return

    uid, error_message = resolve_allowlist_target_user_id(
        update,
        context,
        command_name="deny",
        username_error="Cannot remove by @username. Use user_id.",
    )
    if error_message:
        await reply_to_update(update, error_message)
        return
    if uid is None:
        await reply_to_update(update, "Invalid user_id.")
        return
    if uid in OWNER_USER_IDS:
        await reply_to_update(update, "Cannot remove owner.")
        return
    allowed_user_ids.discard(uid)
    remove_allowed_user_from_db(uid)
    write_allowlist_backup(allowed_user_ids)
    await reply_to_update(update, f"Removed user_id {uid}.")


async def run_text_generation_flow(
    message,
    *,
    user_id: int,
    chat_id: int,
    prompt: str,
    attachments: list[InputAttachment] | None = None,
    selected_only: bool = False,
    allow_attachment_auto_fallback: bool = True,
    short_response: bool = False,
    provider_allowlist: set[str] | None = None,
    append_used_model_footer: bool = False,
    model_key_override: str | None = None,
) -> tuple[str, str, dict[str, int]]:
    await message.chat.send_action(action="typing")
    if model_key_override and model_key_override in MODEL_ORDER_SET:
        model_key = model_key_override
    else:
        model_key = get_user_model_key(user_id)
    group_chat = is_group_chat_type(message.chat)
    history: list[dict[str, str]] = []
    if group_chat:
        clear_chat_history(chat_id)
        reply = getattr(message, "reply_to_message", None)
        if reply is not None:
            reply_text = str(getattr(reply, "text", "") or getattr(reply, "caption", "") or "").strip()
            if reply_text:
                reply_user = getattr(reply, "from_user", None)
                role = "assistant" if reply_user and getattr(reply_user, "is_bot", False) else "user"
                history = [{"role": role, "content": reply_text}]
        memory_on = False
    else:
        memory_on = is_memory_enabled(chat_id, user_id)
        history = get_recent_history(chat_id, MEMORY_CONTEXT_MESSAGES) if memory_on else []
    normalized_attachments = attachments or []
    model_prompt = build_model_prompt(prompt, short=short_response)
    answer, used_model_name, used_provider_id, usage = await asyncio.to_thread(
        generate_text,
        model_prompt,
        model_key,
        history,
        normalized_attachments,
        allow_attachment_auto_fallback=allow_attachment_auto_fallback,
        selected_only=selected_only,
        provider_allowlist=provider_allowlist,
    )
    logger.info(
        "Model usage: user_id=%s chat_id=%s provider=%s model=%s attachments=%d prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        user_id,
        chat_id,
        used_provider_id,
        used_model_name,
        len(normalized_attachments),
        usage.get("prompt_tokens", "n/a"),
        usage.get("completion_tokens", "n/a"),
        usage.get("total_tokens", "n/a"),
    )
    if memory_on:
        stored_prompt = prompt
        if normalized_attachments:
            summary = ", ".join(describe_attachment(item) for item in normalized_attachments)
            stored_prompt = f"{prompt}\n[attachments: {summary}]"
        add_history_messages(
            chat_id,
            user_id,
            [
                ("user", stored_prompt),
                ("assistant", answer),
            ],
        )
    reply_text = answer
    if append_used_model_footer:
        reply_text = (
            f"{reply_text.rstrip()}\n\n"
            f"Used provider/model: `{used_provider_id}/{used_model_name}`"
        )
    await send_reply_text(message, reply_text)
    return used_provider_id, used_model_name, usage


def extract_prompt_from_message(message) -> str:
    return str(message.text or message.caption or "").strip()


def is_group_chat_type(chat) -> bool:
    return chat is not None and chat.type in {"group", "supergroup"}


def strip_keyword(text: str, keyword: str) -> str:
    base = str(text or "")
    needle = str(keyword or "").strip()
    if not base or not needle:
        return base.strip()
    cleaned = re.sub(re.escape(needle), "", base, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def is_group_chat(update: Update) -> bool:
    return is_group_chat_type(update.effective_chat)


def should_trigger_group_keyword(update: Update, prompt: str) -> bool:
    if not GROUP_KEYWORD_TRIGGER:
        return False
    if not is_group_chat(update):
        return False
    text = str(prompt or "")
    if not text:
        return False
    if GROUP_KEYWORD_TRIGGER.casefold() not in text.casefold():
        return False
    if GROUP_KEYWORD_REQUIRE_QUESTION and "?" not in text:
        return False
    return True


def should_trigger_group_photo_keyword(update: Update, prompt: str) -> bool:
    if not GROUP_PHOTO_KEYWORD:
        return False
    if not is_group_chat(update):
        return False
    text = str(prompt or "")
    if not text:
        return False
    return GROUP_PHOTO_KEYWORD.casefold() in text.casefold()


def is_reply_to_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    message = update.effective_message
    if message is None:
        return False
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return False
    reply_user = getattr(reply, "from_user", None)
    if reply_user is None:
        return False
    bot_id = getattr(context.bot, "id", None)
    if bot_id is None:
        return bool(getattr(reply_user, "is_bot", False))
    return reply_user.id == bot_id


async def maybe_handle_model_search_input(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    message,
    *,
    prompt: str,
    attachments: list[InputAttachment],
) -> bool:
    if not context.user_data.get("model_search_waiting_for_query"):
        return False
    if attachments:
        await message.reply_text("Model search accepts text only. Send text query without attachments.")
        return True
    context.user_data["model_search_waiting_for_query"] = False
    context.user_data["model_search_query"] = prompt
    model_key = get_user_model_key(get_effective_user_id(update))
    text, keyboard = build_model_search_view(model_key, prompt, 0)
    await message.reply_text(text, reply_markup=keyboard)
    return True


async def maybe_run_image_mode_flow(
    message,
    *,
    user_id: int,
    chat_id: int,
    prompt: str,
    model_key: str,
    attachments: list[InputAttachment],
    mode: str,
    provider_allowlist: set[str] | None = None,
) -> bool:
    if mode != "image":
        return False
    if attachments:
        await message.reply_text("Attachment analysis is available in text mode. Use /mode text.")
        return True

    if is_group_chat_type(message.chat):
        clear_chat_history(chat_id)
    await message.chat.send_action(action="upload_photo")
    prefer_default_image_model = bool(DEFAULT_IMAGE_MODEL_NAME)
    image_data, used_model_name, used_provider_id = await asyncio.to_thread(
        generate_image,
        prompt,
        model_key,
        provider_allowlist=provider_allowlist,
        prefer_default_image_model=prefer_default_image_model,
    )
    logger.info(
        "Image generation usage: user_id=%s chat_id=%s provider=%s model=%s mode=image",
        user_id,
        chat_id,
        used_provider_id,
        used_model_name,
    )
    caption = f"Model: {used_provider_id}/{used_model_name}"
    await send_reply_image(message, image_data, caption=caption)
    return True


async def maybe_run_attachment_text_flow(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    *,
    user_id: int,
    chat_id: int,
    prompt: str,
    attachments: list[InputAttachment],
    short_response: bool,
    provider_allowlist: set[str] | None,
) -> bool:
    if not attachments:
        return False
    try:
        await run_text_generation_flow(
            message,
            user_id=user_id,
            chat_id=chat_id,
            prompt=prompt,
            attachments=attachments,
            selected_only=True,
            allow_attachment_auto_fallback=False,
            short_response=short_response,
            provider_allowlist=provider_allowlist,
        )
        return True
    except AttachmentDecisionRequiredError as exc:
        set_pending_attachment_request(
            context,
            prompt=prompt,
            attachments=attachments,
            chat_id=chat_id,
        )
        current_model_key = get_user_model_key(user_id)
        await message.reply_text(
            "Current model cannot handle this attachment.\n"
            f"Model: {get_model_label(current_model_key)}\n"
            f"Reason: {exc.reason or 'unsupported attachment input'}\n\n"
            "Choose how to proceed:",
            reply_markup=build_attachment_decision_keyboard(),
        )
        return True


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    if is_special_denied_user(update.effective_user):
        return

    prompt = extract_prompt_from_message(message)
    keyword_trigger = should_trigger_group_keyword(update, prompt)
    photo_keyword_trigger = should_trigger_group_photo_keyword(update, prompt)
    reply_trigger = is_reply_to_bot(update, context)
    group_chat = is_group_chat(update)
    group_trigger = group_chat and (keyword_trigger or reply_trigger or photo_keyword_trigger)
    short_response = group_chat
    provider_allowlist = GROUP_PROVIDER_ALLOWLIST if group_chat else None
    if group_chat and not group_trigger:
        return
    if not group_trigger:
        if not await ensure_allowed_command(update):
            return
    attachments, attachment_notices = await extract_message_attachments(message)
    had_media = bool(message.photo or message.document)
    if had_media and attachment_notices:
        logger.info("Attachment parsing notices: %s", "; ".join(attachment_notices))
    if had_media and not attachments:
        reason = attachment_notices[0] if attachment_notices else "Unsupported attachment type."
        await message.reply_text(f"Could not use attachment: {reason}")
        return
    if not prompt:
        if attachments:
            prompt = "Analyze the attached content and help the user."
        else:
            await message.reply_text("Empty message.")
            return

    if not group_trigger:
        if await maybe_handle_model_search_input(
            update,
            context,
            message,
            prompt=prompt,
            attachments=attachments,
        ):
            return

    try:
        user_id = get_effective_user_id(update)
        chat_id = get_effective_chat_id(update) or user_id
        if user_id is None or chat_id is None:
            await message.reply_text("Chat/user not found.")
            return

        mode = get_chat_mode(chat_id)
        model_key = get_user_model_key(user_id)
        group_fast_model_key = get_group_fast_model_key(provider_allowlist) if group_chat else ""
        if group_chat and photo_keyword_trigger:
            image_prompt = strip_keyword(prompt, GROUP_PHOTO_KEYWORD)
            if not image_prompt:
                image_prompt = "Generate an image."
            await maybe_run_image_mode_flow(
                message,
                user_id=user_id,
                chat_id=chat_id,
                prompt=image_prompt,
                model_key=model_key,
                attachments=[],
                mode="image",
                provider_allowlist=provider_allowlist,
            )
            return

        if await maybe_run_image_mode_flow(
            message,
            user_id=user_id,
            chat_id=chat_id,
            prompt=prompt,
            model_key=model_key,
            attachments=attachments,
            mode=mode,
            provider_allowlist=provider_allowlist,
        ):
            return

        if await maybe_run_attachment_text_flow(
            context,
            message,
            user_id=user_id,
            chat_id=chat_id,
            prompt=prompt,
            attachments=attachments,
            short_response=short_response,
            provider_allowlist=provider_allowlist,
        ):
            return

        await run_text_generation_flow(
            message,
            user_id=user_id,
            chat_id=chat_id,
            prompt=prompt,
            attachments=[],
            selected_only=False,
            allow_attachment_auto_fallback=True,
            short_response=short_response,
            provider_allowlist=provider_allowlist,
            model_key_override=group_fast_model_key,
        )
    except Exception as exc:
        logger.exception("Message generation failed")
        await message.reply_text(f"Generation error: {exc}")


async def inline_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    inline_query = update.inline_query
    if inline_query is None:
        return
    user = inline_query.from_user
    if not is_allowed(user):
        if is_special_denied_user(user):
            return
        denied_text = "Access denied. Ask owner to add you."
        await inline_query.answer(
            [],
            cache_time=1,
            is_personal=True,
            switch_pm_text=denied_text,
            switch_pm_parameter="access_denied",
        )
        return

    candidate_keys = build_inline_model_keys()
    if not candidate_keys:
        await inline_query.answer(
            [],
            cache_time=30,
            is_personal=True,
            switch_pm_text="Inline mode unavailable.",
            switch_pm_parameter="inline_unavailable",
        )
        return

    prompt = str(inline_query.query or "").strip()
    result_id = build_inline_result_id(user.id if user else None, prompt)
    title = build_inline_result_title(prompt)
    description = "Generate with Gemini Flash"
    placeholder = InputTextMessageContent(INLINE_PLACEHOLDER_TEXT)
    reply_markup = build_inline_keyboard()
    result = InlineQueryResultArticle(
        id=result_id,
        title=title,
        description=description,
        input_message_content=placeholder,
        reply_markup=reply_markup,
    )
    await inline_query.answer([result], cache_time=1, is_personal=True)


async def run_inline_generation(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    inline_message_id: str,
    prompt: str,
    user_id: int | None,
) -> None:
    try:
        model_prompt = build_inline_prompt(prompt)
        answer, used_model_name, used_provider_id, usage = await asyncio.to_thread(
            generate_inline_text,
            model_prompt,
        )
        logger.info(
            "Inline usage: user_id=%s provider=%s model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            user_id,
            used_provider_id,
            used_model_name,
            usage.get("prompt_tokens", "n/a"),
            usage.get("completion_tokens", "n/a"),
            usage.get("total_tokens", "n/a"),
        )
        formatted = format_inline_answer(prompt, answer)
        await edit_inline_message_text(context.bot, inline_message_id, formatted)
    except Exception as exc:
        logger.exception("Inline generation failed")
        await edit_inline_message_text(context.bot, inline_message_id, f"Generation error: {exc}")


async def run_inline_generation_with_delay(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    inline_message_id: str,
    prompt: str,
    user_id: int | None,
    delay_seconds: int,
) -> None:
    try:
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        await run_inline_generation(
            context,
            inline_message_id=inline_message_id,
            prompt=prompt,
            user_id=user_id,
        )
    except asyncio.CancelledError:
        try:
            await edit_inline_message_text(context.bot, inline_message_id, "Canceled (superseded).")
        finally:
            raise
    finally:
        if user_id is not None:
            current = asyncio.current_task()
            with INLINE_PENDING_LOCK:
                entry = INLINE_PENDING_REQUESTS.get(user_id)
                if entry and entry.get("task") is current:
                    INLINE_PENDING_REQUESTS.pop(user_id, None)


async def chosen_inline_result_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chosen = update.chosen_inline_result
    if chosen is None:
        return
    user = chosen.from_user
    if not is_allowed(user):
        if chosen.inline_message_id:
            if is_special_denied_user(user):
                return
            await edit_inline_message_text(context.bot, chosen.inline_message_id, "Access denied.")
        return
    inline_message_id = chosen.inline_message_id
    if not inline_message_id:
        logger.info(
            "Inline chosen result missing inline_message_id (user_id=%s).",
            user.id if user else "unknown",
        )
        return
    prompt = str(chosen.query or "").strip() or "Help the user."
    user_id = user.id if user else None
    logger.info(
        "Inline chosen: user_id=%s inline_message_id=%s query_len=%d debounce=%ss",
        user_id,
        inline_message_id,
        len(prompt),
        INLINE_DEBOUNCE_SECONDS,
    )
    if user_id is None:
        context.application.create_task(
            run_inline_generation_with_delay(
                context,
                inline_message_id=inline_message_id,
                prompt=prompt,
                user_id=None,
                delay_seconds=INLINE_DEBOUNCE_SECONDS,
            )
        )
        return

    with INLINE_PENDING_LOCK:
        existing = INLINE_PENDING_REQUESTS.get(user_id)
        if existing:
            task = existing.get("task")
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()
        new_task = context.application.create_task(
            run_inline_generation_with_delay(
                context,
                inline_message_id=inline_message_id,
                prompt=prompt,
                user_id=user_id,
                delay_seconds=INLINE_DEBOUNCE_SECONDS,
            )
        )
        INLINE_PENDING_REQUESTS[user_id] = {
            "task": new_task,
            "inline_message_id": inline_message_id,
            "prompt": prompt,
            "created_at": time.time(),
        }


def build_app():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(8)
        .post_init(on_app_post_init)
        .post_stop(on_app_post_stop)
        .post_shutdown(on_app_post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("keys", keys_status))
    app.add_handler(CommandHandler("list", list_allowed))
    app.add_handler(CommandHandler("clear", clear_memory))
    app.add_handler(CommandHandler("memory", memory_command))
    app.add_handler(CommandHandler("mode", mode_command))
    app.add_handler(CommandHandler("allow", allow))
    app.add_handler(CommandHandler("deny", deny))
    app.add_handler(CommandHandler("api", model_menu))
    app.add_handler(CommandHandler("provider", provider_menu))
    app.add_handler(CommandHandler("model", model_menu))
    app.add_handler(CommandHandler("modelsearch", model_search))
    app.add_handler(CallbackQueryHandler(on_model_providers_callback, pattern=r"^modelproviders:"))
    app.add_handler(CallbackQueryHandler(on_model_provider_callback, pattern=r"^modelprovider:"))
    app.add_handler(CallbackQueryHandler(on_model_search_prompt_callback, pattern=r"^modelsearchprompt$"))
    app.add_handler(CallbackQueryHandler(on_model_search_page_callback, pattern=r"^modelsearchpage:"))
    app.add_handler(CallbackQueryHandler(on_attachment_resolution_callback, pattern=r"^attachresolve:"))
    app.add_handler(CallbackQueryHandler(on_model_page_callback, pattern=r"^modelpage:"))
    app.add_handler(CallbackQueryHandler(on_model_callback, pattern=r"^model:"))
    app.add_handler(InlineQueryHandler(inline_query_handler))
    app.add_handler(ChosenInlineResultHandler(chosen_inline_result_handler))
    app.add_handler(
        MessageHandler((filters.TEXT | filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND, handle_message)
    )
    return app


def main() -> None:
    # Startup blocks until capability check and filtering complete.
    run_startup_model_capability_probe(force_full=MODEL_CAPABILITY_RECHECK_FORCE_FULL)
    apply_model_capability_filter(force=True, strict=True)
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
