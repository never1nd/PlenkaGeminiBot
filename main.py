import asyncio
import concurrent.futures
import hashlib
import html
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from dotenv import load_dotenv
from provider_handlers import (
    BaseProviderHandler,
    load_external_provider_handlers,
)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DEFAULT_OWNER_USER_ID = 8082486311

OWNER_USER_ID = int(os.getenv("OWNER_USER_ID", str(DEFAULT_OWNER_USER_ID)) or str(DEFAULT_OWNER_USER_ID))

ALLOWLIST_FILE = os.getenv("ALLOWLIST_FILE", "allowlist.json").strip()
USERS_DB_FILE = os.getenv("USERS_DB_FILE", "users.db").strip()
MODEL_PREFS_FILE = os.getenv("MODEL_PREFS_FILE", "model_prefs.json").strip()
MEMORY_CONTEXT_MESSAGES = int(os.getenv("MEMORY_CONTEXT_MESSAGES", "20") or "20")
CUSTOM_PROVIDER_CONFIG_FILE = os.getenv("CUSTOM_PROVIDER_CONFIG_FILE", "providers.json").strip()
MODEL_CAPABILITIES_FILE = os.getenv("MODEL_CAPABILITIES_FILE", "model_capabilities.json").strip()

REGULAR_MODEL_TIMEOUT_SECONDS = max(1, int(os.getenv("REGULAR_MODEL_TIMEOUT_SECONDS", "90") or "90"))
REASONING_MODEL_TIMEOUT_SECONDS = max(1, int(os.getenv("REASONING_MODEL_TIMEOUT_SECONDS", "180") or "180"))
FALLBACK_ATTEMPT_TIMEOUT_SECONDS = max(
    5, int(os.getenv("FALLBACK_ATTEMPT_TIMEOUT_SECONDS", "35") or "35")
)
MODEL_CAPABILITY_TTL_SECONDS = max(60, int(os.getenv("MODEL_CAPABILITY_TTL_SECONDS", "21600") or "21600"))
_model_probe_timeout_env = int(os.getenv("MODEL_PROBE_TIMEOUT_SECONDS", "0") or "0")
if _model_probe_timeout_env > 0:
    MODEL_PROBE_TIMEOUT_SECONDS = max(3, _model_probe_timeout_env)
else:
    # Auto mode: probe timeout follows the slowest generation timeout.
    MODEL_PROBE_TIMEOUT_SECONDS = max(REGULAR_MODEL_TIMEOUT_SECONDS, REASONING_MODEL_TIMEOUT_SECONDS)
MODEL_PROBE_WORKERS = max(1, int(os.getenv("MODEL_PROBE_WORKERS", "8") or "8"))
MODEL_PROBE_SCOPE = os.getenv("MODEL_PROBE_SCOPE", "smart").strip().lower() or "smart"
MODEL_PROBE_MAX_MODELS = max(0, int(os.getenv("MODEL_PROBE_MAX_MODELS", "0") or "0"))
MODEL_CAPABILITY_PROBE_ENABLED = os.getenv("MODEL_CAPABILITY_PROBE_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
STARTUP_PROBE_MODE = os.getenv("STARTUP_PROBE_MODE", "none").strip().lower()
MODEL_HIDE_UNAVAILABLE_MODELS = os.getenv("MODEL_HIDE_UNAVAILABLE_MODELS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024") or "1024")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

MODEL_GEMINI_KEY_PREFIX = "gm_"
MODEL_NVIDIA_KEY_PREFIX = "nv_"
MODEL_CUSTOM_PROVIDER_KEY_PREFIX = "cp_"
MODEL_MENU_PAGE_SIZE = 12

MODEL_LABELS: dict[str, str] = {}
GEMINI_MODEL_NAMES: dict[str, str] = {}
NVIDIA_MODEL_NAMES: dict[str, str] = {}
CUSTOM_PROVIDER_MODELS: dict[str, dict[str, str]] = {}
TEXT_PROVIDER_HANDLERS: dict[str, BaseProviderHandler] = {}
MODEL_PROVIDER_BY_KEY: dict[str, str] = {}
MODEL_NAME_BY_KEY: dict[str, str] = {}
PROVIDER_DISPLAY_NAMES: dict[str, str] = {"google": "Google Gemini", "nvidia": "NVIDIA"}
MODEL_ORDER: list[str] = []
MODEL_PROVIDER_PRIORITY: dict[str, list[str]] = {}
ALL_MODEL_ORDER: list[str] = []
MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {}
MODEL_CAPABILITIES_LOCK = threading.Lock()

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

DEFAULT_TEXT_PROVIDER = ""

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("bot")

if not TELEGRAM_BOT_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN is required")

allowlist_path = Path(ALLOWLIST_FILE)
users_db_path = Path(USERS_DB_FILE)
model_prefs_path = Path(MODEL_PREFS_FILE)
model_capabilities_path = Path(MODEL_CAPABILITIES_FILE)


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


def unique_keep_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def get_model_key_prefix(provider_id: str) -> str:
    if provider_id == "google":
        return MODEL_GEMINI_KEY_PREFIX
    if provider_id == "nvidia":
        return MODEL_NVIDIA_KEY_PREFIX
    return f"{MODEL_CUSTOM_PROVIDER_KEY_PREFIX}{provider_id}_"


def model_capability_key(provider_id: str, model_name: str) -> str:
    provider = str(provider_id).strip().lower()
    model = str(model_name).strip().lower()
    return f"{provider}::{model}"


def load_model_capabilities() -> dict[str, dict[str, Any]]:
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


def classify_generation_error(error: str) -> str:
    text = str(error).lower()
    unsupported_markers = (
        "model not found",
        "not found for account",
        "unknown model",
        "unsupported model",
        "invalid model",
        "model is not available",
        "not available for account",
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
        if persist and updated:
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
    entry = get_model_capability(provider_id, model_name)
    if not entry:
        return False, ""
    status = str(entry.get("status", "")).strip().lower()
    if status in MODEL_CAPABILITY_BLOCKING_STATUSES:
        return True, status
    return False, status


def load_model_provider_priority() -> dict[str, list[str]]:
    config_path = Path(CUSTOM_PROVIDER_CONFIG_FILE)
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse provider priority config in %s: %s", config_path, exc)
        return {}

    raw_priority = payload.get("model_provider_priority", {}) if isinstance(payload, dict) else {}
    providers: list[str] = []
    ignored_model_specific_rules = 0

    if isinstance(raw_priority, dict):
        for rule_key in raw_priority.keys():
            if str(rule_key).strip() != "*":
                ignored_model_specific_rules += 1
        wildcard = raw_priority.get("*", [])
        if isinstance(wildcard, list):
            providers = [str(x).strip().lower() for x in wildcard if str(x).strip()]
    elif isinstance(raw_priority, list):
        providers = [str(x).strip().lower() for x in raw_priority if str(x).strip()]
    else:
        logger.warning("model_provider_priority must be an object (with '*') or a list in %s", config_path)
        return {}

    provider_ids = unique_keep_order(providers)
    if ignored_model_specific_rules:
        logger.info(
            "Ignoring %d model-specific provider priority rules; using only general '*' priority.",
            ignored_model_specific_rules,
        )
    if provider_ids:
        logger.info("Loaded general provider priority with %d providers.", len(provider_ids))
        return {"*": provider_ids}
    return {}


def build_text_provider_handlers() -> dict[str, BaseProviderHandler]:
    return load_external_provider_handlers(CUSTOM_PROVIDER_CONFIG_FILE)


def build_model_catalog() -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, str]],
    dict[str, BaseProviderHandler],
    dict[str, str],
    list[str],
    dict[str, str],
    dict[str, str],
]:
    labels: dict[str, str] = {}
    gemini_model_names: dict[str, str] = {}
    nvidia_model_names: dict[str, str] = {}
    custom_provider_models: dict[str, dict[str, str]] = {}
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
            labels[model_key] = build_model_label(model_name)
            model_provider_by_key[model_key] = provider_id
            model_name_by_key[model_key] = model_name
            if provider_id == "google":
                gemini_model_names[model_key] = model_name
            elif provider_id == "nvidia":
                nvidia_model_names[model_key] = model_name
            else:
                custom_provider_models[model_key] = {"provider_id": provider_id, "model_name": model_name}
            order.append(model_key)

    return (
        labels,
        gemini_model_names,
        nvidia_model_names,
        custom_provider_models,
        provider_handlers,
        provider_labels,
        order,
        model_provider_by_key,
        model_name_by_key,
    )


(
    MODEL_LABELS,
    GEMINI_MODEL_NAMES,
    NVIDIA_MODEL_NAMES,
    CUSTOM_PROVIDER_MODELS,
    TEXT_PROVIDER_HANDLERS,
    PROVIDER_DISPLAY_NAMES,
    MODEL_ORDER,
    MODEL_PROVIDER_BY_KEY,
    MODEL_NAME_BY_KEY,
) = build_model_catalog()


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
USER_CTX_MODEL_SEARCH_WAITING = "model_search_waiting_for_query"
USER_CTX_MODEL_SEARCH_QUERY = "model_search_query"


def build_provider_index() -> tuple[list[str], dict[str, list[str]]]:
    mapping: dict[str, list[str]] = {}
    for model_key in MODEL_ORDER:
        provider_id = get_model_provider_id(model_key)
        mapping.setdefault(provider_id, []).append(model_key)
    priority = {"google": 0, "nvidia": 1}
    provider_ids = sorted(mapping.keys(), key=lambda x: (priority.get(x, 2), x))
    return provider_ids, mapping


PROVIDER_ORDER, PROVIDER_MODEL_KEYS = build_provider_index()
MODEL_PROVIDER_PRIORITY = load_model_provider_priority()
MODEL_CAPABILITIES = load_model_capabilities()


def init_users_db() -> None:
    with sqlite3.connect(users_db_path) as conn:
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
                chat_id INTEGER PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.commit()


def add_allowed_user_to_db(user_id: int) -> None:
    with sqlite3.connect(users_db_path) as conn:
        conn.execute("INSERT OR IGNORE INTO allowed_users(user_id) VALUES (?)", (user_id,))
        conn.commit()


def remove_allowed_user_from_db(user_id: int) -> None:
    with sqlite3.connect(users_db_path) as conn:
        conn.execute("DELETE FROM allowed_users WHERE user_id = ?", (user_id,))
        conn.commit()


def load_allowed_user_ids_from_db() -> set[int]:
    with sqlite3.connect(users_db_path) as conn:
        rows = conn.execute("SELECT user_id FROM allowed_users").fetchall()
    return {int(row[0]) for row in rows}


def list_allowed_users_from_db() -> list[tuple[int, str]]:
    with sqlite3.connect(users_db_path) as conn:
        rows = conn.execute(
            "SELECT user_id, added_at FROM allowed_users ORDER BY added_at DESC, user_id DESC"
        ).fetchall()
    return [(int(row[0]), str(row[1])) for row in rows]


def encrypt_text(plain: str) -> str:
    return plain


def decrypt_text(cipher: str) -> str:
    return cipher


def is_memory_enabled(chat_id: int) -> bool:
    with sqlite3.connect(users_db_path) as conn:
        row = conn.execute("SELECT enabled FROM memory_settings WHERE chat_id = ?", (chat_id,)).fetchone()
    if row is None:
        return True
    return bool(int(row[0]))


def set_memory_enabled(chat_id: int, enabled: bool) -> None:
    with sqlite3.connect(users_db_path) as conn:
        conn.execute(
            """
            INSERT INTO memory_settings(chat_id, enabled)
            VALUES(?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET enabled = excluded.enabled
            """,
            (chat_id, 1 if enabled else 0),
        )
        conn.commit()


def add_history_message(chat_id: int, user_id: int, role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        return
    enc = encrypt_text(content)
    with sqlite3.connect(users_db_path) as conn:
        conn.execute(
            "INSERT INTO chat_history(chat_id, user_id, role, content_enc) VALUES (?, ?, ?, ?)",
            (chat_id, user_id, role, enc),
        )
        conn.commit()


def clear_chat_history(chat_id: int) -> None:
    with sqlite3.connect(users_db_path) as conn:
        conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
        conn.commit()


def get_recent_history(chat_id: int, limit: int) -> list[dict[str, str]]:
    with sqlite3.connect(users_db_path) as conn:
        rows = conn.execute(
            """
            SELECT role, content_enc
            FROM chat_history
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
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
                if model_key in MODEL_ORDER:
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
if OWNER_USER_ID:
    allowed_user_ids.add(OWNER_USER_ID)
write_allowlist_backup(allowed_user_ids)
user_model_prefs = load_model_prefs()


def is_allowed(user) -> bool:
    if not user:
        return False
    if user.id in allowed_user_ids:
        return True
    return False


def is_owner(user) -> bool:
    if not user:
        return False
    if OWNER_USER_ID and user.id == OWNER_USER_ID:
        return True
    return False


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
    if model_key not in MODEL_ORDER:
        return DEFAULT_TEXT_PROVIDER
    return model_key


def set_user_model_key(user_id: int, model_key: str) -> None:
    if model_key not in MODEL_ORDER:
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
            parts.append(html.escape(before))

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
        parts.append(html.escape(tail))

    rendered = "".join(parts)
    rendered = re.sub(r"`([^`\n]+)`", lambda m: f"<code>{html.escape(m.group(1))}</code>", rendered)
    return rendered


async def send_reply_text(message, text: str) -> None:
    html_text = markdown_code_to_telegram_html(text)
    try:
        await message.reply_text(html_text, parse_mode=ParseMode.HTML)
    except BadRequest:
        await message.reply_text(text)


def generate_text_with_handler(
    provider_id: str,
    prompt: str,
    model_name: str,
    history: list[dict[str, str]],
    *,
    timeout_seconds: int | None = None,
) -> Tuple[str, str, dict[str, int]]:
    handler = TEXT_PROVIDER_HANDLERS.get(provider_id)
    if not handler:
        raise RuntimeError(f"Provider handler is not loaded: {provider_id}")
    effective_timeout = timeout_seconds if timeout_seconds is not None else get_timeout_for_model(model_name)
    text, usage = handler.generate_text(
        prompt,
        model_name,
        history,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout_seconds=effective_timeout,
        strip_reasoning=strip_reasoning,
    )
    if not text:
        raise RuntimeError(f"{model_name} returned an empty response.")
    return text, model_name, usage


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


def get_provider_priority_for_model(_model_name: str, primary_provider: str) -> list[str]:
    ordered: list[str] = []
    configured: list[str] = list(MODEL_PROVIDER_PRIORITY.get("*", []))

    for provider_id in configured + PROVIDER_ORDER:
        provider_id = str(provider_id).strip().lower()
        if not provider_id or provider_id == primary_provider:
            continue
        if provider_id in ordered:
            continue
        if provider_id not in PROVIDER_MODEL_KEYS:
            continue
        handler = TEXT_PROVIDER_HANDLERS.get(provider_id)
        if handler and handler.key_count() == 0:
            continue
        ordered.append(provider_id)
    return ordered


def build_alternate_model_keys(primary_key: str) -> list[str]:
    primary_provider = MODEL_PROVIDER_BY_KEY.get(primary_key, "")
    primary_model_name = MODEL_NAME_BY_KEY.get(primary_key, "")
    candidates: list[str] = []
    seen: set[str] = {primary_key}

    # Try only the same model on alternate providers, ordered by configured priorities.
    for provider_id in get_provider_priority_for_model(primary_model_name, primary_provider):
        model_key = find_provider_model_key(provider_id, primary_model_name)
        if not model_key or model_key in seen:
            continue
        seen.add(model_key)
        candidates.append(model_key)

    return candidates


def build_model_probe_keys() -> list[str]:
    scope = MODEL_PROBE_SCOPE if MODEL_PROBE_SCOPE in {"all", "smart"} else "smart"
    if scope == "all":
        keys = list(ALL_MODEL_ORDER or MODEL_ORDER)
        if MODEL_PROBE_MAX_MODELS > 0:
            return keys[:MODEL_PROBE_MAX_MODELS]
        return keys

    result: list[str] = []
    seen: set[str] = set()

    # Probe general provider-priority fallback models first.
    general_priority = MODEL_PROVIDER_PRIORITY.get("*", [])
    for provider_id in unique_keep_order([str(x).strip().lower() for x in general_priority]):
        handler = TEXT_PROVIDER_HANDLERS.get(provider_id)
        if not handler:
            continue
        fallback_models = getattr(handler, "fallback_models", []) or []
        if not isinstance(fallback_models, list):
            continue
        for model_name in fallback_models:
            model_key = find_provider_model_key(provider_id, str(model_name))
            if model_key and model_key not in seen:
                seen.add(model_key)
                result.append(model_key)

    # Probe each provider fallback model list as it is commonly used during retries.
    for provider_id, handler in TEXT_PROVIDER_HANDLERS.items():
        fallback_models = getattr(handler, "fallback_models", []) or []
        if not isinstance(fallback_models, list):
            continue
        for model_name in fallback_models:
            model_key = find_provider_model_key(provider_id, str(model_name))
            if model_key and model_key not in seen:
                seen.add(model_key)
                result.append(model_key)

    if DEFAULT_TEXT_PROVIDER and DEFAULT_TEXT_PROVIDER not in seen:
        seen.add(DEFAULT_TEXT_PROVIDER)
        result.append(DEFAULT_TEXT_PROVIDER)

    if not result:
        result = list(ALL_MODEL_ORDER or MODEL_ORDER)

    if MODEL_PROBE_MAX_MODELS > 0:
        result = result[:MODEL_PROBE_MAX_MODELS]
    return result


def probe_single_model_capability(model_key: str) -> tuple[str, str, str]:
    provider_id = MODEL_PROVIDER_BY_KEY.get(model_key, "")
    model_name = MODEL_NAME_BY_KEY.get(model_key, "")
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
        set_model_capability(provider_id, model_name, status, error_text=error_text, persist=False)
        return status, provider_id, model_name


def run_startup_model_capability_probe(*, force_full: bool = False) -> None:
    if not MODEL_CAPABILITY_PROBE_ENABLED and not force_full:
        logger.info("Model capability probe is disabled (MODEL_CAPABILITY_PROBE_ENABLED=0).")
        return
    if force_full:
        probe_keys = list(ALL_MODEL_ORDER or MODEL_ORDER)
    else:
        probe_keys = build_model_probe_keys()
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
    global MODEL_ORDER, PROVIDER_ORDER, PROVIDER_MODEL_KEYS, DEFAULT_TEXT_PROVIDER
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

    PROVIDER_ORDER, PROVIDER_MODEL_KEYS = build_provider_index()
    DEFAULT_TEXT_PROVIDER = MODEL_ORDER[0] if MODEL_ORDER else ""
    if hidden:
        logger.info("Capability filter hidden %d unavailable model entries.", hidden)


def generate_text(prompt: str, model_key: str, history: list[dict[str, str]]) -> Tuple[str, str, str, dict[str, int]]:
    chosen_key = model_key if model_key in MODEL_ORDER else DEFAULT_TEXT_PROVIDER
    if not chosen_key:
        raise RuntimeError("No text models available.")

    raw_attempt_keys: list[str] = [chosen_key]
    raw_attempt_keys.extend(build_alternate_model_keys(chosen_key))
    attempt_keys: list[str] = []
    skipped_candidates: list[str] = []
    for candidate_key in raw_attempt_keys:
        provider_id = MODEL_PROVIDER_BY_KEY.get(candidate_key, "")
        model_name = MODEL_NAME_BY_KEY.get(candidate_key, "")
        if not provider_id or not model_name:
            continue
        should_skip, status = should_skip_model_candidate(provider_id, model_name)
        if should_skip:
            skipped_candidates.append(f"{provider_id}:{model_name}({status})")
            continue
        attempt_keys.append(candidate_key)
    if skipped_candidates:
        logger.info("Skipping cached-unavailable candidates: %s", ", ".join(skipped_candidates))
    if not attempt_keys:
        attempt_keys = list(raw_attempt_keys)

    started_at = time.monotonic()
    attempted: list[str] = []
    last_error: Exception | None = None
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
                timeout_seconds=per_attempt_timeout,
            )
            set_model_capability(provider_id, model_name, MODEL_CAPABILITY_STATUS_AVAILABLE)
            return text, used_model, provider_id, usage
        except Exception as exc:
            last_error = exc
            status = classify_generation_error(exc)
            set_model_capability(provider_id, model_name, status, error_text=str(exc))
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
        raise RuntimeError(
            f"Generation failed after trying: {attempts_str}. Last error: {last_error}"
        ) from last_error
    raise RuntimeError("No text models available.")


def get_model_menu_page_size() -> int:
    return MODEL_MENU_PAGE_SIZE


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
    if model_key not in MODEL_ORDER:
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user):
        await update.message.reply_text("Access denied. Ask owner to add you.")
        return
    user_id = update.effective_user.id if update.effective_user else None
    model_key = get_user_model_key(user_id)
    await update.message.reply_text(
        f"Ready. Current model: {get_model_label(model_key)}\n"
        "Send text, use /model to browse, or /modelsearch <text>."
    )


async def model_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user):
        await update.message.reply_text("Access denied.")
        return
    user_id = update.effective_user.id if update.effective_user else None
    model_key = get_user_model_key(user_id)
    page = find_model_page(model_key)
    page_count = get_model_page_count()
    await update.message.reply_text(
        f"Current model: {get_model_label(model_key)}\n"
        f"Models: {len(MODEL_ORDER)} across {len(PROVIDER_ORDER)} providers.\n"
        f"Select model ({page + 1}/{page_count}):",
        reply_markup=build_model_keyboard(model_key, page),
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
    if not is_allowed(update.effective_user):
        await update.message.reply_text("Access denied.")
        return
    query = " ".join(context.args).strip() if context.args else ""
    if not query:
        context.user_data[USER_CTX_MODEL_SEARCH_WAITING] = True
        await update.message.reply_text(
            "Send text to search models by provider/name.\n"
            "Examples: `google flash`, `qwen thinking`, `minimax`."
        )
        return
    context.user_data[USER_CTX_MODEL_SEARCH_WAITING] = False
    context.user_data[USER_CTX_MODEL_SEARCH_QUERY] = query
    user_id = update.effective_user.id if update.effective_user else None
    model_key = get_user_model_key(user_id)
    text, keyboard = build_model_search_view(model_key, query, 0)
    await update.message.reply_text(text, reply_markup=keyboard)


async def on_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    user = query.from_user
    if not is_allowed(user):
        await query.answer("Access denied.", show_alert=True)
        return
    data = query.data or ""
    if not data.startswith("model:"):
        await query.answer()
        return
    parts = data.split(":")
    model_key = parts[1] if len(parts) > 1 else ""
    if model_key not in MODEL_ORDER:
        await query.answer("Unknown model.", show_alert=True)
        return
    set_user_model_key(user.id, model_key)
    page = find_model_page(model_key)
    await query.answer(f"Selected: {get_model_label(model_key)}")
    await query.edit_message_text(
        f"Current model: {get_model_label(model_key)}\n"
        f"Models: {len(MODEL_ORDER)} across {len(PROVIDER_ORDER)} providers.\n"
        f"Select model ({page + 1}/{get_model_page_count()}):",
        reply_markup=build_model_keyboard(model_key, page),
    )


async def on_model_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    user = query.from_user
    if not is_allowed(user):
        await query.answer("Access denied.", show_alert=True)
        return
    data = query.data or ""
    if not data.startswith("modelpage:"):
        await query.answer()
        return
    page_str = data.split(":", 1)[1]
    try:
        page = clamp_model_page(int(page_str), MODEL_ORDER)
    except ValueError:
        page = 0
    model_key = get_user_model_key(user.id)
    await query.answer()
    await query.edit_message_text(
        f"Current model: {get_model_label(model_key)}\n"
        f"Models: {len(MODEL_ORDER)} across {len(PROVIDER_ORDER)} providers.\n"
        f"Select model ({page + 1}/{get_model_page_count()}):",
        reply_markup=build_model_keyboard(model_key, page),
    )


async def on_model_providers_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    user = query.from_user
    if not is_allowed(user):
        await query.answer("Access denied.", show_alert=True)
        return
    data = query.data or ""
    if not data.startswith("modelproviders:"):
        await query.answer()
        return
    page_str = data.split(":", 1)[1]
    try:
        page = int(page_str)
    except ValueError:
        page = 0
    provider_page_count = max(1, math.ceil(len(PROVIDER_ORDER) / get_model_menu_page_size()))
    page = max(0, min(page, provider_page_count - 1))
    model_key = get_user_model_key(user.id)
    await query.answer()
    await query.edit_message_text(
        f"Current model: {get_model_label(model_key)}\n"
        f"Providers: {len(PROVIDER_ORDER)}\n"
        f"Select provider ({page + 1}/{provider_page_count}):",
        reply_markup=build_provider_keyboard(page),
    )


async def on_model_provider_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    user = query.from_user
    if not is_allowed(user):
        await query.answer("Access denied.", show_alert=True)
        return
    data = query.data or ""
    if not data.startswith("modelprovider:"):
        await query.answer()
        return
    parts = data.split(":")
    if len(parts) < 3:
        await query.answer("Invalid provider request.", show_alert=True)
        return
    try:
        provider_index = int(parts[1])
        page = int(parts[2])
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
    query = update.callback_query
    if not query:
        return
    user = query.from_user
    if not is_allowed(user):
        await query.answer("Access denied.", show_alert=True)
        return
    context.user_data[USER_CTX_MODEL_SEARCH_WAITING] = True
    await query.answer("Send search text in chat.")
    await query.edit_message_text(
        "Model search is ready.\n"
        "Send text to search by provider/name.\n"
        "Examples: `google flash`, `qwen thinking`, `deepseek`."
    )


async def on_model_search_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    user = query.from_user
    if not is_allowed(user):
        await query.answer("Access denied.", show_alert=True)
        return
    data = query.data or ""
    if not data.startswith("modelsearchpage:"):
        await query.answer()
        return
    page_str = data.split(":", 1)[1]
    try:
        page = int(page_str)
    except ValueError:
        page = 0
    search_query = str(context.user_data.get(USER_CTX_MODEL_SEARCH_QUERY, "")).strip()
    if not search_query:
        await query.answer("No active search. Use /modelsearch.", show_alert=True)
        return
    model_key = get_user_model_key(user.id)
    text, keyboard = build_model_search_view(model_key, search_query, page)
    await query.answer()
    await query.edit_message_text(text, reply_markup=keyboard)


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        await update.message.reply_text("Could not identify user.")
        return
    await update.message.reply_text(f"Your user_id: {user.id}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_owner(user):
        await update.message.reply_text("Owner only command.")
        return
    lines = [
        "Owner commands:",
        "/start - start bot",
        "/whoami - show your user_id",
        "/model - choose text model",
        "/modelsearch <text> - search text models",
        "/memory on|off|status - toggle memory for this chat",
        "/clear - clear chat memory",
        "/allow <user_id> - grant access",
        "/deny <user_id> - revoke access",
        "/list - list allowed users",
        "/keys - show key status",
        "/help - show this help",
    ]
    await update.message.reply_text("\n".join(lines))


async def keys_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_owner(user):
        await update.message.reply_text("Owner only command.")
        return
    custom_provider_ids = [pid for pid in TEXT_PROVIDER_HANDLERS.keys() if pid not in {"google", "nvidia"}]
    now_ts = int(time.time())
    blocked_capabilities = 0
    for entry in MODEL_CAPABILITIES.values():
        if not isinstance(entry, dict):
            continue
        if int(entry.get("expires_at", 0) or 0) <= now_ts:
            continue
        if str(entry.get("status", "")).strip().lower() in MODEL_CAPABILITY_BLOCKING_STATUSES:
            blocked_capabilities += 1
    lines = [
        f"USERS_DB_FILE: {users_db_path}",
        f"ALLOWLIST_FILE: {allowlist_path}",
        f"CUSTOM_PROVIDER_CONFIG_FILE: {CUSTOM_PROVIDER_CONFIG_FILE}",
        f"MODEL_CAPABILITIES_FILE: {model_capabilities_path}",
        f"MEMORY_CONTEXT_MESSAGES: {MEMORY_CONTEXT_MESSAGES}",
        f"Regular model timeout: {REGULAR_MODEL_TIMEOUT_SECONDS}s",
        f"Reasoning model timeout: {REASONING_MODEL_TIMEOUT_SECONDS}s",
        f"Fallback attempt timeout: {FALLBACK_ATTEMPT_TIMEOUT_SECONDS}s",
        f"Model probe enabled: {MODEL_CAPABILITY_PROBE_ENABLED}",
        f"Model probe scope: {MODEL_PROBE_SCOPE}",
        f"Model probe timeout/workers: {MODEL_PROBE_TIMEOUT_SECONDS}s/{MODEL_PROBE_WORKERS}",
        f"Hide unavailable models: {MODEL_HIDE_UNAVAILABLE_MODELS}",
        f"Model capability cache: {len(MODEL_CAPABILITIES)} entries ({blocked_capabilities} blocking)",
        f"Models active/all: {len(MODEL_ORDER)}/{len(ALL_MODEL_ORDER)}",
        f"Gemini models loaded: {len(GEMINI_MODEL_NAMES)}",
        f"NVIDIA models loaded: {len(NVIDIA_MODEL_NAMES)}",
        f"Text providers loaded: {len(TEXT_PROVIDER_HANDLERS)}",
        f"Custom providers loaded: {len(custom_provider_ids)}",
        f"General provider priority count: {len(MODEL_PROVIDER_PRIORITY.get('*', []))}",
    ]
    for provider_id in sorted(TEXT_PROVIDER_HANDLERS.keys()):
        handler = TEXT_PROVIDER_HANDLERS[provider_id]
        model_count = len(PROVIDER_MODEL_KEYS.get(provider_id, []))
        lines.append(f"{handler.label}: keys={handler.key_count()}, models={model_count}")
    await update.message.reply_text("\n".join(lines))


async def list_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_owner(user):
        await update.message.reply_text("Owner only command.")
        return
    rows = list_allowed_users_from_db()
    if not rows:
        await update.message.reply_text("Allow list is empty.")
        return
    lines = ["Allowed users:"]
    for uid, added_at in rows:
        lines.append(f"- {uid} (added: {added_at})")
    await update.message.reply_text("\n".join(lines))


async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user):
        await update.message.reply_text("Access denied.")
        return
    if not update.effective_chat:
        await update.message.reply_text("Chat not found.")
        return
    clear_chat_history(update.effective_chat.id)
    await update.message.reply_text("Memory cleared for this chat.")


async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user):
        await update.message.reply_text("Access denied.")
        return
    if not update.effective_chat:
        await update.message.reply_text("Chat not found.")
        return
    chat_id = update.effective_chat.id
    if not context.args:
        state = "on" if is_memory_enabled(chat_id) else "off"
        await update.message.reply_text(f"Memory is {state}. Usage: /memory on|off|status")
        return
    arg = context.args[0].lower().strip()
    if arg == "status":
        state = "on" if is_memory_enabled(chat_id) else "off"
        await update.message.reply_text(f"Memory is {state}.")
        return
    if arg == "on":
        set_memory_enabled(chat_id, True)
        await update.message.reply_text("Memory enabled for this chat.")
        return
    if arg == "off":
        set_memory_enabled(chat_id, False)
        await update.message.reply_text("Memory disabled for this chat.")
        return
    await update.message.reply_text("Usage: /memory on|off|status")


async def allow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_owner(user):
        await update.message.reply_text("Owner only command.")
        return
    if update.effective_chat and update.effective_chat.type != "private":
        await update.message.reply_text("Use this command in private chat with bot.")
        return

    target_user = None
    if update.message and update.message.reply_to_message:
        target_user = update.message.reply_to_message.from_user

    if target_user:
        uid = target_user.id
    else:
        if not context.args:
            await update.message.reply_text("Usage: /allow <user_id> or reply to user message.")
            return
        if context.args[0].startswith("@"):
            await update.message.reply_text("Cannot add by @username. Use user_id.")
            return
        uid = parse_user_id(context.args[0])
        if uid is None:
            await update.message.reply_text("Invalid user_id.")
            return
    allowed_user_ids.add(uid)
    add_allowed_user_to_db(uid)
    write_allowlist_backup(allowed_user_ids)
    await update.message.reply_text(f"Added user_id {uid}.")


async def deny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_owner(user):
        await update.message.reply_text("Owner only command.")
        return
    if update.effective_chat and update.effective_chat.type != "private":
        await update.message.reply_text("Use this command in private chat with bot.")
        return

    target_user = None
    if update.message and update.message.reply_to_message:
        target_user = update.message.reply_to_message.from_user

    if target_user:
        uid = target_user.id
    else:
        if not context.args:
            await update.message.reply_text("Usage: /deny <user_id> or reply to user message.")
            return
        if context.args[0].startswith("@"):
            await update.message.reply_text("Cannot remove by @username. Use user_id.")
            return
        uid = parse_user_id(context.args[0])
        if uid is None:
            await update.message.reply_text("Invalid user_id.")
            return
    if uid == OWNER_USER_ID:
        await update.message.reply_text("Cannot remove owner.")
        return
    allowed_user_ids.discard(uid)
    remove_allowed_user_from_db(uid)
    write_allowlist_backup(allowed_user_ids)
    await update.message.reply_text(f"Removed user_id {uid}.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user):
        await update.message.reply_text("Access denied.")
        return
    prompt = update.message.text.strip() if update.message and update.message.text else ""
    if not prompt:
        await update.message.reply_text("Empty message.")
        return
    if context.user_data.get(USER_CTX_MODEL_SEARCH_WAITING):
        context.user_data[USER_CTX_MODEL_SEARCH_WAITING] = False
        context.user_data[USER_CTX_MODEL_SEARCH_QUERY] = prompt
        user_id = update.effective_user.id if update.effective_user else None
        model_key = get_user_model_key(user_id)
        text, keyboard = build_model_search_view(model_key, prompt, 0)
        await update.message.reply_text(text, reply_markup=keyboard)
        return
    await update.message.chat.send_action(action="typing")
    try:
        user_id = update.effective_user.id if update.effective_user else None
        chat_id = update.effective_chat.id if update.effective_chat else user_id
        if chat_id is None or user_id is None:
            await update.message.reply_text("Chat/user not found.")
            return
        memory_on = is_memory_enabled(chat_id)
        history = get_recent_history(chat_id, MEMORY_CONTEXT_MESSAGES) if memory_on else []
        model_key = get_user_model_key(user_id)
        answer, used_model_name, used_provider_id, usage = await asyncio.to_thread(
            generate_text, prompt, model_key, history
        )
        logger.info(
            "Model usage: user_id=%s chat_id=%s provider=%s model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            user_id,
            chat_id,
            used_provider_id,
            used_model_name,
            usage.get("prompt_tokens", "n/a"),
            usage.get("completion_tokens", "n/a"),
            usage.get("total_tokens", "n/a"),
        )
        if memory_on:
            add_history_message(chat_id, user_id, "user", prompt)
            add_history_message(chat_id, user_id, "assistant", answer)
        await send_reply_text(update.message, answer)
    except Exception as exc:
        logger.exception("Text generation failed")
        await update.message.reply_text(f"Generation error: {exc}")


def build_app():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(8).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("keys", keys_status))
    app.add_handler(CommandHandler("list", list_allowed))
    app.add_handler(CommandHandler("clear", clear_memory))
    app.add_handler(CommandHandler("memory", memory_command))
    app.add_handler(CommandHandler("allow", allow))
    app.add_handler(CommandHandler("deny", deny))
    app.add_handler(CommandHandler("model", model_menu))
    app.add_handler(CommandHandler("modelsearch", model_search))
    app.add_handler(CallbackQueryHandler(on_model_providers_callback, pattern=r"^modelproviders:"))
    app.add_handler(CallbackQueryHandler(on_model_provider_callback, pattern=r"^modelprovider:"))
    app.add_handler(CallbackQueryHandler(on_model_search_prompt_callback, pattern=r"^modelsearchprompt$"))
    app.add_handler(CallbackQueryHandler(on_model_search_page_callback, pattern=r"^modelsearchpage:"))
    app.add_handler(CallbackQueryHandler(on_model_page_callback, pattern=r"^modelpage:"))
    app.add_handler(CallbackQueryHandler(on_model_callback, pattern=r"^model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return app


def main() -> None:
    # Keep startup responsive by default; enable blocking probe only when explicitly requested.
    if STARTUP_PROBE_MODE in {"smart", "full"}:
        force_full = STARTUP_PROBE_MODE == "full"
        logger.info(
            "Startup model probe enabled: mode=%s, scope=%s, timeout=%ss",
            STARTUP_PROBE_MODE,
            MODEL_PROBE_SCOPE,
            MODEL_PROBE_TIMEOUT_SECONDS,
        )
        run_startup_model_capability_probe(force_full=force_full)
        apply_model_capability_filter(force=True, strict=False)
    else:
        logger.info("Startup model probe disabled (STARTUP_PROBE_MODE=none).")
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
