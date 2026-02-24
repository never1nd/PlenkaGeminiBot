import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import google.generativeai as genai
import requests
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_OWNER_USER_ID = 8082486311
DEFAULT_OWNER_USERNAME = "s0ryanix"

OWNER_USER_ID = int(os.getenv("OWNER_USER_ID", str(DEFAULT_OWNER_USER_ID)) or str(DEFAULT_OWNER_USER_ID))
OWNER_USERNAME = os.getenv("OWNER_USERNAME", DEFAULT_OWNER_USERNAME).lstrip("@").strip()

ALLOWED_USER_IDS = os.getenv("ALLOWED_USER_IDS", "").strip()
ALLOWLIST_FILE = os.getenv("ALLOWLIST_FILE", "allowlist.json").strip()
USERS_DB_FILE = os.getenv("USERS_DB_FILE", "users.db").strip()
MODEL_PREFS_FILE = os.getenv("MODEL_PREFS_FILE", "model_prefs.json").strip()
MEMORY_CONTEXT_MESSAGES = int(os.getenv("MEMORY_CONTEXT_MESSAGES", "20") or "20")

TEXT_MODEL_GEMINI_25 = os.getenv("TEXT_MODEL_GEMINI_25", "gemini-2.5-flash").strip()
TEXT_MODEL_GEMINI_3 = os.getenv("TEXT_MODEL_GEMINI_3", "gemini-3-flash-preview").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.0-flash-preview-image-generation").strip()

NVIDIA_API_BASE = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1").strip().rstrip("/")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_API_KEY_QWEN = os.getenv("NVIDIA_API_KEY_QWEN", os.getenv("QWEN_API_KEY", "")).strip()
NVIDIA_API_KEY_KIMI = os.getenv("NVIDIA_API_KEY_KIMI", os.getenv("KIMI_API_KEY", "")).strip()
NVIDIA_API_KEY_MINIMAX = os.getenv("NVIDIA_API_KEY_MINIMAX", os.getenv("MINIMAX_API_KEY", "")).strip()

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7") or "0.7")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024") or "1024")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

MODEL_GEMINI = "gemini"
MODEL_GEMINI_3 = "gemini3"
MODEL_QWEN = "qwen"
MODEL_KIMI = "kimi"
MODEL_MINIMAX = "minimax"

NVIDIA_MODEL_NAMES = {
    MODEL_QWEN: "qwen/qwen3-next-80b-a3b-instruct",
    MODEL_KIMI: "moonshotai/kimi-k2-thinking",
    MODEL_MINIMAX: "minimaxai/minimax-m2.1",
}

MODEL_LABELS = {
    MODEL_GEMINI: "Gemini 2.5 Flash",
    MODEL_GEMINI_3: "Gemini 3 Flash",
    MODEL_QWEN: "Qwen Next 80B",
    MODEL_KIMI: "Kimi K2 Thinking",
    MODEL_MINIMAX: "MiniMax M2.1",
}

DEFAULT_TEXT_PROVIDER = MODEL_GEMINI

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("bot")

if not TELEGRAM_BOT_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN is required")
if not GEMINI_API_KEY:
    raise SystemExit("GEMINI_API_KEY is required")
if not IMAGE_MODEL:
    raise SystemExit("IMAGE_MODEL is empty. Provide an image-capable model name.")

allowlist_path = Path(ALLOWLIST_FILE)
users_db_path = Path(USERS_DB_FILE)
model_prefs_path = Path(MODEL_PREFS_FILE)

genai.configure(api_key=GEMINI_API_KEY)


def _parse_ids(raw_csv: str) -> list[int]:
    result: list[int] = []
    if not raw_csv:
        return result
    for raw in raw_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            result.append(int(raw))
        except ValueError:
            logger.warning("Skipping invalid user id in ALLOWED_USER_IDS: %s", raw)
    return result


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
    ids_to_add = _parse_ids(ALLOWED_USER_IDS)
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
                if model_key in MODEL_LABELS:
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
user_model_prefs = load_model_prefs()


def is_allowed(user) -> bool:
    if not user:
        return False
    if user.id in allowed_user_ids:
        return True
    if OWNER_USERNAME and user.username and user.username.lower() == OWNER_USERNAME.lower():
        return True
    return False


def is_owner(user) -> bool:
    if not user:
        return False
    if OWNER_USER_ID and user.id == OWNER_USER_ID:
        return True
    if OWNER_USERNAME and user.username and user.username.lower() == OWNER_USERNAME.lower():
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


def extract_text(response) -> str:
    if not response or not getattr(response, "candidates", None):
        return ""
    content = response.candidates[0].content
    if not content:
        return ""
    parts = getattr(content, "parts", []) or []
    texts = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def extract_image_bytes(response) -> Optional[Tuple[bytes, str]]:
    if not response or not getattr(response, "candidates", None):
        return None
    content = response.candidates[0].content
    if not content:
        return None
    parts = getattr(content, "parts", []) or []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            mime = getattr(inline, "mime_type", "image/png") or "image/png"
            data = base64.b64decode(inline.data)
            return data, mime
    return None


def get_user_model_key(user_id: Optional[int]) -> str:
    if user_id is None:
        return DEFAULT_TEXT_PROVIDER
    return user_model_prefs.get(str(user_id), DEFAULT_TEXT_PROVIDER)


def set_user_model_key(user_id: int, model_key: str) -> None:
    if model_key not in MODEL_LABELS:
        raise ValueError("Unsupported model key")
    user_model_prefs[str(user_id)] = model_key
    save_model_prefs(user_model_prefs)


def get_nvidia_key(model_key: str) -> str:
    if model_key == MODEL_QWEN:
        return NVIDIA_API_KEY_QWEN or NVIDIA_API_KEY
    if model_key == MODEL_KIMI:
        return NVIDIA_API_KEY_KIMI or NVIDIA_API_KEY
    if model_key == MODEL_MINIMAX:
        return NVIDIA_API_KEY_MINIMAX or NVIDIA_API_KEY
    return ""


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


def generate_text_gemini(prompt: str, model_name: str, history: list[dict[str, str]]) -> Tuple[str, str]:
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

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        full_prompt,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )
    text = extract_text(response)
    if not text:
        raise RuntimeError(f"{model_name} returned an empty response.")
    return strip_reasoning(text), model_name


def generate_text_nvidia(prompt: str, model_key: str, history: list[dict[str, str]]) -> Tuple[str, str]:
    api_key = get_nvidia_key(model_key)
    if not api_key:
        raise RuntimeError(f"NVIDIA API key is not configured for {model_key}.")
    model_name = NVIDIA_MODEL_NAMES[model_key]

    messages = []
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
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    if model_key == MODEL_QWEN:
        payload["top_p"] = 0.7
    elif model_key == MODEL_KIMI:
        payload["top_p"] = 0.9
    elif model_key == MODEL_MINIMAX:
        payload["top_p"] = 0.7

    resp = requests.post(
        f"{NVIDIA_API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=90,
    )
    if not resp.ok:
        raise RuntimeError(f"NVIDIA API error: {resp.status_code} {resp.text[:300]}")
    data = resp.json()
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    text = strip_reasoning(text)
    if not text:
        raise RuntimeError(f"{model_name} returned an empty response.")
    return text, model_name


def generate_text(prompt: str, model_key: str, history: list[dict[str, str]]) -> Tuple[str, str]:
    if model_key == MODEL_GEMINI:
        return generate_text_gemini(prompt, TEXT_MODEL_GEMINI_25, history)
    if model_key == MODEL_GEMINI_3:
        return generate_text_gemini(prompt, TEXT_MODEL_GEMINI_3, history)
    return generate_text_nvidia(prompt, model_key, history)


def generate_image(prompt: str) -> Tuple[bytes, str]:
    model = genai.GenerativeModel(IMAGE_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )
    image = extract_image_bytes(response)
    if image:
        return image
    text = extract_text(response)
    raise RuntimeError(f"Image model returned no image. Text: {text}")


def build_model_keyboard(selected_key: str) -> InlineKeyboardMarkup:
    def label(model_key: str, title: str) -> str:
        return f"* {title}" if selected_key == model_key else title

    rows = [
        [InlineKeyboardButton(label(MODEL_GEMINI, "Gemini 2.5 Flash"), callback_data="model:gemini")],
        [InlineKeyboardButton(label(MODEL_GEMINI_3, "Gemini 3 Flash"), callback_data="model:gemini3")],
        [InlineKeyboardButton(label(MODEL_QWEN, "Qwen Next 80B"), callback_data="model:qwen")],
        [InlineKeyboardButton(label(MODEL_KIMI, "Kimi K2 Thinking"), callback_data="model:kimi")],
        [InlineKeyboardButton(label(MODEL_MINIMAX, "MiniMax M2.1"), callback_data="model:minimax")],
    ]
    return InlineKeyboardMarkup(rows)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user):
        await update.message.reply_text("Access denied. Ask owner to add you.")
        return
    user_id = update.effective_user.id if update.effective_user else None
    model_key = get_user_model_key(user_id)
    await update.message.reply_text(
        f"Ready. Current model: {MODEL_LABELS[model_key]}\n"
        "Send text, use /img <prompt>, or /model to switch text model."
    )


async def model_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user):
        await update.message.reply_text("Access denied.")
        return
    user_id = update.effective_user.id if update.effective_user else None
    model_key = get_user_model_key(user_id)
    await update.message.reply_text(
        f"Current model: {MODEL_LABELS[model_key]}\nSelect model:",
        reply_markup=build_model_keyboard(model_key),
    )


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
    model_key = data.split(":", 1)[1]
    if model_key not in MODEL_LABELS:
        await query.answer("Unknown model.", show_alert=True)
        return
    set_user_model_key(user.id, model_key)
    await query.answer(f"Selected: {MODEL_LABELS[model_key]}")
    await query.edit_message_text(
        f"Current model: {MODEL_LABELS[model_key]}\nSelect model:",
        reply_markup=build_model_keyboard(model_key),
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        await update.message.reply_text("Could not identify user.")
        return
    await update.message.reply_text(f"Your user_id: {user.id}")


async def keys_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_owner(user):
        await update.message.reply_text("Owner only command.")
        return
    lines = [
        f"USERS_DB_FILE: {users_db_path}",
        f"MEMORY_CONTEXT_MESSAGES: {MEMORY_CONTEXT_MESSAGES}",
        f"NVIDIA_API_BASE: {NVIDIA_API_BASE or 'missing'}",
        f"NVIDIA_API_KEY: {mask_key(NVIDIA_API_KEY)}",
        f"NVIDIA_API_KEY_QWEN: {mask_key(NVIDIA_API_KEY_QWEN)}",
        f"NVIDIA_API_KEY_KIMI: {mask_key(NVIDIA_API_KEY_KIMI)}",
        f"NVIDIA_API_KEY_MINIMAX: {mask_key(NVIDIA_API_KEY_MINIMAX)}",
    ]
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
    await update.message.reply_text(f"Removed user_id {uid}.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user):
        await update.message.reply_text("Access denied.")
        return
    prompt = update.message.text.strip() if update.message and update.message.text else ""
    if not prompt:
        await update.message.reply_text("Empty message.")
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
        answer, _ = await asyncio.to_thread(generate_text, prompt, model_key, history)
        if memory_on:
            add_history_message(chat_id, user_id, "user", prompt)
            add_history_message(chat_id, user_id, "assistant", answer)
        await update.message.reply_text(answer)
    except Exception as exc:
        logger.exception("Text generation failed")
        await update.message.reply_text(f"Generation error: {exc}")


async def handle_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user):
        await update.message.reply_text("Access denied.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /img <prompt>")
        return
    prompt = " ".join(context.args).strip()
    if not prompt:
        await update.message.reply_text("Empty prompt.")
        return
    await update.message.chat.send_action(action="upload_photo")
    try:
        image_bytes, mime = await asyncio.to_thread(generate_image, prompt)
        filename = "image.png" if mime == "image/png" else "image.jpg"
        await update.message.reply_photo(photo=image_bytes, filename=filename, caption="Done.")
    except Exception as exc:
        logger.exception("Image generation failed")
        await update.message.reply_text(f"Generation error: {exc}")


def build_app():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("keys", keys_status))
    app.add_handler(CommandHandler("list", list_allowed))
    app.add_handler(CommandHandler("clear", clear_memory))
    app.add_handler(CommandHandler("memory", memory_command))
    app.add_handler(CommandHandler("allow", allow))
    app.add_handler(CommandHandler("deny", deny))
    app.add_handler(CommandHandler("model", model_menu))
    app.add_handler(CommandHandler("img", handle_img))
    app.add_handler(CallbackQueryHandler(on_model_callback, pattern=r"^model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
