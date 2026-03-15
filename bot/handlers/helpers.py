"""Telegram-specific utilities: sending messages, extracting attachments, model menus."""
from __future__ import annotations

import base64
import io
import logging
import math
import mimetypes
from pathlib import Path
from typing import Any

from telegram import (
    InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile, Message,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest
from bot.config import Settings
from bot.formatting import markdown_to_html, split_message, trim_text
from bot.model_utils import (
    DEFAULT_MODEL_KEY,
    DEFAULT_MODEL_LABEL,
    is_default_model_key,
    is_image_model_name,
)
from bot.schemas import InputAttachment
from bot.services.registry import ModelRegistry

logger = logging.getLogger("bot")

MAX_REPLY_DEPTH = 4


# ── user label helpers ─────────────────────────────────────────────

def format_user_label(user: Any) -> str:
    if not user:
        return "unknown"
    first = str(getattr(user, "first_name", "") or "").strip()
    last = str(getattr(user, "last_name", "") or "").strip()
    full = " ".join(part for part in (first, last) if part)
    if full:
        return full
    username = str(getattr(user, "username", "") or "").strip()
    if username:
        return f"@{username}"
    uid = getattr(user, "id", None)
    return f"user_id:{uid}" if uid is not None else "unknown"


# ── sending replies ────────────────────────────────────────────────

async def _send_chunk(msg: Message, text: str, *, html_mode: bool = True, depth: int = 0) -> None:
    if depth >= MAX_REPLY_DEPTH:
        await msg.reply_text(text[:4000])
        return

    payload = markdown_to_html(text) if html_mode else text
    try:
        if html_mode:
            await msg.reply_text(payload, parse_mode=ParseMode.HTML)
        else:
            await msg.reply_text(payload)
    except BadRequest as exc:
        reason = str(exc).lower()
        if "message is too long" in reason:
            for part in split_message(text, max(256, min(len(text) // 2, 2000))):
                await _send_chunk(msg, part, html_mode=html_mode, depth=depth + 1)
            return
        if html_mode:
            await _send_chunk(msg, text, html_mode=False, depth=depth + 1)
            return
        raise


async def send_text(msg: Message, text: str, chunk_limit: int = 4000) -> None:
    full = str(text or "").strip()
    if not full:
        await msg.reply_text("Empty response.")
        return
    for chunk in split_message(full, chunk_limit):
        await _send_chunk(msg, chunk)


async def send_image(msg: Message, data: dict[str, str], caption: str = "") -> None:
    url = str(data.get("url", "")).strip()
    if url:
        await msg.reply_photo(photo=url, caption=caption or None)
        return
    b64 = str(data.get("b64_json", "")).strip()
    if b64:
        decoded = base64.b64decode(b64, validate=False)
        if not decoded:
            raise RuntimeError("Empty image after base64 decoding.")
        buf = io.BytesIO(decoded)
        buf.name = "generated.png"
        await msg.reply_photo(photo=InputFile(buf, filename=buf.name), caption=caption or None)
        return
    raise RuntimeError("No image data in response.")


async def edit_inline(
    bot: Any,
    inline_id: str,
    text: str,
    *,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    trimmed = trim_text(text) or "Empty response."
    payload = markdown_to_html(trimmed)
    try:
        await bot.edit_message_text(
            payload,
            inline_message_id=inline_id,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
    except BadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return
        if "message is too long" in str(exc).lower():
            shorter = markdown_to_html(trim_text(trimmed, 2000))
            try:
                await bot.edit_message_text(
                    shorter,
                    inline_message_id=inline_id,
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
                return
            except Exception:
                pass
        try:
            await bot.edit_message_text(
                trimmed,
                inline_message_id=inline_id,
                reply_markup=reply_markup,
            )
        except BadRequest as e2:
            if "message is not modified" not in str(e2).lower():
                logger.warning("Inline edit failed: %s", e2)
    except Exception as exc:
        logger.warning("Inline edit failed: %s", exc)


def inline_placeholder_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("…", callback_data="inlineplaceholder")]])


# ── attachment extraction ──────────────────────────────────────────

def _guess_mime(name: str) -> str:
    g, _ = mimetypes.guess_type(name)
    return str(g or "").lower()


def _is_text(mime: str, name: str) -> bool:
    m = str(mime or "").lower()
    if m.startswith("text/"):
        return True
    if m in ("application/json", "application/xml", "application/yaml",
             "application/x-yaml", "application/csv"):
        return True
    ext = Path(name or "").suffix.lower()
    return ext in {
        ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".tsx", ".jsx",
        ".json", ".yaml", ".yml", ".xml", ".csv", ".sql", ".log", ".ini", ".toml",
    }


def _decode_text(raw: bytes, max_chars: int) -> str:
    if not raw or b"\x00" in raw[:4096]:
        return ""
    for enc in ("utf-8", "utf-16", "cp1251"):
        try:
            return raw.decode(enc).strip()[:max_chars]
        except UnicodeDecodeError:
            continue
    return ""


async def extract_attachments(msg: Message, s: Settings) -> tuple[list[InputAttachment], list[str]]:
    result: list[InputAttachment] = []
    notices: list[str] = []
    max_count = s.max_input_attachment_count
    max_bytes = s.max_input_attachment_bytes

    async def dl(ref: Any) -> bytes:
        remote = await ref.get_file()
        raw = bytes(await remote.download_as_bytearray() or b"")
        if len(raw) > max_bytes:
            raise RuntimeError(f"exceeds {max_bytes} bytes")
        return raw

    # photo
    if msg.photo and len(result) < max_count:
        photo = msg.photo[-1]
        size = int(getattr(photo, "file_size", 0) or 0)
        if size > max_bytes:
            notices.append(f"Photo skipped: {size} bytes exceeds limit.")
        else:
            try:
                raw = await dl(photo)
                if raw:
                    result.append(InputAttachment(
                        kind="image", mime_type="image/jpeg",
                        file_name="photo.jpg", bytes=raw,
                    ))
            except Exception as exc:
                notices.append(f"Photo download failed: {exc}")

    # document
    doc = getattr(msg, "document", None)
    if doc is not None and len(result) >= max_count:
        notices.append(f"Only first {max_count} attachments processed.")
    elif doc is not None:
        fname = str(getattr(doc, "file_name", "") or "document").strip() or "document"
        mime = str(getattr(doc, "mime_type", "") or "").lower() or _guess_mime(fname)
        size = int(getattr(doc, "file_size", 0) or 0)
        if size > max_bytes:
            notices.append(f"'{fname}' skipped: {size} bytes exceeds limit.")
        else:
            try:
                raw = await dl(doc)
                if not raw:
                    notices.append(f"'{fname}' is empty.")
                else:
                    kind = "image" if mime.startswith("image/") else "file"
                    att = InputAttachment(
                        kind=kind,
                        mime_type=mime or "application/octet-stream",
                        file_name=fname, bytes=raw,
                    )
                    if (kind == "file"
                            and len(raw) <= s.max_input_text_attachment_bytes
                            and _is_text(mime, fname)):
                        text = _decode_text(raw, s.max_input_text_attachment_chars)
                        if text:
                            att.text = text
                    result.append(att)
            except Exception as exc:
                notices.append(f"'{fname}' download failed: {exc}")

    return result, notices


# ── model menu helpers ──────────────────────────────────────────────

def model_page(
    reg: ModelRegistry, keys: list[str], page: int, page_size: int = 12,
) -> tuple[int, int, list[str]]:
    if not keys:
        return 0, 1, []
    pages = max(1, math.ceil(len(keys) / page_size))
    page = max(0, min(page, pages - 1))
    start = page * page_size
    return page, pages, keys[start:start + page_size]


def search_models(reg: ModelRegistry, query: str) -> list[str]:
    keys = selectable_model_keys(reg)
    raw = query.strip().lower()
    if not raw:
        return list(keys)
    terms = raw.split()
    ranked: list[tuple[int, int, str]] = []
    for idx, mk in enumerate(keys):
        label = reg.get_label(mk).lower()
        full = reg.get_full_name(mk).lower()
        prov = reg.get_provider_id(mk).lower()
        score = 0
        if raw in full:
            score += 120
        if raw in label:
            score += 90
        if raw in prov:
            score += 110
        for t in terms:
            if t in prov:
                score += 30
            if t in full:
                score += 25
            if t in label:
                score += 15
        if score > 0:
            ranked.append((score, idx, mk))
    ranked.sort(key=lambda r: (-r[0], r[1]))
    return [r[2] for r in ranked]


def model_select_rows(
    reg: ModelRegistry, selected: str, keys: list[str],
) -> list[list[InlineKeyboardButton]]:
    rows: list[list[InlineKeyboardButton]] = []
    title = DEFAULT_MODEL_LABEL
    if is_default_model_key(selected):
        title = f"* {title}"
    rows.append([InlineKeyboardButton(title, callback_data=f"model:{DEFAULT_MODEL_KEY}")])
    for mk in keys:
        title = reg.get_label(mk)
        if mk == selected:
            title = f"* {title}"
        rows.append([InlineKeyboardButton(title, callback_data=f"model:{mk}")])
    return rows


def model_keyboard(reg: ModelRegistry, selected: str, page: int) -> InlineKeyboardMarkup:
    keys = selectable_model_keys(reg)
    pg, pages, keys = model_page(reg, keys, page)
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton("Providers", callback_data="modelproviders:0"),
         InlineKeyboardButton("Search", callback_data="modelsearchprompt")],
    ]
    rows.extend(model_select_rows(reg, selected, keys))
    nav: list[InlineKeyboardButton] = []
    if pg > 0:
        nav.append(InlineKeyboardButton("Prev", callback_data=f"modelpage:{pg - 1}"))
    if pg < pages - 1:
        nav.append(InlineKeyboardButton("Next", callback_data=f"modelpage:{pg + 1}"))
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(rows)


def find_model_page(reg: ModelRegistry, key: str, page_size: int = 12) -> int:
    keys = selectable_model_keys(reg)
    if key not in keys:
        return 0
    return keys.index(key) // page_size


def is_selectable_model(reg: ModelRegistry, key: str) -> bool:
    if is_default_model_key(key):
        return False
    if key not in reg.order_set:
        return False
    pid = reg.get_provider_id(key)
    handler = reg.handlers.get(pid)
    if not handler or not handler.supports_text():
        return False
    name = reg.model_name.get(key, "")
    if is_image_model_name(name):
        return False
    return True


def selectable_model_keys(reg: ModelRegistry) -> list[str]:
    return [k for k in reg.order if is_selectable_model(reg, k)]


def selectable_provider_ids(reg: ModelRegistry) -> list[str]:
    out: list[str] = []
    for pid in reg.provider_order:
        keys = reg.provider_model_keys.get(pid, [])
        if any(is_selectable_model(reg, k) for k in keys):
            out.append(pid)
    return out


def selectable_provider_model_keys(reg: ModelRegistry, pid: str) -> list[str]:
    return [k for k in reg.provider_model_keys.get(pid, []) if is_selectable_model(reg, k)]


def selectable_or_default_model_key(reg: ModelRegistry, key: str) -> str:
    if is_default_model_key(key):
        return DEFAULT_MODEL_KEY
    if is_selectable_model(reg, key):
        return key
    keys = selectable_model_keys(reg)
    return keys[0] if keys else reg.default_key


def display_model_label(reg: ModelRegistry, key: str) -> str:
    return DEFAULT_MODEL_LABEL if is_default_model_key(key) else reg.get_label(key)
