from __future__ import annotations

import logging
import math

from telegram import Update
from telegram.ext import ContextTypes

from bot.handlers.deps import Deps, get_deps
from bot.handlers.guards import ensure_allowed, ensure_owner, ensure_private, is_group
from bot.handlers.helpers import (
    model_keyboard, find_model_page,
    search_models, send_image, send_text, display_model_label,
    selectable_model_keys, selectable_provider_ids, selectable_or_default_model_key,
)
from bot.formatting import trim_text

logger = logging.getLogger("bot")


# ── /start ──────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_allowed(update, d, "Access denied. Ask owner to add you."):
        return
    user = update.effective_user
    raw_key = d.prefs.get(user.id, d.registry.default_key) if user else d.registry.default_key
    model_key = selectable_or_default_model_key(d.registry, raw_key)
    msg = update.effective_message
    if msg:
        await msg.reply_text(
            f"Ready. Current model: {display_model_label(d.registry, model_key)}\n"
            "Send text/photo/file, use /api to browse models, or /img <prompt> for images."
        )


# ── /whoami ─────────────────────────────────────────────────────────

async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    msg = update.effective_message
    if user and msg:
        await msg.reply_text(f"Your user_id: {user.id}")


# ── /help ───────────────────────────────────────────────────────────

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_owner(update, d):
        return
    text = "\n".join([
        "Owner commands:",
        "/start - start bot",
        "/whoami - show your user_id",
        "/api - choose text API/model",
        "/provider - browse providers",
        "/modelsearch <text> - search text models",
        "/img <prompt> - generate an image",
        "/memory on|off|status - toggle memory context",
        "/memoryshow - view stored memory (DM only)",
        "/clear - clear chat memory",
        "/allow <user_id> - grant access",
        "/deny <user_id> - revoke access",
        "/list - list allowed users",
        "/keys - show key status",
        "/help - show this help",
    ])
    msg = update.effective_message
    if msg:
        await msg.reply_text(text)


# ── /keys ───────────────────────────────────────────────────────────

async def keys_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_owner(update, d):
        return
    reg = d.registry
    s = d.settings
    lines = [
        f"Models active/all: {len(reg.order)}/{len(reg.all_order)}",
        f"Text providers: {len(reg.handlers)}",
        f"Capability cache: {d.capabilities.blocking_count()} blocking",
        f"Provider availability: {d.availability.cache_stats[1]}/{d.availability.cache_stats[0]} unavailable",
        f"Regular timeout: {s.regular_model_timeout}s",
        f"Reasoning timeout: {s.reasoning_model_timeout}s",
    ]
    for pid in sorted(reg.handlers):
        h = reg.handlers[pid]
        count = len(reg.provider_model_keys.get(pid, []))
        lines.append(f"{h.label}: keys={h.key_count()}, models={count}")
    msg = update.effective_message
    if msg:
        await msg.reply_text("\n".join(lines))


# ── /list ───────────────────────────────────────────────────────────

async def list_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_owner(update, d):
        return
    rows = d.allowlist.list_users()
    msg = update.effective_message
    if not msg:
        return
    if not rows:
        await msg.reply_text("Allow list is empty.")
        return
    lines = ["Allowed users:"]
    for uid, added_at in rows:
        lines.append(f"- {uid} (added: {added_at})")
    await msg.reply_text("\n".join(lines))


# ── /allow & /deny ──────────────────────────────────────────────────

def _resolve_uid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int | None:
    message = update.effective_message
    reply = getattr(message, "reply_to_message", None) if message else None
    target = getattr(reply, "from_user", None) if reply else None
    if target:
        return target.id
    raw = str(context.args[0]).strip() if context.args else ""
    if not raw or raw.startswith("@"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def allow_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_owner(update, d):
        return
    if not await ensure_private(update):
        return
    uid = _resolve_uid(update, context)
    msg = update.effective_message
    if not msg:
        return
    if uid is None:
        await msg.reply_text("Usage: /allow <user_id> or reply to user message.")
        return
    d.allowlist.add(uid)
    await msg.reply_text(f"Added user_id {uid}.")


async def deny_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_owner(update, d):
        return
    if not await ensure_private(update):
        return
    uid = _resolve_uid(update, context)
    msg = update.effective_message
    if not msg:
        return
    if uid is None:
        await msg.reply_text("Usage: /deny <user_id> or reply to user message.")
        return
    if d.settings.is_owner(uid):
        await msg.reply_text("Cannot remove owner.")
        return
    d.allowlist.remove(uid)
    await msg.reply_text(f"Removed user_id {uid}.")


# ── /clear ──────────────────────────────────────────────────────────

async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_allowed(update, d):
        return
    chat = update.effective_chat
    msg = update.effective_message
    if chat and msg:
        d.db.clear_history(chat.id)
        await msg.reply_text("Memory cleared for this chat.")


# ── /memory ─────────────────────────────────────────────────────────

async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_allowed(update, d):
        return
    chat, user, msg = update.effective_chat, update.effective_user, update.effective_message
    if not chat or not user or not msg:
        return
    arg = str(context.args[0]).strip().lower() if context.args else ""
    if arg == "on":
        d.db.set_memory_enabled(chat.id, user.id, True)
        await msg.reply_text("Memory enabled for you in this chat.")
    elif arg == "off":
        d.db.set_memory_enabled(chat.id, user.id, False)
        await msg.reply_text("Memory disabled for you in this chat.")
    else:
        state = "on" if d.db.is_memory_enabled(chat.id, user.id) else "off"
        await msg.reply_text(f"Memory is {state}. Usage: /memory on|off")


async def memory_show(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_allowed(update, d):
        return
    if not await ensure_private(update):
        return
    chat, user, msg = update.effective_chat, update.effective_user, update.effective_message
    if not chat or not user or not msg:
        return

    enabled = d.db.is_memory_enabled(chat.id, user.id)
    summary = d.db.get_summary(chat.id)
    history = d.db.recent_history(chat.id, d.settings.memory_context_messages)
    total = d.db.history_count(chat.id)

    lines: list[str] = [f"Memory is {'on' if enabled else 'off'}."]
    if summary:
        lines.append("Memory:")
        lines.append(trim_text(summary, 1200))
    else:
        lines.append("Memory: (empty)")

    if history:
        lines.append(f"Recent messages (last {len(history)} of {total}):")
        for msg_row in history:
            role = msg_row.get("role", "user")
            content = trim_text(msg_row.get("content", ""), 500)
            if content:
                lines.append(f"{role}: {content}")
    else:
        lines.append("Recent messages: (none)")

    await send_text(msg, "\n".join(lines), d.settings.telegram_reply_chunk_chars)


# ── /api & /model ──────────────────────────────────────────────────

async def model_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_allowed(update, d):
        return
    if not await ensure_private(update, "Model settings disabled in groups."):
        return
    user = update.effective_user
    raw_key = d.prefs.get(user.id, d.registry.default_key) if user else d.registry.default_key
    model_key = selectable_or_default_model_key(d.registry, raw_key)
    page = find_model_page(d.registry, model_key)
    selectable = selectable_model_keys(d.registry)
    pages = max(1, math.ceil(len(selectable) / 12))
    msg = update.effective_message
    if msg:
        await msg.reply_text(
            f"Current model: {display_model_label(d.registry, model_key)}\n"
            f"Models: {len(selectable)} across {len(selectable_provider_ids(d.registry))} providers.\n"
            f"Select model ({page + 1}/{pages}):",
            reply_markup=model_keyboard(d.registry, model_key, page),
        )


# ── /provider ───────────────────────────────────────────────────────

async def provider_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_allowed(update, d):
        return
    if not await ensure_private(update, "Model settings disabled in groups."):
        return
    from bot.handlers.callbacks import provider_keyboard
    user = update.effective_user
    raw_key = d.prefs.get(user.id, d.registry.default_key) if user else d.registry.default_key
    model_key = selectable_or_default_model_key(d.registry, raw_key)
    providers = selectable_provider_ids(d.registry)
    pages = max(1, math.ceil(len(providers) / 12))
    msg = update.effective_message
    if msg:
        await msg.reply_text(
            f"Current model: {display_model_label(d.registry, model_key)}\n"
            f"Providers: {len(providers)}\n"
            f"Select provider (1/{pages}):",
            reply_markup=provider_keyboard(d.registry, 0),
        )


# ── /modelsearch ────────────────────────────────────────────────────

async def model_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    if not await ensure_allowed(update, d):
        return
    if not await ensure_private(update, "Model settings disabled in groups."):
        return
    query = " ".join(context.args).strip() if context.args else ""
    msg = update.effective_message
    if not msg:
        return
    if not query:
        context.user_data["model_search_waiting"] = True
        await msg.reply_text("Send text to search models.\nExamples: `google flash`, `qwen thinking`.")
        return

    context.user_data["model_search_waiting"] = False
    context.user_data["model_search_query"] = query
    user = update.effective_user
    raw_key = d.prefs.get(user.id, d.registry.default_key) if user else d.registry.default_key
    model_key = selectable_or_default_model_key(d.registry, raw_key)
    from bot.handlers.callbacks import search_keyboard
    keys = search_models(d.registry, query)
    await msg.reply_text(
        f"Current model: {display_model_label(d.registry, model_key)}\nSearch: {query}\nMatches: {len(keys)}",
        reply_markup=search_keyboard(d.registry, model_key, query, 0),
    )


# ── /img ────────────────────────────────────────────────────────────

async def img_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d = get_deps(context)
    msg = update.effective_message
    if not msg:
        return
    user = update.effective_user
    chat = update.effective_chat

    if not is_group(chat):
        if not await ensure_allowed(update, d):
            return

    prompt = " ".join(context.args).strip() if context.args else ""
    if not prompt:
        await msg.reply_text("Usage: /img <prompt>")
        return
    if not user or not chat:
        return

    model_key = d.prefs.get(user.id, d.registry.default_key)
    allowlist = d.settings.image_providers or None

    try:
        await msg.chat.send_action(action="upload_photo")
        image_data, used_model, used_provider = await d.generation.generate_image(
            prompt, model_key, provider_allowlist=allowlist,
        )
        await send_image(msg, image_data, f"Model: {used_provider}/{used_model}")
    except Exception as exc:
        logger.exception("Image generation failed")
        await msg.reply_text(f"Generation error: {exc}")
