from __future__ import annotations

import logging
import math
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from bot.handlers.deps import get_deps
from bot.handlers.guards import is_group
from bot.handlers.helpers import (
    model_keyboard, model_page, model_select_rows,
    find_model_page, search_models,
    is_selectable_model, selectable_provider_ids, selectable_provider_model_keys,
    selectable_or_default_model_key, display_model_label,
    edit_inline, inline_placeholder_keyboard,
)
from bot.handlers.inline_utils import fetch_inline_prompt
from bot.model_utils import is_default_model_key

logger = logging.getLogger("bot")


# ── shared keyboards ──────────────────────────────────────────────

def provider_keyboard(registry: Any, page: int) -> InlineKeyboardMarkup:
    page_size = 12
    providers = selectable_provider_ids(registry)
    if not providers:
        return InlineKeyboardMarkup([[InlineKeyboardButton("Search", callback_data="modelsearchprompt")]])
    pages = max(1, math.ceil(len(providers) / page_size))
    page = max(0, min(page, pages - 1))
    start = page * page_size
    rows: list[list[InlineKeyboardButton]] = []
    for idx, pid in enumerate(providers[start:start + page_size], start=start):
        models = selectable_provider_model_keys(registry, pid)
        title = f"{registry.get_provider_label(pid)} ({len(models)})"
        rows.append([InlineKeyboardButton(title, callback_data=f"modelprovider:{idx}:0")])
    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("Prev", callback_data=f"modelproviders:{page - 1}"))
    if page < pages - 1:
        nav.append(InlineKeyboardButton("Next", callback_data=f"modelproviders:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([
        InlineKeyboardButton("All Models", callback_data="modelpage:0"),
        InlineKeyboardButton("Search", callback_data="modelsearchprompt"),
    ])
    return InlineKeyboardMarkup(rows)


def search_keyboard(registry: Any, selected: str, query: str, page: int) -> InlineKeyboardMarkup:
    keys = search_models(registry, query)
    pg, pages, page_keys = model_page(registry, keys, page)
    rows = model_select_rows(registry, selected, page_keys)
    nav: list[InlineKeyboardButton] = []
    if pg > 0:
        nav.append(InlineKeyboardButton("Prev", callback_data=f"modelsearchpage:{pg - 1}"))
    if pg < pages - 1:
        nav.append(InlineKeyboardButton("Next", callback_data=f"modelsearchpage:{pg + 1}"))
    if nav:
        rows.append(nav)
    rows.append([
        InlineKeyboardButton("New Search", callback_data="modelsearchprompt"),
        InlineKeyboardButton("All Models", callback_data="modelpage:0"),
    ])
    return InlineKeyboardMarkup(rows)


# ── access check for callbacks ──────────────────────────────────────

async def _check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = get_deps(context)
    query = update.callback_query
    if not query:
        return None, None, None
    user = query.from_user
    if not user or not d.allowlist.is_allowed(user.id):
        if query:
            await query.answer("Access denied.", show_alert=True)
        return None, None, None
    return d, query, user


# ── model selection ─────────────────────────────────────────────────

async def on_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d, query, user = await _check(update, context)
    if not d or not query or not user:
        return
    data = str(query.data or "")
    if not data.startswith("model:"):
        await query.answer()
        return
    key = data[6:]
    if is_default_model_key(key):
        if query.message and is_group(query.message.chat):
            await query.answer("Model settings disabled in groups.", show_alert=True)
            return
    else:
        if key not in d.registry.order_set:
            await query.answer("Unknown model.", show_alert=True)
            return
        if not is_selectable_model(d.registry, key):
            await query.answer("Image models can't be selected.", show_alert=True)
            return
    if query.message and is_group(query.message.chat):
        await query.answer("Model settings disabled in groups.", show_alert=True)
        return
    d.prefs.set(user.id, key)
    page = find_model_page(d.registry, key)
    await query.answer(f"Selected: {display_model_label(d.registry, key)}")
    pages = max(1, math.ceil(len(search_models(d.registry, "")) / 12))
    await query.edit_message_text(
        f"Current model: {display_model_label(d.registry, key)}\n"
        f"Models: {len(search_models(d.registry, ''))} across {len(selectable_provider_ids(d.registry))} providers.\n"
        f"Select model ({page + 1}/{pages}):",
        reply_markup=model_keyboard(d.registry, key, page),
    )


async def on_model_page(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d, query, user = await _check(update, context)
    if not d or not query or not user:
        return
    data = str(query.data or "")
    if not data.startswith("modelpage:"):
        await query.answer()
        return
    try:
        page = int(data[10:])
    except ValueError:
        page = 0
    raw_key = d.prefs.get(user.id, d.registry.default_key)
    key = selectable_or_default_model_key(d.registry, raw_key)
    pages = max(1, math.ceil(len(search_models(d.registry, "")) / 12))
    page = max(0, min(page, pages - 1))
    await query.answer()
    await query.edit_message_text(
        f"Current model: {display_model_label(d.registry, key)}\n"
        f"Models: {len(search_models(d.registry, ''))} across {len(selectable_provider_ids(d.registry))} providers.\n"
        f"Select model ({page + 1}/{pages}):",
        reply_markup=model_keyboard(d.registry, key, page),
    )


async def on_model_providers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d, query, user = await _check(update, context)
    if not d or not query or not user:
        return
    data = str(query.data or "")
    try:
        page = int(data.split(":")[1])
    except (IndexError, ValueError):
        page = 0
    raw_key = d.prefs.get(user.id, d.registry.default_key)
    key = selectable_or_default_model_key(d.registry, raw_key)
    providers = selectable_provider_ids(d.registry)
    pages = max(1, math.ceil(len(providers) / 12))
    page = max(0, min(page, pages - 1))
    await query.answer()
    await query.edit_message_text(
        f"Current model: {display_model_label(d.registry, key)}\n"
        f"Providers: {len(providers)}\n"
        f"Select provider ({page + 1}/{pages}):",
        reply_markup=provider_keyboard(d.registry, page),
    )


async def on_model_provider(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d, query, user = await _check(update, context)
    if not d or not query or not user:
        return
    data = str(query.data or "")
    if not data.startswith("modelprovider:"):
        await query.answer()
        return
    parts = data[14:].split(":")
    if len(parts) != 2:
        await query.answer("Invalid request.", show_alert=True)
        return
    try:
        pidx, page = int(parts[0]), int(parts[1])
    except ValueError:
        await query.answer("Invalid request.", show_alert=True)
        return
    reg = d.registry
    providers = selectable_provider_ids(reg)
    if pidx < 0 or pidx >= len(providers):
        await query.answer("Unknown provider.", show_alert=True)
        return
    pid = providers[pidx]
    pkeys = selectable_provider_model_keys(reg, pid)
    raw_key = d.prefs.get(user.id, reg.default_key)
    key = selectable_or_default_model_key(reg, raw_key)
    pg, pages, pk = model_page(reg, pkeys, page)
    rows = model_select_rows(reg, key, pk)
    nav: list[InlineKeyboardButton] = []
    if pg > 0:
        nav.append(InlineKeyboardButton("Prev", callback_data=f"modelprovider:{pidx}:{pg - 1}"))
    if pg < pages - 1:
        nav.append(InlineKeyboardButton("Next", callback_data=f"modelprovider:{pidx}:{pg + 1}"))
    if nav:
        rows.append(nav)
    rows.append([
        InlineKeyboardButton("Providers", callback_data="modelproviders:0"),
        InlineKeyboardButton("Search", callback_data="modelsearchprompt"),
    ])
    await query.answer()
    await query.edit_message_text(
        f"Current model: {display_model_label(reg, key)}\n"
        f"Provider: {reg.get_provider_label(pid)} ({len(pkeys)} models)\n"
        f"Select model ({pg + 1}/{pages}):",
        reply_markup=InlineKeyboardMarkup(rows),
    )


# ── search callbacks ───────────────────────────────────────────────

async def on_search_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d, query, user = await _check(update, context)
    if not d or not query or not user:
        return
    context.user_data["model_search_waiting"] = True
    await query.answer("Send search text in chat.")
    await query.edit_message_text("Send text to search models.\nExamples: `google flash`, `qwen thinking`.")


async def on_search_page(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d, query, user = await _check(update, context)
    if not d or not query or not user:
        return
    data = str(query.data or "")
    try:
        page = int(data.split(":")[1])
    except (IndexError, ValueError):
        page = 0
    sq = str(context.user_data.get("model_search_query", "")).strip()
    if not sq:
        await query.answer("No active search. Use /modelsearch.", show_alert=True)
        return
    raw_key = d.prefs.get(user.id, d.registry.default_key)
    key = selectable_or_default_model_key(d.registry, raw_key)
    await query.answer()
    await query.edit_message_text(
        f"Current model: {display_model_label(d.registry, key)}\nSearch: {sq}\nMatches: {len(search_models(d.registry, sq))}",
        reply_markup=search_keyboard(d.registry, key, sq, page),
    )


# ── inline generate callback ─────────────────────────────────────────

async def on_inline_generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d, query, user = await _check(update, context)
    if not d or not query or not user:
        return
    data = str(query.data or "")
    if not data.startswith("inlinegen:"):
        await query.answer()
        return
    token = data.split(":", 1)[1].strip()
    if not query.inline_message_id:
        await query.answer("Inline message not found.", show_alert=True)
        return

    prompt = fetch_inline_prompt(context, token, user.id)
    if not prompt:
        await query.answer("Inline request expired. Please retry.", show_alert=True)
        await edit_inline(
            context.bot,
            query.inline_message_id,
            "Inline request expired. Please try again.",
            reply_markup=inline_placeholder_keyboard(),
        )
        return

    await query.answer("Generating...")
    inline_id = query.inline_message_id
    try:
        full = f"{prompt}\n\n{d.settings.reply_instruction_short}".strip()
        answer, used_model, used_provider, usage = await d.generation.generate_inline_text(full)
        logger.info(
            "Inline (button): user=%s provider=%s model=%s tokens=%s",
            user.id, used_provider, used_model, usage.get("total_tokens", "n/a"),
        )
        formatted = f'"{prompt}"\n\n---\n\n"{answer}"'
        await edit_inline(
            context.bot, inline_id, formatted, reply_markup=None,
        )
    except Exception as exc:
        logger.exception("Inline generation failed (button)")
        await edit_inline(
            context.bot, inline_id, f"Error: {exc}", reply_markup=None,
        )


async def on_inline_dm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d, query, user = await _check(update, context)
    if not d or not query or not user:
        return
    data = str(query.data or "")
    if not data.startswith("inlinedm:"):
        await query.answer()
        return
    token = data.split(":", 1)[1].strip()
    prompt = fetch_inline_prompt(context, token, user.id)
    text = "You tapped the inline button."
    if prompt:
        text = f"Inline prompt:\n{prompt}"
    try:
        await context.bot.send_message(chat_id=user.id, text=text)
        await query.answer("Sent you a DM.")
    except Exception as exc:
        logger.warning("Inline DM failed: %s", exc)
        await query.answer("Please open the bot first.", show_alert=True)


async def on_inline_placeholder(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    d, query, user = await _check(update, context)
    if not d or not query or not user:
        return
    if str(query.data or "") != "inlineplaceholder":
        await query.answer()
        return
    await query.answer("Working…")


# ── inline generate callback ─────────────────────────────────────────
