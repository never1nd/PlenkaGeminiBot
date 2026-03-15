"""Access control guards — extracted from commands.py and helpers.py."""
from __future__ import annotations

from telegram import Update
from telegram.ext import ContextTypes

from bot.handlers.deps import Deps


def is_group(chat) -> bool:
    return chat is not None and getattr(chat, "type", "") in ("group", "supergroup")


async def ensure_allowed(update: Update, deps: Deps, denied: str = "Access denied.") -> bool:
    user = update.effective_user
    if user and deps.allowlist.is_allowed(user.id):
        return True
    msg = update.effective_message
    if msg:
        await msg.reply_text(denied)
    return False


async def ensure_owner(update: Update, deps: Deps) -> bool:
    user = update.effective_user
    if user and deps.settings.is_owner(user.id):
        return True
    msg = update.effective_message
    if msg:
        await msg.reply_text("Owner only command.")
    return False


async def ensure_private(update: Update, denied: str = "Use this in private chat.") -> bool:
    chat = update.effective_chat
    if chat and chat.type != "private":
        msg = update.effective_message
        if msg:
            await msg.reply_text(denied)
        return False
    return True
